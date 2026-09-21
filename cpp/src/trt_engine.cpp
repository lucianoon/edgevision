#include "edgevision/trt_engine.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "edgevision/cuda_check.hpp"

namespace edgevision {

void TrtEngine::Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << msg << "\n";
}

size_t element_size(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kINT8: return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL: return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kINT64: return 8;
        default: throw std::runtime_error("unsupported TensorRT tensor data type");
    }
}

size_t volume(const nvinfer1::Dims& dims) {
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) throw std::runtime_error("volume() of a shape with a dynamic dimension");
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

TrtEngine::TrtEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open engine: " + engine_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    // Ultralytics exports prefix the engine with <int32 LE length><JSON metadata>.
    if (blob.size() > 4) {
        const auto* h = reinterpret_cast<const unsigned char*>(blob.data());
        const std::int32_t length = static_cast<std::int32_t>(h[0] | (h[1] << 8) | (h[2] << 16) | (h[3] << 24));
        if (length > 0 && static_cast<size_t>(length) + 4 < blob.size() && blob[4] == '{')
            blob.erase(blob.begin(), blob.begin() + 4 + length);
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), blob.size()));
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed: " + engine_path);
    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("createExecutionContext failed");

    int inputs = 0, outputs = 0;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        Tensor t;
        t.name = engine_->getIOTensorName(i);
        t.type = engine_->getTensorDataType(t.name.c_str());
        const nvinfer1::Dims shape = engine_->getTensorShape(t.name.c_str());
        const bool is_input = engine_->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT;

        nvinfer1::Dims max_shape = shape;
        if (is_input && shape.nbDims > 0 && shape.d[0] < 0) {
            // Dynamic batch: size buffers for the largest shape of optimisation profile 0.
            dynamic_ = true;
            max_shape = engine_->getProfileShape(t.name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            max_batch_ = static_cast<int>(max_shape.d[0]);
            for (int d = 1; d < max_shape.nbDims; ++d)
                if (max_shape.d[d] < 0) throw std::runtime_error("only the batch dimension may be dynamic");
        } else if (is_input && shape.nbDims > 0) {
            max_batch_ = batch_ = static_cast<int>(shape.d[0]);  // static batch (1 or N)
        }
        if (is_input) {
            t.capacity = volume(max_shape) * element_size(t.type);
        } else {
            // Output capacity: assume it scales with the batch like the input does
            // (true for Nx300x6). Resolved exactly after set_batch() via the context.
            nvinfer1::Dims out_max = shape;
            if (out_max.nbDims > 0 && out_max.d[0] < 0) out_max.d[0] = max_batch_;
            for (int d = 1; d < out_max.nbDims; ++d)
                if (out_max.d[d] < 0) throw std::runtime_error("output has a dynamic non-batch dimension");
            t.capacity = volume(out_max) * element_size(t.type);
        }
        EV_CUDA_CHECK(cudaMalloc(&t.device, t.capacity));
        if (!context_->setTensorAddress(t.name.c_str(), t.device))
            throw std::runtime_error("setTensorAddress failed for " + t.name);
        if (is_input) {
            input_ = t;
            ++inputs;
        } else {
            output_ = t;
            ++outputs;
        }
    }
    if (inputs != 1 || outputs != 1)
        throw std::runtime_error("engine must have exactly one input and one output tensor");

    set_batch(dynamic_ ? 1 : batch_);
}

void TrtEngine::set_batch(int n) {
    if (!dynamic_) {
        if (n != batch_)
            throw std::runtime_error("engine has a fixed batch of " + std::to_string(batch_) + ", asked for " +
                                     std::to_string(n));
        refresh_shapes();
        return;
    }
    if (n < 1 || n > max_batch_)
        throw std::runtime_error("batch " + std::to_string(n) + " outside the engine profile [1, " +
                                 std::to_string(max_batch_) + "]");
    nvinfer1::Dims dims = engine_->getTensorShape(input_.name.c_str());
    dims.d[0] = n;
    if (!context_->setInputShape(input_.name.c_str(), dims)) throw std::runtime_error("setInputShape failed");
    batch_ = n;
    refresh_shapes();
}

void TrtEngine::refresh_shapes() {
    input_.dims = context_->getTensorShape(input_.name.c_str());
    output_.dims = context_->getTensorShape(output_.name.c_str());
    input_.bytes = volume(input_.dims) * element_size(input_.type);
    output_.bytes = volume(output_.dims) * element_size(output_.type);
    if (input_.bytes > input_.capacity || output_.bytes > output_.capacity)
        throw std::runtime_error("tensor shape exceeds the allocated capacity");
}

TrtEngine::~TrtEngine() {
    if (input_.device) cudaFree(input_.device);
    if (output_.device) cudaFree(output_.device);
}

void TrtEngine::bind_input(void* device_ptr) {
    if (!context_->setTensorAddress(input_.name.c_str(), device_ptr))
        throw std::runtime_error("setTensorAddress(input) failed");
}

void TrtEngine::enqueue(cudaStream_t stream) {
    if (!context_->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
}

std::string TrtEngine::version() {
    const int v = getInferLibVersion();
    return std::to_string(v / 10000) + "." + std::to_string((v / 100) % 100) + "." + std::to_string(v % 100);
}

}  // namespace edgevision
