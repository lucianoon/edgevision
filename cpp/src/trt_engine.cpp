#include "edgevision/trt_engine.hpp"

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
        if (dims.d[i] < 0) throw std::runtime_error("dynamic tensor shapes are not supported");
        v *= static_cast<size_t>(dims.d[i]);
    }
    return v;
}

TrtEngine::TrtEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open engine: " + engine_path);
    std::vector<char> blob((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

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
        t.dims = engine_->getTensorShape(t.name.c_str());
        t.type = engine_->getTensorDataType(t.name.c_str());
        t.bytes = volume(t.dims) * element_size(t.type);
        EV_CUDA_CHECK(cudaMalloc(&t.device, t.bytes));
        if (!context_->setTensorAddress(t.name.c_str(), t.device))
            throw std::runtime_error("setTensorAddress failed for " + t.name);
        if (engine_->getTensorIOMode(t.name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
            input_ = t;
            ++inputs;
        } else {
            output_ = t;
            ++outputs;
        }
    }
    if (inputs != 1 || outputs != 1)
        throw std::runtime_error("engine must have exactly one input and one output tensor");
}

TrtEngine::~TrtEngine() {
    if (input_.device) cudaFree(input_.device);
    if (output_.device) cudaFree(output_.device);
}

void TrtEngine::enqueue(cudaStream_t stream) {
    if (!context_->enqueueV3(stream)) throw std::runtime_error("enqueueV3 failed");
}

std::string TrtEngine::version() {
    const int v = getInferLibVersion();
    return std::to_string(v / 10000) + "." + std::to_string((v / 100) % 100) + "." + std::to_string(v % 100);
}

}  // namespace edgevision
