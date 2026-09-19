#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace edgevision {

// Thin wrapper over a deserialized TensorRT 10 engine with one input and one output,
// device buffers owned here, execution via enqueueV3 on a caller-provided stream.
class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path);
    ~TrtEngine();
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Device pointer where the input tensor must be written (float32 or float16!).
    void* input_device() const { return input_.device; }
    nvinfer1::Dims input_dims() const { return input_.dims; }
    nvinfer1::DataType input_type() const { return input_.type; }
    size_t input_bytes() const { return input_.bytes; }

    void* output_device() const { return output_.device; }
    nvinfer1::Dims output_dims() const { return output_.dims; }
    nvinfer1::DataType output_type() const { return output_.type; }
    size_t output_bytes() const { return output_.bytes; }

    // Enqueue inference on `stream` (asynchronous).
    void enqueue(cudaStream_t stream);

    static std::string version();

private:
    struct Tensor {
        std::string name;
        nvinfer1::Dims dims{};
        nvinfer1::DataType type{};
        size_t bytes = 0;
        void* device = nullptr;
    };

    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    Tensor input_;
    Tensor output_;
};

size_t element_size(nvinfer1::DataType type);
size_t volume(const nvinfer1::Dims& dims);

}  // namespace edgevision
