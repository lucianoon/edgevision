#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

namespace edgevision {

// Thin wrapper over a deserialized TensorRT 10 engine with one input and one output,
// device buffers owned here (sized for the largest profile shape), execution via
// enqueueV3 on a caller-provided stream. Engines with a dynamic batch dimension are
// supported: set_batch(n) selects the batch before enqueue (default 1).
class TrtEngine {
public:
    explicit TrtEngine(const std::string& engine_path);
    ~TrtEngine();
    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Device pointer where the input tensor must be written (FP32 expected by callers).
    void* input_device() const { return input_.device; }
    nvinfer1::Dims input_dims() const { return input_.dims; }  // current (batch resolved)
    nvinfer1::DataType input_type() const { return input_.type; }
    size_t input_bytes() const { return input_.bytes; }  // for the current batch

    void* output_device() const { return output_.device; }
    nvinfer1::Dims output_dims() const { return output_.dims; }  // current (batch resolved)
    nvinfer1::DataType output_type() const { return output_.type; }
    size_t output_bytes() const { return output_.bytes; }  // for the current batch

    bool dynamic_batch() const { return dynamic_; }
    int max_batch() const { return max_batch_; }
    int batch() const { return batch_; }
    // Select the batch size for the next enqueue (dynamic engines: 1..max_batch();
    // static engines: must equal the engine's batch).
    void set_batch(int n);

    // Point the input tensor at another device buffer (ping-pong batching). The
    // buffer must hold input_bytes(). The engine's own buffer stays allocated.
    void bind_input(void* device_ptr);

    // Enqueue inference on `stream` (asynchronous).
    void enqueue(cudaStream_t stream);

    static std::string version();

private:
    struct Tensor {
        std::string name;
        nvinfer1::Dims dims{};  // current shape
        nvinfer1::DataType type{};
        size_t bytes = 0;     // for the current shape
        size_t capacity = 0;  // allocated (max profile shape)
        void* device = nullptr;
    };

    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
    };

    void refresh_shapes();

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    Tensor input_;
    Tensor output_;
    bool dynamic_ = false;
    int max_batch_ = 1;
    int batch_ = 1;
};

size_t element_size(nvinfer1::DataType type);
size_t volume(const nvinfer1::Dims& dims);

}  // namespace edgevision
