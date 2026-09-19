#include "edgevision/detector.hpp"

#include <algorithm>
#include <stdexcept>

#include "edgevision/cuda_check.hpp"

namespace edgevision {

TensorRTDetector::TensorRTDetector(const std::string& engine_path, float confidence,
                                   std::map<int, std::string> names, PerformanceMetrics* metrics)
    : engine_(engine_path),
      confidence_(confidence),
      names_(std::move(names)),
      metrics_(metrics ? metrics : &own_metrics_),
      timing_(metrics != nullptr) {
    const auto in = engine_.input_dims();
    if (in.nbDims != 4 || in.d[0] != 1 || in.d[1] != 3 || in.d[2] != in.d[3])
        throw std::runtime_error("engine input must be 1x3xSxS");
    if (engine_.input_type() != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error("engine input must be FP32 (build with trtexec default I/O formats)");
    input_size_ = static_cast<int>(in.d[2]);

    const auto out = engine_.output_dims();
    if (out.nbDims != 3 || out.d[0] != 1 || out.d[2] != 6 || engine_.output_type() != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error(
            "engine output must be 1xNx6 FP32: export the ONNX with NMS in the graph (nms=True)");
    max_det_ = static_cast<int>(out.d[1]);

    EV_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    EV_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_output_), engine_.output_bytes(), cudaHostAllocDefault));
}

TensorRTDetector::~TensorRTDetector() {
    if (device_frame_) cudaFree(device_frame_);
    if (host_output_) cudaFreeHost(host_output_);
    if (pinned_frame_) cudaFreeHost(pinned_frame_);
    if (stream_) cudaStreamDestroy(stream_);
}

void TensorRTDetector::ensure_device_frame(size_t bytes) {
    if (bytes <= device_frame_bytes_) return;
    if (device_frame_) EV_CUDA_CHECK(cudaFree(device_frame_));
    EV_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_frame_), bytes));
    device_frame_bytes_ = bytes;
}

void TensorRTDetector::sync_if_timing() {
    if (timing_) EV_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

cv::Mat TensorRTDetector::pinned_frame(int rows, int cols) {
    const size_t bytes = static_cast<size_t>(rows) * cols * 3;
    if (bytes != pinned_frame_bytes_) {
        if (pinned_frame_) EV_CUDA_CHECK(cudaFreeHost(pinned_frame_));
        EV_CUDA_CHECK(cudaHostAlloc(&pinned_frame_, bytes, cudaHostAllocDefault));
        pinned_frame_bytes_ = bytes;
    }
    return cv::Mat(rows, cols, CV_8UC3, pinned_frame_);
}

std::vector<Detection> TensorRTDetector::detect(const cv::Mat& frame) {
    if (frame.type() != CV_8UC3 || frame.empty()) throw std::runtime_error("detect() expects a non-empty BGR 8-bit frame");

    const size_t frame_bytes = frame.step[0] * static_cast<size_t>(frame.rows);
    const LetterboxInfo info = compute_letterbox(frame.cols, frame.rows, input_size_);

    {
        ScopedTimer timer(*metrics_, "preprocess");
        ensure_device_frame(frame_bytes);
        EV_CUDA_CHECK(cudaMemcpyAsync(device_frame_, frame.data, frame_bytes, cudaMemcpyHostToDevice, stream_));
        letterbox_cuda(device_frame_, static_cast<int>(frame.step[0]), info,
                       static_cast<float*>(engine_.input_device()), stream_);
        sync_if_timing();
    }
    return run_and_collect(info);
}

std::vector<Detection> TensorRTDetector::detect(const GpuFrame& frame) {
    if (!frame.y || !frame.uv || frame.width <= 0 || frame.height <= 0)
        throw std::runtime_error("detect() expects a valid NV12 GpuFrame");
    const LetterboxInfo info = compute_letterbox(frame.width, frame.height, input_size_);
    {
        ScopedTimer timer(*metrics_, "preprocess");
        letterbox_nv12_cuda(frame, info, static_cast<float*>(engine_.input_device()), stream_);
        sync_if_timing();
    }
    return run_and_collect(info);
}

std::vector<Detection> TensorRTDetector::run_and_collect(const LetterboxInfo& info) {
    {
        ScopedTimer timer(*metrics_, "inference");
        engine_.enqueue(stream_);
        sync_if_timing();
    }

    std::vector<Detection> detections;
    {
        ScopedTimer timer(*metrics_, "postprocess");
        EV_CUDA_CHECK(cudaMemcpyAsync(host_output_, engine_.output_device(), engine_.output_bytes(),
                                      cudaMemcpyDeviceToHost, stream_));
        EV_CUDA_CHECK(cudaStreamSynchronize(stream_));

        const float w = static_cast<float>(info.source_width), h = static_cast<float>(info.source_height);
        for (int i = 0; i < max_det_; ++i) {
            const float* row = host_output_ + i * 6;
            const float score = row[4];
            if (score < confidence_) continue;  // zero-padded rows land here too
            Detection d;
            d.x1 = std::clamp((row[0] - info.pad_x) / info.scale, 0.0f, w);
            d.y1 = std::clamp((row[1] - info.pad_y) / info.scale, 0.0f, h);
            d.x2 = std::clamp((row[2] - info.pad_x) / info.scale, 0.0f, w);
            d.y2 = std::clamp((row[3] - info.pad_y) / info.scale, 0.0f, h);
            d.confidence = score;
            d.class_id = static_cast<int>(row[5]);
            auto it = names_.find(d.class_id);
            d.class_name = it != names_.end() ? it->second : std::to_string(d.class_id);
            detections.push_back(std::move(d));
        }
    }
    return detections;
}

}  // namespace edgevision
