#pragma once

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "edgevision/detection.hpp"
#include "edgevision/letterbox.hpp"
#include "edgevision/metrics.hpp"
#include "edgevision/trt_engine.hpp"

namespace edgevision {

// End-to-end GPU detector: BGR frame (host) -> H2D copy -> CUDA letterbox written
// straight into the engine input -> enqueueV3 -> D2H of the (1, max_det, 6) NMS'd
// output -> rescale to the source frame.
// Requires an engine exported with NMS in the graph (rows: x1 y1 x2 y2 score class)
// and FP32 input/output tensors (trtexec keeps I/O in FP32 even with --fp16).
class TensorRTDetector {
public:
    // When `metrics` is given, each stage is synchronised and timed separately
    // (preprocess / inference / postprocess), which costs a little throughput.
    TensorRTDetector(const std::string& engine_path, float confidence,
                     std::map<int, std::string> names, PerformanceMetrics* metrics = nullptr);
    ~TensorRTDetector();
    TensorRTDetector(const TensorRTDetector&) = delete;
    TensorRTDetector& operator=(const TensorRTDetector&) = delete;

    std::vector<Detection> detect(const cv::Mat& frame_bgr);

    // Pinned host image of the given size; decode into it to make the H2D copy a DMA
    // instead of a staged copy from pageable memory. Owned by the detector.
    cv::Mat pinned_frame(int rows, int cols);

    int input_size() const { return input_size_; }
    int max_detections() const { return max_det_; }

private:
    void ensure_device_frame(size_t bytes);
    void sync_if_timing();

    TrtEngine engine_;
    float confidence_;
    std::map<int, std::string> names_;
    PerformanceMetrics own_metrics_;
    PerformanceMetrics* metrics_;
    bool timing_;

    cudaStream_t stream_ = nullptr;
    std::uint8_t* device_frame_ = nullptr;
    size_t device_frame_bytes_ = 0;
    float* host_output_ = nullptr;  // pinned
    void* pinned_frame_ = nullptr;
    size_t pinned_frame_bytes_ = 0;

    int input_size_ = 640;
    int max_det_ = 300;
};

}  // namespace edgevision
