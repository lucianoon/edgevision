#pragma once

#include <map>
#include <string>
#include <vector>

#include "edgevision/detection.hpp"
#include "edgevision/metrics.hpp"

namespace edgevision {

// Sprint 6 phase B: N streams share ONE TensorRT execution with a static batch of N.
//
//   worker i (thread): decode -> letterbox kernel into slot i of input buffer [round % 2]
//                      -> signal ready(round)
//   coordinator:       wait all N ready -> bind input buffer -> enqueueV3 (batch N)
//                      -> D2H (N x max_det x 6) -> free buffer for round+2 -> collect
//
// Two input buffers (ping-pong) let workers decode round r+1 while round r infers.
struct BatchOptions {
    std::string engine_path;  // static-batch engine, batch == sources.size()
    std::vector<std::string> sources;
    bool nvdec = true;  // false: OpenCV decode + H2D copy per worker
    float confidence = 0.5f;
    int frames = 300;  // measured rounds (each round = one frame per stream)
    int warmup = 30;
};

struct BatchResult {
    PerformanceMetrics rounds;                // stages: gather, inference, postprocess, end_to_end (per round)
    std::vector<PerformanceMetrics> workers;  // per stream: decode, preprocess
    std::vector<std::vector<Detection>> first_measured;  // per stream, first measured round
    double measured_seconds = 0.0;
    int rounds_measured = 0;
    int batch = 0;
    double aggregate_fps() const { return measured_seconds > 0 ? batch * rounds_measured / measured_seconds : 0.0; }
};

BatchResult run_batched(const BatchOptions& options, const std::map<int, std::string>& names);

}  // namespace edgevision
