#pragma once

#include <map>
#include <string>

#include "edgevision/cli.hpp"
#include "edgevision/report.hpp"

namespace edgevision {

// One independent stream on the calling thread: its own decoder (OpenCV on the CPU or NVDEC),
// its own TensorRT execution context and, with --track, its own ByteTracker. Warm-up frames are
// discarded, `result.metrics` holds the measured window, `result.first_measured` the detections
// of the first measured frame. Short clips are looped until warmup + frames were processed.
void run_stream(const Args& args, const std::map<int, std::string>& names, StreamResult& result);

// TensorRT, CUDA runtime, GPU 0 and OpenCV versions for the report.
Environment runtime_environment();

}  // namespace edgevision
