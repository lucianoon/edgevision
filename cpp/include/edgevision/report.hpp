#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "edgevision/cli.hpp"
#include "edgevision/detection.hpp"
#include "edgevision/metrics.hpp"
#include "edgevision/tracker.hpp"

namespace edgevision {

// What one stream produced: its metrics window (measured frames), the detections of the first
// measured frame (parity tests), wall time and tracking count. Filled by run_stream() or mapped
// from the batched pipeline.
struct StreamResult {
    PerformanceMetrics metrics;
    std::vector<Detection> first_measured;
    double measured_seconds = 0.0;  // wall time of the measured frames (after warm-up)
    std::string source;
    std::string error;
    int index = 0;
    int unique_track_ids = 0;
};

// Versions written into the report; queried from the runtime by main (GPU code), so the
// report writer itself needs no CUDA.
struct Environment {
    std::string tensorrt;
    std::string cuda_runtime;
    std::string gpu;
    std::string opencv;
};

// UTC time as YYYYMMDDTHHMMSSZ (report file names and the timestamp_utc field).
std::string utc_stamp();

// Console table with the same columns as python -m edgevision.benchmark.
void print_table(const char* title, const PerformanceMetrics& m);

// JSON report with the schema of the Python benchmark plus the multi-stream fields
// (decoder, streams, mode, aggregate_fps, per_stream). `merged` is the distribution reported
// under "stages" (all streams, or the per-round metrics in batched mode).
void write_report(std::ostream& out, const Args& a, const std::vector<StreamResult>& streams,
                  const PerformanceMetrics& merged, double aggregate_fps, const std::string& stamp,
                  const Environment& env);
void write_report(const std::string& path, const Args& a, const std::vector<StreamResult>& streams,
                  const PerformanceMetrics& merged, double aggregate_fps, const std::string& stamp,
                  const Environment& env);

// JSON array of detections (tests/test_cpp_runtime.py reads it).
void dump_detections(std::ostream& out, const std::vector<Detection>& dets);
void dump_detections(const std::string& path, const std::vector<Detection>& dets);

// One JSONL line: {"frame": n, "tracks": [{"id", "x1", "y1", "x2", "y2", "score", "class_name"}]}
void dump_tracks_line(std::ostream& out, int frame, const std::vector<Track>& tracks);

}  // namespace edgevision
