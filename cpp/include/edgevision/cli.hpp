#pragma once

#include <string>
#include <vector>

namespace edgevision {

// Command line of edgevision_trt. Plain data: parsing and the report file name live here so
// they can be unit-tested without a GPU; the runtime wiring stays in main.cpp.
struct Args {
    std::string engine;
    std::vector<std::string> sources;
    std::string decoder = "opencv";  // opencv (CPU decode, H2D copy) | nvdec (GPU decode, no copy)
    std::string names;
    std::string label;
    std::string out;
    std::string dump_detections;
    int frames = 300;
    int warmup = 30;
    int streams = 1;
    // Detection score filter. With --track and no explicit --confidence it drops to the
    // tracker's low threshold (kTrackLowThresh): ByteTrack needs the weak detections.
    float confidence = 0.5f;
    bool confidence_set = false;  // --confidence given explicitly
    bool pinned = true;
    bool stage_timing = true;
    bool batched = false;  // phase B: one batch-N inference per round for all streams
    bool track = false;    // Sprint 8: ByteTrack per stream
    float track_thresh = 0.5f;
    int track_buffer = 30;
    std::string dump_tracks;  // JSONL of stream 0's tracks for the measured frames
    std::string render;       // annotated video (opencv decoder only)
    bool help = false;        // --help / -h seen: print usage() and exit 0
};

// Parses argv[1..]; throws std::runtime_error for unknown flags, missing values and invalid
// combinations. Stops at --help (validation is skipped, `help` is set).
Args parse_args(int argc, const char* const* argv);

const char* usage();

// benchmarks/results/cpp_tensorrt_<decoder>[_sN | _bN][_<label>]_<stamp>.json, unless --out was given.
std::string default_report_path(const Args& a, const std::string& stamp);

}  // namespace edgevision
