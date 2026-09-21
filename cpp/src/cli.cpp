#include "edgevision/cli.hpp"

#include <stdexcept>
#include <string>

namespace edgevision {

const char* usage() {
    return "usage: edgevision_trt --engine E [--source S]... [--decoder opencv|nvdec] [--streams N] [--batched]\n"
           "       [--frames N] [--warmup N] [--label L] [--names names.json] [--out report.json]\n"
           "       [--dump-detections dets.json] [--confidence 0.5] [--no-pinned] [--no-stage-timing]\n"
           "  --batched: all streams share one batch-N engine (engine exported with batch=N)\n"
           "  --track [--track-thresh 0.5] [--track-buffer 30] [--dump-tracks f.jsonl] [--render out.mp4]\n"
           "           ByteTrack per stream; use an engine exported with conf 0.1 so weak detections exist\n";
}

Args parse_args(int argc, const char* const* argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
            return argv[++i];
        };
        if (k == "--engine") a.engine = next("--engine");
        else if (k == "--source") a.sources.push_back(next("--source"));
        else if (k == "--decoder") a.decoder = next("--decoder");
        else if (k == "--names") a.names = next("--names");
        else if (k == "--label") a.label = next("--label");
        else if (k == "--out") a.out = next("--out");
        else if (k == "--dump-detections") a.dump_detections = next("--dump-detections");
        else if (k == "--frames") a.frames = std::stoi(next("--frames"));
        else if (k == "--warmup") a.warmup = std::stoi(next("--warmup"));
        else if (k == "--streams") a.streams = std::stoi(next("--streams"));
        else if (k == "--confidence") a.confidence = std::stof(next("--confidence"));
        else if (k == "--no-pinned") a.pinned = false;
        else if (k == "--no-stage-timing") a.stage_timing = false;
        else if (k == "--batched") a.batched = true;
        else if (k == "--track") a.track = true;
        else if (k == "--track-thresh") a.track_thresh = std::stof(next("--track-thresh"));
        else if (k == "--track-buffer") a.track_buffer = std::stoi(next("--track-buffer"));
        else if (k == "--dump-tracks") a.dump_tracks = next("--dump-tracks");
        else if (k == "--render") a.render = next("--render");
        else if (k == "--help" || k == "-h") {
            a.help = true;
            return a;
        } else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.engine.empty()) throw std::runtime_error("--engine is required");
    if (a.sources.empty()) a.sources.emplace_back("videos/pedestrian_area_1080p25.webm");
    if (a.decoder != "opencv" && a.decoder != "nvdec") throw std::runtime_error("--decoder must be opencv or nvdec");
    if (a.streams < 1) throw std::runtime_error("--streams must be >= 1");
    if (a.frames < 1) throw std::runtime_error("--frames must be >= 1");
    if (a.warmup < 0) throw std::runtime_error("--warmup must be >= 0");
    if (!a.render.empty() && a.decoder != "opencv")
        throw std::runtime_error("--render needs --decoder opencv (host frames)");
    if (!a.render.empty() && !a.track) a.track = true;
    return a;
}

std::string default_report_path(const Args& a, const std::string& stamp) {
    if (!a.out.empty()) return a.out;
    std::string name = "benchmarks/results/cpp_tensorrt_" + a.decoder;
    if (a.batched) name += "_b" + std::to_string(a.streams);
    else if (a.streams > 1) name += "_s" + std::to_string(a.streams);
    if (!a.label.empty()) name += "_" + a.label;
    return name + "_" + stamp + ".json";
}

}  // namespace edgevision
