// edgevision_trt: C++ TensorRT runtime benchmark. Same stages, table and JSON report
// schema as `python -m edgevision.benchmark`, so results can be compared side by side.
//
//   edgevision_trt --engine models/tensorrt/yolo26n_nms_fp16.engine \
//                  --source videos/pedestrian_area_1080p25.webm --frames 300 --warmup 30 --label fp16_nmsgraph

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "edgevision/detector.hpp"
#include "edgevision/metrics.hpp"
#include "edgevision/names.hpp"
#include "edgevision/video_decoder.hpp"

using namespace edgevision;

namespace {

struct Args {
    std::string engine;
    std::string source = "videos/pedestrian_area_1080p25.webm";
    std::string decoder = "opencv";  // opencv (CPU decode, H2D copy) | nvdec (GPU decode, no copy)
    std::string names;
    std::string label;
    std::string out;
    std::string dump_detections;
    int frames = 300;
    int warmup = 30;
    float confidence = 0.5f;
    bool pinned = true;
    bool stage_timing = true;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
            return argv[++i];
        };
        if (k == "--engine") a.engine = next("--engine");
        else if (k == "--source") a.source = next("--source");
        else if (k == "--decoder") a.decoder = next("--decoder");
        else if (k == "--names") a.names = next("--names");
        else if (k == "--label") a.label = next("--label");
        else if (k == "--out") a.out = next("--out");
        else if (k == "--dump-detections") a.dump_detections = next("--dump-detections");
        else if (k == "--frames") a.frames = std::stoi(next("--frames"));
        else if (k == "--warmup") a.warmup = std::stoi(next("--warmup"));
        else if (k == "--confidence") a.confidence = std::stof(next("--confidence"));
        else if (k == "--no-pinned") a.pinned = false;
        else if (k == "--no-stage-timing") a.stage_timing = false;
        else if (k == "--help" || k == "-h") {
            std::cout << "usage: edgevision_trt --engine E [--source S] [--decoder opencv|nvdec] [--frames N]\n"
                         "       [--warmup N] [--label L] [--names names.json] [--out report.json]\n"
                         "       [--dump-detections dets.json] [--confidence 0.5] [--no-pinned] [--no-stage-timing]\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.engine.empty()) throw std::runtime_error("--engine is required");
    if (a.decoder != "opencv" && a.decoder != "nvdec") throw std::runtime_error("--decoder must be opencv or nvdec");
    return a;
}

std::string utc_stamp() {
    const std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", std::gmtime(&now));
    return buf;
}

std::string gpu_name() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "unknown";
    return prop.name;
}

std::string cuda_runtime_version() {
    int v = 0;
    cudaRuntimeGetVersion(&v);
    return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10);
}

void print_table(const PerformanceMetrics& m) {
    const auto summary = m.summary();
    std::printf("\nbackend=cpp_tensorrt  fps=%.1f\n", m.fps());  // decoder printed in the JSON
    std::printf("%-14s%9s%9s%9s%9s  n\n", "stage", "mean", "p50", "p95", "max");
    for (const auto& name : m.order()) {
        const auto& s = summary.at(name);
        std::printf("%-14s%9.1f%9.1f%9.1f%9.1f  %zu\n", name.c_str(), s.mean_ms, s.p50_ms, s.p95_ms, s.max_ms,
                    s.samples);
    }
}

void write_report(const Args& a, const PerformanceMetrics& m, const std::string& stamp, const std::string& path) {
    std::ofstream f(path);
    f << "{\n";
    f << "  \"timestamp_utc\": \"" << stamp << "\",\n";
    f << "  \"backend\": \"cpp_tensorrt\",\n";
    f << "  \"label\": \"" << a.label << "\",\n";
    f << "  \"model_path\": \"" << a.engine << "\",\n";
    f << "  \"confidence\": " << a.confidence << ",\n";
    f << "  \"source\": \"" << a.source << "\",\n";
    f << "  \"decoder\": \"" << a.decoder << "\",\n";
    f << "  \"frames\": " << a.frames << ",\n";
    f << "  \"warmup\": " << a.warmup << ",\n";
    f << "  \"pinned_host_frame\": " << (a.pinned ? "true" : "false") << ",\n";
    f << "  \"stage_timing\": " << (a.stage_timing ? "true" : "false") << ",\n";
    f << "  \"fps\": " << m.fps() << ",\n";
    f << "  \"stages\": {\n";
    const auto summary = m.summary();
    const auto order = m.order();
    for (size_t i = 0; i < order.size(); ++i) {
        const auto& s = summary.at(order[i]);
        f << "    \"" << order[i] << "\": {\"mean_ms\": " << s.mean_ms << ", \"p50_ms\": " << s.p50_ms
          << ", \"p95_ms\": " << s.p95_ms << ", \"max_ms\": " << s.max_ms << ", \"samples\": " << s.samples << "}"
          << (i + 1 < order.size() ? ",\n" : "\n");
    }
    f << "  },\n";
    f << "  \"environment\": {\"tensorrt\": \"" << TrtEngine::version() << "\", \"cuda_runtime\": \""
      << cuda_runtime_version() << "\", \"gpu\": \"" << gpu_name() << "\", \"opencv\": \"" << CV_VERSION << "\"}\n";
    f << "}\n";
}

void dump_detections(const std::vector<Detection>& dets, const std::string& path) {
    std::ofstream f(path);
    f << "[\n";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        f << "  {\"x1\": " << d.x1 << ", \"y1\": " << d.y1 << ", \"x2\": " << d.x2 << ", \"y2\": " << d.y2
          << ", \"confidence\": " << d.confidence << ", \"class_id\": " << d.class_id << ", \"class_name\": \""
          << d.class_name << "\"}" << (i + 1 < dets.size() ? ",\n" : "\n");
    }
    f << "]\n";
}

}  // namespace

int main(int argc, char** argv) try {
    const Args args = parse_args(argc, argv);

    const std::string names_path = args.names.empty() ? sidecar_names_path(args.engine) : args.names;
    PerformanceMetrics metrics;
    TensorRTDetector detector(args.engine, args.confidence, load_class_names(names_path),
                              args.stage_timing ? &metrics : nullptr);

    std::vector<Detection> first_measured;
    int processed = 0;
    const int total = args.warmup + args.frames;

    auto account = [&](std::vector<Detection>& dets) {
        metrics.count_frame();
        ++processed;
        if (processed == args.warmup) metrics.reset();
        if (processed == args.warmup + 1) first_measured = dets;
    };

    if (args.decoder == "nvdec") {
        NvVideoDecoder decoder(args.source);
        std::printf("nvdec: %s %dx%d -> %s\n", decoder.codec_name().c_str(), decoder.width(), decoder.height(),
                    decoder.hw_pixel_format().c_str());
        GpuFrame gpu;
        while (processed < total) {
            bool ok;
            {
                ScopedTimer t(metrics, "decode");
                ok = decoder.next(gpu);
            }
            if (!ok) {  // loop short clips
                decoder.reopen();
                continue;
            }
            std::vector<Detection> dets;
            {
                ScopedTimer t(metrics, PerformanceMetrics::kEndToEnd);
                dets = detector.detect(gpu);  // synchronises before returning: frame can be recycled
            }
            account(dets);
        }
    } else {
        // A still image (jpg/png/bmp) is decoded once and re-used as every frame.
        std::string ext = args.source.size() > 4 ? args.source.substr(args.source.size() - 4) : "";
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        const bool still_image = ext == ".jpg" || ext == "jpeg" || ext == ".png" || ext == ".bmp";
        cv::Mat still;
        cv::VideoCapture cap;
        if (still_image) {
            still = cv::imread(args.source);
            if (still.empty()) throw std::runtime_error("cannot read image: " + args.source);
        } else if (!cap.open(args.source)) {
            throw std::runtime_error("cannot open source: " + args.source);
        }

        cv::Mat frame;
        while (processed < total) {
            bool ok;
            {
                ScopedTimer t(metrics, "decode");
                if (still_image) {
                    still.copyTo(frame);
                    ok = true;
                } else {
                    ok = cap.read(frame);
                }
            }
            if (!ok) {  // loop short clips
                cap.release();
                cap.open(args.source);
                if (!cap.isOpened()) throw std::runtime_error("cannot reopen source: " + args.source);
                continue;
            }
            if (args.pinned) {
                // Decode lands in OpenCV's own buffer; keep the frame in pinned memory so the
                // H2D copy is a DMA. One host memcpy (~6 MB at 1080p) instead of a staged copy.
                cv::Mat pinned = detector.pinned_frame(frame.rows, frame.cols);
                frame.copyTo(pinned);
                frame = pinned;
            }

            std::vector<Detection> dets;
            {
                ScopedTimer t(metrics, PerformanceMetrics::kEndToEnd);
                dets = detector.detect(frame);
            }
            account(dets);
        }
    }

    print_table(metrics);
    const std::string stamp = utc_stamp();
    std::string out = args.out;
    if (out.empty())
        out = "benchmarks/results/cpp_tensorrt_" + args.decoder + (args.label.empty() ? "" : "_" + args.label) + "_" +
              stamp + ".json";
    write_report(args, metrics, stamp, out);
    std::printf("saved: %s\n", out.c_str());
    if (!args.dump_detections.empty()) {
        dump_detections(first_measured, args.dump_detections);
        std::printf("detections of first measured frame: %zu -> %s\n", first_measured.size(),
                    args.dump_detections.c_str());
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
