// edgevision_trt: C++ TensorRT runtime benchmark. Same stages, table and JSON report
// schema as `python -m edgevision.benchmark`, so results can be compared side by side.
//
//   edgevision_trt --engine models/tensorrt/yolo26n_nms_fp16.engine \
//                  --source videos/pedestrian_area_1080p25.webm --frames 300 --warmup 30 --label fp16_nmsgraph
//
// Multi-stream (Sprint 6): --streams N runs N independent pipelines (one decoder + one
// TensorRT execution context each, on their own threads) and reports per-stream and
// aggregate throughput. --source may be given several times (one per stream, cycled).

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
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "edgevision/batch_pipeline.hpp"
#include "edgevision/detector.hpp"
#include "edgevision/metrics.hpp"
#include "edgevision/names.hpp"
#include "edgevision/tracker.hpp"
#include "edgevision/video_decoder.hpp"

using namespace edgevision;

namespace {

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
    float confidence = 0.5f;
    bool pinned = true;
    bool stage_timing = true;
    bool batched = false;  // phase B: one batch-N inference per round for all streams
    bool track = false;    // Sprint 8: ByteTrack per stream
    float track_thresh = 0.5f;
    int track_buffer = 30;
    std::string dump_tracks;  // JSONL of stream 0's tracks for the measured frames
    std::string render;       // annotated video (opencv decoder only)
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
            std::cout << "usage: edgevision_trt --engine E [--source S]... [--decoder opencv|nvdec] [--streams N] [--batched]\n"
                         "       [--frames N] [--warmup N] [--label L] [--names names.json] [--out report.json]\n"
                         "       [--dump-detections dets.json] [--confidence 0.5] [--no-pinned] [--no-stage-timing]\n"
                         "  --batched: all streams share one batch-N engine (engine exported with batch=N)\n"
                         "  --track [--track-thresh 0.5] [--track-buffer 30] [--dump-tracks f.jsonl] [--render out.mp4]\n"
                         "           ByteTrack per stream; use an engine exported with conf 0.1 so weak detections exist\n";
            std::exit(0);
        } else throw std::runtime_error("unknown argument: " + k);
    }
    if (a.engine.empty()) throw std::runtime_error("--engine is required");
    if (a.sources.empty()) a.sources.push_back("videos/pedestrian_area_1080p25.webm");
    if (a.decoder != "opencv" && a.decoder != "nvdec") throw std::runtime_error("--decoder must be opencv or nvdec");
    if (a.streams < 1) throw std::runtime_error("--streams must be >= 1");
    if (!a.render.empty() && a.decoder != "opencv") throw std::runtime_error("--render needs --decoder opencv (host frames)");
    if (!a.render.empty() && !a.track) a.track = true;
    return a;
}

void draw_tracks(cv::Mat& frame, const std::vector<Track>& tracks) {
    for (const auto& t : tracks) {
        const cv::Scalar color((t.id * 67) % 256, (t.id * 131) % 256, (t.id * 197) % 256);
        cv::rectangle(frame, cv::Point(int(t.x1), int(t.y1)), cv::Point(int(t.x2), int(t.y2)), color, 2);
        const std::string label = "#" + std::to_string(t.id) + " " + t.class_name;
        cv::putText(frame, label, cv::Point(int(t.x1), std::max(int(t.y1) - 6, 14)), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2);
    }
}

void dump_tracks_line(std::ofstream& f, int frame, const std::vector<Track>& tracks) {
    f << "{\"frame\": " << frame << ", \"tracks\": [";
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i];
        f << (i ? ", " : "") << "{\"id\": " << t.id << ", \"x1\": " << t.x1 << ", \"y1\": " << t.y1 << ", \"x2\": " << t.x2
          << ", \"y2\": " << t.y2 << ", \"score\": " << t.score << ", \"class_name\": \"" << t.class_name << "\"}";
    }
    f << "]}\n";
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

bool is_still_image(const std::string& source) {
    std::string ext = source.size() > 4 ? source.substr(source.size() - 4) : "";
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == "jpeg" || ext == ".png" || ext == ".bmp";
}

// One stream: its own decoder and its own detector (TensorRT execution context).
struct StreamResult {
    PerformanceMetrics metrics;
    std::vector<Detection> first_measured;
    double measured_seconds = 0.0;  // wall time of the measured frames (after warm-up)
    std::string source;
    std::string error;
    int index = 0;
    int unique_track_ids = 0;
};

void run_stream(const Args& args, const std::map<int, std::string>& names, StreamResult& result) {
    PerformanceMetrics& metrics = result.metrics;
    TensorRTDetector detector(args.engine, args.confidence, names, args.stage_timing ? &metrics : nullptr);

    TrackerParams tp;
    tp.track_thresh = args.track_thresh;
    tp.track_buffer = args.track_buffer;
    ByteTracker tracker(tp);
    std::ofstream dump;
    if (args.track && result.index == 0 && !args.dump_tracks.empty()) dump.open(args.dump_tracks);
    cv::VideoWriter writer;

    int processed = 0;
    const int total = args.warmup + args.frames;
    std::chrono::steady_clock::time_point measure_start;
    // Tracking runs inside the end-to-end window (it is part of the frame's work) and is
    // also timed on its own as the "tracking" stage.
    auto track_step = [&](std::vector<Detection>& dets, cv::Mat* host_frame) {
        if (!args.track) return;
        std::vector<Track> tracks;
        {
            ScopedTimer t(metrics, "tracking");
            tracks = tracker.update(dets);
        }
        if (dump.is_open() && processed >= args.warmup) dump_tracks_line(dump, processed - args.warmup + 1, tracks);
        if (host_frame && !args.render.empty()) {
            if (!writer.isOpened())
                writer.open(args.render, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 25.0, host_frame->size());
            draw_tracks(*host_frame, tracks);
            writer.write(*host_frame);
        }
    };
    auto account = [&](std::vector<Detection>& dets) {
        metrics.count_frame();
        ++processed;
        if (processed == args.warmup) {
            metrics.reset();
            measure_start = std::chrono::steady_clock::now();
        }
        if (processed == args.warmup + 1) result.first_measured = dets;
    };
    if (args.warmup == 0) measure_start = std::chrono::steady_clock::now();

    if (args.decoder == "nvdec") {
        NvVideoDecoder decoder(result.source);
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
                track_step(dets, nullptr);
            }
            account(dets);
        }
    } else {
        const bool still_image = is_still_image(result.source);
        cv::Mat still;
        cv::VideoCapture cap;
        if (still_image) {
            still = cv::imread(result.source);
            if (still.empty()) throw std::runtime_error("cannot read image: " + result.source);
        } else if (!cap.open(result.source)) {
            throw std::runtime_error("cannot open source: " + result.source);
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
                cap.open(result.source);
                if (!cap.isOpened()) throw std::runtime_error("cannot reopen source: " + result.source);
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
                track_step(dets, &frame);
            }
            account(dets);
        }
    }
    result.measured_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - measure_start).count();
    result.unique_track_ids = args.track ? tracker.next_id() - 1 : 0;
    if (writer.isOpened()) writer.release();
}

void print_table(const char* title, const PerformanceMetrics& m) {
    const auto summary = m.summary();
    std::printf("\n%s  fps=%.1f\n", title, m.fps());
    std::printf("%-14s%9s%9s%9s%9s  n\n", "stage", "mean", "p50", "p95", "max");
    for (const auto& name : m.order()) {
        const auto& s = summary.at(name);
        std::printf("%-14s%9.1f%9.1f%9.1f%9.1f  %zu\n", name.c_str(), s.mean_ms, s.p50_ms, s.p95_ms, s.max_ms,
                    s.samples);
    }
}

void write_stages(std::ofstream& f, const PerformanceMetrics& m, const char* indent) {
    const auto summary = m.summary();
    const auto order = m.order();
    for (size_t i = 0; i < order.size(); ++i) {
        const auto& s = summary.at(order[i]);
        f << indent << "\"" << order[i] << "\": {\"mean_ms\": " << s.mean_ms << ", \"p50_ms\": " << s.p50_ms
          << ", \"p95_ms\": " << s.p95_ms << ", \"max_ms\": " << s.max_ms << ", \"samples\": " << s.samples << "}"
          << (i + 1 < order.size() ? ",\n" : "\n");
    }
}

void write_report(const Args& a, const std::vector<StreamResult>& streams, const PerformanceMetrics& merged,
                  double aggregate_fps, const std::string& stamp, const std::string& path) {
    std::ofstream f(path);
    f << "{\n";
    f << "  \"timestamp_utc\": \"" << stamp << "\",\n";
    f << "  \"backend\": \"cpp_tensorrt\",\n";
    f << "  \"label\": \"" << a.label << "\",\n";
    f << "  \"model_path\": \"" << a.engine << "\",\n";
    f << "  \"confidence\": " << a.confidence << ",\n";
    f << "  \"source\": \"" << a.sources.front() << "\",\n";
    f << "  \"decoder\": \"" << a.decoder << "\",\n";
    f << "  \"streams\": " << a.streams << ",\n";
    f << "  \"mode\": \"" << (a.batched ? "batched" : "independent") << "\",\n";
    f << "  \"tracking\": " << (a.track ? "true" : "false") << ",\n";
    f << "  \"frames\": " << a.frames << ",\n";
    f << "  \"warmup\": " << a.warmup << ",\n";
    f << "  \"pinned_host_frame\": " << (a.pinned ? "true" : "false") << ",\n";
    f << "  \"stage_timing\": " << (a.stage_timing ? "true" : "false") << ",\n";
    f << "  \"fps\": " << merged.fps() << ",\n";
    f << "  \"aggregate_fps\": " << aggregate_fps << ",\n";
    f << "  \"stages\": {\n";
    write_stages(f, merged, "    ");
    f << "  },\n";
    f << "  \"per_stream\": [\n";
    for (size_t i = 0; i < streams.size(); ++i) {
        const auto& s = streams[i];
        const double fps = s.measured_seconds > 0 ? s.metrics.frame_count() / s.measured_seconds : 0.0;
        f << "    {\"source\": \"" << s.source << "\", \"frames\": " << s.metrics.frame_count()
          << ", \"measured_seconds\": " << s.measured_seconds << ", \"sustained_fps\": " << fps
          << ", \"e2e_mean_ms\": " << s.metrics.average_ms(PerformanceMetrics::kEndToEnd) << "}"
          << (i + 1 < streams.size() ? ",\n" : "\n");
    }
    f << "  ],\n";
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
    const auto names = args.names.empty() ? load_class_names_for_engine(args.engine) : load_class_names(args.names);

    std::vector<StreamResult> results(args.streams);
    for (int i = 0; i < args.streams; ++i) {
        results[i].source = args.sources[i % args.sources.size()];
        results[i].index = i;
    }

    const auto wall_start = std::chrono::steady_clock::now();
    if (args.batched) {
        BatchOptions opt;
        opt.engine_path = args.engine;
        for (const auto& r : results) opt.sources.push_back(r.source);
        opt.nvdec = args.decoder == "nvdec";
        opt.confidence = args.confidence;
        opt.frames = args.frames;
        opt.warmup = args.warmup;
        BatchResult batch = run_batched(opt, names);

        // Map onto the per-stream report: every stream advances one frame per round.
        for (int i = 0; i < args.streams; ++i) {
            results[i].metrics = batch.workers[i];          // decode, preprocess (own thread)
            batch.rounds.merge_into(results[i].metrics);    // gather/inference/postprocess/e2e of the round
            results[i].measured_seconds = batch.measured_seconds;
            results[i].first_measured = batch.first_measured[i];
        }
        const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
        std::printf("\nbatched streams=%d decoder=%s  aggregate=%.1f fps (%d rounds, %.1fs measured, %.1fs wall)\n",
                    args.streams, args.decoder.c_str(), batch.aggregate_fps(), batch.rounds_measured,
                    batch.measured_seconds, wall);
        print_table("per round (all streams)", batch.rounds);
        std::printf("worker means: decode %.2f ms, preprocess %.2f ms\n", results[0].metrics.average_ms("decode"),
                    results[0].metrics.average_ms("preprocess"));

        PerformanceMetrics merged;
        for (const auto& r : results) r.metrics.merge_into(merged);
        const std::string stamp = utc_stamp();
        std::string out = args.out;
        if (out.empty())
            out = "benchmarks/results/cpp_tensorrt_" + args.decoder + "_b" + std::to_string(args.streams) +
                  (args.label.empty() ? "" : "_" + args.label) + "_" + stamp + ".json";
        write_report(args, results, batch.rounds, batch.aggregate_fps(), stamp, out);
        std::printf("saved: %s\n", out.c_str());
        if (!args.dump_detections.empty()) {
            dump_detections(results[0].first_measured, args.dump_detections);
            std::printf("detections of first measured round (stream 0): %zu -> %s\n", results[0].first_measured.size(),
                        args.dump_detections.c_str());
        }
        return 0;
    }
    if (args.streams == 1) {
        run_stream(args, names, results[0]);
    } else {
        std::vector<std::thread> threads;
        for (auto& r : results) {
            StreamResult* result = &r;  // explicit pointer: never capture the loop variable by reference
            threads.emplace_back([&args, &names, result] {
                try {
                    run_stream(args, names, *result);
                } catch (const std::exception& e) {
                    result->error = e.what();
                }
            });
        }
        for (auto& t : threads) t.join();
        for (const auto& r : results)
            if (!r.error.empty()) throw std::runtime_error("stream " + r.source + ": " + r.error);
    }
    const double wall_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();

    // Merged view: all measured samples of all streams in one distribution.
    PerformanceMetrics merged;
    size_t total_frames = 0;
    double aggregate_fps = 0.0;
    for (const auto& r : results) {
        r.metrics.merge_into(merged);
        total_frames += r.metrics.frame_count();
        if (r.measured_seconds > 0) aggregate_fps += r.metrics.frame_count() / r.measured_seconds;
    }

    if (args.track)
        std::printf("tracking: %d unique ids on stream 0 over %d frames%s\n", results[0].unique_track_ids,
                    args.warmup + args.frames, args.dump_tracks.empty() ? "" : (" -> " + args.dump_tracks).c_str());
    if (args.streams == 1) {
        print_table("backend=cpp_tensorrt", merged);
    } else {
        std::printf("\nstreams=%d decoder=%s  aggregate=%.1f fps (%zu frames, %.1fs wall)\n", args.streams,
                    args.decoder.c_str(), aggregate_fps, total_frames, wall_seconds);
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::printf("  stream %zu: %.1f fps sustained, e2e mean %.2f ms, decode mean %.2f ms\n", i,
                        r.metrics.frame_count() / r.measured_seconds, r.metrics.average_ms(PerformanceMetrics::kEndToEnd),
                        r.metrics.average_ms("decode"));
        }
        print_table("all streams merged", merged);
    }

    const std::string stamp = utc_stamp();
    std::string out = args.out;
    if (out.empty())
        out = "benchmarks/results/cpp_tensorrt_" + args.decoder +
              (args.streams > 1 ? "_s" + std::to_string(args.streams) : "") +
              (args.label.empty() ? "" : "_" + args.label) + "_" + stamp + ".json";
    write_report(args, results, merged, aggregate_fps, stamp, out);
    std::printf("saved: %s\n", out.c_str());
    if (!args.dump_detections.empty()) {
        dump_detections(results[0].first_measured, args.dump_detections);
        std::printf("detections of first measured frame (stream 0): %zu -> %s\n", results[0].first_measured.size(),
                    args.dump_detections.c_str());
    }
    return 0;
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
