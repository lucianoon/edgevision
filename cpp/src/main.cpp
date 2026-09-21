// edgevision_trt: C++ TensorRT runtime benchmark. Same stages, table and JSON report
// schema as `edgevision-benchmark` (python/edgevision/benchmark.py), so results can be
// compared side by side.
//
//   edgevision_trt --engine models/tensorrt/yolo26n_nms_fp16.engine \
//                  --source videos/pedestrian_area_1080p25.webm --frames 300 --warmup 30 --label fp16_nmsgraph
//
// Multi-stream (Sprint 6): --streams N runs N independent pipelines (one decoder + one
// TensorRT execution context each, on their own threads) and reports per-stream and
// aggregate throughput; --batched shares one batch-N execution instead. --source may be
// given several times (one per stream, cycled). --track adds ByteTrack per stream.
//
// This file only wires things together: cli.cpp parses, stream_runner.cpp runs one stream,
// batch_pipeline.cpp runs the batched variant, report.cpp prints and writes the JSON.

#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "edgevision/batch_pipeline.hpp"
#include "edgevision/cli.hpp"
#include "edgevision/metrics.hpp"
#include "edgevision/names.hpp"
#include "edgevision/report.hpp"
#include "edgevision/stream_runner.hpp"

using namespace edgevision;

namespace {

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void save_outputs(const Args& args, const std::vector<StreamResult>& results, const PerformanceMetrics& reported,
                  double aggregate_fps, const char* first_measured_what) {
    const std::string stamp = utc_stamp();
    const std::string out = default_report_path(args, stamp);
    write_report(out, args, results, reported, aggregate_fps, stamp, runtime_environment());
    std::printf("saved: %s\n", out.c_str());
    if (!args.dump_detections.empty()) {
        dump_detections(args.dump_detections, results[0].first_measured);
        std::printf("detections of first measured %s (stream 0): %zu -> %s\n", first_measured_what,
                    results[0].first_measured.size(), args.dump_detections.c_str());
    }
}

int run_batched_mode(const Args& args, const std::map<int, std::string>& names, std::vector<StreamResult>& results) {
    const auto wall_start = std::chrono::steady_clock::now();
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
        results[i].metrics = batch.workers[i];        // decode, preprocess (own thread)
        batch.rounds.merge_into(results[i].metrics);  // gather/inference/postprocess/e2e of the round
        results[i].measured_seconds = batch.measured_seconds;
        results[i].first_measured = batch.first_measured[i];
    }
    std::printf("\nbatched streams=%d decoder=%s  aggregate=%.1f fps (%d rounds, %.1fs measured, %.1fs wall)\n",
                args.streams, args.decoder.c_str(), batch.aggregate_fps(), batch.rounds_measured,
                batch.measured_seconds, seconds_since(wall_start));
    print_table("per round (all streams)", batch.rounds);
    std::printf("worker means: decode %.2f ms, preprocess %.2f ms\n", results[0].metrics.average_ms("decode"),
                results[0].metrics.average_ms("preprocess"));

    save_outputs(args, results, batch.rounds, batch.aggregate_fps(), "round");
    return 0;
}

int run_independent_mode(const Args& args, const std::map<int, std::string>& names,
                         std::vector<StreamResult>& results) {
    const auto wall_start = std::chrono::steady_clock::now();
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
    const double wall_seconds = seconds_since(wall_start);

    // Merged view: all measured samples of all streams in one distribution.
    PerformanceMetrics merged;
    size_t total_frames = 0;
    double aggregate_fps = 0.0;
    for (const auto& r : results) {
        r.metrics.merge_into(merged);
        total_frames += r.metrics.frame_count();
        if (r.measured_seconds > 0) aggregate_fps += static_cast<double>(r.metrics.frame_count()) / r.measured_seconds;
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
                        static_cast<double>(r.metrics.frame_count()) / r.measured_seconds,
                        r.metrics.average_ms(PerformanceMetrics::kEndToEnd), r.metrics.average_ms("decode"));
        }
        print_table("all streams merged", merged);
    }

    save_outputs(args, results, merged, aggregate_fps, "frame");
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    const Args args = parse_args(argc, argv);
    if (args.help) {
        std::cout << usage();
        return 0;
    }
    const auto names = args.names.empty() ? load_class_names_for_engine(args.engine) : load_class_names(args.names);

    std::vector<StreamResult> results(static_cast<size_t>(args.streams));
    for (int i = 0; i < args.streams; ++i) {
        results[static_cast<size_t>(i)].source = args.sources[static_cast<size_t>(i) % args.sources.size()];
        results[static_cast<size_t>(i)].index = i;
    }

    return args.batched ? run_batched_mode(args, names, results) : run_independent_mode(args, names, results);
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
}
