#include "edgevision/stream_runner.hpp"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "edgevision/detector.hpp"
#include "edgevision/metrics.hpp"
#include "edgevision/tracker.hpp"
#include "edgevision/trt_engine.hpp"
#include "edgevision/video_decoder.hpp"

namespace edgevision {

namespace {

void draw_tracks(cv::Mat& frame, const std::vector<Track>& tracks) {
    for (const auto& t : tracks) {
        const cv::Scalar color((t.id * 67) % 256, (t.id * 131) % 256, (t.id * 197) % 256);
        cv::rectangle(frame, cv::Point(int(t.x1), int(t.y1)), cv::Point(int(t.x2), int(t.y2)), color, 2);
        const std::string label = "#" + std::to_string(t.id) + " " + t.class_name;
        cv::putText(frame, label, cv::Point(int(t.x1), std::max(int(t.y1) - 6, 14)), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2);
    }
}

bool is_still_image(const std::string& source) {
    std::string ext = source.size() > 4 ? source.substr(source.size() - 4) : "";
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".jpg" || ext == "jpeg" || ext == ".png" || ext == ".bmp";
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

}  // namespace

Environment runtime_environment() {
    Environment env;
    env.tensorrt = TrtEngine::version();
    env.cuda_runtime = cuda_runtime_version();
    env.gpu = gpu_name();
    env.opencv = CV_VERSION;
    return env;
}

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
    result.measured_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - measure_start).count();
    result.unique_track_ids = args.track ? tracker.next_id() - 1 : 0;
    if (writer.isOpened()) writer.release();
}

}  // namespace edgevision
