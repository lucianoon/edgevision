#include "edgevision/report.hpp"

#include <cstdio>
#include <ctime>
#include <fstream>
#include <stdexcept>

namespace edgevision {

namespace {

const char* json_bool(bool v) {
    return v ? "true" : "false";
}

void write_stages(std::ostream& f, const PerformanceMetrics& m, const char* indent) {
    const auto summary = m.summary();
    const auto order = m.order();
    for (size_t i = 0; i < order.size(); ++i) {
        const auto& s = summary.at(order[i]);
        f << indent << "\"" << order[i] << "\": {\"mean_ms\": " << s.mean_ms << ", \"p50_ms\": " << s.p50_ms
          << ", \"p95_ms\": " << s.p95_ms << ", \"max_ms\": " << s.max_ms << ", \"samples\": " << s.samples << "}"
          << (i + 1 < order.size() ? ",\n" : "\n");
    }
}

std::ofstream open_for_write(const std::string& path) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot write " + path);
    return f;
}

}  // namespace

std::string utc_stamp() {
    const std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%dT%H%M%SZ", std::gmtime(&now));
    return buf;
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

void write_report(std::ostream& f, const Args& a, const std::vector<StreamResult>& streams,
                  const PerformanceMetrics& merged, double aggregate_fps, const std::string& stamp,
                  const Environment& env) {
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
    f << "  \"tracking\": " << json_bool(a.track) << ",\n";
    f << "  \"frames\": " << a.frames << ",\n";
    f << "  \"warmup\": " << a.warmup << ",\n";
    f << "  \"pinned_host_frame\": " << json_bool(a.pinned) << ",\n";
    f << "  \"stage_timing\": " << json_bool(a.stage_timing) << ",\n";
    f << "  \"fps\": " << merged.fps() << ",\n";
    f << "  \"aggregate_fps\": " << aggregate_fps << ",\n";
    f << "  \"stages\": {\n";
    write_stages(f, merged, "    ");
    f << "  },\n";
    f << "  \"per_stream\": [\n";
    for (size_t i = 0; i < streams.size(); ++i) {
        const auto& s = streams[i];
        const auto frames = static_cast<double>(s.metrics.frame_count());
        const double fps = s.measured_seconds > 0 ? frames / s.measured_seconds : 0.0;
        f << "    {\"source\": \"" << s.source << "\", \"frames\": " << s.metrics.frame_count()
          << ", \"measured_seconds\": " << s.measured_seconds << ", \"sustained_fps\": " << fps
          << ", \"e2e_mean_ms\": " << s.metrics.average_ms(PerformanceMetrics::kEndToEnd) << "}"
          << (i + 1 < streams.size() ? ",\n" : "\n");
    }
    f << "  ],\n";
    f << "  \"environment\": {\"tensorrt\": \"" << env.tensorrt << "\", \"cuda_runtime\": \"" << env.cuda_runtime
      << "\", \"gpu\": \"" << env.gpu << "\", \"opencv\": \"" << env.opencv << "\"}\n";
    f << "}\n";
}

void write_report(const std::string& path, const Args& a, const std::vector<StreamResult>& streams,
                  const PerformanceMetrics& merged, double aggregate_fps, const std::string& stamp,
                  const Environment& env) {
    auto f = open_for_write(path);
    write_report(f, a, streams, merged, aggregate_fps, stamp, env);
}

void dump_detections(std::ostream& f, const std::vector<Detection>& dets) {
    f << "[\n";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        f << "  {\"x1\": " << d.x1 << ", \"y1\": " << d.y1 << ", \"x2\": " << d.x2 << ", \"y2\": " << d.y2
          << ", \"confidence\": " << d.confidence << ", \"class_id\": " << d.class_id << ", \"class_name\": \""
          << d.class_name << "\"}" << (i + 1 < dets.size() ? ",\n" : "\n");
    }
    f << "]\n";
}

void dump_detections(const std::string& path, const std::vector<Detection>& dets) {
    auto f = open_for_write(path);
    dump_detections(f, dets);
}

void dump_tracks_line(std::ostream& f, int frame, const std::vector<Track>& tracks) {
    f << "{\"frame\": " << frame << ", \"tracks\": [";
    for (size_t i = 0; i < tracks.size(); ++i) {
        const auto& t = tracks[i];
        f << (i ? ", " : "") << "{\"id\": " << t.id << ", \"x1\": " << t.x1 << ", \"y1\": " << t.y1
          << ", \"x2\": " << t.x2 << ", \"y2\": " << t.y2 << ", \"score\": " << t.score << ", \"class_name\": \""
          << t.class_name << "\"}";
    }
    f << "]}\n";
}

}  // namespace edgevision
