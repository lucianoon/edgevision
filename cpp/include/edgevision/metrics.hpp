#pragma once

#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace edgevision {

struct StageStats {
    double mean_ms, p50_ms, p95_ms, max_ms;
    size_t samples;
};

// Per-stage latency recorder; same summary fields as python/edgevision/metrics.py.
class PerformanceMetrics {
public:
    void record(const std::string& stage, double latency_ms);
    void reset();
    size_t frame_count() const { return frames_; }
    void count_frame() { ++frames_; }

    std::map<std::string, StageStats> summary() const;  // insertion order not kept; see order()
    std::vector<std::string> order() const { return order_; }
    double fps() const;  // from end_to_end mean
    double average_ms(const std::string& stage) const;

    // Append every sample of every stage into `other` (multi-stream merged view).
    void merge_into(PerformanceMetrics& other) const;

    static constexpr const char* kEndToEnd = "end_to_end";

private:
    std::map<std::string, std::vector<double>> stages_;
    std::vector<std::string> order_;
    size_t frames_ = 0;
};

// RAII wall-clock timer that records into metrics on destruction.
class ScopedTimer {
public:
    ScopedTimer(PerformanceMetrics& metrics, std::string stage)
        : metrics_(metrics), stage_(std::move(stage)), start_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        auto elapsed = std::chrono::steady_clock::now() - start_;
        metrics_.record(stage_, std::chrono::duration<double, std::milli>(elapsed).count());
    }

private:
    PerformanceMetrics& metrics_;
    std::string stage_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace edgevision
