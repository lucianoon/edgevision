#include "edgevision/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace edgevision {

void PerformanceMetrics::record(const std::string& stage, double latency_ms) {
    auto it = stages_.find(stage);
    if (it == stages_.end()) {
        order_.push_back(stage);
        it = stages_.emplace(stage, std::vector<double>{}).first;
    }
    it->second.push_back(latency_ms);
}

void PerformanceMetrics::reset() {
    stages_.clear();
    order_.clear();
    frames_ = 0;
}

static double percentile(std::vector<double> values, double pct) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    size_t index = static_cast<size_t>(std::lround(pct / 100.0 * (n - 1)));
    index = std::min(index, n - 1);
    return values[index];
}

std::map<std::string, StageStats> PerformanceMetrics::summary() const {
    std::map<std::string, StageStats> result;
    for (const auto& [name, values] : stages_) {
        if (values.empty()) continue;
        StageStats s;
        s.samples = values.size();
        s.mean_ms = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
        s.p50_ms = percentile(values, 50.0);
        s.p95_ms = percentile(values, 95.0);
        s.max_ms = *std::max_element(values.begin(), values.end());
        result[name] = s;
    }
    return result;
}

double PerformanceMetrics::average_ms(const std::string& stage) const {
    auto it = stages_.find(stage);
    if (it == stages_.end() || it->second.empty()) return 0.0;
    return std::accumulate(it->second.begin(), it->second.end(), 0.0) / it->second.size();
}

void PerformanceMetrics::merge_into(PerformanceMetrics& other) const {
    for (const auto& name : order_)
        for (double v : stages_.at(name)) other.record(name, v);
    other.frames_ += frames_;
}

double PerformanceMetrics::fps() const {
    auto it = stages_.find(kEndToEnd);
    if (it == stages_.end() || it->second.empty()) return 0.0;
    const double mean = std::accumulate(it->second.begin(), it->second.end(), 0.0) / it->second.size();
    return mean > 0 ? 1000.0 / mean : 0.0;
}

}  // namespace edgevision
