// CTest for the JSON writers: report schema shared with the Python benchmark, detection dumps,
// track JSONL lines and the UTC stamp. Checked on text (no JSON library in the runtime).
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "edgevision/report.hpp"
#include "expect.hpp"

using namespace edgevision;

namespace {

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

int count(const std::string& text, char c) {
    int n = 0;
    for (const char x : text) n += x == c;
    return n;
}

StreamResult stream(const std::string& source, size_t frames, double e2e_ms) {
    StreamResult r;
    r.source = source;
    for (size_t i = 0; i < frames; ++i) {
        r.metrics.record("decode", 0.5);
        r.metrics.record(PerformanceMetrics::kEndToEnd, e2e_ms);
        r.metrics.count_frame();
    }
    r.measured_seconds = static_cast<double>(frames) * e2e_ms / 1000.0;
    return r;
}

void test_report_schema() {
    Args a;
    a.engine = "m.engine";
    a.sources = {"a.mp4", "b.mp4"};
    a.decoder = "nvdec";
    a.streams = 2;
    a.label = "fp16";
    a.frames = 4;
    a.warmup = 1;
    a.track = true;
    const std::vector<StreamResult> streams = {stream("a.mp4", 4, 2.0), stream("b.mp4", 4, 4.0)};
    PerformanceMetrics merged;
    for (const auto& s : streams) s.metrics.merge_into(merged);

    std::ostringstream out;
    write_report(out, a, streams, merged, 750.0, "20260921T000000Z", {"10.16", "13.2", "Tesla T4", "4.10"});
    const std::string json = out.str();

    for (const char* field : {"\"timestamp_utc\": \"20260921T000000Z\"",
                              "\"backend\": \"cpp_tensorrt\"",
                              "\"label\": \"fp16\"",
                              "\"model_path\": \"m.engine\"",
                              "\"source\": \"a.mp4\"",
                              "\"decoder\": \"nvdec\"",
                              "\"streams\": 2",
                              "\"mode\": \"independent\"",
                              "\"tracking\": true",
                              "\"frames\": 4",
                              "\"warmup\": 1",
                              "\"pinned_host_frame\": true",
                              "\"stage_timing\": true",
                              "\"aggregate_fps\": 750",
                              "\"stages\": {",
                              "\"decode\": {",
                              "\"end_to_end\": {",
                              "\"per_stream\": [",
                              "\"source\": \"b.mp4\"",
                              "\"sustained_fps\": 250",
                              "\"tensorrt\": \"10.16\"",
                              "\"gpu\": \"Tesla T4\"",
                              "\"opencv\": \"4.10\""})
        EXPECT(contains(json, field), "report lacks %s", field);
    EXPECT(contains(json, "\"fps\": 333.33"), "fps from the merged end_to_end mean (3 ms): %s",
           json.substr(json.find("\"fps\""), 16).c_str());
    EXPECT(count(json, '{') == count(json, '}'), "braces balance (%d vs %d)", count(json, '{'), count(json, '}'));
    EXPECT(count(json, '[') == count(json, ']'), "brackets balance");
    EXPECT(json.back() == '\n' && json[json.size() - 2] == '}', "ends with the closing brace");
}

void test_batched_mode_label() {
    Args a;
    a.engine = "m";
    a.sources = {"a"};
    a.batched = true;
    std::ostringstream out;
    write_report(out, a, {stream("a", 1, 1.0)}, PerformanceMetrics{}, 0.0, "S", {});
    EXPECT(contains(out.str(), "\"mode\": \"batched\""), "batched mode");
    EXPECT(contains(out.str(), "\"stages\": {\n  }"), "no stages when the window is empty");
}

void test_dump_detections() {
    std::ostringstream out;
    dump_detections(out, {{1, 2, 3, 4, 0.9f, 0, "person"}, {5, 6, 7, 8, 0.8f, 2, "car"}});
    const std::string json = out.str();
    EXPECT(json.rfind("[\n", 0) == 0 && contains(json, "]\n"), "array");
    EXPECT(contains(json,
                    "{\"x1\": 1, \"y1\": 2, \"x2\": 3, \"y2\": 4, \"confidence\": 0.9, \"class_id\": 0, "
                    "\"class_name\": \"person\"},"),
           "first row: %s", json.c_str());
    EXPECT(contains(json, "\"class_name\": \"car\"}\n]"), "last row has no trailing comma");
    std::ostringstream empty;
    dump_detections(empty, {});
    EXPECT(empty.str() == "[\n]\n", "empty array: '%s'", empty.str().c_str());
}

void test_dump_tracks_line() {
    Track t{};
    t.id = 7;
    t.x1 = 10;
    t.y1 = 20;
    t.x2 = 30;
    t.y2 = 40;
    t.score = 0.75f;
    t.class_name = "person";
    std::ostringstream out;
    dump_tracks_line(out, 3, {t, t});
    EXPECT(out.str() ==
               "{\"frame\": 3, \"tracks\": [{\"id\": 7, \"x1\": 10, \"y1\": 20, \"x2\": 30, \"y2\": 40, "
               "\"score\": 0.75, \"class_name\": \"person\"}, {\"id\": 7, \"x1\": 10, \"y1\": 20, "
               "\"x2\": 30, \"y2\": 40, \"score\": 0.75, \"class_name\": \"person\"}]}\n",
           "line: %s", out.str().c_str());
    std::ostringstream none;
    dump_tracks_line(none, 1, {});
    EXPECT(none.str() == "{\"frame\": 1, \"tracks\": []}\n", "empty tracks: %s", none.str().c_str());
}

void test_utc_stamp_format() {
    const std::string s = utc_stamp();
    EXPECT(s.size() == 16 && s[8] == 'T' && s[15] == 'Z', "stamp %s", s.c_str());
    for (size_t i = 0; i < s.size(); ++i)
        if (i != 8 && i != 15)
            EXPECT(std::isdigit(static_cast<unsigned char>(s[i])), "digit at %zu in %s", i, s.c_str());
}

void test_write_to_unwritable_path_throws() {
    Args a;
    a.engine = "m";
    a.sources = {"a"};
    EXPECT_THROWS(write_report("/nonexistent-dir/x/report.json", a, {}, PerformanceMetrics{}, 0.0, "S", {}),
                  "cannot write");
    EXPECT_THROWS(dump_detections("/nonexistent-dir/x/d.json", {}), "cannot write");
}

}  // namespace

int main() {
    test_report_schema();
    test_batched_mode_label();
    test_dump_detections();
    test_dump_tracks_line();
    test_utc_stamp_format();
    test_write_to_unwritable_path_throws();
    return test_result("report");
}
