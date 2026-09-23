// CTest for the command line of edgevision_trt: defaults, every flag, validation, report names.
#include <stdexcept>
#include <string>
#include <vector>

#include "edgevision/cli.hpp"
#include "edgevision/tracker.hpp"
#include "expect.hpp"

using namespace edgevision;

namespace {

Args parse(std::vector<const char*> argv) {
    argv.insert(argv.begin(), "edgevision_trt");
    return parse_args(static_cast<int>(argv.size()), argv.data());
}

void test_defaults() {
    const Args a = parse({"--engine", "e.engine"});
    EXPECT(a.engine == "e.engine", "engine");
    EXPECT(a.sources.size() == 1 && a.sources[0] == "videos/pedestrian_area_1080p25.webm", "default clip");
    EXPECT(a.decoder == "opencv" && a.streams == 1 && a.frames == 300 && a.warmup == 30, "defaults");
    EXPECT(a.pinned && a.stage_timing && !a.batched && !a.track && !a.help, "flags default");
    EXPECT(a.confidence == 0.5f && a.track_thresh == 0.5f && a.track_buffer == 30, "thresholds");
}

void test_every_flag() {
    const Args a = parse({"--engine",
                          "e",
                          "--source",
                          "a.mp4",
                          "--source",
                          "b.mp4",
                          "--decoder",
                          "nvdec",
                          "--names",
                          "n.json",
                          "--label",
                          "fp16",
                          "--out",
                          "r.json",
                          "--dump-detections",
                          "d.json",
                          "--frames",
                          "10",
                          "--warmup",
                          "2",
                          "--streams",
                          "4",
                          "--confidence",
                          "0.25",
                          "--no-pinned",
                          "--no-stage-timing",
                          "--batched",
                          "--track",
                          "--track-thresh",
                          "0.4",
                          "--track-buffer",
                          "15",
                          "--dump-tracks",
                          "t.jsonl"});
    EXPECT(a.sources.size() == 2 && a.sources[1] == "b.mp4", "two sources");
    EXPECT(a.decoder == "nvdec" && a.names == "n.json" && a.label == "fp16" && a.out == "r.json", "strings");
    EXPECT(a.dump_detections == "d.json" && a.dump_tracks == "t.jsonl", "dump paths");
    EXPECT(a.frames == 10 && a.warmup == 2 && a.streams == 4, "ints");
    EXPECT(a.confidence == 0.25f && a.track_thresh == 0.4f && a.track_buffer == 15, "floats / buffer");
    EXPECT(!a.pinned && !a.stage_timing && a.batched && a.track, "toggles");
}

void test_track_lowers_default_confidence() {
    // ByteTrack's second association needs the weak detections: with --track the default
    // score filter drops to the tracker's low threshold, an explicit --confidence wins.
    EXPECT(parse({"--engine", "e", "--track"}).confidence == kTrackLowThresh, "--track default confidence");
    EXPECT(parse({"--engine", "e", "--render", "o.mp4"}).confidence == kTrackLowThresh, "--render implies it");
    const Args explicit_conf = parse({"--engine", "e", "--track", "--confidence", "0.3"});
    EXPECT(explicit_conf.confidence == 0.3f && explicit_conf.confidence_set, "explicit --confidence kept");
    EXPECT(parse({"--engine", "e"}).confidence == 0.5f, "no tracking: detection default unchanged");
}

void test_render_implies_track_and_needs_opencv() {
    const Args a = parse({"--engine", "e", "--render", "out.mp4"});
    EXPECT(a.track && a.render == "out.mp4", "--render turns tracking on");
    EXPECT_THROWS(parse({"--engine", "e", "--render", "out.mp4", "--decoder", "nvdec"}),
                  "--render needs --decoder opencv");
}

void test_help_stops_parsing_without_validation() {
    const Args a = parse({"--help"});  // no --engine, still fine
    EXPECT(a.help, "help flag");
    EXPECT(parse({"-h", "--bogus"}).help, "-h wins over what follows");
}

void test_validation() {
    EXPECT_THROWS(parse({}), "--engine is required");
    EXPECT_THROWS(parse({"--engine", "e", "--decoder", "gstreamer"}), "--decoder must be opencv or nvdec");
    EXPECT_THROWS(parse({"--engine", "e", "--streams", "0"}), "--streams must be >= 1");
    EXPECT_THROWS(parse({"--engine", "e", "--frames", "0"}), "--frames must be >= 1");
    EXPECT_THROWS(parse({"--engine", "e", "--warmup", "-1"}), "--warmup must be >= 0");
    EXPECT_THROWS(parse({"--engine", "e", "--wat"}), "unknown argument: --wat");
    EXPECT_THROWS(parse({"--engine"}), "missing value for --engine");
}

void test_default_report_path() {
    Args a = parse({"--engine", "e"});
    EXPECT(default_report_path(a, "S") == "benchmarks/results/cpp_tensorrt_opencv_S.json", "single stream: %s",
           default_report_path(a, "S").c_str());
    a = parse({"--engine", "e", "--decoder", "nvdec", "--streams", "12", "--label", "h264"});
    EXPECT(default_report_path(a, "S") == "benchmarks/results/cpp_tensorrt_nvdec_s12_h264_S.json", "12 streams: %s",
           default_report_path(a, "S").c_str());
    a = parse({"--engine", "e", "--decoder", "nvdec", "--streams", "8", "--batched"});
    EXPECT(default_report_path(a, "S") == "benchmarks/results/cpp_tensorrt_nvdec_b8_S.json", "batched: %s",
           default_report_path(a, "S").c_str());
    a = parse({"--engine", "e", "--out", "custom.json", "--streams", "3"});
    EXPECT(default_report_path(a, "S") == "custom.json", "--out wins");
}

}  // namespace

int main() {
    test_defaults();
    test_every_flag();
    test_track_lowers_default_confidence();
    test_render_implies_track_and_needs_opencv();
    test_help_stops_parsing_without_validation();
    test_validation();
    test_default_report_path();
    return test_result("cli");
}
