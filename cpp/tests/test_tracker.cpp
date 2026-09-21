// CTest for the ByteTracker: Hungarian correctness, Kalman sanity and identity
// stability on synthetic scenes (motion, crossing, occlusion, entries/exits).
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

#include "edgevision/tracker.hpp"

using namespace edgevision;

namespace {
int failures = 0;
#define EXPECT(cond, ...)                                    \
    do {                                                     \
        if (!(cond)) {                                       \
            ++failures;                                      \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
        }                                                    \
    } while (0)

Detection det(float cx, float cy, float w, float h, float score = 0.9f, int cls = 0) {
    return Detection{cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, score, cls, "person"};
}

void test_hungarian() {
    // classic 3x3
    const auto a = hungarian({{4, 1, 3}, {2, 0, 5}, {3, 2, 2}});
    EXPECT(a[0] == 1 && a[1] == 0 && a[2] == 2, "3x3 assignment got %d %d %d", a[0], a[1], a[2]);
    // rectangular, more rows than columns: one row stays unassigned
    const auto b = hungarian({{1, 9}, {9, 1}, {5, 5}});
    EXPECT(b[0] == 0 && b[1] == 1 && b[2] == -1, "3x2 assignment got %d %d %d", b[0], b[1], b[2]);
    // more columns than rows
    const auto c = hungarian({{9, 9, 1}, {1, 9, 9}});
    EXPECT(c[0] == 2 && c[1] == 0, "2x3 assignment got %d %d", c[0], c[1]);
}

void test_kalman_follows_constant_velocity() {
    KalmanBox kf;
    kf.initiate({100, 100, 0.5, 80});
    for (int f = 1; f <= 20; ++f) {
        kf.predict();
        kf.update({100.0 + 5.0 * f, 100.0, 0.5, 80});
    }
    kf.predict();  // no measurement: prediction should extrapolate ~5 px/frame
    const auto b = kf.tlbr();
    const double cx = (b[0] + b[2]) / 2;
    EXPECT(std::fabs(cx - 205.0) < 2.0, "kalman extrapolated cx=%.1f, expected ~205", cx);
    EXPECT(std::fabs((b[3] - b[1]) - 80.0) < 1.0, "kalman height drifted to %.1f", b[3] - b[1]);
}

// Two objects moving across the frame for 60 frames: ids must never change or swap.
void test_stable_ids_two_movers() {
    ByteTracker tracker;
    std::map<int, int> id_by_object;  // object -> track id seen
    int switches = 0;
    for (int f = 0; f < 60; ++f) {
        std::vector<Detection> dets = {det(100 + 4 * f, 200, 40, 90), det(900 - 6 * f, 500, 50, 110)};
        const auto tracks = tracker.update(dets);
        if (f == 0) EXPECT(tracks.size() == 2, "frame 0 emitted %zu tracks", tracks.size());
        for (const auto& t : tracks) {
            const float cx = (t.x1 + t.x2) / 2;
            const int object = std::fabs(cx - (100 + 4 * f)) < std::fabs(cx - (900 - 6 * f)) ? 0 : 1;
            auto it = id_by_object.find(object);
            if (it == id_by_object.end()) id_by_object[object] = t.id;
            else if (it->second != t.id) ++switches;
        }
    }
    EXPECT(switches == 0, "%d id switches on two clean movers", switches);
    EXPECT(tracker.next_id() == 3, "expected exactly 2 ids issued, got %d", tracker.next_id() - 1);
}

// An object disappears for 10 frames (occlusion) and comes back near its predicted
// position: same id. A track lost longer than the buffer gets a new id.
void test_occlusion_and_buffer() {
    ByteTracker tracker;
    int id_before = -1, id_after = -1;
    for (int f = 0; f < 40; ++f) {
        std::vector<Detection> dets;
        if (f < 15 || f >= 25) dets.push_back(det(200 + 3 * f, 300, 40, 90));
        dets.push_back(det(1200, 300, 40, 90));  // static anchor object keeps the tracker busy
        const auto tracks = tracker.update(dets);
        for (const auto& t : tracks) {
            if (std::fabs((t.x1 + t.x2) / 2 - 1200) < 20) continue;
            if (f < 15) id_before = t.id;
            if (f >= 25) id_after = t.id;
        }
    }
    EXPECT(id_before > 0 && id_before == id_after, "occlusion 10 frames: id %d -> %d", id_before, id_after);

    ByteTracker tracker2(TrackerParams{});  // default buffer 30
    int id1 = -1, id2 = -1;
    for (int f = 0; f < 80; ++f) {
        std::vector<Detection> dets;
        if (f < 10 || f >= 50) dets.push_back(det(400, 400, 40, 90));  // gone for 40 > 30 frames
        dets.push_back(det(1200, 300, 40, 90));
        for (const auto& t : tracker2.update(dets)) {
            if (std::fabs((t.x1 + t.x2) / 2 - 1200) < 20) continue;
            if (f < 10) id1 = t.id;
            if (f >= 50) id2 = t.id;
        }
    }
    EXPECT(id1 > 0 && id2 > 0 && id1 != id2, "lost > buffer must get a new id: %d vs %d", id1, id2);
}

// Low-score detections keep a track alive (second association) but never create one.
void test_low_score_second_pass() {
    ByteTracker tracker;
    int emitted_low_only = 0;
    for (int f = 0; f < 30; ++f) {
        const float score = f < 5 ? 0.9f : 0.3f;  // becomes a weak detection after 5 frames
        const auto tracks = tracker.update({det(300 + 2 * f, 300, 40, 90, score)});
        if (f >= 5 && !tracks.empty()) ++emitted_low_only;
    }
    EXPECT(emitted_low_only >= 20, "track should survive on low-score detections (%d/25 frames)", emitted_low_only);

    ByteTracker tracker2;
    int created = 0;
    for (int f = 0; f < 10; ++f) created += tracker2.update({det(300, 300, 40, 90, 0.3f)}).size();
    EXPECT(created == 0 && tracker2.next_id() == 1, "low-score detections must not start tracks (%d)", created);
}

// Class-aware association: a person and a car at the same place keep different ids.
void test_class_aware() {
    ByteTracker tracker;
    std::set<int> ids;
    for (int f = 0; f < 10; ++f) {
        std::vector<Detection> dets = {det(500, 500, 60, 120, 0.9f, 0), det(500, 500, 60, 120, 0.9f, 2)};
        for (const auto& t : tracker.update(dets)) ids.insert(t.id);
    }
    EXPECT(ids.size() == 2 && tracker.next_id() == 3, "expected 2 stable ids for 2 classes, got %zu / %d issued",
           ids.size(), tracker.next_id() - 1);
}
}  // namespace

int main() {
    test_hungarian();
    test_kalman_follows_constant_velocity();
    test_stable_ids_two_movers();
    test_occlusion_and_buffer();
    test_low_score_second_pass();
    test_class_aware();
    if (failures == 0) std::printf("tracker tests: OK\n");
    return failures == 0 ? 0 : 1;
}
