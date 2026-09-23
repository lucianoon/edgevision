#pragma once

#include <array>
#include <string>
#include <vector>

#include "edgevision/detection.hpp"

namespace edgevision {

// ByteTrack (Zhang et al., 2022) with a constant-velocity Kalman filter, IoU association
// solved by the Hungarian algorithm, a second pass on low-score detections and a
// lost-track buffer. Dependency-free; per-frame cost is a few tens of microseconds.

// Lowest detection score ByteTrack still uses (second association). The detector must not
// filter below it, or the second association never sees a detection.
inline constexpr float kTrackLowThresh = 0.1f;

struct TrackerParams {
    float track_thresh = 0.5f;           // detections >= this are "high"; new tracks need >= track_thresh + 0.1
    float low_thresh = kTrackLowThresh;  // low_thresh <= score < track_thresh are "low" (second association)
    float match_thresh = 0.8f;           // first association accepts cost (1 - IoU) below this
    float low_match_thresh = 0.5f;
    float unconfirmed_match_thresh = 0.7f;
    int track_buffer = 30;    // frames a lost track is kept before removal
    bool class_aware = true;  // never associate detections of different classes
};

struct Track {
    int id;
    float x1, y1, x2, y2;
    float score;
    int class_id;
    std::string class_name;
    int age;     // frames since the track was created
    int hits;    // frames with a matched detection
    bool fresh;  // matched in this frame (false: predicted only; not emitted by default)
};

// Kalman filter on (cx, cy, aspect, h) + velocities, exactly the ByteTrack/SORT variant.
class KalmanBox {
public:
    using Vec8 = std::array<double, 8>;
    using Mat8 = std::array<double, 64>;
    void initiate(const std::array<double, 4>& xyah);
    void predict();
    void update(const std::array<double, 4>& xyah);
    const Vec8& mean() const { return mean_; }
    std::array<float, 4> tlbr() const;  // x1 y1 x2 y2

private:
    Vec8 mean_{};
    Mat8 cov_{};
};

// Minimum-cost assignment; returns row -> column (-1 when unassigned). Rectangular ok.
std::vector<int> hungarian(const std::vector<std::vector<double>>& cost);

class ByteTracker {
public:
    explicit ByteTracker(TrackerParams params = {});

    // Detections of one frame in -> confirmed tracks (matched this frame) out.
    std::vector<Track> update(const std::vector<Detection>& detections);

    int frame_id() const { return frame_id_; }
    int next_id() const { return next_id_; }  // ids issued so far + 1

private:
    enum class State { New, Tracked, Lost, Removed };
    struct STrack {
        KalmanBox kf;
        State state = State::New;
        bool activated = false;
        int id = 0;
        int class_id = -1;
        std::string class_name;
        float score = 0.f;
        int start_frame = 0;
        int last_frame = 0;
        int tracklet_len = 0;
        int hits = 0;
        std::array<float, 4> box() const { return kf.tlbr(); }
    };

    static std::array<double, 4> to_xyah(const Detection& d);
    void associate(const std::vector<int>& track_idx, const std::vector<int>& det_idx,
                   const std::vector<Detection>& dets, float thresh, std::vector<std::pair<int, int>>& matches,
                   std::vector<int>& unmatched_tracks, std::vector<int>& unmatched_dets) const;
    void refresh(STrack& t, const Detection& d, bool reactivate);
    void start_track(const Detection& d);

    TrackerParams params_;
    std::vector<STrack> tracks_;
    int frame_id_ = 0;
    int next_id_ = 1;
};

}  // namespace edgevision
