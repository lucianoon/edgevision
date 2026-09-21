#include "edgevision/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace edgevision {

// ------------------------------------------------------------------ Kalman filter

namespace {
constexpr double kStdWeightPosition = 1.0 / 20.0;
constexpr double kStdWeightVelocity = 1.0 / 160.0;

inline double& at(KalmanBox::Mat8& m, int r, int c) { return m[r * 8 + c]; }
inline double at(const KalmanBox::Mat8& m, int r, int c) { return m[r * 8 + c]; }

// Inverse of a symmetric positive-definite 4x4 by Gauss-Jordan (projected covariance).
std::array<double, 16> inverse4(std::array<double, 16> a) {
    std::array<double, 16> inv{};
    for (int i = 0; i < 4; ++i) inv[i * 4 + i] = 1.0;
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        for (int r = col + 1; r < 4; ++r)
            if (std::fabs(a[r * 4 + col]) > std::fabs(a[pivot * 4 + col])) pivot = r;
        if (std::fabs(a[pivot * 4 + col]) < 1e-12) throw std::runtime_error("singular covariance");
        if (pivot != col)
            for (int c = 0; c < 4; ++c) {
                std::swap(a[col * 4 + c], a[pivot * 4 + c]);
                std::swap(inv[col * 4 + c], inv[pivot * 4 + c]);
            }
        const double d = a[col * 4 + col];
        for (int c = 0; c < 4; ++c) {
            a[col * 4 + c] /= d;
            inv[col * 4 + c] /= d;
        }
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            const double f = a[r * 4 + col];
            if (f == 0.0) continue;
            for (int c = 0; c < 4; ++c) {
                a[r * 4 + c] -= f * a[col * 4 + c];
                inv[r * 4 + c] -= f * inv[col * 4 + c];
            }
        }
    }
    return inv;
}
}  // namespace

void KalmanBox::initiate(const std::array<double, 4>& m) {
    mean_ = {m[0], m[1], m[2], m[3], 0, 0, 0, 0};
    const double h = m[3];
    const std::array<double, 8> std = {2 * kStdWeightPosition * h, 2 * kStdWeightPosition * h, 1e-2,
                                       2 * kStdWeightPosition * h, 10 * kStdWeightVelocity * h,
                                       10 * kStdWeightVelocity * h, 1e-5, 10 * kStdWeightVelocity * h};
    cov_.fill(0.0);
    for (int i = 0; i < 8; ++i) at(cov_, i, i) = std[i] * std[i];
}

void KalmanBox::predict() {
    const double h = mean_[3];
    const std::array<double, 8> std = {kStdWeightPosition * h, kStdWeightPosition * h, 1e-2, kStdWeightPosition * h,
                                       kStdWeightVelocity * h, kStdWeightVelocity * h, 1e-5, kStdWeightVelocity * h};
    // x' = F x with F = [I dt*I; 0 I], dt = 1
    for (int i = 0; i < 4; ++i) mean_[i] += mean_[i + 4];
    // P' = F P F^T + Q
    Mat8 fp{};
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) at(fp, r, c) = at(cov_, r, c) + (r < 4 ? at(cov_, r + 4, c) : 0.0);
    Mat8 next{};
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) at(next, r, c) = at(fp, r, c) + (c < 4 ? at(fp, r, c + 4) : 0.0);
    for (int i = 0; i < 8; ++i) at(next, i, i) += std[i] * std[i];
    cov_ = next;
}

void KalmanBox::update(const std::array<double, 4>& z) {
    const double h = mean_[3];
    const std::array<double, 4> std = {kStdWeightPosition * h, kStdWeightPosition * h, 1e-1, kStdWeightPosition * h};
    // S = H P H^T + R  (H selects the first 4 states)
    std::array<double, 16> s{};
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) s[r * 4 + c] = at(cov_, r, c);
    for (int i = 0; i < 4; ++i) s[i * 4 + i] += std[i] * std[i];
    const auto s_inv = inverse4(s);
    // K = P H^T S^-1  (8x4)
    std::array<double, 32> k{};
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c) {
            double acc = 0.0;
            for (int j = 0; j < 4; ++j) acc += at(cov_, r, j) * s_inv[j * 4 + c];
            k[r * 4 + c] = acc;
        }
    // x += K (z - H x)
    std::array<double, 4> innov{};
    for (int i = 0; i < 4; ++i) innov[i] = z[i] - mean_[i];
    for (int r = 0; r < 8; ++r) {
        double acc = 0.0;
        for (int j = 0; j < 4; ++j) acc += k[r * 4 + j] * innov[j];
        mean_[r] += acc;
    }
    // P -= K S K^T
    Mat8 ks{};
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 4; ++c) {
            double acc = 0.0;
            for (int j = 0; j < 4; ++j) acc += k[r * 4 + j] * s[j * 4 + c];
            ks[r * 8 + c] = acc;  // reuse Mat8 storage as 8x4 (row stride 8)
        }
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) {
            double acc = 0.0;
            for (int j = 0; j < 4; ++j) acc += ks[r * 8 + j] * k[c * 4 + j];
            at(cov_, r, c) -= acc;
        }
}

std::array<float, 4> KalmanBox::tlbr() const {
    const double cx = mean_[0], cy = mean_[1], a = mean_[2], h = mean_[3];
    const double w = a * h;
    return {static_cast<float>(cx - w / 2), static_cast<float>(cy - h / 2), static_cast<float>(cx + w / 2),
            static_cast<float>(cy + h / 2)};
}

// ------------------------------------------------------------------ Hungarian

std::vector<int> hungarian(const std::vector<std::vector<double>>& cost) {
    const int rows = static_cast<int>(cost.size());
    if (rows == 0) return {};
    const int cols = static_cast<int>(cost[0].size());
    if (cols == 0) return std::vector<int>(rows, -1);
    if (rows > cols) {  // solve the transpose
        std::vector<std::vector<double>> t(cols, std::vector<double>(rows));
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) t[c][r] = cost[r][c];
        const auto col_to_row = hungarian(t);
        std::vector<int> out(rows, -1);
        for (int c = 0; c < cols; ++c)
            if (col_to_row[c] >= 0) out[col_to_row[c]] = c;
        return out;
    }
    // Kuhn-Munkres with potentials, rows <= cols. 1-based internally.
    const double INF = std::numeric_limits<double>::infinity();
    const int n = rows, m = cols;
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(m + 1, INF);
        std::vector<char> used(m + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            int j1 = 0;
            double delta = INF;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> assignment(n, -1);
    for (int j = 1; j <= m; ++j)
        if (p[j]) assignment[p[j] - 1] = j - 1;
    return assignment;
}

// ------------------------------------------------------------------ ByteTracker

namespace {
double iou(const std::array<float, 4>& a, const Detection& d) {
    const double ix1 = std::max<double>(a[0], d.x1), iy1 = std::max<double>(a[1], d.y1);
    const double ix2 = std::min<double>(a[2], d.x2), iy2 = std::min<double>(a[3], d.y2);
    const double inter = std::max(0.0, ix2 - ix1) * std::max(0.0, iy2 - iy1);
    const double area_a = std::max(0.0, double(a[2] - a[0])) * std::max(0.0, double(a[3] - a[1]));
    const double area_d = std::max(0.0, double(d.x2 - d.x1)) * std::max(0.0, double(d.y2 - d.y1));
    const double uni = area_a + area_d - inter;
    return uni > 0 ? inter / uni : 0.0;
}
}  // namespace

ByteTracker::ByteTracker(TrackerParams params) : params_(params) {}

std::array<double, 4> ByteTracker::to_xyah(const Detection& d) {
    const double w = std::max(1e-3, double(d.x2 - d.x1)), h = std::max(1e-3, double(d.y2 - d.y1));
    return {d.x1 + w / 2, d.y1 + h / 2, w / h, h};
}

void ByteTracker::associate(const std::vector<int>& track_idx, const std::vector<int>& det_idx,
                            const std::vector<Detection>& dets, float thresh,
                            std::vector<std::pair<int, int>>& matches, std::vector<int>& unmatched_tracks,
                            std::vector<int>& unmatched_dets) const {
    matches.clear();
    unmatched_tracks.clear();
    unmatched_dets.clear();
    if (track_idx.empty() || det_idx.empty()) {
        unmatched_tracks = track_idx;
        unmatched_dets = det_idx;
        return;
    }
    std::vector<std::vector<double>> cost(track_idx.size(), std::vector<double>(det_idx.size(), 1.0));
    for (size_t r = 0; r < track_idx.size(); ++r) {
        const STrack& t = tracks_[track_idx[r]];
        const auto box = t.box();
        for (size_t c = 0; c < det_idx.size(); ++c) {
            const Detection& d = dets[det_idx[c]];
            if (params_.class_aware && t.class_id != d.class_id) continue;  // cost 1.0: never matched
            cost[r][c] = 1.0 - iou(box, d);
        }
    }
    const auto assignment = hungarian(cost);
    std::vector<char> det_used(det_idx.size(), 0);
    for (size_t r = 0; r < track_idx.size(); ++r) {
        const int c = assignment[r];
        if (c >= 0 && cost[r][c] < thresh) {
            matches.emplace_back(track_idx[r], det_idx[c]);
            det_used[c] = 1;
        } else {
            unmatched_tracks.push_back(track_idx[r]);
        }
    }
    for (size_t c = 0; c < det_idx.size(); ++c)
        if (!det_used[c]) unmatched_dets.push_back(det_idx[c]);
}

void ByteTracker::refresh(STrack& t, const Detection& d, bool reactivate) {
    t.kf.update(to_xyah(d));
    t.score = d.confidence;
    t.class_id = d.class_id;
    t.class_name = d.class_name;
    t.last_frame = frame_id_;
    t.state = State::Tracked;
    t.activated = true;
    t.hits += 1;
    t.tracklet_len = reactivate ? 0 : t.tracklet_len + 1;
}

void ByteTracker::start_track(const Detection& d) {
    STrack t;
    t.kf.initiate(to_xyah(d));
    t.id = next_id_++;
    t.class_id = d.class_id;
    t.class_name = d.class_name;
    t.score = d.confidence;
    t.start_frame = t.last_frame = frame_id_;
    t.state = State::Tracked;
    t.activated = frame_id_ == 1;  // like ByteTrack: first frame's tracks are trusted immediately
    t.hits = 1;
    tracks_.push_back(std::move(t));
}

std::vector<Track> ByteTracker::update(const std::vector<Detection>& dets) {
    ++frame_id_;

    std::vector<int> high, low;
    for (int i = 0; i < static_cast<int>(dets.size()); ++i) {
        if (dets[i].confidence >= params_.track_thresh) high.push_back(i);
        else if (dets[i].confidence >= params_.low_thresh) low.push_back(i);
    }

    std::vector<int> unconfirmed, pool;  // pool = activated tracked + lost
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) {
        STrack& t = tracks_[i];
        if (t.state == State::Removed) continue;
        if (t.state == State::Tracked && !t.activated) unconfirmed.push_back(i);
        else pool.push_back(i);
    }
    for (int i : pool) tracks_[i].kf.predict();
    for (int i : unconfirmed) tracks_[i].kf.predict();

    std::vector<std::pair<int, int>> matches;
    std::vector<int> u_track, u_det;

    // 1. high-score detections vs tracked + lost
    associate(pool, high, dets, params_.match_thresh, matches, u_track, u_det);
    for (const auto& [ti, di] : matches) refresh(tracks_[ti], dets[di], tracks_[ti].state == State::Lost);

    // 2. low-score detections vs tracks still unmatched and currently tracked (not lost)
    std::vector<int> remaining_tracked;
    for (int ti : u_track)
        if (tracks_[ti].state == State::Tracked) remaining_tracked.push_back(ti);
    std::vector<std::pair<int, int>> matches2;
    std::vector<int> u_track2, u_low;
    associate(remaining_tracked, low, dets, params_.low_match_thresh, matches2, u_track2, u_low);
    for (const auto& [ti, di] : matches2) refresh(tracks_[ti], dets[di], false);
    for (int ti : u_track2) tracks_[ti].state = State::Lost;

    // 3. unconfirmed (born last frame) vs remaining high detections
    std::vector<std::pair<int, int>> matches3;
    std::vector<int> u_unconf, u_det3;
    associate(unconfirmed, u_det, dets, params_.unconfirmed_match_thresh, matches3, u_unconf, u_det3);
    for (const auto& [ti, di] : matches3) refresh(tracks_[ti], dets[di], false);
    for (int ti : u_unconf) tracks_[ti].state = State::Removed;

    // 4. new tracks
    for (int di : u_det3)
        if (dets[di].confidence >= params_.track_thresh + 0.1f) start_track(dets[di]);

    // 5. expire lost tracks
    for (STrack& t : tracks_)
        if (t.state == State::Lost && frame_id_ - t.last_frame > params_.track_buffer) t.state = State::Removed;

    std::vector<Track> out;
    for (const STrack& t : tracks_) {
        if (t.state != State::Tracked || !t.activated) continue;
        const auto b = t.box();
        out.push_back(Track{t.id, b[0], b[1], b[2], b[3], t.score, t.class_id, t.class_name, frame_id_ - t.start_frame + 1,
                            t.hits, t.last_frame == frame_id_});
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [](const STrack& t) { return t.state == State::Removed; }),
                  tracks_.end());
    return out;
}

}  // namespace edgevision
