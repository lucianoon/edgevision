// CTest: the CUDA letterbox must match (a) the shared CPU reference bit-for-bit within
// float noise and (b) OpenCV's resize + copyMakeBorder pipeline used by the Python
// preprocess within 8-bit rounding.
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "edgevision/cuda_check.hpp"
#include "edgevision/letterbox.hpp"

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

cv::Mat synthetic_frame(int rows, int cols) {
    cv::Mat img(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y)
        for (int x = 0; x < cols; ++x) {
            auto& p = img.at<cv::Vec3b>(y, x);
            p[0] = static_cast<uchar>((x * 255) / cols);               // B gradient
            p[1] = static_cast<uchar>((y * 255) / rows);               // G gradient
            p[2] = static_cast<uchar>(((x / 16 + y / 16) % 2) * 200);  // R checkerboard
        }
    return img;
}

// Python preprocess() equivalent with OpenCV: resize INTER_LINEAR, pad 114, BGR->RGB, /255, CHW.
std::vector<float> opencv_letterbox(const cv::Mat& frame, const LetterboxInfo& info) {
    const int new_w = static_cast<int>(std::lround(frame.cols * info.scale));
    const int new_h = static_cast<int>(std::lround(frame.rows * info.scale));
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, info.pad_y, info.size - new_h - info.pad_y, info.pad_x,
                       info.size - new_w - info.pad_x, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    std::vector<float> out(3ull * info.size * info.size);
    const size_t plane = static_cast<size_t>(info.size) * info.size;
    for (int y = 0; y < info.size; ++y)
        for (int x = 0; x < info.size; ++x) {
            const auto& p = padded.at<cv::Vec3b>(y, x);
            const size_t idx = static_cast<size_t>(y) * info.size + x;
            out[idx] = p[2] / 255.0f;
            out[plane + idx] = p[1] / 255.0f;
            out[2 * plane + idx] = p[0] / 255.0f;
        }
    return out;
}

void run_case(int rows, int cols, int size) {
    std::printf("case %dx%d -> %d\n", cols, rows, size);
    const cv::Mat frame = synthetic_frame(rows, cols);
    const LetterboxInfo info = compute_letterbox(cols, rows, size);
    const size_t n = 3ull * size * size;

    // GPU
    std::uint8_t* d_src = nullptr;
    float* d_dst = nullptr;
    const size_t src_bytes = frame.step[0] * frame.rows;
    EV_CUDA_CHECK(cudaMalloc(&d_src, src_bytes));
    EV_CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));
    EV_CUDA_CHECK(cudaMemcpy(d_src, frame.data, src_bytes, cudaMemcpyHostToDevice));
    letterbox_cuda(d_src, static_cast<int>(frame.step[0]), info, d_dst, nullptr);
    EV_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> gpu(n);
    EV_CUDA_CHECK(cudaMemcpy(gpu.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_src);
    cudaFree(d_dst);

    // CPU reference (same arithmetic)
    std::vector<float> ref(n);
    letterbox_reference(frame.data, static_cast<int>(frame.step[0]), info, ref.data());
    float max_ref_diff = 0.f;
    for (size_t i = 0; i < n; ++i) max_ref_diff = std::max(max_ref_diff, std::fabs(gpu[i] - ref[i]));
    EXPECT(max_ref_diff < 1e-4f, "gpu vs reference max diff %g", max_ref_diff);

    // OpenCV pipeline
    const std::vector<float> cv_out = opencv_letterbox(frame, info);
    float max_cv_diff = 0.f;
    double sum_cv_diff = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(gpu[i] - cv_out[i]);
        max_cv_diff = std::max(max_cv_diff, d);
        sum_cv_diff += d;
    }
    const double mean_cv_diff = sum_cv_diff / n;
    std::printf("  vs OpenCV: mean %.5f (%.2f/255), max %.5f (%.2f/255)\n", mean_cv_diff, mean_cv_diff * 255,
                max_cv_diff, max_cv_diff * 255);
    EXPECT(mean_cv_diff < 0.5 / 255.0, "mean diff vs OpenCV too high: %g", mean_cv_diff);
    EXPECT(max_cv_diff <= 3.0f / 255.0f, "max diff vs OpenCV too high: %g", max_cv_diff);

    // Padding value and geometry
    const float pad = 114.0f / 255.0f;
    if (info.pad_x > 0) EXPECT(std::fabs(gpu[size / 2 * size + 0] - pad) < 1e-6f, "left pad column not 114");
    if (info.pad_y > 0) EXPECT(std::fabs(gpu[0 * size + size / 2] - pad) < 1e-6f, "top pad row not 114");
}

}  // namespace

int main() {
    run_case(1080, 810, 640);   // portrait (bus.jpg): scale 0.5926, pad_x 80
    run_case(1080, 1920, 640);  // landscape 1080p: pad_y 140
    run_case(480, 640, 640);    // no resize, vertical pad 80
    run_case(1280, 1280, 640);  // square, resize only

    const LetterboxInfo info = compute_letterbox(810, 1080, 640);
    EXPECT(info.pad_x == 80 && info.pad_y == 0, "geometry for 810x1080: pad_x=%d pad_y=%d", info.pad_x, info.pad_y);
    EXPECT(std::fabs(info.scale - 640.0f / 1080.0f) < 1e-6f, "scale for 810x1080: %g", info.scale);

    if (failures == 0) std::printf("letterbox tests: OK\n");
    return failures == 0 ? 0 : 1;
}
