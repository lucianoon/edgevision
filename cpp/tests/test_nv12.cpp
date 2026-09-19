// CTest: the NV12 letterbox kernel (NVDEC path) must match (a) its CPU reference and
// (b) the BGR path fed with OpenCV's NV12->BGR conversion of the same frame, within
// chroma-subsampling tolerance.
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "edgevision/cuda_check.hpp"
#include "edgevision/gpu_frame.hpp"
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
            p[0] = static_cast<uchar>((x * 255) / cols);
            p[1] = static_cast<uchar>((y * 255) / rows);
            p[2] = static_cast<uchar>(128 + 100 * std::sin(x * 0.05) * std::cos(y * 0.03));
        }
    return img;
}

// BGR -> NV12 (Y plane + interleaved UV), BT.601 limited range like a decoder output.
void to_nv12(const cv::Mat& bgr, cv::Mat& y, cv::Mat& uv) {
    cv::Mat i420;
    cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);  // (h*3/2) x w, planar Y, U, V
    const int h = bgr.rows, w = bgr.cols;
    y = i420.rowRange(0, h).clone();
    cv::Mat u = i420.rowRange(h, h + h / 4).reshape(1, h / 2);          // (h/2) x (w/2)
    cv::Mat v = i420.rowRange(h + h / 4, h + h / 2).reshape(1, h / 2);  // (h/2) x (w/2)
    uv.create(h / 2, w, CV_8UC1);
    for (int r = 0; r < h / 2; ++r)
        for (int c = 0; c < w / 2; ++c) {
            uv.at<uchar>(r, 2 * c) = u.at<uchar>(r, c);
            uv.at<uchar>(r, 2 * c + 1) = v.at<uchar>(r, c);
        }
}

void run_case(int rows, int cols, int size) {
    std::printf("case %dx%d -> %d\n", cols, rows, size);
    const cv::Mat bgr = synthetic_frame(rows, cols);
    cv::Mat y, uv;
    to_nv12(bgr, y, uv);
    const LetterboxInfo info = compute_letterbox(cols, rows, size);
    const size_t n = 3ull * size * size;

    // GPU
    std::uint8_t *d_y = nullptr, *d_uv = nullptr;
    float* d_dst = nullptr;
    EV_CUDA_CHECK(cudaMalloc(&d_y, y.total()));
    EV_CUDA_CHECK(cudaMalloc(&d_uv, uv.total()));
    EV_CUDA_CHECK(cudaMalloc(&d_dst, n * sizeof(float)));
    EV_CUDA_CHECK(cudaMemcpy(d_y, y.data, y.total(), cudaMemcpyHostToDevice));
    EV_CUDA_CHECK(cudaMemcpy(d_uv, uv.data, uv.total(), cudaMemcpyHostToDevice));
    GpuFrame gf{d_y, d_uv, cols, cols, cols, rows};
    letterbox_nv12_cuda(gf, info, d_dst, nullptr);
    EV_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> gpu(n);
    EV_CUDA_CHECK(cudaMemcpy(gpu.data(), d_dst, n * sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_y);
    cudaFree(d_uv);
    cudaFree(d_dst);

    // CPU reference of the same arithmetic
    GpuFrame hf{y.data, uv.data, cols, cols, cols, rows};
    std::vector<float> ref(n);
    letterbox_nv12_reference(hf, info, ref.data());
    float max_ref = 0.f;
    for (size_t i = 0; i < n; ++i) max_ref = std::max(max_ref, std::fabs(gpu[i] - ref[i]));
    EXPECT(max_ref < 1e-4f, "gpu vs reference max diff %g", max_ref);

    // OpenCV NV12->BGR then the BGR letterbox reference
    cv::Mat nv12(rows * 3 / 2, cols, CV_8UC1);
    y.copyTo(nv12.rowRange(0, rows));
    uv.copyTo(nv12.rowRange(rows, rows * 3 / 2));
    cv::Mat bgr_from_nv12;
    cv::cvtColor(nv12, bgr_from_nv12, cv::COLOR_YUV2BGR_NV12);
    std::vector<float> via_bgr(n);
    letterbox_reference(bgr_from_nv12.data, static_cast<int>(bgr_from_nv12.step[0]), info, via_bgr.data());
    double sum = 0.0;
    float mx = 0.f;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(gpu[i] - via_bgr[i]);
        sum += d;
        mx = std::max(mx, d);
    }
    const double mean = sum / n;
    std::printf("  vs OpenCV NV12->BGR path: mean %.2f/255, max %.1f/255\n", mean * 255, mx * 255);
    EXPECT(mean < 2.0 / 255.0, "mean diff vs OpenCV path too high: %g", mean);
    EXPECT(mx < 40.0f / 255.0f, "max diff vs OpenCV path too high: %g (chroma edges)", mx);

    // And against the original BGR (round trip through 4:2:0 loses chroma detail)
    std::vector<float> orig(n);
    letterbox_reference(bgr.data, static_cast<int>(bgr.step[0]), info, orig.data());
    sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += std::fabs(gpu[i] - orig[i]);
    std::printf("  vs original BGR: mean %.2f/255\n", sum / n * 255);
    EXPECT(sum / n < 3.0 / 255.0, "mean diff vs original BGR too high: %g", sum / n);

    const float pad = 114.0f / 255.0f;
    if (info.pad_y > 0) EXPECT(std::fabs(gpu[size / 2] - pad) < 1e-6f, "top pad row not 114");
}

}  // namespace

int main() {
    run_case(1080, 1920, 640);
    run_case(1080, 810, 640);
    run_case(720, 1280, 640);
    if (failures == 0) std::printf("nv12 tests: OK\n");
    return failures == 0 ? 0 : 1;
}
