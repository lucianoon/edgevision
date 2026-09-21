#include "edgevision/letterbox.hpp"

#include <algorithm>
#include <cmath>

#include "edgevision/cuda_check.hpp"
#include "edgevision/gpu_frame.hpp"

namespace edgevision {

LetterboxInfo compute_letterbox(int source_width, int source_height, int size) {
    const float scale = std::min(size / static_cast<float>(source_height), size / static_cast<float>(source_width));
    const int new_w = static_cast<int>(std::lround(source_width * scale));
    const int new_h = static_cast<int>(std::lround(source_height * scale));
    const float pad_x = (size - new_w) / 2.0f;
    const float pad_y = (size - new_h) / 2.0f;
    // Python: left = round(pad_x - 0.1), top = round(pad_y - 0.1)
    return LetterboxInfo{scale,
                         static_cast<int>(std::lround(pad_x - 0.1f)),
                         static_cast<int>(std::lround(pad_y - 0.1f)),
                         source_width,
                         source_height,
                         size};
}

namespace {

constexpr float kPad = 114.0f;

struct Geometry {
    int src_w, src_h, src_pitch;
    int new_w, new_h;
    int pad_x, pad_y;
    int size;
};

__host__ __device__ inline Geometry make_geometry(const LetterboxInfo& info, int pitch) {
    Geometry g;
    g.src_w = info.source_width;
    g.src_h = info.source_height;
    g.src_pitch = pitch;
    g.new_w = static_cast<int>(lroundf(info.source_width * info.scale));
    g.new_h = static_cast<int>(lroundf(info.source_height * info.scale));
    g.pad_x = info.pad_x;
    g.pad_y = info.pad_y;
    g.size = info.size;
    return g;
}

// Bilinear sample of channel `c` at continuous coords (sx, sy), OpenCV-style half-pixel
// centres, edge clamped.
__host__ __device__ inline float sample_bilinear(const unsigned char* src, const Geometry& g, float sx, float sy,
                                                 int c) {
    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    const float fx = sx - x0;
    const float fy = sy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = x0 < 0 ? 0 : (x0 > g.src_w - 1 ? g.src_w - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 > g.src_w - 1 ? g.src_w - 1 : x1);
    y0 = y0 < 0 ? 0 : (y0 > g.src_h - 1 ? g.src_h - 1 : y0);
    y1 = y1 < 0 ? 0 : (y1 > g.src_h - 1 ? g.src_h - 1 : y1);
    const unsigned char* r0 = src + static_cast<size_t>(y0) * g.src_pitch;
    const unsigned char* r1 = src + static_cast<size_t>(y1) * g.src_pitch;
    const float v00 = r0[x0 * 3 + c], v01 = r0[x1 * 3 + c];
    const float v10 = r1[x0 * 3 + c], v11 = r1[x1 * 3 + c];
    return (v00 * (1.0f - fx) + v01 * fx) * (1.0f - fy) + (v10 * (1.0f - fx) + v11 * fx) * fy;
}

// One output pixel: writes R, G, B planes (NCHW) normalised to [0, 1].
__host__ __device__ inline void letterbox_pixel(const unsigned char* src, const Geometry& g, int x, int y, float* dst) {
    float b, gch, r;
    const int rx = x - g.pad_x, ry = y - g.pad_y;
    if (rx < 0 || ry < 0 || rx >= g.new_w || ry >= g.new_h) {
        b = gch = r = kPad;
    } else {
        const float sx = (rx + 0.5f) * (g.src_w / static_cast<float>(g.new_w)) - 0.5f;
        const float sy = (ry + 0.5f) * (g.src_h / static_cast<float>(g.new_h)) - 0.5f;
        b = sample_bilinear(src, g, sx, sy, 0);
        gch = sample_bilinear(src, g, sx, sy, 1);
        r = sample_bilinear(src, g, sx, sy, 2);
    }
    const size_t plane = static_cast<size_t>(g.size) * g.size;
    const size_t idx = static_cast<size_t>(y) * g.size + x;
    dst[idx] = r * (1.0f / 255.0f);
    dst[plane + idx] = gch * (1.0f / 255.0f);
    dst[2 * plane + idx] = b * (1.0f / 255.0f);
}

__global__ void letterbox_kernel(const unsigned char* __restrict__ src, Geometry g, float* __restrict__ dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g.size || y >= g.size) return;
    letterbox_pixel(src, g, x, y, dst);
}

}  // namespace

void letterbox_cuda(const std::uint8_t* src_bgr, int src_pitch_bytes, const LetterboxInfo& info, float* dst,
                    cudaStream_t stream) {
    const Geometry g = make_geometry(info, src_pitch_bytes);
    const dim3 block(32, 8);
    const dim3 grid((g.size + block.x - 1) / block.x, (g.size + block.y - 1) / block.y);
    letterbox_kernel<<<grid, block, 0, stream>>>(src_bgr, g, dst);
    EV_CUDA_CHECK(cudaGetLastError());
}

void letterbox_reference(const std::uint8_t* src_bgr, int src_pitch_bytes, const LetterboxInfo& info, float* dst) {
    const Geometry g = make_geometry(info, src_pitch_bytes);
    for (int y = 0; y < g.size; ++y)
        for (int x = 0; x < g.size; ++x) letterbox_pixel(src_bgr, g, x, y, dst);
}

// ---------------------------------------------------------------------------------
// NV12 path (NVDEC output)

namespace {

struct Nv12Geometry {
    const unsigned char* y;
    const unsigned char* uv;
    int pitch_y, pitch_uv;
    int src_w, src_h;
    int new_w, new_h;
    int pad_x, pad_y;
    int size;
};

__host__ __device__ inline float sample_plane(const unsigned char* plane, int pitch, int w, int h, int stride,
                                              int channel, float sx, float sy) {
    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    const float fx = sx - x0, fy = sy - y0;
    int x1 = x0 + 1, y1 = y0 + 1;
    x0 = x0 < 0 ? 0 : (x0 > w - 1 ? w - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 > w - 1 ? w - 1 : x1);
    y0 = y0 < 0 ? 0 : (y0 > h - 1 ? h - 1 : y0);
    y1 = y1 < 0 ? 0 : (y1 > h - 1 ? h - 1 : y1);
    const unsigned char* r0 = plane + static_cast<size_t>(y0) * pitch;
    const unsigned char* r1 = plane + static_cast<size_t>(y1) * pitch;
    const float v00 = r0[x0 * stride + channel], v01 = r0[x1 * stride + channel];
    const float v10 = r1[x0 * stride + channel], v11 = r1[x1 * stride + channel];
    return (v00 * (1.0f - fx) + v01 * fx) * (1.0f - fy) + (v10 * (1.0f - fx) + v11 * fx) * fy;
}

__host__ __device__ inline float clamp255(float v) {
    return v < 0.f ? 0.f : (v > 255.f ? 255.f : v);
}

__host__ __device__ inline void nv12_pixel(const Nv12Geometry& g, int x, int y, float* dst) {
    float r, gch, b;
    const int rx = x - g.pad_x, ry = y - g.pad_y;
    if (rx < 0 || ry < 0 || rx >= g.new_w || ry >= g.new_h) {
        r = gch = b = kPad;
    } else {
        const float sx = (rx + 0.5f) * (g.src_w / static_cast<float>(g.new_w)) - 0.5f;
        const float sy = (ry + 0.5f) * (g.src_h / static_cast<float>(g.new_h)) - 0.5f;
        const float Y = sample_plane(g.y, g.pitch_y, g.src_w, g.src_h, 1, 0, sx, sy);
        // chroma is half resolution, sited at the centre of each 2x2 luma block
        const float cx = (sx + 0.5f) * 0.5f - 0.5f, cy = (sy + 0.5f) * 0.5f - 0.5f;
        const float U = sample_plane(g.uv, g.pitch_uv, g.src_w / 2, g.src_h / 2, 2, 0, cx, cy);
        const float V = sample_plane(g.uv, g.pitch_uv, g.src_w / 2, g.src_h / 2, 2, 1, cx, cy);
        // BT.601 limited range (what FFmpeg/OpenCV use for untagged 8-bit video)
        const float c = 1.164383f * (Y - 16.0f), d = U - 128.0f, e = V - 128.0f;
        r = clamp255(c + 1.596027f * e);
        gch = clamp255(c - 0.391762f * d - 0.812968f * e);
        b = clamp255(c + 2.017232f * d);
    }
    const size_t plane = static_cast<size_t>(g.size) * g.size;
    const size_t idx = static_cast<size_t>(y) * g.size + x;
    dst[idx] = r * (1.0f / 255.0f);
    dst[plane + idx] = gch * (1.0f / 255.0f);
    dst[2 * plane + idx] = b * (1.0f / 255.0f);
}

__global__ void nv12_kernel(Nv12Geometry g, float* __restrict__ dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g.size || y >= g.size) return;
    nv12_pixel(g, x, y, dst);
}

Nv12Geometry make_nv12_geometry(const GpuFrame& f, const LetterboxInfo& info) {
    Nv12Geometry g;
    g.y = f.y;
    g.uv = f.uv;
    g.pitch_y = f.pitch_y;
    g.pitch_uv = f.pitch_uv;
    g.src_w = f.width;
    g.src_h = f.height;
    g.new_w = static_cast<int>(std::lround(f.width * info.scale));
    g.new_h = static_cast<int>(std::lround(f.height * info.scale));
    g.pad_x = info.pad_x;
    g.pad_y = info.pad_y;
    g.size = info.size;
    return g;
}

}  // namespace

void letterbox_nv12_cuda(const GpuFrame& frame, const LetterboxInfo& info, float* dst, cudaStream_t stream) {
    const Nv12Geometry g = make_nv12_geometry(frame, info);
    const dim3 block(32, 8);
    const dim3 grid((g.size + block.x - 1) / block.x, (g.size + block.y - 1) / block.y);
    nv12_kernel<<<grid, block, 0, stream>>>(g, dst);
    EV_CUDA_CHECK(cudaGetLastError());
}

void letterbox_nv12_reference(const GpuFrame& frame_on_host, const LetterboxInfo& info, float* dst) {
    const Nv12Geometry g = make_nv12_geometry(frame_on_host, info);
    for (int y = 0; y < g.size; ++y)
        for (int x = 0; x < g.size; ++x) nv12_pixel(g, x, y, dst);
}

}  // namespace edgevision
