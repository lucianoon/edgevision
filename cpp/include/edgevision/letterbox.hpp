#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace edgevision {

// Geometry of the letterbox, mirrors python/edgevision/preprocess.py::LetterboxInfo.
struct LetterboxInfo {
    float scale;
    int pad_x;
    int pad_y;
    int source_width;
    int source_height;
    int size;  // output side (square)
};

// Computes the geometry the same way as the Python letterbox (round(), pad centered).
LetterboxInfo compute_letterbox(int source_width, int source_height, int size);

// GPU letterbox: BGR uint8 HWC (device) -> RGB float32 planar NCHW in [0,1] (device),
// bilinear resize, pad value 114. `dst` must hold 3*size*size floats.
// Launch is asynchronous on `stream`.
void letterbox_cuda(const std::uint8_t* src_bgr, int src_pitch_bytes, const LetterboxInfo& info, float* dst,
                    cudaStream_t stream);

// CPU reference of the exact same arithmetic (for tests; slow).
void letterbox_reference(const std::uint8_t* src_bgr, int src_pitch_bytes, const LetterboxInfo& info, float* dst);

struct GpuFrame;

// Same output, but straight from an NV12 frame in device memory (NVDEC output):
// bilinear in Y and in the half-resolution UV plane, BT.601 limited-range to RGB.
void letterbox_nv12_cuda(const GpuFrame& frame, const LetterboxInfo& info, float* dst, cudaStream_t stream);

// CPU reference of the NV12 path (host pointers).
void letterbox_nv12_reference(const GpuFrame& frame_on_host, const LetterboxInfo& info, float* dst);

}  // namespace edgevision
