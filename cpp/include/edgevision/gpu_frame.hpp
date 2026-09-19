#pragma once

#include <cstdint>

namespace edgevision {

// A decoded NV12 frame that already lives in GPU memory (e.g. from NVDEC).
// Y plane: height rows of width bytes; UV plane: height/2 rows of width bytes (interleaved).
struct GpuFrame {
    const std::uint8_t* y = nullptr;
    const std::uint8_t* uv = nullptr;
    int pitch_y = 0;   // bytes per row
    int pitch_uv = 0;  // bytes per row
    int width = 0;
    int height = 0;
};

}  // namespace edgevision
