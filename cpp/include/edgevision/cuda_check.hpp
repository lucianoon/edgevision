#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#define EV_CUDA_CHECK(expr)                                                                       \
    do {                                                                                          \
        cudaError_t ev_err__ = (expr);                                                            \
        if (ev_err__ != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(ev_err__) + \
                                     " at " __FILE__ ":" + std::to_string(__LINE__));             \
        }                                                                                         \
    } while (0)
