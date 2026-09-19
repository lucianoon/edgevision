#pragma once

#include <string>

namespace edgevision {

// Same fields as the Python dataclass; coordinates in source-frame pixels.
struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    std::string class_name;
};

}  // namespace edgevision
