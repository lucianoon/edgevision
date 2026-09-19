#include "edgevision/names.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace edgevision {

std::string sidecar_names_path(const std::string& engine_path) {
    const auto dot = engine_path.rfind('.');
    const auto slash = engine_path.rfind('/');
    const bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
    return (has_ext ? engine_path.substr(0, dot) : engine_path) + ".names.json";
}

// Extracts consecutive quoted strings as key/value pairs: {"0": "person", "1": "bicycle"}.
std::map<int, std::string> load_class_names(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open class names file: " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();

    std::map<int, std::string> names;
    std::string key;
    bool have_key = false;
    size_t pos = 0;
    while ((pos = text.find('"', pos)) != std::string::npos) {
        const size_t end = text.find('"', pos + 1);
        if (end == std::string::npos) break;
        const std::string token = text.substr(pos + 1, end - pos - 1);
        pos = end + 1;
        if (!have_key) {
            key = token;
            have_key = true;
        } else {
            names[std::stoi(key)] = token;
            have_key = false;
        }
    }
    if (names.empty()) throw std::runtime_error("no class names parsed from " + path);
    return names;
}

}  // namespace edgevision
