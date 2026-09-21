#include "edgevision/names.hpp"

#include <cstdint>
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

std::string read_ultralytics_metadata(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open engine: " + engine_path);
    unsigned char head[4];
    if (!file.read(reinterpret_cast<char*>(head), 4)) return "";
    const std::int32_t length = static_cast<std::int32_t>(head[0] | (head[1] << 8) | (head[2] << 16) | (head[3] << 24));
    if (length <= 0 || length > (1 << 20)) return "";  // implausible for metadata: plain engine
    std::string json(static_cast<size_t>(length), '\0');
    if (!file.read(json.data(), length)) return "";
    if (json.empty() || json[0] != '{') return "";
    return json;
}

// Extracts consecutive quoted strings as key/value pairs: {"0": "person", "1": "bicycle"}.
std::map<int, std::string> parse_names_object(const std::string& text) {
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
            try {
                names[std::stoi(key)] = token;
            } catch (const std::exception&) {  // not an "int": "name" pair
            }
            have_key = false;
        }
    }
    return names;
}

std::map<int, std::string> load_class_names_for_engine(const std::string& engine_path) {
    std::ifstream sidecar(sidecar_names_path(engine_path));
    if (sidecar) return load_class_names(sidecar_names_path(engine_path));
    const std::string meta = read_ultralytics_metadata(engine_path);
    const size_t at = meta.find("\"names\"");
    if (at == std::string::npos) throw std::runtime_error("no class names: neither " + sidecar_names_path(engine_path) +
                                                          " nor Ultralytics metadata in " + engine_path);
    const size_t open = meta.find('{', at);
    const size_t close = meta.find('}', open);
    if (open == std::string::npos || close == std::string::npos) throw std::runtime_error("malformed names metadata");
    auto names = parse_names_object(meta.substr(open, close - open + 1));
    if (names.empty()) throw std::runtime_error("empty names in engine metadata");
    return names;
}

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
