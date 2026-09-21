#pragma once

#include <map>
#include <string>

namespace edgevision {

// Reads the `<engine>.names.json` sidecar written by scripts/build_engine.sh:
// a flat JSON object {"0": "person", "1": "bicycle", ...}. Minimal parser, no deps.
std::map<int, std::string> load_class_names(const std::string& path);

// `<dir>/<stem>.engine` -> `<dir>/<stem>.names.json`
std::string sidecar_names_path(const std::string& engine_path);

// Ultralytics-exported engines start with a 4-byte little-endian length followed by a
// JSON metadata document (names, imgsz, task...). Returns that JSON, or "" for a plain
// serialized engine (e.g. built with trtexec).
std::string read_ultralytics_metadata(const std::string& engine_path);

// Class names from the "names" object inside such a metadata document.
std::map<int, std::string> parse_names_object(const std::string& json_text);

// Sidecar first, then the engine's embedded metadata; throws if neither has names.
std::map<int, std::string> load_class_names_for_engine(const std::string& engine_path);

}  // namespace edgevision
