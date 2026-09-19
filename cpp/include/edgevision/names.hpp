#pragma once

#include <map>
#include <string>

namespace edgevision {

// Reads the `<engine>.names.json` sidecar written by scripts/build_engine.sh:
// a flat JSON object {"0": "person", "1": "bicycle", ...}. Minimal parser, no deps.
std::map<int, std::string> load_class_names(const std::string& path);

// `<dir>/<stem>.engine` -> `<dir>/<stem>.names.json`
std::string sidecar_names_path(const std::string& engine_path);

}  // namespace edgevision
