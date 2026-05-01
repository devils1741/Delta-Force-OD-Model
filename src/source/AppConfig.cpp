#include "AppConfig.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

template <typename T>
T readOr(YAML::Node const& node, char const* key, T fallback) {
    if (!node || !node[key]) {
        return fallback;
    }
    return node[key].as<T>();
}

fs::path resolvePath(fs::path const& baseDir, fs::path path) {
    if (path.empty() || path.is_absolute()) {
        return path;
    }
    return fs::absolute(baseDir / path);
}

void validatePositive(char const* name, int value) {
    if (value <= 0) {
        throw std::runtime_error(std::string(name) + " must be greater than 0");
    }
}

} // namespace

AppConfig& AppConfig::instance() {
    static AppConfig config;
    return config;
}

void AppConfig::load(fs::path const& path) {
    YAML::Node root = YAML::LoadFile(path.string());
    fs::path configDir = fs::absolute(path).parent_path();
    fs::path projectDir = configDir.parent_path();

    auto model = root["model"];
    model_.onnxPath = resolvePath(projectDir, fs::path(readOr(model, "onnx_path", model_.onnxPath.string())));
    model_.enginePath = resolvePath(projectDir, fs::path(readOr(model, "engine_path", model_.enginePath.string())));

    auto inference = root["inference"];
    inference_.targetFps = readOr(inference, "target_fps", inference_.targetFps);
    inference_.scoreThreshold = readOr(inference, "score_threshold", inference_.scoreThreshold);
    inference_.nmsThreshold = readOr(inference, "nms_threshold", inference_.nmsThreshold);
    inference_.maxDetections = readOr(inference, "max_detections", inference_.maxDetections);

    auto capture = root["capture"];
    capture_.outputIndex = readOr(capture, "output_index", capture_.outputIndex);
    capture_.roiWidth = readOr(capture, "roi_width", capture_.roiWidth);
    capture_.roiHeight = readOr(capture, "roi_height", capture_.roiHeight);

    auto tensorrt = root["tensorrt"];
    tensorrt_.fp16 = readOr(tensorrt, "fp16", tensorrt_.fp16);
    tensorrt_.workspaceMb = readOr(tensorrt, "workspace_mb", tensorrt_.workspaceMb);

    validatePositive("inference.target_fps", inference_.targetFps);
    validatePositive("inference.max_detections", inference_.maxDetections);
    validatePositive("capture.roi_width", capture_.roiWidth);
    validatePositive("capture.roi_height", capture_.roiHeight);
    validatePositive("tensorrt.workspace_mb", tensorrt_.workspaceMb);
    if (capture_.outputIndex < 0) {
        throw std::runtime_error("capture.output_index must be greater than or equal to 0");
    }

    inference_.scoreThreshold = std::clamp(inference_.scoreThreshold, 0.0f, 1.0f);
    inference_.nmsThreshold = std::clamp(inference_.nmsThreshold, 0.0f, 1.0f);
}
