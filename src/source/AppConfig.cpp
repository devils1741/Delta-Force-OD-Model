#include "AppConfig.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

template <typename T>
/**
 * @brief 从YAML节点读取指定键，缺失时返回默认值。
 * @tparam T需要读取的目标类型。
 * @param node YAML节点。
 * @param key 需要读取的键名。
 * @param fallback 键不存在时使用的默认值。
 * @return YAML中的配置值，或fallback默认值。
 */
T readOr(YAML::Node const& node, char const* key, T fallback) {
    if (!node || !node[key]) {
        return fallback;
    }
    return node[key].as<T>();
}

/**
 * @brief 将配置中的相对路径解析为项目绝对路径。
 * @param baseDir 相对路径的基准目录。
 * @param path 待解析路径。
 * @return 绝对路径；空路径或已是绝对路径时原样返回。
 */
fs::path resolvePath(fs::path const& baseDir, fs::path path) {
    if (path.empty() || path.is_absolute()) {
        return path;
    }
    return fs::absolute(baseDir / path);
}

/**
 * @brief 校验整数配置必须为正数。
 * @param name 配置项名称，用于错误提示。
 * @param value 配置项数值。
 * @note 无返回值。
 */
void validatePositive(char const* name, int value) {
    if (value <= 0) {
        throw std::runtime_error(std::string(name) + " must be greater than 0");
    }
}

std::string lowerCopy(std::string value) {
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
    return value;
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
    model_.inputWidth = readOr(model, "input_width", model_.inputWidth);
    model_.inputHeight = readOr(model, "input_height", model_.inputHeight);

    auto inference = root["inference"];
    inference_.targetFps = readOr(inference, "target_fps", inference_.targetFps);
    inference_.scoreThreshold = readOr(inference, "score_threshold", inference_.scoreThreshold);
    inference_.maxDetections = readOr(inference, "max_detections", inference_.maxDetections);
    inference_.lostTargetFrameLimit =
        readOr(inference, "lost_target_frame_limit", inference_.lostTargetFrameLimit);
    inference_.minTargetWidthPx = readOr(inference, "min_target_width_px", inference_.minTargetWidthPx);
    inference_.minTargetHeightPx = readOr(inference, "min_target_height_px", inference_.minTargetHeightPx);
    inference_.logIntervalFrames = readOr(inference, "log_interval_frames", inference_.logIntervalFrames);
    inference_.overlayIntervalFrames =
        readOr(inference, "overlay_interval_frames", inference_.overlayIntervalFrames);

    auto capture = root["capture"];
    capture_.outputIndex = readOr(capture, "output_index", capture_.outputIndex);
    capture_.roiWidth = readOr(capture, "roi_width", capture_.roiWidth);
    capture_.roiHeight = readOr(capture, "roi_height", capture_.roiHeight);

    auto mouse = root["mouse"];
    mouse_.mode = lowerCopy(readOr(mouse, "mode", mouse_.mode));
    mouse_.relativeScale = readOr(mouse, "relative_scale", mouse_.relativeScale);
    mouse_.moveCooldownFrames = readOr(mouse, "move_cooldown_frames", mouse_.moveCooldownFrames);

    auto tensorrt = root["tensorrt"];
    tensorrt_.fp16 = readOr(tensorrt, "fp16", tensorrt_.fp16);
    tensorrt_.int8 = readOr(tensorrt, "int8", tensorrt_.int8);
    tensorrt_.calibrationCachePath = resolvePath(
        projectDir,
        fs::path(readOr(tensorrt, "calibration_cache", tensorrt_.calibrationCachePath.string())));
    tensorrt_.workspaceMb = readOr(tensorrt, "workspace_mb", tensorrt_.workspaceMb);

    validatePositive("model.input_width", model_.inputWidth);
    validatePositive("model.input_height", model_.inputHeight);
    validatePositive("inference.target_fps", inference_.targetFps);
    validatePositive("inference.max_detections", inference_.maxDetections);
    validatePositive("inference.lost_target_frame_limit", inference_.lostTargetFrameLimit);
    validatePositive("inference.log_interval_frames", inference_.logIntervalFrames);
    validatePositive("inference.overlay_interval_frames", inference_.overlayIntervalFrames);
    validatePositive("capture.roi_width", capture_.roiWidth);
    validatePositive("capture.roi_height", capture_.roiHeight);
    validatePositive("tensorrt.workspace_mb", tensorrt_.workspaceMb);
    if (capture_.outputIndex < 0) {
        throw std::runtime_error("capture.output_index must be greater than or equal to 0");
    }

    inference_.scoreThreshold = std::clamp(inference_.scoreThreshold, 0.0f, 1.0f);
    if (inference_.minTargetWidthPx < 0.0f || inference_.minTargetHeightPx < 0.0f) {
        throw std::runtime_error("inference min target size must be greater than or equal to 0");
    }
    if (mouse_.mode != "absolute" && mouse_.mode != "relative" && mouse_.mode != "both") {
        throw std::runtime_error("mouse.mode must be absolute, relative, or both");
    }
    if (mouse_.relativeScale <= 0.0f) {
        throw std::runtime_error("mouse.relative_scale must be greater than 0");
    }
    if (mouse_.moveCooldownFrames < 0) {
        throw std::runtime_error("mouse.move_cooldown_frames must be greater than or equal to 0");
    }
}
