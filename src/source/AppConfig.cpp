#include "AppConfig.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
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
    inference_.nmsThreshold = readOr(inference, "nms_threshold", inference_.nmsThreshold);
    inference_.maxDetections = readOr(inference, "max_detections", inference_.maxDetections);

    auto capture = root["capture"];
    capture_.outputIndex = readOr(capture, "output_index", capture_.outputIndex);
    capture_.roiWidth = readOr(capture, "roi_width", capture_.roiWidth);
    capture_.roiHeight = readOr(capture, "roi_height", capture_.roiHeight);

    auto tensorrt = root["tensorrt"];
    tensorrt_.fp16 = readOr(tensorrt, "fp16", tensorrt_.fp16);
    tensorrt_.workspaceMb = readOr(tensorrt, "workspace_mb", tensorrt_.workspaceMb);

    validatePositive("model.input_width", model_.inputWidth);
    validatePositive("model.input_height", model_.inputHeight);
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
