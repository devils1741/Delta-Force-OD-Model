#pragma once

#include <filesystem>
#include <string>

/**
 * @brief 模型文件和输入尺寸配置。
 *
 * 记录ONNX模型、TensorRT engine缓存以及模型输入张量的宽高。
 */
struct ModelConfig {
    std::filesystem::path onnxPath{"weights/best.onnx"};
    std::filesystem::path enginePath{"weights/best_640_trt10_16_sm89_fp16.engine"};
    int inputWidth{640};
    int inputHeight{640};
};

/**
 * @brief 推理后处理配置。
 *
 * 控制目标帧率、置信度阈值和最大检测数量。
 */
struct InferenceConfig {
    int targetFps{60};
    float scoreThreshold{0.30f};
    int maxDetections{300};
    int lostTargetFrameLimit{3};
    float minTargetWidthPx{24.0f};
    float minTargetHeightPx{48.0f};
    int logIntervalFrames{30};
    int overlayIntervalFrames{4};
};

/**
 * @brief 屏幕采集配置。
 *
 * 指定采集的显示器索引以及中心ROI的宽高。默认是1600*900
 */
struct CaptureConfig {
    int outputIndex{0};
    int roiWidth{1600};
    int roiHeight{900};
};

struct MouseConfig {
    std::string mode{"relative"};
    float relativeScale{1.0f};
    int moveCooldownFrames{12};
};

/**
 * @brief TensorRT构建配置。
 *
 * 描述是否启用FP16以及engine构建时的工作空间大小。
 */
struct TensorRtConfig {
    bool fp16{true};
    bool int8{false};
    std::filesystem::path calibrationCachePath{"weights/calibration.cache"};
    int workspaceMb{1024};
};

/**
 * @brief 应用全局配置单例。
 *
 * 从YAML文件加载运行参数，并向各模块提供只读配置访问。
 */
class AppConfig {
public:
    /**
     * @brief 获取全局配置实例。
     * @return 全局唯一的AppConfig引用。
     */
    static AppConfig& instance();

    /**
     * @brief 从YAML配置文件加载应用参数。
     * @param path 配置文件路径。
     * @note 无返回值。
     */
    void load(std::filesystem::path const& path);

    /**
     * @brief 获取模型配置。
     * @return 模型配置的只读引用。
     */
    ModelConfig const& model() const { return model_; }
    /**
     * @brief 获取推理配置。
     * @return 推理配置的只读引用。
     */
    InferenceConfig const& inference() const { return inference_; }
    /**
     * @brief 获取采集配置。
     * @return 采集配置的只读引用。
     */
    CaptureConfig const& capture() const { return capture_; }
    MouseConfig const& mouse() const { return mouse_; }
    /**
     * @brief 获取TensorRT配置。
     * @return TensorRT配置的只读引用。
     */
    TensorRtConfig const& tensorrt() const { return tensorrt_; }

private:
    /**
     * @brief 构造全局配置对象。
     * @note 无返回值。
     */
    AppConfig() = default;

    ModelConfig model_;
    InferenceConfig inference_;
    CaptureConfig capture_;
    MouseConfig mouse_;
    TensorRtConfig tensorrt_;
};
