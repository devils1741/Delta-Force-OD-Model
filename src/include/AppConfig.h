#pragma once

#include <filesystem>

struct ModelConfig {
    std::filesystem::path onnxPath{"weights/best.onnx"};
    std::filesystem::path enginePath{"weights/best_640_trt10_16_sm89_fp16.engine"};
};

struct InferenceConfig {
    int targetFps{60};
    float scoreThreshold{0.30f};
    float nmsThreshold{0.45f};
    int maxDetections{300};
};

struct CaptureConfig {
    int outputIndex{0};
    int roiWidth{1600};
    int roiHeight{900};
};

struct TensorRtConfig {
    bool fp16{true};
    int workspaceMb{1024};
};

class AppConfig {
public:
    static AppConfig& instance();

    void load(std::filesystem::path const& path);

    ModelConfig const& model() const { return model_; }
    InferenceConfig const& inference() const { return inference_; }
    CaptureConfig const& capture() const { return capture_; }
    TensorRtConfig const& tensorrt() const { return tensorrt_; }

private:
    AppConfig() = default;

    ModelConfig model_;
    InferenceConfig inference_;
    CaptureConfig capture_;
    TensorRtConfig tensorrt_;
};
