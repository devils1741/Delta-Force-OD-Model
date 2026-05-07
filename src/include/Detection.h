#pragma once

#include <vector>

/**
 * @brief 目标检测框。
 *
 * 保存屏幕坐标中的矩形边界和置信度分数。
 */
struct Box {
    float x1{};
    float y1{};
    float x2{};
    float y2{};
    float score{};
};

/**
 * @brief Letterbox缩放和坐标映射信息。
 *
 * 保存屏幕ROI到模型输入尺寸之间的缩放、填充和反变换参数。
 */
struct LetterboxInfo {
    int inputW{};
    int inputH{};
    int screenW{};
    int screenH{};
    int desktopX{};
    int desktopY{};
    int captureX{};
    int captureY{};
    int captureW{};
    int captureH{};
    int resizedW{};
    int resizedH{};
    int padX{};
    int padY{};
    float scale{1.0f};
};

/**
 * @brief 解码端到端模型输出。
 * @param output TensorRT输出缓冲区，按每个候选框6个浮点值组织。
 * @param letterbox 模型输入和屏幕坐标之间的letterbox映射信息。
 * @return 经过置信度过滤和坐标还原后的检测框列表。
 */
std::vector<Box> decodeDetections(std::vector<float> const& output, LetterboxInfo const& letterbox);
