#pragma once

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
    int classId{};
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
