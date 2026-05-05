#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <vector>

#include "Detection.h"

/**
 * @brief 基于GDI的CPU屏幕采集器。
 *
 * 采集屏幕并在CPU上完成letterbox、BGRA到RGB、归一化和CHW排布。
 */
class ScreenCapture {
public:
    /**
     * @brief 创建屏幕采集器并分配像素和输入张量缓冲区。
     * @note 无返回值。
     */
    ScreenCapture();
    /**
     * @brief 释放GDI资源。
     * @note 无返回值。
     */
    ~ScreenCapture();

    /**
     * @brief 禁止复制构造采集器。
     * @param other 另一个采集器对象。
     * @note 无返回值。
     */
    ScreenCapture(ScreenCapture const& other) = delete;
    /**
     * @brief 禁止复制赋值采集器。
     * @param other 另一个采集器对象。
     * @return 当前对象引用。
     */
    ScreenCapture& operator=(ScreenCapture const& other) = delete;

    /**
     * @brief 采集屏幕并转换为模型输入张量。
     * @return 指向内部RGB CHW浮点输入张量的指针。
     */
    float* captureToTensor();

    /**
     * @brief 获取letterbox后的BGRA像素。
     * @return 像素缓冲区的只读引用。
     */
    std::vector<uint8_t> const& pixels() const { return pixels_; }
    /**
     * @brief 获取预览图像的BITMAPINFO。
     * @return BITMAPINFO的只读引用。
     */
    BITMAPINFO const& bmi() const { return bmi_; }
    /**
     * @brief 获取屏幕宽度。
     * @return 屏幕宽度，单位为像素。
     */
    int screenW() const { return screenW_; }
    /**
     * @brief 获取屏幕高度。
     * @return 屏幕高度，单位为像素。
     */
    int screenH() const { return screenH_; }
    /**
     * @brief 获取当前letterbox参数。
     * @return letterbox信息的只读引用。
     */
    LetterboxInfo const& letterbox() const { return letterbox_; }

private:
    int screenW_{};
    int screenH_{};
    LetterboxInfo letterbox_{};
    HDC screenDc_{};
    HDC memDc_{};
    HBITMAP bitmap_{};
    HBITMAP oldBitmap_{};
    BITMAPINFO screenBmi_{};
    BITMAPINFO bmi_{};
    std::vector<uint8_t> screenPixels_;
    std::vector<uint8_t> pixels_;
    std::vector<float> input_;
};
