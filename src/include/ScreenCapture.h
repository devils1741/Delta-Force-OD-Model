#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdint>
#include <vector>

#include "Detection.h"

class ScreenCapture {
public:
    ScreenCapture();
    ~ScreenCapture();

    ScreenCapture(ScreenCapture const&) = delete;
    ScreenCapture& operator=(ScreenCapture const&) = delete;

    float* captureToTensor();

    std::vector<uint8_t> const& pixels() const { return pixels_; }
    BITMAPINFO const& bmi() const { return bmi_; }
    int screenW() const { return screenW_; }
    int screenH() const { return screenH_; }
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
