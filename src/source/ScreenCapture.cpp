#include "ScreenCapture.h"

#include "Detection.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

ScreenCapture::ScreenCapture() {
    DEVMODEW mode{};
    mode.dmSize = sizeof(mode);
    if (EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &mode)) {
        screenW_ = static_cast<int>(mode.dmPelsWidth);
        screenH_ = static_cast<int>(mode.dmPelsHeight);
    } else {
        screenW_ = GetSystemMetrics(SM_CXSCREEN);
        screenH_ = GetSystemMetrics(SM_CYSCREEN);
    }

    letterbox_.screenW = screenW_;
    letterbox_.screenH = screenH_;
    letterbox_.captureX = 0;
    letterbox_.captureY = 0;
    letterbox_.captureW = screenW_;
    letterbox_.captureH = screenH_;
    letterbox_.scale = std::min(
        static_cast<float>(kInputW) / static_cast<float>(letterbox_.captureW),
        static_cast<float>(kInputH) / static_cast<float>(letterbox_.captureH));
    letterbox_.resizedW = std::max(1, static_cast<int>(std::round(letterbox_.captureW * letterbox_.scale)));
    letterbox_.resizedH = std::max(1, static_cast<int>(std::round(letterbox_.captureH * letterbox_.scale)));
    letterbox_.padX = (kInputW - letterbox_.resizedW) / 2;
    letterbox_.padY = (kInputH - letterbox_.resizedH) / 2;

    screenDc_ = GetDC(nullptr);
    memDc_ = CreateCompatibleDC(screenDc_);
    bitmap_ = CreateCompatibleBitmap(screenDc_, screenW_, screenH_);
    oldBitmap_ = static_cast<HBITMAP>(SelectObject(memDc_, bitmap_));

    screenPixels_.resize(screenW_ * screenH_ * 4);
    pixels_.resize(kInputW * kInputH * 4);
    input_.resize(3 * kInputW * kInputH);

    ZeroMemory(&screenBmi_, sizeof(screenBmi_));
    screenBmi_.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    screenBmi_.bmiHeader.biWidth = screenW_;
    screenBmi_.bmiHeader.biHeight = -screenH_;
    screenBmi_.bmiHeader.biPlanes = 1;
    screenBmi_.bmiHeader.biBitCount = 32;
    screenBmi_.bmiHeader.biCompression = BI_RGB;

    ZeroMemory(&bmi_, sizeof(bmi_));
    bmi_.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi_.bmiHeader.biWidth = kInputW;
    bmi_.bmiHeader.biHeight = -kInputH;
    bmi_.bmiHeader.biPlanes = 1;
    bmi_.bmiHeader.biBitCount = 32;
    bmi_.bmiHeader.biCompression = BI_RGB;
}

ScreenCapture::~ScreenCapture() {
    if (memDc_ && oldBitmap_) {
        SelectObject(memDc_, oldBitmap_);
    }
    if (bitmap_) {
        DeleteObject(bitmap_);
    }
    if (memDc_) {
        DeleteDC(memDc_);
    }
    if (screenDc_) {
        ReleaseDC(nullptr, screenDc_);
    }
}

float* ScreenCapture::captureToTensor() {
    if (!BitBlt(memDc_, 0, 0, screenW_, screenH_, screenDc_, 0, 0, SRCCOPY)) {
        throw std::runtime_error("BitBlt failed");
    }
    if (!GetDIBits(memDc_, bitmap_, 0, screenH_, screenPixels_.data(), &screenBmi_, DIB_RGB_COLORS)) {
        throw std::runtime_error("GetDIBits failed");
    }

    std::fill(pixels_.begin(), pixels_.end(), 114);

    for (int y = 0; y < letterbox_.resizedH; ++y) {
        int srcY = std::clamp(
            letterbox_.captureY + static_cast<int>(y / letterbox_.scale),
            0,
            screenH_ - 1);
        int dstY = y + letterbox_.padY;
        for (int x = 0; x < letterbox_.resizedW; ++x) {
            int srcX = std::clamp(
                letterbox_.captureX + static_cast<int>(x / letterbox_.scale),
                0,
                screenW_ - 1);
            int dstX = x + letterbox_.padX;
            std::memcpy(
                &pixels_[(dstY * kInputW + dstX) * 4],
                &screenPixels_[(srcY * screenW_ + srcX) * 4],
                4);
        }
    }

    auto plane = kInputW * kInputH;
    for (int y = 0; y < kInputH; ++y) {
        for (int x = 0; x < kInputW; ++x) {
            int src = (y * kInputW + x) * 4;
            int dst = y * kInputW + x;
            input_[dst] = pixels_[src + 2] / 255.0f;
            input_[plane + dst] = pixels_[src + 1] / 255.0f;
            input_[2 * plane + dst] = pixels_[src + 0] / 255.0f;
        }
    }
    return input_.data();
}
