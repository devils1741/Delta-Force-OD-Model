#pragma once

#include <vector>

constexpr int kInputW = 640;
constexpr int kInputH = 640;
constexpr float kScoreThreshold = 0.30f;
constexpr float kNmsThreshold = 0.45f;

struct Box {
    float x1{};
    float y1{};
    float x2{};
    float y2{};
    float score{};
};

struct LetterboxInfo {
    int screenW{};
    int screenH{};
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

std::vector<Box> decodeAndNms(std::vector<float> const& output, LetterboxInfo const& letterbox);
