#pragma once

#include <vector>

struct Box {
    float x1{};
    float y1{};
    float x2{};
    float y2{};
    float score{};
};

struct LetterboxInfo {
    int inputW{};
    int inputH{};
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
