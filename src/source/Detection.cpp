#include "Detection.h"

#include "AppConfig.h"

#include <algorithm>
#include <cmath>

namespace {

float iou(Box const& a, Box const& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float areaA = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float areaB = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    return inter / std::max(1e-6f, areaA + areaB - inter);
}

} // namespace

std::vector<Box> decodeAndNms(std::vector<float> const& output, LetterboxInfo const& letterbox) {
    auto const& config = AppConfig::instance().inference();
    std::vector<Box> boxes;
    int maxDetections = std::min(config.maxDetections, static_cast<int>(output.size() / 6));
    for (int i = 0; i < maxDetections; ++i) {
        float a = output[i * 6 + 0];
        float b = output[i * 6 + 1];
        float c = output[i * 6 + 2];
        float d = output[i * 6 + 3];
        float score = output[i * 6 + 4];

        if (score < config.scoreThreshold) {
            continue;
        }

        if (std::max({std::fabs(a), std::fabs(b), std::fabs(c), std::fabs(d)}) <= 2.0f) {
            a *= kInputW;
            c *= kInputW;
            b *= kInputH;
            d *= kInputH;
        }

        float x1 = std::min(a, c);
        float y1 = std::min(b, d);
        float x2 = std::max(a, c);
        float y2 = std::max(b, d);

        x1 = (x1 - letterbox.padX) / letterbox.scale + letterbox.captureX;
        y1 = (y1 - letterbox.padY) / letterbox.scale + letterbox.captureY;
        x2 = (x2 - letterbox.padX) / letterbox.scale + letterbox.captureX;
        y2 = (y2 - letterbox.padY) / letterbox.scale + letterbox.captureY;

        x1 = std::clamp(x1, 0.0f, static_cast<float>(letterbox.screenW - 1));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(letterbox.screenH - 1));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(letterbox.screenW - 1));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(letterbox.screenH - 1));

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        boxes.push_back({x1, y1, x2, y2, score});
    }

    std::sort(boxes.begin(), boxes.end(), [](Box const& lhs, Box const& rhs) {
        return lhs.score > rhs.score;
    });

    std::vector<Box> kept;
    for (auto const& box : boxes) {
        bool suppressed = false;
        for (auto const& selected : kept) {
            if (iou(box, selected) > config.nmsThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(box);
        }
    }
    return kept;
}
