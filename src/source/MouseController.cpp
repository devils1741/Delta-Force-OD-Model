#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "MouseController.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace {
#ifndef NDEBUG
constexpr bool kLogMoves = true;
#else
constexpr bool kLogMoves = false;
#endif

bool isLeftButtonRequested() {
    SHORT state = GetAsyncKeyState(VK_LBUTTON);
    return (state & 0x8000) != 0;
}

bool aimFromScreenCenter(MouseConfig const& mouse) {
    return mouse.mode == "relative";
}

int nearestTargetIndexToPoint(
    std::vector<Box> const& boxes,
    POINT const& point) {
    int bestIndex = -1;
    double bestDistanceSq = 0.0;
    for (size_t i = 0; i < boxes.size(); ++i) {
        Box const& box = boxes[i];
        double centerX = (box.x1 + box.x2) * 0.5;
        double centerY = (box.y1 + box.y2) * 0.5;
        double dx = centerX - point.x;
        double dy = centerY - point.y;
        double distanceSq = dx * dx + dy * dy;
        if (bestIndex < 0 || distanceSq < bestDistanceSq) {
            bestIndex = static_cast<int>(i);
            bestDistanceSq = distanceSq;
        }
    }
    return bestIndex;
}

POINT targetCenter(Box const& target) {
    return POINT{
        static_cast<LONG>(std::lround((target.x1 + target.x2) * 0.5f)),
        static_cast<LONG>(std::lround((target.y1 + target.y2) * 0.5f)),
    };
}

POINT targetGlobalCenter(Box const& target, LetterboxInfo const& letterbox) {
    POINT center = targetCenter(target);
    return POINT{
        center.x + letterbox.desktopX,
        center.y + letterbox.desktopY,
    };
}

struct MouseMovePlan {
    POINT targetGlobal{};
    float errorX = 0.0f;
    float errorY = 0.0f;
    POINT delta{};
};

MouseMovePlan planMouseMove(Box const& target, LetterboxInfo const& letterbox, POINT const& cursor) {
    MouseMovePlan plan{};
    plan.targetGlobal = targetGlobalCenter(target, letterbox);
    plan.errorX = static_cast<float>(plan.targetGlobal.x - cursor.x);
    plan.errorY = static_cast<float>(plan.targetGlobal.y - cursor.y);

    constexpr float kDeadzonePx = 2.0f;
    if (std::fabs(plan.errorX) <= kDeadzonePx && std::fabs(plan.errorY) <= kDeadzonePx) {
        return plan;
    }

    int dx = static_cast<int>(std::lround(plan.errorX));
    int dy = static_cast<int>(std::lround(plan.errorY));
    if (dx == 0 && std::fabs(plan.errorX) > kDeadzonePx) {
        dx = plan.errorX > 0.0f ? 1 : -1;
    }
    if (dy == 0 && std::fabs(plan.errorY) > kDeadzonePx) {
        dy = plan.errorY > 0.0f ? 1 : -1;
    }
    plan.delta = POINT{dx, dy};
    return plan;
}

bool sendRelativeMouseMove(POINT const& delta, MouseConfig const& mouse) {
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dx = static_cast<LONG>(std::lround(static_cast<float>(delta.x) * mouse.relativeScale));
    input.mi.dy = static_cast<LONG>(std::lround(static_cast<float>(delta.y) * mouse.relativeScale));
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    if (input.mi.dx == 0 && delta.x != 0) {
        input.mi.dx = delta.x > 0 ? 1 : -1;
    }
    if (input.mi.dy == 0 && delta.y != 0) {
        input.mi.dy = delta.y > 0 ? 1 : -1;
    }
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

bool applyMouseMovePlan(
    MouseMovePlan const& plan,
    POINT const& referencePoint,
    MouseConfig const& mouse) {
    if (!mouse.enabled) {
        return false;
    }
    if (plan.delta.x == 0 && plan.delta.y == 0) {
        return true;
    }
    if (mouse.mode == "absolute") {
        return SetCursorPos(referencePoint.x + plan.delta.x, referencePoint.y + plan.delta.y) != FALSE;
    }
    if (mouse.mode == "both") {
        // Compatibility mode: absolute first, relative only if the call fails.
        if (SetCursorPos(referencePoint.x + plan.delta.x, referencePoint.y + plan.delta.y)) {
            return true;
        }
        return sendRelativeMouseMove(plan.delta, mouse);
    }
    return sendRelativeMouseMove(plan.delta, mouse);
}

} // namespace

MouseController::MouseController(MouseConfig const& config) : config_(config) {}

void MouseController::update(std::vector<Box> const& boxes,
                             LetterboxInfo const& letterbox, uint64_t sequence) {
    bool leftRequested = isLeftButtonRequested();
    POINT cursor{};
    bool hasCursor = GetCursorPos(&cursor) != FALSE;
    POINT screenCenter{
        letterbox.screenW / 2,
        letterbox.screenH / 2,
    };
    POINT screenCenterGlobal{
        letterbox.desktopX + screenCenter.x,
        letterbox.desktopY + screenCenter.y,
    };
    bool useScreenCenterAim = aimFromScreenCenter(config_);
    POINT aimReferenceGlobal = useScreenCenterAim ? screenCenterGlobal : cursor;
    POINT aimReferenceLocal{
        aimReferenceGlobal.x - letterbox.desktopX,
        aimReferenceGlobal.y - letterbox.desktopY,
    };
    if (!config_.enabled || !leftRequested || !hasCursor || boxes.empty()) {
        hasLockedTarget_ = false;
        cooldownFramesRemaining_ = 0;
        return;
    }
    // Associate the previous box by class and overlap, independently of vector order.
    int selectedTargetIndex = -1;
    float bestOverlap = 0.25f;
    if (hasLockedTarget_) {
        for (size_t i = 0; i < boxes.size(); ++i) {
            auto const& box = boxes[i];
            if (box.classId != lockedTarget_.classId) continue;
            float intersection = std::max(0.0f, std::min(box.x2, lockedTarget_.x2) - std::max(box.x1, lockedTarget_.x1)) *
                                 std::max(0.0f, std::min(box.y2, lockedTarget_.y2) - std::max(box.y1, lockedTarget_.y1));
            float total = (box.x2-box.x1)*(box.y2-box.y1) +
                          (lockedTarget_.x2-lockedTarget_.x1)*(lockedTarget_.y2-lockedTarget_.y1) - intersection;
            float overlap = total > 0.0f ? intersection / total : 0.0f;
            if (overlap > bestOverlap) {
                bestOverlap = overlap;
                selectedTargetIndex = static_cast<int>(i);
            }
        }
    }
    if (selectedTargetIndex < 0) {
        selectedTargetIndex = nearestTargetIndexToPoint(boxes, aimReferenceLocal);
    }
    lockedTarget_ = boxes[static_cast<size_t>(selectedTargetIndex)];
    hasLockedTarget_ = true;
    if (cooldownFramesRemaining_ > 0) {
        --cooldownFramesRemaining_;
        return;
    }
    if (selectedTargetIndex >= 0) {
        MouseMovePlan selectedMovePlan = planMouseMove(
            boxes[static_cast<size_t>(selectedTargetIndex)],
            letterbox,
            aimReferenceGlobal);
        bool moveSucceeded =
            moveBy(selectedMovePlan.delta.x, selectedMovePlan.delta.y,
                   aimReferenceGlobal.x, aimReferenceGlobal.y);
        if (moveSucceeded && (selectedMovePlan.delta.x != 0 || selectedMovePlan.delta.y != 0)) {
            cooldownFramesRemaining_ = config_.moveCooldownFrames;
        }
        if constexpr (kLogMoves) {
            std::cout << "move frame=" << sequence
                      << " target=" << selectedTargetIndex
                      << " dx=" << selectedMovePlan.delta.x
                      << " dy=" << selectedMovePlan.delta.y
                      << " ok=" << (moveSucceeded ? 1 : 0)
                      << '\n';
        }
    }
}

bool MouseController::moveBy(int dx, int dy, int referenceX, int referenceY) {
    MouseMovePlan plan{};
    plan.delta = POINT{dx, dy};
    return applyMouseMovePlan(plan, POINT{referenceX, referenceY}, config_);
}
