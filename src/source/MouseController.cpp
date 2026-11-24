#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "MouseController.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <future>
#include <limits>
#include <stdexcept>
#include <syncstream>
#include <vector>

namespace {
struct ButtonEventState {
    std::atomic<bool>* down;
    std::atomic<uint64_t>* rawEvents;
    std::atomic<uint64_t>* downEvents;
    std::atomic<uint64_t>* upEvents;
    std::atomic<uint64_t>* injectedEvents;
    std::atomic<uint64_t>* injectedDown;
    std::atomic<uint64_t>* injectedUp;
};

thread_local ButtonEventState* hookButtonState = nullptr;

LRESULT CALLBACK mouseButtonHook(int code, WPARAM message, LPARAM data) {
    if (code == HC_ACTION && hookButtonState) {
        auto const* event = reinterpret_cast<MSLLHOOKSTRUCT const*>(data);
        auto& state = *hookButtonState;
        if (event->flags & LLMHF_INJECTED) {
            ++*state.injectedEvents;
            if (message == WM_LBUTTONDOWN) ++*state.injectedDown;
            if (message == WM_LBUTTONUP) ++*state.injectedUp;
        }
        // Only physical input controls the trigger; never consume other apps' events.
        if (!(event->flags & LLMHF_INJECTED)) {
            ++*state.rawEvents;
            if (message == WM_LBUTTONDOWN) {
                ++*state.downEvents;
                state.down->store(true);
            } else if (message == WM_LBUTTONUP) {
                ++*state.upEvents;
                state.down->store(false);
            }
        }
    }
    return CallNextHookEx(nullptr, code, message, data);
}

bool aimFromScreenCenter(MouseConfig const& mouse) {
    return mouse.mode == "relative";
}

struct AimTarget {
    Box source;
    float x;
    float y;
    int priority;
};

bool validEnemy(Box const& box) {
    return box.classId == 0 && std::isfinite(box.x1) && std::isfinite(box.y1) &&
           std::isfinite(box.x2) && std::isfinite(box.y2) && box.x2 > box.x1 && box.y2 > box.y1;
}

bool sameTarget(Box const& current, Box const& previous) {
    float intersection = std::max(0.0f, std::min(current.x2, previous.x2) - std::max(current.x1, previous.x1)) *
                         std::max(0.0f, std::min(current.y2, previous.y2) - std::max(current.y1, previous.y1));
    float total = (current.x2-current.x1)*(current.y2-current.y1) +
                  (previous.x2-previous.x1)*(previous.y2-previous.y1) - intersection;
    return total > 0 && intersection / total >= 0.25f;
}

int nearestTargetIndexToPoint(
    std::vector<AimTarget> const& boxes,
    POINT const& point) {
    int bestIndex = -1;
    double bestDistanceSq = 0.0;
    for (size_t i = 0; i < boxes.size(); ++i) {
        auto const& box = boxes[i];
        double centerX = box.x;
        double centerY = box.y;
        double dx = centerX - point.x;
        double dy = centerY - point.y;
        double distanceSq = dx * dx + dy * dy;
        if (bestIndex < 0 || box.priority < boxes[bestIndex].priority ||
            (box.priority == boxes[bestIndex].priority && distanceSq < bestDistanceSq)) {
            bestIndex = static_cast<int>(i);
            bestDistanceSq = distanceSq;
        }
    }
    return bestIndex;
}

// Associate geometry, not vector order or the point's distance from the crosshair.
// Reject ambiguous matches rather than steering to a different nearby enemy.
int associatedTargetIndex(std::vector<AimTarget> const& boxes, Box const& expected) {
    int best = -1;
    double bestCost = 1e9, secondCost = 1e9;
    double w = expected.x2 - expected.x1, h = expected.y2 - expected.y1;
    for (size_t i = 0; i < boxes.size(); ++i) {
        auto const& box = boxes[i].source;
        double wr = (box.x2 - box.x1) / w, hr = (box.y2 - box.y1) / h;
        if (wr < 0.5 || wr > 2 || hr < 0.5 || hr > 2) continue;
        double dx = ((box.x1 + box.x2) - (expected.x1 + expected.x2)) * 0.5 / std::max(24.0, w);
        double dy = ((box.y1 + box.y2) - (expected.y1 + expected.y2)) * 0.5 / std::max(24.0, h);
        if (std::abs(dx) > 0.65 || std::abs(dy) > 0.65) continue;
        double intersection = std::max(0.0f, std::min(box.x2, expected.x2) - std::max(box.x1, expected.x1)) *
                              std::max(0.0f, std::min(box.y2, expected.y2) - std::max(box.y1, expected.y1));
        double total = w*h + (box.x2-box.x1)*(box.y2-box.y1) - intersection;
        double cost = dx*dx + dy*dy + 0.25*(std::abs(std::log(wr)) + std::abs(std::log(hr))) - intersection/total;
        if (cost < bestCost) { secondCost = bestCost; bestCost = cost; best = static_cast<int>(i); }
        else secondCost = std::min(secondCost, cost);
    }
    return secondCost - bestCost < 0.15 ? -1 : best;
}

POINT targetGlobalCenter(AimTarget const& target, LetterboxInfo const& letterbox) {
    return POINT{
        static_cast<LONG>(std::lround(target.x)) + letterbox.desktopX,
        static_cast<LONG>(std::lround(target.y)) + letterbox.desktopY,
    };
}

struct MouseMovePlan {
    POINT targetGlobal{};
    float errorX = 0.0f;
    float errorY = 0.0f;
    POINT delta{};
};

MouseMovePlan planMouseMove(AimTarget const& target, LetterboxInfo const& letterbox, POINT const& cursor) {
    MouseMovePlan plan{};
    plan.targetGlobal = targetGlobalCenter(target, letterbox);
    plan.errorX = static_cast<float>(plan.targetGlobal.x - cursor.x);
    plan.errorY = static_cast<float>(plan.targetGlobal.y - cursor.y);

    int dx = static_cast<int>(std::lround(plan.errorX));
    int dy = static_cast<int>(std::lround(plan.errorY));
    plan.delta = POINT{dx, dy};
    return plan;
}

POINT scaledRelativeDelta(POINT const& delta, MouseConfig const& mouse) {
    POINT scaled{
        static_cast<LONG>(std::lround(static_cast<float>(delta.x) * mouse.relativeScale)),
        static_cast<LONG>(std::lround(static_cast<float>(delta.y) * mouse.relativeScale)),
    };
    if (scaled.x == 0 && delta.x != 0) {
        scaled.x = delta.x > 0 ? 1 : -1;
    }
    if (scaled.y == 0 && delta.y != 0) {
        scaled.y = delta.y > 0 ? 1 : -1;
    }
    return scaled;
}

LONG relativeAimAxis(LONG error, MouseConfig const& mouse, float axisScale) {
    if (std::abs(static_cast<double>(error)) <= mouse.relativeDeadzonePx) return 0;
    double gain = mouse.relativeDirect ? 1.0 : mouse.relativeGain;
    double limit = mouse.relativeDirect
        ? static_cast<double>(std::numeric_limits<LONG>::max()) : mouse.relativeMaxStep;
    double step = std::clamp(static_cast<double>(error) * mouse.relativeScale * axisScale * gain, -limit, limit);
    LONG rounded = static_cast<LONG>(std::lround(step));
    return rounded != 0 ? rounded : (error > 0 ? 1 : -1);
}

// Smoothing must never keep steering toward an old point after the camera passes it.
double currentRelativeError(double filtered, double current, float deadzone) {
    if (std::abs(current) <= deadzone) return 0;
    if (filtered * current <= 0) return current;
    return std::copysign(std::min(std::abs(filtered), std::abs(current)), current);
}

bool sendRelativeMouseMove(POINT const& delta, MouseConfig const& mouse, POINT& relativeDelta,
                           bool aimCorrection) {
    relativeDelta = aimCorrection
        ? POINT{relativeAimAxis(delta.x, mouse, mouse.relativeScaleX),
                relativeAimAxis(delta.y, mouse, mouse.relativeScaleY)}
        : scaledRelativeDelta(delta, mouse);
    if (relativeDelta.x == 0 && relativeDelta.y == 0) return true;
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dx = relativeDelta.x;
    input.mi.dy = relativeDelta.y;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

bool sendAbsoluteMouseMove(POINT const& target) {
    int originX = GetSystemMetrics(SM_XVIRTUALSCREEN);
    int originY = GetSystemMetrics(SM_YVIRTUALSCREEN);
    int width = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    int height = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    if (width <= 0 || height <= 0) {
        SetLastError(ERROR_INVALID_PARAMETER);
        return false;
    }
    // Aim at the pixel center in the 0..65535 virtual-desktop coordinate range.
    auto normalize = [](LONG position, int origin, int extent) -> LONG {
        int64_t pixel = std::clamp<int64_t>(static_cast<int64_t>(position) - origin, 0, extent - 1);
        return static_cast<LONG>(((pixel * 2 + 1) * 65536) / (static_cast<int64_t>(extent) * 2));
    };
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dx = normalize(target.x, originX, width);
    input.mi.dy = normalize(target.y, originY, height);
    input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK;
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

bool applyMouseMovePlan(
    MouseMovePlan const& plan,
    POINT const& referencePoint,
    MouseConfig const& mouse,
    POINT& relativeDelta,
    bool aimCorrection) {
    if (!mouse.enabled) {
        return false;
    }
    if (plan.delta.x == 0 && plan.delta.y == 0) {
        return true;
    }
    if (mouse.mode == "absolute") {
        return sendAbsoluteMouseMove(POINT{referencePoint.x + plan.delta.x, referencePoint.y + plan.delta.y});
    }
    if (mouse.mode == "both") {
        // Compatibility mode: absolute first, relative only if the call fails.
        if (SetCursorPos(referencePoint.x + plan.delta.x, referencePoint.y + plan.delta.y)) {
            return true;
        }
        return sendRelativeMouseMove(plan.delta, mouse, relativeDelta, aimCorrection);
    }
    return sendRelativeMouseMove(plan.delta, mouse, relativeDelta, aimCorrection);
}

} // namespace

MouseController::MouseController(MouseConfig const& config) : config_(config) {
    std::promise<DWORD> ready;
    auto result = ready.get_future();
    inputThread_ = std::thread([this, ready = std::move(ready)]() mutable {
        ButtonEventState state{&leftButtonDown_,
            &rawMouseEvents_, &leftDownEvents_, &leftUpEvents_,
            &injectedMouseEvents_, &injectedDownEvents_, &injectedUpEvents_};
        hookButtonState = &state;
        MSG message{};
        // Create the thread message queue before publishing its ID.
        PeekMessageW(&message, nullptr, 0, 0, PM_NOREMOVE);
        HHOOK hook = SetWindowsHookExW(WH_MOUSE_LL, mouseButtonHook,
            GetModuleHandleW(nullptr), 0);
        if (!hook) {
            hookButtonState = nullptr;
            ready.set_value(0);
            return;
        }
        leftButtonDown_.store((GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0);
        ready.set_value(GetCurrentThreadId());
        while (GetMessageW(&message, nullptr, 0, 0) > 0) {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        UnhookWindowsHookEx(hook);
        hookButtonState = nullptr;
        leftButtonDown_.store(false);
    });
    inputThreadId_ = result.get();
    if (!inputThreadId_) {
        inputThread_.join();
        throw std::runtime_error("Failed to install low-level mouse hook");
    }

}

MouseController::~MouseController() {
    if (inputThreadId_) PostThreadMessageW(inputThreadId_, WM_QUIT, 0, 0);
    if (inputThread_.joinable()) inputThread_.join();
}

void MouseController::update(std::vector<Box> const& allBoxes,
                             LetterboxInfo const& letterbox, uint64_t sequence, float keypointThreshold) {
    std::vector<AimTarget> boxes;
    boxes.reserve(allBoxes.size());
    // Keep each enemy's best eligible point, including lower tiers for a locked enemy.
    auto excludedPoint = [&](float x, float y) {
        for (auto const& region : allBoxes) {
            if (region.classId != 1 && region.classId != 2 && region.classId != 3) continue;
            if (x >= region.x1 && x <= region.x2 && y >= region.y1 && y <= region.y2)
                return true;
        }
        return false;
    };
    for (auto const& box : allBoxes) {
        if (!validEnemy(box)) continue;
        bool found = false;
        for (int keypoint = 0; keypoint < 2; ++keypoint) {
            auto const* point = box.keypoints + keypoint * 3;
            if (!std::isfinite(point[0]) || !std::isfinite(point[1]) ||
                !std::isfinite(point[2]) || point[2] <= 0 || point[2] < keypointThreshold ||
                point[0] < box.x1 || point[0] > box.x2 || point[1] < box.y1 || point[1] > box.y2 ||
                excludedPoint(point[0], point[1])) continue;
            boxes.push_back(AimTarget{box, point[0], point[1], keypoint});
            found = true;
            break;
        }
        if (!found) {
            float x = (box.x1 + box.x2) * 0.5f;
            float y = box.y1 + (box.y2 - box.y1) * 0.25f;
            if (!excludedPoint(x, y)) boxes.push_back(AimTarget{box, x, y, 2});
        }
    }
    lastInputError_ = 0;
    lastRelativeDx_ = 0;
    lastRelativeDy_ = 0;
    // Latch physical presses even when down and up both occur between updates.
    bool leftHeld = leftButtonDown_.load();
    uint64_t leftDownEvents = leftDownEvents_.load();
    bool leftRequested = leftHeld || leftDownEvents != handledLeftDownEvents_;
    bool useScreenCenterAim = aimFromScreenCenter(config_);
    POINT cursor{};
    // Relative camera control uses the display center, not the desktop cursor.
    bool hasCursor = useScreenCenterAim || GetCursorPos(&cursor) != FALSE;
    POINT screenCenter{
        letterbox.screenW / 2,
        letterbox.screenH / 2,
    };
    POINT screenCenterGlobal{
        letterbox.desktopX + screenCenter.x,
        letterbox.desktopY + screenCenter.y,
    };
    POINT aimReferenceGlobal = useScreenCenterAim ? screenCenterGlobal : cursor;
    POINT aimReferenceLocal{
        aimReferenceGlobal.x - letterbox.desktopX,
        aimReferenceGlobal.y - letterbox.desktopY,
    };
    if (!config_.enabled || !leftRequested || !hasCursor) {
        // Do not carry a click without a valid target into a later detection.
        handledLeftDownEvents_ = leftDownEvents;
        hasLockedTarget_ = false;
        hasPredictedTarget_ = false;
        targetLostSince_ = {};
        lockPressEvents_ = leftDownEvents;
        lastTargetUpdate_ = {};
        logState(!config_.enabled ? "disabled" : !leftRequested ? "waiting-left-button" :
                 !hasCursor ? "cursor-unavailable" : "no-targets", sequence, boxes.size());
        return;
    }
    if (config_.lockTarget && leftDownEvents != lockPressEvents_) {
        // A new physical press explicitly requests a fresh nearest-target acquisition.
        lockPressEvents_ = leftDownEvents;
        hasLockedTarget_ = false;
        hasPredictedTarget_ = false;
        targetLostSince_ = {};
        lastTargetUpdate_ = {};
    }
    auto selectionTime = std::chrono::steady_clock::now();
    int feedbackMs = useScreenCenterAim
        ? std::max(config_.moveIntervalMs, config_.relativeFeedbackMs) : config_.moveIntervalMs;
    if (config_.lockTarget && hasPredictedTarget_ &&
        selectionTime-lastMoveTime_ < std::chrono::milliseconds(feedbackMs)) return;

    int selectedTargetIndex = -1;
    bool retained = false;
    if (config_.lockTarget && hasLockedTarget_) {
        selectedTargetIndex = associatedTargetIndex(boxes, hasPredictedTarget_ ? predictedTarget_ : lockedTarget_);
        retained = selectedTargetIndex >= 0;
        if (!retained) {
            if (targetLostSince_ == std::chrono::steady_clock::time_point{}) targetLostSince_ = selectionTime;
            if (selectionTime-targetLostSince_ < std::chrono::milliseconds(config_.targetLostGraceMs)) {
                logState("locked-target-missing", sequence, boxes.size());
                return; // Never move toward a stale coordinate or another enemy during a brief miss.
            }
            hasLockedTarget_ = false;
            hasPredictedTarget_ = false;
            lastTargetUpdate_ = {};
        }
    }
    if (selectedTargetIndex < 0) selectedTargetIndex = nearestTargetIndexToPoint(boxes, aimReferenceLocal);
    if (selectedTargetIndex < 0) {
        handledLeftDownEvents_ = leftDownEvents;
        hasLockedTarget_ = false;
        hasPredictedTarget_ = false;
        lastTargetUpdate_ = {};
        logState("no-targets", sequence, boxes.size());
        return;
    }
    targetLostSince_ = {};
    hasPredictedTarget_ = false;
    auto const& selectedTarget = boxes[static_cast<size_t>(selectedTargetIndex)];
    if (config_.lockTarget && !retained) ++targetLockId_;
    if (!hasLockedTarget_ || lockedTargetPriority_ != selectedTarget.priority ||
        !sameTarget(selectedTarget.source, lockedTarget_)) {
        lastTargetUpdate_ = {};
    }
    lockedTarget_ = selectedTarget.source;
    lockedTargetPriority_ = selectedTarget.priority;
    hasLockedTarget_ = true;
    if (selectedTargetIndex >= 0) {
        MouseMovePlan selectedMovePlan = planMouseMove(
            boxes[static_cast<size_t>(selectedTargetIndex)],
            letterbox,
            aimReferenceGlobal);
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - lastTargetUpdate_).count();
        // Absolute positioning uses this frame's exact desktop coordinate, without filtering.
        if (config_.mode == "absolute" || (useScreenCenterAim && config_.relativeDirect) ||
            lastTargetUpdate_ == std::chrono::steady_clock::time_point{} ||
            dt > 0.1 || config_.targetSmoothingMs == 0) {
            filteredTargetX_ = selectedMovePlan.targetGlobal.x;
            filteredTargetY_ = selectedMovePlan.targetGlobal.y;
        } else {
            double alpha = -std::expm1(-dt / (config_.targetSmoothingMs * 0.001));
            filteredTargetX_ += alpha * (selectedMovePlan.targetGlobal.x - filteredTargetX_);
            filteredTargetY_ += alpha * (selectedMovePlan.targetGlobal.y - filteredTargetY_);
        }
        lastTargetUpdate_ = now;
        double errorX = filteredTargetX_ - aimReferenceGlobal.x;
        double errorY = filteredTargetY_ - aimReferenceGlobal.y;
        if (useScreenCenterAim) {
            errorX = currentRelativeError(errorX, selectedMovePlan.errorX, config_.relativeDeadzonePx);
            errorY = currentRelativeError(errorY, selectedMovePlan.errorY, config_.relativeDeadzonePx);
        }
        selectedMovePlan.delta.x = static_cast<LONG>(std::lround(errorX));
        selectedMovePlan.delta.y = static_cast<LONG>(std::lround(errorY));
        int intervalMs = useScreenCenterAim
            ? std::max(config_.moveIntervalMs, config_.relativeFeedbackMs) : config_.moveIntervalMs;
        if (lastMoveTime_ != std::chrono::steady_clock::time_point{} &&
            now - lastMoveTime_ < std::chrono::milliseconds(intervalMs)) {
            return;
        }
        // Consume only after the interval check, so throttling cannot lose a tap.
        handledLeftDownEvents_ = leftDownEvents;
        POINT requestedPosition{
            aimReferenceGlobal.x + selectedMovePlan.delta.x,
            aimReferenceGlobal.y + selectedMovePlan.delta.y,
        };
        std::osyncstream(std::cout)
            << "[mouse-move] frame=" << sequence << " mode=" << config_.mode
            << " lock_id=" << targetLockId_ << " lock_retained=" << retained
            << " target_box=(" << selectedTarget.source.x1 << ',' << selectedTarget.source.y1 << ','
            << selectedTarget.source.x2 << ',' << selectedTarget.source.y2 << ')'
            << " point=" << (selectedTarget.priority == 0 ? "keypoint1" :
                            selectedTarget.priority == 1 ? "keypoint2" : "box-upper-center")
            << " target_screen=(" << selectedTarget.x << ',' << selectedTarget.y << ')'
            << " target_desktop=(" << selectedMovePlan.targetGlobal.x << ',' << selectedMovePlan.targetGlobal.y << ')'
            << " requested_desktop=(" << requestedPosition.x << ',' << requestedPosition.y << ')'
            << " reference=(" << aimReferenceGlobal.x << ',' << aimReferenceGlobal.y << ')'
            << " delta=(" << selectedMovePlan.delta.x << ',' << selectedMovePlan.delta.y << ')'
            << " screen=(" << letterbox.screenW << ',' << letterbox.screenH << ')'
            << " desktop_origin=(" << letterbox.desktopX << ',' << letterbox.desktopY << ')'
            << " roi=(" << letterbox.captureX << ',' << letterbox.captureY << ','
            << letterbox.captureW << ',' << letterbox.captureH << ")\n";
        bool moveSucceeded =
            submitMove(selectedMovePlan.delta.x, selectedMovePlan.delta.y,
                       aimReferenceGlobal.x, aimReferenceGlobal.y, useScreenCenterAim);
        POINT cursorAfter{};
        bool hasCursorAfter = !useScreenCenterAim && GetCursorPos(&cursorAfter) != FALSE;
        std::osyncstream(std::cout)
            << "[mouse-move-result] frame=" << sequence
            << " submitted=" << (moveSucceeded && (selectedMovePlan.delta.x != 0 || selectedMovePlan.delta.y != 0))
            << " skipped_zero=" << (selectedMovePlan.delta.x == 0 && selectedMovePlan.delta.y == 0)
            << " relative_input=(" << lastRelativeDx_ << ',' << lastRelativeDy_ << ')'
            << " win32_error=" << lastInputError_ << " cursor_valid=" << hasCursorAfter
            << " cursor_after=(" << cursorAfter.x << ',' << cursorAfter.y << ")\n";
        if (moveSucceeded && (selectedMovePlan.delta.x != 0 || selectedMovePlan.delta.y != 0))
            lastMoveTime_ = std::chrono::steady_clock::now();
        if (moveSucceeded && useScreenCenterAim && config_.lockTarget &&
            (lastRelativeDx_ != 0 || lastRelativeDy_ != 0)) {
            predictedTarget_ = lockedTarget_;
            float cameraX = lastRelativeDx_ / (config_.relativeScale * config_.relativeScaleX);
            float cameraY = lastRelativeDy_ / (config_.relativeScale * config_.relativeScaleY);
            predictedTarget_.x1 -= cameraX; predictedTarget_.x2 -= cameraX;
            predictedTarget_.y1 -= cameraY; predictedTarget_.y2 -= cameraY;
            hasPredictedTarget_ = true;
        }
        bool noMovement = selectedMovePlan.delta.x == 0 && selectedMovePlan.delta.y == 0;
        logState(noMovement ? "centered" : moveSucceeded ? "input-submitted" : "input-failed",
                 sequence, boxes.size(), selectedMovePlan.delta.x, selectedMovePlan.delta.y,
                 !noMovement);
    }
}

bool MouseController::moveBy(int dx, int dy, int referenceX, int referenceY) {
    return submitMove(dx, dy, referenceX, referenceY, false);
}

bool MouseController::submitMove(int dx, int dy, int referenceX, int referenceY, bool aimCorrection) {
    MouseMovePlan plan{};
    plan.delta = POINT{dx, dy};
    bool attempted = config_.enabled && (dx != 0 || dy != 0);
    if (attempted) {
        ++moveAttempts_;
    }
    POINT relativeDelta{};
    SetLastError(ERROR_SUCCESS);
    bool succeeded = applyMouseMovePlan(plan, POINT{referenceX, referenceY}, config_, relativeDelta, aimCorrection);
    lastInputError_ = succeeded ? ERROR_SUCCESS : GetLastError();
    lastRelativeDx_ = relativeDelta.x;
    lastRelativeDy_ = relativeDelta.y;
    if (attempted && succeeded) {
        ++movesSubmitted_;
    }
    return succeeded;
}

void MouseController::logState(char const* state, uint64_t sequence, std::size_t boxCount,
                               int dx, int dy, bool force) {
    auto now = std::chrono::steady_clock::now();
    if (now < nextDiagnosticLog_) return;
    nextDiagnosticLog_ = now + std::chrono::seconds(1);
    std::cout << "[mouse] mode=" << config_.mode << " state=" << state
              << " frame=" << sequence << " targets=" << boxCount
              << " delta=(" << dx << ',' << dy << ") attempts=" << moveAttempts_
              << " submitted=" << movesSubmitted_ << " win32_error=" << lastInputError_ << '\n';
}
