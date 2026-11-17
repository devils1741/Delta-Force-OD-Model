#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "MouseController.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <future>
#include <stdexcept>
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

bool sendRelativeMouseMove(POINT const& delta, MouseConfig const& mouse, POINT& relativeDelta) {
    relativeDelta = scaledRelativeDelta(delta, mouse);
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dx = relativeDelta.x;
    input.mi.dy = relativeDelta.y;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

bool applyMouseMovePlan(
    MouseMovePlan const& plan,
    POINT const& referencePoint,
    MouseConfig const& mouse,
    POINT& relativeDelta) {
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
        return sendRelativeMouseMove(plan.delta, mouse, relativeDelta);
    }
    return sendRelativeMouseMove(plan.delta, mouse, relativeDelta);
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
    std::cout << "[mouse-input] listener=low-level-hook ready=1\n";
}

MouseController::~MouseController() {
    if (inputThreadId_) PostThreadMessageW(inputThreadId_, WM_QUIT, 0, 0);
    if (inputThread_.joinable()) inputThread_.join();
}

void MouseController::update(std::vector<Box> const& boxes,
                             LetterboxInfo const& letterbox, uint64_t sequence) {
    lastInputError_ = 0;
    lastRelativeDx_ = 0;
    lastRelativeDy_ = 0;
    bool leftRequested = leftButtonDown_.load();
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
    if (!config_.enabled || !leftRequested || !hasCursor || boxes.empty()) {
        hasLockedTarget_ = false;
        logState(!config_.enabled ? "disabled" : !leftRequested ? "waiting-left-button" :
                 !hasCursor ? "cursor-unavailable" : "no-targets", sequence, boxes.size());
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
    if (selectedTargetIndex >= 0) {
        MouseMovePlan selectedMovePlan = planMouseMove(
            boxes[static_cast<size_t>(selectedTargetIndex)],
            letterbox,
            aimReferenceGlobal);
        bool moveSucceeded =
            moveBy(selectedMovePlan.delta.x, selectedMovePlan.delta.y,
                   aimReferenceGlobal.x, aimReferenceGlobal.y);
        bool noMovement = selectedMovePlan.delta.x == 0 && selectedMovePlan.delta.y == 0;
        logState(noMovement ? "centered" : moveSucceeded ? "input-submitted" : "input-failed",
                 sequence, boxes.size(), selectedMovePlan.delta.x, selectedMovePlan.delta.y,
                 !noMovement);
    }
}

bool MouseController::moveBy(int dx, int dy, int referenceX, int referenceY) {
    MouseMovePlan plan{};
    plan.delta = POINT{dx, dy};
    bool attempted = config_.enabled && (dx != 0 || dy != 0);
    if (attempted) {
        ++moveAttempts_;
    }
    POINT relativeDelta{};
    SetLastError(ERROR_SUCCESS);
    bool succeeded = applyMouseMovePlan(plan, POINT{referenceX, referenceY}, config_, relativeDelta);
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
    if (!force && now < nextDiagnosticLog_) {
        return;
    }
    nextDiagnosticLog_ = now + std::chrono::seconds(2);
    DWORD foregroundPid = 0;
    GetWindowThreadProcessId(GetForegroundWindow(), &foregroundPid);
    std::cout << "[mouse] state=" << state << " frame=" << sequence
              << " elapsed_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(now - diagnosticsStart_).count()
              << " boxes=" << boxCount << " mode=" << config_.mode
              << " left_down=" << leftButtonDown_.load()
              << " hook_events=" << rawMouseEvents_.load()
              << " injected_events=" << injectedMouseEvents_.load()
              << " injected_down=" << injectedDownEvents_.load()
              << " injected_up=" << injectedUpEvents_.load()
              << " down_events=" << leftDownEvents_.load()
              << " up_events=" << leftUpEvents_.load()
              << " target_class=" << (hasLockedTarget_ ? lockedTarget_.classId : -1)
              << " target_score=" << (hasLockedTarget_ ? lockedTarget_.score : 0.0f)
              << " planned_dx=" << dx << " planned_dy=" << dy
              << " relative_dx=" << lastRelativeDx_ << " relative_dy=" << lastRelativeDy_
              << " relative_scale=" << config_.relativeScale
              << " attempts=" << moveAttempts_ << " submitted=" << movesSubmitted_
              << " win32_error=" << lastInputError_
              << " foreground_pid=" << foregroundPid << '\n';
}
