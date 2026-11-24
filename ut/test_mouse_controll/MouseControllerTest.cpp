// Regression tests execute the real controller with fake OS input and cursor APIs.
// No physical hooks are installed and no mouse movement leaves this process.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include "AppConfig.h"
#include "Detection.h"
#include <cstdint>
#include <chrono>
#include <cstddef>
#include <vector>
#include <atomic>
#include <thread>
#include <iostream>
#include <stdexcept>
#include <cmath>
// Test-only access makes interval/smoothing checks independent of scheduler timing.
#define private public
#include "MouseController.h"
#undef private

HHOOK WINAPI fakeSetWindowsHookExW(int, HOOKPROC, HINSTANCE, DWORD);
BOOL WINAPI fakeUnhookWindowsHookEx(HHOOK);
SHORT WINAPI fakeGetAsyncKeyState(int);
BOOL WINAPI fakeGetCursorPos(LPPOINT);
BOOL WINAPI fakeSetCursorPos(int, int);
UINT WINAPI fakeSendInput(UINT, LPINPUT, int);
int WINAPI fakeGetSystemMetrics(int);
LRESULT WINAPI fakeCallNextHookEx(HHOOK, int, WPARAM, LPARAM);
#define SetWindowsHookExW fakeSetWindowsHookExW
#define UnhookWindowsHookEx fakeUnhookWindowsHookEx
#define GetAsyncKeyState fakeGetAsyncKeyState
#define GetCursorPos fakeGetCursorPos
#define SetCursorPos fakeSetCursorPos
#define SendInput fakeSendInput
#define GetSystemMetrics fakeGetSystemMetrics
#define CallNextHookEx fakeCallNextHookEx
#include "../../src/source/MouseController.cpp"
#undef SetWindowsHookExW
#undef UnhookWindowsHookEx
#undef GetAsyncKeyState
#undef GetCursorPos
#undef SetCursorPos
#undef SendInput
#undef GetSystemMetrics
#undef CallNextHookEx

namespace {
struct FakeMove { bool relative; LONG x; LONG y; };
std::atomic<void*> capturedState{};
std::vector<FakeMove> moves;
POINT fakeCursor{500, 500};
bool cursorAvailable = true;
int cursorReads = 0;

void require(bool condition, char const* message) {
    if (!condition) throw std::runtime_error(message);
}

MouseConfig testConfig(char const* mode = "absolute") {
    MouseConfig config;
    config.mode = mode;
    config.moveIntervalMs = 0;
    config.targetSmoothingMs = 0;
    config.relativeScale = 1.0f;
    config.relativeDirect = false;
    config.lockTarget = false;
    config.relativeGain = 0.2f;
    config.relativeMaxStep = 12;
    config.relativeDeadzonePx = 2.0f;
    config.relativeFeedbackMs = 0;
    return config;
}

Box liveBox(float x1 = 450, float y1 = 300, float x2 = 750, float y2 = 700) {
    return Box{x1, y1, x2, y2, 0.9f, 0, {}};
}
void keypoint(Box& box, int index, float x, float y, float confidence = 0.9f) {
    box.keypoints[index * 3] = x;
    box.keypoints[index * 3 + 1] = y;
    box.keypoints[index * 3 + 2] = confidence;
}

struct Fixture {
    MouseController controller;
    LetterboxInfo letterbox{};
    uint64_t sequence = 1;

    explicit Fixture(MouseConfig const& config = testConfig()) : controller(config) {
        moves.clear();
        fakeCursor = POINT{500, 500};
        cursorAvailable = true;
        cursorReads = 0;
        hookButtonState = static_cast<ButtonEventState*>(capturedState.load());
        require(hookButtonState != nullptr, "fake hook installation failed");
        letterbox.screenW = 1000;
        letterbox.screenH = 1000;
    }
    ~Fixture() { hookButtonState = nullptr; }
    void event(WPARAM message, bool injected = false) {
        MSLLHOOKSTRUCT input{};
        input.flags = injected ? LLMHF_INJECTED : 0;
        mouseButtonHook(HC_ACTION, message, reinterpret_cast<LPARAM>(&input));
    }
    void tap(bool injected = false) {
        event(WM_LBUTTONDOWN, injected);
        event(WM_LBUTTONUP, injected);
    }
    void update(std::vector<Box> const& boxes, float threshold = 0.3f) {
        controller.update(boxes, letterbox, sequence++, threshold);
    }
    void ageLastMove() {
        controller.lastMoveTime_ = std::chrono::steady_clock::now() - std::chrono::seconds(3);
    }
};

void expectAbsolute(LONG x, LONG y) {
    require(!moves.empty(), "expected an absolute movement");
    auto const& move = moves.back();
    require(!move.relative && move.x == x && move.y == y, "wrong absolute target");
}
void expectRelative(LONG x, LONG y) {
    require(!moves.empty(), "expected a relative movement");
    auto const& move = moves.back();
    require(move.relative && move.x == x && move.y == y, "wrong relative correction");
}

void checkClickAndHold() {
    Fixture f;
    std::vector<Box> targets{liveBox(600, 600, 640, 680)};
    f.update(targets);
    require(moves.empty(), "idle moved without input");
    f.tap(); f.update(targets);
    require(moves.size() == 1, "quick tap between frames did not move once");
    f.update(targets);
    require(moves.size() == 1, "quick tap repeated while idle");
    f.event(WM_LBUTTONDOWN); f.update(targets); f.update(targets); f.update(targets);
    require(moves.size() == 4, "held input did not continue moving");
    f.event(WM_LBUTTONUP); f.update(targets);
    require(moves.size() == 4, "release caused an extra movement");

    f.controller.config_.moveIntervalMs = 1000;
    f.controller.lastMoveTime_ = std::chrono::steady_clock::now();
    f.tap(); f.update(targets);
    require(moves.size() == 4, "movement interval was ignored");
    f.ageLastMove(); f.update(targets);
    require(moves.size() == 5, "rate limiting lost a quick tap");
    f.update(targets);
    require(moves.size() == 5, "pending tap repeated after consumption");
}

void checkClearedAndInjectedClicks() {
    Fixture f;
    std::vector<Box> targets{liveBox(600, 600, 640, 680)};
    f.tap(); f.update({}); f.update(targets);
    require(moves.empty(), "no-target tap was carried into a later detection");
    f.tap(true); f.update(targets);
    require(moves.empty(), "injected click triggered movement");
    f.controller.config_.enabled = false;
    f.tap(); f.update(targets);
    f.controller.config_.enabled = true; f.update(targets);
    require(moves.empty(), "disabled controller retained a stale tap");
    cursorAvailable = false;
    f.tap(); f.update(targets);
    cursorAvailable = true; f.update(targets);
    require(moves.empty(), "missing-cursor tap was retained");
}

void checkTargetPriorities() {
    Fixture f;
    Box farEnemy = liveBox(490, 480, 810, 1040);
    Box nearEnemy = liveBox(495, 495, 585, 675);
    keypoint(farEnemy, 0, 650, 620); keypoint(farEnemy, 1, 501, 500);
    keypoint(nearEnemy, 0, 540, 530); keypoint(nearEnemy, 1, 502, 500);
    f.event(WM_LBUTTONDOWN);
    f.update({farEnemy, nearEnemy}); expectAbsolute(540, 530);
    // First-point tier wins even when a second point is closer to the reference.
    farEnemy.keypoints[2] = nearEnemy.keypoints[2] = 0.2f;
    f.update({farEnemy, nearEnemy}); expectAbsolute(501, 500);
    farEnemy.keypoints[5] = nearEnemy.keypoints[5] = 0;
    f.update({farEnemy, nearEnemy}); expectAbsolute(540, 540);

    keypoint(farEnemy, 0, 650, 620); keypoint(nearEnemy, 0, 540, 530);
    Box excluded = liveBox(530, 520, 550, 540);
    keypoint(excluded, 0, 540, 530);
    for (int excludedClass : {1, 2, 3}) {
        excluded.classId = excludedClass;
        f.update({farEnemy, nearEnemy, excluded}); expectAbsolute(650, 620);
    }
    size_t before = moves.size();
    Box teammate = excluded; teammate.classId = 2;
    Box corpse = excluded; corpse.classId = 1;
    f.update({teammate, corpse});
    require(moves.size() == before, "teammate or corpse became an aim target");
}

void checkBoundedRelativeControl() {
    MouseConfig config = testConfig("relative");
    config.relativeScale = 2.0f;
    Fixture f(config);
    Box target = liveBox();
    keypoint(target, 0, 600, 400);
    f.event(WM_LBUTTONDOWN); f.update({target}); expectRelative(12, -12);
    keypoint(target, 0, 510, 490);
    f.update({target}); expectRelative(4, -4);
    size_t before = moves.size();
    keypoint(target, 0, 501, 499);
    f.update({target});
    require(moves.size() == before, "deadzone emitted relative movement");
    require(cursorReads == 0, "relative aiming consulted the desktop cursor");
    for (auto const& move : moves) {
        require(move.relative && std::abs(move.x) <= 12 && std::abs(move.y) <= 12,
            "automatic relative correction exceeded its per-axis step bound");
    }
}

void checkFeedbackIntervals() {
    MouseConfig config = testConfig("relative");
    config.relativeFeedbackMs = 1000;
    Fixture f(config);
    Box target = liveBox(); keypoint(target, 0, 600, 500);
    f.tap(); f.update({target});
    require(moves.size() == 1, "first relative tap was lost");
    f.tap(); f.update({target});
    require(moves.size() == 1, "relative feedback interval was ignored");
    f.ageLastMove(); f.update({target});
    require(moves.size() == 2, "feedback interval lost a pending tap");

    f.controller.config_.moveIntervalMs = 1000;
    f.controller.config_.relativeFeedbackMs = 20;
    f.controller.lastMoveTime_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(100);
    f.tap(); f.update({target});
    require(moves.size() == 2, "feedback interval bypassed longer move interval");
    f.ageLastMove(); f.update({target});
    require(moves.size() == 3, "tap did not resume after both intervals");
}

void checkSmoothingAndCurrentError() {
    MouseConfig config = testConfig("relative");
    config.targetSmoothingMs = 1000;
    Fixture f(config);
    Box target = liveBox(); keypoint(target, 0, 550, 500);
    f.event(WM_LBUTTONDOWN); f.update({target}); expectRelative(10, 0);
    keypoint(target, 0, 560, 500);
    f.controller.lastTargetUpdate_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(20);
    f.update({target});
    require(f.controller.filteredTargetX_ > 550 && f.controller.filteredTargetX_ < 555,
        "same-source target smoothing reset between frames");

    // A stale filtered target must not command a larger or wrong-way correction.
    keypoint(target, 0, 510, 500);
    f.controller.lastTargetUpdate_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(20);
    f.update({target});
    require(moves.back().relative && moves.back().x > 0 && moves.back().x <= 2,
        "stale smoothing exceeded the latest remaining error");
    keypoint(target, 0, 480, 500);
    f.controller.lastTargetUpdate_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(20);
    f.update({target});
    require(moves.back().relative && moves.back().x < 0 && moves.back().x >= -4,
        "smoothing commanded the wrong direction after crossing the target");
    size_t before = moves.size();
    keypoint(target, 0, 501, 500);
    f.controller.lastTargetUpdate_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(20);
    f.update({target});
    require(moves.size() == before, "stale filtered error defeated the latest raw deadzone");

    Box other = liveBox(100, 300, 350, 700); keypoint(other, 0, 200, 500);
    f.update({other});
    require(f.controller.filteredTargetX_ == 200, "target change retained previous target smoothing");
    other.keypoints[2] = 0; keypoint(other, 1, 250, 500);
    f.update({other});
    require(f.controller.filteredTargetX_ == 250, "keypoint tier change retained stale smoothing");
}

void checkRoiAndMonitorCoordinates() {
    Fixture f(testConfig("relative"));
    f.letterbox.screenW = 3840; f.letterbox.screenH = 2160;
    f.letterbox.captureX = 1280; f.letterbox.captureY = 440;
    f.letterbox.captureW = f.letterbox.captureH = 1280;
    f.letterbox.inputW = f.letterbox.inputH = 640;
    f.letterbox.scale = 0.5f;
    Box target = liveBox(1800, 900, 2100, 1400);
    keypoint(target, 0, 1920, 1080);
    f.event(WM_LBUTTONDOWN); f.update({target});
    require(moves.empty(), "center of 4K monitor was confused with ROI or model center");
    keypoint(target, 0, 2020, 1080);
    f.update({target}); expectRelative(12, 0);
    f.letterbox.desktopX = 3840; f.letterbox.desktopY = -400;
    f.update({target}); expectRelative(12, 0);
    require(cursorReads == 0, "offset monitor relative control used desktop cursor");
}

void checkDirectMoveByCompatibility() {
    Fixture f(testConfig("relative"));
    require(f.controller.moveBy(100, -50, 500, 500), "direct moveBy failed");
    expectRelative(100, -50);
}

void checkPerAxisCalibration() {
    auto config = testConfig("relative");
    config.relativeScaleX = 1.40f;
    config.relativeScaleY = 1.70f;
    Fixture f(config);
    auto target = liveBox();
    keypoint(target, 0, 510, 520);
    f.tap();
    f.update({target});
    expectRelative(3, 7);
    require(cursorReads == 0, "relative calibration used the hidden desktop cursor");
}

void checkFullCalibratedMovement() {
    auto config = testConfig("relative");
    config.relativeDirect = true;
    config.relativeScaleX = 1.40f;
    config.relativeScaleY = 1.70f;
    config.relativeFeedbackMs = 50;
    config.targetSmoothingMs = 1000;
    Fixture f(config);
    auto target = liveBox(350, 300, 750, 700);
    keypoint(target, 0, 600, 600);
    f.tap(); f.update({target}); expectRelative(140, 170);
    require(moves.size() == 1, "direct tap emitted multiple moves");
    f.update({target});
    require(moves.size() == 1, "released direct tap repeated");
    f.tap(); f.update({target});
    require(moves.size() == 1, "direct mode bypassed feedback wait");
    f.ageLastMove();
    keypoint(target, 0, 450, 480);
    f.update({target}); expectRelative(-70, -34);
}

void checkStableMultiTargetLock() {
    auto config = testConfig("relative");
    config.relativeDirect = true;
    config.lockTarget = true;
    Fixture f(config);
    auto a = liveBox(490, 400, 530, 600);
    auto b = liveBox(520, 400, 560, 600);
    keypoint(a, 0, 500, 500); keypoint(b, 0, 540, 500);
    f.event(WM_LBUTTONDOWN); f.update({a, b});
    require(f.controller.lockedTarget_.x1 == a.x1, "initial lock did not choose nearest");
    uint64_t id = f.controller.targetLockId_;
    b = liveBox(470, 400, 510, 600);
    keypoint(a, 0, 505, 500); keypoint(b, 0, 501, 500);
    f.update({b, a}); expectRelative(5, 0);
    require(f.controller.targetLockId_ == id && f.controller.lockedTarget_.x1 == a.x1,
        "closer neighbor or reordered detections stole target lock");
    a.keypoints[2] = 0;
    keypoint(a, 1, 505, 501);
    f.update({b, a}); expectRelative(5, 0);
    require(f.controller.targetLockId_ == id && f.controller.lockedTargetPriority_ == 1,
        "missing first point switched enemy instead of using its second point");
    f.event(WM_LBUTTONUP); f.update({a, b});
    f.tap(); f.update({a, b});
    require(f.controller.targetLockId_ != id && f.controller.lockedTarget_.x1 == b.x1,
        "new press could not choose a new nearest enemy");
}

void checkLostTargetGrace() {
    auto config = testConfig("relative");
    config.relativeDirect = true;
    config.lockTarget = true;
    config.targetLostGraceMs = 150;
    Fixture f(config);
    auto a = liveBox(480, 400, 520, 600);
    auto b = liveBox(800, 400, 940, 600);
    keypoint(a, 0, 500, 500); keypoint(b, 0, 900, 500);
    f.event(WM_LBUTTONDOWN); f.update({a, b});
    auto id = f.controller.targetLockId_;
    f.update({b}); f.update({});
    require(moves.empty() && f.controller.hasLockedTarget_, "brief miss moved or discarded target lock");
    f.update({b, a});
    require(f.controller.targetLockId_ == id, "reappearing target did not resume its lock");
    a.classId = 1;
    f.update({a, b});
    require(moves.empty(), "dead target caused an immediate jump to another enemy");
    f.controller.targetLostSince_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(200);
    f.update({a, b}); expectRelative(400, 0);
    require(f.controller.targetLockId_ != id, "expired grace did not allow reacquisition");
}

void checkCameraMotionAssociation() {
    auto config = testConfig("relative");
    config.relativeDirect = true;
    config.lockTarget = true;
    config.relativeFeedbackMs = 1000;
    Fixture f(config);
    auto a = liveBox(650, 450, 690, 550);
    auto b = liveBox(250, 350, 290, 450);
    keypoint(a, 0, 670, 500); keypoint(b, 0, 270, 400);
    f.event(WM_LBUTTONDOWN); f.update({a, b}); expectRelative(170, 0);
    auto id = f.controller.targetLockId_;
    f.update({b, a});
    require(moves.size() == 1 && f.controller.targetLockId_ == id,
        "pre-feedback frame changed lock or resent an old error");
    a = liveBox(480, 450, 520, 550); keypoint(a, 0, 500, 500);
    b = liveBox(80, 350, 120, 450); keypoint(b, 0, 100, 400);
    f.ageLastMove(); f.update({b, a});
    require(f.controller.targetLockId_ == id && f.controller.lockedTarget_.x1 == 480,
        "camera translation was mistaken for a different target");
    require(moves.size() == 1, "centered tracked target generated another movement");
    auto duplicate = a; keypoint(duplicate, 0, 510, 500);
    f.update({a, duplicate});
    require(moves.size() == 1 && f.controller.targetLockId_ == id,
        "ambiguous overlapping targets caused a random selection");
}

void checkExactAbsoluteCoordinates() {
    auto config = testConfig("absolute");
    config.targetSmoothingMs = 1000;
    config.relativeScale = 0.01f;
    Fixture f(config);
    f.letterbox.screenW = 3840; f.letterbox.screenH = 2160;
    f.letterbox.captureX = 1280; f.letterbox.captureY = 440;
    f.letterbox.captureW = f.letterbox.captureH = 1280;
    f.letterbox.desktopX = -3840; f.letterbox.desktopY = 200;
    auto target = liveBox(1800, 900, 2100, 1300);
    keypoint(target, 0, 1920, 1080);
    f.tap();
    f.update({target}); expectAbsolute(-1920, 1280);
    f.update({target});
    require(moves.size() == 1, "absolute tap repeated without a new press");
    f.event(WM_LBUTTONDOWN);
    keypoint(target, 0, 2020, 1100);
    f.update({target}); expectAbsolute(-1820, 1300);
    require(moves.size() == 2, "absolute movement was throttled by relative feedback");
    fakeCursor = POINT{-1820, 1300};
    f.update({target});
    require(moves.size() == 2, "stationary absolute target sent another movement event");
}
} // namespace

HHOOK WINAPI fakeSetWindowsHookExW(int, HOOKPROC, HINSTANCE, DWORD) {
    capturedState.store(hookButtonState);
    return reinterpret_cast<HHOOK>(1);
}
BOOL WINAPI fakeUnhookWindowsHookEx(HHOOK) { return TRUE; }
SHORT WINAPI fakeGetAsyncKeyState(int) { return 0; }
BOOL WINAPI fakeGetCursorPos(LPPOINT point) {
    ++cursorReads; *point = fakeCursor; return cursorAvailable ? TRUE : FALSE;
}
BOOL WINAPI fakeSetCursorPos(int x, int y) {
    moves.push_back(FakeMove{false, x, y}); return TRUE;
}
UINT WINAPI fakeSendInput(UINT count, LPINPUT input, int) {
    for (UINT i = 0; i < count; ++i) {
        require(input[i].type == INPUT_MOUSE,
            "controller sent an unexpected input event");
        if (input[i].mi.dwFlags & MOUSEEVENTF_ABSOLUTE) {
            require(input[i].mi.dwFlags == (MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK),
                "absolute event is missing virtual-desktop flags");
            LONG x = static_cast<LONG>(static_cast<int64_t>(input[i].mi.dx) * 7680 / 65536 - 3840);
            LONG y = static_cast<LONG>(static_cast<int64_t>(input[i].mi.dy) * 2560 / 65536 - 400);
            moves.push_back(FakeMove{false, x, y});
        } else {
            require(input[i].mi.dwFlags == MOUSEEVENTF_MOVE, "unexpected relative flags");
            moves.push_back(FakeMove{true, input[i].mi.dx, input[i].mi.dy});
        }
    }
    return count;
}
int WINAPI fakeGetSystemMetrics(int index) {
    switch (index) {
    case SM_XVIRTUALSCREEN: return -3840;
    case SM_YVIRTUALSCREEN: return -400;
    case SM_CXVIRTUALSCREEN: return 7680;
    case SM_CYVIRTUALSCREEN: return 2560;
    default: throw std::runtime_error("unexpected screen metric");
    }
}
LRESULT WINAPI fakeCallNextHookEx(HHOOK, int, WPARAM, LPARAM) { return 0; }

int main() {
    try {
        struct Test { char const* name; void (*run)(); };
        Test const tests[] = {
            {"click, hold, release, and throttling", checkClickAndHold},
            {"stale and injected clicks", checkClearedAndInjectedClicks},
            {"target priority and excluded regions", checkTargetPriorities},
            {"bounded relative corrections and deadzone", checkBoundedRelativeControl},
            {"feedback and movement intervals", checkFeedbackIntervals},
            {"smoothing and latest raw error", checkSmoothingAndCurrentError},
            {"4K monitor, square ROI, and desktop offsets", checkRoiAndMonitorCoordinates},
            {"direct movement probe compatibility", checkDirectMoveByCompatibility},
            {"independent horizontal and vertical calibration", checkPerAxisCalibration},
            {"full calibrated movement on first update", checkFullCalibratedMovement},
            {"stable target across neighbors, vector order and keypoint tiers", checkStableMultiTargetLock},
            {"pause during target loss and reacquire after grace", checkLostTargetGrace},
            {"preserve identity across camera motion and ambiguous frames", checkCameraMotionAssociation},
            {"exact absolute coordinates without smoothing", checkExactAbsoluteCoordinates},
        };
        for (auto const& test : tests) {
            test.run();
            std::cout << "PASS " << test.name << '\n';
        }
        return 0;
    } catch (std::exception const& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
