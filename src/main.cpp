#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "AppConfig.h"
#include "CudaPostprocess.h"
#include "Detection.h"
#include "DxgiScreenCapture.h"
#include "OverlayWindow.h"
#include "RealtimePipeline.h"
#include "TensorRtDetector.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

#ifndef NDEBUG
constexpr bool kLogFrameTimings = true;
#else
constexpr bool kLogFrameTimings = false;
#endif
/**
 * @brief 启用进程DPI感知，避免高DPI屏幕下坐标缩放错误。
 * @note 无返回值。
 */
void enableDpiAwareness() {
    using SetDpiAwarenessContextFn = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
    auto* user32 = GetModuleHandleW(L"user32.dll");
    auto* setDpiAwarenessContext = user32
        ? reinterpret_cast<SetDpiAwarenessContextFn>(
              GetProcAddress(user32, "SetProcessDpiAwarenessContext"))
        : nullptr;

    if (setDpiAwarenessContext &&
        setDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
        return;
    }
    SetProcessDPIAware();
}

/**
 * @brief 在常见运行目录中查找配置文件。
 * @return 配置文件的绝对路径。
 */
fs::path findConfigFile() {
    for (auto const& candidate : {
             fs::path("config/config.yaml"),
             fs::path("../config/config.yaml"),
             fs::current_path() / "config/config.yaml",
             fs::current_path().parent_path() / "config/config.yaml",
         }) {
        if (fs::exists(candidate)) {
            return fs::absolute(candidate);
        }
    }
    throw std::runtime_error("Cannot find config/config.yaml. Run from the project root or cmake-build-debug.");
}

bool isUsableTarget(Box const& box, InferenceConfig const& inference) {
    return (box.x2 - box.x1) >= inference.minTargetWidthPx &&
           (box.y2 - box.y1) >= inference.minTargetHeightPx;
}

void removeSmallTargets(std::vector<Box>& boxes, InferenceConfig const& inference) {
    boxes.erase(
        std::remove_if(
            boxes.begin(),
            boxes.end(),
            [&](Box const& box) {
                return !isUsableTarget(box, inference);
            }),
        boxes.end());
}

bool isLeftButtonRequested() {
    SHORT state = GetAsyncKeyState(VK_LBUTTON);
    return (state & 0x8000) != 0;
}

bool aimFromScreenCenter(MouseConfig const& mouse) {
    return mouse.mode != "absolute";
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
    if (plan.delta.x == 0 && plan.delta.y == 0) {
        return true;
    }
    if (mouse.mode == "absolute") {
        return SetCursorPos(referencePoint.x + plan.delta.x, referencePoint.y + plan.delta.y) != FALSE;
    }
    if (mouse.mode == "both") {
        bool absoluteOk =
            SetCursorPos(referencePoint.x + plan.delta.x, referencePoint.y + plan.delta.y) != FALSE;
        bool relativeOk = sendRelativeMouseMove(plan.delta, mouse);
        return absoluteOk || relativeOk;
    }
    return sendRelativeMouseMove(plan.delta, mouse);
}

} // namespace

/**
 * @brief 程序入口，启动配置加载、TensorRT推理、DXGI采集和覆盖窗口绘制。
 * @return 0表示正常退出；1表示发生异常并已提示错误。
 */
int main() {
    try {
        std::cout << std::unitbuf;
        enableDpiAwareness();
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

        fs::path configPath = findConfigFile();
        AppConfig::instance().load(configPath);
        auto const& config = AppConfig::instance();

        fs::path onnxPath = config.model().onnxPath;
        fs::path enginePath = config.model().enginePath;
        std::cout << "Config: " << configPath << '\n';
        std::cout << "ONNX model: " << onnxPath << '\n';
        std::cout << "Engine cache: " << enginePath << '\n';

        TensorRtDetector detector(onnxPath, enginePath);
        CudaPostprocessor postprocessor(config.inference().maxDetections);
        int candidateCount = std::min(
            config.inference().maxDetections,
            static_cast<int>(detector.outputCount() / 6));

        DxgiScreenCapture capture(detector.inputW(), detector.inputH());
        std::cout << "Primary display: " << capture.screenW() << "x" << capture.screenH() << '\n';
        std::cout << "Letterbox resize: " << capture.letterbox().resizedW << "x"
                  << capture.letterbox().resizedH << ", pad=(" << capture.letterbox().padX
                  << ", " << capture.letterbox().padY << "), scale="
                  << capture.letterbox().scale << '\n';
        std::cout << "Capture ROI: x=" << capture.letterbox().captureX
                  << " y=" << capture.letterbox().captureY
                  << " w=" << capture.letterbox().captureW
                  << " h=" << capture.letterbox().captureH << '\n';
        std::cout << "Mouse mode: " << config.mouse().mode
                  << ", relative_scale=" << config.mouse().relativeScale
                  << ", move_cooldown_frames=" << config.mouse().moveCooldownFrames << '\n';
        std::cout << "Diagnostics: log_interval_frames=" << config.inference().logIntervalFrames
                  << ", overlay_interval_frames=" << config.inference().overlayIntervalFrames << '\n';
        LetterboxInfo initialLetterbox = capture.letterbox();

        std::cout << "Running realtime screen inference.\n";

        HWND overlay = createOverlayWindow(
            GetModuleHandle(nullptr),
            initialLetterbox.desktopX,
            initialLetterbox.desktopY,
            capture.screenW(),
            capture.screenH());
        LatestBoxes latestBoxes;
        ThreadError threadError;
        std::atomic_bool running{true};
        HANDLE drawEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!drawEvent) {
            throw std::runtime_error("CreateEventW drawEvent failed");
        }

        std::thread inferenceThread([&] {
            try {
                SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
                uint64_t frameIndex = 0;
                auto nextFrameTime = Clock::now();
                while (running.load(std::memory_order_relaxed)) {
                    nextFrameTime += std::chrono::microseconds(1'000'000 / config.inference().targetFps);

                    auto start = Clock::now();
                    if (!capture.captureToDevice(detector.deviceInput(), detector.stream())) {
                        Sleep(1);
                        continue;
                    }

                    auto afterCapture = Clock::now();
                    LetterboxInfo frameLetterbox = capture.letterbox();
                    detector.enqueueDeviceInput();
                    auto afterInfer = Clock::now();
                    auto detections = postprocessor.decodeDetections(
                        detector.deviceOutput(),
                        detector.deviceInput(),
                        candidateCount,
                        frameLetterbox,
                        config.inference().scoreThreshold,
                        detector.stream());
                    int beforeSmallFilter = static_cast<int>(detections.size());
                    removeSmallTargets(detections, config.inference());
                    int smallFiltered = beforeSmallFilter - static_cast<int>(detections.size());

                    if constexpr (kLogFrameTimings) {
                        auto afterPost = Clock::now();
                        auto captureMs =
                            std::chrono::duration<double, std::milli>(afterCapture - start).count();
                        auto inferMs =
                            std::chrono::duration<double, std::milli>(afterInfer - afterCapture).count();
                        auto postMs =
                            std::chrono::duration<double, std::milli>(afterPost - afterInfer).count();
                        auto totalMs =
                            std::chrono::duration<double, std::milli>(afterPost - start).count();
                        bool shouldLogFrame =
                            !detections.empty() &&
                            (config.inference().logIntervalFrames <= 1 ||
                             frameIndex % static_cast<uint64_t>(config.inference().logIntervalFrames) == 0);
                        if (shouldLogFrame) {
                            std::cout << "frame=" << frameIndex
                                      << " ms=" << std::fixed << std::setprecision(2) << totalMs
                                      << " cap=" << captureMs
                                      << " infer=" << inferMs
                                      << " post=" << postMs
                                      << " raw=" << postprocessor.rawDetectionCount()
                                      << " det=" << detections.size()
                                      << " team_filtered=" << postprocessor.teamFilteredCount()
                                      << " small_filtered=" << smallFiltered
                                      << '\n';
                        }
                    }
                    frameIndex++;

                    latestBoxes.publish(std::move(detections), frameLetterbox);
                    SetEvent(drawEvent);

                    auto now = Clock::now();
                    if (now < nextFrameTime) {
                        std::this_thread::sleep_until(nextFrameTime);
                    } else {
                        nextFrameTime = now;
                    }
                }
            } catch (...) {
                threadError.capture(std::current_exception());
                running.store(false, std::memory_order_relaxed);
                SetEvent(drawEvent);
            }
        });

        MSG msg{};
        uint64_t drawnSequence = 0;
        int moveCooldownFramesRemaining = 0;
        while (running.load(std::memory_order_relaxed)) {
            DWORD waitResult = MsgWaitForMultipleObjectsEx(
                1,
                &drawEvent,
                INFINITE,
                QS_ALLINPUT,
                MWMO_INPUTAVAILABLE);
            bool shouldDraw = waitResult == WAIT_OBJECT_0;

            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) {
                    running.store(false, std::memory_order_relaxed);
                    break;
                }
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
            std::vector<Box> boxes;
            LetterboxInfo drawLetterbox{};
            if (shouldDraw && latestBoxes.snapshot(drawnSequence, boxes, drawLetterbox)) {
                bool leftRequested = isLeftButtonRequested();
                POINT cursor{};
                bool hasCursor = GetCursorPos(&cursor) != FALSE;
                POINT screenCenter{
                    drawLetterbox.screenW / 2,
                    drawLetterbox.screenH / 2,
                };
                POINT screenCenterGlobal{
                    drawLetterbox.desktopX + screenCenter.x,
                    drawLetterbox.desktopY + screenCenter.y,
                };
                bool useScreenCenterAim = aimFromScreenCenter(config.mouse());
                POINT aimReferenceGlobal = useScreenCenterAim ? screenCenterGlobal : cursor;
                POINT aimReferenceLocal{
                    aimReferenceGlobal.x - drawLetterbox.desktopX,
                    aimReferenceGlobal.y - drawLetterbox.desktopY,
                };
                bool canMoveThisFrame = moveCooldownFramesRemaining <= 0;
                int selectedTargetIndex = leftRequested && hasCursor && canMoveThisFrame
                    ? nearestTargetIndexToPoint(boxes, aimReferenceLocal)
                    : -1;
                if (selectedTargetIndex >= 0) {
                    MouseMovePlan selectedMovePlan = planMouseMove(
                        boxes[static_cast<size_t>(selectedTargetIndex)],
                        drawLetterbox,
                        aimReferenceGlobal);
                    bool moveSucceeded =
                        applyMouseMovePlan(selectedMovePlan, aimReferenceGlobal, config.mouse());
                    if (moveSucceeded) {
                        moveCooldownFramesRemaining = config.mouse().moveCooldownFrames;
                    }
                    if constexpr (kLogFrameTimings) {
                        std::cout << "move frame=" << drawnSequence
                                  << " target=" << selectedTargetIndex
                                  << " dx=" << selectedMovePlan.delta.x
                                  << " dy=" << selectedMovePlan.delta.y
                                  << " ok=" << (moveSucceeded ? 1 : 0)
                                  << '\n';
                    }
                } else if (moveCooldownFramesRemaining > 0) {
                    --moveCooldownFramesRemaining;
                }
                bool shouldDrawOverlay =
                    config.inference().overlayIntervalFrames <= 1 ||
                    drawnSequence % static_cast<uint64_t>(config.inference().overlayIntervalFrames) == 0 ||
                    boxes.empty();
                if (shouldDrawOverlay) {
                    drawOverlay(overlay, boxes, drawLetterbox);
                }
            }
        }

        running.store(false, std::memory_order_relaxed);
        if (inferenceThread.joinable()) {
            inferenceThread.join();
        }
        CloseHandle(drawEvent);
        threadError.rethrowIfAny();

        return 0;
    } catch (std::exception const& e) {
        MessageBoxA(nullptr, e.what(), "TensorRT screen demo failed", MB_ICONERROR | MB_OK);
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
