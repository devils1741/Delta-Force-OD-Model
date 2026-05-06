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

#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
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
        LetterboxInfo letterbox = capture.letterbox();

        std::cout << "Running realtime screen inference.\n";

        HWND overlay = createOverlayWindow(GetModuleHandle(nullptr), capture.screenW(), capture.screenH());
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
                    detector.enqueueDeviceInput();
                    auto afterInfer = Clock::now();
                    auto detections = postprocessor.decodeDetections(
                        detector.deviceOutput(),
                        candidateCount,
                        letterbox,
                        config.inference().scoreThreshold,
                        detector.stream());

                    if constexpr (kLogFrameTimings) {
                    if (!detections.empty()) {
                        auto afterPost = Clock::now();
                        auto captureMs =
                            std::chrono::duration<double, std::milli>(afterCapture - start).count();
                        auto inferMs =
                            std::chrono::duration<double, std::milli>(afterInfer - afterCapture).count();
                        auto postMs =
                            std::chrono::duration<double, std::milli>(afterPost - afterInfer).count();
                        auto totalMs =
                            std::chrono::duration<double, std::milli>(afterPost - start).count();
                        std::cout << "frame=" << frameIndex
                                  << " total_ms=" << std::fixed << std::setprecision(2) << totalMs
                                  << " capture_gpu_ms=" << captureMs
                                  << " infer_ms=" << inferMs
                                  << " post_ms=" << postMs
                                  << " detections=" << detections.size() << '\n';
                    }
                    }
                    frameIndex++;

                    latestBoxes.publish(std::move(detections));
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
            if (shouldDraw && latestBoxes.snapshot(drawnSequence, boxes)) {
                drawOverlay(overlay, boxes);
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
