#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "AppConfig.h"
#include "CudaPostprocess.h"
#include "Detection.h"
#include "DxgiScreenCapture.h"
#include "OverlayWindow.h"
#include "MouseController.h"
#include "RealtimePipeline.h"
#include "TensorRtDetector.h"

#include <algorithm>
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

// CUDA events are recorded on the inference stream; postprocessing already synchronizes it.
struct PerfEvents {
    cudaEvent_t captureStart{}, captureEnd{}, inferEnd{};
    PerfEvents() {
        if (cudaEventCreate(&captureStart) != cudaSuccess ||
            cudaEventCreate(&captureEnd) != cudaSuccess ||
            cudaEventCreate(&inferEnd) != cudaSuccess)
            throw std::runtime_error("Cannot create performance CUDA events");
    }
    ~PerfEvents() { cudaEventDestroy(captureStart); cudaEventDestroy(captureEnd); cudaEventDestroy(inferEnd); }
};

namespace {

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
        int candidateCount = static_cast<int>(detector.outputCount() / 6);

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
                PerfEvents perf;
                auto windowStart = Clock::now();
                double totalSum=0, totalMax=0, capSum=0, gpuCapSum=0, gpuInferSum=0, postSum=0;
                uint64_t windowFrames=0, captureMisses=0;
                std::cout << "[perf] timings in ms; GPU capture includes stream idle/capture wait; post_cpu includes GPU wait.\n";
                auto nextFrameTime = Clock::now();
                while (running.load(std::memory_order_relaxed)) {
                    nextFrameTime += std::chrono::microseconds(1'000'000 / config.inference().targetFps);

                    auto start = Clock::now();
                    cudaEventRecord(perf.captureStart, detector.stream());
                    if (!capture.captureToDevice(detector.deviceInput(), detector.stream())) {
                        ++captureMisses;
                        // Request 10 us; actual wake-up latency depends on Windows scheduling.
                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                        continue;
                    }

                    cudaEventRecord(perf.captureEnd, detector.stream());
                    auto afterCapture = Clock::now();
                    LetterboxInfo frameLetterbox = capture.letterbox();
                    detector.enqueueDeviceInput();
                    cudaEventRecord(perf.inferEnd, detector.stream());
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

                    auto afterPost = Clock::now();
                    auto ms=[](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
                    float gpuCapture=0, gpuInfer=0;
                    if(cudaEventElapsedTime(&gpuCapture,perf.captureStart,perf.captureEnd)!=cudaSuccess ||
                       cudaEventElapsedTime(&gpuInfer,perf.captureEnd,perf.inferEnd)!=cudaSuccess)
                        throw std::runtime_error("Cannot read performance CUDA events");
                    double total=ms(start,afterPost);
                    totalSum+=total; totalMax=std::max(totalMax,total);
                    capSum+=ms(start,afterCapture); gpuCapSum+=gpuCapture;
                    gpuInferSum+=gpuInfer; postSum+=ms(afterInfer,afterPost); ++windowFrames;
                    if(ms(windowStart,afterPost)>=2000) {
                        std::cout << "[perf] frames=" << windowFrames
                                  << " fps=" << std::fixed << std::setprecision(2) << windowFrames*1000.0/ms(windowStart,afterPost)
                                  << " total_avg=" << totalSum/windowFrames << " total_max=" << totalMax
                                  << " capture_cpu=" << capSum/windowFrames << " capture_gpu_span=" << gpuCapSum/windowFrames
                                  << " infer_gpu=" << gpuInferSum/windowFrames << " post_cpu_wait=" << postSum/windowFrames
                                  << " capture_misses=" << captureMisses << " raw=" << postprocessor.rawDetectionCount()
                                  << " final=" << detections.size() << " small_filtered=" << smallFiltered << std::endl;
                        windowStart=afterPost; windowFrames=0; captureMisses=0;
                        totalSum=totalMax=capSum=gpuCapSum=gpuInferSum=postSum=0;
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
        MouseController mouseController(config.mouse());
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
                mouseController.update(boxes, drawLetterbox, drawnSequence);
                bool shouldDrawOverlay =
                    config.inference().overlayIntervalFrames <= 1 ||
                    drawnSequence % static_cast<uint64_t>(config.inference().overlayIntervalFrames) == 0 ||
                    boxes.empty();
                if (shouldDrawOverlay) {
                    auto drawStart=Clock::now();
                    drawOverlay(overlay, boxes, drawLetterbox);
                    static auto lastDrawLog=Clock::now();
                    auto drawEnd=Clock::now();
                    if(drawEnd-lastDrawLog>=std::chrono::seconds(2)) {
                        std::cout << "[draw] sequence=" << drawnSequence << " boxes=" << boxes.size()
                                  << " submit_ms=" << std::chrono::duration<double,std::milli>(drawEnd-drawStart).count()
                                  << " (CPU submission, excludes compositor/display latency)" << std::endl;
                        lastDrawLog=drawEnd;
                    }
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
