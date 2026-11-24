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
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <string_view>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

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

// High-resolution waits avoid rounding a short frame interval up to the system timer tick.
class FrameWaiter {
public:
    FrameWaiter() : timer_(CreateWaitableTimerExW(nullptr,nullptr,
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,TIMER_ALL_ACCESS)) {
        if(!timer_) throw std::runtime_error("Cannot create high-resolution frame timer");
    }
    ~FrameWaiter() { CloseHandle(timer_); }
    void wait(Clock::duration duration) {
        auto ticks=std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()/100;
        if(ticks<=0) return;
        LARGE_INTEGER due{}; due.QuadPart=-ticks;
        if(!SetWaitableTimerEx(timer_,&due,0,nullptr,nullptr,nullptr,0))
            throw std::runtime_error("Cannot set frame timer");
        DWORD result=MsgWaitForMultipleObjectsEx(1,&timer_,INFINITE,QS_ALLINPUT,MWMO_INPUTAVAILABLE);
        if(result==WAIT_FAILED) throw std::runtime_error("Frame timer wait failed");
    }
private:
    HANDLE timer_{};
};

} // namespace

/**
 * @brief 程序入口，启动配置加载、TensorRT推理、DXGI采集和覆盖窗口绘制。
 * @return 0表示正常退出；1表示发生异常并已提示错误。
 */
int main(int argc, char** argv) {
    bool smokeTest=argc==2 && std::string_view(argv[1])=="--smoke-test";
    try {
        SetConsoleOutputCP(CP_UTF8);
        std::cout << std::unitbuf;
        enableDpiAwareness();
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        auto path=findConfigFile();
        AppConfig::instance().load(path);
        auto const& config=AppConfig::instance();
        std::cout << "[mouse-config] file=" << path.string() << " enabled=" << config.mouse().enabled
                  << " mode=" << config.mouse().mode << '\n';
        // Capture selects the CUDA adapter before TensorRT allocates any GPU resources.
        DxgiScreenCapture capture(config.model().inputWidth,config.model().inputHeight);
        TensorRtDetector detector(config.model().onnxPath,config.model().enginePath);
        int candidates=static_cast<int>(detector.outputCount()/(detector.poseOutput()?12:6));
        if(candidates<=0) throw std::runtime_error("Model has no detection candidates");
        int capacity=std::min(config.inference().maxDetections,candidates);
        CudaPostprocessor postprocessor(capacity);
        GpuOverlay overlay(capture.device(),capture.context(),capture.letterbox(),capacity);
        auto mouseConfig=config.mouse();
        if(smokeTest) mouseConfig.enabled=false;
        AsyncMousePipeline mouse(mouseConfig,capacity,config.inference().keypointThreshold);
        FrameWaiter waiter;
        std::cout << "检测已启动，按 Ctrl+Shift+F12 退出。坐标为显示器内像素坐标。\n";
        struct TimingEvents {
            cudaEvent_t begin{}, end{};
            TimingEvents() {
                if (cudaEventCreate(&begin)!=cudaSuccess || cudaEventCreate(&end)!=cudaSuccess)
                    throw std::runtime_error("创建推理计时事件失败");
            }
            ~TimingEvents() { if(begin) cudaEventDestroy(begin); if(end) cudaEventDestroy(end); }
        } timing;
        auto cudaCheck=[](cudaError_t status) {
            if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        };
        std::vector<Box> logBoxes(capacity);
        // Read raw pose rows only on existing diagnostic frames, before CPU reporting.
        std::vector<float> logRawPose(detector.poseOutput() ? detector.outputCount() : 0);
        bool running=true;
        uint64_t sequence=0, frames=0;
        auto windowStart=Clock::now();
        auto nextFrame=Clock::now();
        auto runStart=nextFrame;
        while(running) {
            MSG msg{};
            while(PeekMessageW(&msg,nullptr,0,0,PM_REMOVE)) {
                if(msg.message==WM_QUIT) { std::cout << "程序退出：收到窗口关闭消息。\n"; running=false; break; }
                TranslateMessage(&msg); DispatchMessageW(&msg);
            }
            if(!running) break;
            if ((GetAsyncKeyState(VK_CONTROL)&0x8000) &&
                (GetAsyncKeyState(VK_SHIFT)&0x8000) &&
                (GetAsyncKeyState(VK_F12)&0x8000)) {
                std::cout << "程序退出：按下 Ctrl+Shift+F12。\n";
                break;
            }
            mouse.checkError();
            if(smokeTest && Clock::now()-runStart>std::chrono::seconds(10))
                throw std::runtime_error("Smoke test timed out waiting for screen frames");
            auto captureStart=Clock::now();
            if(!capture.captureToDevice(detector.deviceInput(),detector.stream())) {
                waiter.wait(std::chrono::microseconds(100));
                continue;
            }
            auto info=capture.letterbox();
            double captureMs=std::chrono::duration<double,std::milli>(Clock::now()-captureStart).count();
            bool logFrame=(sequence % config.inference().logIntervalFrames)==0;
            if(logFrame) cudaCheck(cudaEventRecord(timing.begin,detector.stream()));
            detector.enqueueDeviceInput();
            if(logFrame) cudaCheck(cudaEventRecord(timing.end,detector.stream()));
            postprocessor.enqueue(detector.deviceOutput(),candidates,info,
                config.inference().scoreThreshold,config.inference().minTargetWidthPx,
                config.inference().minTargetHeightPx,detector.stream(),detector.poseOutput(),config.inference().logoScoreThreshold,config.inference().logoMinWidthPx,config.inference().logoMinHeightPx);
            // Draw every result. No CPU box readback, count query or frame-skipping interval.
            overlay.draw(postprocessor.deviceBoxes(),postprocessor.deviceCount(),info,
                config.capture().showRegionBox,detector.stream(),config.inference().showKeypoints,config.inference().keypointThreshold,config.inference().keypointRadiusPx,config.inference().showFriendBoxes,config.inference().showFriendLogoBoxes);
            mouse.submit(postprocessor.deviceBoxes(),postprocessor.deviceCount(),info,++sequence,detector.stream());
            if(logFrame) {
                // Only sampled log frames wait for GPU readback; other frames remain asynchronous.
                int count=0;
                cudaCheck(cudaMemcpyAsync(&count,postprocessor.deviceCount(),sizeof(count),cudaMemcpyDeviceToHost,detector.stream()));
                cudaCheck(cudaMemcpyAsync(logBoxes.data(),postprocessor.deviceBoxes(),capacity*sizeof(Box),cudaMemcpyDeviceToHost,detector.stream()));
                if (!logRawPose.empty())
                    cudaCheck(cudaMemcpyAsync(logRawPose.data(),detector.deviceOutput(),
                        logRawPose.size()*sizeof(float),cudaMemcpyDeviceToHost,detector.stream()));
                cudaCheck(cudaStreamSynchronize(detector.stream()));
                float inferMs=0;
                cudaCheck(cudaEventElapsedTime(&inferMs,timing.begin,timing.end));
                count=std::clamp(count,0,capacity);
                std::cout << std::fixed << std::setprecision(2)
                          << "\n【检测日志】帧=" << sequence << "，图像接收耗时=" << captureMs
                          << " 毫秒（采集及预处理提交），GPU推理耗时=" << inferMs << " 毫秒，目标数=" << count << '\n';
                std::cout << "[detection-region] screen=(" << info.screenW << ',' << info.screenH
                          << ") roi=(" << info.captureX << ',' << info.captureY << ','
                          << info.captureW << ',' << info.captureH << ")\n";
                if (!logRawPose.empty()) {
                    int bestAlive = -1;
                    float bestScore = -1;
                    for (int i=0; i<candidates; ++i) {
                        auto const* row=logRawPose.data()+i*12;
                        if (row[5]!=0 || !std::isfinite(row[4]) || row[4]<0 || row[4]>1 ||
                            !std::isfinite(row[0]) || !std::isfinite(row[1]) ||
                            !std::isfinite(row[2]) || !std::isfinite(row[3]) ||
                            row[2]<=row[0] || row[3]<=row[1]) continue;
                        if (row[4]>bestScore) { bestScore=row[4]; bestAlive=i; }
                    }
                    std::cout << "[detection-raw] frame=" << sequence << " best_alive_score=" << bestScore
                              << " score_threshold=" << config.inference().scoreThreshold
                              << " min_size=(" << config.inference().minTargetWidthPx << ','
                              << config.inference().minTargetHeightPx << ") kept_total=" << count;
                    if (bestAlive>=0) {
                        auto const* row=logRawPose.data()+bestAlive*12;
                        float x1=std::clamp((std::clamp(row[0],0.0f,float(info.inputW-1))-info.padX)/info.scale+info.captureX,0.0f,float(info.screenW-1));
                        float y1=std::clamp((std::clamp(row[1],0.0f,float(info.inputH-1))-info.padY)/info.scale+info.captureY,0.0f,float(info.screenH-1));
                        float x2=std::clamp((std::clamp(row[2],0.0f,float(info.inputW-1))-info.padX)/info.scale+info.captureX,0.0f,float(info.screenW-1));
                        float y2=std::clamp((std::clamp(row[3],0.0f,float(info.inputH-1))-info.padY)/info.scale+info.captureY,0.0f,float(info.screenH-1));
                        std::cout << " best_alive_screen_box=(" << x1 << ',' << y1 << ',' << x2 << ',' << y2
                                  << ") best_alive_size=(" << x2-x1 << ',' << y2-y1 << ") rejection="
                                  << (bestScore<config.inference().scoreThreshold ? "score" :
                                      x2-x1<config.inference().minTargetWidthPx || y2-y1<config.inference().minTargetHeightPx
                                      ? "size" : "passed-filters");
                    } else std::cout << " rejection=no-valid-alive-row";
                    std::cout << '\n';
                }
                char const* names[]={"活敌","尸体","队友","队友标志"};
                for(int i=0;i<count;++i) {
                    auto const& box=logBoxes[i];
                    auto name=box.classId>=0 && box.classId<4 ? names[box.classId] : "未知类别";
                    std::cout << "  目标" << i+1 << "：" << name << "，置信度=" << box.score
                              << "，中心坐标=(" << (box.x1+box.x2)*.5f << "," << (box.y1+box.y2)*.5f
                              << ")，边框=[" << box.x1 << "," << box.y1 << "," << box.x2 << "," << box.y2
                              << "]，宽高=(" << box.x2-box.x1 << "," << box.y2-box.y1 << ")\n";
                    if(detector.poseOutput() && box.classId!=3) {
                        for(int k=0;k<2;++k)
                            std::cout << "    关键点" << k+1 << "：坐标=(" << box.keypoints[k*3] << "," << box.keypoints[k*3+1]
                                      << ")，置信度=" << box.keypoints[k*3+2] << "，状态="
                                      << (box.keypoints[k*3+2]>0 && box.keypoints[k*3+2]>=config.inference().keypointThreshold ? "显示" : "低于阈值") << '\n';
                    } else std::cout << "    关键点：不适用\n";
                }
            }
            ++frames;
            if(smokeTest && sequence>=20) {
                auto status=cudaStreamSynchronize(detector.stream());
                if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
                std::cout << "自检通过：完成20帧采集、推理和绘制，鼠标控制已禁用\n";
                break;
            }
            auto now=Clock::now();
            nextFrame+=std::chrono::microseconds(1'000'000/config.inference().targetFps);
            if(now<nextFrame) waiter.wait(nextFrame-now);
            else nextFrame=now;
        }
        return 0;
    } catch(std::exception const& error) {
        std::cerr << "错误：" << error.what() << '\n';
        if(!smokeTest) MessageBoxA(nullptr,error.what(),"GPU screen inference failed",MB_ICONERROR|MB_OK);
        return 1;
    }
}
