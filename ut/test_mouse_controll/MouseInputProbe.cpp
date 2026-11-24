#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "MouseInputProbe.h"

#include <chrono>
#include <iostream>
#include <thread>

MouseConfig MouseInputProbe::configuration() {
    MouseConfig config;
    config.enabled = true;
    config.mode = "relative";
    config.relativeScale = 1.0f;
    config.moveIntervalMs = 0;
    return config;
}

MouseInputProbe::MouseInputProbe() : controller_(configuration()) {}

int MouseInputProbe::run() {
    std::cout << std::unitbuf
              << "Relative input probe: focus the cloud game within 5 seconds.\n"
              << "No detection or mouse button is required. ESC cancels.\n"
              << "The probe sends 100 horizontal input units over 1 second.\n";
    for (int remaining = 5; remaining > 0; --remaining) {
        std::cout << "Starting in " << remaining << " seconds...\n";
        for (int tick = 0; tick < 50; ++tick) {
            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
                std::cout << "Cancelled.\n";
                return 2;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    HWND foreground = GetForegroundWindow();
    DWORD foregroundPid = 0;
    GetWindowThreadProcessId(foreground, &foregroundPid);
    std::cout << "[probe] foreground_pid=" << foregroundPid << '\n';
    auto start = std::chrono::steady_clock::now();
    for (int tick = 0; tick < 50; ++tick) {
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            std::cout << "Cancelled.\n";
            return 2;
        }
        if (!foreground || GetForegroundWindow() != foreground) {
            std::cerr << "Foreground window changed; probe stopped.\n";
            return 2;
        }
        if (!controller_.moveBy(2, 0, 0, 0)) {
            DWORD error = GetLastError();
            std::cerr << "[probe] input-failed step=" << tick
                      << " win32_error=" << error << '\n';
            return 1;
        }
        std::this_thread::sleep_until(start + std::chrono::milliseconds((tick + 1) * 20));
    }
    std::cout << "[probe] input-submitted events=50 total_dx=100\n"
              << "This confirms Windows submission only. Observe whether the cloud game view moved.\n";
    return 0;
}
