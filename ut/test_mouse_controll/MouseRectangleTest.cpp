#include "MouseRectangleTest.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>

MouseRectangleTest::MouseRectangleTest(std::string mode)
    : controller_(configuration(mode)) {}

int MouseRectangleTest::run() {
    SetProcessDPIAware();
    int width = GetSystemMetrics(SM_CXSCREEN);
    int height = GetSystemMetrics(SM_CYSCREEN);
    if (width < 740 || height < 740) throw std::runtime_error("Display must be at least 740x740");
    left_ = (width - 720) / 2;
    top_ = (height - 720) / 2;
    WNDCLASSW wc{};
    wc.lpfnWndProc = windowProc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"MouseRectangleTest";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    if (!RegisterClassW(&wc)) throw std::runtime_error("RegisterClass failed");
    HWND window = CreateWindowExW(WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW,
        wc.lpszClassName, L"Mouse rectangle test", WS_POPUP,
        left_ - 10, top_ - 10, 741, 741, nullptr, nullptr, wc.hInstance, nullptr);
    if (!window) throw std::runtime_error("CreateWindow failed");
    SetLayeredWindowAttributes(window, RGB(0,0,0), 0, LWA_COLORKEY);
    ShowWindow(window, SW_SHOWNOACTIVATE);
    UpdateWindow(window);
    std::cout << "720x720 rectangle. Starting in 3 seconds. ESC cancels.\n";
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start < std::chrono::seconds(3)) {
        if (!pump()) { DestroyWindow(window); return 2; }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    // Position at top-left using the same controller before traversing the perimeter.
    POINT position{};
    GetCursorPos(&position);
    if (!controller_.moveBy(left_ - position.x, top_ - position.y, position.x, position.y))
        throw std::runtime_error("Initial movement failed");
    POINT corners[] = {{left_+720,top_}, {left_+720,top_+720},
                       {left_,top_+720}, {left_,top_}};
    int maxError = 0;
    for (int edge = 0; edge < 4; ++edge) {
        POINT origin = edge == 0 ? POINT{left_,top_} : corners[edge-1];
        auto edgeStart = std::chrono::steady_clock::now();
        for (int step = 1; step <= 360; ++step) {
            if (!pump()) { DestroyWindow(window); return 2; }
            POINT desired{origin.x + (corners[edge].x-origin.x)*step/360,
                          origin.y + (corners[edge].y-origin.y)*step/360};
            if (!GetCursorPos(&position) ||
                !controller_.moveBy(desired.x-position.x, desired.y-position.y, position.x, position.y))
                throw std::runtime_error("Mouse movement failed");
            std::this_thread::sleep_until(edgeStart + std::chrono::milliseconds(step*16));
            if (!GetCursorPos(&position)) throw std::runtime_error("GetCursorPos failed");
            maxError = std::max(maxError, static_cast<int>(std::max(
                std::abs(position.x-desired.x), std::abs(position.y-desired.y))));
        }
        std::cout << "Edge " << edge+1 << " complete\n";
    }
    DestroyWindow(window);
    std::cout << "Finished clockwise traversal; maximum observed error: " << maxError << " px\n";
    return maxError <= 3 ? 0 : 1;
}

MouseConfig MouseRectangleTest::configuration(std::string const& mode) {
    if (mode != "absolute" && mode != "relative") throw std::runtime_error("Use absolute or relative");
    MouseConfig config;
    config.mode = mode;
    config.moveCooldownFrames = 0;
    return config;
}

bool MouseRectangleTest::pump() {
    MSG msg{};
    while (PeekMessageW(&msg,nullptr,0,0,PM_REMOVE)) {
        if (msg.message == WM_QUIT) return false;
        TranslateMessage(&msg); DispatchMessageW(&msg);
    }
    return !(GetAsyncKeyState(VK_ESCAPE)&0x8000);
}

LRESULT CALLBACK MouseRectangleTest::windowProc(HWND hwnd, UINT msg, WPARAM w, LPARAM l) {
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps{}; HDC dc = BeginPaint(hwnd,&ps);
        RECT area{}; GetClientRect(hwnd,&area);
        FillRect(dc,&area,static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH)));
        HPEN pen = CreatePen(PS_SOLID,2,RGB(0,255,80));
        auto oldPen = SelectObject(dc,pen);
        auto oldBrush = SelectObject(dc,GetStockObject(NULL_BRUSH));
        Rectangle(dc,10,10,730,730);
        SelectObject(dc,oldBrush); SelectObject(dc,oldPen); DeleteObject(pen);
        EndPaint(hwnd,&ps); return 0;
    }
    return DefWindowProcW(hwnd,msg,w,l);
}
