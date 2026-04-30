#include "OverlayWindow.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

LRESULT CALLBACK overlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

} // namespace

HWND createOverlayWindow(HINSTANCE instance, int width, int height) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = overlayWndProc;
    wc.hInstance = instance;
    wc.lpszClassName = L"TensorRTScreenOverlay";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        wc.lpszClassName,
        L"TensorRT person detector overlay",
        WS_POPUP,
        0,
        0,
        width,
        height,
        nullptr,
        nullptr,
        instance,
        nullptr);
    if (!hwnd) {
        throw std::runtime_error("Create overlay window failed");
    }

    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    return hwnd;
}

void drawOverlay(HWND hwnd, std::vector<Box> const& boxes) {
    RECT client{};
    GetClientRect(hwnd, &client);

    HDC dc = GetDC(hwnd);
    HBRUSH clearBrush = CreateSolidBrush(RGB(0, 0, 0));
    FillRect(dc, &client, clearBrush);
    DeleteObject(clearBrush);

    HPEN pen = CreatePen(PS_SOLID, 4, RGB(0, 255, 0));
    HGDIOBJ oldPen = SelectObject(dc, pen);
    HGDIOBJ oldBrush = SelectObject(dc, GetStockObject(HOLLOW_BRUSH));
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(0, 255, 0));

    for (auto const& box : boxes) {
        int x1 = static_cast<int>(std::round(box.x1));
        int y1 = static_cast<int>(std::round(box.y1));
        int x2 = static_cast<int>(std::round(box.x2));
        int y2 = static_cast<int>(std::round(box.y2));
        Rectangle(dc, x1, y1, x2, y2);

        wchar_t text[64]{};
        swprintf_s(text, L"person %.2f", box.score);
        TextOutW(dc, x1, std::max(0, y1 - 22), text, static_cast<int>(wcslen(text)));
    }

    SelectObject(dc, oldBrush);
    SelectObject(dc, oldPen);
    DeleteObject(pen);
    ReleaseDC(hwnd, dc);
}
