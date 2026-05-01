#include "PreviewWindow.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

LRESULT CALLBACK wndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

} // namespace

HWND createPreviewWindow(HINSTANCE instance) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = wndProc;
    wc.hInstance = instance;
    wc.lpszClassName = L"TensorRTScreenDemo";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);

    RECT rect{0, 0, kInputW, kInputH};
    AdjustWindowRect(&rect, WS_OVERLAPPEDWINDOW, FALSE);
    HWND hwnd = CreateWindowExW(
        0, wc.lpszClassName, L"TensorRT person detector - ESC to exit",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left, rect.bottom - rect.top,
        nullptr, nullptr, instance, nullptr);
    if (!hwnd) {
        throw std::runtime_error("CreateWindowEx failed");
    }
    ShowWindow(hwnd, SW_SHOW);
    return hwnd;
}

void drawPreview(HWND hwnd, ScreenCapture const& capture, std::vector<Box> const& boxes) {
    HDC dc = GetDC(hwnd);
    RECT client{};
    GetClientRect(hwnd, &client);
    int clientW = std::max(1L, client.right - client.left);
    int clientH = std::max(1L, client.bottom - client.top);
    int viewSize = std::max(1, std::min(clientW, clientH));
    int viewX = (clientW - viewSize) / 2;
    int viewY = (clientH - viewSize) / 2;
    float viewScale = static_cast<float>(viewSize) / static_cast<float>(kInputW);

    HBRUSH background = CreateSolidBrush(RGB(0, 0, 0));
    FillRect(dc, &client, background);
    DeleteObject(background);

    StretchDIBits(
        dc, viewX, viewY, viewSize, viewSize, 0, 0, kInputW, kInputH,
        capture.pixels().data(), &capture.bmi(), DIB_RGB_COLORS, SRCCOPY);

    HPEN pen = CreatePen(PS_SOLID, 3, RGB(0, 255, 0));
    HGDIOBJ oldPen = SelectObject(dc, pen);
    HGDIOBJ oldBrush = SelectObject(dc, GetStockObject(HOLLOW_BRUSH));
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(0, 255, 0));

    for (auto const& box : boxes) {
        auto const& lb = capture.letterbox();
        int x1 = static_cast<int>(std::round(box.x1 * lb.scale + lb.padX));
        int y1 = static_cast<int>(std::round(box.y1 * lb.scale + lb.padY));
        int x2 = static_cast<int>(std::round(box.x2 * lb.scale + lb.padX));
        int y2 = static_cast<int>(std::round(box.y2 * lb.scale + lb.padY));
        x1 = viewX + static_cast<int>(std::round(x1 * viewScale));
        y1 = viewY + static_cast<int>(std::round(y1 * viewScale));
        x2 = viewX + static_cast<int>(std::round(x2 * viewScale));
        y2 = viewY + static_cast<int>(std::round(y2 * viewScale));
        Rectangle(dc, x1, y1, x2, y2);

        wchar_t text[64]{};
        swprintf_s(text, L"person %.2f", box.score);
        TextOutW(dc, x1, std::max(0, y1 - 20), text, static_cast<int>(wcslen(text)));
    }

    SelectObject(dc, oldBrush);
    SelectObject(dc, oldPen);
    DeleteObject(pen);
    ReleaseDC(hwnd, dc);
}
