#include "OverlayWindow.h"
#include "AppConfig.h"

#include <algorithm>
#include <cmath>
#include <cwchar>
#include <stdexcept>
#include <vector>

namespace {

/**
 * @brief 透明覆盖窗口的Windows消息过程。
 * @param hwnd 接收消息的窗口句柄。
 * @param msg Windows消息编号。
 * @param wParam 消息的附加参数。
 * @param lParam 消息的附加参数。
 * @return 消息处理结果。
 */
LRESULT CALLBACK overlayWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (msg == WM_DESTROY) {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wParam, lParam);
}

RECT boxDirtyRect(Box const& box) {
    RECT rect{};
    rect.left = static_cast<LONG>(std::floor(box.x1)) - 8;
    rect.top = static_cast<LONG>(std::floor(box.y1)) - 8;
    rect.right = static_cast<LONG>(std::ceil(box.x2)) + 8;
    rect.bottom = static_cast<LONG>(std::ceil(box.y2)) + 8;
    return rect;
}

RECT confidenceRect(HDC dc, Box const& box, RECT const& client) {
    wchar_t text[32]{};
    swprintf_s(text, L"%.1f%%", box.score * 100.0f);
    SIZE size{};
    GetTextExtentPoint32W(dc, text, static_cast<int>(wcslen(text)), &size);
    LONG x = std::clamp(static_cast<LONG>(std::round(box.x1)),
        client.left, std::max(client.left, client.right - size.cx));
    LONG y = std::clamp(static_cast<LONG>(std::round(box.y1)) - size.cy - 4,
        client.top, std::max(client.top, client.bottom - size.cy));
    return RECT{x, y, x + size.cx, y + size.cy};
}

void clampRect(RECT& rect, RECT const& bounds) {
    rect.left = std::clamp(rect.left, bounds.left, bounds.right);
    rect.top = std::clamp(rect.top, bounds.top, bounds.bottom);
    rect.right = std::clamp(rect.right, bounds.left, bounds.right);
    rect.bottom = std::clamp(rect.bottom, bounds.top, bounds.bottom);
}

bool isEmpty(RECT const& rect) {
    return rect.right <= rect.left || rect.bottom <= rect.top;
}

RECT roiRect(LetterboxInfo const& letterbox) {
    RECT rect{};
    rect.left = letterbox.captureX;
    rect.top = letterbox.captureY;
    rect.right = letterbox.captureX + letterbox.captureW;
    rect.bottom = letterbox.captureY + letterbox.captureH;
    return rect;
}

} // namespace

HWND createOverlayWindow(HINSTANCE instance, int x, int y, int width, int height) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = overlayWndProc;
    wc.hInstance = instance;
    wc.lpszClassName = L"TensorRTScreenOverlay";
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClassW(&wc);

    HWND hwnd = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        wc.lpszClassName,
        L"TensorRT person detector overlay",
        WS_POPUP,
        x,
        y,
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
    SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        x,
        y,
        width,
        height,
        SWP_NOACTIVATE | SWP_SHOWWINDOW);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);

    HDC dc = GetDC(hwnd);
    RECT client{0, 0, width, height};
    HBRUSH clearBrush = CreateSolidBrush(RGB(0, 0, 0));
    FillRect(dc, &client, clearBrush);
    DeleteObject(clearBrush);
    ReleaseDC(hwnd, dc);

    return hwnd;
}

void drawOverlay(HWND hwnd, std::vector<Box> const& boxes, LetterboxInfo const& letterbox) {
    static std::vector<Box> previousBoxes;
    static HBRUSH clearBrush = CreateSolidBrush(RGB(0, 0, 0));
    static HPEN roiPen = CreatePen(PS_SOLID, 3, RGB(255, 0, 0));
    static HPEN boxPen = CreatePen(PS_SOLID, 4, RGB(0, 255, 0));

    SetWindowPos(
        hwnd,
        HWND_TOPMOST,
        letterbox.desktopX,
        letterbox.desktopY,
        letterbox.screenW,
        letterbox.screenH,
        SWP_NOACTIVATE | SWP_SHOWWINDOW);

    RECT client{};
    GetClientRect(hwnd, &client);

    HDC dc = GetDC(hwnd);
    std::vector<RECT> dirtyRects;
    dirtyRects.reserve(previousBoxes.size() + boxes.size());

    for (auto const& box : previousBoxes) {
        RECT rect = boxDirtyRect(box);
        RECT label = confidenceRect(dc, box, client);
        UnionRect(&rect, &rect, &label);
        clampRect(rect, client);
        if (!isEmpty(rect)) {
            dirtyRects.push_back(rect);
        }
    }
    for (auto const& box : boxes) {
        RECT rect = boxDirtyRect(box);
        RECT label = confidenceRect(dc, box, client);
        UnionRect(&rect, &rect, &label);
        clampRect(rect, client);
        if (!isEmpty(rect)) {
            dirtyRects.push_back(rect);
        }
    }

    RECT roi = roiRect(letterbox);
    clampRect(roi, client);
    if (!isEmpty(roi)) {
        dirtyRects.push_back(roi);
    }

    for (auto const& rect : dirtyRects) {
        FillRect(dc, &rect, clearBrush);
    }

    HGDIOBJ oldBrush = SelectObject(dc, GetStockObject(HOLLOW_BRUSH));

    HGDIOBJ oldPen = SelectObject(dc, roiPen);
    if (AppConfig::instance().capture().showRegionBox && !isEmpty(roi)) {
        Rectangle(dc, roi.left, roi.top, roi.right, roi.bottom);
    }

    int oldBkMode = SetBkMode(dc, TRANSPARENT);
    COLORREF oldTextColor = SetTextColor(dc, RGB(0, 255, 0));
    SelectObject(dc, boxPen);
    for (auto const& box : boxes) {
        int x1 = static_cast<int>(std::round(box.x1));
        int y1 = static_cast<int>(std::round(box.y1));
        int x2 = static_cast<int>(std::round(box.x2));
        int y2 = static_cast<int>(std::round(box.y2));
        Rectangle(dc, x1, y1, x2, y2);
        wchar_t text[32]{};
        swprintf_s(text, L"%.1f%%", box.score * 100.0f);
        RECT label = confidenceRect(dc, box, client);
        TextOutW(dc, label.left, label.top, text, static_cast<int>(wcslen(text)));
    }

    SetTextColor(dc, oldTextColor);
    SetBkMode(dc, oldBkMode);
    SelectObject(dc, oldBrush);
    SelectObject(dc, oldPen);
    ReleaseDC(hwnd, dc);
    previousBoxes = boxes;
}
