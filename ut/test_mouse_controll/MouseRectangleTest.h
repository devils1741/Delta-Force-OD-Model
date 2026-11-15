#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "MouseController.h"
#include <string>

/** @brief 在屏幕中央绘制矩形并验证鼠标顺时针移动。 */
class MouseRectangleTest {
public:
    explicit MouseRectangleTest(std::string mode);
    int run();

private:
    static MouseConfig configuration(std::string const& mode);
    static bool pump();
    static LRESULT CALLBACK windowProc(HWND hwnd, UINT msg, WPARAM w, LPARAM l);

    MouseController controller_;
    int left_{};
    int top_{};
};
