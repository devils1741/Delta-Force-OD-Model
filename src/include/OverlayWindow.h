#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "Detection.h"

#include <vector>

/**
 * @brief 创建全屏透明置顶检测框覆盖窗口。
 * @param instance 当前进程实例句柄。
 * @param width 覆盖窗口宽度。
 * @param height 覆盖窗口高度。
 * @return 创建成功的窗口句柄。
 */
HWND createOverlayWindow(HINSTANCE instance, int x, int y, int width, int height);

/**
 * @brief 在透明覆盖窗口上绘制检测框。
 * @param hwnd 覆盖窗口句柄。
 * @param boxes 需要绘制的检测框列表。
 * @note 无返回值。
 */
void drawOverlay(HWND hwnd, std::vector<Box> const& boxes, LetterboxInfo const& letterbox);
