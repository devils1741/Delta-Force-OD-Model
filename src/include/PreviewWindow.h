#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "Detection.h"
#include "ScreenCapture.h"

#include <vector>

/**
 * @brief 创建 CPU 预览窗口。
 * @param instance 当前进程实例句柄。
 * @return 创建成功的窗口句柄。
 */
HWND createPreviewWindow(HINSTANCE instance);

/**
 * @brief 绘制预处理后的输入画面和检测框。
 * @param hwnd 预览窗口句柄。
 * @param capture 提供预处理像素和 letterbox 信息的采集对象。
 * @param boxes 需要叠加绘制的检测框列表。
 * @note 无返回值。
 */
void drawPreview(HWND hwnd, ScreenCapture const& capture, std::vector<Box> const& boxes);
