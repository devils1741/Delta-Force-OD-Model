#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "Detection.h"
#include "ScreenCapture.h"

#include <vector>

HWND createPreviewWindow(HINSTANCE instance);
void drawPreview(HWND hwnd, ScreenCapture const& capture, std::vector<Box> const& boxes);
