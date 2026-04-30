#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "Detection.h"

#include <vector>

HWND createOverlayWindow(HINSTANCE instance, int width, int height);
void drawOverlay(HWND hwnd, std::vector<Box> const& boxes);
