#pragma once

#include "MouseController.h"

/** @brief 不依赖检测、左键和光标位置，单独检查相对输入链路。 */
class MouseInputProbe {
public:
    MouseInputProbe();
    int run();

private:
    static MouseConfig configuration();
    MouseController controller_;
};
