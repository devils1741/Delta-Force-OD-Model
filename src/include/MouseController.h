#pragma once

#include "AppConfig.h"
#include "Detection.h"

#include <cstdint>
#include <vector>

/** @brief 封装目标选择、鼠标移动和逐帧冷却；由主线程调用。 */
class MouseController {
public:
    explicit MouseController(MouseConfig const& config);

    /** @brief 每次收到新的检测结果时更新鼠标控制。 */
    void update(std::vector<Box> const& boxes, LetterboxInfo const& letterbox,
                uint64_t sequence);

    /** @brief 使用当前移动模式执行偏移；坐标单位为屏幕像素。 */
    bool moveBy(int dx, int dy, int referenceX, int referenceY);

private:
    MouseConfig config_;
    Box lockedTarget_{};
    bool hasLockedTarget_{false};
    int cooldownFramesRemaining_{0};
};
