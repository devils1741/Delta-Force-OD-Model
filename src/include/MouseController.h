#pragma once

#include "AppConfig.h"
#include "Detection.h"

#include <cstdint>
#include <chrono>
#include <cstddef>
#include <vector>
#include <atomic>
#include <thread>

/** @brief 封装目标选择和每次检测更新时的鼠标移动；由主线程调用。 */
class MouseController {
public:
    explicit MouseController(MouseConfig const& config);
    ~MouseController();
    MouseController(MouseController const&) = delete;
    MouseController& operator=(MouseController const&) = delete;

    /** @brief 每次收到新的检测结果时更新鼠标控制。 */
    void update(std::vector<Box> const& boxes, LetterboxInfo const& letterbox,
                uint64_t sequence, float keypointThreshold = 0.3f);

    /** @brief 使用当前移动模式执行偏移；坐标单位为屏幕像素。 */
    bool moveBy(int dx, int dy, int referenceX, int referenceY);

private:
    bool submitMove(int dx, int dy, int referenceX, int referenceY, bool aimCorrection);
    void logState(char const* state, uint64_t sequence, std::size_t boxCount,
                  int dx = 0, int dy = 0, bool force = false);

    MouseConfig config_;
    std::atomic<bool> leftButtonDown_{false};
    std::atomic<uint64_t> rawMouseEvents_{0};
    std::atomic<uint64_t> injectedMouseEvents_{0};
    std::atomic<uint64_t> injectedDownEvents_{0};
    std::atomic<uint64_t> injectedUpEvents_{0};
    std::atomic<uint64_t> leftDownEvents_{0};
    std::atomic<uint64_t> leftUpEvents_{0};
    uint64_t handledLeftDownEvents_{0}; // Only accessed by the update thread.
    std::thread inputThread_;
    unsigned long inputThreadId_{0};
    Box lockedTarget_{};
    Box predictedTarget_{};
    bool hasPredictedTarget_{false};
    uint64_t lockPressEvents_{0};
    uint64_t targetLockId_{0};
    std::chrono::steady_clock::time_point targetLostSince_{};
    int lockedTargetPriority_{-1};
    double filteredTargetX_{}, filteredTargetY_{};
    std::chrono::steady_clock::time_point lastTargetUpdate_{};
    bool hasLockedTarget_{false};
    std::chrono::steady_clock::time_point lastMoveTime_{};
    unsigned long lastInputError_{0};
    uint64_t moveAttempts_{0};
    uint64_t movesSubmitted_{0};
    int lastRelativeDx_{0};
    int lastRelativeDy_{0};
    std::chrono::steady_clock::time_point diagnosticsStart_{std::chrono::steady_clock::now()};
    std::chrono::steady_clock::time_point nextDiagnosticLog_{};
};
