#pragma once

#include "Detection.h"

#include <cstdint>
#include <exception>
#include <mutex>
#include <vector>

/**
 * @brief 线程安全的最新检测框缓存。
 *
 * 用序列号标记检测结果是否更新，供绘制线程按需刷新。
 */
class LatestBoxes {
public:
    /**
     * @brief 发布最新检测框。
     * @param boxes 新的检测框列表，函数会接管其内容。
     * @note 无返回值。
     */
    void publish(std::vector<Box> boxes, LetterboxInfo letterbox);
    /**
     * @brief 获取检测框快照。
     * @param lastSequence 调用方持有的上次序列号，函数会在有更新时写入新序列号。
     * @param boxes 输出参数，接收最新检测框。
     * @return 有新检测结果时返回true；序列号未变化时返回false。
     */
    bool snapshot(uint64_t& lastSequence, std::vector<Box>& boxes, LetterboxInfo& letterbox);

private:
    std::mutex mutex_;
    std::vector<Box> boxes_;
    LetterboxInfo letterbox_{};
    uint64_t sequence_{};
};

/**
 * @brief 跨线程异常传递器。
 *
 * 捕获工作线程中的第一个异常，并在主线程中重新抛出。
 */
class ThreadError {
public:
    /**
     * @brief 保存线程中捕获到的异常。
     * @param error 需要保存的异常指针。
     * @note 无返回值。
     */
    void capture(std::exception_ptr error);
    /**
     * @brief 如果已有异常则在当前线程重新抛出。
     * @note 无返回值。
     */
    void rethrowIfAny();

private:
    std::mutex mutex_;
    std::exception_ptr error_;
};
