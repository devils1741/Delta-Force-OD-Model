#pragma once

#include "Detection.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <vector>

/**
 * @brief 最新帧队列返回的帧引用。
 *
 * 保存槽位编号、序列号以及指向输入张量数据的只读指针。
 */
struct LatestFrame {
    int slot{-1};
    uint64_t sequence{};
    float const* data{};
};

/**
 * @brief 只保留最新输入帧的双缓冲队列。
 *
 * 用于采集线程和推理线程之间传递最新一帧，避免旧帧堆积。
 */
class LatestFrameQueue {
public:
    /**
     * @brief 创建最新帧队列并按模型输入尺寸分配槽位。
     * @note 无返回值。
     */
    LatestFrameQueue();

    /**
     * @brief 发布一帧输入张量到队列。
     * @param input 输入张量数据指针。
     * @note 无返回值。
     */
    void publish(float const* input);
    /**
     * @brief 等待并取出最新帧。
     * @param frame 输出参数，接收最新帧槽位、序列号和数据指针。
     * @return 成功取到帧时返回true；队列停止且没有待处理帧时返回false。
     */
    bool waitLatest(LatestFrame& frame);
    /**
     * @brief 释放正在读取的槽位。
     * @param slot 需要释放的槽位编号。
     * @note 无返回值。
     */
    void release(int slot);
    /**
     * @brief 停止队列并唤醒等待线程。
     * @note 无返回值。
     */
    void stop();

private:
    /**
     * @brief 选择可写入的缓冲槽位。
     * @return 可安全覆盖的槽位编号。
     */
    int chooseWritableSlot() const;

    std::array<std::vector<float>, 2> slots_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int latestSlot_{-1};
    int readingSlot_{-1};
    uint64_t latestSequence_{};
    bool stopped_{};
};

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
