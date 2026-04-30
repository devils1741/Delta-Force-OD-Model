#pragma once

#include "Detection.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <vector>

struct LatestFrame {
    int slot{-1};
    uint64_t sequence{};
    float const* data{};
};

class LatestFrameQueue {
public:
    LatestFrameQueue();

    void publish(float const* input);
    bool waitLatest(LatestFrame& frame);
    void release(int slot);
    void stop();

private:
    int chooseWritableSlot() const;

    std::array<std::vector<float>, 2> slots_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int latestSlot_{-1};
    int readingSlot_{-1};
    uint64_t latestSequence_{};
    bool stopped_{};
};

class LatestBoxes {
public:
    void publish(std::vector<Box> boxes);
    bool snapshot(uint64_t& lastSequence, std::vector<Box>& boxes);

private:
    std::mutex mutex_;
    std::vector<Box> boxes_;
    uint64_t sequence_{};
};

class ThreadError {
public:
    void capture(std::exception_ptr error);
    void rethrowIfAny();

private:
    std::mutex mutex_;
    std::exception_ptr error_;
};
