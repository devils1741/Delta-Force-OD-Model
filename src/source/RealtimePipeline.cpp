#include "RealtimePipeline.h"

#include <algorithm>
#include <utility>

LatestFrameQueue::LatestFrameQueue() {
    for (auto& slot : slots_) {
        slot.resize(3 * kInputW * kInputH);
    }
}

void LatestFrameQueue::publish(float const* input) {
    std::unique_lock lock(mutex_);
    if (stopped_) {
        return;
    }

    int slot = chooseWritableSlot();
    std::copy(input, input + slots_[slot].size(), slots_[slot].begin());
    latestSlot_ = slot;
    latestSequence_++;

    lock.unlock();
    cv_.notify_one();
}

bool LatestFrameQueue::waitLatest(LatestFrame& frame) {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] {
        return stopped_ || latestSlot_ >= 0;
    });
    if (stopped_ && latestSlot_ < 0) {
        return false;
    }

    readingSlot_ = latestSlot_;
    latestSlot_ = -1;
    frame = {readingSlot_, latestSequence_, slots_[readingSlot_].data()};
    return true;
}

void LatestFrameQueue::release(int slot) {
    std::lock_guard lock(mutex_);
    if (readingSlot_ == slot) {
        readingSlot_ = -1;
    }
}

void LatestFrameQueue::stop() {
    {
        std::lock_guard lock(mutex_);
        stopped_ = true;
    }
    cv_.notify_all();
}

int LatestFrameQueue::chooseWritableSlot() const {
    if (latestSlot_ >= 0 && latestSlot_ != readingSlot_) {
        return latestSlot_;
    }
    for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
        if (i != readingSlot_) {
            return i;
        }
    }
    return 0;
}

void LatestBoxes::publish(std::vector<Box> boxes) {
    std::lock_guard lock(mutex_);
    boxes_ = std::move(boxes);
    sequence_++;
}

bool LatestBoxes::snapshot(uint64_t& lastSequence, std::vector<Box>& boxes) {
    std::lock_guard lock(mutex_);
    if (sequence_ == lastSequence) {
        return false;
    }
    boxes = boxes_;
    lastSequence = sequence_;
    return true;
}

void ThreadError::capture(std::exception_ptr error) {
    std::lock_guard lock(mutex_);
    if (!error_) {
        error_ = error;
    }
}

void ThreadError::rethrowIfAny() {
    std::exception_ptr error;
    {
        std::lock_guard lock(mutex_);
        error = error_;
    }
    if (error) {
        std::rethrow_exception(error);
    }
}
