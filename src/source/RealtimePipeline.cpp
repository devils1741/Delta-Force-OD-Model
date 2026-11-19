#include "RealtimePipeline.h"

#include <utility>

void LatestBoxes::publish(std::vector<Box> boxes, LetterboxInfo letterbox) {
    std::lock_guard lock(mutex_);
    boxes_ = std::move(boxes);
    letterbox_ = letterbox;
    sequence_++;
}

bool LatestBoxes::snapshot(uint64_t& lastSequence, std::vector<Box>& boxes, LetterboxInfo& letterbox) {
    std::lock_guard lock(mutex_);
    if (sequence_ == lastSequence) {
        return false;
    }
    boxes = boxes_;
    letterbox = letterbox_;
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
