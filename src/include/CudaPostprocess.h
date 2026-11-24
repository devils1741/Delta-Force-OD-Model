#pragma once
#include "Detection.h"
#include <cuda_runtime_api.h>
#include <memory>

// GPU-only decode, NMS and size filtering. No host readback or stream synchronization.
class CudaPostprocessor {
public:
    explicit CudaPostprocessor(int maxDetections);
    ~CudaPostprocessor();
    CudaPostprocessor(CudaPostprocessor const&) = delete;
    CudaPostprocessor& operator=(CudaPostprocessor const&) = delete;
    void enqueue(float const* output, int candidates, LetterboxInfo const& letterbox,
                 float threshold, float minWidth, float minHeight, cudaStream_t stream, bool poseOutput = false, float logoThreshold = .3f, float logoMinWidth = 0.f, float logoMinHeight = 0.f);
    Box const* deviceBoxes() const { return static_cast<Box const*>(boxes_.get()); }
    int const* deviceCount() const { return static_cast<int const*>(count_.get()); }
    int capacity() const { return maxDetections_; }
private:
    struct CudaDeleter { void operator()(void* ptr) const; };
    using CudaPtr = std::unique_ptr<void, CudaDeleter>;
    int maxDetections_{};
    int allocatedCandidates_{};
    CudaPtr boxes_, count_;
};
