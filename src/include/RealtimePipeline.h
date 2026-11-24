#pragma once
#include "AppConfig.h"
#include "Detection.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <memory>
#ifdef DELTA_GPU_TESTING
#include <functional>
#include <vector>
#endif

// Optional CPU mouse consumer. GPU rendering never waits for its D2H transfers.
class AsyncMousePipeline {
public:
    AsyncMousePipeline(MouseConfig const& config, int maxDetections, float keypointThreshold = 0.3f);
    ~AsyncMousePipeline();
#ifdef DELTA_GPU_TESTING
    using TestConsumer = std::function<void(std::vector<Box> const&, LetterboxInfo const&, uint64_t)>;
    AsyncMousePipeline(int capacity, TestConsumer consumer);
#endif
    AsyncMousePipeline(AsyncMousePipeline const&) = delete;
    AsyncMousePipeline& operator=(AsyncMousePipeline const&) = delete;
    void submit(Box const* boxes, int const* count, LetterboxInfo const& letterbox,
                uint64_t sequence, cudaStream_t producer);
    void checkError();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
