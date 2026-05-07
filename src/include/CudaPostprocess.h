#pragma once

#include "Detection.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <vector>

class CudaPostprocessor {
public:
    explicit CudaPostprocessor(int maxDetections);
    ~CudaPostprocessor();

    CudaPostprocessor(CudaPostprocessor const&) = delete;
    CudaPostprocessor& operator=(CudaPostprocessor const&) = delete;

    std::vector<Box> decodeDetections(
        float const* deviceOutput,
        float const* deviceInput,
        int candidateCount,
        LetterboxInfo const& letterbox,
        float scoreThreshold,
        cudaStream_t stream);

private:
    struct CudaDeleter {
        void operator()(void* ptr) const;
    };

    using CudaPtr = std::unique_ptr<void, CudaDeleter>;

    int maxDetections_{};
    CudaPtr deviceBoxes_;
    CudaPtr deviceCount_;
    std::vector<Box> hostBoxes_;
    int hostCount_{};
};
