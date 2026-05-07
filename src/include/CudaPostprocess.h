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

    int rawDetectionCount() const { return hostRawCount_; }
    int teamFilteredCount() const { return hostTeamFilteredCount_; }

private:
    struct CudaDeleter {
        void operator()(void* ptr) const;
    };

    using CudaPtr = std::unique_ptr<void, CudaDeleter>;

    int maxDetections_{};
    CudaPtr deviceBoxes_;
    CudaPtr deviceCount_;
    CudaPtr deviceRawCount_;
    CudaPtr deviceTeamFilteredCount_;
    std::vector<Box> hostBoxes_;
    int hostCount_{};
    int hostRawCount_{};
    int hostTeamFilteredCount_{};
};
