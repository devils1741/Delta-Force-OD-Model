#pragma once
#include "Detection.h"
#include <cuda_runtime_api.h>
#include <memory>
#ifdef DELTA_GPU_TESTING
#include <vector>
#include <cstdint>
#endif
struct ID3D11Device;
struct ID3D11DeviceContext;

// CUDA writes the D3D11 shader buffer; DirectComposition presents transparent GPU frames.
class GpuOverlay {
public:
    GpuOverlay(ID3D11Device* device, ID3D11DeviceContext* context,
               LetterboxInfo const& letterbox, int maxDetections);
    ~GpuOverlay();
    GpuOverlay(GpuOverlay const&) = delete;
    GpuOverlay& operator=(GpuOverlay const&) = delete;
    void draw(Box const* boxes, int const* count, LetterboxInfo const& letterbox,
              bool showRegion, cudaStream_t stream, bool showKeypoints = true, float keypointThreshold = .3f, float keypointRadius = 4.f, bool showFriends = true, bool showLogos = true);
#ifdef DELTA_GPU_TESTING
    std::vector<uint32_t> readPixelsForTest();
#endif
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
