#include "CudaPostprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(cudaError_t status, char const* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ Box decodeBox(float const* output, int i, LetterboxInfo letterbox, float scoreThreshold) {
    float a = output[i * 6 + 0];
    float b = output[i * 6 + 1];
    float c = output[i * 6 + 2];
    float d = output[i * 6 + 3];
    float score = output[i * 6 + 4];

    Box box{};
    box.score = -1.0f;
    if (score < scoreThreshold) {
        return box;
    }

    if (fmaxf(fmaxf(fabsf(a), fabsf(b)), fmaxf(fabsf(c), fabsf(d))) <= 2.0f) {
        a *= static_cast<float>(letterbox.inputW);
        c *= static_cast<float>(letterbox.inputW);
        b *= static_cast<float>(letterbox.inputH);
        d *= static_cast<float>(letterbox.inputH);
    }

    float x1 = fminf(a, c);
    float y1 = fminf(b, d);
    float x2 = fmaxf(a, c);
    float y2 = fmaxf(b, d);

    x1 = (x1 - static_cast<float>(letterbox.padX)) / letterbox.scale + static_cast<float>(letterbox.captureX);
    y1 = (y1 - static_cast<float>(letterbox.padY)) / letterbox.scale + static_cast<float>(letterbox.captureY);
    x2 = (x2 - static_cast<float>(letterbox.padX)) / letterbox.scale + static_cast<float>(letterbox.captureX);
    y2 = (y2 - static_cast<float>(letterbox.padY)) / letterbox.scale + static_cast<float>(letterbox.captureY);

    x1 = fminf(fmaxf(x1, 0.0f), static_cast<float>(letterbox.screenW - 1));
    y1 = fminf(fmaxf(y1, 0.0f), static_cast<float>(letterbox.screenH - 1));
    x2 = fminf(fmaxf(x2, 0.0f), static_cast<float>(letterbox.screenW - 1));
    y2 = fminf(fmaxf(y2, 0.0f), static_cast<float>(letterbox.screenH - 1));

    if (x2 <= x1 || y2 <= y1) {
        return box;
    }

    box.x1 = x1;
    box.y1 = y1;
    box.x2 = x2;
    box.y2 = y2;
    box.score = score;
    return box;
}

__global__ void decodeDetectionsKernel(
    float const* output,
    int candidateCount,
    LetterboxInfo letterbox,
    float scoreThreshold,
    int maxDetections,
    Box* keptBoxes,
    int* keptCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= candidateCount || i >= maxDetections) {
        return;
    }

    Box box = decodeBox(output, i, letterbox, scoreThreshold);
    if (box.score < scoreThreshold) {
        return;
    }

    int out = atomicAdd(keptCount, 1);
    if (out < maxDetections) {
        keptBoxes[out] = box;
    }
}

} // namespace

void CudaPostprocessor::CudaDeleter::operator()(void* ptr) const {
    if (ptr) {
        cudaFree(ptr);
    }
}

CudaPostprocessor::CudaPostprocessor(int maxDetections)
    : maxDetections_(std::max(1, maxDetections)),
      hostBoxes_(static_cast<size_t>(std::max(1, maxDetections))) {
    void* boxes{};
    void* count{};
    checkCuda(cudaMalloc(&boxes, hostBoxes_.size() * sizeof(Box)), "cudaMalloc postprocess boxes");
    checkCuda(cudaMalloc(&count, sizeof(int)), "cudaMalloc postprocess count");
    deviceBoxes_.reset(boxes);
    deviceCount_.reset(count);
}

CudaPostprocessor::~CudaPostprocessor() = default;

std::vector<Box> CudaPostprocessor::decodeDetections(
    float const* deviceOutput,
    int candidateCount,
    LetterboxInfo const& letterbox,
    float scoreThreshold,
    cudaStream_t stream) {
    candidateCount = std::clamp(candidateCount, 0, maxDetections_);

    checkCuda(cudaMemsetAsync(deviceCount_.get(), 0, sizeof(int), stream), "cudaMemsetAsync postprocess count");

    int constexpr threadsPerBlock = 128;
    int blocks = (candidateCount + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 0) {
        decodeDetectionsKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            deviceOutput,
            candidateCount,
            letterbox,
            scoreThreshold,
            maxDetections_,
            static_cast<Box*>(deviceBoxes_.get()),
            static_cast<int*>(deviceCount_.get()));
        checkCuda(cudaGetLastError(), "decodeDetectionsKernel");
    }
    checkCuda(cudaMemcpyAsync(&hostCount_, deviceCount_.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync postprocess count");
    checkCuda(cudaMemcpyAsync(hostBoxes_.data(), deviceBoxes_.get(), hostBoxes_.size() * sizeof(Box),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync postprocess boxes");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize postprocess");

    hostCount_ = std::clamp(hostCount_, 0, maxDetections_);
    return {hostBoxes_.begin(), hostBoxes_.begin() + hostCount_};
}
