#include "CudaPostprocess.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

struct DecodedBox {
    Box screen;
    float inputX1{};
    float inputY1{};
    float inputX2{};
    float inputY2{};
};

void checkCuda(cudaError_t status, char const* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

__device__ float rgbHue(float r, float g, float b, float& saturation, float& value) {
    float maxChannel = fmaxf(r, fmaxf(g, b));
    float minChannel = fminf(r, fminf(g, b));
    float delta = maxChannel - minChannel;
    value = maxChannel;
    if (maxChannel < 0.42f || delta < 0.10f) {
        saturation = 0.0f;
        return -1.0f;
    }

    saturation = delta / maxChannel;
    if (maxChannel == r) {
        float hue = 60.0f * fmodf((g - b) / delta, 6.0f);
        return hue < 0.0f ? hue + 360.0f : hue;
    }
    if (maxChannel == g) {
        return 60.0f * ((b - r) / delta + 2.0f);
    }
    return 60.0f * ((r - g) / delta + 4.0f);
}

__device__ bool isTeamLogoPixel(float r, float g, float b) {
    float saturation = 0.0f;
    float value = 0.0f;
    float hue = rgbHue(r, g, b, saturation, value);
    if (hue < 0.0f || saturation < 0.35f) {
        return false;
    }

    bool greenLogo = hue >= 75.0f && hue <= 165.0f && g >= r && g >= b;
    bool blueLogo = hue >= 175.0f && hue <= 235.0f && b >= r && g >= r;
    return greenLogo || blueLogo;
}

__device__ bool hasTeamLogo(float const* input, DecodedBox const& box, LetterboxInfo letterbox) {
    int width = letterbox.inputW;
    int height = letterbox.inputH;
    int plane = width * height;

    float boxW = box.inputX2 - box.inputX1;
    float boxH = box.inputY2 - box.inputY1;
    if (boxW < 12.0f || boxH < 20.0f) {
        return false;
    }

    int x1 = max(0, min(width - 1, static_cast<int>(box.inputX1 + boxW * 0.10f)));
    int x2 = max(0, min(width - 1, static_cast<int>(box.inputX2 - boxW * 0.10f)));
    int y1 = max(0, min(height - 1, static_cast<int>(box.inputY1 + 6.0f)));
    int y2 = max(0, min(height - 1, static_cast<int>(box.inputY1 + boxH * 0.45f)));
    if (x2 <= x1 || y2 <= y1) {
        return false;
    }

    int step = (x2 - x1 > 96 || y2 - y1 > 96) ? 2 : 1;
    int greenPixels = 0;
    int sampledPixels = 0;

    for (int y = y1; y <= y2; y += step) {
        int row = y * width;
        for (int x = x1; x <= x2; x += step) {
            int idx = row + x;
            float r = input[idx];
            float g = input[plane + idx];
            float b = input[2 * plane + idx];
            sampledPixels++;
            greenPixels += isTeamLogoPixel(r, g, b) ? 1 : 0;
        }
    }

    if (sampledPixels <= 0) {
        return false;
    }

    float greenRatio = static_cast<float>(greenPixels) / static_cast<float>(sampledPixels);
    return greenPixels >= 8 && greenRatio >= 0.006f;
}

__device__ DecodedBox decodeBox(float const* output, int i, LetterboxInfo letterbox, float scoreThreshold) {
    float a = output[i * 6 + 0];
    float b = output[i * 6 + 1];
    float c = output[i * 6 + 2];
    float d = output[i * 6 + 3];
    float score = output[i * 6 + 4];

    DecodedBox box{};
    box.screen.score = -1.0f;
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

    box.inputX1 = fminf(fmaxf(x1, 0.0f), static_cast<float>(letterbox.inputW - 1));
    box.inputY1 = fminf(fmaxf(y1, 0.0f), static_cast<float>(letterbox.inputH - 1));
    box.inputX2 = fminf(fmaxf(x2, 0.0f), static_cast<float>(letterbox.inputW - 1));
    box.inputY2 = fminf(fmaxf(y2, 0.0f), static_cast<float>(letterbox.inputH - 1));

    x1 = (box.inputX1 - static_cast<float>(letterbox.padX)) / letterbox.scale +
         static_cast<float>(letterbox.captureX);
    y1 = (box.inputY1 - static_cast<float>(letterbox.padY)) / letterbox.scale +
         static_cast<float>(letterbox.captureY);
    x2 = (box.inputX2 - static_cast<float>(letterbox.padX)) / letterbox.scale +
         static_cast<float>(letterbox.captureX);
    y2 = (box.inputY2 - static_cast<float>(letterbox.padY)) / letterbox.scale +
         static_cast<float>(letterbox.captureY);

    x1 = fminf(fmaxf(x1, 0.0f), static_cast<float>(letterbox.screenW - 1));
    y1 = fminf(fmaxf(y1, 0.0f), static_cast<float>(letterbox.screenH - 1));
    x2 = fminf(fmaxf(x2, 0.0f), static_cast<float>(letterbox.screenW - 1));
    y2 = fminf(fmaxf(y2, 0.0f), static_cast<float>(letterbox.screenH - 1));

    if (x2 <= x1 || y2 <= y1) {
        return box;
    }

    box.screen.x1 = x1;
    box.screen.y1 = y1;
    box.screen.x2 = x2;
    box.screen.y2 = y2;
    box.screen.score = score;
    return box;
}

__global__ void decodeDetectionsKernel(
    float const* output,
    float const* input,
    int candidateCount,
    LetterboxInfo letterbox,
    float scoreThreshold,
    int maxDetections,
    Box* keptBoxes,
    int* keptCount,
    int* rawCount,
    int* teamFilteredCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= candidateCount || i >= maxDetections) {
        return;
    }

    DecodedBox box = decodeBox(output, i, letterbox, scoreThreshold);
    if (box.screen.score < scoreThreshold) {
        return;
    }

    atomicAdd(rawCount, 1);
    if (hasTeamLogo(input, box, letterbox)) {
        atomicAdd(teamFilteredCount, 1);
        return;
    }

    int out = atomicAdd(keptCount, 1);
    if (out < maxDetections) {
        keptBoxes[out] = box.screen;
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
    void* rawCount{};
    void* teamFilteredCount{};
    checkCuda(cudaMalloc(&boxes, hostBoxes_.size() * sizeof(Box)), "cudaMalloc postprocess boxes");
    checkCuda(cudaMalloc(&count, sizeof(int)), "cudaMalloc postprocess count");
    checkCuda(cudaMalloc(&rawCount, sizeof(int)), "cudaMalloc postprocess raw count");
    checkCuda(cudaMalloc(&teamFilteredCount, sizeof(int)), "cudaMalloc postprocess team filtered count");
    deviceBoxes_.reset(boxes);
    deviceCount_.reset(count);
    deviceRawCount_.reset(rawCount);
    deviceTeamFilteredCount_.reset(teamFilteredCount);
}

CudaPostprocessor::~CudaPostprocessor() = default;

std::vector<Box> CudaPostprocessor::decodeDetections(
    float const* deviceOutput,
    float const* deviceInput,
    int candidateCount,
    LetterboxInfo const& letterbox,
    float scoreThreshold,
    cudaStream_t stream) {
    candidateCount = std::clamp(candidateCount, 0, maxDetections_);

    checkCuda(cudaMemsetAsync(deviceCount_.get(), 0, sizeof(int), stream), "cudaMemsetAsync postprocess count");
    checkCuda(cudaMemsetAsync(deviceRawCount_.get(), 0, sizeof(int), stream),
              "cudaMemsetAsync postprocess raw count");
    checkCuda(cudaMemsetAsync(deviceTeamFilteredCount_.get(), 0, sizeof(int), stream),
              "cudaMemsetAsync postprocess team filtered count");

    int constexpr threadsPerBlock = 128;
    int blocks = (candidateCount + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks > 0) {
        decodeDetectionsKernel<<<blocks, threadsPerBlock, 0, stream>>>(
            deviceOutput,
            deviceInput,
            candidateCount,
            letterbox,
            scoreThreshold,
            maxDetections_,
            static_cast<Box*>(deviceBoxes_.get()),
            static_cast<int*>(deviceCount_.get()),
            static_cast<int*>(deviceRawCount_.get()),
            static_cast<int*>(deviceTeamFilteredCount_.get()));
        checkCuda(cudaGetLastError(), "decodeDetectionsKernel");
    }
    checkCuda(cudaMemcpyAsync(&hostCount_, deviceCount_.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync postprocess count");
    checkCuda(cudaMemcpyAsync(&hostRawCount_, deviceRawCount_.get(), sizeof(int), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync postprocess raw count");
    checkCuda(cudaMemcpyAsync(
                  &hostTeamFilteredCount_,
                  deviceTeamFilteredCount_.get(),
                  sizeof(int),
                  cudaMemcpyDeviceToHost,
                  stream),
              "cudaMemcpyAsync postprocess team filtered count");
    checkCuda(cudaMemcpyAsync(hostBoxes_.data(), deviceBoxes_.get(), hostBoxes_.size() * sizeof(Box),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync postprocess boxes");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize postprocess");

    hostCount_ = std::clamp(hostCount_, 0, maxDetections_);
    return {hostBoxes_.begin(), hostBoxes_.begin() + hostCount_};
}
