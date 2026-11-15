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

__device__ DecodedBox decodeBox(float const* output, int i, int count, LetterboxInfo letterbox, float scoreThreshold) {
    float cx=output[i], cy=output[count+i], w=output[2*count+i], h=output[3*count+i];
    float s0=output[4*count+i], s1=output[5*count+i];
    float score=fmaxf(s0,s1);
    DecodedBox box{};
    box.screen.score=-1.0f;
    if (!isfinite(cx)||!isfinite(cy)||!isfinite(w)||!isfinite(h)||!isfinite(s0)||!isfinite(s1)||score<scoreThreshold||score>1||w<=0||h<=0) return box;
    box.screen.classId=s1>s0 ? 1 : 0;
    float x1=cx-w*.5f, y1=cy-h*.5f, x2=cx+w*.5f, y2=cy+h*.5f;
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
    if (i >= candidateCount) {
        return;
    }

    DecodedBox box = decodeBox(output, i, candidateCount, letterbox, scoreThreshold);
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

// One block cooperatively selects the highest score and suppresses same-class overlaps.
__global__ void nmsKernel(Box* boxes, int* count, int capacity, int limit) {
    __shared__ float scores[256];
    __shared__ int indices[256];
    __shared__ Box selected;
    __shared__ int kept;
    int t=threadIdx.x;
    int n=min(*count,capacity);
    if(t==0) kept=0;
    __syncthreads();
    for(int k=0;k<min(n,limit);++k) {
        float best=-1; int idx=-1;
        for(int i=k+t;i<n;i+=blockDim.x) {
            if(boxes[i].score>best) {best=boxes[i].score;idx=i;}
        }
        scores[t]=best; indices[t]=idx; __syncthreads();
        for(int step=128;step;step/=2) {
            if(t<step && (scores[t+step]>scores[t] ||
                (scores[t+step]==scores[t] && indices[t+step]>=0 && (indices[t]<0 || indices[t+step]<indices[t])))) {
                scores[t]=scores[t+step]; indices[t]=indices[t+step];
            }
            __syncthreads();
        }
        if(scores[0]<0) break;
        if(t==0) {
            int winner=indices[0]; selected=boxes[winner];
            boxes[winner]=boxes[k]; boxes[k]=selected; kept=k+1;
        }
        __syncthreads();
        for(int i=k+1+t;i<n;i+=blockDim.x) {
            Box b=boxes[i];
            if(b.score<0 || b.classId!=selected.classId) continue;
            float iw=fmaxf(0,fminf(b.x2,selected.x2)-fmaxf(b.x1,selected.x1));
            float ih=fmaxf(0,fminf(b.y2,selected.y2)-fmaxf(b.y1,selected.y1));
            float inter=iw*ih;
            float uni=(b.x2-b.x1)*(b.y2-b.y1)+(selected.x2-selected.x1)*(selected.y2-selected.y1)-inter;
            if(uni>0 && inter/uni>0.45f) boxes[i].score=-1;
        }
        __syncthreads();
    }
    if(t==0) *count=kept;
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
    if(candidateCount<=0) return {};
    if(static_cast<size_t>(candidateCount)>hostBoxes_.size()) {
        void* boxes{};
        checkCuda(cudaMalloc(&boxes,static_cast<size_t>(candidateCount)*sizeof(Box)),"cudaMalloc candidates");
        deviceBoxes_.reset(boxes);
        hostBoxes_.resize(candidateCount);
    }

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
            candidateCount,
            static_cast<Box*>(deviceBoxes_.get()),
            static_cast<int*>(deviceCount_.get()),
            static_cast<int*>(deviceRawCount_.get()),
            static_cast<int*>(deviceTeamFilteredCount_.get()));
        checkCuda(cudaGetLastError(), "decodeDetectionsKernel");
    }
    nmsKernel<<<1,256,0,stream>>>(static_cast<Box*>(deviceBoxes_.get()),static_cast<int*>(deviceCount_.get()),candidateCount,maxDetections_);
    checkCuda(cudaGetLastError(),"nmsKernel");
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
    checkCuda(cudaMemcpyAsync(hostBoxes_.data(), deviceBoxes_.get(), static_cast<size_t>(std::min(candidateCount,maxDetections_)) * sizeof(Box),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync postprocess boxes");
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize postprocess");

    hostCount_ = std::clamp(hostCount_, 0, maxDetections_);
    return {hostBoxes_.begin(), hostBoxes_.begin() + hostCount_};
}
