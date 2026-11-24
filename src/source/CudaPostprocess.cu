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

__device__ DecodedBox decodeBox(float const* output, int i, int count, LetterboxInfo letterbox, float scoreThreshold, bool poseOutput, float logoThreshold) {
    DecodedBox box{};
    box.screen.score=-1.0f;
    float x1,y1,x2,y2,score;
    if (poseOutput) {
        auto row=output+i*12;
        x1=row[0]; y1=row[1]; x2=row[2]; y2=row[3]; score=row[4];
        if (!isfinite(row[5]) || row[5]<0 || row[5]>3 || row[5]!=floorf(row[5])) return box;
        box.screen.classId=static_cast<int>(row[5]);
        for (int k=0;k<2;++k) {
            float x=row[6+k*3],y=row[7+k*3],c=row[8+k*3];
            if (!isfinite(x)||!isfinite(y)||!isfinite(c)||c<0||c>1) continue;
            box.screen.keypoints[k*3]=fminf(fmaxf((x-letterbox.padX)/letterbox.scale+letterbox.captureX,0.0f),float(letterbox.screenW-1));
            box.screen.keypoints[k*3+1]=fminf(fmaxf((y-letterbox.padY)/letterbox.scale+letterbox.captureY,0.0f),float(letterbox.screenH-1));
            box.screen.keypoints[k*3+2]=c;
        }
    } else {
        float cx=output[i],cy=output[count+i],w=output[2*count+i],h=output[3*count+i];
        float s0=output[4*count+i],s1=output[5*count+i];
        if (!isfinite(s0)||!isfinite(s1)||w<=0||h<=0) return box;
        score=fmaxf(s0,s1); box.screen.classId=s1>s0?1:0;
        x1=cx-w*.5f;y1=cy-h*.5f;x2=cx+w*.5f;y2=cy+h*.5f;
    }
    if (poseOutput && box.screen.classId==3) scoreThreshold=logoThreshold;
    if (!isfinite(x1)||!isfinite(y1)||!isfinite(x2)||!isfinite(y2)||!isfinite(score)||score<scoreThreshold||score>1) return box;
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
    int candidateCount,
    LetterboxInfo letterbox,
    float scoreThreshold,
    int maxDetections,
    Box* keptBoxes,
    int* keptCount, bool poseOutput, float logoThreshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= candidateCount) {
        return;
    }

    DecodedBox box = decodeBox(output, i, candidateCount, letterbox, scoreThreshold, poseOutput, logoThreshold);
    if (box.screen.score < ((poseOutput && box.screen.classId==3) ? logoThreshold : scoreThreshold)) {
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

// Preserve the previous behavior: size filtering happens after NMS.
__global__ void filterSmallKernel(Box* boxes, int* count, float minWidth, float minHeight, float logoMinWidth, float logoMinHeight) {
    if (threadIdx.x || blockIdx.x) return;
    int kept = 0;
    for (int i = 0; i < *count; ++i) {
        Box box = boxes[i];
        if (box.x2 - box.x1 >= (box.classId==3?logoMinWidth:minWidth) && box.y2 - box.y1 >= (box.classId==3?logoMinHeight:minHeight))
            boxes[kept++] = box;
    }
    *count = kept;
}
} // namespace

void CudaPostprocessor::CudaDeleter::operator()(void* ptr) const {
    if (ptr) cudaFree(ptr);
}
CudaPostprocessor::CudaPostprocessor(int maxDetections)
    : maxDetections_(std::max(1, maxDetections)) {
    void* count{};
    checkCuda(cudaMalloc(&count, sizeof(int)), "cudaMalloc detection count");
    count_.reset(count);
}
CudaPostprocessor::~CudaPostprocessor() = default;

void CudaPostprocessor::enqueue(float const* output, int candidates,
    LetterboxInfo const& letterbox, float threshold, float minWidth, float minHeight,
    cudaStream_t stream, bool poseOutput, float logoThreshold, float logoMinWidth, float logoMinHeight) {
    checkCuda(cudaMemsetAsync(count_.get(), 0, sizeof(int), stream), "clear detection count");
    if (candidates <= 0) return;
    if (candidates > allocatedCandidates_) {
        void* boxes{};
        checkCuda(cudaMalloc(&boxes, static_cast<size_t>(candidates) * sizeof(Box)), "cudaMalloc candidates");
        boxes_.reset(boxes);
        allocatedCandidates_ = candidates;
    }
    decodeDetectionsKernel<<<(candidates + 127) / 128, 128, 0, stream>>>(output,
        candidates, letterbox, threshold, candidates,
        static_cast<Box*>(boxes_.get()), static_cast<int*>(count_.get()), poseOutput, logoThreshold);
    checkCuda(cudaGetLastError(), "decodeDetectionsKernel");
    if (!poseOutput) nmsKernel<<<1, 256, 0, stream>>>(static_cast<Box*>(boxes_.get()),
        static_cast<int*>(count_.get()), candidates, maxDetections_);
    checkCuda(cudaGetLastError(), "nmsKernel");
    filterSmallKernel<<<1, 1, 0, stream>>>(static_cast<Box*>(boxes_.get()),
        static_cast<int*>(count_.get()), minWidth, minHeight, logoMinWidth, logoMinHeight);
    checkCuda(cudaGetLastError(), "filterSmallKernel");
}
