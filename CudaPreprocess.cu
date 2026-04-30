#include "CudaPreprocess.h"

#include <cuda_runtime.h>

namespace {

__global__ void preprocessKernel(
    cudaTextureObject_t texture,
    int sourceW,
    int sourceH,
    LetterboxInfo letterbox,
    float* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= kInputW || y >= kInputH) {
        return;
    }

    int dst = y * kInputW + x;
    int plane = kInputW * kInputH;
    float r = 114.0f / 255.0f;
    float g = 114.0f / 255.0f;
    float b = 114.0f / 255.0f;

    int localX = x - letterbox.padX;
    int localY = y - letterbox.padY;
    if (localX >= 0 && localY >= 0 &&
        localX < letterbox.resizedW && localY < letterbox.resizedH) {
        int srcX = letterbox.captureX + static_cast<int>(localX / letterbox.scale);
        int srcY = letterbox.captureY + static_cast<int>(localY / letterbox.scale);
        srcX = min(max(srcX, 0), sourceW - 1);
        srcY = min(max(srcY, 0), sourceH - 1);
        uchar4 bgra = tex2D<uchar4>(texture, srcX + 0.5f, srcY + 0.5f);
        b = bgra.x / 255.0f;
        g = bgra.y / 255.0f;
        r = bgra.z / 255.0f;
    }

    output[dst] = r;
    output[plane + dst] = g;
    output[2 * plane + dst] = b;
}

} // namespace

cudaError_t launchDxgiPreprocess(
    cudaArray_t source,
    int sourceW,
    int sourceH,
    LetterboxInfo letterbox,
    float* output,
    cudaStream_t stream) {
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = source;

    cudaTextureDesc textureDesc{};
    textureDesc.addressMode[0] = cudaAddressModeClamp;
    textureDesc.addressMode[1] = cudaAddressModeClamp;
    textureDesc.filterMode = cudaFilterModePoint;
    textureDesc.readMode = cudaReadModeElementType;
    textureDesc.normalizedCoords = 0;

    cudaTextureObject_t texture{};
    cudaError_t status = cudaCreateTextureObject(&texture, &resource, &textureDesc, nullptr);
    if (status != cudaSuccess) {
        return status;
    }

    dim3 block(16, 16);
    dim3 grid((kInputW + block.x - 1) / block.x, (kInputH + block.y - 1) / block.y);
    preprocessKernel<<<grid, block, 0, stream>>>(texture, sourceW, sourceH, letterbox, output);
    status = cudaGetLastError();

    cudaError_t destroyStatus = cudaDestroyTextureObject(texture);
    return status == cudaSuccess ? destroyStatus : status;
}
