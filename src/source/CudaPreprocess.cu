#include "CudaPreprocess.h"

#include <cuda_runtime.h>

namespace {

/**
 * @brief 将BGRA CUDA纹理预处理为RGB CHW浮点输入张量。
 * @param texture CUDA纹理对象，来源为DXGI ROI。
 * @param sourceW 源ROI宽度。
 * @param sourceH 源ROI高度。
 * @param inputW 模型输入宽度。
 * @param inputH 模型输入高度。
 * @param letterbox letterbox缩放和填充信息。
 * @param output GPU输出张量指针。
 * @note 无返回值。
 */
__global__ void preprocessKernel(
    cudaTextureObject_t texture,
    int sourceW,
    int sourceH,
    int inputW,
    int inputH,
    LetterboxInfo letterbox,
    float* output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= inputW || y >= inputH) {
        return;
    }

    int dst = y * inputW + x;
    int plane = inputW * inputH;
    float r = 114.0f / 255.0f;
    float g = 114.0f / 255.0f;
    float b = 114.0f / 255.0f;

    int localX = x - letterbox.padX;
    int localY = y - letterbox.padY;
    if (localX >= 0 && localY >= 0 &&
        localX < letterbox.resizedW && localY < letterbox.resizedH) {
        int srcX = static_cast<int>(localX / letterbox.scale);
        int srcY = static_cast<int>(localY / letterbox.scale);
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
    int inputW,
    int inputH,
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
    dim3 grid((inputW + block.x - 1) / block.x, (inputH + block.y - 1) / block.y);
    preprocessKernel<<<grid, block, 0, stream>>>(texture, sourceW, sourceH, inputW, inputH, letterbox, output);
    status = cudaGetLastError();

    cudaError_t destroyStatus = cudaDestroyTextureObject(texture);
    return status == cudaSuccess ? destroyStatus : status;
}
