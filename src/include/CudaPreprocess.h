#pragma once

#include "Detection.h"

#include <cuda_runtime_api.h>

/**
 * @brief 启动DXGI纹理到模型输入张量的CUDA预处理。
 * @param source 已映射的CUDA数组，内容为BGRA屏幕ROI。
 * @param sourceW 源ROI宽度。
 * @param sourceH 源ROI高度。
 * @param inputW 模型输入宽度。
 * @param inputH 模型输入高度。
 * @param letterbox ROI到模型输入的letterbox参数。
 * @param output 输出到GPU的RGB CHW浮点张量指针。
 * @param stream 执行kernel的CUDA stream。
 * @return CUDA 调用状态，cudaSuccess表示成功。
 */
cudaError_t launchDxgiPreprocess(
    cudaArray_t source,
    int sourceW,
    int sourceH,
    int inputW,
    int inputH,
    LetterboxInfo letterbox,
    float* output,
    cudaStream_t stream);
