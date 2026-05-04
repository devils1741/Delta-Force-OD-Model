#pragma once

#include "Detection.h"

#include <cuda_runtime_api.h>

cudaError_t launchDxgiPreprocess(
    cudaArray_t source,
    int sourceW,
    int sourceH,
    int inputW,
    int inputH,
    LetterboxInfo letterbox,
    float* output,
    cudaStream_t stream);
