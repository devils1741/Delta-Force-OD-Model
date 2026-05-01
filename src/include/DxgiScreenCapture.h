#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include "Detection.h"

#include <cuda_runtime_api.h>
#include <wrl/client.h>

struct IDXGIOutputDuplication;
struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11Texture2D;
struct cudaGraphicsResource;

class DxgiScreenCapture {
public:
    DxgiScreenCapture();
    ~DxgiScreenCapture();

    DxgiScreenCapture(DxgiScreenCapture const&) = delete;
    DxgiScreenCapture& operator=(DxgiScreenCapture const&) = delete;

    bool captureToDevice(float* deviceInput, cudaStream_t stream);

    int screenW() const { return screenW_; }
    int screenH() const { return screenH_; }
    LetterboxInfo const& letterbox() const { return letterbox_; }

private:
    void initD3d();
    void initDuplication();
    void initCudaTexture();
    bool recreateDuplication();

    int screenW_{};
    int screenH_{};
    LetterboxInfo letterbox_{};
    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> cudaTexture_;
    cudaGraphicsResource* cudaResource_{};
};
