#include "DxgiScreenCapture.h"

#include "AppConfig.h"
#include "CudaPreprocess.h"

#include <cuda_d3d11_interop.h>
#include <d3d11.h>
#include <dxgi1_2.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <string>

using Microsoft::WRL::ComPtr;

namespace {

void checkHr(HRESULT hr, char const* what) {
    if (FAILED(hr)) {
        throw std::runtime_error(std::string(what) + " failed, HRESULT=0x" +
                                 std::to_string(static_cast<unsigned long>(hr)));
    }
}

void checkCuda(cudaError_t status, char const* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

LetterboxInfo makeLetterbox(int screenW, int screenH) {
    auto const& capture = AppConfig::instance().capture();
    int captureW = std::min(capture.roiWidth, screenW);
    int captureH = std::min(capture.roiHeight, screenH);

    LetterboxInfo info{};
    info.screenW = screenW;
    info.screenH = screenH;
    info.captureX = (screenW - captureW) / 2;
    info.captureY = (screenH - captureH) / 2;
    info.captureW = captureW;
    info.captureH = captureH;
    info.scale = std::min(
        static_cast<float>(kInputW) / static_cast<float>(captureW),
        static_cast<float>(kInputH) / static_cast<float>(captureH));
    info.resizedW = std::max(1, static_cast<int>(std::round(captureW * info.scale)));
    info.resizedH = std::max(1, static_cast<int>(std::round(captureH * info.scale)));
    info.padX = (kInputW - info.resizedW) / 2;
    info.padY = (kInputH - info.resizedH) / 2;
    return info;
}

} // namespace

DxgiScreenCapture::DxgiScreenCapture() {
    initD3d();
    initDuplication();
    initCudaTexture();
}

DxgiScreenCapture::~DxgiScreenCapture() {
    if (cudaResource_) {
        cudaGraphicsUnregisterResource(cudaResource_);
    }
}

bool DxgiScreenCapture::captureToDevice(float* deviceInput, cudaStream_t stream) {
    if (!duplication_ || !cudaResource_) {
        return recreateDuplication();
    }

    DXGI_OUTDUPL_FRAME_INFO frameInfo{};
    ComPtr<IDXGIResource> frameResource;
    HRESULT hr = duplication_->AcquireNextFrame(0, &frameInfo, &frameResource);
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
        return false;
    }
    if (hr == DXGI_ERROR_ACCESS_LOST) {
        return recreateDuplication();
    }
    checkHr(hr, "IDXGIOutputDuplication::AcquireNextFrame");

    ComPtr<ID3D11Texture2D> frameTexture;
    checkHr(frameResource.As(&frameTexture), "Query desktop frame texture");

    D3D11_BOX roi{};
    roi.left = static_cast<UINT>(letterbox_.captureX);
    roi.top = static_cast<UINT>(letterbox_.captureY);
    roi.front = 0;
    roi.right = static_cast<UINT>(letterbox_.captureX + letterbox_.captureW);
    roi.bottom = static_cast<UINT>(letterbox_.captureY + letterbox_.captureH);
    roi.back = 1;
    context_->CopySubresourceRegion(cudaTexture_.Get(), 0, 0, 0, 0, frameTexture.Get(), 0, &roi);
    checkHr(duplication_->ReleaseFrame(), "IDXGIOutputDuplication::ReleaseFrame");

    checkCuda(cudaGraphicsMapResources(1, &cudaResource_, stream), "cudaGraphicsMapResources");
    cudaArray_t source{};
    cudaError_t status = cudaGraphicsSubResourceGetMappedArray(&source, cudaResource_, 0, 0);
    if (status == cudaSuccess) {
        status = launchDxgiPreprocess(
            source,
            letterbox_.captureW,
            letterbox_.captureH,
            letterbox_,
            deviceInput,
            stream);
    }
    cudaError_t unmapStatus = cudaGraphicsUnmapResources(1, &cudaResource_, stream);
    checkCuda(status, "launchDxgiPreprocess");
    checkCuda(unmapStatus, "cudaGraphicsUnmapResources");
    return true;
}

void DxgiScreenCapture::initD3d() {
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

    D3D_FEATURE_LEVEL levels[] = {
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
    };
    D3D_FEATURE_LEVEL selected{};
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        flags,
        levels,
        static_cast<UINT>(std::size(levels)),
        D3D11_SDK_VERSION,
        &device_,
        &selected,
        &context_);
    checkHr(hr, "D3D11CreateDevice");
}

void DxgiScreenCapture::initDuplication() {
    ComPtr<IDXGIDevice> dxgiDevice;
    checkHr(device_.As(&dxgiDevice), "Query IDXGIDevice");

    ComPtr<IDXGIAdapter> adapter;
    checkHr(dxgiDevice->GetAdapter(&adapter), "IDXGIDevice::GetAdapter");

    ComPtr<IDXGIOutput> output;
    checkHr(adapter->EnumOutputs(static_cast<UINT>(AppConfig::instance().capture().outputIndex), &output),
            "IDXGIAdapter::EnumOutputs");

    DXGI_OUTPUT_DESC desc{};
    checkHr(output->GetDesc(&desc), "IDXGIOutput::GetDesc");
    screenW_ = desc.DesktopCoordinates.right - desc.DesktopCoordinates.left;
    screenH_ = desc.DesktopCoordinates.bottom - desc.DesktopCoordinates.top;
    letterbox_ = makeLetterbox(screenW_, screenH_);

    ComPtr<IDXGIOutput1> output1;
    checkHr(output.As(&output1), "Query IDXGIOutput1");
    checkHr(output1->DuplicateOutput(device_.Get(), &duplication_), "IDXGIOutput1::DuplicateOutput");
}

void DxgiScreenCapture::initCudaTexture() {
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = static_cast<UINT>(letterbox_.captureW);
    desc.Height = static_cast<UINT>(letterbox_.captureH);
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    checkHr(device_->CreateTexture2D(&desc, nullptr, &cudaTexture_), "ID3D11Device::CreateTexture2D");
    checkCuda(cudaGraphicsD3D11RegisterResource(&cudaResource_, cudaTexture_.Get(), cudaGraphicsRegisterFlagsNone),
              "cudaGraphicsD3D11RegisterResource");
}

bool DxgiScreenCapture::recreateDuplication() {
    try {
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        if (cudaResource_) {
            checkCuda(cudaGraphicsUnregisterResource(cudaResource_), "cudaGraphicsUnregisterResource");
            cudaResource_ = nullptr;
        }
        cudaTexture_.Reset();
        duplication_.Reset();

        initDuplication();
        initCudaTexture();
        return false;
    } catch (...) {
        duplication_.Reset();
        cudaTexture_.Reset();
        cudaResource_ = nullptr;
        Sleep(50);
        return false;
    }
}
