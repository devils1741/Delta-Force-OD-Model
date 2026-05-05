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

/**
 * @brief 基于DXGI Desktop Duplication的GPU屏幕采集器。
 *
 * 将屏幕中心ROI复制到D3D11纹理，并通过CUDA interop预处理为模型输入。
 */
class DxgiScreenCapture {
public:
    /**
     * @brief 创建DXGI屏幕采集器。
     * @param inputW 模型输入宽度。
     * @param inputH 模型输入高度。
     * @note 无返回值。
     */
    DxgiScreenCapture(int inputW, int inputH);
    /**
     * @brief 释放DXGI和CUDA interop资源。
     * @note 无返回值。
     */
    ~DxgiScreenCapture();

    /**
     * @brief 禁止复制构造采集器。
     * @param other 另一个采集器对象。
     * @note 无返回值。
     */
    DxgiScreenCapture(DxgiScreenCapture const& other) = delete;
    /**
     * @brief 禁止复制赋值采集器。
     * @param other 另一个采集器对象。
     * @return 当前对象引用。
     */
    DxgiScreenCapture& operator=(DxgiScreenCapture const& other) = delete;

    /**
     * @brief 采集一帧屏幕并写入GPU输入张量。
     * @param deviceInput GPU上的模型输入张量指针。
     * @param stream 用于CUDA预处理的stream。
     * @return 捕获到新帧并完成预处理时返回true；无新帧或重建采集资源时返回false。
     */
    bool captureToDevice(float* deviceInput, cudaStream_t stream);

    /**
     * @brief 获取屏幕宽度。
     * @return 屏幕宽度，单位为像素。
     */
    int screenW() const { return screenW_; }
    /**
     * @brief 获取屏幕高度。
     * @return 屏幕高度，单位为像素。
     */
    int screenH() const { return screenH_; }
    /**
     * @brief 获取当前letterbox参数。
     * @return letterbox信息的只读引用。
     */
    LetterboxInfo const& letterbox() const { return letterbox_; }

private:
    /**
     * @brief 初始化D3D11设备和上下文。
     * @note 无返回值。
     */
    void initD3d();
    /**
     * @brief 初始化DXGI输出复制接口并计算采集ROI。
     * @note 无返回值。
     */
    void initDuplication();
    /**
     * @brief 创建可被CUDA映射的D3D11纹理。
     * @note 无返回值。
     */
    void initCudaTexture();
    /**
     * @brief 在桌面复制失效后重建DXGI和CUDA资源。
     * @return 当前调用不产生可推理帧，固定返回false。
     */
    bool recreateDuplication();

    int screenW_{};
    int screenH_{};
    int inputW_{};
    int inputH_{};
    LetterboxInfo letterbox_{};
    Microsoft::WRL::ComPtr<ID3D11Device> device_;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context_;
    Microsoft::WRL::ComPtr<IDXGIOutputDuplication> duplication_;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> cudaTexture_;
    cudaGraphicsResource* cudaResource_{};
};
