#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

/**
 * @brief TensorRT 日志适配器。
 *
 * 将 TensorRT 警告及以上级别日志写入标准错误流。
 */
class TrtLogger final : public nvinfer1::ILogger {
public:
    /**
     * @brief 处理 TensorRT 日志消息。
     * @param severity 日志严重级别。
     * @param msg TensorRT 输出的日志文本。
     * @note 无返回值。
     */
    void log(Severity severity, char const* msg) noexcept override;
};

/**
 * @brief TensorRT 目标检测推理器。
 *
 * 负责构建或加载 TensorRT engine、管理 CUDA 缓冲区并执行推理。
 */
class TensorRtDetector {
public:
    /**
     * @brief 创建 TensorRT 检测器。
     * @param onnxPath ONNX 模型路径。
     * @param cachePath TensorRT engine 缓存路径。
     * @note 无返回值。
     */
    TensorRtDetector(std::filesystem::path const& onnxPath, std::filesystem::path const& cachePath);
    /**
     * @brief 销毁检测器并释放 CUDA stream。
     * @note 无返回值。
     */
    ~TensorRtDetector();

    /**
     * @brief 禁止复制构造检测器。
     * @param other 另一个检测器对象。
     * @note 无返回值。
     */
    TensorRtDetector(TensorRtDetector const& other) = delete;
    /**
     * @brief 禁止复制赋值检测器。
     * @param other 另一个检测器对象。
     * @return 当前对象引用。
     */
    TensorRtDetector& operator=(TensorRtDetector const& other) = delete;

    /**
     * @brief 使用主机端输入张量执行一次推理。
     * @param input CPU 内存中的 RGB CHW 浮点输入张量。
     * @return TensorRT 输出张量数据。
     */
    std::vector<float> infer(float const* input);
    /**
     * @brief 使用已经写入 GPU 的输入张量执行一次推理。
     * @return TensorRT 输出张量数据。
     */
    std::vector<float> inferDeviceInput();
    /**
     * @brief 获取 GPU 输入缓冲区指针。
     * @return GPU 输入张量指针。
     */
    float* deviceInput() const { return static_cast<float*>(deviceInput_.get()); }
    /**
     * @brief 获取检测器使用的 CUDA stream。
     * @return CUDA stream 句柄。
     */
    cudaStream_t stream() const { return stream_; }
    /**
     * @brief 获取 TensorRT 输入张量名称。
     * @return 输入张量名称的只读引用。
     */
    std::string const& inputName() const { return inputName_; }
    /**
     * @brief 获取 TensorRT 输出张量名称。
     * @return 输出张量名称的只读引用。
     */
    std::string const& outputName() const { return outputName_; }
    /**
     * @brief 获取模型输入宽度。
     * @return 输入宽度，单位为像素。
     */
    int inputW() const { return inputW_; }
    /**
     * @brief 获取模型输入高度。
     * @return 输入高度，单位为像素。
     */
    int inputH() const { return inputH_; }

private:
    /**
     * @brief CUDA 设备内存释放器。
     */
    struct CudaDeleter {
        /**
         * @brief 释放 CUDA 设备内存。
         * @param ptr 需要释放的设备内存指针。
         * @note 无返回值。
         */
        void operator()(void* ptr) const;
    };

    using CudaPtr = std::unique_ptr<void, CudaDeleter>;

    /**
     * @brief 从缓存加载或从 ONNX 构建 TensorRT engine。
     * @param onnxPath ONNX 模型路径。
     * @param cachePath engine 缓存路径。
     * @note 无返回值。
     */
    void buildOrLoadEngine(std::filesystem::path const& onnxPath, std::filesystem::path const& cachePath);
    /**
     * @brief 发现 TensorRT engine 的输入和输出张量名称。
     * @note 无返回值。
     */
    void discoverTensors();
    /**
     * @brief 读取 TensorRT 张量形状并分配输出容量。
     * @note 无返回值。
     */
    void discoverTensorShapes();

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string inputName_;
    std::string outputName_;
    int inputW_{};
    int inputH_{};
    CudaPtr deviceInput_;
    CudaPtr deviceOutput_;
    cudaStream_t stream_{};
    size_t inputBytes_{};
    size_t outputBytes_{};
    size_t outputCount_{};
    std::vector<float> output_;
};
