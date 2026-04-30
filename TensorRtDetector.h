#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class TrtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* msg) noexcept override;
};

class TensorRtDetector {
public:
    TensorRtDetector(std::filesystem::path const& onnxPath, std::filesystem::path const& cachePath);
    ~TensorRtDetector();

    TensorRtDetector(TensorRtDetector const&) = delete;
    TensorRtDetector& operator=(TensorRtDetector const&) = delete;

    std::vector<float> infer(float const* input);
    std::vector<float> inferDeviceInput();
    float* deviceInput() const { return static_cast<float*>(deviceInput_.get()); }
    cudaStream_t stream() const { return stream_; }
    std::string const& inputName() const { return inputName_; }
    std::string const& outputName() const { return outputName_; }

private:
    struct CudaDeleter {
        void operator()(void* ptr) const;
    };

    using CudaPtr = std::unique_ptr<void, CudaDeleter>;

    void buildOrLoadEngine(std::filesystem::path const& onnxPath, std::filesystem::path const& cachePath);
    void discoverTensors();

    TrtLogger logger_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string inputName_;
    std::string outputName_;
    CudaPtr deviceInput_;
    CudaPtr deviceOutput_;
    cudaStream_t stream_{};
    size_t inputBytes_{};
    size_t outputBytes_{};
    size_t outputCount_{};
    std::vector<float> output_;
};
