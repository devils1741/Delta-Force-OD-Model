#include "TensorRtDetector.h"

#include "AppConfig.h"
#include "Detection.h"

#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

template <typename T>
using TrtUniquePtr = std::unique_ptr<T>;

/**
 * @brief 检查CUDA状态并在失败时抛出异常。
 * @param status CUDA API返回状态。
 * @param what 当前操作名称。
 * @note 无返回值。
 */
void checkCuda(cudaError_t status, char const* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

/**
 * @brief 以二进制方式读取整个文件。
 * @param path 文件路径。
 * @return 文件内容字节数组。
 */
std::vector<char> readFile(fs::path const& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

/**
 * @brief 以二进制方式写入文件。
 * @param path 文件路径。
 * @param data 需要写入的数据指针。
 * @param size 需要写入的字节数。
 * @note 无返回值。
 */
void writeFile(fs::path const& path, void const* data, size_t size) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot write file: " + path.string());
    }
    file.write(static_cast<char const*>(data), static_cast<std::streamsize>(size));
}

/**
 * @brief 计算TensorRT维度中的元素总数。
 * @param dims TensorRT张量维度。
 * @return 张量元素数量。
 */
size_t volume(nvinfer1::Dims const& dims) {
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            throw std::runtime_error("Dynamic TensorRT tensor shapes are not configured");
        }
        count *= static_cast<size_t>(dims.d[i]);
    }
    return count;
}

} // namespace

void TrtLogger::log(Severity severity, char const* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TensorRT] " << msg << '\n';
    }
}

void TensorRtDetector::CudaDeleter::operator()(void* ptr) const {
    if (ptr) {
        cudaFree(ptr);
    }
}

TensorRtDetector::TensorRtDetector(std::filesystem::path const& onnxPath, std::filesystem::path const& cachePath) {
    initLibNvInferPlugins(&logger_, "");
    buildOrLoadEngine(onnxPath, cachePath);
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    discoverTensors();
    discoverTensorShapes();

    inputBytes_ = 3ULL * static_cast<size_t>(inputW_) * static_cast<size_t>(inputH_) * sizeof(float);
    outputBytes_ = outputCount_ * sizeof(float);
    output_.resize(outputCount_);

    void* input{};
    void* output{};
    checkCuda(cudaMalloc(&input, inputBytes_), "cudaMalloc input");
    checkCuda(cudaMalloc(&output, outputBytes_), "cudaMalloc output");
    deviceInput_.reset(input);
    deviceOutput_.reset(output);
    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    if (!context_->setTensorAddress(inputName_.c_str(), deviceInput_.get()) ||
        !context_->setTensorAddress(outputName_.c_str(), deviceOutput_.get())) {
        throw std::runtime_error("Failed to set TensorRT tensor addresses");
    }
}

TensorRtDetector::~TensorRtDetector() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

std::vector<float> TensorRtDetector::infer(float const* input) {
    checkCuda(cudaMemcpyAsync(deviceInput_.get(), input, inputBytes_, cudaMemcpyHostToDevice, stream_),
              "cudaMemcpyAsync H2D");
    return inferDeviceInput();
}

std::vector<float> TensorRtDetector::inferDeviceInput() {
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed");
    }
    checkCuda(cudaMemcpyAsync(output_.data(), deviceOutput_.get(), outputBytes_, cudaMemcpyDeviceToHost, stream_),
              "cudaMemcpyAsync D2H");
    checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    return output_;
}

void TensorRtDetector::buildOrLoadEngine(fs::path const& onnxPath, fs::path const& cachePath) {
    TrtUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger_));
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    if (fs::exists(cachePath) && fs::last_write_time(cachePath) >= fs::last_write_time(onnxPath)) {
        auto bytes = readFile(cachePath);
        if (auto* engine = runtime->deserializeCudaEngine(bytes.data(), bytes.size())) {
            std::cout << "Loaded TensorRT engine cache: " << cachePath << '\n';
            engine_.reset(engine);
            return;
        }
        std::cerr << "Engine cache is incompatible, rebuilding from ONNX.\n";
    } else if (fs::exists(cachePath)) {
        std::cerr << "Engine cache is older than ONNX, rebuilding from ONNX.\n";
    }

    std::cout << "Building TensorRT engine from ONNX. First launch may take a while...\n";
    TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
    if (!builder) {
        throw std::runtime_error("Failed to create TensorRT builder");
    }

    TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
    if (!network) {
        throw std::runtime_error("Failed to create TensorRT network");
    }

    TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger_));
    if (!parser) {
        throw std::runtime_error("Failed to create ONNX parser");
    }
    if (!parser->parseFromFile(onnxPath.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << '\n';
        }
        throw std::runtime_error("Failed to parse ONNX: " + onnxPath.string());
    }

    TrtUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) {
        throw std::runtime_error("Failed to create TensorRT builder config");
    }
    auto const& trt = AppConfig::instance().tensorrt();
    config->setMemoryPoolLimit(
        nvinfer1::MemoryPoolType::kWORKSPACE,
        static_cast<size_t>(trt.workspaceMb) * 1024ULL * 1024ULL);
    if (trt.fp16) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    TrtUniquePtr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
        throw std::runtime_error("Failed to build TensorRT engine");
    }
    writeFile(cachePath, serialized->data(), serialized->size());
    std::cout << "Saved TensorRT engine cache: " << cachePath << '\n';

    auto* engine = runtime->deserializeCudaEngine(serialized->data(), serialized->size());
    if (!engine) {
        throw std::runtime_error("Failed to deserialize freshly built engine");
    }
    engine_.reset(engine);
}

void TensorRtDetector::discoverTensors() {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        char const* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            inputName_ = name;
        } else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName_ = name;
        }
    }
    if (inputName_.empty() || outputName_.empty()) {
        throw std::runtime_error("Could not find TensorRT input/output tensors");
    }
    std::cout << "Input tensor: " << inputName_ << '\n';
    std::cout << "Output tensor: " << outputName_ << '\n';
}

void TensorRtDetector::discoverTensorShapes() {
    auto inputDims = engine_->getTensorShape(inputName_.c_str());
    auto outputDims = engine_->getTensorShape(outputName_.c_str());

    if (inputDims.nbDims != 4 || inputDims.d[0] != 1 || inputDims.d[1] != 3) {
        throw std::runtime_error("Expected input shape [1, 3, H, W]");
    }

    inputH_ = static_cast<int>(inputDims.d[2]);
    inputW_ = static_cast<int>(inputDims.d[3]);
    outputCount_ = volume(outputDims);

    auto const& model = AppConfig::instance().model();
    if (model.inputWidth != inputW_ || model.inputHeight != inputH_) {
        throw std::runtime_error(
            "Configured model input size does not match TensorRT engine. config=" +
            std::to_string(model.inputWidth) + "x" + std::to_string(model.inputHeight) +
            ", engine=" + std::to_string(inputW_) + "x" + std::to_string(inputH_));
    }

    std::cout << "Input shape: [";
    for (int i = 0; i < inputDims.nbDims; ++i) {
        std::cout << (i ? ", " : "") << inputDims.d[i];
    }
    std::cout << "]\n";

    std::cout << "Output shape: [";
    for (int i = 0; i < outputDims.nbDims; ++i) {
        std::cout << (i ? ", " : "") << outputDims.d[i];
    }
    std::cout << "]\n";
}
