# Delta TensorRT Screen Detector

一个基于 Windows DXGI + CUDA + TensorRT 的实时屏幕目标检测 Demo。程序会捕获屏幕中心区域，将画面在 GPU 上预处理后送入 TensorRT 推理，并用透明置顶窗口在屏幕上绘制检测框。

## 效果展示

![检测效果 1](assert/2026-04-30%2011-45-07_03m17s-03m19s.gif)

![检测效果 2](assert/2026-04-30%2011-45-07_03m21s-03m23s.gif)

## 当前功能

- 使用 DXGI Desktop Duplication 捕获屏幕画面
- 使用 CUDA 在 GPU 上完成 ROI 采样、letterbox、BGRA 转 RGB、归一化和 CHW 排布
- 使用 TensorRT FP16 engine 进行推理
- 检测框通过透明 overlay 窗口绘制到屏幕上
- 默认输入尺寸为 `640x640`
- 默认捕获屏幕中心 `1600x900` ROI
- 默认检测频率上限为 `60 FPS`

## 推理流程

```text
DXGI screen capture
  -> CUDA preprocess
  -> TensorRT FP16 inference
  -> CPU decode + NMS
  -> transparent overlay draw
```

## 依赖

- Windows10/11
- CUDA Toolkit
- TensorRT 10.x
- CMake
- MSVC / Visual Studio Build Tools

项目默认 TensorRT 路径在 [CMakeLists.txt](CMakeLists.txt) 中配置：

```cmake
set(TENSORRT_ROOT "C:/Program Files/TensorRT-10.16.1.11")
```

如本机路径不同，需要修改该变量。

## 模型文件

默认查找：

```text
weights/best.onnx
```

首次运行会生成 TensorRT engine 缓存：

```text
weights/best_640_trt10_16_sm89_fp16.engine
```

如果替换模型、输入尺寸或 TensorRT/CUDA 环境变化，建议删除旧 engine，让程序重新构建。

## 主要参数

输入尺寸在 [Detection.h](Detection.h)：

```cpp
constexpr int kInputW = 640;
constexpr int kInputH = 640;
```

检测阈值：

```cpp
constexpr float kScoreThreshold = 0.30f;
constexpr float kNmsThreshold = 0.45f;
```

检测频率上限在 [main.cpp](main.cpp)：

```cpp
constexpr int kTargetInferenceFps = 60;
```

中心 ROI 尺寸在 [DxgiScreenCapture.cpp](DxgiScreenCapture.cpp)：

```cpp
int captureW = std::min(1600, screenW);
int captureH = std::min(900, screenH);
```

## 运行建议

- 游戏建议使用无边框窗口或窗口化全屏，DXGI 捕获更稳定。
- 如果游戏帧数下降明显，可以降低 `kTargetInferenceFps`，例如改为 `45` 或 `30`。
- 如果远处小目标漏检，可以尝试减小 ROI，让目标在输入中占比更大。
