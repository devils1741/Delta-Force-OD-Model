# Delta TensorRT Screen Detector

一个基于 Windows DXGI + CUDA + TensorRT 的实时屏幕目标检测 Demo。程序会捕获屏幕中心区域，将画面在 GPU 上预处理后送入 TensorRT 推理，并用透明置顶窗口在屏幕上绘制检测框。

## 效果展示

![检测效果 1](assert/2026-04-30%2011-45-07_03m17s-03m19s.gif)

![检测效果 2](assert/2026-04-30%2011-45-07_03m21s-03m23s.gif)

## 当前功能

- 使用 DXGI Desktop Duplication 捕获屏幕画面，并只复制中心 ROI 到 CUDA texture
- 使用 CUDA 在 GPU 上完成 ROI letterbox、BGRA 转 RGB、归一化和 CHW 排布
- 使用 TensorRT FP16 engine 进行推理
- 检测框通过透明 overlay 窗口绘制到屏幕上
- 运行参数集中放在 `config/config.yaml`
- 默认输入尺寸为 `640x640`
- 默认捕获屏幕中心 `1600x900` ROI
- 默认检测频率上限为 `60 FPS`

## 推理流程

```text
DXGI screen capture
  -> D3D11 copy center ROI
  -> CUDA preprocess ROI to 640x640 tensor
  -> TensorRT FP16 inference
  -> CPU decode + NMS
  -> transparent overlay draw
```

## 项目结构

```text
delta/
  CMakeLists.txt
  README.md
  config/
    config.yaml
  src/
    main.cpp
    include/
    source/
  weights/
  assert/
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

默认模型路径在 [config/config.yaml](config/config.yaml) 中配置：

```text
weights/best.onnx
```

首次运行会生成 TensorRT engine 缓存：

```text
weights/best_640_trt10_16_sm89_fp16.engine
```

如果替换模型、输入尺寸或 TensorRT/CUDA 环境变化，建议删除旧 engine，让程序重新构建。

## 主要参数

大部分运行参数在 [config/config.yaml](config/config.yaml) 中配置：

```yaml
model:
  onnx_path: weights/best.onnx
  engine_path: weights/best_640_trt10_16_sm89_fp16.engine

inference:
  target_fps: 60
  score_threshold: 0.30
  nms_threshold: 0.45
  max_detections: 300

capture:
  output_index: 0
  roi_width: 1600
  roi_height: 900

tensorrt:
  fp16: true
  workspace_mb: 1024
```

配置文件中的相对路径按项目根目录解析，所以 `weights/best.onnx` 会指向项目根目录下的 `weights/`。

输入尺寸仍是编译期常量，位置在 [src/include/Detection.h](src/include/Detection.h)：

```cpp
constexpr int kInputW = 640;
constexpr int kInputH = 640;
```

如果修改输入尺寸，需要同步重新构建 TensorRT engine。建议删除旧 engine 缓存后再运行。

## 构建

```powershell
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug --config Debug
```

## 运行建议

- 游戏建议使用无边框窗口或窗口化全屏，DXGI 捕获更稳定。
- 如果游戏帧数下降明显，可以降低 `inference.target_fps`，例如改为 `45` 或 `30`。
- 如果远处小目标漏检，可以尝试减小 `capture.roi_width` 和 `capture.roi_height`，让目标在输入中占比更大。
- 如果替换模型或 TensorRT/CUDA 环境变化，建议删除旧 engine，让程序重新构建。
