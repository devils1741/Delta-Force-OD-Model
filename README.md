# Delta TensorRT Screen Detector

一个基于 Windows DXGI + CUDA + TensorRT 的实时屏幕目标检测 Demo。程序会捕获屏幕中心区域，将画面在 GPU 上预处理后送入 TensorRT 推理，并用透明置顶窗口在屏幕上绘制检测框。

## 效果展示

![检测效果 1](assert/001.gif)


## 当前功能

- 使用DXGI Desktop Duplication 捕获屏幕画面，并只复制中心ROI到CUDA texture
- 使用CUDA在GPU上完成ROIletterbox、BGRA转RGB、归一化和CHW排布
- 使用TensorRT FP16 engine进行推理
- 检测框通过透明overlay窗口绘制到屏幕上
- 运行参数集中放在`config/config.yaml`

## 推理流程

```text
DXGI screen capture
  -> D3D11 copy center ROI
  -> CUDA preprocess ROI to tensor
  -> TensorRT FP16 inference
  -> CUDA decode end-to-end detections
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


## 主要参数

大部分运行参数在 [config/config.yaml](config/config.yaml) 中配置.
```yaml
model:
  # ONNX 模型文件路径，相对项目根目录解析
  onnx_path: weights/best_640.onnx

  # TensorRT engine 缓存路径；首次运行会由 ONNX 构建生成
  engine_path: weights/best_640.engine 

  # 模型输入宽度，必须与 ONNX / engine 的输入尺寸一致
  input_width: 640

  # 模型输入高度，必须与 ONNX / engine 的输入尺寸一致
  input_height: 640

inference:
  # 推理目标帧率上限
  target_fps: 120

  # 置信度阈值，低于该分数的检测结果会被过滤
  score_threshold: 0.30

  # 每帧最多保留的候选检测数量
  max_detections: 300

capture:
  # 采集的显示器索引，0 表示主显示器
  output_index: 0

  # 屏幕中心 ROI 采集宽度
  roi_width: 1600

  # 屏幕中心 ROI 采集高度
  roi_height: 900

tensorrt:
  # 是否启用 FP16 构建 TensorRT engine
  fp16: true

  # TensorRT 构建 engine 时可使用的 workspace 大小，单位 MB
  workspace_mb: 1024

```

## 编译
首次编译时间会比较长
```powershell
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug --config Debug
```

## 运行建议

- 游戏建议使用无边框窗口或窗口化全屏，DXGI 捕获更稳定。
- 如果游戏帧数下降明显，可以降低 `inference.target_fps`，例如改为 `45` 或 `30`。
- 如果远处小目标漏检，可以尝试减小 `capture.roi_width` 和 `capture.roi_height`，让目标在输入中占比更大。
- 如果替换模型或 TensorRT/CUDA 环境变化，建议删除旧 engine，让程序重新构建。
