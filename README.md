# Delta TensorRT Screen Detector

一个基于 Windows DXGI + CUDA + TensorRT 的实时屏幕目标检测 Demo。程序会捕获屏幕中心区域，将画面在 GPU 上预处理后送入 TensorRT 推理，并用透明置顶窗口在屏幕上绘制检测框。

## 效果展示

![检测效果 1](assert/001.gif)


## 当前功能

- 使用DXGI Desktop Duplication 捕获屏幕画面，并只复制中心ROI到CUDA texture
- 使用CUDA在GPU上完成ROIletterbox、BGRA转RGB、归一化和CHW排布
- 使用TensorRT FP16 engine进行推理
- 检测框、置信度和 ROI 边框由 D3D11 着色器直接绘制，通过 DirectComposition 透明置顶显示
- 图像和绘框数据不回传 CPU；鼠标控制在独立线程异步接收检测结果快照
- 覆盖窗口支持跨应用鼠标穿透，并排除在屏幕采集中
- 运行参数集中放在`config/config.yaml`

## 推理流程

```text
DXGI screen capture
  -> D3D11 copy center ROI
  -> CUDA preprocess ROI to tensor
  -> TensorRT FP16 inference
  -> CUDA decode + NMS + small-target filter
  -> CUDA/D3D11 shared box buffer
  -> D3D11 GPU overlay + DirectComposition
       \-> independent GPU snapshot -> async pinned-memory D2H -> CPU mouse control
```

## 依赖

- Windows 10 2004+ / Windows 11
- CUDA Toolkit
- TensorRT 11.x（当前默认链接配置）
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

## GPU 绘制与异步鼠标控制

主循环顺序提交捕获、预处理、推理和绘制，没有 CPU 帧队列。CUDA/D3D11
映射与解除映射负责资源所有权和 GPU 执行顺序；这不代表驱动调用或显示合成完全没有等待。
每份检测结果都会尝试绘制，旧的 `overlay_interval_frames` 配置不再使用。

鼠标功能使用三个独立 GPU 快照槽、固定页主机内存和非阻塞 CUDA 传输流。
快照先在推理流上生成，传输流通过 CUDA event 等待快照完成，再异步回传。
CPU 工作线程用 `cudaEventQuery` 检查完成状态，只使用最新已完成结果；全部槽位忙时
跳过鼠标回传，不阻塞绘框。超过 100 ms 的控制结果会丢弃。
当前保留 CPU 目标锁定逻辑，因此回传的是最多 `max_detections` 个框及数量，
不是图像或完整模型输出。`mouse.enabled: false` 时不分配回传资源，也不安装鼠标钩子。

GPU 图像共享要求采集显示器所属显卡支持 CUDA，且与推理、渲染使用同一张显卡。
跨显卡场景不会静默退回 CPU 图像传输。

验证命令（项目根目录）：

```powershell
cmake --build build --config Release --parallel 4
ctest --test-dir build -C Release --output-on-failure
.\build\Release\delta.exe --smoke-test
```

`gpu_pipeline_test` 检查 GPU 解码/NMS/尺寸过滤、共享缓冲区、透明绘框和置信度、
空结果清屏、跨线程窗口鼠标穿透，以及慢 CPU 消费者下的异步快照一致性。
只有测试版本会把渲染像素读回 CPU 作断言。`--smoke-test` 使用实际截图和模型运行
20 帧后退出，并强制关闭鼠标控制。正常运行时可按 Esc 退出。

接口说明：[CUDA 图形互操作同步](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html)、
[Windows 分层窗口鼠标穿透](https://learn.microsoft.com/en-us/windows/win32/winmsg/window-features#layered-windows)。

## 鼠标移动

当前默认使用 `mouse.mode: relative`，以采集显示器中心为准星，发送相对鼠标输入。
配置中的 `relative_scale_x: 1.40` 和 `relative_scale_y: 1.70` 分别为水平、垂直的初步校准比例。
它们取两次对应方向测量的换算系数中间值：X 为 `(100/74+100/69)/2`，Y 为 `(100/85+500/224)/2`。
垂直测量差异较大，这些值供手动微调。通用倍率 `relative_scale` 保持 1。
默认启用 `relative_direct: true`，第一次有效触发立即发送完整的 `像素偏差 × 对应轴比例 × 通用倍率`，
忽略 `relative_gain`、`relative_max_step` 和目标平滑；保留死区和后续反馈间隔。
例如目标相对准星偏右/下各 100 像素，一次发送 `(140,170)`，而非拆成每次 12 单位。
50ms 反馈间隔只限制后续修正，首次发送不等待；点按发送一次，按住等待新画面持续纠偏。
关闭 `relative_direct` 后，使用下述增益、限幅和平滑控制。

默认启用 `lock_target: true`：首次按关键点优先级选择最近敌人，按住期间关联同一敌人的原始框，
不会因为旁边敌人更靠近准星或某人的关键点暂时消失就切换敌人。
相对输入后，使用轴比例估计相机引起的框位移，等待反馈后再关联；匹配不明确时暂停移动。
目标短暂丢检时保留锁定 `target_lost_grace_ms`（默认 150ms），期间不发送旧坐标，超时后重新选目标。
松开并再次点按可立即重新选择。日志中的 `lock_id` 仅在重新选敌人时改变，`lock_retained` 表示是否保持关联。
该关联依赖检测框几何位置，没有外观身份识别；目标重叠时会优先暂停。
构建仅在输出目录没有配置文件时复制默认配置，后续重新编译会保留用户修改的运行配置。

启用 `mouse.enabled` 后，点按左键触发一次修正，按住左键持续按检测反馈修正。
快速按下并松开也会被记录；移动间隔内保留本次触发，间隔结束后按最新检测结果执行。
当次没有有效目标时丢弃点按请求，避免后续突然移动。

先选最近的有效第一关键点，再选第二关键点；都没有时选活敌框上半部分中心：
`x=(x1+x2)/2`，`y=y1+(y2-y1)/4`。队友、尸体以及这些区域内的目标点不参与选择。
关键点还必须位于所属活敌框内，并满足 `inference.keypoint_threshold`。

检测坐标先还原缩放和 padding，再加上 ROI 在整屏中的偏移：
`screenX=(modelX-padX)/scale+captureX`，Y 同理。
3840×2160 屏幕、居中 1280×1280 ROI、640×640 输入时，`captureX=1280`、`captureY=440`、`scale=0.5`；
模型中心 `(320,320)` 对应整屏 `(1920,1080)`，相对准星误差为零。
`relative` 模式以采集显示器中心为准星，适用于游戏画面及准星居中的情况。

同一目标的点按 `mouse.target_smoothing_ms` 做指数平滑；更换目标或关键点优先级、丢失目标、
停止触发或检测间隔超过 100 ms 时重置。相对模式还将平滑误差限制在本帧误差的方向和幅度内，
避免越过目标后仍向旧位置移动。

相对模式的自动修正按 `误差 × relative_scale × 对应轴的 relative_scale_x/y × relative_gain` 换算，并设置以下限制：

- `relative_gain`：每次修正比例，默认 0.2。
- `relative_max_step`：单次每轴输入上限，默认 12。
- `relative_deadzone_px`：停止修正的像素范围，默认每轴 2 像素。
- `relative_feedback_ms`：两次修正至少间隔的画面反馈时间，默认 50 ms；与 `move_interval_ms` 取较大值。

屏幕像素偏差与游戏鼠标输入量不是固定 1:1，`relative_scale` 仍需按游戏灵敏度及视野校准。
点按只执行一次受限修正，不能保证一次到点；按住时持续使用最新检测结果收敛。
`absolute`/`both` 模式仍以桌面光标为参考，不应用游戏相对修正限制。
