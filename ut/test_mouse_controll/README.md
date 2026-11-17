# 鼠标矩形可视化测试

在本目录执行：

```powershell
cmake -S . -B build
cmake --build build --config Release
```

运行：`.\build\Release\mouse_rectangle_test.exe absolute`

将鼠标沿主屏幕中央的 720×720 绿色矩形顺时针移动一圈，每条边约 5.76 秒。
倒计时 3 秒后从左上角开始，Esc 随时取消。运行时请勿手动移动鼠标。
使用 MouseController::moveBy，与主程序共用移动执行代码；不测试识别、左键触发与帧冷却。
传入 relative 可测试 SendInput 相对移动；Windows 指针加速可能影响轨迹。
程序输出实际光标与计划位置的最大像素偏差；不超过 3 像素返回 0，超限返回 1，取消返回 2。

## 云游戏相对输入诊断

单独运行 `mouse_input_probe.exe`，不加载模型、不依赖目标检测或左键。
程序倒计时 5 秒，请在此期间将焦点切回云游戏；随后用 `MouseController::moveBy`
在约 1 秒内发送总计 100 单位的向右相对输入，然后退出。Esc 可取消，发送过程中切换前台窗口也会停止。
请先关闭自动控制主程序，避免两份输入相互干扰。

在此目录独立编译后运行：`.\build\Release\mouse_input_probe.exe`。
根项目编译后位于：`build/ut/test_mouse_controll/Release/mouse_input_probe.exe`。
`input-submitted` 只证明 Windows 已接收输入，是否转发到云端仍需观察游戏视角。
不要用桌面光标是否回到中心判断游戏内相对输入是否生效。
