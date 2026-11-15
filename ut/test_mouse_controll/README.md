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
