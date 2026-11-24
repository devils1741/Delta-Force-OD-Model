"""Windows relative mouse movement probe; default: wait 3 seconds, move once.

Uses the same SendInput/MOUSEEVENTF_MOVE route as MouseController.cpp.
No game detection, clicks, sensitivity changes, or third-party packages.
"""

import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import sys
import time


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [("type", wintypes.DWORD), ("value", INPUTUNION)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="等待后发送一次相对鼠标位移，自动申请管理员权限。")
    parser.add_argument("--delay", type=float, default=3.0, help="移动前等待秒数，默认 3")
    parser.add_argument("--dx", type=int, default=100, help="水平输入量：正数向右，负数向左，默认 100")
    parser.add_argument("--dy", type=int, default=0, help="垂直输入量：正数向下，负数向上，默认 0")
    parser.add_argument("--dry-run", action="store_true", help="仅检查参数，不提权、不移动")
    args = parser.parse_args(argv)
    if not 0 <= args.delay <= 3600:
        parser.error("--delay 必须在 0 到 3600 秒之间")
    if not (-10000 <= args.dx <= 10000 and -10000 <= args.dy <= 10000):
        parser.error("--dx 和 --dy 必须在 -10000 到 10000 之间")
    if args.dx == 0 and args.dy == 0:
        parser.error("--dx 和 --dy 不能同时为 0")
    return args


def ensure_admin():
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.IsUserAnAdmin.restype = wintypes.BOOL
    if shell.IsUserAnAdmin():
        return True
    shell.ShellExecuteW.argtypes = [
        wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR,
        wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.c_int,
    ]
    shell.ShellExecuteW.restype = ctypes.c_void_p
    arguments = subprocess.list2cmdline([str(Path(__file__).resolve()), *sys.argv[1:]])
    result = shell.ShellExecuteW(None, "runas", sys.executable, arguments,
                                str(Path(__file__).resolve().parent), 1)
    if not result or result <= 32:
        raise RuntimeError("未取得管理员权限：管理员确认可能已取消。")
    return False


def move_after_delay(args):
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
    user32.GetAsyncKeyState.restype = wintypes.SHORT
    user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
    user32.SendInput.restype = wintypes.UINT
    print(f"管理员权限已启用。{args.delay:g} 秒后移动一次：dx={args.dx}, dy={args.dy}。", flush=True)
    print("现在切回游戏；移动前按住 Esc 可取消。", flush=True)
    deadline = time.perf_counter() + args.delay
    while True:
        if user32.GetAsyncKeyState(0x1B) & 0x8000:
            print("已取消，未发送鼠标位移。", flush=True)
            return
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        time.sleep(min(0.01, remaining))
    event = INPUT(type=0)
    event.mi = MOUSEINPUT(args.dx, args.dy, 0, 0x0001, 0, 0)
    ctypes.set_last_error(0)
    if user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
        error = ctypes.get_last_error()
        raise RuntimeError(f"SendInput 未成功提交，Windows 错误码：{error}")
    print("已提交一次位移。输入量不等于游戏画面的像素位移。", flush=True)


def main():
    args = parse_args()
    if args.dry_run:
        print(f"参数检查通过：延迟 {args.delay:g} 秒，dx={args.dx}, dy={args.dy}；未移动鼠标。")
        return 0
    if sys.platform != "win32":
        raise RuntimeError("此脚本仅支持 Windows。")
    if ensure_admin():
        move_after_delay(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("已取消。")
        sys.exit(2)
    except (OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)
