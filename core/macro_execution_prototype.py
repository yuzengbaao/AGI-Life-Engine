core/desktop_automation.py 中宏操作功能的实现现状分析：

当前 core/desktop_automation.py 文件中的宏操作功能主要用于执行预定义的一系列自动化操作，例如模拟键盘输入、鼠标点击、窗口控制等。从现有实现来看，宏操作通常以硬编码的方式组织在函数或类中，缺乏动态录制能力，回放过程也较为静态，难以支持用户自定义流程的灵活扩展。主要问题包括：

1. 宏指令存储方式简单，多采用列表或字典结构保存动作序列，缺乏统一的动作模型。
2. 无录制机制，用户无法通过实际操作生成宏，必须手动编写代码定义动作。
3. 回放逻辑耦合在主流程中，不易进行暂停、恢复、速度调节等控制。
4. 扩展性差，新增动作类型需要修改核心回放逻辑，违反开闭原则。
5. 缺乏错误处理与状态反馈机制，在回放过程中出现异常时难以恢复或提示。

为解决上述问题，需设计一个可扩展的宏录制与回放示例代码框架，具备以下特性：

- 支持动态录制用户操作并生成宏脚本
- 提供清晰的命令模式结构，便于扩展新的操作类型
- 实现解耦的录制器与播放器组件
- 支持宏的持久化存储与加载
- 提供基础的控制接口（如开始、暂停、停止、速度设置）

以下是可扩展的宏录制与回放示例代码框架：

import json
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional
import threading

# 操作类型枚举
class ActionType(Enum):
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    KEYBOARD_PRESS = "key_press"
    KEYBOARD_TYPE = "key_type"
    WAIT = "wait"
    WINDOW_FOCUS = "window_focus"

# 动作基类
class Action(ABC):
    def __init__(self, action_type: ActionType, timestamp: float, **kwargs):
        self.action_type = action_type
        self.timestamp = timestamp
        self.params = kwargs

    @abstractmethod
    def execute(self):
        pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.action_type.value,
            "timestamp": self.timestamp,
            "params": self.params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        action_map = {
            ActionType.MOUSE_CLICK.value: MouseClickAction,
            ActionType.MOUSE_MOVE.value: MouseMoveAction,
            ActionType.KEYBOARD_PRESS.value: KeyPressAction,
            ActionType.KEYBOARD_TYPE.value: KeyTypeAction,
            ActionType.WAIT.value: WaitAction,
            ActionType.WINDOW_FOCUS.value: WindowFocusAction
        }
        action_type = data["type"]
        if action_type in action_map:
            return action_map[action_type].from_dict(data)
        else:
            raise ValueError(f"未知的操作类型: {action_type}")

# 鼠标点击动作
class MouseClickAction(Action):
    def __init__(self, x: int, y: int, button: str = "left", timestamp: float = None):
        super().__init__(ActionType.MOUSE_CLICK, timestamp or time.time(), x=x, y=y, button=button)

    def execute(self):
        x, y = self.params["x"], self.params["y"]
        button = self.params["button"]
        # 此处调用底层桌面自动化接口
        print(f"[执行] 鼠标点击 ({x}, {y}) 按钮={button}")
        # 示例：pyautogui.click(x, y, button=button)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MouseClickAction':
        params = data["params"]
        return cls(
            x=params["x"],
            y=params["y"],
            button=params.get("button", "left"),
            timestamp=data["timestamp"]
        )

# 鼠标移动动作
class MouseMoveAction(Action):
    def __init__(self, x: int, y: int, duration: float = 0.0, timestamp: float = None):
        super().__init__(ActionType.MOUSE_MOVE, timestamp or time.time(), x=x, y=y, duration=duration)

    def execute(self):
        x, y = self.params["x"], self.params["y"]
        duration = self.params["duration"]
        print(f"[执行] 鼠标移动至 ({x}, {y}) 耗时={duration}s")
        # 示例：pyautogui.moveTo(x, y, duration=duration)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MouseMoveAction':
        params = data["params"]
        return cls(
            x=params["x"],
            y=params["y"],
            duration=params.get("duration", 0.0),
            timestamp=data["timestamp"]
        )

# 键盘按键动作
class KeyPressAction(Action):
    def __init__(self, key: str, timestamp: float = None):
        super().__init__(ActionType.KEYBOARD_PRESS, timestamp or time.time(), key=key)

    def execute(self):
        key = self.params["key"]
        print(f"[执行] 按下按键: {key}")
        # 示例：pyautogui.press(key)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyPressAction':
        params = data["params"]
        return cls(key=params["key"], timestamp=data["timestamp"])

# 键盘输入文本动作
class KeyTypeAction(Action):
    def __init__(self, text: str, interval: float = 0.01, timestamp: float = None):
        super().__init__(ActionType.KEYBOARD_TYPE, timestamp or time.time(), text=text, interval=interval)

    def execute(self):
        text = self.params["text"]
        interval = self.params["interval"]
        print(f"[执行] 输入文本: '{text}' 间隔={interval}s")
        # 示例：pyautogui.typewrite(text, interval=interval)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KeyTypeAction':
        params = data["params"]
        return cls(
            text=params["text"],
            interval=params.get("interval", 0.01),
            timestamp=data["timestamp"]
        )

# 等待动作
class WaitAction(Action):
    def __init__(self, duration: float, timestamp: float = None):
        super().__init__(ActionType.WAIT, timestamp or time.time(), duration=duration)

    def execute(self):
        duration = self.params["duration"]
        print(f"[执行] 等待 {duration} 秒")
        time.sleep(duration)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WaitAction':
        params = data["params"]
        return cls(duration=params["duration"], timestamp=data["timestamp"])

# 窗口聚焦动作
class WindowFocusAction(Action):
    def __init__(self, window_title: str, timeout: float = 5.0, timestamp: float = None):
        super().__init__(ActionType.WINDOW_FOCUS, timestamp or time.time(), window_title=window_title, timeout=timeout)

    def execute(self):
        title = self.params["window_title"]
        timeout = self.params["timeout"]
        print(f"[执行] 聚焦窗口: '{title}' 超时={timeout}s")
        # 示例：使用 win32gui 查找并激活窗口

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WindowFocusAction':
        params = data["params"]
        return cls(
            window_title=params["window_title"],
            timeout=params.get("timeout", 5.0),
            timestamp=data["timestamp"]
        )

# 宏记录器
class MacroRecorder:
    def __init__(self):
        self.actions: List[Action] = []
        self.is_recording = False
        self.start_time: Optional[float] = None
        self.listeners = []

    def start_recording(self):
        if self.is_recording:
            return
        self.actions.clear()
        self.is_recording = True
        self.start_time = time.time()
        print("开始录制宏...")
        self._setup_listeners()

    def stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self._remove_listeners()
        print("停止录制宏")

    def add_action(self, action: Action):
        if self.is_recording:
            self.actions.append(action)
            print(f"已记录: {action.action_type.value}")

    def _setup_listeners(self):
        # 示例：使用 pynput 监听鼠标和键盘事件
        try:
            from pynput import mouse, keyboard

            self.mouse_listener = mouse.Listener(
                on_move=self._on_mouse_move,
                on_click=self._on_mouse_click
            )
            self.keyboard_listener = keyboard.Listener(
                on_press=self._on_key_press
            )

            self.mouse_listener.start()
            self.keyboard_listener.start()
        except ImportError:
            print("警告: pynput 未安装，无法启用实时监听")

    def _remove_listeners(self):
        if hasattr(self, 'mouse_listener'):
            self.mouse_listener.stop()
        if hasattr(self, 'keyboard_listener'):
            self.keyboard_listener.stop()

    def _on_mouse_move(self, x, y):
        action = MouseMoveAction(x=int(x), y=int(y))
        self.add_action(action)

    def _on_mouse_click(self, x, y, button, pressed):
        if pressed:
            btn_name = "left" if "left" in str(button) else "right"
            action = MouseClickAction(x=int(x), y=int(y), button=btn_name)
            self.add_action(action)

    def _on_key_press(self, key):
        try:
            key_char = key.char
            action = KeyPressAction(key=key_char)
            self.add_action(action)
        except AttributeError:
            # 特殊键处理
            key_name = str(key).replace("Key.", "")
            if key_name in ["f1", "f2", "esc", "enter", "tab", "shift", "ctrl"]:
                action = KeyPressAction(key=key_name)
                self.add_action(action)

    def save_to_file(self, filepath: str):
        data = [action.to_dict() for action in self.actions]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"宏已保存至: {filepath}")

    def load_from_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.actions = [Action.from_dict(item) for item in data]
            print(f"宏已从 {filepath} 加载，共 {len(self.actions)} 个动作")
        except Exception as e:
            print(f"加载宏失败: {e}")

# 宏播放器
class MacroPlayer:
    def __init__(self):
        self.is_playing = False
        self.playback_thread: Optional[threading.Thread] = None
        self.speed_factor = 1.0  # 播放速度倍率

    def play(self, actions: List[Action], callback=None):
        if self.is_playing:
            print("正在播放中，请先停止")
            return

        self.is_playing = True

        def playback():
            start_time = actions[0].timestamp if actions else time.time()
            for i, action in enumerate(actions):
                if not self.is_playing:
                    break

                elapsed = action.timestamp - start_time
                adjusted_delay = elapsed / self.speed_factor
                if i > 0:
                    time.sleep(max(0, adjusted_delay))

                if self.is_playing:
                    try:
                        action.execute()
                    except Exception as e:
                        print(f"执行动作 {i+1} 失败: {e}")

            self.is_playing = False
            if callback:
                callback()

        self.playback_thread = threading.Thread(target=playback)
        self.playback_thread.start()

    def stop(self):
        self.is_playing = False
        if self.playback_thread:
            self.playback_thread.join(timeout=1.0)

    def set_speed(self, factor: float):
        if factor <= 0:
            raise ValueError("播放速度必须大于0")
        self.speed_factor = factor

# 使用示例
def example_usage():
    recorder = MacroRecorder()
    player = MacroPlayer()

    # 录制宏
    recorder.start_recording()
    time.sleep(5)  # 录制5秒操作
    recorder.stop_recording()

    # 保存宏
    recorder.save_to_file("my_macro.json")

    # 或加载已有宏
    # recorder.load_from_file("my_macro.json")

    # 播放宏
    player.set_speed(1.0)
    player.play(recorder.actions, callback=lambda: print("宏播放完成"))

if __name__ == "__main__":
    example_usage()

说明：

该框架采用命令模式（Command Pattern）设计，每个操作封装为独立的 Action 子类，支持未来扩展更多操作类型（如截图、剪贴板操作等）。Recorder 负责监听用户输入并生成动作序列，Player 负责异步回放，支持速度调节与中断控制。动作可序列化为 JSON 格式，便于存储与共享。

此设计实现了录制与回放逻辑的分离，新增动作类型只需继承 Action 并注册到工厂映射中，符合开闭原则，具备良好的可扩展性，适用于构建桌面自动化工具中的宏功能模块。