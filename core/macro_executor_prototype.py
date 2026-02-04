在 core/visual_feedback_loop.py 基础上开发的宏操作原型，支持从录制到回放的完整流程，并集成视觉确认与点击失败后的自动修正机制。以下是该原型的实现代码（纯文本格式）：

import os
import time
import json
import cv2
import numpy as np
import pyautogui
import threading
from datetime import datetime
from pynput import mouse, keyboard
from typing import List, Dict, Tuple, Optional

# 配置参数
RECORDINGS_DIR = "recordings"
SCREENSHOT_DIR = "screenshots"
CONFIDENCE_THRESHOLD = 0.8
MAX_RETRY_ATTEMPTS = 3
CHECK_INTERVAL = 0.5  # 视觉确认轮询间隔（秒）
TEMPLATE_RESIZE_FACTOR = 1.0  # 模板匹配缩放因子

# 确保目录存在
os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

class MacroAction:
    def __init__(self, action_type: str, timestamp: float, **kwargs):
        self.action_type = action_type
        self.timestamp = timestamp
        self.params = kwargs

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "timestamp": self.timestamp,
            "params": self.params
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["action_type"], data["timestamp"], **data["params"])

class VisualFeedbackLoop:
    def __init__(self):
        self.recorded_actions: List[MacroAction] = []
        self.is_recording = False
        self.is_playing = False
        self.listener_mouse = None
        self.listener_keyboard = None
        self.last_timestamp = 0

    def capture_screen_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        screenshot = pyautogui.screenshot(region=region)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def find_template_on_screen(self, template_path: str, region=None, confidence: float = CONFIDENCE_THRESHOLD) -> Optional[Tuple[int, int]]:
        screen_img = pyautogui.screenshot(region=region)
        screen_array = np.array(screen_img)
        screen_gray = cv2.cvtColor(screen_array, cv2.COLOR_RGB2GRAY)

        template = cv2.imread(template_path, 0)
        if TEMPLATE_RESIZE_FACTOR != 1.0:
            w = int(template.shape[1] * TEMPLATE_RESIZE_FACTOR)
            h = int(template.shape[0] * TEMPLATE_RESIZE_FACTOR)
            template = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA)

        res = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val >= confidence:
            h, w = template.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return (center_x, center_y)
        return None

    def wait_for_element_and_click(self, template_path: str, timeout: float = 10.0, offset_x: int = 0, offset_y: int = 0) -> bool:
        start_time = time.time()
        last_check = 0
        position = None

        while time.time() - start_time < timeout:
            current_time = time.time()
            if current_time - last_check >= CHECK_INTERVAL:
                position = self.find_template_on_screen(template_path)
                if position:
                    break
                last_check = current_time
            time.sleep(0.1)

        if not position:
            print(f"[ERROR] 元素未在 {timeout} 秒内出现: {template_path}")
            return False

        target_x = position[0] + offset_x
        target_y = position[1] + offset_y

        for attempt in range(MAX_RETRY_ATTEMPTS):
            try:
                pyautogui.click(target_x, target_y)
                time.sleep(0.5)
                # 验证点击是否成功（通过检查目标是否仍然可见）
                new_pos = self.find_template_on_screen(template_path)
                if not new_pos or attempt == MAX_RETRY_ATTEMPTS - 1:
                    return True
                else:
                    print(f"[INFO] 点击可能未生效，重试中... ({attempt + 1}/{MAX_RETRY_ATTEMPTS})")
                    time.sleep(1)
            except Exception as e:
                print(f"[ERROR] 点击失败: {e}")
                time.sleep(1)

        return False

    def on_mouse_click(self, x, y, button, pressed):
        if not self.is_recording or not pressed:
            return True

        timestamp = time.time()
        event_time = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        if button == mouse.Button.left:
            # 截图并保存点击位置附近区域作为模板
            width, height = 80, 80
            left = max(0, x - width // 2)
            top = max(0, y - height // 2)
            region = (left, top, width, height)

            screen_img = pyautogui.screenshot(region=region)
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            template_filename = f"click_{timestamp_str}.png"
            template_path = os.path.join(SCREENSHOT_DIR, template_filename)
            screen_img.save(template_path)

            action = MacroAction(
                action_type="click",
                timestamp=timestamp,
                x=x,
                y=y,
                template_path=template_path,
                delay=event_time
            )
            self.recorded_actions.append(action)
            print(f"[RECORD] 左键点击已记录: ({x}, {y}), 模板: {template_filename}")

        return True

    def on_key_press(self, key):
        if not self.is_recording:
            return True

        try:
            key_name = key.char
        except AttributeError:
            key_name = str(key)

        timestamp = time.time()
        event_time = timestamp - self.last_timestamp
        self.last_timestamp = timestamp

        action = MacroAction(
            action_type="keypress",
            timestamp=timestamp,
            key=key_name,
            delay=event_time
        )
        self.recorded_actions.append(action)
        print(f"[RECORD] 键盘输入已记录: {key_name}")

        # 使用特定快捷键停止录制
        if key_name.lower() == 'esc':
            self.stop_recording()
            print("[INFO] 录制已停止")

    def start_recording(self):
        if self.is_recording:
            print("[WARNING] 正在录制中，无法重复开始")
            return

        self.recorded_actions.clear()
        self.last_timestamp = time.time()
        self.is_recording = True

        self.listener_mouse = mouse.Listener(on_click=self.on_mouse_click)
        self.listener_keyboard = keyboard.Listener(on_press=self.on_key_press)

        self.listener_mouse.start()
        self.listener_keyboard.start()

        print("[INFO] 开始录制... 按 ESC 停止")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        if self.listener_mouse:
            self.listener_mouse.stop()
        if self.listener_keyboard:
            self.listener_keyboard.stop()

        # 保存录制动作
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(RECORDINGS_DIR, f"macro_{timestamp}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([action.to_dict() for action in self.recorded_actions], f, indent=2, ensure_ascii=False)

        print(f"[INFO] 录制完成，已保存至: {filename}")

    def play_macro(self, macro_file: str, playback_speed: float = 1.0):
        if self.is_playing:
            print("[WARNING] 正在播放宏，无法重复启动")
            return

        if not os.path.exists(macro_file):
            print(f"[ERROR] 宏文件不存在: {macro_file}")
            return

        with open(macro_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        actions = [MacroAction.from_dict(item) for item in data]
        self.is_playing = True

        print(f"[INFO] 开始回放宏: {macro_file}")

        for idx, action in enumerate(actions):
            if not self.is_playing:
                break

            # 处理延迟
            if idx > 0 and action.delay > 0:
                adjusted_delay = action.delay / playback_speed
                time.sleep(adjusted_delay)

            if action.action_type == "click":
                template_path = action.params.get("template_path")
                x = action.params.get("x")
                y = action.params.get("y")
                success = self.wait_for_element_and_click(template_path, offset_x=x - (x - 40), offset_y=y - (y - 40))
                if not success:
                    print(f"[ERROR] 点击操作失败，终止回放")
                    self.is_playing = False
                    break
                else:
                    print(f"[PLAY] 成功点击: ({x}, {y})")

            elif action.action_type == "keypress":
                key = action.params.get("key")
                try:
                    pyautogui.press(key)
                    print(f"[PLAY] 输入按键: {key}")
                except Exception as e:
                    print(f"[ERROR] 按键模拟失败: {e}")

        self.is_playing = False
        print("[INFO] 宏回放完成")

    def stop_playback(self):
        self.is_playing = False
        print("[INFO] 宏回放已手动停止")

    def list_recorded_macros(self) -> List[str]:
        if not os.path.exists(RECORDINGS_DIR):
            return []
        files = [f for f in os.listdir(RECORDINGS_DIR) if f.endswith('.json')]
        return sorted(files)

# 主程序入口点示例
if __name__ == "__main__":
    feedback_loop = VisualFeedbackLoop()

    print("宏录制与回放系统启动")
    print("命令:")
    print("  record  - 开始录制")
    print("  stop    - 停止录制或回放")
    print("  list    - 列出所有已录制宏")
    print("  play <index> - 回放指定宏（索引）")
    print("  exit    - 退出程序")

    while True:
        cmd = input("\n请输入命令: ").strip().lower()

        if cmd == "record":
            if not feedback_loop.is_recording:
                feedback_loop.start_recording()
            else:
                print("[INFO] 正在录制中...")

        elif cmd == "stop":
            if feedback_loop.is_recording:
                feedback_loop.stop_recording()
            elif feedback_loop.is_playing:
                feedback_loop.stop_playback()
            else:
                print("[INFO] 当前无运行任务")

        elif cmd == "list":
            macros = feedback_loop.list_recorded_macros()
            if not macros:
                print("暂无已录制宏")
            else:
                print("已录制的宏:")
                for i, name in enumerate(macros):
                    print(f"  {i}: {name}")

        elif cmd.startswith("play"):
            parts = cmd.split()
            if len(parts) < 2:
                print("用法: play <index>")
                continue
            try:
                idx = int(parts[1])
                macros = feedback_loop.list_recorded_macros()
                if 0 <= idx < len(macros):
                    filepath = os.path.join(RECORDINGS_DIR, macros[idx])
                    threading.Thread(target=feedback_loop.play_macro, args=(filepath,), daemon=True).start()
                else:
                    print(f"索引超出范围，有效范围: 0-{len(macros)-1}")
            except ValueError:
                print("请输入有效的数字索引")
            except Exception as e:
                print(f"播放错误: {e}")

        elif cmd == "exit":
            if feedback_loop.is_recording:
                feedback_loop.stop_recording()
            if feedback_loop.is_playing:
                feedback_loop.stop_playback()
            print("程序退出")
            break

        else:
            print("未知命令，请重新输入")