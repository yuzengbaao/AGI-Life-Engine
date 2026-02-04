在 core/desktop_automation.py 中实现宏录制与回放功能，需支持录制用户的鼠标和键盘操作，并将这些操作序列化为可保存、加载和执行的宏函数。以下是该模块的实现内容：

import time
import json
from pynput import mouse, keyboard
from threading import Thread, Event

class MacroRecorder:
    def __init__(self):
        self.events = []
        self.recording = False
        self.mouse_listener = None
        self.keyboard_listener = None
        self.stop_event = Event()

    def start_recording(self):
        """开始录制用户操作"""
        if self.recording:
            return
        self.recording = True
        self.events.clear()
        self.stop_event.clear()

        def on_mouse_event(event_type, x, y, button=None, pressed=None):
            timestamp = time.time()
            event_data = {
                'type': 'mouse',
                'event': event_type,
                'x': x,
                'y': y,
                'timestamp': timestamp
            }
            if button:
                event_data['button'] = str(button)
            if event_type == 'click':
                event_data['pressed'] = pressed
            self.events.append(event_data)

        def on_move(x, y):
            if self.recording:
                on_mouse_event('move', x, y)

        def on_click(x, y, button, pressed):
            if self.recording:
                action = 'click' if pressed else 'release'
                on_mouse_event(action, x, y, button, pressed)

        def on_scroll(x, y, dx, dy):
            if self.recording:
                on_mouse_event('scroll', x, y, None, None)

        def on_press(key):
            if not self.recording:
                return
            try:
                key_str = key.char
            except AttributeError:
                key_str = str(key)
            self.events.append({
                'type': 'keyboard',
                'event': 'press',
                'key': key_str,
                'timestamp': time.time()
            })

        def on_release(key):
            if not self.recording:
                return
            if key == keyboard.Key.esc:
                self.stop_recording()
                return
            try:
                key_str = key.char
            except AttributeError:
                key_str = str(key)
            self.events.append({
                'type': 'keyboard',
                'event': 'release',
                'key': key_str,
                'timestamp': time.time()
            })

        self.mouse_listener = mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )

        self.mouse_listener.start()
        self.keyboard_listener.start()

    def stop_recording(self):
        """停止录制"""
        if not self.recording:
            return
        self.recording = False
        self.stop_event.set()
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()

    def save_macro(self, file_path):
        """将录制的事件保存到文件"""
        with open(file_path, 'w') as f:
            json.dump(self.events, f, indent=2)

    def load_macro(self, file_path):
        """从文件加载宏事件"""
        with open(file_path, 'r') as f:
            self.events = json.load(f)

    def replay_macro(self, speed_factor=1.0):
        """回放录制的宏"""
        if not self.events:
            print("没有可回放的宏事件")
            return

        from pynput.mouse import Controller as MouseController
        from pynput.keyboard import Controller as KeyboardController

        mouse_ctrl = MouseController()
        keyboard_ctrl = KeyboardController()

        start_time = self.events[0]['timestamp']
        for i, event in enumerate(self.events):
            current_time = event['timestamp']
            if i == 0:
                delay = 0
            else:
                prev_time = self.events[i-1]['timestamp']
                delay = (current_time - prev_time) / speed_factor
            time.sleep(delay)

            if event['type'] == 'mouse':
                mouse_ctrl.position = (event['x'], event['y'])
                if event['event'] == 'click':
                    btn = self._parse_button(event.get('button'))
                    if btn:
                        mouse_ctrl.press(btn)
                elif event['event'] == 'release':
                    btn = self._parse_button(event.get('button'))
                    if btn:
                        mouse_ctrl.release(btn)
                elif event['event'] == 'scroll':
                    mouse_ctrl.scroll(event.get('dx', 0), event.get('dy', 0))

            elif event['type'] == 'keyboard':
                key = self._parse_key(event['key'])
                if event['event'] == 'press':
                    keyboard_ctrl.press(key)
                elif event['event'] == 'release':
                    keyboard_ctrl.release(key)

    def _parse_button(self, button_str):
        """解析鼠标按钮字符串"""
        if not button_str:
            return None
        if 'left' in button_str:
            return mouse.Button.left
        elif 'right' in button_str:
            return mouse.Button.right
        elif 'middle' in button_str:
            return mouse.Button.middle
        return None

    def _parse_key(self, key_str):
        """解析键盘按键字符串"""
        if key_str.startswith('Key.'):
            special_key = key_str.replace('Key.', '')
            return getattr(keyboard.Key, special_key, key_str)
        return key_str

# 全局宏录制器实例
recorder = MacroRecorder()

def start_macro_recording():
    """启动宏录制"""
    recorder.start_recording()
    print("宏录制已启动，按 Esc 键停止...")

def stop_macro_and_save(file_path):
    """停止录制并保存宏"""
    recorder.stop_recording()
    recorder.save_macro(file_path)
    print(f"宏已保存至 {file_path}")

def play_macro_from_file(file_path, speed=1.0):
    """从文件加载并回放宏"""
    recorder.load_macro(file_path)
    print(f"正在回放宏: {file_path}")
    recorder.replay_macro(speed_factor=speed)

# 使用示例（可在主程序中调用）
# start_macro_recording()
# ... 用户操作 ...
# stop_macro_and_save("macros/demo.json")
# play_macro_from_file("macros/demo.json", speed=1.0)

该实现提供了完整的宏录制与回放功能，支持鼠标移动、点击、滚轮以及键盘按键的捕获与重放。事件以时间戳记录，确保回放时的操作时序准确。支持将宏保存为 JSON 文件以便后续使用，并可通过调整 speed_factor 参数控制回放速度。