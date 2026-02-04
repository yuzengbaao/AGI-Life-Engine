分析 core/desktop_automation.py 文件中的宏操作支持逻辑，发现当前系统已具备基础的桌面自动化能力，包括窗口管理、控件查找与点击、应用启动等操作。现有设计采用面向对象方式封装操作接口，但缺乏对宏录制与回放功能的支持。为此，设计并实现一个可配置的宏录制与回放功能原型。

一、需求理解  
目标：支持至少两个连续UI操作的录制（如打开应用、点击按钮）与回放，并允许用户通过配置控制录制行为和回放参数。

关键能力：
1. 开始/停止宏录制
2. 存储操作序列（时间戳、操作类型、参数）
3. 回放宏（按顺序执行记录的操作）
4. 可配置性（如回放速度、重试机制）

二、设计思路

1. 宏操作数据结构  
定义宏操作为字典格式，包含以下字段：
- timestamp: 操作发生的时间（用于计算延迟）
- action_type: 操作类型（如 "launch_app", "click", "type_text"）
- params: 操作所需参数（如 app_name, control_selector）
- window_info: 操作时的窗口上下文（如标题、进程ID）

示例：
{
  "timestamp": 1712345678.123,
  "action_type": "launch_app",
  "params": {"app_name": "Notepad++"}
}
{
  "timestamp": 1712345682.456,
  "action_type": "click",
  "params": {"selector": {"name": "File", "control_type": "MenuItem"}}
}

2. 核心组件设计  
- MacroRecorder：负责监听和记录用户操作
- MacroPlayer：负责按序列回放操作
- MacroConfig：配置类，支持回放速度、失败策略等
- OperationLog：持久化存储宏序列（JSON格式）

3. 与现有系统的集成  
利用 desktop_automation.py 中已有的接口：
- DesktopAutomation.launch_application(app_name)
- DesktopAutomation.find_control(selector)
- DesktopAutomation.click_control(control)

在这些操作外层封装事件捕获逻辑。

三、实现原型

1. 新增文件：core/macro.py

代码实现如下：

import time
import json
from typing import List, Dict, Any
from core.desktop_automation import DesktopAutomation

class MacroConfig:
    def __init__(self):
        self.playback_speed = 1.0  # 回放速度倍率
        self.on_failure = "stop"   # 失败策略：stop / retry / skip
        self.retry_count = 3

class MacroRecorder:
    def __init__(self, automation: DesktopAutomation):
        self.automation = automation
        self.is_recording = False
        self.recorded_actions: List[Dict] = []
        self.start_time = None

    def start_recording(self):
        self.recorded_actions.clear()
        self.start_time = time.time()
        self.is_recording = True
        print("宏录制已开始")

    def stop_recording(self) -> List[Dict]:
        self.is_recording = False
        print(f"宏录制结束，共记录 {len(self.recorded_actions)} 个操作")
        return self.recorded_actions.copy()

    def _record_action(self, action_type: str, params: Dict):
        if not self.is_recording:
            return
        entry = {
            "timestamp": time.time() - self.start_time,
            "action_type": action_type,
            "params": params,
            "window_info": {
                "active_window": self.automation.get_active_window_title(),
                "process": self.automation.get_active_process_name()
            }
        }
        self.recorded_actions.append(entry)

    def record_launch_app(self, app_name: str):
        self._record_action("launch_app", {"app_name": app_name})

    def record_click(self, selector: Dict[str, Any]):
        self._record_action("click", {"selector": selector})


class MacroPlayer:
    def __init__(self, automation: DesktopAutomation, config: MacroConfig = None):
        self.automation = automation
        self.config = config or MacroConfig()

    def play(self, actions: List[Dict]) -> bool:
        prev_timestamp = 0
        for idx, action in enumerate(actions):
            timestamp = action["timestamp"]
            action_type = action["action_type"]
            params = action["params"]

            # 控制延迟
            delay = (timestamp - prev_timestamp) / self.config.playback_speed
            if delay > 0:
                time.sleep(delay)

            success = False
            if action_type == "launch_app":
                try:
                    self.automation.launch_application(params["app_name"])
                    success = True
                except Exception as e:
                    print(f"启动应用失败: {e}")

            elif action_type == "click":
                selector = params["selector"]
                for _ in range(self.config.retry_count if self.config.on_failure == "retry" else 1):
                    try:
                        ctrl = self.automation.find_control(selector)
                        if ctrl:
                            self.automation.click_control(ctrl)
                            success = True
                            break
                    except Exception as e:
                        print(f"查找或点击控件失败: {e}")
                        time.sleep(0.5)

            if not success:
                print(f"操作 {idx} 执行失败: {action}")
                if self.config.on_failure == "stop":
                    return False
                elif self.config.on_failure == "skip":
                    continue
            prev_timestamp = timestamp
        return True


四、使用示例

# 初始化自动化引擎
auto = DesktopAutomation()
recorder = MacroRecorder(auto)
player = MacroPlayer(auto, MacroConfig())

# 开始录制
recorder.start_recording()

# 用户执行操作
auto.launch_application("Notepad++")
recorder.record_launch_app("Notepad++")

time.sleep(2)

file_menu = {"name": "File", "control_type": "MenuItem"}
ctrl = auto.find_control(file_menu)
if ctrl:
    auto.click_control(ctrl)
    recorder.record_click(file_menu)

# 停止录制
macro_sequence = recorder.stop_recording()

# 保存宏到文件
with open("macros/open_file_menu.json", "w", encoding="utf-8") as f:
    json.dump(macro_sequence, f, indent=2)

# 回放宏
print("开始回放...")
player.play(macro_sequence)

五、可配置性说明

通过 MacroConfig 支持以下配置项：
- playback_speed: 控制回放速度，1.0为原速，2.0为加速，0.5为减速
- on_failure: 定义操作失败后的处理策略
  - "stop": 立即终止回放
  - "retry": 重试最多 retry_count 次
  - "skip": 跳过当前操作继续
- retry_count: 重试次数上限

六、扩展性考虑

1. 支持更多操作类型：键盘输入、文本输入、等待条件等
2. 添加断言操作：验证界面状态（如窗口是否存在）
3. 提供图形化录制界面（后续可集成）
4. 宏编辑功能：修改、删除、插入操作步骤

七、结论

本方案基于现有的 desktop_automation.py 实现了一个轻量级、可配置的宏录制与回放原型。支持至少两个连续UI操作的录制与回放，具备良好的可扩展性和稳定性。通过事件记录与参数化解耦，实现了与底层自动化逻辑的无缝集成，为后续构建完整RPA功能打下基础。