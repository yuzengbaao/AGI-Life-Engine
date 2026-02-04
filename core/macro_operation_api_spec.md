宏操作 API 设计文档

1. 概述

本设计文档旨在定义基于 core/desktop_automation.py 分析结果的宏操作功能接口结构，重点实现录制与回放功能。系统提供对桌面应用程序用户交互行为的自动化捕获、存储、执行能力，支持状态管理、错误处理和可扩展性。

2. 功能目标

- 支持启动/停止宏操作的录制
- 将用户操作序列化为可存储的宏脚本
- 支持加载并回放已录制的宏
- 提供运行时状态监控与控制（暂停、继续、终止）
- 确保跨平台兼容性（Windows、macOS、Linux）
- 支持宏执行过程中的异常检测与恢复机制

3. 接口定义

3.1 宏管理器（MacroManager）

核心类，负责协调录制、存储、回放和状态管理。

方法：

start_recording()
  启动用户操作录制。
  返回：无
  异常：RuntimeError（若正在录制中）

stop_recording() -> MacroScript
  停止当前录制，生成宏脚本对象。
  返回：MacroScript 实例
  异常：RuntimeError（若未在录制状态）

save_macro(script: MacroScript, path: str) -> bool
  将宏脚本序列化并保存至指定路径。
  参数：
    script: 要保存的宏脚本
    path: 文件保存路径（JSON格式）
  返回：保存成功返回True，否则False

load_macro(path: str) -> MacroScript
  从文件加载宏脚本。
  参数：path - 文件路径
  返回：MacroScript 实例
  异常：FileNotFoundError, ValueError（格式错误）

playback(macro: MacroScript, callback: callable = None) -> PlaybackResult
  执行宏回放。
  参数：
    macro: 要回放的宏脚本
    callback: 可选回调函数，用于接收状态更新（如进度、事件）
  返回：PlaybackResult 对象，包含执行结果
  异常：RuntimeError（若正在回放或录制中）

is_recording() -> bool
  查询是否处于录制状态。
  返回：True表示正在录制

is_playing() -> bool
  查询是否处于回放状态。
  返回：True表示正在回放

abort_playback() -> bool
  立即终止当前回放。
  返回：True表示成功终止，False表示无回放在进行

3.2 宏脚本（MacroScript）

表示一段录制的操作序列，包含元数据和操作列表。

属性：

- name: str（宏名称）
- description: str（描述信息）
- created_at: datetime（创建时间）
- events: List[MacroEvent]（操作事件列表）
- platform: str（录制平台，如 "Windows"）
- app_context: str（目标应用标识，可选）

方法：

add_event(event: MacroEvent)
  添加一个操作事件到脚本末尾。

validate() -> (bool, List[str])
  验证脚本完整性。
  返回：(是否有效, 错误信息列表)

to_dict() -> dict
  序列化为字典，用于持久化存储。

from_dict(data: dict) -> MacroScript
  从字典反序列化构建宏脚本。

3.3 操作事件（MacroEvent）

表示一次用户交互动作。

通用属性：

- timestamp: float（相对于宏开始的时间戳，单位秒）
- event_type: str（事件类型，如 "mouse_move", "mouse_click", "key_press", "wait"）
- target_app: str（目标进程名或窗口标题，可选）

子类型示例：

- MouseMoveEvent:
  - x: int
  - y: int
  - absolute: bool

- MouseClickEvent:
  - button: str ("left", "right", "middle")
  - clicks: int
  - x: int
  - y: int

- KeyPressEvent:
  - key: str（键名，如 "a", "ctrl", "enter"）
  - modifiers: List[str]（修饰键，如 ["shift"]）

- WaitEvent:
  - duration: float（等待时间，秒）

3.4 回放结果（PlaybackResult）

记录一次回放执行的结果。

属性：

- success: bool（是否完整执行完成）
- error: str（错误信息，若存在）
- start_time: datetime
- end_time: datetime
- executed_events: int（实际执行的事件数）
- failed_event_index: int（失败事件索引，若适用）
- logs: List[str]（执行日志）

4. 状态管理机制

4.1 全局状态机

MacroManager 维护一个有限状态机，状态包括：

- IDLE：初始状态，无录制或回放
- RECORDING：正在录制用户操作
- PLAYBACK：正在执行宏回放
- PAUSED：回放被暂停（仅在PLAYBACK下可进入）
- ERROR：发生不可恢复错误

状态转换规则：

IDLE → RECORDING：调用 start_recording()
RECORDING → IDLE：调用 stop_recording()
IDLE → PLAYBACK：调用 playback()
PLAYBACK → PAUSED：调用 pause_playback()（待扩展）
PAUSED → PLAYBACK：调用 resume_playback()（待扩展）
PAUSED → IDLE：调用 abort_playback()
PLAYBACK → IDLE：回放正常结束或被终止
任何状态 → ERROR：发生严重异常
ERROR → IDLE：调用 reset()

4.2 状态同步与通知

- 使用观察者模式通知状态变更
- 提供 get_status() 方法返回当前状态对象：

  {
    "state": "RECORDING",
    "active_macro": null,
    "recorded_events": 42,
    "timestamp": "2025-04-05T10:00:00Z"
  }

- 回放过程中可通过 callback 函数流式推送事件级状态

5. 异常处理策略

- 录制阶段：底层输入监听异常应触发警告但不中断录制
- 回放阶段：
  - 单个事件执行失败（如目标窗口未找到）记录日志并尝试继续
  - 连续失败超过阈值（默认3次）自动 abort 并标记为失败
  - 提供重试配置选项（retry_count, retry_delay）

6. 扩展性设计

- 插件式事件处理器：支持注册自定义事件类型处理逻辑
- 上下文感知回放：可根据当前应用环境动态调整坐标或键位映射
- 宏脚本版本控制：通过 version 字段支持未来格式升级

7. 安全与权限

- 在需要操作系统级输入模拟的平台（如 macOS），首次运行提示用户授予权限
- 敏感操作（如密码输入）支持标记为掩码事件，不记录具体键值
- 提供“沙箱回放”模式，限制宏只能在指定应用区域内执行

8. 示例流程

# 录制宏
manager = MacroManager()
manager.start_recording()
# 用户执行若干操作...
script = manager.stop_recording()
manager.save_macro(script, "login_sequence.json")

# 回放宏
loaded = manager.load_macro("login_sequence.json")
result = manager.playback(loaded)
if result.success:
    print("宏执行成功")
else:
    print(f"执行失败：{result.error}")

9. 未来扩展建议

- 支持宏参数化（变量注入）
- 提供可视化编辑器接口
- 增加条件判断与循环控制结构
- 支持多宏组合调度

10. 结语

本API设计提供了清晰、稳定且可扩展的宏操作能力框架，满足自动化测试、重复任务简化等场景需求，同时确保状态可控、行为可预测。