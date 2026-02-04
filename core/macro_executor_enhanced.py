分析 core/macro_recorder_prototype.py 与 core/desktop_automation.py 的集成缺口：

1. 功能职责划分不清晰：
   - macro_recorder_prototype.py 主要负责宏的录制、回放和基本事件序列存储，但缺乏对执行环境变化的适应能力。
   - desktop_automation.py 提供了图像识别、控件查找等底层自动化能力，但在宏回放过程中未被深度整合用于动态定位。

2. 错误恢复机制缺失：
   - 当前宏回放在目标界面元素发生变化（如窗口位置移动、控件不可见）时，无法检测失败或尝试恢复。
   - 缺少基于视觉反馈的重试与重新定位逻辑，导致一旦操作失败即中断整个流程。

3. 目标定位方式静态化：
   - 宏记录使用绝对坐标或固定偏移量进行点击/输入操作，未结合图像模板匹配或OCR技术实现动态定位。
   - 回放时若界面布局改变，原有坐标失效，导致操作错位或失败。

4. 状态感知能力不足：
   - 两个模块之间缺乏运行时状态同步机制，desktop_automation.py 的识别结果未能反馈给 macro_recorder 进行决策调整。

设计目标：构建一个支持错误恢复的宏执行增强模块，融合视觉反馈机制，在操作失败时自动尝试重新定位并继续执行。

解决方案设计：

1. 模块名称：RobustMacroExecutor（鲁棒宏执行器）

2. 核心组件设计：

   a) ExecutionMonitor（执行监控器）
      - 包装原始宏指令执行过程，捕获异常和操作失败信号
      - 记录每步操作的预期目标区域与实际结果差异

   b) VisionBasedRecoveryEngine（基于视觉的恢复引擎）
      - 利用 desktop_automation.py 中的图像匹配功能（如 match_template, find_image_region）
      - 在操作失败时启动全局屏幕搜索，尝试重新定位目标控件
      - 支持多级重试策略：先局部微调搜索，再全屏扫描，最后使用相似度降级匹配

   c) AdaptiveActionRunner（自适应动作执行器）
      - 替代原有的简单坐标回放逻辑
      - 执行前验证目标是否存在，不存在则触发恢复流程
      - 成功定位后动态更新操作坐标并执行

   d) ContextSnapshotManager（上下文快照管理器）
      - 在录制阶段保存关键步骤的目标图像快照（小图）
      - 回放时用于比对和定位，提升准确性

3. 集成接口设计：

   - 向 macro_recorder_prototype.py 注入 RobustMacroExecutor 实例作为默认播放引擎
   - 扩展事件结构，增加“target_template”字段，存储该步骤对应的图像模板路径
   - 调用 desktop_automation.py 的视觉函数实现 locate_on_screen 和 wait_until_appears

4. 错误恢复流程：

   步骤1：尝试执行原定操作（如点击坐标）
   步骤2：检查操作后状态（例如通过颜色变化、新窗口出现等确认效果）
   步骤3：若无预期变化，则判定为失败，进入恢复模式
   步骤4：使用保存的图像模板在屏幕上搜索目标
   步骤5：若找到新位置，更新坐标并重试操作
   步骤6：最多重试3次，每次间隔1秒，可配置
   步骤7：仍失败则抛出 RecoverableMacroError，允许外部处理或跳过

5. 增强功能特性：

   - 自动等待机制：在关键步骤前自动等待目标出现（超时可设）
   - 日志记录：详细记录每次重试过程，便于调试
   - 回退策略：支持执行预设的“纠错宏”片段，如重新登录、刷新页面等
   - 用户干预接口：提供 pause-on-error 模式，允许人工介入后继续

实现代码结构（核心类框架）：

class RobustMacroExecutor:
    def __init__(self, automation_backend=None):
        self.automation = automation_backend or DesktopAutomation()
        self.max_retries = 3
        self.retry_interval = 1.0
        self.recovery_enabled = True

    def execute_step(self, step):
        action = step['action']
        target_pos = step.get('position')
        template_path = step.get('target_template')

        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    # 首次尝试使用原始坐标
                    result = self._perform_action(action, target_pos)
                else:
                    # 恢复模式：尝试视觉定位
                    if not template_path:
                        raise MacroExecutionError("No template for recovery")
                    located_pos = self.automation.find_image_region(template_path)
                    if located_pos:
                        updated_pos = self._center_of_region(located_pos)
                        result = self._perform_action(action, updated_pos)
                        # 更新step中的位置供后续使用
                        step['position'] = updated_pos
                    else:
                        raise MacroLocateError(f"Cannot find {template_path}")

                # 验证操作是否生效
                if self._verify_effect(step):
                    return result
                else:
                    raise MacroEffectError("Action executed but no expected change")

            except (MacroLocateError, MacroEffectError) as e:
                if attempt < self.max_retries and self.recovery_enabled:
                    time.sleep(self.retry_interval)
                    continue
                else:
                    raise RecoverableMacroError(f"Failed after {attempt+1} attempts", step=step) from e

    def _verify_effect(self, step):
        # 可基于下一目标存在、颜色变化、文本出现等方式判断
        post_condition = step.get('post_condition')
        if not post_condition:
            return True  # 无验证条件视为成功
        return self.automation.check_condition(post_condition)

    def play_macro(self, macro_sequence):
        for idx, step in enumerate(macro_sequence):
            try:
                self.execute_step(step)
            except RecoverableMacroError as e:
                print(f"[ERROR] Step {idx} failed: {e}")
                if not self.handle_failure(e, macro_sequence, idx):
                    break

    def handle_failure(self, error, sequence, current_index):
        # 默认策略：记录日志，可扩展为执行恢复宏或询问用户
        recovery_macro = self._get_recovery_procedure(current_index)
        if recovery_macro:
            print(f"Attempting recovery procedure: {recovery_macro}")
            try:
                self.play_macro(recovery_macro)
                return True  # 恢复成功，继续原流程
            except Exception as re:
                print(f"Recovery failed: {re}")
        return False  # 不可恢复，停止执行

6. 与现有模块集成方式：

   - 修改 macro_recorder_prototype.py 中的 playback 函数，使用 RobustMacroExecutor 替代直接坐标回放
   - 在录制过程中，每当检测到可识别控件（按钮、输入框等），自动截取其图像并保存为模板文件
   - 将模板路径写入对应事件对象中，形成 enriched event record
   - desktop_automation.py 需暴露 find_image_region, check_condition 等接口供本模块调用

7. 配置选项：

   - enable_visual_recovery: 是否启用视觉恢复
   - recovery_timeout: 单次查找超时时间
   - confidence_threshold: 图像匹配最低置信度
   - auto_retry_count: 自动重试次数
   - snapshot_interval: 录制时截图频率控制

结论：

通过引入 RobustMacroExecutor 模块，有效填补了 macro_recorder_prototype 与 desktop_automation 之间的集成缺口。该模块利用视觉反馈实现动态目标定位，增强了宏执行的鲁棒性。在面对界面变化、延迟加载等常见问题时，能够自动恢复而非直接失败，显著提升自动化脚本的可用性和稳定性。建议将此模块作为默认执行引擎，并逐步迁移旧有回放逻辑。