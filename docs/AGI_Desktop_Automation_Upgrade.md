# AGI 桌面自动化与任务规划升级报告
## (AGI Desktop Automation & Task Planning Upgrade Report)

**日期**: 2025-12-11
**版本**: v1.0
**状态**: 已实施 (Implemented)

### 1. 背景与灵感 (Context)
受到 **豆包手机 (Doubao Mobile Agent)** 的启发，我们将 AGI 的能力从单一的 API 调用（如仅控制 AutoCAD）扩展到了**系统级桌面自动化**。豆包手机的核心优势在于其能够跨越 APP 边界，通过视觉和模拟触控执行长链路任务。本次元更新旨在为 Windows 平台的 AGI 复刻这一能力。

### 2. 核心架构变更 (Core Architecture Changes)

#### 2.1 新增模块：桌面控制器 (`core/desktop_automation.py`)
赋予了 AGI “双手”，使其能够像人类用户一样操作 Windows 桌面环境。
*   **类名**: `DesktopController`
*   **关键能力**:
    *   `open_app(app_name)`: 通过 Win 键或运行命令启动任意应用程序（如 Chrome, Notepad, Word）。
    *   `switch_to_window(keyword)`: 智能查找并切换到包含特定关键词的窗口（支持模糊匹配）。
    *   `type_text(text)`: 模拟键盘输入文本。
    *   `press_hotkey(*keys)`: 模拟组合键（如 `Ctrl+C`, `Alt+Tab`, `Enter`）。
    *   **安全机制**: 鼠标移动到屏幕左上角可触发 `FAILSAFE` 紧急停止。

#### 2.2 引擎升级：智能任务规划器 (`AGI_Life_Engine.py`)
赋予了 AGI “前额叶”，使其具备处理复杂、多步骤任务的规划能力。
*   **任务拆解 (`_decompose_task`)**:
    *   引入了 LLM 驱动的规划层。当检测到复杂指令（如“搜索并保存”）时，不再尝试单步执行。
    *   自动将自然语言目标拆解为 **原子动作序列 (Atomic Action Sequence)**。
    *   **示例**:
        *   用户指令: *"去网上查一下这个报错，然后把解决方法写在文档里"*
        *   自动拆解:
            1.  `["Open App Chrome",`
            2.  `"Type Text 'AutoCAD Error 0x3344'",`
            3.  `"Press Hotkey Enter",`
            4.  `"Summarize results",`
            5.  `"Switch to Window Notepad",`
            6.  `"Type Text [Summary]"]`
*   **执行层更新 (`_execute_task`)**:
    *   新增了对 `Open App`, `Switch to Window`, `Type Text`, `Press Hotkey` 等指令的原生支持。

### 3. 能力提升对比 (Capability Comparison)

| 能力维度 | 升级前 (Legacy) | 升级后 (Current) |
| :--- | :--- | :--- |
| **操作范围** | 仅限 AutoCAD (通过 COM 接口) | **全桌面 (Cross-App)**：浏览器、文档、IDE、系统设置均可操作。 |
| **任务复杂度** | 只能执行单步原子指令 | **多步链式任务 (Chain of Execution)**：自动拆解、顺序执行。 |
| **交互方式** | 后台静默执行 | **前台模拟操作**：可见鼠标移动、窗口切换、文字输入（更符合“学徒”设定）。 |
| **感知-行动** | 依赖 API 返回值 | **视觉+动作闭环**：结合 VLM 视觉感知与桌面模拟操作。 |

### 4. 下一步计划 (Next Steps)
*   **视觉反馈闭环**: 目前虽然能操作，但还需要结合 `VisionObserver` 确认操作结果（例如：确认网页是否加载完成）。
*   **鼠标精准点击**: 结合 VLM 返回的坐标 (`[x, y]`) 实现点击特定按钮（如“发送”、“下载”），彻底摆脱对快捷键的依赖。

---
*本报告由 AGI 自动生成，记录了系统向“通用电脑操作代理 (General Computer Control Agent)”进化的关键一步。*
