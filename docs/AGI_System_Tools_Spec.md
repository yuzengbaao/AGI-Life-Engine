# AGI System Tools & Execution Specification
## (AGI 系统工具与执行规范)

**Version**: 1.0
**Date**: 2025-12-12
**Status**: Implemented
**Module**: `core/system_tools.py`

## 1. 概述 (Overview)
本规范定义了 AGI 在 Windows 宿主环境下的**系统级执行能力 (System Execution Capabilities)**。作为 AGI 的“双手”之一（另一只手是 DesktopController），本模块负责处理所有非 GUI 的底层系统交互。

## 2. 设计原则 (Design Principles)
1.  **安全性 (Safety)**: 所有文件操作限制在工作目录内（Path Traversal Protection）。
2.  **原子性 (Atomicity)**: 每个工具函数执行单一、明确的任务。
3.  **可观测性 (Observability)**: 所有操作必须返回明确的 `Success` 或 `Error` 状态字符串，供 LLM 认知层解析。
4.  **无状态性 (Statelessness)**: 工具本身不保存状态，依赖调用者（Global Workspace）维持上下文。

## 3. 接口定义 (Interface Definition)

### 3.1 文件系统操作 (File System)
| 方法名 | 参数 | 返回值 | 描述 |
| :--- | :--- | :--- | :--- |
| `write_file` | `filename` (str), `content` (str) | `str` (Status) | 创建或覆盖文件。强制使用 UTF-8。 |
| `read_file` | `filename` (str), `limit` (int=2000) | `str` (Content) | 读取文件内容。支持截断以保护上下文窗口。 |
| `list_files` | `directory` (str=".") | `str` (List) | 列出指定目录下的非隐藏文件。 |

### 3.2 终端与进程控制 (Terminal & Process)
| 方法名 | 参数 | 返回值 | 描述 |
| :--- | :--- | :--- | :--- |
| `run_command` | `command` (str) | `str` (Output) | 执行 Shell 命令 (PowerShell/CMD)。包含 30s 超时机制。 |
| `run_python_script` | `script_name` (str) | `str` (Output) | 专用接口，用于运行生成的 Python 代码。 |

## 4. 错误处理规范 (Error Handling)
- **输入验证**: 必须检查 `filename` 是否包含 `..` 或绝对路径试图逃逸工作区。
- **异常捕获**: 所有 OS 级异常（PermissionError, FileNotFoundError）必须被捕获并转化为人类可读的字符串返回，**严禁**直接抛出导致 AGI 崩溃。
- **超时机制**: `run_command` 必须设置 `timeout=30`，防止 AGI 因执行死循环脚本而挂起。

## 5. 集成计划 (Integration Plan)
- **调用方**: `AGI_Life_Engine.py` -> `_execute_task`
- **触发逻辑**: 
    - 意图识别为 `write file` -> 调用 `write_file`
    - 意图识别为 `run command` -> 调用 `run_command`
- **认知反馈**: 执行结果直接写入 `Global Workspace` 的 `sensory_buffer`，供下一轮 `Reflect` 阶段评估。

---
*本规范作为 `SystemTools` 模块的开发与验收标准。*
