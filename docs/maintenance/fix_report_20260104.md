# 系统修复报告 - 2026-01-04 (Update)

## 1. 概述
本次维护修复了 AGI 系统启动和对话交互中的关键阻断性问题，并进行了二次迭代以解决上下文获取时的键名匹配错误。
1. `agi_chat_cli.py` 与 `DialogueController` 之间的接口不匹配。
2. `agi_system_fully_integrated.py` 中缺失 `get_visual_dashboard` 函数。
3. `DialogueController.get_recent_history` 返回的数据结构与 CLI 期望不一致。

## 2. 问题详情与修复方案

### 2.1 对话控制器接口不匹配 (已修复)
**问题描述**：
CLI 客户端调用了 `add_message` 和 `get_recent_history`，但原控制器未实现。

**修复方案**：
在 `dialogue_controller.py` 中扩展 `DialogueController` 类，添加兼容性适配方法。

### 2.2 仪表盘启动函数缺失 (已修复)
**问题描述**：
全集成系统主程序调用未定义的 `get_visual_dashboard(self)`。

**修复方案**：
在 `agi_system_fully_integrated.py` 中添加 Mock 适配器工厂函数。

### 2.3 对话历史键名不匹配 (本次新增修复)
**问题描述**：
用户反馈输入 `help` 后系统报错 `KeyError: 'content'`。
经查，`persistent_context_engine.py` 返回的字典键为 `speaker` 和 `message`，而 `agi_chat_cli.py` 期望的是 `role` and `content`。

**修复方案**：
修改 `dialogue_controller.py` 中的 `get_recent_history` 方法，在返回前对字典键名进行映射转换。

**代码变更**：
```python
    def get_recent_history(self, session_id, limit=10):
        """
        获取最近的会话历史 (兼容 CLI 接口)
        """
        history = self.get_history(session_id)
        if history and len(history) > limit:
            history = history[-limit:]
            
        # 转换键名以匹配 CLI 期望 (speaker->role, message->content)
        formatted_history = []
        if history:
            for item in history:
                formatted_item = item.copy()
                if 'speaker' in item:
                    formatted_item['role'] = item['speaker']
                if 'message' in item:
                    formatted_item['content'] = item['message']
                formatted_history.append(formatted_item)
                
        return formatted_history
```

## 3. 验证结果
- **系统状态**：
    - Terminal #8 已重启，初始化流程顺利完成。
    - 修复了上下文历史获取时的键名错误，预期对话功能现已完全正常。

## 4. 后续建议
- 统一系统内部的数据模型（如统一使用 Role/Content 或 Speaker/Message）。
