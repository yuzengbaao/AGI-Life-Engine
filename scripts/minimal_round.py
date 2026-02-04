#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小对话回合验证脚本
- 验证 ActiveAGIWrapper 6步流程能跑通
- 验证 MotivationSystem 的主动建议生成
- 验证 EnhancedToolManager 的工具调用接口
"""

import asyncio
import os
import sys

# 将项目根目录加入模块搜索路径，确保脚本可直接运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from enhanced_llm_core import EnhancedLLMCore
from active_agi_wrapper import ActiveAGIWrapper
from enhanced_tools_collection import get_tool_manager


async def main():
    # 初始化 LLM 核心（内置记忆系统）
    llm_core = EnhancedLLMCore()
    memory = llm_core.memory_system

    # 创建并启动 Active AGI 包装器
    wrapper = ActiveAGIWrapper(memory, llm_core)
    await wrapper.start()

    # 跑通一次最小对话回合
    result = await wrapper.process_user_input("做一个简单的系统自检，然后给出2条主动建议。", user_id="tester")
    print("\n=== 最小对话回合结果 ===")
    print({k: result[k] for k in [
        'dominant_motivation', 'actions', 'tasks', 'tasks_completed', 'reward'
    ] if k in result})

    # 生成主动建议
    suggestions = await wrapper.get_proactive_suggestions(count=2)
    print("\n=== 主动建议(2条) ===")
    for i, s in enumerate(suggestions, 1):
        desc = s.get('description', s.get('content', '无描述'))
        print(f"{i}. [{s.get('motivation')}] {s.get('type')} - {desc}")

    # 工具调用验证（通过工具管理器调用系统信息）
    tm = get_tool_manager()
    tool_res = tm.execute_tool("system_info", info_type="cpu")
    print("\n=== 工具调用(system_info) ===")
    if tool_res.success:
        cpu = tool_res.data.get('cpu', {})
        print(f"CPU核数: {cpu.get('count')}, 使用率: {cpu.get('percent')}%")
    else:
        print(f"工具调用失败: {tool_res.error}")


if __name__ == "__main__":
    asyncio.run(main())
