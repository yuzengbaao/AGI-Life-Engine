#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时间查询强制拦截器
[P0 修复] 覆盖基座模型的"意志"，强制使用 system_time 工具

问题：基座模型（如 Qwen）具有强烈的预训练行为，
     常规提示词难以完全覆盖其使用 web_search 的倾向

解决方案：在代码层面强制拦截时间查询，绕过 LLM 判断
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TimeQueryOverride:
    """
    时间查询强制拦截器

    功能：
    1. 检测用户输入中的时间查询意图
    2. 强制生成 system_time 工具调用
    3. 覆盖 LLM 的原始响应
    """

    # 时间查询关键词模式
    TIME_QUERY_PATTERNS = [
        # 直接时间查询
        r'现在.*?[几点时分时刻]',
        r'当前.*?[时间时刻]',
        r'今天.*?[几点时分时刻]',
        r'.*?[几时点]了',
        r'几点了',
        r'现在时间',

        # 日期查询
        r'今天.*?[日期号天]',
        r'今天是',
        r'当前日期',
        r'今天几号',
        r'现在是.*?[年月日]',

        # 时间相关的询问
        r'查询.*时间',
        r'告诉我.*时间',
        r'看看.*时间',
        r'时间多少',
        r'什么时间',

        # 英文查询
        r'what time',
        r'current time',
        r'what.*date',
        r'what.*day',
        r'tell me.*time',
    ]

    @classmethod
    def detect_time_query(cls, user_input: str) -> bool:
        """
        检测用户输入是否为时间查询

        Args:
            user_input: 用户输入文本

        Returns:
            bool: 如果是时间查询返回 True
        """
        if not user_input:
            return False

        user_input_lower = user_input.lower()

        # 检查所有模式
        for pattern in cls.TIME_QUERY_PATTERNS:
            if re.search(pattern, user_input_lower):
                logger.debug(f"[TimeQueryOverride] 检测到时间查询: {user_input[:50]}")
                return True

        return False

    @classmethod
    def generate_tool_call(cls, user_input: str) -> Optional[Dict[str, Any]]:
        """
        为时间查询生成工具调用

        Args:
            user_input: 用户输入文本

        Returns:
            工具调用字典，或 None
        """
        if not cls.detect_time_query(user_input):
            return None

        # 根据查询类型决定使用哪个操作
        user_input_lower = user_input.lower()

        # 检测是否需要日期格式
        if any(keyword in user_input_lower for keyword in
               ['几号', '日期', '年', '月', 'day', 'date']):
            return {
                'tool_name': 'system_time',
                'params': {
                    '_method': 'format',
                    'format': '%Y年%m月%d日'
                },
                'format': 'forced_time_query',
                'reason': 'Detected date query'
            }

        # 默认：获取当前时间
        return {
            'tool_name': 'system_time',
            'params': {
                '_method': 'get'
            },
            'format': 'forced_time_query',
            'reason': 'Detected time query'
        }

    @classmethod
    def override_llm_response(cls, user_input: str, llm_response: str) -> str:
        """
        覆盖 LLM 的响应，强制使用 system_time 工具

        Args:
            user_input: 用户输入
            llm_response: LLM 的原始响应

        Returns:
            修改后的响应（如果检测到时间查询）
        """
        # 检测是否为时间查询
        if not cls.detect_time_query(user_input):
            return llm_response

        # 生成工具调用
        tool_call = cls.generate_tool_call(user_input)

        if not tool_call:
            return llm_response

        # 构造强制工具调用字符串
        tool_name = tool_call['tool_name']
        params = tool_call['params']

        # 格式化参数
        if params.get('_method') == 'format':
            format_str = params.get('format', '%Y-%m-%d %H:%M:%S')
            tool_call_str = f'TOOL_CALL: {tool_name}.format(format="{format_str}")'
        else:
            tool_call_str = f'TOOL_CALL: {tool_name}.get()'

        logger.warning(f"[TimeQueryOverride] 强制覆盖 LLM 响应")
        logger.warning(f"  原始用户输入: {user_input[:100]}")
        logger.warning(f"  强制工具调用: {tool_call_str}")

        # 返回强制的工具调用
        return f"系统检测到时间查询，自动调用系统时间工具。\n\n{tool_call_str}"

    @classmethod
    def process_before_llm(cls, user_input: str) -> Optional[str]:
        """
        在 LLM 调用之前处理用户输入

        如果检测到时间查询，直接返回工具调用，
        跳过 LLM 处理

        Args:
            user_input: 用户输入

        Returns:
            如果是时间查询，返回工具调用字符串
            否则返回 None，继续正常流程
        """
        if not cls.detect_time_query(user_input):
            return None

        tool_call = cls.generate_tool_call(user_input)

        if not tool_call:
            return None

        tool_name = tool_call['tool_name']
        params = tool_call['params']

        if params.get('_method') == 'format':
            format_str = params.get('format', '%Y-%m-%d %H:%M:%S')
            return f'TOOL_CALL: {tool_name}.format(format="{format_str}")'
        else:
            return f'TOOL_CALL: {tool_name}.get()'


# 测试函数
def test_time_query_detector():
    """测试时间查询检测器"""
    test_cases = [
        ("现在几点了？", True),
        ("当前时间", True),
        ("今天是几月几号？", True),
        ("查询当前时间", True),
        ("What time is it now?", True),
        ("帮我写一段代码", False),
        ("分析这个文档", False),
        ("搜索人工智能", False),
    ]

    print("=" * 60)
    print("Time Query Detector Test")
    print("=" * 60)

    for user_input, expected in test_cases:
        detected = TimeQueryOverride.detect_time_query(user_input)
        status = "[OK]" if detected == expected else "[FAIL]"
        print(f"{status} '{user_input}' -> {detected} (expected: {expected})")

    print("=" * 60)
    print("Test Completed")
    print("=" * 60)


if __name__ == "__main__":
    test_time_query_detector()
