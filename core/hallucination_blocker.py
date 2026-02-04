#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幻觉阻断层 (Hallucination Blocker)
=================================

核心功能：
1. 检测LLM输出中依赖工具结果的断言
2. 验证这些断言是否有真实工具回执支持
3. 阻断/删除没有证据支持的断言

作者：AGI Self-Improvement Module
创建日期：2026-01-17
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Assertion:
    """LLM输出中的断言"""
    text: str
    claimed_tool: Optional[str]
    claimed_result: Optional[str]
    line_number: int
    verified: bool = False
    blocked: bool = False
    block_reason: str = ""


class HallucinationBlocker:
    """
    幻觉阻断器
    
    工作原理：
    1. 扫描LLM输出，识别工具相关断言
    2. 对比实际工具执行结果
    3. 阻断/标注没有证据支持的断言
    """
    
    def __init__(self):
        # 断言模式：识别LLM声称工具执行结果的模式
        self.assertion_patterns = [
            # 模式1: "xxx返回/得到yyy"
            r'(\w+)[\s\.]*(?:返回|得到|输出|显示|生成|执行成功)\s*[:：]?\s*(.+)',
            # 模式2: "调用xxx() → yyy"
            r'(?:调用|执行|使用)\s*[\`\'"]?(\w+)[\`\'"]?.*?→\s*(.+)',
            # 模式3: "xxx.method() → result"
            r'(\w+\.\w+)\s*\([^)]*\)\s*→\s*(.+)',
            # 模式4: "[工具名] 状态: 成功/失败"
            r'\[(\w+)\]\s*状态\s*[:：]\s*(成功|失败|✅|❌)',
            # 模式5: "证据: tool.method = result"
            r'证据\s*[:：]\s*(\w+\.\w+)\s*=\s*(.+)',
        ]
        
        # 成功声称模式
        self.success_patterns = [
            r'已成功',
            r'执行成功',
            r'✅',
            r'创建完成',
            r'写入完成',
            r'保存成功',
        ]
        
        logger.info("[幻觉阻断器] 初始化完成")
    
    def process(
        self,
        llm_output: str,
        tool_results: List[Dict[str, Any]]
    ) -> Tuple[str, List[Assertion]]:
        """
        处理LLM输出，阻断幻觉
        
        Args:
            llm_output: LLM的原始输出
            tool_results: 实际工具执行结果列表
            
        Returns:
            (处理后的输出, 断言列表)
        """
        # 构建工具结果索引
        tool_result_index = self._build_result_index(tool_results)
        
        # 扫描并分析断言
        assertions = self._scan_assertions(llm_output)
        
        # 验证每个断言
        for assertion in assertions:
            self._verify_assertion(assertion, tool_result_index)
        
        # 生成处理后的输出
        processed_output = self._generate_blocked_output(llm_output, assertions)
        
        # 统计
        blocked_count = sum(1 for a in assertions if a.blocked)
        verified_count = sum(1 for a in assertions if a.verified)
        logger.info(f"[幻觉阻断器] 扫描 {len(assertions)} 个断言, 验证 {verified_count} 个, 阻断 {blocked_count} 个")
        
        return processed_output, assertions
    
    def _build_result_index(self, tool_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """构建工具结果索引"""
        index = {}
        
        for result in tool_results:
            # 提取工具名
            tool_name = (
                result.get('tool_name') or 
                result.get('tool') or 
                'unknown'
            ).lower()
            
            # 提取操作名
            params = result.get('params') if isinstance(result.get('params'), dict) else {}
            operation = (
                result.get('operation') or
                result.get('result', {}).get('operation') or
                params.get('operation') or
                params.get('_method') or
                'unknown'
            ).lower()
            
            # 提取成功状态
            success = result.get('result', {}).get('success', False)
            error = result.get('result', {}).get('error', '')
            data = result.get('result', {}).get('data', {})
            
            # 建立索引
            key = f"{tool_name}.{operation}"
            index[key] = {
                'success': success,
                'error': error,
                'data': data,
                'raw': result
            }
            
            # 也建立工具名索引（不含操作）
            index[tool_name] = {
                'success': success,
                'error': error,
                'data': data,
                'raw': result
            }
        
        return index
    
    def _scan_assertions(self, llm_output: str) -> List[Assertion]:
        """扫描LLM输出中的断言"""
        assertions = []
        lines = llm_output.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for pattern in self.assertion_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    claimed_tool = match.group(1) if match.lastindex >= 1 else None
                    claimed_result = match.group(2) if match.lastindex >= 2 else None
                    
                    assertion = Assertion(
                        text=line.strip(),
                        claimed_tool=claimed_tool,
                        claimed_result=claimed_result,
                        line_number=line_num
                    )
                    assertions.append(assertion)
                    break  # 每行只取第一个匹配
        
        return assertions
    
    def _verify_assertion(
        self,
        assertion: Assertion,
        tool_result_index: Dict[str, Dict[str, Any]]
    ):
        """验证单个断言"""
        if not assertion.claimed_tool:
            # 没有声称工具，无法验证
            return
        
        tool_key = assertion.claimed_tool.lower()
        
        # 查找对应的工具结果
        tool_result = tool_result_index.get(tool_key)
        
        if tool_result is None:
            # 工具未执行，断言是幻觉
            assertion.blocked = True
            assertion.block_reason = f"工具 '{assertion.claimed_tool}' 未被执行"
            return
        
        # 检查成功/失败一致性
        actual_success = tool_result.get('success', False)
        
        # 检测断言是否声称成功
        claimed_success = any(
            re.search(pattern, assertion.text, re.IGNORECASE)
            for pattern in self.success_patterns
        )
        
        if claimed_success and not actual_success:
            # 声称成功但实际失败 → 幻觉
            assertion.blocked = True
            assertion.block_reason = f"工具实际执行失败: {tool_result.get('error', '未知错误')}"
        elif not claimed_success and actual_success:
            # 声称失败但实际成功 → 也是不一致，但不太常见
            assertion.verified = True  # 宽容处理
        else:
            # 一致
            assertion.verified = True
    
    def _generate_blocked_output(
        self,
        llm_output: str,
        assertions: List[Assertion]
    ) -> str:
        """生成阻断后的输出"""
        blocked_assertions = [a for a in assertions if a.blocked]
        
        if not blocked_assertions:
            return llm_output
        
        # 构建阻断报告
        block_report = "\n\n" + "═" * 50 + "\n"
        block_report += "🚫 **幻觉阻断报告**\n"
        block_report += "═" * 50 + "\n\n"
        block_report += "以下断言因缺乏工具回执支持而被阻断：\n\n"
        
        for i, assertion in enumerate(blocked_assertions, 1):
            block_report += f"**[阻断 #{i}]** (第{assertion.line_number}行)\n"
            block_report += f"- 原始断言: {assertion.text[:100]}...\n"
            block_report += f"- 声称工具: {assertion.claimed_tool}\n"
            block_report += f"- 阻断原因: {assertion.block_reason}\n\n"
        
        block_report += "═" * 50 + "\n"
        block_report += "⚠️ 上述内容为LLM幻觉，请以实际工具执行结果为准\n"
        block_report += "═" * 50 + "\n"
        
        # 在原输出中标注被阻断的行
        lines = llm_output.split('\n')
        blocked_lines = {a.line_number for a in blocked_assertions}
        
        processed_lines = []
        for line_num, line in enumerate(lines, 1):
            if line_num in blocked_lines:
                processed_lines.append(f"~~{line}~~ [🚫 幻觉已阻断]")
            else:
                processed_lines.append(line)
        
        return '\n'.join(processed_lines) + block_report


# ==================== 集成接口 ====================

_blocker_instance: Optional[HallucinationBlocker] = None


def get_hallucination_blocker() -> HallucinationBlocker:
    """获取幻觉阻断器实例"""
    global _blocker_instance
    if _blocker_instance is None:
        _blocker_instance = HallucinationBlocker()
    return _blocker_instance


def block_hallucinations(llm_output: str, tool_results: List[Dict]) -> str:
    """便捷函数：阻断LLM输出中的幻觉"""
    blocker = get_hallucination_blocker()
    processed_output, _ = blocker.process(llm_output, tool_results)
    return processed_output
