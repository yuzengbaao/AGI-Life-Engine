#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
失败原因归因引擎 (Failure Attribution Engine)
===============================================

元认知层第三组件：分析失败原因并区分架构问题与数据问题

功能：
- 失败原因归因（是架构问题还是数据问题？）
- 根因分析（失败的根本原因是什么？）
- 改进路径生成（如何改进？）
- 失败模式识别（避免重复失败）

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class FailureType(Enum):
    """失败类型"""
    DATA_INSUFFICIENCY = "data_insufficiency"  # 数据不足
    CAPABILITY_MISSING = "capability_missing"  # 能力缺失
    ARCHITECTURE_LIMITATION = "architecture_limitation"  # 架构限制
    EXECUTION_ERROR = "execution_error"  # 执行错误
    LOGIC_ERROR = "logic_error"  # 逻辑错误
    EXTERNAL_FAILURE = "external_failure"  # 外部失败
    UNKNOWN = "unknown"  # 未知


class RootCause(Enum):
    """根因类别"""
    ARCHITECTURAL = "architectural"  # 架构问题（需要重构）
    DATA = "data"  # 数据问题（需要更多训练）
    IMPLEMENTATION = "implementation"  # 实现问题（需要调试）
    ENVIRONMENT = "environment"  # 环境问题（外部限制）
    KNOWLEDGE = "knowledge"  # 知识问题（需要学习）
    UNKNOWN = "unknown"  # 未知原因（无法确定）


@dataclass
class FailureAnalysis:
    """失败分析结果"""
    task: str
    failure_type: FailureType
    root_cause: RootCause
    confidence: float  # 0.0-1.0
    evidence: List[str] = field(default_factory=list)
    attribution_chain: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    prevention_strategies: List[str] = field(default_factory=list)
    estimated_fix_effort: str = "unknown"  # trivial, easy, medium, hard, expert


class FailureAttributionEngine:
    """
    失败原因归因引擎

    核心功能：
    1. 分析失败日志和结果
    2. 区分架构问题与数据问题
    3. 追溯失败根因
    4. 生成改进建议
    """

    def __init__(self, capability_matcher=None):
        """
        初始化归因引擎

        Args:
            capability_matcher: 能力匹配器引用
        """
        self.capability_matcher = capability_matcher

        # 失败模式数据库
        self.failure_patterns = self._initialize_failure_patterns()

        # 归因规则
        self.attribution_rules = self._initialize_attribution_rules()

    def _initialize_failure_patterns(self) -> Dict[str, Dict]:
        """初始化失败模式数据库"""
        return {
            "no_matching_capability": {
                "type": FailureType.CAPABILITY_MISSING,
                "root_cause": RootCause.ARCHITECTURAL,
                "indicators": ["无法找到匹配的能力", "能力注册表中无对应项"],
                "fix_category": "需要增加新能力或工具"
            },
            "worldmodel_failure": {
                "type": FailureType.DATA_INSUFFICIENCY,
                "root_cause": RootCause.DATA,
                "indicators": ["WorldModel预测失败", "no sufficient data"],
                "fix_category": "需要更多训练数据"
            },
            "execution_exception": {
                "type": FailureType.EXECUTION_ERROR,
                "root_cause": RootCause.IMPLEMENTATION,
                "indicators": ["Exception", "Error", "Traceback"],
                "fix_category": "需要调试代码"
            },
            "low_confidence": {
                "type": FailureType.LOGIC_ERROR,
                "root_cause": RootCause.DATA,
                "indicators": ["置信度 < 0.5", "不确定性高"],
                "fix_category": "需要更多上下文信息"
            },
            "external_service_failure": {
                "type": FailureType.EXTERNAL_FAILURE,
                "root_cause": RootCause.ENVIRONMENT,
                "indicators": ["API错误", "网络超时", "服务不可用"],
                "fix_category": "需要检查外部依赖"
            },
            "planner_failure": {
                "type": FailureType.ARCHITECTURE_LIMITATION,
                "root_cause": RootCause.ARCHITECTURAL,
                "indicators": ["Planner无法生成计划", "计划分解失败"],
                "fix_category": "需要增强规划能力"
            },
            "tool_call_failure": {
                "type": FailureType.EXECUTION_ERROR,
                "root_cause": RootCause.IMPLEMENTATION,
                "indicators": ["工具调用失败", "参数错误"],
                "fix_category": "需要修复工具调用逻辑"
            },
        }

    def _initialize_attribution_rules(self) -> List[Dict]:
        """初始化归因规则"""
        return [
            {
                "name": "能力缺失优先规则",
                "condition": lambda task, result: "无法" in result or "缺少" in result,
                "attribution": RootCause.ARCHITECTURAL,
                "confidence": 0.9
            },
            {
                "name": "异常错误优先规则",
                "condition": lambda task, result: any(e in str(result) for e in ["Exception", "Error", "Traceback"]),
                "attribution": RootCause.IMPLEMENTATION,
                "confidence": 0.85
            },
            {
                "name": "低置信度优先规则",
                "condition": lambda task, result: isinstance(result, dict) and result.get("confidence", 1.0) < 0.5,
                "attribution": RootCause.DATA,
                "confidence": 0.7
            },
            {
                "name": "WorldModel失败优先规则",
                "condition": lambda task, result: "no sufficient data" in str(result),
                "attribution": RootCause.DATA,
                "confidence": 0.95
            },
            {
                "name": "外部服务失败优先规则",
                "condition": lambda task, result: any(e in str(result).lower() for e in ["timeout", "unavailable", "404", "500"]),
                "attribution": RootCause.ENVIRONMENT,
                "confidence": 0.8
            },
        ]

    def analyze(self, task: str, result: Any, context: Optional[Dict] = None) -> FailureAnalysis:
        """
        分析失败原因

        Args:
            task: 任务描述
            result: 执行结果（可能包含错误信息）
            context: 额外上下文

        Returns:
            FailureAnalysis: 详细的失败分析
        """
        print(f"\n{'='*60}")
        print(f"[MetaCognitive] 失败原因归因分析")
        print(f"{'='*60}")
        print(f"任务: {task}")
        print(f"结果: {str(result)[:100]}")

        # 1. 识别失败模式
        failure_type, patterns = self._identify_failure_type(result)

        # 2. 应用归因规则
        root_cause, confidence = self._apply_attribution_rules(task, result, patterns)

        # 3. 收集证据
        evidence = self._collect_evidence(task, result, context)

        # 4. 构建归因链
        attribution_chain = self._build_attribution_chain(failure_type, root_cause, evidence)

        # 5. 生成改进建议
        improvements = self._generate_improvements(failure_type, root_cause, evidence)

        # 6. 生成预防策略
        preventions = self._generate_preventions(failure_type, patterns)

        # 7. 评估修复工作量
        fix_effort = self._estimate_fix_effort(failure_type, root_cause)

        # 构建分析结果
        analysis = FailureAnalysis(
            task=task,
            failure_type=failure_type,
            root_cause=root_cause,
            confidence=confidence,
            evidence=evidence,
            attribution_chain=attribution_chain,
            improvement_suggestions=improvements,
            prevention_strategies=preventions,
            estimated_fix_effort=fix_effort
        )

        # 输出分析结果
        self._print_analysis(analysis)

        return analysis

    def _identify_failure_type(self, result: Any) -> Tuple[FailureType, List[str]]:
        """识别失败类型"""
        matched_patterns = []
        result_str = str(result).lower()

        for pattern_name, pattern_info in self.failure_patterns.items():
            if any(indicator.lower() in result_str for indicator in pattern_info["indicators"]):
                matched_patterns.append(pattern_name)

        # 确定主要失败类型
        if "no_matching_capability" in matched_patterns:
            failure_type = FailureType.CAPABILITY_MISSING
        elif "worldmodel_failure" in matched_patterns:
            failure_type = FailureType.DATA_INSUFFICIENCY
        elif "execution_exception" in matched_patterns or "tool_call_failure" in matched_patterns:
            failure_type = FailureType.EXECUTION_ERROR
        elif "external_service_failure" in matched_patterns:
            failure_type = FailureType.EXTERNAL_FAILURE
        elif "low_confidence" in matched_patterns:
            failure_type = FailureType.LOGIC_ERROR
        elif "planner_failure" in matched_patterns:
            failure_type = FailureType.ARCHITECTURE_LIMITATION
        else:
            failure_type = FailureType.UNKNOWN

        return failure_type, matched_patterns

    def _apply_attribution_rules(self, task: str, result: Any, patterns: List[str]) -> Tuple[RootCause, float]:
        """应用归因规则"""
        # 收集所有匹配规则的归因建议
        attributions = []

        for rule in self.attribution_rules:
            try:
                if rule["condition"](task, result):
                    attributions.append((rule["attribution"], rule["confidence"]))
            except Exception as e:
                # 规则应用失败，跳过
                continue

        # 如果没有规则匹配，使用默认归因
        if not attributions:
            # 根据失败模式推断
            if any(p in self.failure_patterns for p in patterns):
                pattern_info = self.failure_patterns[patterns[0]]
                root_cause = pattern_info["root_cause"]
                confidence = 0.5
            else:
                root_cause = RootCause.UNKNOWN
                confidence = 0.3
        else:
            # 选择置信度最高的归因
            attributions.sort(key=lambda x: x[1], reverse=True)
            root_cause, confidence = attributions[0]

        return root_cause, confidence

    def _collect_evidence(self, task: str, result: Any, context: Optional[Dict]) -> List[str]:
        """收集证据"""
        evidence = []

        result_str = str(result)

        # 检查是否包含错误信息
        if "Exception" in result_str or "Error" in result_str:
            evidence.append("包含异常错误信息")

        # 检查是否包含置信度
        if "confidence" in result_str.lower() or "置信度" in result_str:
            evidence.append("包含置信度评估")

        # 检查是否提及数据不足
        if "sufficient data" in result_str or "数据不足" in result_str:
            evidence.append("明确提及数据不足")

        # 检查是否提及能力缺失
        if "无法" in result_str or "不支持" in result_str:
            evidence.append("明确提及能力限制")

        # 检查上下文信息
        if context:
            if context.get("execution_success") == False:
                evidence.append("执行标记为失败")
            if context.get("planner_success") == False:
                evidence.append("规划阶段失败")

        return evidence

    def _build_attribution_chain(self, failure_type: FailureType, root_cause: RootCause, evidence: List[str]) -> List[str]:
        """构建归因链"""
        chain = []

        chain.append(f"1. 失败类型: {failure_type.value}")
        chain.append(f"2. 根因分析: {root_cause.value}")
        chain.append(f"3. 证据支持: {len(evidence)}项")

        # 详细归因链
        if root_cause == RootCause.ARCHITECTURAL:
            chain.append("4. 归因结论: 这是架构层面的问题")
            chain.append("5. 解决方向: 需要重构或扩展系统能力")
        elif root_cause == RootCause.DATA:
            chain.append("4. 归因结论: 这是数据层面的问题")
            chain.append("5. 解决方向: 需要更多训练数据或上下文")
        elif root_cause == RootCause.IMPLEMENTATION:
            chain.append("4. 归因结论: 这是实现层面的问题")
            chain.append("5. 解决方向: 需要调试和修复代码")
        elif root_cause == RootCause.ENVIRONMENT:
            chain.append("4. 归因结论: 这是环境层面的问题")
            chain.append("5. 解决方向: 需要检查外部依赖")
        elif root_cause == RootCause.KNOWLEDGE:
            chain.append("4. 归因结论: 这是知识层面的问题")
            chain.append("5. 解决方向: 需要学习相关知识")

        return chain

    def _generate_improvements(self, failure_type: FailureType, root_cause: RootCause, evidence: List[str]) -> List[str]:
        """生成改进建议"""
        improvements = []

        if failure_type == FailureType.CAPABILITY_MISSING:
            improvements.append("实现缺失的能力模块")
            improvements.append("集成外部工具或服务")
            improvements.append("降低任务复杂度，分解为子任务")

        elif failure_type == FailureType.DATA_INSUFFICIENCY:
            improvements.append("收集更多训练样本")
            improvements.append("增强WorldModel训练")
            improvements.append("利用外部知识库补充数据")

        elif failure_type == FailureType.EXECUTION_ERROR:
            improvements.append("调试执行代码逻辑")
            improvements.append("增强错误处理机制")
            improvements.append("添加执行前验证")

        elif failure_type == FailureType.ARCHITECTURE_LIMITATION:
            improvements.append("重构相关组件")
            improvements.append("增强Planner规划能力")
            improvements.append("引入更灵活的架构")

        elif failure_type == FailureType.LOGIC_ERROR:
            improvements.append("提供更多上下文信息")
            improvements.append("改进提示词模板")
            improvements.append("增加推理步骤")

        elif failure_type == FailureType.EXTERNAL_FAILURE:
            improvements.append("检查外部服务状态")
            improvements.append("实现重试机制")
            improvements.append("添加降级方案")

        else:
            improvements.append("进一步分析失败原因")
            improvements.append("收集更多日志信息")

        return improvements

    def _generate_preventions(self, failure_type: FailureType, patterns: List[str]) -> List[str]:
        """生成预防策略"""
        preventions = []

        # 通用预防策略
        preventions.append("在任务执行前进行能力评估")
        preventions.append("在失败时记录详细日志")

        # 特定类型预防策略
        if failure_type == FailureType.CAPABILITY_MISSING:
            preventions.append("增强能力注册表")
            preventions.append("实现能力检测前置")

        elif failure_type == FailureType.EXECUTION_ERROR:
            preventions.append("增加代码测试覆盖")
            preventions.append("实现沙箱测试机制")

        elif failure_type == FailureType.LOGIC_ERROR:
            preventions.append("改进提示词工程")
            preventions.append("增加推理验证步骤")

        return preventions

    def _estimate_fix_effort(self, failure_type: FailureType, root_cause: RootCause) -> str:
        """评估修复工作量"""
        if root_cause == RootCause.ARCHITECTURAL:
            return "hard"  # 需要重构，工作量大
        elif root_cause == RootCause.DATA:
            return "medium"  # 需要数据收集
        elif root_cause == RootCause.IMPLEMENTATION:
            return "easy"  # 代码调试
        elif root_cause == RootCause.ENVIRONMENT:
            return "easy"  # 外部依赖问题
        elif root_cause == RootCause.KNOWLEDGE:
            return "medium"  # 学习成本
        else:
            return "unknown"

    def _print_analysis(self, analysis: FailureAnalysis):
        """打印分析结果"""
        print(f"\n{'─'*60}")
        print(f"[归因分析结果]")
        print(f"{'─'*60}")
        print(f"失败类型: {analysis.failure_type.value}")
        print(f"根因类别: {analysis.root_cause.value}")
        print(f"置信度:   {analysis.confidence:.2f}")
        print(f"修复难度: {analysis.estimated_fix_effort}")

        if analysis.evidence:
            print(f"\n📋 支持证据:")
            for i, evidence in enumerate(analysis.evidence, 1):
                print(f"  {i}. {evidence}")

        if analysis.attribution_chain:
            print(f"\n🔗 归因链:")
            for step in analysis.attribution_chain:
                print(f"  {step}")

        if analysis.improvement_suggestions:
            print(f"\n💡 改进建议:")
            for i, suggestion in enumerate(analysis.improvement_suggestions, 1):
                print(f"  {i}. {suggestion}")

        if analysis.prevention_strategies:
            print(f"\n🛡️ 预防策略:")
            for i, strategy in enumerate(analysis.prevention_strategies, 1):
                print(f"  {i}. {strategy}")

        print(f"\n{'='*60}")

        # 关键输出：根因认知
        if analysis.root_cause == RootCause.ARCHITECTURAL:
            print(f"[MetaCognitive] 🔴 架构问题: 需要重构或扩展系统")
            print(f"[MetaCognitive] ⚠️ 警告: 这不是简单的代码问题，而是架构设计的局限")
        elif analysis.root_cause == RootCause.DATA:
            print(f"[MetaCognitive] 📊 数据问题: 需要更多训练或上下文")
            print(f"[MetaCognitive] 💡 建议: 这可以通过数据收集解决")
        elif analysis.root_cause == RootCause.IMPLEMENTATION:
            print(f"[MetaCognitive] 🐛 实现问题: 需要调试代码")
            print(f"[MetaCognitive] 🔧 建议: 这是可以通过修复解决的")
        else:
            print(f"[MetaCognitive] 🔍 根因识别: {analysis.root_cause.value}")


# ============ 使用示例 ============

if __name__ == "__main__":
    print("="*60)
    print("失败原因归因引擎测试")
    print("="*60)

    engine = FailureAttributionEngine()

    # 测试1: 能力缺失导致的失败
    print("\n[测试1] 能力缺失失败")
    result1 = {
        "success": False,
        "error": "无法处理3D点云数据",
        "missing_capability": "3d_geometry"
    }
    analysis1 = engine.analyze("分析3D点云数据", result1)

    # 测试2: 数据不足导致的失败
    print("\n[测试2] 数据不足失败")
    result2 = {
        "success": False,
        "error": "WorldModel unable to predict: no sufficient data",
        "confidence": 0.3
    }
    analysis2 = engine.analyze("预测未来趋势", result2)

    # 测试3: 执行错误
    print("\n[测试3] 执行错误")
    result3 = {
        "success": False,
        "error": "Exception: FileNotFoundError",
        "traceback": "FileNotFoundError: [Errno 2] No such file or directory"
    }
    analysis3 = engine.analyze("读取配置文件", result3)
