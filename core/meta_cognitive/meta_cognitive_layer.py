#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元认知层集成包装器 (Meta-Cognitive Layer Integration Wrapper)
============================================================

将三个元认知组件整合为统一的元认知层

架构：
┌─────────────────────────────────────────────────────────┐
│              Meta-Cognitive Layer (V2)                   │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │ TaskUnderstanding│  │ CapabilityMatcher │            │
│  │   Evaluator      │  │                  │            │
│  └────────┬─────────┘  └────────┬─────────┘            │
│           │                     │                       │
│           └──────────┬──────────┘                       │
│                      ▼                                  │
│           ┌──────────────────────┐                     │
│           │ FailureAttribution   │                     │
│           │     Engine           │                     │
│           └──────────────────────┘                     │
│                      │                                  │
│                      ▼                                  │
│           ┌──────────────────────┐                     │
│           │  Meta-Cognitive      │                     │
│           │     Report           │                     │
│           └──────────────────────┘                     │
└─────────────────────────────────────────────────────────┘

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .task_understanding_evaluator import (
    TaskUnderstandingEvaluator,
    TaskAnalysis,
    UnderstandingLevel
)
from .capability_matcher import (
    CapabilityMatcher,
    MatchResult,
    MatchLevel
)
from .failure_attribution_engine import (
    FailureAttributionEngine,
    FailureAnalysis,
    FailureType,
    RootCause
)


class DecisionOutcome(Enum):
    """元认知决策结果"""
    PROCEED = "proceed"              # 继续执行
    PROCEED_WITH_CAUTION = "proceed_with_caution"  # 谨慎执行
    DECLINE = "decline"              # 拒绝执行
    ESCALATE = "escalate"            # 升级到人类


@dataclass
class MetaCognitiveReport:
    """元认知分析报告"""
    task: str

    # 任务理解评估
    task_analysis: Optional[TaskAnalysis] = None
    understanding_level: Optional[UnderstandingLevel] = None
    understanding_confidence: float = 0.0

    # 能力匹配评估
    capability_match: Optional[MatchResult] = None
    match_level: Optional[MatchLevel] = None
    capability_confidence: float = 0.0

    # 失败归因评估（仅在失败后使用）
    failure_analysis: Optional[FailureAnalysis] = None

    # 综合决策
    decision: DecisionOutcome = DecisionOutcome.PROCEED
    overall_confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)

    # 建议
    suggestions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # 元数据
    should_proceed: bool = True
    requires_human_intervention: bool = False
    estimated_success_probability: float = 0.0


class MetaCognitiveLayer:
    """
    元认知层 - "思考自己的思考"

    核心功能：
    1. 在任务执行前进行自我评估
    2. 在任务失败后进行归因分析
    3. 提供决策建议（继续/拒绝/升级）
    4. 生成详细的元认知报告
    """

    def __init__(self, knowledge_graph=None, memory_system=None):
        """
        初始化元认知层

        Args:
            knowledge_graph: 知识图谱引用
            memory_system: 记忆系统引用
        """
        # 初始化三个核心组件
        self.task_evaluator = TaskUnderstandingEvaluator(
            knowledge_graph=knowledge_graph,
            memory_system=memory_system
        )

        self.capability_matcher = CapabilityMatcher()

        self.failure_engine = FailureAttributionEngine(
            capability_matcher=self.capability_matcher
        )

        # 元认知统计
        self.stats = {
            "total_evaluations": 0,
            "proceed_count": 0,
            "decline_count": 0,
            "escalate_count": 0,
            "with_caution_count": 0,
        }

    def evaluate_before_execution(self, task: str, context: Optional[Dict] = None) -> MetaCognitiveReport:
        """
        任务执行前的元认知评估

        Args:
            task: 任务描述
            context: 额外上下文

        Returns:
            MetaCognitiveReport: 元认知分析报告
        """
        print(f"\n{'='*70}")
        print(f"[Meta-Cognitive Layer] 任务执行前评估")
        print(f"{'='*70}")
        print(f"任务: {task}")

        self.stats["total_evaluations"] += 1

        # 1. 评估任务理解深度
        print(f"\n[步骤 1/3] 评估任务理解深度...")
        task_analysis = self.task_evaluator.evaluate(task, context)

        # 2. 评估能力匹配
        print(f"\n[步骤 2/3] 评估能力匹配...")
        capability_match = self.capability_matcher.match(task, context)

        # 3. 生成综合决策
        print(f"\n[步骤 3/3] 生成综合决策...")
        decision, confidence, reasoning, suggestions, warnings = self._make_decision(
            task_analysis, capability_match
        )

        # 构建报告
        report = MetaCognitiveReport(
            task=task,
            task_analysis=task_analysis,
            understanding_level=task_analysis.understanding_level,
            understanding_confidence=task_analysis.confidence,
            capability_match=capability_match,
            match_level=capability_match.match_level,
            capability_confidence=capability_match.confidence,
            decision=decision,
            overall_confidence=confidence,
            reasoning=reasoning,
            suggestions=suggestions,
            warnings=warnings,
            should_proceed=decision in [DecisionOutcome.PROCEED, DecisionOutcome.PROCEED_WITH_CAUTION],
            requires_human_intervention=decision == DecisionOutcome.ESCALATE,
            estimated_success_probability=confidence
        )

        # 更新统计
        if decision == DecisionOutcome.PROCEED:
            self.stats["proceed_count"] += 1
        elif decision == DecisionOutcome.PROCEED_WITH_CAUTION:
            self.stats["with_caution_count"] += 1
        elif decision == DecisionOutcome.DECLINE:
            self.stats["decline_count"] += 1
        elif decision == DecisionOutcome.ESCALATE:
            self.stats["escalate_count"] += 1

        # 打印报告
        self._print_report(report)

        return report

    def analyze_after_failure(self, task: str, result: Any, context: Optional[Dict] = None) -> FailureAnalysis:
        """
        任务失败后的归因分析

        Args:
            task: 任务描述
            result: 执行结果（包含错误信息）
            context: 额外上下文

        Returns:
            FailureAnalysis: 失败归因分析
        """
        print(f"\n{'='*70}")
        print(f"[Meta-Cognitive Layer] 任务失败后归因分析")
        print(f"{'='*70}")

        # 使用失败归因引擎分析
        failure_analysis = self.failure_engine.analyze(task, result, context)

        return failure_analysis

    def _make_decision(
        self,
        task_analysis: TaskAnalysis,
        capability_match: MatchResult
    ) -> Tuple[DecisionOutcome, float, List[str], List[str], List[str]]:
        """
        基于任务分析和能力匹配生成决策

        Returns:
            (决策, 置信度, 推理链, 建议, 警告)
        """
        reasoning = []
        suggestions = []
        warnings = []

        # 计算综合置信度
        understanding_weight = 0.4
        capability_weight = 0.6
        overall_confidence = (
            task_analysis.confidence * understanding_weight +
            capability_match.confidence * capability_weight
        )

        # 决策逻辑
        # 1. 如果任务不可行 → 拒绝
        if not task_analysis.can_solve:
            reasoning.append("任务可行性评估：不可行")
            reasoning.append(f"原因：{', '.join(task_analysis.missing_capabilities)}")

            warnings.append("⚠️ 该任务超出系统能力边界")
            warnings.append(f"缺失能力：{len(task_analysis.missing_capabilities)}项")

            suggestions.extend([
                "建议：将任务分解为更小的子任务",
                "建议：寻求外部工具或专业知识支持",
            ])

            # 如果完全无匹配，建议升级
            if capability_match.match_level == MatchLevel.NONE:
                return DecisionOutcome.ESCALATE, overall_confidence, reasoning, suggestions, warnings
            else:
                return DecisionOutcome.DECLINE, overall_confidence, reasoning, suggestions, warnings

        # 2. 如果匹配度低 → 拒绝或谨慎执行
        elif capability_match.match_level in [MatchLevel.POOR, MatchLevel.NONE]:
            reasoning.append("能力匹配评估：匹配度低")
            reasoning.append(f"匹配等级：{capability_match.match_level.value}")

            warnings.append("⚠️ 系统能力不足以可靠完成此任务")
            warnings.append(f"缺失能力：{', '.join(capability_match.missing_capabilities)}")

            suggestions.extend([
                "建议：先获取缺失的能力或工具",
                "建议：尝试替代方案",
            ])
            suggestions.extend(capability_match.suggested_alternatives)

            if capability_match.match_level == MatchLevel.NONE:
                return DecisionOutcome.DECLINE, overall_confidence, reasoning, suggestions, warnings
            else:
                return DecisionOutcome.PROCEED_WITH_CAUTION, overall_confidence * 0.7, reasoning, suggestions, warnings

        # 3. 如果理解深度浅 → 谨慎执行
        elif task_analysis.understanding_level in [UnderstandingLevel.SURFACE, UnderstandingLevel.SHALLOW]:
            reasoning.append("任务理解评估：理解深度不足")
            reasoning.append(f"理解等级：{task_analysis.understanding_level.value}")

            warnings.append("⚠️ 系统对任务的理解可能不完整")
            if task_analysis.knowledge_gaps:
                warnings.append(f"知识缺口：{', '.join(task_analysis.knowledge_gaps)}")

            suggestions.extend([
                "建议：提供更多上下文信息",
                "建议：将任务描述得更具体",
                "建议：分步骤明确需求",
            ])

            # 如果能力匹配度高，可以谨慎执行
            if capability_match.match_level in [MatchLevel.PERFECT, MatchLevel.GOOD]:
                return DecisionOutcome.PROCEED_WITH_CAUTION, overall_confidence * 0.8, reasoning, suggestions, warnings
            else:
                return DecisionOutcome.PROCEED_WITH_CAUTION, overall_confidence * 0.6, reasoning, suggestions, warnings

        # 4. 如果一切良好 → 继续执行
        else:
            reasoning.append("任务理解评估：充分")
            reasoning.append(f"理解等级：{task_analysis.understanding_level.value}")

            reasoning.append("能力匹配评估：匹配良好")
            reasoning.append(f"匹配等级：{capability_match.match_level.value}")

            if task_analysis.suggested_approach:
                suggestions.append(f"建议方法：\n{task_analysis.suggested_approach}")

            return DecisionOutcome.PROCEED, overall_confidence, reasoning, suggestions, warnings

    def _print_report(self, report: MetaCognitiveReport):
        """打印元认知报告"""
        print(f"\n{'─'*70}")
        print(f"[元认知分析报告]")
        print(f"{'─'*70}")

        # 决策结果
        decision_icons = {
            DecisionOutcome.PROCEED: "✅",
            DecisionOutcome.PROCEED_WITH_CAUTION: "⚠️",
            DecisionOutcome.DECLINE: "❌",
            DecisionOutcome.ESCALATE: "🆘",
        }
        icon = decision_icons.get(report.decision, "❓")

        print(f"\n{icon} 决策结果: {report.decision.value.upper()}")
        print(f"📊 综合置信度: {report.overall_confidence:.2%}")
        print(f"🎯 预估成功概率: {report.estimated_success_probability:.2%}")

        # 推理链
        if report.reasoning:
            print(f"\n🔗 推理链:")
            for i, reason in enumerate(report.reasoning, 1):
                print(f"  {i}. {reason}")

        # 警告
        if report.warnings:
            print(f"\n⚠️ 警告:")
            for warning in report.warnings:
                print(f"  {warning}")

        # 建议
        if report.suggestions:
            print(f"\n💡 建议:")
            for i, suggestion in enumerate(report.suggestions, 1):
                print(f"  {i}. {suggestion}")

        # 详细评估结果摘要
        print(f"\n📋 评估摘要:")
        print(f"  • 任务理解: {report.understanding_level.value} (置信度: {report.understanding_confidence:.2%})")
        print(f"  • 能力匹配: {report.match_level.value} (置信度: {report.capability_confidence:.2%})")
        print(f"  • 匹配能力: {len(report.capability_match.matching_capabilities) if report.capability_match else 0}项")
        print(f"  • 缺失能力: {len(report.capability_match.missing_capabilities) if report.capability_match else 0}项")

        print(f"\n{'='*70}")

        # 关键输出
        if report.decision == DecisionOutcome.PROCEED:
            print(f"[Meta-Cognitive] ✅ 系统具备充分的理解和能力，建议继续执行")
        elif report.decision == DecisionOutcome.PROCEED_WITH_CAUTION:
            print(f"[Meta-Cognitive] ⚠️ 系统建议谨慎执行，注意潜在风险")
        elif report.decision == DecisionOutcome.DECLINE:
            print(f"[Meta-Cognitive] ❌ 系统建议拒绝此任务，超出能力边界")
        elif report.decision == DecisionOutcome.ESCALATE:
            print(f"[Meta-Cognitive] 🆘 系统建议升级到人类干预")

    def get_stats(self) -> Dict[str, Any]:
        """获取元认知统计信息"""
        total = self.stats["total_evaluations"]
        if total == 0:
            return self.stats

        return {
            **self.stats,
            "proceed_rate": self.stats["proceed_count"] / total,
            "decline_rate": self.stats["decline_count"] / total,
            "escalate_rate": self.stats["escalate_count"] / total,
            "caution_rate": self.stats["with_caution_count"] / total,
        }


# ============ 使用示例 ============

if __name__ == "__main__":
    print("="*70)
    print("元认知层集成测试")
    print("="*70)

    # 创建元认知层
    meta_layer = MetaCognitiveLayer()

    # 测试1: 简单任务（应该继续执行）
    print("\n" + "▶"*35)
    print("测试1: 简单任务")
    print("▶"*35)
    report1 = meta_layer.evaluate_before_execution("读取文件hello.txt并统计行数")

    # 测试2: 复杂任务（应该谨慎执行）
    print("\n" + "▶"*35)
    print("测试2: 复杂任务")
    print("▶"*35)
    report2 = meta_layer.evaluate_before_execution(
        "分析项目中所有Python文件的代码质量，生成优化建议报告"
    )

    # 测试3: 超出能力范围（应该拒绝）
    print("\n" + "▶"*35)
    print("测试3: 超出能力范围")
    print("▶"*35)
    report3 = meta_layer.evaluate_before_execution(
        "分析3D点云数据的几何特征，提取表面法向量"
    )

    # 测试4: 完全超出知识范围（应该升级）
    print("\n" + "▶"*35)
    print("测试4: 完全超出知识范围")
    print("▶"*35)
    report4 = meta_layer.evaluate_before_execution(
        "解释量子纠缠的物理机制及其在量子计算中的应用"
    )

    # 测试5: 失败归因分析
    print("\n" + "▶"*35)
    print("测试5: 失败归因分析")
    print("▶"*35)
    result5 = {
        "success": False,
        "error": "WorldModel unable to predict: no sufficient data",
        "confidence": 0.3
    }
    failure_analysis = meta_layer.analyze_after_failure("预测未来趋势", result5)

    # 打印统计
    print("\n" + "="*70)
    print("元认知统计")
    print("="*70)
    stats = meta_layer.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
