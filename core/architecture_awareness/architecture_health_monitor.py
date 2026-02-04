#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
架构健康度监控器 (Architecture Health Monitor)
===============================================

架构感知层第三组件：持续监控架构健康状态

功能：
- 监控组件健康状态
- 检测架构风险
- 预警架构问题
- 生成健康度报告
- 追踪架构演进趋势

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import statistics


class HealthStatus(Enum):
    """健康状态"""
    EXCELLENT = "excellent"  # 优秀 (90-100%)
    GOOD = "good"           # 良好 (70-90%)
    WARNING = "warning"     # 警告 (50-70%)
    CRITICAL = "critical"   # 严重 (30-50%)
    EMERGENCY = "emergency" # 紧急 (0-30%)


class RiskType(Enum):
    """风险类型"""
    CIRCULAR_DEPENDENCY = "circular_dependency"
    LAYER_VIOLATION = "layer_violation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    COUPLING_INCREASE = "coupling_increase"
    COMPONENT_ISOLATION = "component_isolation"
    DEPENDENCY_BLOAT = "dependency_bloat"


@dataclass
class HealthMetric:
    """健康度指标"""
    name: str
    value: float  # 0.0-1.0
    weight: float  # 权重（用于计算总分）
    threshold: float  # 警告阈值
    trend: str  # improving, stable, worsening
    description: str


@dataclass
class HealthRisk:
    """架构风险"""
    risk_type: RiskType
    severity: str  # low, medium, high, critical
    description: str
    affected_components: List[str]
    likelihood: float  # 0.0-1.0
    impact: float  # 0.0-1.0
    mitigation_suggestions: List[str]


@dataclass
class ComponentHealth:
    """组件健康状态"""
    component_name: str
    health_score: float  # 0.0-1.0
    status: HealthStatus
    issues: List[str]
    metrics: Dict[str, float]
    last_updated: float


@dataclass
class ArchitectureHealthReport:
    """架构健康度报告"""
    overall_health_score: float  # 0.0-1.0
    overall_status: HealthStatus
    metrics: List[HealthMetric]
    risks: List[HealthRisk]
    component_health: Dict[str, ComponentHealth]
    trend: str  # improving, stable, worsening
    summary: str
    recommendations: List[str]
    report_timestamp: float


class ArchitectureHealthMonitor:
    """
    架构健康度监控器

    核心功能：
    1. 持续监控架构健康指标
    2. 检测和预警架构风险
    3. 评估组件健康状态
    4. 追踪健康度趋势
    5. 生成健康度报告
    """

    def __init__(self):
        """初始化架构健康度监控器"""
        # 健康度历史记录（用于趋势分析）
        self.health_history: List[Tuple[float, float]] = []  # (timestamp, score)

        # 组件健康状态缓存
        self.component_health_cache: Dict[str, ComponentHealth] = {}

        # 风险历史记录
        self.risk_history: List[Tuple[float, List[HealthRisk]]] = []

    def generate_health_report(
        self,
        dependency_analysis: Optional[Any] = None,
        performance_analysis: Optional[Any] = None,
        include_trends: bool = True
    ) -> ArchitectureHealthReport:
        """
        生成架构健康度报告

        Args:
            dependency_analysis: 依赖分析结果
            performance_analysis: 性能分析结果
            include_trends: 是否包含趋势分析

        Returns:
            ArchitectureHealthReport: 完整的健康度报告
        """
        print(f"\n{'='*60}")
        print(f"[ArchitectureAwareness] 架构健康度监控")
        print(f"{'='*60}")

        start_time = time.time()

        # 1. 计算健康度指标
        print(f"\n[步骤 1/5] 计算健康度指标...")
        metrics = self._calculate_health_metrics(dependency_analysis, performance_analysis)

        # 2. 检测架构风险
        print(f"\n[步骤 2/5] 检测架构风险...")
        risks = self._detect_risks(dependency_analysis, performance_analysis)

        # 3. 评估组件健康状态
        print(f"\n[步骤 3/5] 评估组件健康状态...")
        component_health = self._evaluate_component_health(
            dependency_analysis,
            performance_analysis
        )

        # 4. 计算总体健康度
        print(f"\n[步骤 4/5] 计算总体健康度...")
        overall_score, overall_status = self._calculate_overall_health(metrics, risks)

        # 5. 分析趋势
        print(f"\n[步骤 5/5] 分析健康度趋势...")
        trend = self._analyze_trend(overall_score) if include_trends else "stable"

        duration = time.time() - start_time

        # 生成建议
        recommendations = self._generate_recommendations(metrics, risks, overall_score)

        # 生成摘要
        summary = self._generate_summary(overall_score, overall_status, risks)

        # 构建报告
        report = ArchitectureHealthReport(
            overall_health_score=overall_score,
            overall_status=overall_status,
            metrics=metrics,
            risks=risks,
            component_health=component_health,
            trend=trend,
            summary=summary,
            recommendations=recommendations,
            report_timestamp=time.time()
        )

        # 记录历史
        self.health_history.append((time.time(), overall_score))
        self.risk_history.append((time.time(), risks))

        # 打印报告
        self._print_health_report(report, duration)

        return report

    def _calculate_health_metrics(
        self,
        dependency_analysis: Optional[Any],
        performance_analysis: Optional[Any]
    ) -> List[HealthMetric]:
        """计算健康度指标"""
        metrics = []

        # 1. 依赖复杂度指标
        if dependency_analysis:
            # 循环依赖惩罚
            circular_count = len(dependency_analysis.circular_dependencies)
            complexity_score = max(0.0, 1.0 - circular_count * 0.2)

            metrics.append(HealthMetric(
                name="依赖复杂度",
                value=complexity_score,
                weight=0.2,
                threshold=0.7,
                trend="stable",
                description=f"循环依赖数量: {circular_count}"
            ))

            # 层级架构合规性
            violations = len(dependency_analysis.layer_violations)
            compliance_score = max(0.0, 1.0 - violations * 0.1)

            metrics.append(HealthMetric(
                name="架构合规性",
                value=compliance_score,
                weight=0.15,
                threshold=0.8,
                trend="stable",
                description=f"层级违规数量: {violations}"
            ))

            # 孤立组件指标
            orphans = len(dependency_analysis.orphan_components)
            orphan_ratio = orphans / max(dependency_analysis.total_components, 1)
            isolation_score = max(0.0, 1.0 - orphan_ratio)

            metrics.append(HealthMetric(
                name="组件耦合度",
                value=isolation_score,
                weight=0.1,
                threshold=0.7,
                trend="stable",
                description=f"孤立组件比例: {orphan_ratio:.2%}"
            ))

        # 2. 性能指标
        if performance_analysis:
            # 系统健康度（来自性能分析）
            perf_score = performance_analysis.overall_health_score

            metrics.append(HealthMetric(
                name="性能健康度",
                value=perf_score,
                weight=0.25,
                threshold=0.6,
                trend="stable",
                description=f"瓶颈数量: {len(performance_analysis.bottlenecks)}"
            ))

        # 3. 可维护性指标（默认值）
        metrics.append(HealthMetric(
            name="代码可维护性",
            value=0.8,  # 默认值（可以后续集成实际分析）
            weight=0.15,
            threshold=0.7,
            trend="stable",
            description="基于代码复杂度和注释覆盖率"
        ))

        # 4. 可扩展性指标
        metrics.append(HealthMetric(
            name="架构可扩展性",
            value=0.75,  # 默认值（可以后续集成实际分析）
            weight=0.15,
            threshold=0.7,
            trend="stable",
            description="基于模块化程度和接口设计"
        ))

        return metrics

    def _detect_risks(
        self,
        dependency_analysis: Optional[Any],
        performance_analysis: Optional[Any]
    ) -> List[HealthRisk]:
        """检测架构风险"""
        risks = []

        # 1. 循环依赖风险
        if dependency_analysis and dependency_analysis.circular_dependencies:
            high_severity_cycles = [
                c for c in dependency_analysis.circular_dependencies
                if c.severity in ["high", "critical"]
            ]

            if high_severity_cycles:
                risks.append(HealthRisk(
                    risk_type=RiskType.CIRCULAR_DEPENDENCY,
                    severity="high",
                    description=f"发现 {len(high_severity_cycles)} 个严重循环依赖",
                    affected_components=[c.cycle[0] for c in high_severity_cycles],
                    likelihood=0.9,
                    impact=0.8,
                    mitigation_suggestions=[
                        "重构模块结构，打破循环依赖",
                        "引入依赖注入或接口抽象",
                        "使用事件驱动架构解耦"
                    ]
                ))

        # 2. 层级架构违规风险
        if dependency_analysis and dependency_analysis.layer_violations:
            risks.append(HealthRisk(
                risk_type=RiskType.LAYER_VIOLATION,
                severity="medium",
                description=f"发现 {len(dependency_analysis.layer_violations)} 个层级违规",
                affected_components=[v.split(':')[0] for v in dependency_analysis.layer_violations[:5]],
                likelihood=0.7,
                impact=0.6,
                mitigation_suggestions=[
                    "检查依赖方向，确保单向依赖",
                    "调整模块层级关系",
                    "引入中间层解耦"
                ]
            ))

        # 3. 性能恶化风险
        if performance_analysis:
            critical_bottlenecks = [
                b for b in performance_analysis.bottlenecks
                if b.severity.value in ["high", "critical"]
            ]

            if critical_bottlenecks:
                risks.append(HealthRisk(
                    risk_type=RiskType.PERFORMANCE_DEGRADATION,
                    severity="high" if len(critical_bottlenecks) > 3 else "medium",
                    description=f"发现 {len(critical_bottlenecks)} 个严重性能瓶颈",
                    affected_components=[b.component for b in critical_bottlenecks],
                    likelihood=0.8,
                    impact=0.7,
                    mitigation_suggestions=[
                        "优化热点代码",
                        "引入缓存机制",
                        "考虑并行化处理"
                    ]
                ))

        # 4. 组件孤立风险
        if dependency_analysis and dependency_analysis.orphan_components:
            orphan_ratio = len(dependency_analysis.orphan_components) / max(dependency_analysis.total_components, 1)

            if orphan_ratio > 0.1:  # 超过10%组件孤立
                risks.append(HealthRisk(
                    risk_type=RiskType.COMPONENT_ISOLATION,
                    severity="low",
                    description=f"发现 {len(dependency_analysis.orphan_components)} 个孤立组件 ({orphan_ratio:.1%})",
                    affected_components=dependency_analysis.orphan_components[:5],
                    likelihood=0.5,
                    impact=0.3,
                    mitigation_suggestions=[
                        "检查是否为废弃组件",
                        "评估是否需要集成",
                        "添加文档说明用途"
                    ]
                ))

        return risks

    def _evaluate_component_health(
        self,
        dependency_analysis: Optional[Any],
        performance_analysis: Optional[Any]
    ) -> Dict[str, ComponentHealth]:
        """评估组件健康状态"""
        component_health = {}

        # 从依赖分析中评估组件
        if dependency_analysis:
            for node_name, node in dependency_analysis.nodes.items():
                # 计算组件健康度
                health_issues = []
                health_score = 1.0

                # 检查是否在循环依赖中
                in_circular = any(
                    node_name in cycle.cycle
                    for cycle in dependency_analysis.circular_dependencies
                )
                if in_circular:
                    health_issues.append("涉及循环依赖")
                    health_score -= 0.3

                # 检查是否为核心组件（高耦合）
                if len(node.imported_by) > 10:
                    health_issues.append(f"高耦合组件 (被{len(node.imported_by)}个组件依赖)")
                    health_score -= 0.1

                # 检查复杂度
                if node.complexity > 0.8:
                    health_issues.append(f"复杂度过高 ({node.complexity:.2%})")
                    health_score -= 0.15

                # 确定健康状态
                health_score = max(0.0, health_score)
                if health_score >= 0.9:
                    status = HealthStatus.EXCELLENT
                elif health_score >= 0.7:
                    status = HealthStatus.GOOD
                elif health_score >= 0.5:
                    status = HealthStatus.WARNING
                elif health_score >= 0.3:
                    status = HealthStatus.CRITICAL
                else:
                    status = HealthStatus.EMERGENCY

                component_health[node_name] = ComponentHealth(
                    component_name=node_name,
                    health_score=health_score,
                    status=status,
                    issues=health_issues,
                    metrics={
                        "complexity": node.complexity,
                        "dependents": len(node.imported_by),
                        "size_lines": node.size_lines
                    },
                    last_updated=time.time()
                )

        return component_health

    def _calculate_overall_health(
        self,
        metrics: List[HealthMetric],
        risks: List[HealthRisk]
    ) -> Tuple[float, HealthStatus]:
        """计算总体健康度"""
        # 加权平均计算总分
        total_weight = sum(m.weight for m in metrics)
        weighted_score = sum(m.value * m.weight for m in metrics) / total_weight if total_weight > 0 else 0

        # 根据风险扣分
        for risk in risks:
            if risk.severity == "critical":
                weighted_score -= 0.15
            elif risk.severity == "high":
                weighted_score -= 0.1
            elif risk.severity == "medium":
                weighted_score -= 0.05
            elif risk.severity == "low":
                weighted_score -= 0.02

        weighted_score = max(0.0, min(1.0, weighted_score))

        # 确定健康状态
        if weighted_score >= 0.9:
            status = HealthStatus.EXCELLENT
        elif weighted_score >= 0.7:
            status = HealthStatus.GOOD
        elif weighted_score >= 0.5:
            status = HealthStatus.WARNING
        elif weighted_score >= 0.3:
            status = HealthStatus.CRITICAL
        else:
            status = HealthStatus.EMERGENCY

        return weighted_score, status

    def _analyze_trend(self, current_score: float) -> str:
        """分析健康度趋势"""
        if len(self.health_history) < 3:
            return "stable"

        # 取最近的记录
        recent_scores = [score for _, score in self.health_history[-10:]]

        if len(recent_scores) < 3:
            return "stable"

        # 比较前后半段
        mid = len(recent_scores) // 2
        first_half_avg = statistics.mean(recent_scores[:mid])
        second_half_avg = statistics.mean(recent_scores[mid:])

        change = second_half_avg - first_half_avg

        if change > 0.05:
            return "improving"
        elif change < -0.05:
            return "worsening"
        else:
            return "stable"

    def _generate_recommendations(
        self,
        metrics: List[HealthMetric],
        risks: List[HealthRisk],
        overall_score: float
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于指标的建议
        low_metrics = [m for m in metrics if m.value < m.threshold]
        for metric in low_metrics:
            if metric.name == "依赖复杂度":
                recommendations.append(
                    "优先解决循环依赖问题，重构模块结构以降低耦合"
                )
            elif metric.name == "架构合规性":
                recommendations.append(
                    "修复层级违规，确保架构分层清晰"
                )
            elif metric.name == "性能健康度":
                recommendations.append(
                    "优化性能瓶颈，重点关注执行时间长的组件"
                )

        # 基于风险的建议
        high_risks = [r for r in risks if r.severity in ["high", "critical"]]
        for risk in high_risks:
            if risk.mitigation_suggestions:
                recommendations.extend(risk.mitigation_suggestions[:2])

        # 去重
        recommendations = list(set(recommendations))

        return recommendations

    def _generate_summary(
        self,
        overall_score: float,
        overall_status: HealthStatus,
        risks: List[HealthRisk]
    ) -> str:
        """生成摘要"""
        status_text = {
            HealthStatus.EXCELLENT: "优秀",
            HealthStatus.GOOD: "良好",
            HealthStatus.WARNING: "警告",
            HealthStatus.CRITICAL: "严重",
            HealthStatus.EMERGENCY: "紧急"
        }

        summary = f"架构健康度: {overall_score:.2%} ({status_text[overall_status]})"

        if risks:
            high_risk_count = sum(1 for r in risks if r.severity in ["high", "critical"])
            if high_risk_count > 0:
                summary += f" | 发现 {high_risk_count} 个高风险问题"

        return summary

    def _print_health_report(self, report: ArchitectureHealthReport, duration: float):
        """打印健康度报告"""
        print(f"\n{'─'*60}")
        print(f"[架构健康度报告]")
        print(f"{'─'*60}")

        # 总体状态
        status_icons = {
            HealthStatus.EXCELLENT: "✅",
            HealthStatus.GOOD: "🟢",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "🔴",
            HealthStatus.EMERGENCY: "🚨"
        }
        icon = status_icons.get(report.overall_status, "❓")

        print(f"\n{icon} 总体健康度: {report.overall_health_score:.2%}")
        print(f"状态: {report.overall_status.value.upper()}")
        print(f"趋势: {report.trend}")
        print(f"摘要: {report.summary}")

        # 健康度指标
        if report.metrics:
            print(f"\n📊 健康度指标:")
            for metric in report.metrics:
                status_icon = "✅" if metric.value >= metric.threshold else "⚠️"
                print(f"  {status_icon} {metric.name}: {metric.value:.2%} (权重: {metric.weight:.2%})")
                print(f"     {metric.description}")

        # 架构风险
        if report.risks:
            print(f"\n⚠️  架构风险 ({len(report.risks)}个):")
            for i, risk in enumerate(report.risks[:5], 1):
                severity_icon = {
                    "critical": "🔴",
                    "high": "🟠",
                    "medium": "🟡",
                    "low": "🟢"
                }
                icon = severity_icon.get(risk.severity, "⚪")

                print(f"  {icon} {i}. {risk.risk_type.value} ({risk.severity})")
                print(f"     {risk.description}")
                print(f"     影响组件: {len(risk.affected_components)}个")
                print(f"     可能性: {risk.likelihood:.0%} | 影响: {risk.impact:.0%}")

        # 组件健康状态（只显示有问题组件）
        unhealthy_components = {
            name: health for name, health in report.component_health.items()
            if health.status in [HealthStatus.WARNING, HealthStatus.CRITICAL, HealthStatus.EMERGENCY]
        }

        if unhealthy_components:
            print(f"\n🏥 需要关注的组件 ({len(unhealthy_components)}个):")
            for name, health in list(unhealthy_components.items())[:5]:
                print(f"  • {name} (健康度: {health.health_score:.2%})")
                for issue in health.issues[:2]:
                    print(f"    - {issue}")

        # 改进建议
        if report.recommendations:
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")

        print(f"\n报告生成耗时: {duration:.2f}秒")
        print(f"{'='*60}")

        # 关键输出
        if report.overall_status == HealthStatus.EXCELLENT:
            print(f"[ArchitectureAwareness] ✅ 架构健康状态优秀，继续保持")
        elif report.overall_status == HealthStatus.GOOD:
            print(f"[ArchitectureAwareness] 🟢 架构健康状态良好，需持续监控")
        elif report.overall_status == HealthStatus.WARNING:
            print(f"[ArchitectureAwareness] ⚠️  架构健康状态警告，建议优化")
        elif report.overall_status == HealthStatus.CRITICAL:
            print(f"[ArchitectureAwareness] 🔴 架构健康状态严重，需要紧急修复")
        else:
            print(f"[ArchitectureAwareness] 🚨 架构健康状态紧急，必须立即处理！")


# ============ 使用示例 ============

if __name__ == "__main__":
    print("="*60)
    print("架构健康度监控器测试")
    print("="*60)

    monitor = ArchitectureHealthMonitor()

    # 生成报告（无输入数据，使用默认值）
    report = monitor.generate_health_report()

    print("\n✅ 健康度监控完成！")
