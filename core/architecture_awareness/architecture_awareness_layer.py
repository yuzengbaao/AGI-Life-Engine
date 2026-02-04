#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
架构感知层集成包装器 (Architecture Awareness Layer Integration Wrapper)
======================================================================

将三个架构感知组件整合为统一的架构感知层

架构：
┌─────────────────────────────────────────────────────────┐
│         Architecture Awareness Layer (V2)                │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │ Component        │  │ Performance      │            │
│  │ Dependency       │  │ Bottleneck       │            │
│  │ Mapper           │  │ Analyzer         │            │
│  └────────┬─────────┘  └────────┬─────────┘            │
│           │                     │                       │
│           └──────────┬──────────┘                       │
│                      ▼                                  │
│           ┌──────────────────────┐                     │
│           │ Architecture Health  │                     │
│           │     Monitor          │                     │
│           └──────────────────────┘                     │
│                      │                                  │
│                      ▼                                  │
│           ┌──────────────────────┐                     │
│           │  Architecture        │                     │
│           │  Awareness Report    │                     │
│           └──────────────────────┘                     │
└─────────────────────────────────────────────────────────┘

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

import os
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .component_dependency_mapper import (
    ComponentDependencyMapper,
    DependencyAnalysis
)
from .performance_bottleneck_analyzer import (
    PerformanceBottleneckAnalyzer,
    PerformanceAnalysis,
    PerformanceMonitor,
    PerformanceSample,
    PerformanceMetric
)
from .architecture_health_monitor import (
    ArchitectureHealthMonitor,
    ArchitectureHealthReport,
    HealthStatus
)


@dataclass
class ArchitectureAwarenessReport:
    """架构感知综合报告"""
    # 依赖分析
    dependency_analysis: Optional[DependencyAnalysis] = None

    # 性能分析
    performance_analysis: Optional[PerformanceAnalysis] = None

    # 健康度报告
    health_report: Optional[ArchitectureHealthReport] = None

    # 综合评估
    overall_architecture_score: float = 0.0  # 0.0-1.0
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # 元数据
    analysis_timestamp: float = 0.0
    analysis_duration: float = 0.0


class ArchitectureAwarenessLayer:
    """
    架构感知层 - "理解自己的架构"

    核心功能：
    1. 分析系统组件依赖关系
    2. 监控性能瓶颈
    3. 评估架构健康度
    4. 生成综合架构报告
    5. 提供架构优化建议
    """

    def __init__(self, project_root: str):
        """
        初始化架构感知层

        Args:
            project_root: 项目根目录
        """
        self.project_root = Path(project_root)

        # 初始化三个核心组件
        self.dependency_mapper = ComponentDependencyMapper(str(self.project_root))
        self.performance_analyzer = PerformanceBottleneckAnalyzer()
        self.health_monitor = ArchitectureHealthMonitor()

        # 性能监控状态
        self._performance_monitoring_enabled = False
        self._monitor_thread = None

        print(f"[ArchitectureAwareness] 🏗️  初始化完成")
        print(f"   项目根目录: {self.project_root}")

    def analyze_comprehensive(
        self,
        include_dependency: bool = True,
        include_performance: bool = True,
        include_health: bool = True
    ) -> ArchitectureAwarenessReport:
        """
        执行完整的架构感知分析

        Args:
            include_dependency: 是否包含依赖分析
            include_performance: 是否包含性能分析
            include_health: 是否包含健康度分析

        Returns:
            ArchitectureAwarenessReport: 综合架构感知报告
        """
        print(f"\n{'='*70}")
        print(f"[Architecture Awareness Layer] 完整架构感知分析")
        print(f"{'='*70}")

        start_time = time.time()

        # 1. 依赖分析
        dependency_analysis = None
        if include_dependency:
            print(f"\n[阶段 1/3] 组件依赖分析...")
            dependency_analysis = self.dependency_mapper.analyze()

        # 2. 性能分析
        performance_analysis = None
        if include_performance:
            print(f"\n[阶段 2/3] 性能瓶颈分析...")
            performance_analysis = self.performance_analyzer.analyze(min_samples=5)

        # 3. 健康度评估
        health_report = None
        if include_health:
            print(f"\n[阶段 3/3] 架构健康度评估...")
            health_report = self.health_monitor.generate_health_report(
                dependency_analysis=dependency_analysis,
                performance_analysis=performance_analysis
            )

        duration = time.time() - start_time

        # 4. 综合评估
        overall_score, critical_issues, recommendations = self._synthesize_findings(
            dependency_analysis,
            performance_analysis,
            health_report
        )

        # 构建综合报告
        report = ArchitectureAwarenessReport(
            dependency_analysis=dependency_analysis,
            performance_analysis=performance_analysis,
            health_report=health_report,
            overall_architecture_score=overall_score,
            critical_issues=critical_issues,
            recommendations=recommendations,
            analysis_timestamp=time.time(),
            analysis_duration=duration
        )

        # 打印综合报告
        self._print_comprehensive_report(report)

        return report

    def _synthesize_findings(
        self,
        dependency_analysis: Optional[DependencyAnalysis],
        performance_analysis: Optional[PerformanceAnalysis],
        health_report: Optional[ArchitectureHealthReport]
    ) -> tuple:
        """综合分析结果"""
        scores = []
        critical_issues = []
        recommendations = []

        # 依赖健康度
        if dependency_analysis:
            # 计算依赖健康度 (0-1)
            dep_health = 1.0
            if dependency_analysis.circular_dependencies:
                dep_health -= 0.3 * len(dependency_analysis.circular_dependencies)
            if dependency_analysis.layer_violations:
                dep_health -= 0.1 * len(dependency_analysis.layer_violations)

            dep_health = max(0.0, dep_health)
            scores.append(dep_health)

            # 收集关键问题
            for dep in dependency_analysis.circular_dependencies[:3]:
                if dep.severity in ["high", "critical"]:
                    critical_issues.append(
                        f"严重循环依赖: {' -> '.join(dep.cycle)}"
                    )

            for violation in dependency_analysis.layer_violations[:3]:
                critical_issues.append(f"层级违规: {violation}")

        # 性能健康度
        if performance_analysis:
            perf_health = performance_analysis.overall_health_score
            scores.append(perf_health)

            # 收集关键问题
            critical_bottlenecks = [
                b for b in performance_analysis.bottlenecks
                if b.severity.value in ["high", "critical"]
            ]
            for bottleneck in critical_bottlenecks[:3]:
                critical_issues.append(
                    f"性能瓶颈: {bottleneck.component} - {bottleneck.metric_type.value}"
                )

        # 架构健康度
        if health_report:
            arch_health = health_report.overall_health_score
            scores.append(arch_health)

            # 收集高风险
            high_risks = [
                r for r in health_report.risks
                if r.severity in ["high", "critical"]
            ]
            for risk in high_risks[:3]:
                critical_issues.append(
                    f"架构风险: {risk.risk_type.value} - {risk.description}"
                )

            # 收集建议
            recommendations.extend(health_report.recommendations[:3])

        # 计算总分
        overall_score = sum(scores) / len(scores) if scores else 0.5

        # 生成通用建议
        if overall_score < 0.5:
            recommendations.insert(0, "架构健康度严重不足，建议立即进行全面重构")
        elif overall_score < 0.7:
            recommendations.insert(0, "架构健康度较低，建议优先处理关键问题")

        return overall_score, critical_issues, recommendations

    def _print_comprehensive_report(self, report: ArchitectureAwarenessReport):
        """打印综合报告"""
        print(f"\n{'='*70}")
        print(f"[架构感知综合报告]")
        print(f"{'='*70}")

        # 总体评分
        print(f"\n📊 总体架构评分: {report.overall_architecture_score:.2%}")

        # 评分等级
        if report.overall_architecture_score >= 0.9:
            grade = "A (优秀)"
            icon = "✅"
        elif report.overall_architecture_score >= 0.8:
            grade = "B (良好)"
            icon = "🟢"
        elif report.overall_architecture_score >= 0.7:
            grade = "C (一般)"
            icon = "⚠️"
        elif report.overall_architecture_score >= 0.6:
            grade = "D (较差)"
            icon = "🟡"
        else:
            grade = "F (差)"
            icon = "🔴"

        print(f"{icon} 评级: {grade}")

        # 关键问题
        if report.critical_issues:
            print(f"\n🚨 关键问题 ({len(report.critical_issues)}个):")
            for i, issue in enumerate(report.critical_issues[:5], 1):
                print(f"  {i}. {issue}")

        # 改进建议
        if report.recommendations:
            print(f"\n💡 改进建议:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")

        # 各分项评分
        print(f"\n📈 分项评分:")

        if report.dependency_analysis:
            dep_health = max(0.0, 1.0 - 0.3 * len(report.dependency_analysis.circular_dependencies))
            print(f"  • 依赖健康度: {dep_health:.2%}")
            print(f"    - 组件数: {report.dependency_analysis.total_components}")
            print(f"    - 依赖数: {report.dependency_analysis.total_dependencies}")
            print(f"    - 循环依赖: {len(report.dependency_analysis.circular_dependencies)}")

        if report.performance_analysis:
            print(f"  • 性能健康度: {report.performance_analysis.overall_health_score:.2%}")
            print(f"    - 瓶颈数: {len(report.performance_analysis.bottlenecks)}")

        if report.health_report:
            print(f"  • 架构健康度: {report.health_report.overall_health_score:.2%}")
            print(f"    - 状态: {report.health_report.overall_status.value}")

        print(f"\n⏱️  分析耗时: {report.analysis_duration:.2f}秒")
        print(f"{'='*70}")

        # 关键输出
        if report.overall_architecture_score >= 0.8:
            print(f"[ArchitectureAwareness] ✅ 架构状态健康，系统设计良好")
        elif report.overall_architecture_score >= 0.6:
            print(f"[ArchitectureAwareness] ⚠️  架构状态一般，有改进空间")
        else:
            print(f"[ArchitectureAwareness] 🔴 架构状态不佳，需要关注和改进")

    def get_performance_monitor(self) -> PerformanceMonitor:
        """获取性能监控装饰器"""
        return self.performance_analyzer.get_monitor()

    def export_dependency_graph(self, output_path: str):
        """导出依赖图"""
        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.dependency_mapper.export_graph(str(output_path))

    def enable_continuous_monitoring(self, interval_seconds: int = 60):
        """启用持续监控（后台线程）"""
        # TODO: 实现后台监控线程
        print(f"[ArchitectureAwareness] 🔄 持续监控功能待实现")

    def get_architecture_insights(self) -> Dict[str, Any]:
        """获取架构洞察（快速摘要）"""
        insights = {
            "project_root": str(self.project_root),
            "components": 0,
            "dependencies": 0,
            "health_score": 0.0,
            "critical_issues": 0
        }

        # 如果有缓存的分析结果，直接返回
        if self.health_monitor.health_history:
            latest_score = self.health_monitor.health_history[-1][1]
            insights["health_score"] = latest_score

        return insights


# ============ 使用示例 ============

if __name__ == "__main__":
    print("="*70)
    print("架构感知层集成测试")
    print("="*70)

    # 创建架构感知层
    arch_layer = ArchitectureAwarenessLayer(
        project_root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    # 执行完整分析
    report = arch_layer.analyze_comprehensive()

    # 导出依赖图
    arch_layer.export_dependency_graph("data/architecture/dependency_graph_full.json")

    print("\n✅ 架构感知分析完成！")
