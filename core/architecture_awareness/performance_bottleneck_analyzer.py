#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能瓶颈分析器 (Performance Bottleneck Analyzer)
================================================

架构感知层第二组件：识别系统性能瓶颈

功能：
- 监控组件执行时间
- 分析资源使用模式
- 识别性能热点
- 评估性能趋势
- 生成优化建议

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime
import statistics


class PerformanceMetric(Enum):
    """性能指标类型"""
    EXECUTION_TIME = "execution_time"      # 执行时间
    CPU_USAGE = "cpu_usage"                # CPU使用率
    MEMORY_USAGE = "memory_usage"          # 内存使用
    IO_OPERATIONS = "io_operations"        # IO操作
    NETWORK_CALLS = "network_calls"        # 网络调用


class BottleneckSeverity(Enum):
    """瓶颈严重程度"""
    LOW = "low"              # 低（可接受）
    MEDIUM = "medium"        # 中（需关注）
    HIGH = "high"            # 高（需优化）
    CRITICAL = "critical"    # 严重（立即处理）


@dataclass
class PerformanceSample:
    """性能样本"""
    component: str
    metric_type: PerformanceMetric
    value: float
    unit: str  # ms, %, MB, etc.
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Bottleneck:
    """性能瓶颈"""
    component: str
    metric_type: PerformanceMetric
    severity: BottleneckSeverity
    current_value: float
    threshold: float
    impact: str  # 对系统的影响描述
    trend: str  # improving, stable, worsening
    suggested_actions: List[str]


@dataclass
class PerformanceTrend:
    """性能趋势"""
    component: str
    metric_type: PerformanceMetric
    trend: str  # improving, stable, worsening
    change_rate: float  # 每秒变化率
    confidence: float  # 0.0-1.0


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    component: str
    priority: str  # high, medium, low
    category: str  # caching, parallelization, algorithm, etc.
    description: str
    expected_improvement: str
    implementation_effort: str  # easy, medium, hard


@dataclass
class PerformanceAnalysis:
    """性能分析结果"""
    bottlenecks: List[Bottleneck]
    trends: List[PerformanceTrend]
    suggestions: List[OptimizationSuggestion]
    overall_health_score: float  # 0.0-1.0
    critical_components: List[str]
    analysis_timestamp: float


class PerformanceMonitor:
    """性能监控器 - 装饰器模式"""

    def __init__(self, analyzer: 'PerformanceBottleneckAnalyzer'):
        self.analyzer = analyzer

    def __call__(self, component_name: str):
        """装饰器：监控函数性能"""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_cpu = psutil.cpu_percent()
                start_mem = psutil.virtual_memory().percent

                try:
                    result = func(*args, **kwargs)

                    # 记录成功执行
                    execution_time = time.time() - start_time
                    cpu_usage = psutil.cpu_percent() - start_cpu
                    memory_usage = psutil.virtual_memory().percent - start_mem

                    self.analyzer.record_sample(
                        PerformanceSample(
                            component=component_name,
                            metric_type=PerformanceMetric.EXECUTION_TIME,
                            value=execution_time * 1000,  # 转换为毫秒
                            unit="ms",
                            timestamp=time.time(),
                            context={"status": "success"}
                        )
                    )

                    return result

                except Exception as e:
                    # 记录失败执行
                    execution_time = time.time() - start_time

                    self.analyzer.record_sample(
                        PerformanceSample(
                            component=component_name,
                            metric_type=PerformanceMetric.EXECUTION_TIME,
                            value=execution_time * 1000,
                            unit="ms",
                            timestamp=time.time(),
                            context={"status": "error", "error": str(e)}
                        )
                    )

                    raise

            return wrapper
        return decorator


class PerformanceBottleneckAnalyzer:
    """
    性能瓶颈分析器

    核心功能：
    1. 实时监控组件性能
    2. 分析性能趋势
    3. 识别性能瓶颈
    4. 生成优化建议
    5. 评估系统整体健康度
    """

    def __init__(self, max_samples: int = 1000):
        """
        初始化性能分析器

        Args:
            max_samples: 每个组件保留的最大样本数
        """
        # 性能样本存储
        self.samples: Dict[str, Dict[PerformanceMetric, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=max_samples))
        )

        # 性能阈值配置
        self.thresholds = {
            PerformanceMetric.EXECUTION_TIME: {
                "low": 100,        # ms
                "medium": 500,
                "high": 1000,
                "critical": 5000
            },
            PerformanceMetric.CPU_USAGE: {
                "low": 20,         # %
                "medium": 50,
                "high": 80,
                "critical": 95
            },
            PerformanceMetric.MEMORY_USAGE: {
                "low": 30,         # %
                "medium": 60,
                "high": 85,
                "critical": 95
            }
        }

        # 系统信息
        self.process = psutil.Process()

    def record_sample(self, sample: PerformanceSample):
        """
        记录性能样本

        Args:
            sample: 性能样本
        """
        self.samples[sample.component][sample.metric_type].append(sample)

    def analyze(self, min_samples: int = 10) -> PerformanceAnalysis:
        """
        执行完整的性能分析

        Args:
            min_samples: 最小样本数要求

        Returns:
            PerformanceAnalysis: 完整的性能分析结果
        """
        print(f"\n{'='*60}")
        print(f"[ArchitectureAwareness] 性能瓶颈分析")
        print(f"{'='*60}")

        start_time = time.time()

        # 1. 识别性能瓶颈
        print(f"\n[步骤 1/4] 识别性能瓶颈...")
        bottlenecks = self._identify_bottlenecks(min_samples)
        print(f"  发现 {len(bottlenecks)} 个性能瓶颈")

        # 2. 分析性能趋势
        print(f"\n[步骤 2/4] 分析性能趋势...")
        trends = self._analyze_trends(min_samples)
        print(f"  分析了 {len(trends)} 个组件的趋势")

        # 3. 生成优化建议
        print(f"\n[步骤 3/4] 生成优化建议...")
        suggestions = self._generate_suggestions(bottlenecks, trends)
        print(f"  生成了 {len(suggestions)} 条优化建议")

        # 4. 评估系统健康度
        print(f"\n[步骤 4/4] 评估系统健康度...")
        health_score = self._calculate_health_score(bottlenecks, trends)
        print(f"  健康度评分: {health_score:.2%}")

        duration = time.time() - start_time

        # 识别关键组件
        critical_components = list(set(b.component for b in bottlenecks if b.severity in [BottleneckSeverity.HIGH, BottleneckSeverity.CRITICAL]))

        # 构建分析结果
        analysis = PerformanceAnalysis(
            bottlenecks=bottlenecks,
            trends=trends,
            suggestions=suggestions,
            overall_health_score=health_score,
            critical_components=critical_components,
            analysis_timestamp=time.time()
        )

        # 打印分析报告
        self._print_analysis_report(analysis, duration)

        return analysis

    def _identify_bottlenecks(self, min_samples: int) -> List[Bottleneck]:
        """识别性能瓶颈"""
        bottlenecks = []

        for component, metrics in self.samples.items():
            for metric_type, samples in metrics.items():
                if len(samples) < min_samples:
                    continue

                if metric_type not in self.thresholds:
                    continue

                # 计算统计值
                values = [s.value for s in samples]
                avg_value = statistics.mean(values)
                max_value = max(values)
                recent_value = values[-1]

                # 确定严重程度
                thresholds = self.thresholds[metric_type]
                if recent_value >= thresholds["critical"]:
                    severity = BottleneckSeverity.CRITICAL
                    impact = "严重阻塞系统运行"
                elif recent_value >= thresholds["high"]:
                    severity = BottleneckSeverity.HIGH
                    impact = "显著影响系统性能"
                elif recent_value >= thresholds["medium"]:
                    severity = BottleneckSeverity.MEDIUM
                    impact = "适度影响性能"
                elif recent_value >= thresholds["low"]:
                    severity = BottleneckSeverity.LOW
                    impact = "轻微影响"
                else:
                    continue  # 性能正常

                # 分析趋势
                trend = self._get_trend(values)

                bottlenecks.append(Bottleneck(
                    component=component,
                    metric_type=metric_type,
                    severity=severity,
                    current_value=recent_value,
                    threshold=thresholds[str(severity.value)],
                    impact=impact,
                    trend=trend,
                    suggested_actions=[]
                ))

        # 按严重程度排序
        severity_order = {
            BottleneckSeverity.CRITICAL: 0,
            BottleneckSeverity.HIGH: 1,
            BottleneckSeverity.MEDIUM: 2,
            BottleneckSeverity.LOW: 3
        }
        bottlenecks.sort(key=lambda b: severity_order[b.severity])

        return bottlenecks

    def _analyze_trends(self, min_samples: int) -> List[PerformanceTrend]:
        """分析性能趋势"""
        trends_list = []

        for component, metrics in self.samples.items():
            for metric_type, samples in metrics.items():
                if len(samples) < min_samples:
                    continue

                values = [s.value for s in samples]
                timestamps = [s.timestamp for s in samples]

                # 计算变化率（线性回归）
                if len(values) >= 3:
                    # 简单趋势分析：比较前后半段的平均值
                    mid = len(values) // 2
                    first_half_avg = statistics.mean(values[:mid])
                    second_half_avg = statistics.mean(values[mid:])

                    change_rate = (second_half_avg - first_half_avg) / (first_half_avg + 1e-6)

                    # 确定趋势方向
                    if change_rate > 0.1:
                        trend = "worsening"
                    elif change_rate < -0.1:
                        trend = "improving"
                    else:
                        trend = "stable"

                    # 计算置信度（基于样本数量）
                    confidence = min(len(values) / 100.0, 1.0)

                    trends_list.append(PerformanceTrend(
                        component=component,
                        metric_type=metric_type,
                        trend=trend,
                        change_rate=change_rate,
                        confidence=confidence
                    ))

        return trends_list

    def _get_trend(self, values: List[float]) -> str:
        """获取趋势方向"""
        if len(values) < 3:
            return "stable"

        # 比较最近3个值与之前的平均值
        recent = values[-3:]
        previous = values[:-3]

        if not previous:
            return "stable"

        recent_avg = statistics.mean(recent)
        previous_avg = statistics.mean(previous)

        change = (recent_avg - previous_avg) / (previous_avg + 1e-6)

        if change > 0.1:
            return "worsening"
        elif change < -0.1:
            return "improving"
        else:
            return "stable"

    def _generate_suggestions(
        self,
        bottlenecks: List[Bottleneck],
        trends: List[PerformanceTrend]
    ) -> List[OptimizationSuggestion]:
        """生成优化建议"""
        suggestions = []

        for bottleneck in bottlenecks:
            if bottleneck.severity in [BottleneckSeverity.LOW]:
                continue  # 低严重度不生成建议

            # 根据指标类型生成建议
            if bottleneck.metric_type == PerformanceMetric.EXECUTION_TIME:
                if bottleneck.trend == "worsening":
                    suggestions.append(OptimizationSuggestion(
                        component=bottleneck.component,
                        priority="high" if bottleneck.severity == BottleneckSeverity.CRITICAL else "medium",
                        category="caching",
                        description=f"组件执行时间过长且持续恶化",
                        expected_improvement="可减少50-80%执行时间",
                        implementation_effort="medium"
                    ))

                suggestions.append(OptimizationSuggestion(
                    component=bottleneck.component,
                    priority="medium",
                    category="algorithm",
                    description=f"优化算法复杂度或使用更高效的数据结构",
                    expected_improvement="可减少30-60%执行时间",
                    implementation_effort="hard"
                ))

            elif bottleneck.metric_type == PerformanceMetric.CPU_USAGE:
                suggestions.append(OptimizationSuggestion(
                    component=bottleneck.component,
                    priority="high",
                    category="parallelization",
                    description=f"CPU使用率过高，考虑并行化处理",
                    expected_improvement="可提升2-4倍吞吐量",
                    implementation_effort="medium"
                ))

            elif bottleneck.metric_type == PerformanceMetric.MEMORY_USAGE:
                suggestions.append(OptimizationSuggestion(
                    component=bottleneck.component,
                    priority="medium",
                    category="memory_optimization",
                    description=f"内存使用过高，检查内存泄漏或优化数据结构",
                    expected_improvement="可减少30-50%内存占用",
                    implementation_effort="medium"
                ))

        return suggestions

    def _calculate_health_score(
        self,
        bottlenecks: List[Bottleneck],
        trends: List[PerformanceTrend]
    ) -> float:
        """计算系统健康度评分"""
        score = 1.0

        # 根据瓶颈严重程度扣分
        for bottleneck in bottlenecks:
            if bottleneck.severity == BottleneckSeverity.CRITICAL:
                score -= 0.3
            elif bottleneck.severity == BottleneckSeverity.HIGH:
                score -= 0.15
            elif bottleneck.severity == BottleneckSeverity.MEDIUM:
                score -= 0.05
            elif bottleneck.severity == BottleneckSeverity.LOW:
                score -= 0.01

        # 根据趋势扣分
        for trend in trends:
            if trend.trend == "worsening" and trend.confidence > 0.7:
                score -= 0.05

        return max(0.0, min(1.0, score))

    def _print_analysis_report(self, analysis: PerformanceAnalysis, duration: float):
        """打印分析报告"""
        print(f"\n{'─'*60}")
        print(f"[性能分析报告]")
        print(f"{'─'*60}")

        print(f"\n📊 系统健康度: {analysis.overall_health_score:.2%}")
        print(f"分析耗时: {duration:.2f}秒")

        if analysis.bottlenecks:
            print(f"\n⚠️  性能瓶颈 ({len(analysis.bottlenecks)}个):")
            for i, bottleneck in enumerate(analysis.bottlenecks[:10], 1):
                severity_icon = {
                    BottleneckSeverity.CRITICAL: "🔴",
                    BottleneckSeverity.HIGH: "🟠",
                    BottleneckSeverity.MEDIUM: "🟡",
                    BottleneckSeverity.LOW: "🟢"
                }
                icon = severity_icon.get(bottleneck.severity, "⚪")

                print(f"  {icon} {i}. {bottleneck.component}")
                print(f"     指标: {bottleneck.metric_type.value}")
                print(f"     当前值: {bottleneck.current_value:.2f} {bottleneck.severity.value}")
                print(f"     影响: {bottleneck.impact}")
                print(f"     趋势: {bottleneck.trend}")

            if len(analysis.bottlenecks) > 10:
                print(f"  ... 还有 {len(analysis.bottlenecks) - 10} 个瓶颈")
        else:
            print(f"\n✅ 未发现性能瓶颈")

        if analysis.trends:
            worsening = [t for t in analysis.trends if t.trend == "worsening"]
            if worsening:
                print(f"\n📈 性能恶化趋势 ({len(worsening)}个):")
                for trend in worsening[:5]:
                    print(f"  • {trend.component} ({trend.metric_type.value}): {trend.change_rate:+.2%}/秒")

        if analysis.suggestions:
            print(f"\n💡 优化建议 ({len(analysis.suggestions)}条):")
            for i, suggestion in enumerate(analysis.suggestions[:5], 1):
                print(f"  {i}. [{suggestion.priority.upper()}] {suggestion.component}")
                print(f"     类别: {suggestion.category}")
                print(f"     描述: {suggestion.description}")
                print(f"     预期收益: {suggestion.expected_improvement}")
                print(f"     实施难度: {suggestion.implementation_effort}")

        if analysis.critical_components:
            print(f"\n🔥 关键组件 (需优先处理):")
            for component in analysis.critical_components:
                print(f"  • {component}")

        print(f"\n{'='*60}")

        # 关键输出
        if analysis.overall_health_score > 0.8:
            print(f"[ArchitectureAwareness] ✅ 性能状态: 优秀")
        elif analysis.overall_health_score > 0.6:
            print(f"[ArchitectureAwareness] ⚠️  性能状态: 良好")
        elif analysis.overall_health_score > 0.4:
            print(f"[ArchitectureAwareness] ⚠️  性能状态: 一般，建议优化")
        else:
            print(f"[ArchitectureAwareness] 🔴 性能状态: 差，需要紧急优化")

    def get_monitor(self) -> PerformanceMonitor:
        """获取性能监控装饰器"""
        return PerformanceMonitor(self)


# ============ 使用示例 ============

if __name__ == "__main__":
    print("="*60)
    print("性能瓶颈分析器测试")
    print("="*60)

    analyzer = PerformanceBottleneckAnalyzer()

    # 创建监控装饰器
    monitor = analyzer.get_monitor()

    # 模拟一些性能数据
    @monitor(component_name="test_component")
    def slow_function():
        time.sleep(0.1)  # 100ms
        return "done"

    @monitor(component_name="fast_component")
    def fast_function():
        time.sleep(0.01)  # 10ms
        return "done"

    # 执行多次以收集数据
    print("\n收集性能样本...")
    for _ in range(20):
        slow_function()
        fast_function()

    # 执行分析
    analysis = analyzer.analyze(min_samples=5)

    print("\n✅ 性能分析完成！")
