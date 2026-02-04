#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolution Monitor - 进化效果监控系统
=====================================

功能：
1. 性能指标追踪
2. 进化历史记录
3. 效果对比分析
4. 可视化报告生成

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """性能指标"""
    timestamp: str
    component_id: str
    version: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    error_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionSnapshot:
    """进化快照"""
    evolution_id: str
    component_id: str
    old_version: str
    new_version: str
    timestamp: str
    before_metrics: PerformanceMetric
    after_metrics: PerformanceMetric
    improvement_percent: float
    success: bool


class EvolutionMonitor:
    """
    进化效果监控器

    追踪和分析进化效果
    """

    def __init__(self, storage_dir: str = ".agi_evolution_monitor"):
        """
        初始化进化监控器

        Args:
            storage_dir: 存储目录
        """
        self.storage_dir = storage_dir
        self.metrics_history: List[PerformanceMetric] = []
        self.evolution_snapshots: List[EvolutionSnapshot] = []

        # 创建存储目录
        os.makedirs(storage_dir, exist_ok=True)

        # 加载历史数据
        self._load_history()

        logger.info(f"[进化监控] 初始化完成 (存储: {storage_dir})")

    def record_metric(self, metric: PerformanceMetric):
        """
        记录性能指标

        Args:
            metric: 性能指标
        """
        self.metrics_history.append(metric)

        logger.debug(
            f"[进化监控] 记录指标: "
            f"{metric.component_id} v{metric.version} "
            f"(执行时间: {metric.execution_time_ms:.2f}ms)"
        )

    def record_evolution(
        self,
        evolution_id: str,
        component_id: str,
        old_version: str,
        new_version: str,
        before_metrics: PerformanceMetric,
        after_metrics: PerformanceMetric,
        success: bool
    ) -> EvolutionSnapshot:
        """
        记录进化快照

        Args:
            evolution_id: 进化ID
            component_id: 组件ID
            old_version: 旧版本
            new_version: 新版本
            before_metrics: 进化前指标
            after_metrics: 进化后指标
            success: 是否成功

        Returns:
            进化快照
        """
        # 计算改进
        improvement = self._calculate_improvement(before_metrics, after_metrics)

        snapshot = EvolutionSnapshot(
            evolution_id=evolution_id,
            component_id=component_id,
            old_version=old_version,
            new_version=new_version,
            timestamp=datetime.now().isoformat(),
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percent=improvement,
            success=success
        )

        self.evolution_snapshots.append(snapshot)

        logger.info(
            f"[进化监控] 记录进化: "
            f"{component_id} v{old_version} -> v{new_version}, "
            f"改进: {improvement:.1%}"
        )

        return snapshot

    def compare_performance(
        self,
        component_id: str,
        version_a: str,
        version_b: str
    ) -> Dict[str, Any]:
        """
        对比两个版本的性能

        Args:
            component_id: 组件ID
            version_a: 版本A
            version_b: 版本B

        Returns:
            对比结果
        """
        # 获取版本A的指标
        metrics_a = [
            m for m in self.metrics_history
            if m.component_id == component_id and m.version == version_a
        ]

        # 获取版本B的指标
        metrics_b = [
            m for m in self.metrics_history
            if m.component_id == component_id and m.version == version_b
        ]

        if not metrics_a or not metrics_b:
            return {
                'error': '版本数据不足',
                'component_id': component_id,
                'version_a': version_a,
                'version_b': version_b
            }

        # 计算平均值
        avg_a = self._average_metrics(metrics_a)
        avg_b = self._average_metrics(metrics_b)

        # 计算差异
        comparison = {
            'component_id': component_id,
            'version_a': version_a,
            'version_b': version_b,
            'execution_time_diff_ms': avg_b['execution_time_ms'] - avg_a['execution_time_ms'],
            'execution_time_improvement': (
                (avg_a['execution_time_ms'] - avg_b['execution_time_ms']) /
                avg_a['execution_time_ms']
                if avg_a['execution_time_ms'] > 0 else 0
            ),
            'memory_diff_mb': avg_b['memory_usage_mb'] - avg_a['memory_usage_mb'],
            'cpu_diff_percent': avg_b['cpu_usage_percent'] - avg_a['cpu_usage_percent'],
            'error_diff': avg_b['error_count'] - avg_a['error_count']
        }

        return comparison

    def get_evolution_trend(
        self,
        component_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        获取进化趋势

        Args:
            component_id: 组件ID
            hours: 时间范围（小时）

        Returns:
            趋势分析
        """
        # 时间范围
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # 筛选时间范围内的指标
        recent_metrics = [
            m for m in self.metrics_history
            if m.component_id == component_id
            and datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        if not recent_metrics:
            return {
                'error': '无数据',
                'component_id': component_id,
                'hours': hours
            }

        # 按版本分组
        by_version: Dict[str, List[PerformanceMetric]] = defaultdict(list)
        for m in recent_metrics:
            by_version[m.version].append(m)

        # 计算每个版本的平均性能
        version_performance = {}
        for version, metrics in by_version.items():
            version_performance[version] = self._average_metrics(metrics)

        # 识别趋势
        versions = sorted(version_performance.keys())
        if len(versions) < 2:
            return {
                'component_id': component_id,
                'versions': list(version_performance.keys()),
                'trend': 'insufficient_data'
            }

        # 比较最早和最新版本
        earliest_version = versions[0]
        latest_version = versions[-1]

        earliest_perf = version_performance[earliest_version]
        latest_perf = version_performance[latest_version]

        improvement = (
            (earliest_perf['execution_time_ms'] - latest_perf['execution_time_ms']) /
            earliest_perf['execution_time_ms']
            if earliest_perf['execution_time_ms'] > 0 else 0
        )

        trend = 'improving' if improvement > 0.05 else 'stable' if improvement > -0.05 else 'degrading'

        return {
            'component_id': component_id,
            'hours': hours,
            'versions': versions,
            'version_performance': version_performance,
            'earliest_version': earliest_version,
            'latest_version': latest_version,
            'overall_improvement': improvement,
            'trend': trend
        }

    def generate_report(
        self,
        component_id: Optional[str] = None,
        hours: int = 24
    ) -> str:
        """
        生成监控报告

        Args:
            component_id: 组件ID（None表示所有组件）
            hours: 时间范围

        Returns:
            报告文本
        """
        lines = []
        lines.append("=" * 70)
        lines.append("进化效果监控报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"时间范围: 最近{hours}小时")
        lines.append("=" * 70)
        lines.append("")

        # 进化统计
        lines.append("## 进化统计")
        lines.append("")

        total_evolutions = len(self.evolution_snapshots)
        successful = sum(1 for s in self.evolution_snapshots if s.success)

        lines.append(f"总进化次数: {total_evolutions}")
        lines.append(f"成功次数: {successful}")
        lines.append(f"成功率: {successful/total_evolutions:.1%}" if total_evolutions > 0 else "成功率: N/A")
        lines.append("")

        # 按组件统计
        lines.append("## 组件性能统计")
        lines.append("")

        components = set(m.component_id for m in self.metrics_history)

        for comp_id in sorted(components):
            trend = self.get_evolution_trend(comp_id, hours)

            if 'error' not in trend:
                lines.append(f"### {comp_id}")
                lines.append(f"  趋势: {trend['trend']}")
                lines.append(f"  版本: {', '.join(trend['versions'])}")

                if 'overall_improvement' in trend:
                    improvement = trend['overall_improvement']
                    lines.append(f"  整体改进: {improvement:.1%}")

                lines.append("")

        # 最近进化
        lines.append("## 最近进化")
        lines.append("")

        recent_evolutions = sorted(
            self.evolution_snapshots,
            key=lambda s: s.timestamp,
            reverse=True
        )[:5]

        for snapshot in recent_evolutions:
            status = "✅" if snapshot.success else "❌"
            lines.append(
                f"{status} {snapshot.component_id}: "
                f"v{snapshot.old_version} -> v{snapshot.new_version} "
                f"({snapshot.improvement_percent:.1%})"
            )

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    def save_report(self, report: str, filename: Optional[str] = None):
        """
        保存报告到文件

        Args:
            report: 报告内容
            filename: 文件名（可选）
        """
        if filename is None:
            filename = f"evolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        filepath = os.path.join(self.storage_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"[进化监控] 报告已保存: {filepath}")

    def _calculate_improvement(
        self,
        before: PerformanceMetric,
        after: PerformanceMetric
    ) -> float:
        """计算改进百分比"""
        if before.execution_time_ms == 0:
            return 0.0

        return (before.execution_time_ms - after.execution_time_ms) / before.execution_time_ms

    def _average_metrics(self, metrics: List[PerformanceMetric]) -> Dict[str, float]:
        """计算平均指标"""
        if not metrics:
            return {}

        return {
            'execution_time_ms': sum(m.execution_time_ms for m in metrics) / len(metrics),
            'memory_usage_mb': sum(m.memory_usage_mb for m in metrics) / len(metrics),
            'cpu_usage_percent': sum(m.cpu_usage_percent for m in metrics) / len(metrics),
            'error_count': sum(m.error_count for m in metrics) / len(metrics)
        }

    def _load_history(self):
        """加载历史数据"""
        # 尝试加载指标历史
        metrics_file = os.path.join(self.storage_dir, "metrics_history.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.metrics_history = [
                        PerformanceMetric(**m) for m in data
                    ]
                logger.info(f"[进化监控] 加载指标历史: {len(self.metrics_history)}条")
            except Exception as e:
                logger.warning(f"[进化监控] 加载指标历史失败: {e}")

        # 尝试加载进化快照
        snapshots_file = os.path.join(self.storage_dir, "evolution_snapshots.json")
        if os.path.exists(snapshots_file):
            try:
                with open(snapshots_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.evolution_snapshots = [
                        EvolutionSnapshot(**s) for s in data
                    ]
                logger.info(f"[进化监控] 加载进化快照: {len(self.evolution_snapshots)}条")
            except Exception as e:
                logger.warning(f"[进化监控] 加载进化快照失败: {e}")

    def save_history(self):
        """保存历史数据"""
        # 保存指标历史
        metrics_file = os.path.join(self.storage_dir, "metrics_history.json")
        try:
            with open(metrics_file, 'w', encoding='utf-8') as f:
                data = [asdict(m) for m in self.metrics_history]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[进化监控] 保存指标历史失败: {e}")

        # 保存进化快照
        snapshots_file = os.path.join(self.storage_dir, "evolution_snapshots.json")
        try:
            with open(snapshots_file, 'w', encoding='utf-8') as f:
                data = [asdict(s) for s in self.evolution_snapshots]
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[进化监控] 保存进化快照失败: {e}")

    def get_monitor_statistics(self) -> Dict[str, Any]:
        """获取监控统计"""
        return {
            'total_metrics': len(self.metrics_history),
            'total_snapshots': len(self.evolution_snapshots),
            'components_tracked': len(set(m.component_id for m in self.metrics_history)),
            'storage_dir': self.storage_dir
        }


# 全局单例
_global_monitor: Optional[EvolutionMonitor] = None


def get_evolution_monitor() -> EvolutionMonitor:
    """获取全局进化监控器"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = EvolutionMonitor()
    return _global_monitor
