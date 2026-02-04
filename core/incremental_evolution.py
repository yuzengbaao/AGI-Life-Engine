#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incremental Evolution - 增量进化策略系统
==========================================

功能：
1. 性能瓶颈分析
2. 进化路径规划
3. 增量进化控制
4. 进化效果评估

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import cProfile
import io
import logging
import pstats
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBottleneck:
    """性能瓶颈"""
    component_name: str
    function_name: str
    bottleneck_type: str  # 'cpu', 'memory', 'io'
    severity: float  # 0-1
    cumulative_time: float
    call_count: int
    per_call_time: float


@dataclass
class EvolutionPlan:
    """进化计划"""
    component_id: str
    target_version: str
    optimization_goal: str
    bottlenecks: List[PerformanceBottleneck]
    estimated_improvement: float
    risk_level: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvolutionResult:
    """进化结果"""
    component_id: str
    old_version: str
    new_version: str
    success: bool
    performance_improvement: float  # 百分比
    bottlenecks_resolved: int
    side_effects: List[str]
    timestamp: str


class IncrementalEvolution:
    """
    增量进化控制器

    实现组件的渐进式进化优化
    """

    def __init__(self, self_modifying_engine=None):
        """
        初始化增量进化控制器

        Args:
            self_modifying_engine: 自修改引擎实例（可选）
        """
        self.engine = self_modifying_engine
        self.performance_history: Dict[str, List[Dict]] = {}
        self.evolution_results: List[EvolutionResult] = []
        self.active_plans: List[EvolutionPlan] = []

        logger.info("[增量进化] 初始化完成")

    def analyze_performance_bottleneck(
        self,
        component_id: str,
        component_instance: Any
    ) -> Tuple[bool, Optional[PerformanceBottleneck], Optional[str]]:
        """
        分析性能瓶颈

        Args:
            component_id: 组件ID
            component_instance: 组件实例

        Returns:
            (成功, 瓶颈信息, 错误信息)
        """
        try:
            logger.info(f"[增量进化] 分析性能瓶颈: {component_id}")

            # 性能分析
            pr = cProfile.Profile()
            pr.enable()

            # 执行组件（如果有run方法）
            if hasattr(component_instance, 'run'):
                component_instance.run()
            elif hasattr(component_instance, '__call__'):
                component_instance()
            else:
                return False, None, "组件无可执行的接口"

            pr.disable()

            # 解析结果
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
            ps.print_stats(20)

            output = s.getvalue()

            # 识别瓶颈
            bottleneck = self._extract_bottleneck(output, component_id)

            if bottleneck:
                logger.info(
                    f"[增量进化] ✅ 发现瓶颈: "
                    f"{bottleneck.function_name} "
                    f"(严重度: {bottleneck.severity:.2f})"
                )
                return True, bottleneck, None
            else:
                logger.info(f"[增量进化] 未发现明显瓶颈")
                return True, None, None

        except Exception as e:
            error_msg = f"瓶颈分析失败: {e}"
            logger.error(f"[增量进化] {error_msg}")
            return False, None, error_msg

    def create_evolution_plan(
        self,
        component_id: str,
        bottleneck: PerformanceBottleneck,
        optimization_goal: str = "performance"
    ) -> EvolutionPlan:
        """
        创建进化计划

        Args:
            component_id: 组件ID
            bottleneck: 性能瓶颈
            optimization_goal: 优化目标

        Returns:
            进化计划
        """
        plan = EvolutionPlan(
            component_id=component_id,
            target_version=self._get_next_version(component_id),
            optimization_goal=optimization_goal,
            bottlenecks=[bottleneck],
            estimated_improvement=self._estimate_improvement(bottleneck),
            risk_level=self._assess_risk(bottleneck)
        )

        self.active_plans.append(plan)

        logger.info(
            f"[增量进化] ✅ 进化计划创建: "
            f"{component_id} -> v{plan.target_version}, "
            f"预期提升: {plan.estimated_improvement:.1%}"
        )

        return plan

    def execute_evolution(
        self,
        plan: EvolutionPlan,
        new_component_code: str
    ) -> EvolutionResult:
        """
        执行进化

        Args:
            plan: 进化计划
            new_component_code: 新组件代码

        Returns:
            进化结果
        """
        try:
            logger.info(
                f"[增量进化] 执行进化: "
                f"{plan.component_id} -> v{plan.target_version}"
            )

            # 记录旧版本性能
            old_metrics = self._measure_component(plan.component_id)

            # 应用新代码（通过自修改引擎）
            if self.engine:
                from core.self_modifying_engine import CodePatch, CodeLocation

                patch = CodePatch(
                    location=CodeLocation(file_path=f"{plan.component_id}.py"),
                    original_code="",
                    modified_code=new_component_code,
                    risk_level=plan.risk_level
                )

                record = self.engine.apply_or_reject(patch)

                if record.status.name != "applied":
                    return EvolutionResult(
                        component_id=plan.component_id,
                        old_version="unknown",
                        new_version=plan.target_version,
                        success=False,
                        performance_improvement=0.0,
                        bottlenecks_resolved=0,
                        side_effects=["应用失败"],
                        timestamp=datetime.now().isoformat()
                    )

            # 测量新版本性能
            new_metrics = self._measure_component(plan.component_id)

            # 计算改进
            improvement = self._calculate_improvement(old_metrics, new_metrics)

            result = EvolutionResult(
                component_id=plan.component_id,
                old_version=old_metrics.get('version', 'unknown'),
                new_version=plan.target_version,
                success=True,
                performance_improvement=improvement,
                bottlenecks_resolved=len(plan.bottlenecks),
                side_effects=[],
                timestamp=datetime.now().isoformat()
            )

            self.evolution_results.append(result)

            # 从活动计划中移除
            if plan in self.active_plans:
                self.active_plans.remove(plan)

            logger.info(
                f"[增量进化] ✅ 进化成功: "
                f"{plan.component_id}, "
                f"性能提升: {improvement:.1%}"
            )

            return result

        except Exception as e:
            logger.error(f"[增量进化] 进化失败: {e}")

            return EvolutionResult(
                component_id=plan.component_id,
                old_version="unknown",
                new_version=plan.target_version,
                success=False,
                performance_improvement=0.0,
                bottlenecks_resolved=0,
                side_effects=[str(e)],
                timestamp=datetime.now().isoformat()
            )

    def monitor_performance(self, component_id: str) -> Dict[str, Any]:
        """
        监控组件性能

        Args:
            component_id: 组件ID

        Returns:
            性能指标
        """
        if component_id not in self.performance_history:
            self.performance_history[component_id] = []

        # 测量当前性能
        metrics = self._measure_component(component_id)
        metrics['timestamp'] = datetime.now().isoformat()

        # 记录历史
        self.performance_history[component_id].append(metrics)

        # 计算趋势
        if len(self.performance_history[component_id]) > 1:
            recent = self.performance_history[component_id][-5:]
            trend = self._calculate_trend(recent)
            metrics['trend'] = trend

        return metrics

    def _extract_bottleneck(
        self,
        profile_output: str,
        component_id: str
    ) -> Optional[PerformanceBottleneck]:
        """从性能分析输出中提取瓶颈"""
        try:
            lines = profile_output.split('\n')

            # 跳过头部
            data_lines = []
            for line in lines:
                if line.strip() and not line.startswith('   ') and ':' in line:
                    data_lines.append(line)

            if not data_lines:
                return None

            # 解析第一行（最耗时的函数）
            first_line = data_lines[0]
            parts = first_line.split()

            if len(parts) < 6:
                return None

            # 提取信息
            cumulative_time = float(parts[1])  # 累计时间
            call_count = int(parts[0])  # 调用次数
            per_call_time = cumulative_time / call_count if call_count > 0 else 0

            # 提取函数名
            function_name = parts[-1]
            if '{built-in method' in function_name:
                function_name = function_name.split("'")[1]

            # 评估严重度（基于耗时）
            severity = min(cumulative_time / 1.0, 1.0)  # 假设1秒为严重瓶颈

            return PerformanceBottleneck(
                component_name=component_id,
                function_name=function_name,
                bottleneck_type='cpu',
                severity=severity,
                cumulative_time=cumulative_time,
                call_count=call_count,
                per_call_time=per_call_time
            )

        except Exception as e:
            logger.warning(f"瓶颈提取失败: {e}")
            return None

    def _estimate_improvement(
        self,
        bottleneck: PerformanceBottleneck
    ) -> float:
        """估算改进幅度"""
        # 基于瓶颈严重度估算
        return bottleneck.severity * 0.3  # 最多30%改进

    def _assess_risk(self, bottleneck: PerformanceBottleneck) -> str:
        """评估风险等级"""
        if bottleneck.severity > 0.8:
            return "high"
        elif bottleneck.severity > 0.5:
            return "medium"
        else:
            return "low"

    def _get_next_version(self, component_id: str) -> str:
        """获取下一个版本号"""
        # 查找最近的进化结果
        for result in reversed(self.evolution_results):
            if result.component_id == component_id:
                parts = result.new_version.split('.')
                if len(parts) >= 3:
                    parts[2] = str(int(parts[2]) + 1)
                    return '.'.join(parts)
        return "1.0.1"

    def _measure_component(self, component_id: str) -> Dict[str, float]:
        """测量组件性能"""
        # 简化实现：返回模拟数据
        # 实际应该运行组件并测量
        return {
            'execution_time': 0.1,
            'memory_usage': 1024,
            'version': '1.0.0'
        }

    def _calculate_improvement(
        self,
        old_metrics: Dict,
        new_metrics: Dict
    ) -> float:
        """计算性能改进百分比"""
        old_time = old_metrics.get('execution_time', 1.0)
        new_time = new_metrics.get('execution_time', 1.0)

        if old_time == 0:
            return 0.0

        improvement = (old_time - new_time) / old_time
        return improvement

    def _calculate_trend(self, history: List[Dict]) -> str:
        """计算性能趋势"""
        if len(history) < 2:
            return "stable"

        recent_times = [h.get('execution_time', 0) for h in history]

        if len(recent_times) < 2:
            return "stable"

        # 简单线性回归
        avg_old = sum(recent_times[:len(recent_times)//2]) / (len(recent_times)//2)
        avg_new = sum(recent_times[len(recent_times)//2:]) / (len(recent_times) - len(recent_times)//2)

        if avg_new < avg_old * 0.9:
            return "improving"
        elif avg_new > avg_old * 1.1:
            return "degrading"
        else:
            return "stable"

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """获取进化统计信息"""
        successful = sum(1 for r in self.evolution_results if r.success)
        total = len(self.evolution_results)

        avg_improvement = 0.0
        if successful > 0:
            avg_improvement = sum(
                r.performance_improvement for r in self.evolution_results
                if r.success
            ) / successful

        return {
            'total_evolutions': total,
            'successful': successful,
            'failed': total - successful,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_improvement': avg_improvement,
            'active_plans': len(self.active_plans),
            'components_tracked': len(self.performance_history)
        }


# 全局单例
_global_evolution: Optional[IncrementalEvolution] = None


def get_incremental_evolution() -> IncrementalEvolution:
    """获取全局增量进化控制器"""
    global _global_evolution
    if _global_evolution is None:
        _global_evolution = IncrementalEvolution()
    return _global_evolution
