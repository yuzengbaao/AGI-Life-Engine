#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Recursion Limiter - 动态递归限制器
==========================================

功能：
1. 基于系统状态动态调整递归深度
2. 从硬编码3提升到动态10
3. 尾递归优化支持
4. 性能历史记录

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import functools
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class RecursionCall:
    """尾递归调用标记"""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def tail_recursive_optimized(fn):
    """
    将尾递归转换为迭代，避免堆栈溢出

    Args:
        fn: 要优化的函数

    Returns:
        优化后的包装函数
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        while True:
            result = fn(*args, **kwargs)
            if isinstance(result, RecursionCall):
                args, kwargs = result.args, result.kwargs
                continue
            return result
    return wrapper


@dataclass
class RecursionPerformanceRecord:
    """递归性能记录"""
    timestamp: float
    depth: int
    success: bool
    execution_time_ms: float
    system_load: float
    task_complexity: float


class DynamicRecursionLimiter:
    """
    动态递归深度限制器

    基于系统状态动态调整递归深度限制
    """

    def __init__(self, base_depth: int = 3, max_depth: int = 10):
        """
        初始化动态递归限制器

        Args:
            base_depth: 基础深度（默认3，向后兼容）
            max_depth: 最大深度（提升到10）
        """
        self.base_depth = base_depth
        self.max_depth = max_depth
        self.performance_history: List[RecursionPerformanceRecord] = []
        self.current_limit = base_depth

        logger.info(
            f"[递归限制器] 初始化: "
            f"基础深度={base_depth}, 最大深度={max_depth}"
        )

    def get_current_limit(self, context: Optional[Dict[str, Any]] = None) -> int:
        """
        基于系统状态动态调整递归深度

        Args:
            context: 上下文信息（可选）

        Returns:
            当前递归深度限制
        """
        if context is None:
            context = {}

        # 基础深度
        current_limit = self.base_depth
        adjustments = []

        # 因素1: 系统负载
        try:
            import psutil
            load = psutil.cpu_percent(interval=0.1)

            if load < 30:
                depth_adjustment = +2
                adjustments.append(f"低负载(+2)")
            elif load > 80:
                depth_adjustment = -1
                adjustments.append(f"高负载(-1)")
            else:
                depth_adjustment = 0
                adjustments.append(f"正常负载(0)")
        except Exception:
            # 如果psutil不可用，使用默认
            depth_adjustment = 0
            adjustments.append("负载未知(0)")

        current_limit += depth_adjustment

        # 因素2: 任务复杂度
        complexity = context.get('task_complexity', 0.5)
        if complexity > 0.8:
            current_limit += 1
            adjustments.append(f"高复杂度(+1)")
        elif complexity < 0.3:
            adjustments.append("低复杂度(0)")
        else:
            adjustments.append("中复杂度(0)")

        # 因素3: 历史性能
        if self.performance_history:
            recent = self.performance_history[-20:]
            if recent:
                recent_avg = np.mean([1.0 if r.success else 0.0 for r in recent])

                if recent_avg > 0.9:
                    current_limit += 1
                    adjustments.append(f"高成功率(+1)")
                elif recent_avg < 0.5:
                    current_limit -= 1
                    adjustments.append(f"低成功率(-1)")
                else:
                    adjustments.append("中成功率(0)")
        else:
            adjustments.append("无历史(0)")

        # 因素4: 内存使用
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            if memory_percent > 90:
                current_limit -= 2
                adjustments.append(f"高内存(-2)")
            elif memory_percent < 50:
                current_limit += 1
                adjustments.append(f"低内存(+1)")
            else:
                adjustments.append("正常内存(0)")
        except Exception:
            adjustments.append("内存未知(0)")

        # 应用最大深度限制
        current_limit = max(1, min(current_limit, self.max_depth))

        # 记录调整
        if len(adjustments) > 0:
            logger.debug(
                f"[递归限制器] 深度调整: "
                f"{self.current_limit} -> {current_limit}, "
                f"调整因素: {', '.join(adjustments)}"
            )

        self.current_limit = current_limit
        return current_limit

    def record_performance(
        self,
        depth: int,
        success: bool,
        execution_time_ms: float,
        system_load: float = 0.0,
        task_complexity: float = 0.5
    ):
        """
        记录递归性能

        Args:
            depth: 使用的递归深度
            success: 是否成功
            execution_time_ms: 执行时间（毫秒）
            system_load: 系统负载
            task_complexity: 任务复杂度
        """
        import time

        record = RecursionPerformanceRecord(
            timestamp=time.time(),
            depth=depth,
            success=success,
            execution_time_ms=execution_time_ms,
            system_load=system_load,
            task_complexity=task_complexity
        )

        self.performance_history.append(record)

        # 保留最近100条记录
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

        logger.debug(
            f"[递归限制器] 记录性能: "
            f"深度={depth}, 成功={success}, "
            f"耗时={execution_time_ms:.2f}ms"
        )

    def get_average_depth(self) -> float:
        """获取平均递归深度"""
        if not self.performance_history:
            return self.base_depth

        return np.mean([r.depth for r in self.performance_history])

    def get_success_rate(self) -> float:
        """获取递归成功率"""
        if not self.performance_history:
            return 1.0

        successful = sum(1 for r in self.performance_history if r.success)
        return successful / len(self.performance_history)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.performance_history:
            return {
                'base_depth': self.base_depth,
                'max_depth': self.max_depth,
                'current_limit': self.current_limit,
                'total_recursions': 0,
                'avg_depth': self.base_depth,
                'success_rate': 1.0
            }

        return {
            'base_depth': self.base_depth,
            'max_depth': self.max_depth,
            'current_limit': self.current_limit,
            'total_recursions': len(self.performance_history),
            'avg_depth': self.get_average_depth(),
            'success_rate': self.get_success_rate(),
            'recent_performance': [
                {
                    'depth': r.depth,
                    'success': r.success,
                    'time_ms': r.execution_time_ms
                }
                for r in self.performance_history[-10:]
            ]
        }


# 全局单例
_global_limiter: Optional[DynamicRecursionLimiter] = None


def get_recursion_limiter() -> DynamicRecursionLimiter:
    """获取全局递归限制器"""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = DynamicRecursionLimiter()
    return _global_limiter


def enable_tail_recursion(cls):
    """
    为类的所有方法启用尾递归优化

    Args:
        cls: 要优化的类
    """
    for name, method in cls.__dict__.items():
        if callable(method) and not name.startswith('_'):
            setattr(cls, name, tail_recursive_optimized(method))
    return cls
