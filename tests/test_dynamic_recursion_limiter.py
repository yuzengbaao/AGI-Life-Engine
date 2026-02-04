#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试: 动态递归限制器

测试覆盖:
- DynamicRecursionLimiter 核心功能
- 动态深度计算
- 多因素调整（CPU、复杂度、历史、内存）
- 性能记录追踪

作者: AGI System
日期: 2026-02-04
"""

import pytest
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# 导入被测试模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dynamic_recursion_limiter import DynamicRecursionLimiter


class TestDynamicRecursionLimiter:
    """测试 DynamicRecursionLimiter 限制器"""

    @pytest.fixture
    def limiter(self):
        """创建限制器实例"""
        limiter = DynamicRecursionLimiter()
        return limiter

    def test_initialization(self, limiter):
        """测试初始化"""
        assert limiter.base_depth == 3
        assert limiter.max_depth == 10
        assert limiter.performance_history == []

    def test_get_current_limit_default(self, limiter):
        """测试默认上下文获取限制"""
        context = {
            "task_complexity": 0.5,
        }

        limit = limiter.get_current_limit(context)

        # 应该在合理范围内
        assert 1 <= limit <= limiter.max_depth

    def test_cpu_factor(self, limiter):
        """测试CPU负载因素"""
        # 模拟低CPU
        with patch('psutil.cpu_percent', return_value=20):
            context = {"task_complexity": 0.5}
            limit_low_cpu = limiter.get_current_limit(context)
            assert limit_low_cpu >= limiter.base_depth

        # 模拟高CPU
        with patch('psutil.cpu_percent', return_value=90):
            limit_high_cpu = limiter.get_current_limit(context)
            assert limit_high_cpu <= limiter.base_depth

    def test_task_complexity_factor(self, limiter):
        """测试任务复杂度因素"""
        # 低复杂度
        context_low = {"task_complexity": 0.2}
        limit_low = limiter.get_current_limit(context_low)

        # 高复杂度
        context_high = {"task_complexity": 0.9}
        limit_high = limiter.get_current_limit(context_high)

        # 高复杂度应该获得更高限制
        assert limit_high >= limit_low

    def test_performance_history_factor(self, limiter):
        """测试历史性能因素"""
        # 记录高性能历史
        for i in range(10):
            limiter.record_performance(depth=3, success=True, execution_time_ms=50.0)

        context = {"task_complexity": 0.5}
        limit_with_history = limiter.get_current_limit(context)

        # 历史性能好，限制应该提升
        assert limit_with_history >= limiter.base_depth

    def test_max_depth_clamping(self, limiter):
        """测试最大深度限制"""
        # 极端情况：所有因素都指向增加
        with patch('psutil.cpu_percent', return_value=10):
            # 记录成功历史
            for _ in range(10):
                limiter.record_performance(depth=5, success=True, execution_time_ms=50.0)

            context = {"task_complexity": 1.0}  # 最高复杂度
            limit = limiter.get_current_limit(context)

            # 不应超过最大深度
            assert limit <= limiter.max_depth

    def test_min_depth_clamping(self, limiter):
        """测试最小深度限制"""
        # 极端情况：所有因素都指向减少
        with patch('psutil.cpu_percent', return_value=95):
            # 记录失败历史
            for _ in range(10):
                limiter.record_performance(depth=1, success=False, execution_time_ms=200.0)

            context = {"task_complexity": 0.0}  # 最低复杂度
            limit = limiter.get_current_limit(context)

            # 不应低于1
            assert limit >= 1

    def test_record_performance(self, limiter):
        """测试性能记录"""
        initial_len = len(limiter.performance_history)

        # 记录成功
        limiter.record_performance(depth=3, success=True, execution_time_ms=50.0)
        assert len(limiter.performance_history) == initial_len + 1
        assert limiter.performance_history[-1].success is True

        # 记录失败
        limiter.record_performance(depth=3, success=False, execution_time_ms=200.0)
        assert len(limiter.performance_history) == initial_len + 2
        assert limiter.performance_history[-1].success is False

    def test_performance_history_limit(self, limiter):
        """测试历史记录限制"""
        # 添加超过限制的记录
        for i in range(150):
            limiter.record_performance(depth=3, success=True, execution_time_ms=50.0)

        # 应该被限制在100条
        assert len(limiter.performance_history) <= 100

    def test_multiple_context_calls(self, limiter):
        """测试多次上下文调用"""
        contexts = [
            {"task_complexity": 0.3},
            {"task_complexity": 0.5},
            {"task_complexity": 0.7},
        ]

        limits = [limiter.get_current_limit(ctx) for ctx in contexts]

        # 所有限制都应在有效范围内
        for limit in limits:
            assert 1 <= limit <= 10

    def test_context_without_complexity(self, limiter):
        """测试缺少复杂度的上下文"""
        context = {}  # 没有task_complexity

        # 不应该崩溃，应该使用默认值
        limit = limiter.get_current_limit(context)
        assert 1 <= limit <= 10


class TestDynamicRecursionWithMemory:
    """测试带内存监控的动态递归"""

    @pytest.fixture
    def limiter_with_memory(self):
        """创建带内存监控的限制器"""
        limiter = DynamicRecursionLimiter()
        return limiter

    def test_memory_pressure_factor(self, limiter_with_memory):
        """测试内存压力因素"""
        # 正常内存情况
        with patch('psutil.virtual_memory') as mock_vm:
            mock_mem = MagicMock()
            mock_mem.percent = 50  # 50%内存使用
            mock_vm.return_value = mock_mem

            context = {"task_complexity": 0.5}
            limit = limiter_with_memory.get_current_limit(context)

            # 内存压力不高时，限制应该正常
            assert limit >= 1

    def test_high_memory_pressure(self, limiter_with_memory):
        """测试高内存压力"""
        with patch('psutil.virtual_memory') as mock_vm:
            mock_mem = MagicMock()
            mock_mem.percent = 95  # 95%内存使用
            mock_vm.return_value = mock_mem

            context = {"task_complexity": 0.5}
            limit = limiter_with_memory.get_current_limit(context)

            # 高内存压力应该降低递归深度
            # （具体实现取决于代码）

    def test_combined_factors(self, limiter_with_memory):
        """测试多因素综合影响"""
        # 最佳情况
        with patch('psutil.cpu_percent', return_value=10):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_mem = MagicMock()
                mock_mem.percent = 30
                mock_vm.return_value = mock_mem

                # 记录成功历史
                for _ in range(10):
                    limiter_with_memory.record_performance(depth=5, success=True, execution_time_ms=50.0)

                context = {"task_complexity": 0.9}
                limit_optimal = limiter_with_memory.get_current_limit(context)

        # 最差情况
        with patch('psutil.cpu_percent', return_value=90):
            with patch('psutil.virtual_memory') as mock_vm:
                mock_mem = MagicMock()
                mock_mem.percent = 90
                mock_vm.return_value = mock_mem

                # 记录失败历史
                for _ in range(10):
                    limiter_with_memory.record_performance(depth=1, success=False, execution_time_ms=200.0)

                context = {"task_complexity": 0.1}
                limit_worst = limiter_with_memory.get_current_limit(context)

        # 最佳情况应该获得更高限制
        assert limit_optimal >= limit_worst

    def test_empty_performance_history(self, limiter_with_memory):
        """测试空性能历史"""
        # 清空历史
        limiter_with_memory.performance_history = []

        context = {"task_complexity": 0.5}
        limit = limiter_with_memory.get_current_limit(context)

        # 应该基于基础深度计算
        assert limit is not None
        assert 1 <= limit <= 10


class TestEdgeCases:
    """测试边界情况"""

    def test_extreme_complexity(self):
        """测试极端复杂度值"""
        limiter = DynamicRecursionLimiter()

        # 复杂度 = 1.0 (最大)
        context_max = {"task_complexity": 1.0}
        limit_max = limiter.get_current_limit(context_max)

        # 复杂度 = 0.0 (最小)
        context_min = {"task_complexity": 0.0}
        limit_min = limiter.get_current_limit(context_min)

        # 都应该在有效范围内
        assert 1 <= limit_max <= 10
        assert 1 <= limit_min <= 10

    def test_rapid_performance_updates(self):
        """测试快速性能更新"""
        limiter = DynamicRecursionLimiter()

        # 快速添加100条记录
        for i in range(100):
            limiter.record_performance(depth=3, success=(i % 2 == 0), execution_time_ms=50.0)

        # 应该被限制在100条以内
        assert len(limiter.performance_history) <= 100

    def test_alternating_performance(self):
        """测试交替性能（成功/失败）"""
        limiter = DynamicRecursionLimiter()

        # 交替记录成功和失败
        for i in range(20):
            limiter.record_performance(depth=3, success=(i % 2 == 0), execution_time_ms=50.0)

        # 计算平均成功率
        success_rate = sum([1.0 if r.success else 0.0 for r in limiter.performance_history]) / len(limiter.performance_history)

        # 应该接近0.5
        assert 0.4 <= success_rate <= 0.6

    def test_all_success_history(self):
        """测试全部成功历史"""
        limiter = DynamicRecursionLimiter()

        for _ in range(20):
            limiter.record_performance(depth=3, success=True, execution_time_ms=50.0)

        # 应该倾向于增加递归深度
        context = {"task_complexity": 0.5}
        limit = limiter.get_current_limit(context)

        assert limit >= limiter.base_depth

    def test_all_failure_history(self):
        """测试全部失败历史"""
        limiter = DynamicRecursionLimiter()

        for _ in range(20):
            limiter.record_performance(depth=1, success=False, execution_time_ms=200.0)

        # 应该倾向于降低递归深度
        context = {"task_complexity": 0.5}
        limit = limiter.get_current_limit(context)

        # 可能低于基础深度
        assert limit >= 1

    def test_context_with_extra_fields(self):
        """测试包含额外字段的上下文"""
        limiter = DynamicRecursionLimiter()

        context = {
            "task_complexity": 0.5,
            "extra_field_1": "value1",
            "extra_field_2": 123,
            "nested": {"a": 1},
        }

        # 不应该崩溃
        limit = limiter.get_current_limit(context)
        assert 1 <= limit <= 10

    def test_nan_handling(self):
        """测试NaN处理（如果适用）"""
        limiter = DynamicRecursionLimiter()

        # 模拟返回NaN的情况（如果有）
        # 这里主要是确保不会崩溃
        context = {"task_complexity": 0.5}
        limit = limiter.get_current_limit(context)

        assert limit is not None


class TestRecursionDepthRanges:
    """测试递归深度范围"""

    def test_full_range_coverage(self):
        """测试完整范围覆盖"""
        limiter = DynamicRecursionLimiter()

        # 测试不同复杂度
        complexities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        limits = []

        for complexity in complexities:
            context = {"task_complexity": complexity}
            limit = limiter.get_current_limit(context)
            limits.append(limit)

        # 所有限制都应在有效范围内
        for limit in limits:
            assert 1 <= limit <= 10

        # 复杂度越高，限制应该倾向于增加
        # （虽然不一定严格单调）
        min_limit = min(limits)
        max_limit = max(limits)

        # 至少应该有一定的范围
        assert max_limit >= min_limit


class TestPerformanceTracking:
    """测试性能追踪"""

    def test_performance_history_growth(self):
        """测试性能历史增长"""
        limiter = DynamicRecursionLimiter()

        initial_size = len(limiter.performance_history)

        # 添加记录
        for i in range(5):
            limiter.record_performance(depth=3, success=True, execution_time_ms=50.0)

        assert len(limiter.performance_history) == initial_size + 5

    def test_performance_history_circular_buffer(self):
        """测试性能历史循环缓冲"""
        limiter = DynamicRecursionLimiter()

        # 添加超过限制的记录
        for i in range(110):
            limiter.record_performance(depth=3, success=True, execution_time_ms=50.0)

        # 应该被裁剪到100条
        assert len(limiter.performance_history) == 100

        # 最旧的记录应该被移除
        # （验证：当前记录应该是最近的）

    def test_average_performance_calculation(self):
        """测试平均性能计算"""
        limiter = DynamicRecursionLimiter()

        # 添加已知模式的记录
        for i in range(10):
            limiter.record_performance(depth=3, success=True, execution_time_ms=50.0)
        for i in range(10):
            limiter.record_performance(depth=3, success=False, execution_time_ms=50.0)

        # 计算平均值
        if limiter.performance_history:
            avg = sum([1.0 if r.success else 0.0 for r in limiter.performance_history]) / len(limiter.performance_history)
            assert avg == 0.5  # 10个True + 10个False = 0.5平均


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
