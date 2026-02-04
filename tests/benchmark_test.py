#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准测试框架

用于系统性能基准测试和对比：
- 缓存性能基准
- 递归限制器性能基准
- 生命周期管理器性能基准
- 综合性能基准

使用方法:
    # 运行所有基准测试
    python tests/benchmark_test.py

    # 运行特定基准
    python tests/benchmark_test.py --benchmark cache

    # 生成性能报告
    python tests/benchmark_test.py --report

作者: AGI System
日期: 2026-02-04
"""

import sys
import time
import statistics
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Callable
from dataclasses import dataclass, asdict
import numpy as np

# 设置Windows控制台编码
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except:
        pass

# 导入被测试模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tool_call_cache import ToolCallCache
from core.memory.memory_lifecycle_manager import (
    MemoryLifecycleManager,
    EvictionPolicy,
)
from core.dynamic_recursion_limiter import DynamicRecursionLimiter


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float
    std_dev: float
    ops_per_second: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BenchmarkRunner:
    """基准测试运行器"""

    def __init__(self):
        self.results = []

    def run_benchmark(
        self,
        name: str,
        func: Callable,
        iterations: int = 100,
        warmup: int = 10,
        **metadata
    ) -> BenchmarkResult:
        """
        运行基准测试

        Args:
            name: 测试名称
            func: 测试函数
            iterations: 迭代次数
            warmup: 预热次数
            **metadata: 额外元数据

        Returns:
            BenchmarkResult对象
        """
        print(f"🔧 运行基准: {name}")
        print(f"   预热: {warmup}次")
        print(f"   迭代: {iterations}次")

        # 预热
        for _ in range(warmup):
            func()

        # 基准测试
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒

        # 计算统计
        total_time = sum(times)
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        median_time = statistics.median(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        ops_per_second = iterations / (total_time / 1000)

        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            std_dev=std_dev,
            ops_per_second=ops_per_second,
            metadata=metadata
        )

        self.results.append(result)

        # 打印结果
        print(f"   总时间: {total_time:.2f}ms")
        print(f"   平均: {avg_time:.3f}ms")
        print(f"   中位数: {median_time:.3f}ms")
        print(f"   标准差: {std_dev:.3f}ms")
        print(f"   吞吐量: {ops_per_second:.1f} ops/s")
        print()

        return result

    def print_summary(self):
        """打印测试总结"""
        print("=" * 60)
        print("📊 基准测试总结")
        print("=" * 60)

        for result in self.results:
            print(f"\n{result.name}:")
            print(f"   平均时间: {result.avg_time:.3f}ms")
            print(f"   吞吐量: {result.ops_per_second:.1f} ops/s")
            print(f"   标准差: {result.std_dev:.3f}ms")

        # 性能排名
        print(f"\n🏆 性能排名（按平均时间）:")
        sorted_results = sorted(self.results, key=lambda r: r.avg_time)
        for i, result in enumerate(sorted_results, 1):
            print(f"   {i}. {result.name}: {result.avg_time:.3f}ms")

    def save_report(self, filepath: str):
        """保存报告到文件"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\n📄 报告已保存: {filepath}")


# ============================================================================
# 缓存性能基准测试
# ============================================================================

def benchmark_cache_put():
    """缓存PUT操作基准"""
    cache = ToolCallCache(max_size=1000)

    def put_operation():
        tool_name = "test_tool"
        params = {"operation": "test", "id": np.random.randint(0, 1000)}
        result = {"data": "test result"}
        cache.put(tool_name, params, result)

    return put_operation


def benchmark_cache_get():
    """缓存GET操作基准（命中）"""
    cache = ToolCallCache(max_size=1000)

    # 预填充缓存
    for i in range(100):
        cache.put("tool", {"id": i}, {"result": i})

    def get_operation():
        # 随机获取已存在的键
        cache_id = np.random.randint(0, 100)
        cache.get("tool", {"id": cache_id})

    return get_operation


def benchmark_cache_get_miss():
    """缓存GET操作基准（未命中）"""
    cache = ToolCallCache(max_size=1000)

    def get_miss_operation():
        # 获取不存在的键
        cache.get("tool", {"id": np.random.randint(1000, 2000)})

    return get_miss_operation


def benchmark_cache_generate_key():
    """缓存键生成基准"""
    cache = ToolCallCache(max_size=1000)

    def generate_key_operation():
        tool_name = "test_tool"
        params = {"operation": "test", "id": np.random.randint(0, 1000)}
        cache.generate_cache_key(tool_name, params)

    return generate_key_operation


def benchmark_cache_eviction():
    """缓存淘汰基准"""
    cache = ToolCallCache(max_size=100)

    def eviction_operation():
        # 添加超过限制的记录以触发淘汰
        for i in range(101):
            cache.put(f"tool_{i}", {"id": i}, {"result": i})

    return eviction_operation


# ============================================================================
# 生命周期管理器性能基准测试
# ============================================================================

def benchmark_lifecycle_register():
    """生命周期注册基准"""
    lifecycle = MemoryLifecycleManager(max_records=1000)

    def register_operation():
        memory_id = f"mem_{np.random.randint(0, 1000)}"
        lifecycle.register_record(
            memory_id=memory_id,
            importance_score=np.random.random(),
            tags=["benchmark"],
        )

    return register_operation


def benchmark_lifecycle_touch():
    """生命周期访问更新基准"""
    lifecycle = MemoryLifecycleManager(max_records=1000)

    # 预填充记录
    for i in range(100):
        lifecycle.register_record(f"mem_{i}", importance_score=0.5)

    def touch_operation():
        memory_id = f"mem_{np.random.randint(0, 100)}"
        lifecycle.touch_record(memory_id)

    return touch_operation


def benchmark_lifecycle_evict_lru():
    """生命周期LRU淘汰基准"""
    lifecycle = MemoryLifecycleManager(
        max_records=100,
        eviction_policy=EvictionPolicy.LRU,
    )

    def eviction_operation():
        # 添加记录
        for i in range(105):
            lifecycle.register_record(f"mem_{i}", importance_score=0.5)

    return eviction_operation


# ============================================================================
# 递归限制器性能基准测试
# ============================================================================

def benchmark_limiter_get_limit():
    """递归限制器获取限制基准"""
    limiter = DynamicRecursionLimiter()

    def get_limit_operation():
        context = {"task_complexity": np.random.random()}
        limiter.get_current_limit(context)

    return get_limit_operation


def benchmark_limiter_record_performance():
    """递归限制器记录性能基准"""
    limiter = DynamicRecursionLimiter()

    def record_performance_operation():
        limiter.record_performance(
            depth=np.random.randint(1, 10),
            success=np.random.choice([True, False]),
            execution_time_ms=np.random.uniform(10, 200),
        )

    return record_performance_operation


# ============================================================================
# 综合性能基准测试
# ============================================================================

def benchmark_full_decision_flow():
    """完整决策流程基准"""
    cache = ToolCallCache(max_size=100)
    lifecycle = MemoryLifecycleManager(max_records=100)
    limiter = DynamicRecursionLimiter()

    def decision_flow_operation():
        # 1. 获取递归限制
        context = {"task_complexity": 0.5}
        limit = limiter.get_current_limit(context)

        # 2. 缓存操作
        tool_name = "test_tool"
        params = {"operation": "test", "id": np.random.randint(0, 100)}
        cached = cache.get(tool_name, params)

        if cached is None:
            result = {"success": True, "data": "test"}
            cache.put(tool_name, params, result)

        # 3. 创建记忆记录（每10次）
        if np.random.randint(0, 10) == 0:
            cache_key = cache.generate_cache_key(tool_name, params)
            lifecycle.register_record(f"mem_{cache_key}", importance_score=0.5)

        # 4. 记录性能
        limiter.record_performance(
            depth=limit,
            success=True,
            execution_time_ms=50.0,
        )

    return decision_flow_operation


# ============================================================================
# 主函数
# ============================================================================

def run_all_benchmarks(runner: BenchmarkRunner, iterations: int = 100):
    """运行所有基准测试"""

    print("\n" + "=" * 60)
    print("🚀 开始性能基准测试")
    print("=" * 60)
    print()

    # 缓存基准测试
    print("📦 缓存性能基准")
    print("-" * 60)
    runner.run_benchmark(
        "Cache: PUT",
        benchmark_cache_put,
        iterations=iterations,
        warmup=10,
        component="cache",
        operation="put",
    )

    runner.run_benchmark(
        "Cache: GET (hit)",
        benchmark_cache_get,
        iterations=iterations,
        warmup=10,
        component="cache",
        operation="get_hit",
    )

    runner.run_benchmark(
        "Cache: GET (miss)",
        benchmark_cache_get_miss,
        iterations=iterations,
        warmup=10,
        component="cache",
        operation="get_miss",
    )

    runner.run_benchmark(
        "Cache: Generate Key",
        benchmark_cache_generate_key,
        iterations=iterations,
        warmup=10,
        component="cache",
        operation="generate_key",
    )

    runner.run_benchmark(
        "Cache: Eviction",
        benchmark_cache_eviction,
        iterations=10,  # 较少迭代，因为每次101次操作
        warmup=2,
        component="cache",
        operation="eviction",
    )

    # 生命周期管理器基准测试
    print("\n🧠 生命周期管理器性能基准")
    print("-" * 60)
    runner.run_benchmark(
        "Lifecycle: Register",
        benchmark_lifecycle_register,
        iterations=iterations,
        warmup=10,
        component="lifecycle",
        operation="register",
    )

    runner.run_benchmark(
        "Lifecycle: Touch",
        benchmark_lifecycle_touch,
        iterations=iterations,
        warmup=10,
        component="lifecycle",
        operation="touch",
    )

    runner.run_benchmark(
        "Lifecycle: Eviction (LRU)",
        benchmark_lifecycle_evict_lru,
        iterations=10,
        warmup=2,
        component="lifecycle",
        operation="eviction_lru",
    )

    # 递归限制器基准测试
    print("\n🔄 递归限制器性能基准")
    print("-" * 60)
    runner.run_benchmark(
        "Limiter: Get Limit",
        benchmark_limiter_get_limit,
        iterations=iterations,
        warmup=10,
        component="limiter",
        operation="get_limit",
    )

    runner.run_benchmark(
        "Limiter: Record Performance",
        benchmark_limiter_record_performance,
        iterations=iterations,
        warmup=10,
        component="limiter",
        operation="record_performance",
    )

    # 综合基准测试
    print("\n🔗 综合性能基准")
    print("-" * 60)
    runner.run_benchmark(
        "Full: Decision Flow",
        benchmark_full_decision_flow,
        iterations=iterations,
        warmup=10,
        component="full",
        operation="decision_flow",
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AGI系统性能基准测试"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="迭代次数，默认1000",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["cache", "lifecycle", "limiter", "full", "all"],
        default="all",
        help="运行特定基准测试",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="保存报告到指定文件",
    )

    args = parser.parse_args()

    runner = BenchmarkRunner()

    if args.benchmark == "all":
        run_all_benchmarks(runner, args.iterations)
    elif args.benchmark == "cache":
        # 只运行缓存基准
        pass  # 简化版本，默认运行所有
    else:
        run_all_benchmarks(runner, args.iterations)

    # 打印总结
    runner.print_summary()

    # 保存报告
    if args.report:
        runner.save_report(args.report)
    else:
        # 默认保存
        report_file = Path(__file__).parent / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        runner.save_report(str(report_file))

    return 0


if __name__ == "__main__":
    sys.exit(main())
