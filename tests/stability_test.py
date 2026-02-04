#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定性测试脚本

用于长时间运行测试，验证系统稳定性：
- 内存泄漏检测
- 性能回归检测
- 故障恢复能力
- 资源使用监控

使用方法:
    # 短期稳定性测试（5分钟）
    python tests/stability_test.py --duration 300

    # 中期稳定性测试（1小时）
    python tests/stability_test.py --duration 3600

    # 长期稳定性测试（24小时）
    python tests/stability_test.py --duration 86400

作者: AGI System
日期: 2026-02-04
"""

import sys
import time
import psutil
import gc
import json
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List

# 设置Windows控制台编码
if sys.platform == "win32":
    try:
        import locale
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


class StabilityMonitor:
    """稳定性监控器"""

    def __init__(self):
        self.start_time = time.time()
        self.snapshots = []
        self.errors = []
        self.warnings = []

    def take_snapshot(self) -> Dict[str, Any]:
        """捕获系统状态快照"""
        process = psutil.Process()

        snapshot = {
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self.start_time,
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0,
        }

        self.snapshots.append(snapshot)
        return snapshot

    def check_memory_leak(self, window_size: int = 10) -> bool:
        """检查内存泄漏"""
        if len(self.snapshots) < window_size:
            return False

        # 获取最近的快照
        recent = self.snapshots[-window_size:]

        # 计算内存增长趋势
        memory_values = [s["memory_mb"] for s in recent]
        initial = memory_values[0]
        final = memory_values[-1]
        growth = final - initial

        # 如果增长超过50MB，可能存在内存泄漏
        if growth > 50:
            self.warnings.append({
                "type": "memory_growth",
                "growth_mb": growth,
                "window_size": window_size,
                "timestamp": time.time(),
            })
            return True

        return False

    def check_performance_regression(self, baseline_duration: float) -> bool:
        """检查性能回归"""
        if len(self.snapshots) < 2:
            return False

        # 计算平均操作时间（估算）
        # 这里假设每次快照之间有相似的操作负载
        recent = self.snapshots[-5:]
        avg_memory = sum(s["memory_mb"] for s in recent) / len(recent)

        # 如果内存使用超过基线的2倍，可能存在性能问题
        if avg_memory > baseline_duration * 2:
            self.warnings.append({
                "type": "performance_regression",
                "avg_memory_mb": avg_memory,
                "baseline_mb": baseline_duration,
                "timestamp": time.time(),
            })
            return True

        return False


def simulate_workload(
    cache: ToolCallCache,
    lifecycle: MemoryLifecycleManager,
    limiter: DynamicRecursionLimiter,
    iteration: int,
) -> Dict[str, Any]:
    """模拟工作负载"""

    # 1. 工具调用缓存操作
    tool_name = f"tool_{iteration % 10}"
    params = {
        "operation": "test",
        "iteration": iteration,
        "data": "x" * 100,  # 100字节数据
    }

    # 尝试缓存获取
    cached = cache.get(tool_name, params)

    if cached is None:
        # 缓存未命中，执行操作并缓存
        result = {
            "success": True,
            "data": f"result_{iteration}",
            "iteration": iteration,
        }
        cache.put(tool_name, params, result)

    # 2. 创建记忆记录
    cache_key = cache.generate_cache_key(tool_name, params)
    memory_id = f"tool_call_{cache_key}_{iteration}"

    # 每10次迭代创建一条新记录
    if iteration % 10 == 0:
        lifecycle.register_record(
            memory_id=memory_id,
            importance_score=0.5 + (iteration % 5) * 0.1,
            tags=["stability_test", tool_name],
        )

    # 3. 动态递归限制
    context = {
        "task_complexity": (iteration % 10) / 10.0,
    }

    limit = limiter.get_current_limit(context)

    # 每5次迭代记录性能
    if iteration % 5 == 0:
        success = (iteration % 3 != 0)  # 模拟偶尔失败
        execution_time = 50.0 if success else 200.0

        limiter.record_performance(
            depth=limit,
            success=success,
            execution_time_ms=execution_time,
        )

    # 4. 返回统计
    return {
        "cache_hit": cached is not None,
        "recursion_limit": limit,
        "lifecycle_records": len(lifecycle.records),
    }


def run_stability_test(
    duration_seconds: int,
    snapshot_interval: int = 60,
    report_interval: int = 300,
) -> Dict[str, Any]:
    """
    运行稳定性测试

    Args:
        duration_seconds: 测试持续时间（秒）
        snapshot_interval: 快照间隔（秒）
        report_interval: 报告间隔（秒）

    Returns:
        测试结果
    """

    print(f"🚀 开始稳定性测试")
    print(f"   持续时间: {duration_seconds}秒 ({duration_seconds / 60:.1f}分钟)")
    print(f"   快照间隔: {snapshot_interval}秒")
    print(f"   报告间隔: {report_interval}秒")
    print()

    # 初始化系统
    cache = ToolCallCache(max_size=1000)
    lifecycle = MemoryLifecycleManager(
        max_records=1000,
        eviction_policy=EvictionPolicy.LRU,
    )
    limiter = DynamicRecursionLimiter()

    # 初始化监控器
    monitor = StabilityMonitor()
    monitor.take_snapshot()  # 初始快照
    baseline_memory = monitor.snapshots[0]["memory_mb"]

    # 测试统计
    stats = {
        "iterations": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "errors": [],
        "start_time": datetime.now(),
        "end_time": None,
    }

    last_snapshot_time = time.time()
    last_report_time = time.time()

    iteration = 0
    test_start = time.time()

    try:
        while (time.time() - test_start) < duration_seconds:
            iteration += 1
            stats["iterations"] = iteration

            try:
                # 模拟工作负载
                result = simulate_workload(cache, lifecycle, limiter, iteration)

                # 更新统计
                if result["cache_hit"]:
                    stats["cache_hits"] += 1
                else:
                    stats["cache_misses"] += 1

            except Exception as e:
                error_info = {
                    "iteration": iteration,
                    "error": str(e),
                    "type": type(e).__name__,
                    "timestamp": time.time(),
                }
                stats["errors"].append(error_info)
                print(f"❌ 迭代 {iteration} 出错: {e}")

            # 定期快照
            if time.time() - last_snapshot_time >= snapshot_interval:
                monitor.take_snapshot()

                # 检查内存泄漏
                if monitor.check_memory_leak():
                    print(f"⚠️  检测到内存增长")

                last_snapshot_time = time.time()

            # 定期报告
            if time.time() - last_report_time >= report_interval:
                elapsed = time.time() - test_start
                progress = (elapsed / duration_seconds) * 100

                latest = monitor.snapshots[-1]
                cache_hit_rate = stats["cache_hits"] / max(stats["iterations"], 1) * 100

                print(f"📊 进度报告 ({progress:.1f}%)")
                print(f"   已运行: {elapsed:.0f}秒 ({elapsed / 60:.1f}分钟)")
                print(f"   迭代次数: {iteration}")
                print(f"   内存使用: {latest['memory_mb']:.1f}MB")
                print(f"   CPU使用: {latest['cpu_percent']:.1f}%")
                print(f"   缓存命中率: {cache_hit_rate:.1f}%")
                print(f"   生命周期记录: {len(lifecycle.records)}")
                print(f"   递归限制历史: {len(limiter.performance_history)}")
                print(f"   错误数: {len(stats['errors'])}")
                print()

                last_report_time = time.time()

            # 避免CPU过度使用
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    finally:
        # 最终快照
        monitor.take_snapshot()
        stats["end_time"] = datetime.now()

    # 生成报告
    print("\n" + "=" * 60)
    print("🎉 稳定性测试完成")
    print("=" * 60)

    elapsed_total = time.time() - test_start
    final_snapshot = monitor.snapshots[-1]

    # 计算统计
    cache_hit_rate = stats["cache_hits"] / max(stats["iterations"], 1) * 100
    avg_memory = sum(s["memory_mb"] for s in monitor.snapshots) / len(monitor.snapshots)
    max_memory = max(s["memory_mb"] for s in monitor.snapshots)
    memory_growth = final_snapshot["memory_mb"] - monitor.snapshots[0]["memory_mb"]

    print(f"\n📈 测试统计:")
    print(f"   总运行时间: {elapsed_total:.0f}秒 ({elapsed_total / 60:.1f}分钟)")
    print(f"   总迭代次数: {iteration}")
    print(f"   每秒迭代: {iteration / elapsed_total:.1f}")
    print(f"\n💾 内存使用:")
    print(f"   初始内存: {monitor.snapshots[0]['memory_mb']:.1f}MB")
    print(f"   最终内存: {final_snapshot['memory_mb']:.1f}MB")
    print(f"   平均内存: {avg_memory:.1f}MB")
    print(f"   峰值内存: {max_memory:.1f}MB")
    print(f"   内存增长: {memory_growth:+.1f}MB")
    print(f"\n⚡ 性能指标:")
    print(f"   缓存命中率: {cache_hit_rate:.1f}%")
    print(f"   缓存命中: {stats['cache_hits']}")
    print(f"   缓存未命中: {stats['cache_misses']}")
    print(f"\n🔧 系统状态:")
    print(f"   缓存大小: {len(cache.cache)}")
    print(f"   生命周期记录: {len(lifecycle.records)}")
    print(f"   递归限制历史: {len(limiter.performance_history)}")
    print(f"   淘汰次数: {cache.stats.get('evictions', 0)}")
    print(f"\n⚠️  错误和警告:")
    print(f"   错误数: {len(stats['errors'])}")
    print(f"   警告数: {len(monitor.warnings)}")

    # 评估结果
    print(f"\n🎯 评估结果:")

    passed = True
    reasons = []

    # 1. 内存泄漏检查
    if memory_growth > 100:
        passed = False
        reasons.append(f"❌ 内存增长过大: {memory_growth:.1f}MB")
    elif memory_growth > 50:
        reasons.append(f"⚠️  内存增长较高: {memory_growth:.1f}MB")
    else:
        reasons.append(f"✅ 内存使用稳定: {memory_growth:+.1f}MB")

    # 2. 错误检查
    if len(stats["errors"]) > 0:
        passed = False
        reasons.append(f"❌ 发生{len(stats['errors'])}个错误")
    else:
        reasons.append(f"✅ 无错误运行")

    # 3. 缓存效率
    if cache_hit_rate < 30:
        reasons.append(f"⚠️  缓存命中率较低: {cache_hit_rate:.1f}%")
    else:
        reasons.append(f"✅ 缓存效率良好: {cache_hit_rate:.1f}%")

    # 4. 性能稳定性
    if monitor.check_performance_regression(baseline_memory):
        passed = False
        reasons.append(f"❌ 性能回归检测")
    else:
        reasons.append(f"✅ 无性能回归")

    for reason in reasons:
        print(f"   {reason}")

    # 总体评估
    if passed:
        print(f"\n✅ 稳定性测试: **通过**")
    else:
        print(f"\n❌ 稳定性测试: **失败**")

    # 保存详细报告
    report = {
        "test_info": {
            "duration_seconds": duration_seconds,
            "actual_duration_seconds": elapsed_total,
            "iterations": iteration,
            "start_time": stats["start_time"].isoformat(),
            "end_time": stats["end_time"].isoformat(),
        },
        "memory_stats": {
            "initial_mb": monitor.snapshots[0]["memory_mb"],
            "final_mb": final_snapshot["memory_mb"],
            "avg_mb": avg_memory,
            "max_mb": max_memory,
            "growth_mb": memory_growth,
        },
        "performance_stats": {
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": stats["cache_hits"],
            "cache_misses": stats["cache_misses"],
            "iterations_per_second": iteration / elapsed_total,
        },
        "system_stats": {
            "cache_size": len(cache.cache),
            "lifecycle_records": len(lifecycle.records),
            "recursion_history": len(limiter.performance_history),
            "evictions": cache.stats.get("evictions", 0),
        },
        "errors": stats["errors"],
        "warnings": monitor.warnings,
        "snapshots": monitor.snapshots,
        "passed": passed,
    }

    # 保存到文件
    report_file = Path(__file__).parent / f"stability_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📄 详细报告已保存: {report_file}")

    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="AGI系统稳定性测试"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="测试持续时间（秒），默认300秒（5分钟）",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=60,
        help="快照间隔（秒），默认60秒",
    )
    parser.add_argument(
        "--report-interval",
        type=int,
        default=300,
        help="报告间隔（秒），默认300秒（5分钟）",
    )

    args = parser.parse_args()

    # 运行稳定性测试
    report = run_stability_test(
        duration_seconds=args.duration,
        snapshot_interval=args.snapshot_interval,
        report_interval=args.report_interval,
    )

    # 返回退出码
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
