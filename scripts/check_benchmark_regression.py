#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准回归检测脚本

用于对比两次基准测试的结果，检测性能回归。

使用方法:
    python scripts/check_benchmark_regression.py baseline.json current.json

作者: AGI System
日期: 2026-02-04
"""

import sys
import json
import argparse
from pathlib import Path

# 设置Windows控制台编码
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    except:
        pass


def load_report(filepath: str) -> dict:
    """加载基准测试报告"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def check_regression(baseline: dict, current: dict, threshold: float = 1.5) -> bool:
    """
    检查性能回归

    Args:
        baseline: 基线报告
        current: 当前报告
        threshold: 回归阈值（倍数），默认1.5倍即50%性能下降

    Returns:
        True if regression detected, False otherwise
    """
    baseline_results = {r["name"]: r for r in baseline["results"]}
    current_results = {r["name"]: r for r in current["results"]}

    regressions = []
    improvements = []

    for name, current_result in current_results.items():
        if name not in baseline_results:
            continue

        baseline_result = baseline_results[name]
        baseline_time = baseline_result["avg_time"]
        current_time = current_result["avg_time"]

        # 计算性能变化
        if baseline_time > 0:
            time_ratio = current_time / baseline_time
            percent_change = ((current_time - baseline_time) / baseline_time) * 100
        else:
            time_ratio = 1.0
            percent_change = 0.0

        # 检测回归（性能下降超过阈值）
        if time_ratio > threshold:
            regressions.append({
                "name": name,
                "baseline_time": baseline_time,
                "current_time": current_time,
                "time_ratio": time_ratio,
                "percent_change": percent_change,
            })
        elif time_ratio < (1 / threshold):
            improvements.append({
                "name": name,
                "baseline_time": baseline_time,
                "current_time": current_time,
                "time_ratio": time_ratio,
                "percent_change": percent_change,
            })

    # 打印结果
    print("=" * 60)
    print("📊 性能回归检测报告")
    print("=" * 60)

    if regressions:
        print(f"\n⚠️  检测到性能回归 ({len(regressions)}项):")
        print("-" * 60)

        for reg in regressions:
            print(f"\n操作: {reg['name']}")
            print(f"  基线时间: {reg['baseline_time']:.3f}ms")
            print(f"  当前时间: {reg['current_time']:.3f}ms")
            print(f"  时间比率: {reg['time_ratio']:.2f}x")
            print(f"  性能下降: {reg['percent_change']:+.1f}%")
            print(f"  状态: ❌ 性能回归")

    if improvements:
        print(f"\n✅ 性能提升 ({len(improvements)}项):")
        print("-" * 60)

        for imp in improvements[:5]:  # 只显示前5个
            print(f"\n操作: {imp['name']}")
            print(f"  基线时间: {imp['baseline_time']:.3f}ms")
            print(f"  当前时间: {imp['current_time']:.3f}ms")
            print(f"  时间比率: {imp['time_ratio']:.2f}x")
            print(f"  性能提升: {imp['percent_change']:+.1f}%")
            print(f"  状态: ✅ 性能提升")

    # 无显著变化
    if not regressions and not improvements:
        print("\n✅ 无显著性能变化")
        return False

    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"基线报告: {baseline.get('timestamp', 'N/A')}")
    print(f"当前报告: {current.get('timestamp', 'N/A')}")
    print(f"回归阈值: {threshold}x ({(threshold - 1) * 100:.0f}% 性能下降)")
    print(f"性能回归: {len(regressions)}项")
    print(f"性能提升: {len(improvements)}项")

    if regressions:
        print("\n❌ 检测到性能回归！")
        return True
    else:
        print("\n✅ 无性能回归")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="检查性能基准回归"
    )
    parser.add_argument(
        "baseline",
        type=str,
        help="基线报告文件路径",
    )
    parser.add_argument(
        "current",
        type=str,
        help="当前报告文件路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="回归阈值（倍数），默认1.5（即50%性能下降）",
    )

    args = parser.parse_args()

    # 加载报告
    baseline = load_report(args.baseline)
    current = load_report(args.current)

    # 检查回归
    has_regression = check_regression(baseline, current, args.threshold)

    # 返回退出码
    sys.exit(1 if has_regression else 0)


if __name__ == "__main__":
    main()
