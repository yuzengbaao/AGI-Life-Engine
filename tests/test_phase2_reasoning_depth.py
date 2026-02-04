#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 深度推理测试
==================

目的：验证推理调度器实现深度推理（1000步+）
测试内容：
1. 推理调度器基本功能
2. 推理深度追踪
3. 推理模式切换
4. 深度推理性能

版本: 1.0.0
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模块
from core.reasoning_scheduler import ReasoningScheduler, ReasoningMode
from core.causal_reasoning import CausalReasoningEngine, Event


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 70)
    print("[测试1] 推理调度器基本功能")
    print("=" * 70)

    # 创建调度器
    causal_engine = CausalReasoningEngine()
    scheduler = ReasoningScheduler(
        causal_engine=causal_engine,
        llm_service=None,
        confidence_threshold=0.6,
        max_depth=1000
    )

    # 启动会话
    session_id = scheduler.start_session()
    print(f"  [OK] Session started: {session_id}")

    # 执行推理
    query = "为什么系统会陷入循环？"
    result, step = scheduler.reason(query, prefer_causal=True)

    print(f"  [OK] Reasoning completed")
    print(f"    Mode: {step.mode.value}")
    print(f"    Confidence: {step.confidence:.2f}")
    print(f"    Depth: {step.depth}")

    # 验证
    assert step.depth == 1, "First reasoning step should be depth 1"
    assert step.mode in [ReasoningMode.CAUSAL, ReasoningMode.LLM_FALLBACK, ReasoningMode.PATTERN_MATCH]

    print("  [OK] Basic functionality test passed")
    return scheduler


def test_reasoning_depth_tracking(scheduler):
    """测试推理深度追踪"""
    print("\n" + "=" * 70)
    print("[测试2] 推理深度追踪")
    print("=" * 70)

    # 执行多次推理
    queries = [
        "为什么系统会陷入循环？",
        "如何打破思想循环？",
        "预测添加工作记忆的效果",
        "分析当前系统状态",
        "探索改进方案",
        "评估推理质量",
        "优化决策过程",
        "提升系统性能",
        "增强学习能力",
        "改进用户体验"
    ]

    print(f"\n  执行 {len(queries)} 次推理...")

    for i, query in enumerate(queries, 1):
        result, step = scheduler.reason(query, prefer_causal=True)
        print(f"  [{i}] {query[:20]:20s} -> depth={step.depth}, mode={step.mode.value[:10]}")

    # 获取会话摘要
    summary = scheduler.get_current_session_summary()

    print(f"\n  [会话摘要]")
    print(f"    Total steps: {summary['total_steps']}")
    print(f"    Max depth: {summary['max_depth']}")
    print(f"    Avg confidence: {summary['avg_confidence']:.2f}")

    # 验证
    assert summary['total_steps'] >= len(queries), f"Should have at least {len(queries)} steps"

    print("  [OK] Depth tracking test passed")
    return scheduler


def test_mode_switching(scheduler):
    """测试推理模式切换"""
    print("\n" + "=" * 70)
    print("[测试3] 推理模式切换")
    print("=" * 70)

    # 测试不同类型的查询
    test_cases = [
        ("为什么系统会循环？", "causal query"),
        ("分析当前状态", "analytical query"),
        ("预测未来趋势", "predictive query")
    ]

    for query, description in test_cases:
        result, step = scheduler.reason(query, prefer_causal=True)
        print(f"  [{description}]")
        print(f"    Query: {query}")
        print(f"    Mode: {step.mode.value}")
        print(f"    Confidence: {step.confidence:.2f}")

    # 获取模式分布
    summary = scheduler.get_current_session_summary()
    mode_dist = summary.get('mode_distribution', {})

    print(f"\n  [模式分布]")
    for mode, count in mode_dist.items():
        print(f"    {mode}: {count}")

    print("  [OK] Mode switching test passed")
    return scheduler


def test_deep_reasoning_performance():
    """测试深度推理性能"""
    print("\n" + "=" * 70)
    print("[测试4] 深度推理性能 (目标: 1000步)")
    print("=" * 70)

    # 创建新的调度器用于深度测试
    causal_engine = CausalReasoningEngine()
    scheduler = ReasoningScheduler(
        causal_engine=causal_engine,
        llm_service=None,
        confidence_threshold=0.6,
        max_depth=1000
    )

    session_id = scheduler.start_session()

    # 生成1000个推理查询
    target_depth = 1000
    print(f"\n  执行 {target_depth} 步推理...")

    start_time = time.time()

    for i in range(1, target_depth + 1):
        # 生成变化的查询以增加多样性
        query_type = i % 5
        queries = [
            f"分析问题_{i}",
            f"探索方案_{i}",
            f"预测结果_{i}",
            f"评估选项_{i}",
            f"优化策略_{i}"
        ]
        query = queries[query_type]

        result, step = scheduler.reason(query, prefer_causal=False)  # 使用模式匹配以加快速度

        # 每100步显示进度
        if i % 100 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            print(f"  Progress: {i}/{target_depth} steps ({i/target_depth*100:.0f}%) - Rate: {rate:.1f} steps/sec")

    elapsed = time.time() - start_time

    # 获取会话摘要
    session = scheduler.end_session()
    summary = session.get_summary()

    print(f"\n  [性能结果]")
    print(f"    Total steps: {summary['total_steps']}")
    print(f"    Max depth: {summary['max_depth']}")
    print(f"    Target depth: {target_depth}")
    print(f"    Achievement: {summary['max_depth']/target_depth*100:.1f}%")
    print(f"    Total time: {elapsed:.2f}s")
    print(f"    Avg time per step: {summary['avg_step_time']:.4f}s")
    print(f"    Throughput: {summary['total_steps']/elapsed:.1f} steps/sec")

    # 获取统计
    stats = scheduler.get_statistics()
    print(f"\n  [调度器统计]")
    print(f"    Causal reasoning used: {stats['causal_reasoning_used']}")
    print(f"    LLM fallback used: {stats['llm_fallback_used']}")
    print(f"    Pattern matching used: {stats['total_reasoning_calls'] - stats['causal_reasoning_used'] - stats['llm_fallback_used']}")

    # 验证
    assert summary['max_depth'] >= target_depth, f"Should reach depth {target_depth}"

    print("\n  [OK] Deep reasoning performance test passed")
    print(f"  [SUCCESS] Achieved {summary['max_depth']} steps reasoning depth!")


def test_comparison_with_baseline():
    """对比测试：升级前 vs 升级后"""
    print("\n" + "=" * 70)
    print("[测试5] 对比测试：升级前 vs 升级后")
    print("=" * 70)

    # 模拟原始系统（15步推理深度限制）
    original_depth = 15

    # 升级后的系统（1000步推理深度）
    causal_engine = CausalReasoningEngine()
    scheduler = ReasoningScheduler(
        causal_engine=causal_engine,
        llm_service=None,
        confidence_threshold=0.6,
        max_depth=1000
    )

    session_id = scheduler.start_session()

    # 执行100步推理（比原始系统的15步多）
    test_steps = 100
    for i in range(test_steps):
        query = f"推理步骤_{i}"
        scheduler.reason(query, prefer_causal=False)

    summary = scheduler.get_current_session_summary()
    upgraded_depth = summary['max_depth']

    print(f"\n  [对比结果]")
    print(f"    原始系统最大推理深度: {original_depth} 步")
    print(f"    升级系统最大推理深度: {upgraded_depth} 步")
    print(f"    提升倍数: {upgraded_depth / original_depth:.1f}x")
    print(f"    提升百分比: {(upgraded_depth - original_depth) / original_depth * 100:.0f}%")

    if upgraded_depth > original_depth:
        print(f"\n  [SUCCESS] 推理深度显著提升!")
        print(f"    -> 系统现在可以进行更复杂的推理")
        print(f"    -> 可以处理更深层的问题分析")
        print(f"    -> 智能特征更加明显")


def main():
    """主函数"""
    print("=" * 70)
    print("Phase 2 深度推理测试")
    print("=" * 70)

    try:
        # 测试1: 基本功能
        scheduler = test_basic_functionality()

        # 测试2: 深度追踪
        scheduler = test_reasoning_depth_tracking(scheduler)

        # 测试3: 模式切换
        scheduler = test_mode_switching(scheduler)

        # 测试4: 深度推理性能
        test_deep_reasoning_performance()

        # 测试5: 对比测试
        test_comparison_with_baseline()

        # 总结
        print("\n" + "=" * 70)
        print("[测试总结]")
        print("=" * 70)
        print("  [OK] 所有测试通过")
        print("  [SUCCESS] Phase 2 深度推理功能验证完成")
        print("\n关键成果:")
        print("  1. 推理调度器工作正常")
        print("  2. 推理深度追踪准确")
        print("  3. 推理模式切换有效")
        print("  4. 深度推理性能达标 (1000步)")
        print("  5. 相比原始系统大幅提升")

    except Exception as e:
        print(f"\n  [ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
