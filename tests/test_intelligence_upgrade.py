#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能升级系统集成测试
======================

目的：演示如何将新模块集成到现有AGI系统
测试内容：
1. 短期工作记忆打破循环
2. 因果推理引擎
3. 智能提升效果对比

版本: 1.0.0
"""

import sys
import time
import random
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入新模块
from core.working_memory import ShortTermWorkingMemory
from core.causal_reasoning import CausalReasoningEngine, Event


class SimulatedAGI:
    """模拟的AGI系统（用于测试）"""

    def __init__(self, use_new_modules=True):
        self.use_new_modules = use_new_modules

        # 新模块
        if use_new_modules:
            self.working_memory = ShortTermWorkingMemory(capacity=7)
            self.causal_engine = CausalReasoningEngine()
            print("[System] [OK] 启用新模块: WorkingMemory + CausalEngine")
        else:
            print("[System] [WARNING] 使用原始系统（无新模块）")

        # 状态追踪
        self.tick_count = 0
        self.thought_chain = []

    def run_tick(self):
        """运行一个Tick"""
        self.tick_count += 1

        if self.use_new_modules:
            return self._tick_with_new_modules()
        else:
            return self._tick_original()

    def _tick_with_new_modules(self):
        """使用新模块的Tick"""
        print(f"\n[Tick {self.tick_count}] (新模块)")

        # 1. 生成思想（带工作记忆）
        action = random.choice(['analyze', 'explore', 'create'])
        concept = f"Concept_{random.randint(0, 5)}"  # 故意限制范围以触发循环

        thought = self.working_memory.add_thought(action, concept)

        # 2. 记录思想链
        self.thought_chain.append(str(thought))

        # 3. 检查是否需要因果推理
        if action == 'analyze' and self.tick_count % 3 == 0:
            # 模拟因果推理
            events = self._generate_mock_events()
            self.causal_engine.infer_causality(events)

        # 4. 获取上下文摘要
        summary = self.working_memory.get_context_summary()

        print(f"  动作: {thought.action}")
        print(f"  概念: {thought.concept_id}")
        print(f"  多样性: {summary['diversity']:.2f}")
        print(f"  统计: {summary['stats']}")

        return thought

    def _tick_original(self):
        """原始系统的Tick（无新模块）"""
        print(f"\n[Tick {self.tick_count}] (原始系统)")

        # 模拟思想循环
        action = 'analyze'
        concept = f"Concept_11110111"

        thought = f"({action}) -> {concept}"

        # 记录（会重复）
        self.thought_chain.append(thought)

        print(f"  思想: {thought}")
        print(f"  (无循环检测)")

        return thought

    def _generate_mock_events(self):
        """生成模拟事件（用于因果推理）"""
        base_time = time.time()
        return [
            Event(id=f"E{self.tick_count}1",
                  type="user_action",
                  timestamp=base_time,
                  properties={"action": "query"}),
            Event(id=f"E{self.tick_count}2",
                  type="system_processing",
                  timestamp=base_time + 0.5,
                  properties={"component": "LLM"}),
            Event(id=f"E{self.tick_count}3",
                  type="response_generated",
                  timestamp=base_time + 1.0,
                  properties={"success": True}),
        ]

    def get_statistics(self):
        """获取统计信息"""
        if not self.use_new_modules:
            return {
                'ticks': self.tick_count,
                'total_thoughts': len(self.thought_chain),
                'unique_thoughts': len(set(self.thought_chain)),
                'diversity': len(set(self.thought_chain)) / len(self.thought_chain) if self.thought_chain else 0
            }

        return {
            'ticks': self.tick_count,
            'total_thoughts': len(self.thought_chain),
            'unique_thoughts': len(set(self.thought_chain)),
            'diversity': self.working_memory._calculate_diversity(),
            'working_memory_stats': self.working_memory.stats,
            'causal_stats': self.causal_engine.get_statistics()
        }


def run_comparison_test():
    """运行对比测试"""
    print("=" * 70)
    print("智能升级对比测试")
    print("=" * 70)

    # 测试参数
    ticks = 20

    # 测试1: 原始系统
    print("\n" + "=" * 70)
    print("[测试1] 原始系统（无新模块）")
    print("=" * 70)

    agi_original = SimulatedAGI(use_new_modules=False)
    for _ in range(ticks):
        agi_original.run_tick()

    stats_original = agi_original.get_statistics()

    # 测试2: 升级系统
    print("\n" + "=" * 70)
    print("[测试2] 升级系统（新模块）")
    print("=" * 70)

    agi_upgraded = SimulatedAGI(use_new_modules=True)
    for _ in range(ticks):
        agi_upgraded.run_tick()

    stats_upgraded = agi_upgraded.get_statistics()

    # 对比结果
    print("\n" + "=" * 70)
    print("[对比结果]")
    print("=" * 70)

    print(f"\n{'指标':<25} {'原始系统':<15} {'升级系统':<15} {'提升':<10}")
    print("-" * 70)
    print(f"{'总思想数':<25} {stats_original['total_thoughts']:<15} {stats_upgraded['total_thoughts']:<15} -")
    print(f"{'独特思想数':<25} {stats_original['unique_thoughts']:<15} {stats_upgraded['unique_thoughts']:<15} -")
    print(f"{'思想多样性':<25} {stats_original['diversity']:.2f}{'':<14} {stats_upgraded['diversity']:.2f}{'':<14}", end="")

    improvement = (stats_upgraded['diversity'] - stats_original['diversity']) / max(stats_original['diversity'], 0.01)
    print(f" x{improvement:.1f}")

    # 新模块统计
    if 'working_memory_stats' in stats_upgraded:
        wm_stats = stats_upgraded['working_memory_stats']
        print(f"\n[工作记忆统计]")
        print(f"  循环检测次数: {wm_stats['loops_detected']}")
        print(f"  循环打破次数: {wm_stats['loops_broken']}")
        print(f"  发散思想数: {wm_stats['divergent_thoughts']}")

    if 'causal_stats' in stats_upgraded:
        c_stats = stats_upgraded['causal_stats']
        print(f"\n[因果推理统计]")
        print(f"  事件数: {c_stats['total_events']}")
        print(f"  因果关系数: {c_stats['causal_relations']}")
        print(f"  平均置信度: {c_stats['avg_confidence']:.2f}")

    # 结论
    print("\n" + "=" * 70)
    print("[结论]")
    print("=" * 70)

    if stats_upgraded['diversity'] > stats_original['diversity']:
        print("[OK] 新模块显著提升了思想多样性")
        print(f"   提升倍数: {improvement:.1f}x")
        print("   -> 系统探索能力增强")
        print("   -> 智能特征更明显")
    else:
        print("[WARNING] 提升不明显，需要进一步调优")

    print("\n建议:")
    if stats_upgraded['working_memory_stats']['loops_detected'] == 0:
        print("- 增加测试复杂度以触发循环检测")
    else:
        print("- 循环检测机制工作正常")

    if stats_upgraded['causal_stats']['causal_relations'] == 0:
        print("- 需要更多真实事件以进行因果推理")
    else:
        print("- 因果推理机制工作正常")


def demonstrate_intelligence_features():
    """演示智能特征"""
    print("\n" + "=" * 70)
    print("[演示] 新系统的智能特征")
    print("=" * 70)

    # 创建升级系统
    agi = SimulatedAGI(use_new_modules=True)

    # 特征1: 循环检测与打破
    print("\n[特征1] 循环检测与打破")
    print("-" * 70)
    print("添加10个重复思想...")

    for i in range(10):
        thought = agi.working_memory.add_thought("analyze", "Concept_11110111")

    summary = agi.working_memory.get_context_summary()
    print(f"循环检测: {summary['stats']['loops_detected']}")
    print(f"循环打破: {summary['stats']['loops_broken']}")
    print(f"最终多样性: {summary['diversity']:.2f}")

    # 特征2: 因果推理
    print("\n[特征2] 因果推理能力")
    print("-" * 70)

    events = [
        Event(id="E1", type="问题提出", timestamp=time.time(),
              properties={"content": "系统为什么会陷入循环？"}),
        Event(id="E2", type="原因分析", timestamp=time.time()+0.5,
              properties={"cause": "缺乏工作记忆"}),
        Event(id="E3", type="解决方案", timestamp=time.time()+1.0,
              properties={"solution": "添加工作记忆模块"}),
    ]

    for event in events:
        agi.causal_engine.graph.add_event(event)

    causal_graph = agi.causal_engine.infer_causality(events)

    explanation = agi.causal_engine.explain_reasoning("为什么会提出解决方案？")
    print(f"\n{explanation}")

    # 特征3: 预测干预效果
    print("\n[特征3] 干预预测")
    print("-" * 70)

    intervention = {"问题提出": {"urgency": "high"}}
    prediction = agi.causal_engine.predict_intervention(intervention)
    print(f"干预: {intervention}")
    print(f"预测效果: {prediction['effect']:.3f}")


if __name__ == "__main__":
    # 运行对比测试
    run_comparison_test()

    # 演示智能特征
    demonstrate_intelligence_features()

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
