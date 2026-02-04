#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3 完整功能测试
==================

目的：验证Phase 3三大核心模块功能
测试内容：
1. 贝叶斯世界模型
2. 层级目标管理器
3. 创造性探索引擎
4. 综合集成测试

版本: 1.0.0
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模块
from core.bayesian_world_model import BayesianWorldModel
from core.hierarchical_goal_manager import HierarchicalGoalManager, GoalLevel, GoalStatus
from core.creative_exploration_engine import CreativeExplorationEngine, ExplorationMode


def test_world_model():
    """测试世界模型"""
    print("\n" + "=" * 70)
    print("[测试1] 贝叶斯世界模型")
    print("=" * 70)

    # 创建世界模型
    world = BayesianWorldModel(learning_rate=0.1)
    print("  [OK] World model created")

    # 测试信念更新
    print("\n  [1.1] 贝叶斯信念更新")
    for i in range(5):
        value = 0.5 + (i * 0.1)
        belief = world.observe("system_performance", value, confidence=0.8)
        print(f"    Observation[{i}]: {value:.2f} -> P={belief.probability:.3f}, conf={belief.confidence:.3f}")

    # 测试因果关系
    print("\n  [1.2] 因果关系学习")
    world.add_causal_link("cpu_usage", "performance", strength=0.7, mechanism="资源竞争")
    world.add_causal_link("memory_usage", "performance", strength=0.5, mechanism="内存压力")
    print("    [OK] 2 causal links added")

    # 测试预测
    print("\n  [1.3] 预测推理")
    predicted, conf = world.predict("performance", context={"cpu_usage": 80})
    print(f"    Predict performance: {predicted:.2f}, confidence: {conf:.2f}")

    # 测试干预
    print("\n  [1.4] 干预 do-calculus")
    intervention = world.intervene("cpu_usage", 50)
    print(f"    Intervention: do(cpu_usage=50)")
    print(f"    Predicted effects: {intervention.effect_prediction}")

    # 测试状态摘要
    summary = world.get_state_summary()
    print(f"\n  [摘要]")
    print(f"    Beliefs: {summary['total_beliefs']}")
    print(f"    Causal links: {summary['total_causal_links']}")
    print(f"    Interventions: {summary['total_interventions']}")

    print("\n  [OK] World model test passed")
    return world


def test_goal_manager():
    """测试目标管理器"""
    print("\n" + "=" * 70)
    print("[测试2] 层级目标管理器")
    print("=" * 70)

    # 创建目标管理器
    manager = HierarchicalGoalManager(max_active_goals=10)
    print("  [OK] Goal manager created")

    # 创建层级目标
    print("\n  [2.1] 创建目标层级")
    lifetime = manager.create_goal(
        name="achieve_intelligence",
        level=GoalLevel.LIFETIME,
        description="实现真正的人工智能",
        priority=1.0
    )
    print(f"    Created: {lifetime}")

    # 目标分解
    print("\n  [2.2] 目标分解")
    subgoals = manager.decompose_goal(lifetime.id)
    print(f"    Decomposed into {len(subgoals)} sub-goals")

    # 继续分解
    for subgoal in subgoals:
        if subgoal.level == GoalLevel.LONG_TERM:
            manager.decompose_goal(subgoal.id)

    # 激活目标
    print("\n  [2.3] 激活目标")
    manager.activate_goal(lifetime.id)
    active = manager.get_active_goals()
    print(f"    Active goals: {len(active)}")
    for goal in active[:3]:
        print(f"      - {goal.name} ({goal.level.value})")

    # 目标完成
    print("\n  [2.4] 目标完成")
    if len(active) > 1:
        first_active = active[0]
        manager.complete_goal(first_active.id, success=True)
        print(f"    Completed: {first_active.name}")

    # 冲突检测
    print("\n  [2.5] 冲突检测")
    manager.create_goal("competing_1", GoalLevel.SHORT_TERM, "竞争目标1", priority=0.9)
    manager.create_goal("competing_2", GoalLevel.SHORT_TERM, "竞争目标2", priority=0.9)
    manager.activate_goal("competing_1")
    manager.activate_goal("competing_2")
    conflicts = manager.detect_conflicts()
    print(f"    Conflicts detected: {len(conflicts)}")

    # 自主目标生成
    print("\n  [2.6] 自主目标生成")
    context = {"entropy": 0.8, "curiosity": 0.6, "performance": 0.4}
    new_goal = manager.generate_autonomous_goal(context)
    if new_goal:
        print(f"    Generated: {new_goal.name}")

    # 摘要
    summary = manager.get_summary()
    print(f"\n  [摘要]")
    print(f"    Total goals: {summary['total_goals']}")
    print(f"    Active: {summary['active_goals']}")
    print(f"    By level: {summary['by_level']}")

    print("\n  [OK] Goal manager test passed")
    return manager


def test_creative_engine():
    """测试创造性探索引擎"""
    print("\n" + "=" * 70)
    print("[测试3] 创造性探索引擎")
    print("=" * 70)

    # 创建探索引擎
    engine = CreativeExplorationEngine(temperature=0.7)
    print("  [OK] Creative engine created")

    # 测试类比推理
    print("\n  [3.1] 类比推理")
    result1 = engine.explore("优化算法性能", mode=ExplorationMode.ANALOGICAL)
    print(f"    Idea: {result1.output_idea[:80]}...")
    print(f"    Novelty: {result1.novelty_score:.2f}, Value: {result1.value_score:.2f}")

    # 测试概念组合
    print("\n  [3.2] 概念组合")
    result2 = engine.explore("创新系统架构", mode=ExplorationMode.COMBINATORIAL)
    print(f"    Idea: {result2.output_idea[:80]}...")
    print(f"    Novelty: {result2.novelty_score:.2f}, Value: {result2.value_score:.2f}")

    # 测试随机探索
    print("\n  [3.3] 随机探索")
    result3 = engine.explore("突破瓶颈", mode=ExplorationMode.STOCHASTIC)
    print(f"    Idea: {result3.output_idea[:80]}...")
    print(f"    Novelty: {result3.novelty_score:.2f}, Value: {result3.value_score:.2f}")

    # 测试自动模式选择
    print("\n  [3.4] 自动模式选择")
    contexts = [
        {"entropy": 0.2, "curiosity": 0.3},  # 低惊奇
        {"entropy": 0.5, "curiosity": 0.5},  # 中惊奇
        {"entropy": 0.9, "curiosity": 0.8}   # 高惊奇
    ]
    for i, ctx in enumerate(contexts, 1):
        result = engine.explore(f"查询{i}", context=ctx)
        print(f"    Context (E={ctx['entropy']:.1f}) -> {result.mode.value}")

    # 统计
    stats = engine.get_statistics()
    print(f"\n  [摘要]")
    print(f"    Total explorations: {stats['total_explorations']}")
    print(f"    Novel ideas: {stats['novel_ideas_generated']}")
    print(f"    Avg novelty: {stats['avg_novelty']:.3f}")

    print("\n  [OK] Creative engine test passed")
    return engine


def test_integration():
    """综合集成测试"""
    print("\n" + "=" * 70)
    print("[测试4] 综合集成测试")
    print("=" * 70)

    # 创建所有模块
    world = BayesianWorldModel(learning_rate=0.1)
    manager = HierarchicalGoalManager(max_active_goals=10)
    engine = CreativeExplorationEngine(temperature=0.7)

    print("  [OK] All modules created")

    # 模拟一个完整的智能循环
    print("\n  [4.1] 观察世界状态")
    world.observe("system_entropy", 0.75, confidence=0.8)
    world.observe("system_performance", 0.45, confidence=0.7)
    belief = world.get_belief("system_entropy")
    print(f"    Entropy belief: {belief}")

    print("\n  [4.2] 生成目标")
    context = {
        "entropy": belief.probability,
        "curiosity": 0.7,
        "performance": 0.45
    }
    goal = manager.generate_autonomous_goal(context)
    if goal:
        print(f"    Generated goal: {goal.name}")
        manager.activate_goal(goal.id)

    print("\n  [4.3] 创造性探索")
    result = engine.explore("如何降低系统熵", context=context)
    print(f"    Exploration mode: {result.mode.value}")
    print(f"    Novelty: {result.novelty_score:.2f}")

    print("\n  [4.4] 干预预测")
    world.add_causal_link("optimization", "entropy", strength=-0.6, mechanism="优化降低熵")
    intervention = world.intervene("optimization", 1.0)
    print(f"    Intervention: do(optimization=1.0)")
    print(f"    Predicted effects: {intervention.effect_prediction}")

    print("\n  [OK] Integration test passed")


def main():
    """主函数"""
    print("=" * 70)
    print("Phase 3 完整功能测试")
    print("=" * 70)

    try:
        # 测试1: 世界模型
        world = test_world_model()

        # 测试2: 目标管理器
        manager = test_goal_manager()

        # 测试3: 创造性探索
        engine = test_creative_engine()

        # 测试4: 综合集成
        test_integration()

        # 总结
        print("\n" + "=" * 70)
        print("[测试总结]")
        print("=" * 70)
        print("  [OK] 所有测试通过")
        print("  [SUCCESS] Phase 3 三大核心模块验证完成")
        print("\n关键成果:")
        print("  1. 贝叶斯世界模型 - 信念更新、因果推理、干预预测")
        print("  2. 层级目标管理器 - 目标分解、冲突检测、自主生成")
        print("  3. 创造性探索引擎 - 类比推理、概念组合、随机探索")
        print("  4. 综合集成 - 模块协同工作")

    except Exception as e:
        print(f"\n  [ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
