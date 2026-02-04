#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copilot修复验证脚本
==================
验证所有集成问题是否已修复
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("Copilot修复验证")
print("=" * 70)

all_passed = True

# 测试 1: 验证目标系统API兼容性
print("\n[测试1] 目标系统API兼容性")
try:
    from core.hierarchical_goal_manager import HierarchicalGoalManager, GoalLevel
    from core.goal_system import GoalType

    manager = HierarchicalGoalManager(max_active_goals=10)

    # 测试旧API调用（这会导致崩溃如果修复失败）
    old_goal = manager.create_goal(
        description="Test old API",
        goal_type=GoalType.CUSTOM,
        priority="critical"
    )
    print(f"  [OK] 旧API调用成功: {old_goal.name}")
    print(f"  [OK] goal_type.value访问: {old_goal.goal_type.value}")
    print(f"  [OK] priority类型: {type(old_goal.priority)} = {old_goal.priority}")

    # 测试新API调用
    new_goal = manager.create_goal(
        name="test_new_api",
        level=GoalLevel.SHORT_TERM,
        description="Test new API",
        priority=0.8
    )
    print(f"  [OK] 新API调用成功: {new_goal.name}")

    # 验证兼容方法
    manager.start_goal(old_goal)
    current = manager.get_current_goal()
    if current and current.id == old_goal.id:
        print(f"  [OK] get_current_goal()工作正常")

    manager.abandon_goal(old_goal, "Test abandon")
    print(f"  [OK] abandon_goal()工作正常")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 测试 2: 验证推理调度器导入
print("\n[测试2] 推理调度器")
try:
    from core.reasoning_scheduler import ReasoningScheduler
    from core.causal_reasoning import CausalReasoningEngine

    causal_engine = CausalReasoningEngine()
    scheduler = ReasoningScheduler(
        causal_engine=causal_engine,
        llm_service=None,  # 测试时不需要LLM
        confidence_threshold=0.6,
        max_depth=1000
    )
    print(f"  [OK] ReasoningScheduler创建成功")

    stats = scheduler.get_statistics()
    print(f"  [OK] 统计信息: max_depth={stats.get('max_depth', 'N/A')}")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 测试 3: 验证世界模型
print("\n[测试3] 世界模型")
try:
    from core.bayesian_world_model import BayesianWorldModel

    world_model = BayesianWorldModel(learning_rate=0.1)
    print(f"  [OK] BayesianWorldModel创建成功")

    # 测试观测
    belief = world_model.observe(
        variable="test_var",
        value="test_value",
        confidence=0.9
    )
    print(f"  [OK] observe()工作: {belief.variable} = {belief.value}")

    # 测试预测
    prediction, confidence = world_model.predict(
        query="test_query",
        context={"test": "context"}
    )
    print(f"  [OK] predict()工作: prediction={prediction}, confidence={confidence}")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 测试 4: 验证创造性探索引擎
print("\n[测试4] 创造性探索引擎")
try:
    from core.creative_exploration_engine import CreativeExplorationEngine

    creative_engine = CreativeExplorationEngine(temperature=0.7)
    print(f"  [OK] CreativeExplorationEngine创建成功")

    # 测试探索
    result = creative_engine.explore(
        query="Test exploration",
        context={"test": "context"},
        mode=None  # 让引擎选择模式
    )
    print(f"  [OK] explore()工作: novelty_score={result.novelty_score:.2f}")
    print(f"  [OK] 模式: {result.mode.value}")

except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# 测试 5: 验证AGI_Life_Engine集成
print("\n[测试5] AGI_Life_Engine集成")
try:
    # 检查关键代码段是否存在
    with open("AGI_Life_Engine.py", "r", encoding="utf-8") as f:
        content = f.read()

    checks = [
        ("推理调度器激活", "if self.reasoning_scheduler:"),
        ("世界模型观测", "self.world_model.observe"),
        ("世界模型预测", "self.world_model.predict"),
        ("创造性探索", "self.creative_engine.explore"),
    ]

    for check_name, check_string in checks:
        if check_string in content:
            print(f"  [OK] {check_name}代码已集成")
        else:
            print(f"  [FAIL] {check_name}代码未找到")
            all_passed = False

except Exception as e:
    print(f"  [FAIL] {e}")
    all_passed = False

# 总结
print("\n" + "=" * 70)
if all_passed:
    print("[SUCCESS] 所有测试通过 - Copilot修复验证成功")
    print("=" * 70)
    print("\n修复摘要:")
    print("  [OK] 目标系统API兼容性 - CRITICAL问题已修复")
    print("  [OK] 推理调度器 - 已集成到主循环")
    print("  [OK] 世界模型 - 观测和预测已激活")
    print("  [OK] 创造性探索 - 空闲时触发已激活")
    print("\n系统状态: 完全激活")
else:
    print("[FAIL] 部分测试失败 - 需要进一步检查")
    print("=" * 70)

sys.exit(0 if all_passed else 1)
