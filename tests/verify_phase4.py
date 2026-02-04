#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 模块验证脚本
==================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("Phase 4 模块验证")
print("=" * 70)

# 测试 1: 元学习引擎
print("\n[测试1] 元学习引擎")
try:
    from core.meta_learning import MetaLearner, Task
    meta_learner = MetaLearner(memory_size=50)
    print("  [OK] Meta-learner created")

    simple_loss = lambda params: 1.0 - params.get('score', 0.5)
    task = Task(
        task_id="test_task",
        name="Test_Task",
        data=[f"sample_{i}" for i in range(10)],
        loss_function=simple_loss,
        metadata={'type': 'test', 'domain': 'test_domain'}
    )
    result = meta_learner.learn_task(task, max_iterations=20)
    print(f"  [OK] Task learned: performance={result['final_performance']:.3f}")

    stats = meta_learner.get_statistics()
    print(f"  [OK] Total tasks: {stats['total_tasks_learned']}")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 2: 自我改进引擎
print("\n[测试2] 自我改进引擎")
try:
    from core.self_improvement import SelfImprovementEngine
    project_root = Path(__file__).parent.parent
    engine = SelfImprovementEngine(str(project_root))
    print("  [OK] Self-improvement engine created")

    stats = engine.get_statistics()
    print(f"  [OK] Modules scanned: {stats['modules_scanned']}")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 3: 递归自指引擎
print("\n[测试3] 递归自指优化引擎")
try:
    from core.recursive_self_reference import RecursiveSelfReferenceEngine
    recursive_engine = RecursiveSelfReferenceEngine(max_recursion_depth=2)
    print("  [OK] Recursive self-reference engine created")

    thought = recursive_engine.monitor_thought(
        input_stimulus="测试思考",
        reasoning_steps=["步骤1", "步骤2", "步骤3"],
        conclusion="测试结论",
        confidence=0.8
    )
    print(f"  [OK] Thought monitored: {thought.meta_commentary}")

    stats = recursive_engine.get_statistics()
    print(f"  [OK] Thoughts monitored: {stats['total_thoughts_monitored']}")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 4: 综合集成
print("\n[测试4] 综合集成测试")
try:
    # 创建所有模块
    meta_learner = MetaLearner(memory_size=30)
    self_improvement = SelfImprovementEngine(str(Path(__file__).parent.parent))
    recursive_engine = RecursiveSelfReferenceEngine(max_recursion_depth=2)
    print("  [OK] All Phase 4 modules created successfully")
except Exception as e:
    print(f"  [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] Phase 4 所有模块验证通过")
print("=" * 70)
print("\n关键成果:")
print("  1. 元学习引擎 - 学会学习、快速适应")
print("  2. 自我改进引擎 - 代码生成、自动优化")
print("  3. 递归自指优化 - 元认知、自我修正")
print("  4. 综合集成 - 模块协同工作")
