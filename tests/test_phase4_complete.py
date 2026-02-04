#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 完整功能测试
==================

目的：验证Phase 4三大核心模块功能
测试内容：
1. 元学习引擎（学会学习）
2. 自我改进引擎（代码生成、优化）
3. 递归自指优化（元认知）
4. 综合集成测试

版本: 1.0.0
"""

import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入模块
from core.meta_learning import MetaLearner, Task
from core.self_improvement import SelfImprovementEngine
from core.recursive_self_reference import RecursiveSelfReferenceEngine


def test_meta_learning():
    """测试元学习引擎"""
    print("\n" + "=" * 70)
    print("[测试1] 元学习引擎")
    print("=" * 70)

    # 创建元学习器
    meta_learner = MetaLearner(memory_size=50)
    print("  [OK] Meta-learner created")

    # 测试1.1: 学习多个任务
    print("\n  [1.1] 元学习 - 学习多个任务")
    simple_loss = lambda params: 1.0 - params.get('score', 0.5)

    for i in range(3):
        task = Task(
            task_id=f"task_{i}",
            name=f"Optimization_Task_{i}",
            data=[f"sample_{j}" for j in range(10)],
            loss_function=simple_loss,
            metadata={'type': 'optimization', 'domain': 'machine_learning'}
        )
        result = meta_learner.learn_task(task, max_iterations=20)
        print(f"    Task {i}: performance={result['final_performance']:.3f}, iterations={result['iterations']}")

    # 测试1.2: 快速适应
    print("\n  [1.2] 快速适应（少样本学习）")
    new_task = Task(
        task_id="new_task",
        name="New_Task",
        data=[],
        loss_function=simple_loss,
        metadata={'type': 'optimization', 'domain': 'machine_learning'}
    )
    support_examples = [f"sample_{i}" for i in range(3)]
    adaptation = meta_learner.adapt_to_new_task(new_task, support_examples)
    print(f"    Adaptation performance: {adaptation['performance']:.3f}")
    print(f"    Method: {adaptation['method']}")

    # 测试1.3: 提取元知识
    print("\n  [1.3] 提取元知识")
    knowledge = meta_learner.extract_meta_knowledge('machine_learning')
    print(f"    Domain: {knowledge.domain}")
    print(f"    Patterns: {len(knowledge.patterns)}")
    print(f"    Transferability: {knowledge.transferability_score:.2f}")

    # 统计
    stats = meta_learner.get_statistics()
    print(f"\n  [摘要]")
    print(f"    Total tasks: {stats['total_tasks_learned']}")
    print(f"    Adaptations: {stats['total_adaptations']}")
    print(f"    Experience count: {stats['experience_count']}")

    print("\n  [OK] Meta-learner test passed")
    return meta_learner


def test_self_improvement():
    """测试自我改进引擎"""
    print("\n" + "=" * 70)
    print("[测试2] 自我改进引擎")
    print("=" * 70)

    # 创建自我改进引擎
    project_root = Path(__file__).parent.parent
    engine = SelfImprovementEngine(str(project_root))
    print("  [OK] Self-improvement engine created")

    # 测试2.1: 扫描项目
    print("\n  [2.1] 扫描项目代码")
    stats = engine.get_statistics()
    print(f"    Modules scanned: {stats['modules_scanned']}")
    print(f"    Total LOC: {stats['total_lines_of_code']}")
    print(f"    Total complexity: {stats['total_complexity']:.1f}")

    # 测试2.2: 生成改进提案
    print("\n  [2.2] 生成改进提案")
    if engine.code_modules:
        module_path = list(engine.code_modules.keys())[0]
        proposals = engine.analyze_and_propose(module_path)
        print(f"    Generated {len(proposals)} proposals")

        for i, prop in enumerate(proposals[:2], 1):
            print(f"      [{i}] {prop.improvement_type.value}")
            print(f"          Description: {prop.description}")
            print(f"          Expected benefit: {prop.expected_benefit:.2%}")

    # 测试2.3: 生成自我改进代码
    print("\n  [2.3] 生成自我改进代码")
    task = "优化数据处理流程"
    generated_code = engine.generate_self_improvement_code(task)
    print(f"    Task: {task}")
    print(f"    Generated code length: {len(generated_code)} chars")
    print(f"    Preview: {generated_code[:200]}...")

    print("\n  [OK] Self-improvement engine test passed")
    return engine


def test_recursive_self_reference():
    """测试递归自指引擎"""
    print("\n" + "=" * 70)
    print("[测试3] 递归自指优化引擎")
    print("=" * 70)

    # 创建递归自指引擎
    engine = RecursiveSelfReferenceEngine(max_recursion_depth=2)
    print("  [OK] Recursive self-reference engine created")

    # 测试3.1: 监控思考过程
    print("\n  [3.1] 监控思考过程")
    thought = engine.monitor_thought(
        input_stimulus="如何提升系统性能",
        reasoning_steps=[
            "分析瓶颈",
            "设计优化方案",
            "实施改进"
        ],
        conclusion="需要重构核心模块",
        confidence=0.8
    )
    print(f"    Meta-commentary: {thought.meta_commentary}")

    # 测试3.2: 自我评估
    print("\n  [3.2] 自我评估")
    evaluation = engine.evaluate_self()

    # 测试3.3: 自我反思
    print("\n  [3.3] 自我反思")
    reflection = engine.reflect_on(
        topic="learning_efficiency",
        context={'complexity': 'medium', 'time_pressure': True}
    )
    print(f"    Observations: {len(reflection.observations)}")
    print(f"    Insights: {len(reflection.insights)}")
    print(f"    Action items: {len(reflection.action_items)}")

    # 测试3.4: 递归改进
    print("\n  [3.4] 递归改进")
    improvement = engine.recursive_improve()
    print(f"    Depth: {improvement['depth']}")
    print(f"    Performance trend: {improvement['evaluation']['performance_trend']}")

    # 统计
    stats = engine.get_statistics()
    print(f"\n  [摘要]")
    print(f"    Thoughts monitored: {stats['total_thoughts_monitored']}")
    print(f"    Reflections: {stats['total_reflections']}")
    print(f"    Self-awareness: {stats['self_awareness']:.3f}")
    print(f"    Meta-cognitive cycles: {stats['meta_cognitive_cycles']}")

    print("\n  [OK] Recursive self-reference test passed")
    return engine


def test_integration():
    """综合集成测试"""
    print("\n" + "=" * 70)
    print("[测试4] 综合集成测试")
    print("=" * 70)

    # 创建所有模块
    meta_learner = MetaLearner(memory_size=30)
    self_improvement = SelfImprovementEngine(str(Path(__file__).parent.parent))
    recursive_engine = RecursiveSelfReferenceEngine(max_recursion_depth=2)

    print("  [OK] All Phase 4 modules created")

    # 模拟完整的智能增强循环
    print("\n  [4.1] 学习阶段 - 元学习")
    task = Task(
        task_id="test_task",
        name="Test_Task",
        data=[],
        loss_function=lambda x: 1.0,
        metadata={'type': 'test', 'domain': 'test_domain'}
    )
    meta_learner.learn_task(task, max_iterations=10)

    print("\n  [4.2] 反思阶段 - 元认知")
    recursive_engine.reflect_on(
        topic="test_performance",
        context={'meta_learner_performance': 0.7}
    )

    print("\n  [4.3] 改进阶段 - 自我改进")
    improvement_code = self_improvement.generate_self_improvement_code("增强推理能力")
    print(f"    Generated improvement code: {len(improvement_code)} chars")

    print("\n  [4.4] 递归优化阶段")
    recursive_engine.recursive_improve()

    print("\n  [OK] Integration test passed")


def main():
    """主函数"""
    print("=" * 70)
    print("Phase 4 完整功能测试")
    print("=" * 70)

    try:
        # 测试1: 元学习
        meta_learner = test_meta_learning()

        # 测试2: 自我改进
        self_improvement = test_self_improvement()

        # 测试3: 递归自指
        recursive_engine = test_recursive_self_reference()

        # 测试4: 综合集成
        test_integration()

        # 总结
        print("\n" + "=" * 70)
        print("[测试总结]")
        print("=" * 70)
        print("  [OK] 所有测试通过")
        print("  [SUCCESS] Phase 4 三大核心模块验证完成")
        print("\n关键成果:")
        print("  1. 元学习引擎 - 学会学习、快速适应")
        print("  2. 自我改进引擎 - 代码生成、自动优化")
        print("  3. 递归自指优化 - 元认知、自我修正")
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
