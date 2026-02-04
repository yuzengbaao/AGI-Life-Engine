"""
🔧 [2026-01-11] 元认知调查空转循环修复验证测试

验证修复的四个层面:
1. WorkTemplates.meta_cognitive_investigation() 创建带有明确验证标准的目标
2. PlannerAgent._heuristic_plan() 生成证据驱动的调查步骤
3. CriticAgent.verify_outcome() 根据证据评分而非恒定1.0
4. AGI_Life_Engine 的冷却机制防止循环触发
"""

import asyncio
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.goal_system import WorkTemplates, GoalType, GoalStatus, GoalVerifier
from core.agents.planner import PlannerAgent
from core.agents.critic import CriticAgent
from core.llm_client import LLMService


def test_work_template_has_evidence_requirements():
    """测试1: WorkTemplates 生成的目标必须有明确的验证标准"""
    print("\n" + "="*60)
    print("测试1: WorkTemplates 证据要求")
    print("="*60)
    
    goal = WorkTemplates.meta_cognitive_investigation(entropy=0.8, curiosity=0.65)
    
    # 检查目标类型
    assert goal.goal_type == GoalType.ANALYSIS, f"Expected ANALYSIS, got {goal.goal_type}"
    print(f"✅ 目标类型: {goal.goal_type.value}")
    
    # 检查必须有 success_criteria
    assert goal.success_criteria, "目标必须有 success_criteria"
    print(f"✅ success_criteria 存在: {list(goal.success_criteria.keys())}")
    
    # 检查必须有 output_file 要求
    assert "output_file" in goal.success_criteria, "必须要求输出文件"
    print(f"✅ 要求输出文件: {goal.success_criteria['output_file']}")
    
    # 检查必须有 required_keywords
    assert "required_keywords" in goal.success_criteria, "必须要求关键词证据"
    keywords = goal.success_criteria["required_keywords"]
    assert len(keywords) >= 3, f"至少需要3个证据关键词, 实际: {len(keywords)}"
    print(f"✅ 证据关键词: {keywords}")
    
    # 检查 max_attempts = 1 (不重试)
    assert goal.max_attempts == 1, f"Expected max_attempts=1, got {goal.max_attempts}"
    print(f"✅ 最大尝试次数: {goal.max_attempts} (避免重试循环)")
    
    print("\n✅ 测试1通过: WorkTemplates 正确生成带证据要求的目标")
    return True


def test_planner_generates_evidence_steps():
    """测试2: Planner 为元认知调查生成证据驱动的步骤"""
    print("\n" + "="*60)
    print("测试2: Planner 证据驱动步骤")
    print("="*60)
    
    llm = LLMService()
    planner = PlannerAgent(llm)
    
    # 模拟元认知调查任务
    task = "[Meta] Investigate high entropy state (Entropy: 0.85, Curiosity: 0.72)"
    
    # 直接调用启发式计划 (不依赖 LLM)
    steps = planner._heuristic_plan(task)
    
    print(f"生成的步骤 ({len(steps)} 步):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step[:80]}...")
    
    # 验证步骤包含证据生成工具
    steps_str = str(steps).lower()
    
    evidence_tools = [
        "analyze_entropy_sources",
        "check_memory_drift", 
        "evaluate_uncertainty_distribution",
        "synthesize_investigation_report"
    ]
    
    found_tools = [t for t in evidence_tools if t in steps_str]
    print(f"\n找到的证据生成工具: {found_tools}")
    
    assert len(found_tools) >= 2, f"至少需要2个证据工具, 实际: {len(found_tools)}"
    print(f"✅ 包含 {len(found_tools)} 个证据生成工具")
    
    # 确保不是只有 log
    non_log_steps = [s for s in steps if '"tool": "log"' not in s]
    assert len(non_log_steps) >= 2, "必须有非log的实质步骤"
    print(f"✅ 包含 {len(non_log_steps)} 个非log步骤")
    
    print("\n✅ 测试2通过: Planner 生成证据驱动的调查步骤")
    return True


async def test_critic_evidence_based_scoring():
    """测试3: Critic 根据证据评分"""
    print("\n" + "="*60)
    print("测试3: Critic 证据评分")
    print("="*60)
    
    llm = LLMService()
    critic = CriticAgent(llm)
    
    # 场景A: 仅日志输出 (应该得低分)
    action_a = "[Meta] Investigate high entropy state"
    result_a = "Logged: Starting investigation..."
    score_a = await critic.verify_outcome(action_a, result_a)
    print(f"\n场景A - 仅日志:")
    print(f"  动作: {action_a}")
    print(f"  结果: {result_a}")
    print(f"  评分: {score_a}")
    assert score_a < 0.5, f"仅日志应得低分(<0.5), 实际: {score_a}"
    print(f"  ✅ 正确: 仅日志得低分 {score_a}")
    
    # 场景B: 有2个证据标记 (应该得0.7+)
    action_b = "[Meta] Investigate high entropy"
    result_b = "entropy_source detected | memory_drift analysis complete"
    score_b = await critic.verify_outcome(action_b, result_b)
    print(f"\n场景B - 2个证据:")
    print(f"  动作: {action_b}")
    print(f"  结果: {result_b}")
    print(f"  评分: {score_b}")
    assert score_b >= 0.7, f"2个证据应得0.7+, 实际: {score_b}"
    print(f"  ✅ 正确: 2个证据得分 {score_b}")
    
    # 场景C: 有4个证据标记 (应该得0.9+)
    action_c = "[Meta] Investigate entropy state"
    result_c = "entropy_source: high | memory_drift: 0.2 | uncertainty_analysis: complete | root_cause: identified"
    score_c = await critic.verify_outcome(action_c, result_c)
    print(f"\n场景C - 4个证据:")
    print(f"  动作: {action_c}")
    print(f"  结果: {result_c}")
    print(f"  评分: {score_c}")
    assert score_c >= 0.9, f"4个证据应得0.9+, 实际: {score_c}"
    print(f"  ✅ 正确: 4个证据得分 {score_c}")
    
    print("\n✅ 测试3通过: Critic 正确根据证据评分")
    return True


def test_cooldown_mechanism():
    """测试4: 冷却机制防止循环触发"""
    print("\n" + "="*60)
    print("测试4: 冷却机制")
    print("="*60)
    
    # 检查 AGI_Life_Engine 中是否有冷却相关属性
    import ast
    
    engine_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "AGI_Life_Engine.py")
    with open(engine_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查冷却机制属性
    cooldown_attrs = [
        "_last_meta_investigation_ts",
        "_meta_investigation_cooldown",
        "_curiosity_satisfaction_decay"
    ]
    
    found = []
    for attr in cooldown_attrs:
        if attr in content:
            found.append(attr)
            print(f"✅ 找到冷却属性: {attr}")
    
    assert len(found) == 3, f"需要3个冷却属性, 找到: {found}"
    
    # 检查冷却检查逻辑
    assert "meta_cooldown_remaining" in content, "需要冷却检查逻辑"
    print("✅ 找到冷却检查逻辑: meta_cooldown_remaining")
    
    # 检查好奇心衰减恢复逻辑
    assert "curiosity_satisfaction_decay - 0.05" in content or "_curiosity_satisfaction_decay - 0.05" in content, \
        "需要好奇心衰减恢复逻辑"
    print("✅ 找到好奇心衰减恢复逻辑")
    
    print("\n✅ 测试4通过: 冷却机制正确实现")
    return True


async def main():
    print("\n" + "="*70)
    print("🔧 元认知调查空转循环修复验证")
    print("="*70)
    
    results = []
    
    # 测试1: WorkTemplates
    try:
        results.append(("WorkTemplates证据要求", test_work_template_has_evidence_requirements()))
    except Exception as e:
        print(f"❌ 测试1失败: {e}")
        results.append(("WorkTemplates证据要求", False))
    
    # 测试2: Planner
    try:
        results.append(("Planner证据步骤", test_planner_generates_evidence_steps()))
    except Exception as e:
        print(f"❌ 测试2失败: {e}")
        results.append(("Planner证据步骤", False))
    
    # 测试3: Critic
    try:
        results.append(("Critic证据评分", await test_critic_evidence_based_scoring()))
    except Exception as e:
        print(f"❌ 测试3失败: {e}")
        results.append(("Critic证据评分", False))
    
    # 测试4: 冷却机制
    try:
        results.append(("冷却机制", test_cooldown_mechanism()))
    except Exception as e:
        print(f"❌ 测试4失败: {e}")
        results.append(("冷却机制", False))
    
    # 汇总
    print("\n" + "="*70)
    print("📊 测试汇总")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有修复验证通过！元认知调查空转循环问题已解决。")
        return 0
    else:
        print("\n⚠️ 部分测试失败，请检查修复代码。")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
