#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
False Positive修复验证测试
===========================

测试元认知层是否正确识别系统内部任务，避免false positive

Version: 1.0.0
Date: 2026-01-16
"""

import sys
import os
import io

# 修复UTF-8输出问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.meta_cognitive import TaskUnderstandingEvaluator, CapabilityMatcher


def test_false_positive_fix():
    """测试false positive修复"""
    print("="*70)
    print("False Positive修复验证测试")
    print("="*70)

    evaluator = TaskUnderstandingEvaluator()
    matcher = CapabilityMatcher()

    # 测试用例
    test_cases = [
        {
            "name": "系统idle任务",
            "task": "Wait for Evolution Loop to generate new strategy (Resting)",
            "expected_feasible": True,  # 应该可行
            "expected_no_gaps": True,   # 不应该有知识缺口
        },
        {
            "name": "系统maintenance任务",
            "task": "Triggering evolution loop for self-improvement",
            "expected_feasible": True,
            "expected_no_gaps": True,
        },
        {
            "name": "真正的3D任务",
            "task": "Analyze 3D point cloud data and extract surface normals",
            "expected_feasible": False,
            "expected_no_gaps": False,
        },
        {
            "name": "真正的分子生物学任务",
            "task": "Analyze protein structure and predict molecular interactions",
            "expected_feasible": False,
            "expected_no_gaps": False,
        },
        {
            "name": "普通代码任务",
            "task": "Read Python file and refactor the code structure",
            "expected_feasible": True,
            "expected_no_gaps": True,
        },
    ]

    passed = 0
    failed = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"[测试 {i}/{len(test_cases)}] {test_case['name']}")
        print(f"{'='*70}")
        print(f"任务: {test_case['task']}")
        print()

        # 评估任务理解
        task_analysis = evaluator.evaluate(test_case['task'])

        # 评估能力匹配
        match_result = matcher.match(test_case['task'])

        # 验证结果
        test_passed = True

        # 检查可行性
        if task_analysis.can_solve != test_case['expected_feasible']:
            print(f"❌ 失败: 可行性判断错误")
            print(f"   期望: {test_case['expected_feasible']}, 实际: {task_analysis.can_solve}")
            test_passed = False
        else:
            print(f"✅ 可行性判断正确: {task_analysis.can_solve}")

        # 检查知识缺口
        has_gaps = len(task_analysis.knowledge_gaps) > 0
        if has_gaps != (not test_case['expected_no_gaps']):
            print(f"❌ 失败: 知识缺口判断错误")
            print(f"   期望是否有缺口: {not test_case['expected_no_gaps']}, 实际: {has_gaps}")
            if has_gaps:
                print(f"   缺口: {task_analysis.knowledge_gaps}")
            test_passed = False
        else:
            if test_case['expected_no_gaps']:
                print(f"✅ 正确识别无知识缺口")
            else:
                print(f"✅ 正确识别有知识缺口: {task_analysis.knowledge_gaps}")

        # 检查能力匹配
        if test_case['expected_feasible']:
            # 应该是好的匹配
            if match_result.match_level.value in ["none", "poor"]:
                print(f"❌ 失败: 能力匹配等级错误")
                print(f"   期望: good/perfect/partial, 实际: {match_result.match_level.value}")
                test_passed = False
            else:
                print(f"✅ 能力匹配正确: {match_result.match_level.value}")
        else:
            # 应该是差的匹配
            if match_result.match_level.value in ["perfect", "good"]:
                print(f"❌ 失败: 能力匹配等级错误")
                print(f"   期望: none/poor, 实际: {match_result.match_level.value}")
                test_passed = False
            else:
                print(f"✅ 能力匹配正确: {match_result.match_level.value}")

        if test_passed:
            passed += 1
            print(f"\n✅ 测试 {i} 通过")
        else:
            failed += 1
            print(f"\n❌ 测试 {i} 失败")

    # 总结
    print(f"\n{'='*70}")
    print(f"测试总结")
    print(f"{'='*70}")
    print(f"总测试数: {len(test_cases)}")
    print(f"通过: {passed} ✅")
    print(f"失败: {failed} ❌")
    print(f"通过率: {passed/len(test_cases)*100:.1f}%")

    if failed == 0:
        print(f"\n🎉 所有测试通过！False positive问题已修复！")
        return True
    else:
        print(f"\n⚠️  有{failed}个测试失败，需要进一步调试")
        return False


if __name__ == "__main__":
    success = test_false_positive_fix()
    sys.exit(0 if success else 1)
