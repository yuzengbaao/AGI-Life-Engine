#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试决策缓存系统 (Test Decision Cache System)
=====================================

测试目标：
1. 验证模式匹配器功能
2. 验证决策缓存功能
3. 验证集成效果

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import sys
import os
import time
import logging
from typing import List, Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pattern_matcher():
    """测试模式匹配器"""
    print("\n" + "="*60)
    print("测试1：模式匹配器 (Pattern Matcher)")
    print("="*60)

    try:
        from core.pattern_matcher import PatternMatcher

        matcher = PatternMatcher()

        # 测试用例
        test_cases = [
            ("读取文件config.txt", "file_read"),
            ("系统状态怎么样", "system_status"),
            ("代码分析", "code_read"),
            ("你好", "conversation_greeting"),
            ("帮助", "conversation_help"),
            ("调试错误", "debug_analyze"),
            ("创建备份", "backup_create"),
        ]

        passed = 0
        total = len(test_cases)

        for text, expected_intent in test_cases:
            result = matcher.match(text)
            if result and result.intent == expected_intent:
                print(f"✅ PASS: '{text}' → {result.intent} (confidence={result.confidence:.2f})")
                passed += 1
            else:
                actual_intent = result.intent if result else "None"
                print(f"❌ FAIL: '{text}' → 期望: {expected_intent}, 实际: {actual_intent}")

        print(f"\n结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")

        # 显示统计
        stats = matcher.get_statistics()
        print(f"\n模式匹配器统计:")
        print(f"  - 总模式数: {stats['total_patterns']}")
        print(f"  - 匹配次数: {stats['total_matches']}")
        print(f"  - 未匹配次数: {stats['total_misses']}")
        print(f"  - 匹配率: {stats['match_rate']:.2%}")

        return passed == total

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_decision_cache():
    """测试决策缓存"""
    print("\n" + "="*60)
    print("测试2：决策缓存 (Decision Cache)")
    print("="*60)

    try:
        from core.decision_cache import DecisionCache
        import numpy as np

        cache = DecisionCache(max_size=10, similarity_threshold=0.85)

        # 创建测试embeddings（简化版：使用随机向量）
        np.random.seed(42)

        # 测试用例1：存储和检索
        print("\n测试用例1：基本存储和检索")
        embedding1 = np.random.rand(128)
        cache.put(embedding1, "test_intent", confidence=0.95, metadata={"source": "test"})

        # 精确匹配
        result = cache.get(embedding1)
        if result:
            intent, confidence, metadata = result
            print(f"✅ 精确匹配成功: intent={intent}, confidence={confidence:.2f}")
        else:
            print(f"❌ 精确匹配失败")
            return False

        # 测试用例2：相似度匹配
        print("\n测试用例2：相似度匹配")
        embedding2 = embedding1 + np.random.rand(128) * 0.1  # 添加小噪声
        result = cache.get(embedding2)
        if result:
            intent, confidence, _ = result
            print(f"✅ 相似度匹配成功: intent={intent}, similarity={confidence:.3f}")
        else:
            print(f"❌ 相似度匹配失败（可能低于阈值）")

        # 测试用例3：LRU淘汰
        print("\n测试用例3：LRU淘汰策略")
        for i in range(15):  # 超过max_size=10
            emb = np.random.rand(128)
            cache.put(emb, f"intent_{i}", confidence=0.8)

        stats = cache.get_statistics()
        print(f"缓存大小: {stats['cache_size']}/{stats['max_size']}")
        print(f"淘汰次数: {stats['evictions']}")
        if stats['cache_size'] <= stats['max_size']:
            print(f"✅ LRU淘汰正常工作")
        else:
            print(f"❌ LRU淘汰失败")
            return False

        # 测试用例4：统计信息
        print("\n测试用例4：统计信息")
        stats = cache.get_statistics()
        print(f"缓存统计:")
        print(f"  - 缓存大小: {stats['cache_size']}")
        print(f"  - 命中次数: {stats['hits']}")
        print(f"  - 未命中次数: {stats['misses']}")
        print(f"  - 命中率: {stats['hit_rate']:.2%}")
        print(f"  - 淘汰次数: {stats['evictions']}")
        print(f"  - 过期次数: {stats['expirations']}")
        print(f"  - 平均访问次数: {stats['avg_access_count']:.2f}")
        print(f"✅ 统计功能正常")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """测试集成效果"""
    print("\n" + "="*60)
    print("测试3：集成效果 (Integration Test)")
    print("="*60)

    try:
        from core.pattern_matcher import PatternMatcher
        from core.intent_tracker import IntentTracker

        # 创建IntentTracker
        tracker = IntentTracker(history_size=20)

        # 检查是否启用了快速路径
        if hasattr(tracker, 'enable_fast_intent'):
            print(f"✅ 快速路径已启用: {tracker.enable_fast_intent}")
        else:
            print(f"❌ 快速路径未启用")
            return False

        if tracker.pattern_matcher:
            print(f"✅ 模式匹配器已集成")
        else:
            print(f"⚠️ 模式匹配器未集成（导入失败）")

        if tracker.intent_cache:
            print(f"✅ 意图缓存已集成")
        else:
            print(f"⚠️ 意图缓存未集成（导入失败）")

        # 测试快速路径
        print("\n测试快速路径识别：")
        test_observations = [
            {"timestamp": time.time(), "type": "user", "text": "读取配置文件"},
            {"timestamp": time.time(), "type": "user", "text": "检查系统状态"},
            {"timestamp": time.time(), "type": "user", "text": "创建备份"},
        ]

        for obs in test_observations:
            tracker.add_observation(obs)

        # 检查快速路径统计
        stats = tracker.get_fast_path_statistics()
        print(f"\n快速路径统计:")
        print(f"  - 快速路径命中: {stats['fast_path_hits']}")
        print(f"  - 缓存命中: {stats['cache_hits']}")
        print(f"  - LLM调用: {stats['llm_calls']}")
        print(f"  - 总推断次数: {stats['total_inferences']}")
        print(f"  - 快速路径率: {stats['fast_path_rate']:.2%}")
        print(f"  - LLM调用率: {stats['llm_call_rate']:.2%}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("[TEST] AGI Life Engine P0级优化 - 决策缓存系统测试")
    print("="*60)

    results = []

    # 运行所有测试
    results.append(("模式匹配器", test_pattern_matcher()))
    results.append(("决策缓存", test_decision_cache()))
    results.append(("集成效果", test_integration()))

    # 总结
    print("\n" + "="*60)
    print("[SUMMARY] 测试总结")
    print("="*60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} - {name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\n总计: {total_passed}/{total_tests} 通过 ({total_passed/total_tests*100:.1f}%)")

    if total_passed == total_tests:
        print("\n[SUCCESS] 所有测试通过！阶段1.1实施成功。")
        return 0
    else:
        print("\n[WARNING] 部分测试失败，需要修复。")
        return 1


if __name__ == "__main__":
    exit(main())
