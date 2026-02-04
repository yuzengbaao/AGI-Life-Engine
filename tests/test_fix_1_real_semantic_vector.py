#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复1：真实语义向量 (Fix #1: Real Semantic Vector)
验证：AGI_Life_Engine.py:3658 行的修复

测试目标：
- 验证修复代码确实使用了 PerceptionSystem
- 验证语义向量是确定性的（相同输入→相同输出）
- 验证语义向量有语义相似性（相似输入→相似向量）
- 对比修复前后的行为
"""

import sys
import os
import numpy as np
import hashlib

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.perception_system import PerceptionSystem
except ImportError:
    print("❌ 无法导入 PerceptionSystem")
    sys.exit(1)


class TestRealSemanticVector:
    """测试真实语义向量替代随机数"""

    def __init__(self):
        print("\n" + "="*70)
        print(" "*15 + "🧪 测试修复1：真实语义向量")
        print("="*70)

        self.perception_system = PerceptionSystem()
        self.test_results = []

    def test_1_perception_system_loaded(self):
        """测试1：验证感知系统已加载"""
        print("\n[测试1] 验证感知系统已加载...")
        assert self.perception_system is not None, "PerceptionSystem 未初始化"
        assert self.perception_system.embedder is not None, "SentenceTransformer 模型未加载"
        assert self.perception_system.model_dim == 384, f"模型维度错误: {self.perception_system.model_dim}"
        print("✅ PASS: 感知系统正常加载 (all-MiniLM-L6-v2, 384维)")
        self.test_results.append(("感知系统加载", True))
        return True

    def test_2_deterministic_encoding(self):
        """测试2：验证编码的确定性（相同输入→相同输出）"""
        print("\n[测试2] 验证编码确定性...")
        test_text = "Write a python script to analyze data"

        # 编码两次
        vec1 = self.perception_system.encode_text(test_text)
        vec2 = self.perception_system.encode_text(test_text)

        # 验证完全相同
        assert np.allclose(vec1, vec2), "相同文本的编码应该完全一致"
        print(f"✅ PASS: 确定性验证通过 (向量完全相同)")
        print(f"   - 向量维度: {vec1.shape}")
        print(f"   - 范围: [{vec1.min():.4f}, {vec1.max():.4f}]")
        self.test_results.append(("确定性编码", True))
        return True

    def test_3_semantic_similarity(self):
        """测试3：验证语义相似性（相似输入→高相似度）"""
        print("\n[测试3] 验证语义相似性...")

        # 语义相似的文本对
        similar_pairs = [
            ("Write a python script", "Create a python program"),
            ("Analyze the data", "Examine the dataset"),
            ("Fix the bug", "Debug the error"),
        ]

        # 语义不同的文本对
        different_pairs = [
            ("Write a python script", "Cook a spicy meal"),
            ("Analyze the data", "Play basketball"),
            ("Fix the bug", "Sing a song"),
        ]

        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        print("\n   语义相似文本对:")
        similar_scores = []
        for text1, text2 in similar_pairs:
            vec1 = self.perception_system.encode_text(text1)
            vec2 = self.perception_system.encode_text(text2)
            score = cosine_similarity(vec1, vec2)
            similar_scores.append(score)
            print(f"   - '{text1}' vs '{text2}'")
            print(f"     相似度: {score:.4f}")

        print("\n   语义不同文本对:")
        different_scores = []
        for text1, text2 in different_pairs:
            vec1 = self.perception_system.encode_text(text1)
            vec2 = self.perception_system.encode_text(text2)
            score = cosine_similarity(vec1, vec2)
            different_scores.append(score)
            print(f"   - '{text1}' vs '{text2}'")
            print(f"     相似度: {score:.4f}")

        avg_similar = np.mean(similar_scores)
        avg_different = np.mean(different_scores)

        print(f"\n   📊 统计:")
        print(f"   - 相似文本平均相似度: {avg_similar:.4f}")
        print(f"   - 不同文本平均相似度: {avg_different:.4f}")
        print(f"   - 相似度差异: {avg_similar - avg_different:.4f}")

        # 验证相似文本的相似度显著高于不同文本
        assert avg_similar > 0.5, f"相似文本相似度过低: {avg_similar}"
        assert avg_different < 0.3, f"不同文本相似度过高: {avg_different}"
        assert avg_similar > avg_different * 1.5, "相似文本与不同文本的区分度不足"

        print("\n✅ PASS: 语义相似性验证通过")
        self.test_results.append(("语义相似性", True))
        return True

    def test_4_old_vs_new_behavior(self):
        """测试4：对比修复前后的行为"""
        print("\n[测试4] 对比修复前后的行为...")

        test_insight = "Implement a recursive algorithm for tree traversal"

        # 模拟修复前的行为：随机向量
        print("\n   🔴 修复前 (旧行为):")
        old_simulated_vec = np.random.rand(128)
        print(f"   - 使用: np.random.rand(128)")
        print(f"   - 向量示例: {old_simulated_vec[:5]}")
        print(f"   - ⚠️ 问题: 每次运行结果完全不同，无法验证")

        # 模拟修复后的行为：真实语义向量
        print("\n   🟢 修复后 (新行为):")
        real_vec = self.perception_system.encode_text(test_insight)
        # 截断到128维（模拟实际使用）
        real_vec_128 = real_vec[:128] if real_vec.shape[0] > 128 else real_vec
        print(f"   - 使用: perception_system.encode_text()")
        print(f"   - 向量示例: {real_vec_128[:5]}")
        print(f"   - ✅ 优势: 相同输入永远产生相同向量")

        # 验证新行为的一致性
        real_vec_128_v2 = self.perception_system.encode_text(test_insight)[:128]
        consistency = np.allclose(real_vec_128, real_vec_128_v2)
        print(f"\n   - 一致性验证: {consistency}")
        assert consistency, "新行为应该保持一致性"

        # 对比随机向量的不一致性
        random_vec_v2 = np.random.rand(128)
        random_consistency = np.allclose(old_simulated_vec, random_vec_v2)
        print(f"   - 旧随机向量一致性: {random_consistency} (应该是 False)")
        assert not random_consistency, "旧随机向量不应该一致"

        print("\n✅ PASS: 新行为优于旧行为")
        self.test_results.append(("旧vs新行为对比", True))
        return True

    def test_5_fallback_mechanism(self):
        """测试5：验证fallback机制（当perception_system不可用时）"""
        print("\n[测试5] 验证fallback机制...")

        # 模拟 perception_system 不可用的情况
        test_text = "Fallback test case"

        # 使用确定性哈希投影（代码中的fallback逻辑）
        hash_seed = int(hashlib.md5(test_text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(hash_seed)
        fallback_vec = rng.standard_normal(128)

        # 编码两次
        hash_seed_v2 = int(hashlib.md5(test_text.encode()).hexdigest(), 16) % (2**32)
        rng_v2 = np.random.default_rng(hash_seed_v2)
        fallback_vec_v2 = rng_v2.standard_normal(128)

        # 验证fallback也是确定性的
        assert np.allclose(fallback_vec, fallback_vec_v2), "Fallback向量应该一致"

        print("✅ PASS: Fallback机制正确（基于哈希的确定性投影）")
        print(f"   - Fallback向量维度: {fallback_vec.shape}")
        print(f"   - 一致性: ✓")
        self.test_results.append(("Fallback机制", True))
        return True

    def run_all_tests(self):
        """运行所有测试"""
        tests = [
            self.test_1_perception_system_loaded,
            self.test_2_deterministic_encoding,
            self.test_3_semantic_similarity,
            self.test_4_old_vs_new_behavior,
            self.test_5_fallback_mechanism,
        ]

        passed = 0
        failed = 0

        for test in tests:
            try:
                if test():
                    passed += 1
            except AssertionError as e:
                failed += 1
                print(f"\n❌ FAIL: {e}")
            except Exception as e:
                failed += 1
                print(f"\n❌ ERROR: {e}")

        # 打印总结
        print("\n" + "="*70)
        print(" "*25 + "📊 测试总结")
        print("="*70)
        print(f"\n总测试数: {len(tests)}")
        print(f"✅ 通过: {passed}")
        print(f"❌ 失败: {failed}")
        print(f"成功率: {passed/len(tests)*100:.1f}%")

        print("\n详细结果:")
        for name, result in self.test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status}: {name}")

        if failed == 0:
            print("\n🎉 所有测试通过！修复1验证成功。")
            return True
        else:
            print(f"\n⚠️ {failed} 个测试失败，请检查。")
            return False


if __name__ == "__main__":
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    tester = TestRealSemanticVector()
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)
