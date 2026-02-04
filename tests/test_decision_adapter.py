#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVP单元测试
===========

测试DecisionAdapter的各个组件：
- encode_state
- decode_decision
- decide
- 边界条件

版本: 1.0.0
日期: 2026-01-14
作者: Claude Code (Sonnet 4.5)
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import time
from decision_adapter import DecisionAdapter, DecisionResult
from baseline_decision_system import BaselineDecisionSystem
from mvp_utils import generate_test_scenario, calculate_decision_quality


class TestDecisionAdapter:
    """DecisionAdapter单元测试"""

    @pytest.fixture
    def adapter(self):
        """创建DecisionAdapter实例"""
        return DecisionAdapter(
            state_dim=64,
            action_dim=4,
            enable_nonlinear=True,
            enable_meta_learning=True,
            enable_dialogue=False
        )

    def test_init(self, adapter):
        """测试1: 初始化"""
        assert adapter.helix_engine is not None
        assert adapter.state_dim == 64
        assert adapter.action_dim == 4
        print("[OK] test_init passed")

    def test_encode_intent(self, adapter):
        """测试2: 意图编码"""
        intent = {
            'type': 'visual_analysis',
            'confidence': 0.8,
            'depth': 'deep'
        }
        encoded = adapter._encode_intent(intent)

        assert encoded.shape == (16,)
        assert encoded.dtype == np.float32
        assert encoded[0] == 1.0  # visual_analysis one-hot
        assert encoded[6] == 0.8  # confidence
        assert encoded[7] == 1.0  # deep

        print("[OK] test_encode_intent passed")

    def test_encode_world_state(self, adapter):
        """测试3: 世界状态编码"""
        world_state = {
            'visual_objects': [1, 2, 3, 4, 5],
            'actions': [{'type': 'move'}, {'type': 'click'}],
            'timestamp': 3600.0  # 1小时
        }
        encoded = adapter._encode_world_state(world_state)

        assert encoded.shape == (16,)
        assert encoded.dtype == np.float32
        assert encoded[0] == 0.5  # 5个对象 / 10 = 0.5
        assert encoded[1] == 0.2  # 2个动作 / 10 = 0.2
        # 时间戳编码应该在0-1范围内
        assert 0.0 <= encoded[4] <= 1.0

        print("[OK] test_encode_world_state passed")

    def test_encode_prediction(self, adapter):
        """测试4: 预测编码"""
        prediction = {
            'confidence': 0.75,
            'next_actions': [{'type': 'A'}, {'type': 'B'}, {'type': 'C'}],
            'probabilities': [0.6, 0.3, 0.1]
        }
        encoded = adapter._encode_prediction(prediction)

        assert encoded.shape == (16,)
        assert encoded.dtype == np.float32
        assert encoded[0] == 0.3  # 3个动作 / 10
        assert encoded[1] == 0.75  # confidence
        assert encoded[3] == 0.6  # 第一个概率
        assert encoded[4] == 0.3  # 第二个概率

        print("[OK] test_encode_prediction passed")

    def test_encode_creativity(self, adapter):
        """测试5: 创造性编码"""
        creativity = {
            'insights': ['insight1', 'insight2', 'insight3'],
            'novelty_score': 0.85,
            'cross_domain_connections': [{'A': 'B'}]
        }
        encoded = adapter._encode_creativity(creativity)

        assert encoded.shape == (16,)
        assert encoded.dtype == np.float32
        assert encoded[0] == 0.3  # 3个洞察 / 10
        assert encoded[1] == 0.85  # novelty_score
        assert encoded[2] == 0.1  # 1个连接 / 10

        print("[OK] test_encode_creativity passed")

    def test_encode_state(self, adapter):
        """测试6: 完整state编码"""
        context = {
            'intent': {'type': 'navigation', 'confidence': 0.7},
            'world_state': {'visual_objects': [1, 2]},
            'prediction': {'confidence': 0.6},
            'creativity': {'novelty_score': 0.5}
        }
        state = adapter.encode_state(context)

        assert state.shape == (64,)
        assert state.dtype == np.float32

        # 检查各部分都在合理范围内
        assert np.all(state >= 0.0)
        assert np.all(state <= 1.0)

        print("[OK] test_encode_state passed")

    def test_decode_decision(self, adapter):
        """测试7: 决策解码"""
        from core.double_helix_engine_v2 import DoubleHelixResult

        # 创建一个模拟的DoubleHelixResult
        helix_result = DoubleHelixResult(
            action=2,
            confidence=0.85,
            weight_A=0.4,
            weight_B=0.6,
            phase=0.5,
            fusion_method='creative',
            emergence_score=0.7,
            explanation='Test reasoning',
            response_time_ms=15.0,
            dialogue_length=3,
            consensus_quality=0.8
        )

        decision = adapter.decode_decision(helix_result)

        assert isinstance(decision, DecisionResult)
        assert decision.action == 2
        assert decision.confidence == 0.85
        assert decision.fusion_method == 'creative'
        assert decision.emergence_score == 0.7
        assert decision.is_creative == True
        assert decision.is_emergent == True

        print("[OK] test_decode_decision passed")

    def test_decide(self, adapter):
        """测试8: 完整决策流程"""
        context = {
            'intent': {'type': 'question', 'confidence': 0.6},
            'world_state': {},
            'prediction': {},
            'creativity': {}
        }

        decision = adapter.decide(context)

        assert isinstance(decision, DecisionResult)
        assert 0 <= decision.action < 5  # action应该在0-4之间
        assert 0.0 <= decision.confidence <= 1.0
        assert len(decision.reasoning) > 0
        assert decision.response_time_ms > 0

        print("[OK] test_decide passed")

    def test_empty_context(self, adapter):
        """测试9: 空上下文（边界条件）"""
        context = {}

        decision = adapter.decide(context)

        assert decision is not None
        assert 0 <= decision.action < 5
        # 空上下文应该也能返回有效决策

        print("[OK] test_empty_context passed")

    def test_malformed_context(self, adapter):
        """测试10: 畸形上下文（边界条件）"""
        context = {
            'unknown_field': 123,
            'intent': 'invalid_type',  # 错误的类型
            'world_state': None
        }

        decision = adapter.decide(context)

        assert decision is not None
        assert 0 <= decision.action < 5
        # 畸形上下文应该也能返回有效决策

        print("[OK] test_malformed_context passed")


class TestEncodingQuality:
    """编码质量验证测试"""

    @pytest.fixture
    def adapter(self):
        return DecisionAdapter(state_dim=64, action_dim=4)

    def test_encoding_distinguishability(self, adapter):
        """测试11: 验证编码能否区分不同上下文"""
        contexts = [
            generate_test_scenario(0),
            generate_test_scenario(1),
            generate_test_scenario(2),
            generate_test_scenario(3),
            generate_test_scenario(4)
        ]

        states = [adapter.encode_state(ctx) for ctx in contexts]

        # 检查所有state都是不同的
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                # 确保不同上下文编码后的state是不同的
                assert not np.allclose(states[i], states[j]), \
                    f"Context {i} and {j} encoded to the same state!"

        print("[OK] test_encoding_distinguishability passed")

    def test_encoding_information_loss(self, adapter):
        """测试12: 验证编码没有严重信息损失"""
        context = generate_test_scenario(42)

        # 记录原始信息
        original_intent_type = context['intent']['type']
        original_confidence = context['intent']['confidence']
        original_objects = len(context['world_state']['visual_objects'])

        # 编码
        state = adapter.encode_state(context)

        # 验证关键信息保留（通过解码state的特征）
        # 意图类型应该在state的前6维中有体现
        intent_part = state[0:6]
        assert np.max(intent_part) > 0, "意图类型信息丢失"

        # 置信度应该在第6维
        assert state[6] == original_confidence, "置信度信息丢失"

        # 对象数量应该在第16维
        assert state[16] > 0, "世界状态信息丢失"

        print("[OK] test_encoding_information_loss passed")


class TestBaselineComparison:
    """Baseline对比测试（Day 8-9准备）"""

    @pytest.fixture
    def systems(self):
        baseline = BaselineDecisionSystem(action_dim=4)
        adapter = DecisionAdapter(state_dim=64, action_dim=4)
        return baseline, adapter

    def test_both_systems_work(self, systems):
        """测试13: 确保两个系统都能工作"""
        baseline, adapter = systems

        context = generate_test_scenario(42)

        # Baseline决策
        baseline_result = baseline.decide(context)
        assert 0 <= baseline_result.action < 4
        assert 0.0 <= baseline_result.confidence <= 1.0

        # Adapter决策
        adapter_result = adapter.decide(context)
        assert 0 <= adapter_result.action < 4
        assert 0.0 <= adapter_result.confidence <= 1.0

        print("[OK] test_both_systems_work passed")

    def test_decision_quality_metrics(self, systems):
        """测试14: 决策质量指标计算"""
        baseline, adapter = systems

        # 生成多个决策
        num_decisions = 10
        baseline_decisions = []
        adapter_decisions = []

        for i in range(num_decisions):
            context = generate_test_scenario(i)

            baseline_result = baseline.decide(context)
            baseline_decisions.append({
                'action': baseline_result.action,
                'confidence': baseline_result.confidence,
                'is_creative': False,
                'emergence_score': 0.0
            })

            adapter_result = adapter.decide(context)
            adapter_decisions.append({
                'action': adapter_result.action,
                'confidence': adapter_result.confidence,
                'is_creative': adapter_result.is_creative,
                'emergence_score': adapter_result.emergence_score
            })

        # 计算质量指标
        baseline_quality = calculate_decision_quality(baseline_decisions)
        adapter_quality = calculate_decision_quality(adapter_decisions)

        assert baseline_quality['overall'] > 0
        assert adapter_quality['overall'] > 0

        # 打印比较
        print(f"\n[对比测试] {num_decisions}次决策")
        print(f"  Baseline: overall={baseline_quality['overall']:.3f}, "
              f"conf={baseline_quality['confidence']:.3f}")
        print(f"  Adapter:  overall={adapter_quality['overall']:.3f}, "
              f"conf={adapter_quality['confidence']:.3f}, "
              f"creative={adapter_quality['creative_rate']:.1%}")

        print("[OK] test_decision_quality_metrics passed")


# 运行所有测试
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*20 + "MVP单元测试")
    print("="*70)

    # 创建adapter实例
    adapter = DecisionAdapter()

    print("\n[测试组1] DecisionAdapter核心功能")
    print("-" * 70)

    try:
        test = TestDecisionAdapter()

        print("测试1: 初始化...")
        test.test_init(adapter)

        print("测试2: 意图编码...")
        test.test_encode_intent(adapter)

        print("测试3: 世界状态编码...")
        test.test_encode_world_state(adapter)

        print("测试4: 预测编码...")
        test.test_encode_prediction(adapter)

        print("测试5: 创造性编码...")
        test.test_encode_creativity(adapter)

        print("测试6: 完整state编码...")
        test.test_encode_state(adapter)

        print("测试7: 决策解码...")
        test.test_decode_decision(adapter)

        print("测试8: 完整决策流程...")
        test.test_decide(adapter)

        print("测试9: 空上下文...")
        test.test_empty_context(adapter)

        print("测试10: 畸形上下文...")
        test.test_malformed_context(adapter)

    except Exception as e:
        print(f"\n[ERROR] DecisionAdapter测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n[测试组2] 编码质量验证")
    print("-" * 70)

    try:
        test_quality = TestEncodingQuality()

        print("测试11: 编码区分性...")
        test_quality.test_encoding_distinguishability(adapter)

        print("测试12: 编码信息保留...")
        test_quality.test_encoding_information_loss(adapter)

    except Exception as e:
        print(f"\n[ERROR] 编码质量测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n[测试组3] Baseline对比")
    print("-" * 70)

    try:
        test_comparison = TestBaselineComparison()

        baseline = BaselineDecisionSystem(action_dim=4)
        adapter = DecisionAdapter(state_dim=64, action_dim=4)

        print("测试13: 两个系统都能工作...")
        test_comparison.test_both_systems_work((baseline, adapter))

        print("测试14: 决策质量指标...")
        test_comparison.test_decision_quality_metrics((baseline, adapter))

    except Exception as e:
        print(f"\n[ERROR] Baseline对比测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("[SUCCESS] 所有单元测试完成！")
    print("="*70)
