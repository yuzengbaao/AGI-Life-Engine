"""
MetaCognition Module Tests
元认知模块集成测试

测试目标:
1. MetaCognition 类初始化
2. observe() 方法正确记录思维帧
3. extend_thought_chain() 延长到20步
4. register_intention() 意图管理
5. 与 TheSeed 和 EvolutionController 的集成
"""

import pytest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.metacognition import MetaCognition, ThoughtFrame, Intention, MetaInsight, create_metacognition


class TestMetaCognitionBasic:
    """基础功能测试"""
    
    def test_initialization(self):
        """测试 MetaCognition 初始化"""
        mc = create_metacognition(horizon=20)
        
        assert mc is not None
        assert mc.current_horizon == 20
        assert mc.MIN_HORIZON == 15
        assert mc.MAX_HORIZON == 25
        assert len(mc.thought_frames) == 0
        assert len(mc.intentions) == 0
    
    def test_horizon_bounds(self):
        """测试 horizon 边界约束"""
        mc = MetaCognition()
        
        # 超出上界
        mc.set_horizon(30)
        assert mc.current_horizon == 25
        
        # 超出下界
        mc.set_horizon(10)
        assert mc.current_horizon == 15
        
        # 正常范围
        mc.set_horizon(20)
        assert mc.current_horizon == 20


class TestObserve:
    """observe() 方法测试"""
    
    def test_observe_single_frame(self):
        """测试单次观察"""
        mc = create_metacognition()
        
        state = np.random.randn(64).astype(np.float32)
        frame = mc.observe(
            state_vector=state,
            action_taken=0,
            action_name="explore",
            uncertainty=0.5,
            thought_chain=["Concept_ABC"],
            neural_confidence=0.7
        )
        
        assert isinstance(frame, ThoughtFrame)
        assert frame.tick_id == 1
        assert frame.action_name == "explore"
        assert frame.uncertainty == 0.5
        assert len(mc.thought_frames) == 1
    
    def test_observe_multiple_frames(self):
        """测试多次观察"""
        mc = create_metacognition()
        
        for i in range(10):
            state = np.random.randn(64).astype(np.float32)
            mc.observe(
                state_vector=state,
                action_taken=i % 4,
                action_name=f"action_{i % 4}",
                uncertainty=0.3 + 0.1 * i,
                thought_chain=[f"Thought_{i}"],
                neural_confidence=0.5
            )
        
        assert len(mc.thought_frames) == 10
        assert mc._tick_counter == 10


class TestIntentions:
    """意图管理测试"""
    
    def test_register_intention(self):
        """测试意图注册"""
        mc = create_metacognition()
        
        intention = mc.register_intention(
            description="探索未知领域",
            priority=0.8
        )
        
        assert isinstance(intention, Intention)
        assert intention.description == "探索未知领域"
        assert intention.priority == 0.8
        assert intention.status == "active"
        assert len(mc.intentions) == 1
    
    def test_update_intention_progress(self):
        """测试意图进度更新"""
        mc = create_metacognition()
        
        intention = mc.register_intention("测试意图", priority=0.5)
        
        mc.update_intention_progress(intention.id, 0.5)
        assert mc.intentions[intention.id].progress == 0.5
        
        mc.update_intention_progress(intention.id, 1.0)
        assert mc.intentions[intention.id].status == "completed"
    
    def test_get_active_intentions(self):
        """测试获取活跃意图"""
        mc = create_metacognition()
        
        mc.register_intention("意图1", priority=0.5)
        intent2 = mc.register_intention("意图2", priority=0.8)
        mc.register_intention("意图3", priority=0.3)
        
        # 完成一个意图
        mc.update_intention_progress(intent2.id, 1.0)
        
        active = mc.get_active_intentions()
        assert len(active) == 2


class TestExtendedThoughtChain:
    """延长思维链测试"""
    
    def test_extend_without_seed(self):
        """测试无 TheSeed 时的延长"""
        mc = create_metacognition()
        
        state = np.random.randn(64).astype(np.float32)
        thoughts, trajectory = mc.extend_thought_chain(state, 0)
        
        # 无 seed 时应返回空
        assert thoughts == []
        assert trajectory == []
    
    def test_extend_with_mock_seed(self):
        """测试模拟 TheSeed 的延长"""
        # 创建模拟 Seed
        class MockSeed:
            def simulate_trajectory(self, state, action, horizon=20):
                return [
                    (np.random.randn(64), 0.3, i % 4)
                    for i in range(horizon)
                ]
            
            def project_thought(self, state):
                return f"Concept_{hash(state.tobytes()) % 10000:04d}"
        
        mc = create_metacognition(seed_ref=MockSeed(), horizon=20)
        
        state = np.random.randn(64).astype(np.float32)
        thoughts, trajectory = mc.extend_thought_chain(state, 0)
        
        # 应有 20 步 + 1 个 META 观察
        assert len(thoughts) == 21
        assert len(trajectory) == 20
        assert "[META]" in thoughts[-1]


class TestPatternDetection:
    """模式检测测试"""
    
    def test_entropy_lock_detection(self):
        """测试熵锁定检测"""
        mc = create_metacognition()
        
        # 模拟低变化状态
        for i in range(10):
            state = np.zeros(64).astype(np.float32)
            mc.observe(
                state_vector=state,
                action_taken=0,  # 总是相同动作
                action_name="rest",
                uncertainty=0.05,  # 极低不确定性
                thought_chain=["Same"],
                neural_confidence=0.95
            )
        
        # 检测熵锁定
        insight = mc.detect_entropy_lock()
        
        # 应该检测到异常
        assert insight is not None or len(mc.meta_insights) > 0


class TestIntrospectionReport:
    """内省报告测试"""
    
    def test_generate_report(self):
        """测试生成内省报告"""
        mc = create_metacognition()
        
        # 添加一些数据
        for i in range(5):
            state = np.random.randn(64).astype(np.float32)
            mc.observe(
                state_vector=state,
                action_taken=i % 4,
                action_name=f"action_{i}",
                uncertainty=0.5,
                thought_chain=[f"Thought_{i}"],
                neural_confidence=0.6
            )
        
        mc.register_intention("测试意图", priority=0.7)
        
        report = mc.generate_introspection_report()
        
        assert "timestamp" in report
        assert "total_ticks" in report
        assert report["total_ticks"] == 5
        assert "statistics" in report
        assert report["statistics"]["frames_recorded"] == 5
        assert report["statistics"]["active_intentions"] == 1


class TestIntegration:
    """集成测试"""
    
    def test_seed_integration(self):
        """测试与 TheSeed 的集成"""
        try:
            from core.seed import TheSeed
            
            seed = TheSeed(state_dim=64, action_dim=4)
            mc = create_metacognition(seed_ref=seed, horizon=20)
            
            # 验证 horizon 已更新
            assert mc.current_horizon == 20
            
            # 测试延长思维链
            state = np.random.randn(64).astype(np.float32)
            thoughts, trajectory = mc.extend_thought_chain(state, 0)
            
            assert len(trajectory) == 20
            print(f"✅ TheSeed 集成成功: 延长思维链 {len(trajectory)} 步")
            
        except ImportError as e:
            pytest.skip(f"TheSeed 不可用: {e}")
    
    def test_evolution_controller_integration(self):
        """测试与 EvolutionController 的集成"""
        try:
            # 检查是否能导入
            from core.evolution.impl import EvolutionController
            from core.llm_client import LLMService
            
            # 这个测试仅验证导入成功
            # 实际初始化需要完整环境
            print("✅ EvolutionController 导入成功")
            
        except ImportError as e:
            pytest.skip(f"EvolutionController 不可用: {e}")


class TestPersistence:
    """持久化测试"""
    
    def test_save_and_load_state(self, tmp_path):
        """测试状态保存和加载"""
        mc = create_metacognition()
        
        # 添加数据
        for i in range(5):
            state = np.random.randn(64).astype(np.float32)
            mc.observe(
                state_vector=state,
                action_taken=i % 4,
                action_name=f"action_{i}",
                uncertainty=0.5,
                thought_chain=[f"Thought_{i}"],
                neural_confidence=0.6
            )
        
        mc.register_intention("持久化测试", priority=0.9)
        mc.set_horizon(22)
        
        # 保存
        filepath = str(tmp_path / "metacognition_state.json")
        mc.save_state(filepath)
        
        # 加载到新实例
        mc2 = MetaCognition()
        mc2.load_state(filepath)
        
        assert mc2._tick_counter == 5
        assert mc2.current_horizon == 22
        assert len(mc2.intentions) == 1


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MetaCognition Module Tests")
    print("=" * 60)
    
    # 基础测试
    print("\n[1] 基础功能测试...")
    test_basic = TestMetaCognitionBasic()
    test_basic.test_initialization()
    test_basic.test_horizon_bounds()
    print("   ✅ 基础功能测试通过")
    
    # observe 测试
    print("\n[2] Observe 方法测试...")
    test_observe = TestObserve()
    test_observe.test_observe_single_frame()
    test_observe.test_observe_multiple_frames()
    print("   ✅ Observe 测试通过")
    
    # 意图测试
    print("\n[3] 意图管理测试...")
    test_intent = TestIntentions()
    test_intent.test_register_intention()
    test_intent.test_update_intention_progress()
    test_intent.test_get_active_intentions()
    print("   ✅ 意图管理测试通过")
    
    # 延长思维链测试
    print("\n[4] 延长思维链测试...")
    test_chain = TestExtendedThoughtChain()
    test_chain.test_extend_without_seed()
    test_chain.test_extend_with_mock_seed()
    print("   ✅ 延长思维链测试通过")
    
    # 模式检测测试
    print("\n[5] 模式检测测试...")
    test_pattern = TestPatternDetection()
    test_pattern.test_entropy_lock_detection()
    print("   ✅ 模式检测测试通过")
    
    # 内省报告测试
    print("\n[6] 内省报告测试...")
    test_report = TestIntrospectionReport()
    test_report.test_generate_report()
    print("   ✅ 内省报告测试通过")
    
    # 集成测试
    print("\n[7] 集成测试...")
    test_integration = TestIntegration()
    try:
        test_integration.test_seed_integration()
    except Exception as e:
        print(f"   ⚠️ TheSeed 集成: {e}")
    try:
        test_integration.test_evolution_controller_integration()
    except Exception as e:
        print(f"   ⚠️ EvolutionController 集成: {e}")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
