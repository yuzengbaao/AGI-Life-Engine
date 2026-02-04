#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active AGI Wrapper 单元测试
测试核心流程的正确性，防止回归
"""

import sys
import os
import asyncio
import pytest

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from active_agi.multi_agent_system import Task, AgentType
from active_agi.motivation_system import MotivationSystem
from unified_memory_system import UnifiedMemorySystem


class TestTaskInstantiation:
    """测试Task对象实例化参数"""
    
    def test_task_with_assigned_to_and_priority(self):
        """验证Task接受assigned_to和priority参数"""
        task = Task(
            task_id="test_1",
            description="test task",
            assigned_to=AgentType.EXECUTOR,
            priority=1
        )
        
        assert task.task_id == "test_1"
        assert task.description == "test task"
        assert task.assigned_to == AgentType.EXECUTOR
        assert task.priority == 1
        assert task.status == "pending"  # 默认状态
    
    def test_task_without_role_parameter(self):
        """确保Task不再接受role参数（已废弃）"""
        # 这个测试确保不会出现role参数的调用
        with pytest.raises(TypeError):
            Task(
                task_id="test_2",
                description="test",
                role="some_role",  # 应该触发TypeError
                priority=1
            )


class TestMotivationSystem:
    """测试动机系统的异步行为"""
    
    @pytest.mark.asyncio
    async def test_generate_motivated_actions_is_async(self):
        """验证generate_motivated_actions是异步方法且返回列表"""
        # 创建最小记忆系统
        memory = UnifiedMemorySystem()
        
        # 创建动机系统
        motivation = MotivationSystem(memory)
        
        # 调用异步方法
        actions = await motivation.generate_motivated_actions()
        
        # 验证返回类型
        assert isinstance(actions, list), "应该返回列表"
        assert len(actions) > 0, "应该生成至少一个行动"
        
        # 验证每个行动包含基本字段
        for action in actions[:3]:  # 检查前3个
            assert "type" in action, "行动应包含type字段"
            assert "motivation" in action, "行动应包含motivation字段"
    
    @pytest.mark.asyncio
    async def test_evaluate_all_motivations_returns_dict(self):
        """验证动机评估返回字典"""
        memory = UnifiedMemorySystem()
        motivation = MotivationSystem(memory)
        
        motivations = await motivation.evaluate_all_motivations()
        
        assert isinstance(motivations, dict), "应该返回字典"
        assert len(motivations) > 0, "应该有至少一种动机"
        
        # 验证值在合理范围内（动机系统返回0-100的强度值）
        for mot_type, score in motivations.items():
            assert 0.0 <= score <= 100.0, f"动机分数{score}应在0-100之间"


class TestDecisionLayerAPI:
    """测试决策层的API接口"""
    
    def test_perceive_state_creates_state(self):
        """验证perceive_state能创建State对象"""
        from active_agi.decision_layer import AutonomousDecisionLayer
        
        decision = AutonomousDecisionLayer()
        state = decision.perceive_state({"test": "context"})
        
        assert state is not None
        assert state.context == {"test": "context"}
        assert state.state_id is not None
    
    def test_decide_returns_action(self):
        """验证decide返回Action对象"""
        from active_agi.decision_layer import AutonomousDecisionLayer
        
        decision = AutonomousDecisionLayer()
        state = decision.perceive_state({"input": "test"})
        
        action = decision.decide(state, method="q_learning")
        
        assert action is not None
        assert hasattr(action, "action_type")
        assert hasattr(action, "action_id")
        assert hasattr(action, "estimated_value")
    
    def test_execute_and_learn_returns_reward(self):
        """验证execute_and_learn返回奖励值"""
        from active_agi.decision_layer import AutonomousDecisionLayer
        
        decision = AutonomousDecisionLayer()
        state = decision.perceive_state({"input": "test"})
        action = decision.decide(state, method="q_learning")
        
        reward = decision.execute_and_learn(state, action)
        
        assert isinstance(reward, float)
        assert -1.0 <= reward <= 1.0, "奖励应在-1到1之间"


class TestEventDrivenSystem:
    """测试事件驱动系统的接口"""
    
    def test_internal_source_register_thought(self):
        """验证InternalEventSource.register_thought接口"""
        from active_agi.event_driven_system import InternalEventSource
        
        source = InternalEventSource()
        source.register_thought("test thought", 0.8)
        
        assert len(source.thought_queue) == 1
        assert source.thought_queue[0]["content"] == "test thought"
        assert source.thought_queue[0]["quality"] == 0.8
    
    def test_user_source_add_request(self):
        """验证UserEventSource.add_request接口"""
        from active_agi.event_driven_system import UserEventSource, EventPriority
        
        source = UserEventSource()
        source.add_request({"query": "test"}, EventPriority.NORMAL)
        
        assert len(source.pending_requests) == 1
        assert source.pending_requests[0]["request"]["query"] == "test"


class TestEnhancedToolManager:
    """测试增强工具管理器"""
    
    def test_execute_tool_exists(self):
        """验证execute_tool方法存在"""
        from enhanced_tools_collection import EnhancedToolManager
        
        manager = EnhancedToolManager()
        
        # 验证方法存在
        assert hasattr(manager, "execute_tool")
        assert callable(manager.execute_tool)
    
    def test_get_stats_exists(self):
        """验证get_stats方法存在"""
        from enhanced_tools_collection import EnhancedToolManager
        
        manager = EnhancedToolManager()
        
        assert hasattr(manager, "get_stats")
        assert callable(manager.get_stats)
    
    def test_get_tools_by_category_exists(self):
        """验证get_tools_by_category方法存在"""
        from enhanced_tools_collection import EnhancedToolManager
        
        manager = EnhancedToolManager()
        
        assert hasattr(manager, "get_tools_by_category")
        assert callable(manager.get_tools_by_category)


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "--tb=short"])
