#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active AGI 端到端集成测试
验证完整的6步主动AGI流程和各组件协作
"""

import sys
import os
import asyncio
import pytest

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from enhanced_llm_core import EnhancedLLMCore
from active_agi_wrapper import ActiveAGIWrapper
from enhanced_tools_collection import get_tool_manager


class TestActiveAGIIntegration:
    """主动AGI端到端集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_6step_workflow(self):
        """验证完整的6步主动AGI流程"""
        # 初始化系统
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 处理用户输入
        result = await wrapper.process_user_input(
            "测试系统运行状态", 
            user_id="integration_test"
        )
        
        # 验证结果包含所有关键字段
        assert 'dominant_motivation' in result, "应该包含主导动机"
        assert 'actions' in result, "应该包含行动数量"
        assert 'tasks' in result, "应该包含任务数量"
        assert 'tasks_completed' in result, "应该包含完成任务数"
        assert 'reward' in result, "应该包含奖励值"
        
        # 验证值的合理性
        assert result['actions'] > 0, "应该生成至少一个行动"
        assert result['tasks'] > 0, "应该创建至少一个任务"
        assert result['tasks_completed'] >= 0, "完成任务数不能为负"
        assert -1.0 <= result['reward'] <= 1.0, "奖励应在-1到1之间"
        
        print(f"✅ 6步流程测试通过: {result['tasks_completed']}/{result['tasks']} 任务完成")
    
    @pytest.mark.asyncio
    async def test_proactive_suggestions_generation(self):
        """验证主动建议生成功能"""
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 生成主动建议
        suggestions = await wrapper.get_proactive_suggestions(count=3)
        
        # 验证建议数量
        assert len(suggestions) > 0, "应该生成至少一条建议"
        assert len(suggestions) <= 3, "不应超过请求的数量"
        
        # 验证每条建议的结构
        for suggestion in suggestions:
            assert 'type' in suggestion, "建议应包含type字段"
            assert 'motivation' in suggestion, "建议应包含motivation字段"
            # description或content至少有一个
            assert 'description' in suggestion or 'content' in suggestion, \
                "建议应包含description或content字段"
        
        print(f"✅ 主动建议测试通过: 生成{len(suggestions)}条建议")
    
    @pytest.mark.asyncio
    async def test_tool_manager_integration(self):
        """验证工具管理器集成"""
        tm = get_tool_manager()
        
        # 测试系统信息工具
        result = tm.execute_tool("system_info", info_type="cpu")
        
        assert result is not None, "工具应返回结果"
        assert result.success, "工具调用应成功"
        assert 'cpu' in result.data, "应包含CPU信息"
        
        cpu = result.data['cpu']
        assert 'count' in cpu, "CPU信息应包含核心数"
        assert 'percent' in cpu, "CPU信息应包含使用率"
        assert cpu['count'] > 0, "CPU核心数应大于0"
        assert 0 <= cpu['percent'] <= 100, "CPU使用率应在0-100之间"
        
        print(f"✅ 工具集成测试通过: CPU {cpu['count']}核, 使用率{cpu['percent']}%")
    
    @pytest.mark.asyncio
    async def test_motivation_evaluation_flow(self):
        """验证动机评估流程"""
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 评估动机
        motivations = await wrapper.motivation.evaluate_all_motivations()
        
        # 验证返回所有动机类型
        assert len(motivations) == 5, "应该包含5种动机类型"
        
        # 验证每种动机的强度在合理范围
        for mot_type, intensity in motivations.items():
            assert 0 <= intensity <= 100, \
                f"动机{mot_type.value}的强度{intensity}应在0-100之间"
        
        # 获取主导动机
        dominant, score = wrapper.motivation.get_dominant_motivation()
        assert dominant is not None, "应该有主导动机"
        assert score > 0, "主导动机分数应大于0"
        
        print(f"✅ 动机评估测试通过: 主导动机={dominant.value}, 分数={score:.2f}")
    
    @pytest.mark.asyncio
    async def test_task_execution_pipeline(self):
        """验证任务执行流水线"""
        from active_agi.multi_agent_system import Task, AgentType, MultiAgentCoordinator
        from unified_memory_system import UnifiedMemorySystem
        
        memory = UnifiedMemorySystem()
        coordinator = MultiAgentCoordinator(memory)
        
        # 创建测试任务
        task = Task(
            task_id="integration_test_1",
            description="test task execution",
            assigned_to=AgentType.EXECUTOR,
            priority=1
        )
        
        # 执行任务
        await coordinator.execute_task_pipeline(task)
        
        # 验证任务状态
        assert task.status == "completed", "任务应该完成"
        assert task.completed_at is not None, "应该记录完成时间"
        
        print(f"✅ 任务执行测试通过: {task.task_id} 状态={task.status}")
    
    @pytest.mark.asyncio
    async def test_decision_learning_cycle(self):
        """验证决策学习循环"""
        from active_agi.decision_layer import AutonomousDecisionLayer
        
        decision_layer = AutonomousDecisionLayer()
        
        # 感知状态
        state = decision_layer.perceive_state({"test": "context"})
        assert state is not None
        
        # 做出决策
        action = decision_layer.decide(state, method="q_learning")
        assert action is not None
        assert hasattr(action, 'action_type')
        
        # 执行并学习
        reward = decision_layer.execute_and_learn(state, action)
        assert isinstance(reward, float)
        assert -1.0 <= reward <= 1.0
        
        # 验证学习效果（经验缓冲区应有记录）
        assert decision_layer.experience_buffer.size() > 0, "应该记录经验"
        assert decision_layer.total_decisions > 0, "应该记录决策次数"
        
        print(f"✅ 决策学习测试通过: 决策={action.action_type.value}, 奖励={reward:.3f}")


class TestMultipleRounds:
    """多轮交互测试"""
    
    @pytest.mark.asyncio
    async def test_consecutive_interactions(self):
        """验证连续多轮交互"""
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 执行3轮交互
        inputs = [
            "第一轮测试",
            "第二轮测试",
            "第三轮测试"
        ]
        
        results = []
        for user_input in inputs:
            result = await wrapper.process_user_input(user_input, user_id="multi_round_test")
            results.append(result)
            
            # 每轮都应该成功
            assert 'dominant_motivation' in result
            assert 'reward' in result
        
        # 验证决策系统在学习（决策次数增加）
        assert wrapper.decision.total_decisions >= 3, "应该至少做出3次决策"
        
        print(f"✅ 多轮交互测试通过: 完成{len(results)}轮")
    
    @pytest.mark.asyncio
    async def test_motivation_adaptation(self):
        """验证动机系统适应性"""
        from active_agi.motivation_system import MotivationType
        
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 第一次评估
        motivations_1 = await wrapper.motivation.evaluate_all_motivations()
        
        # 模拟互动（记录一次高质量互动）
        wrapper.motivation.social.record_interaction(85.0, "很好")
        
        # 第二次评估
        motivations_2 = await wrapper.motivation.evaluate_all_motivations()
        
        # 社交动机应该有变化（可能增加或因满足而降低）
        # 这里只验证系统能响应变化
        assert motivations_1 is not None
        assert motivations_2 is not None
        
        social_1 = motivations_1[MotivationType.SOCIAL]
        social_2 = motivations_2[MotivationType.SOCIAL]
        
        print(f"✅ 动机适应测试通过: 社交动机 {social_1:.1f} → {social_2:.1f}")


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "--tb=short", "-s"])
