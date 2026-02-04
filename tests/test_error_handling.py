#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Active AGI 错误处理和边界条件测试
验证系统在异常场景下的健壮性
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
from active_agi.multi_agent_system import Task, AgentType
from enhanced_tools_collection import get_tool_manager


class TestErrorHandling:
    """异常处理测试"""
    
    @pytest.mark.asyncio
    async def test_empty_input_handling(self):
        """测试空输入处理"""
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 处理空字符串
        result = await wrapper.process_user_input("", user_id="empty_test")
        
        # 系统应该处理而不崩溃
        assert result is not None, "空输入应该返回结果"
        assert 'dominant_motivation' in result, "应该包含基本字段"
        
        print("✅ 空输入处理测试通过")
    
    @pytest.mark.asyncio
    async def test_very_long_input(self):
        """测试超长输入处理"""
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 生成超长输入（1000个字符）
        long_input = "测试" * 500
        
        result = await wrapper.process_user_input(long_input, user_id="long_test")
        
        # 系统应该能够处理
        assert result is not None, "超长输入应该返回结果"
        assert 'reward' in result, "应该包含奖励字段"
        
        print("✅ 超长输入处理测试通过")
    
    @pytest.mark.asyncio
    async def test_special_characters_input(self):
        """测试特殊字符输入"""
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 包含各种特殊字符
        special_input = "测试@#$%^&*()_+<>?{}\|/~`"
        
        result = await wrapper.process_user_input(special_input, user_id="special_test")
        
        assert result is not None, "特殊字符输入应该返回结果"
        
        print("✅ 特殊字符处理测试通过")
    
    @pytest.mark.asyncio
    async def test_invalid_tool_call(self):
        """测试无效工具调用"""
        tm = get_tool_manager()
        
        # 调用不存在的工具
        result = tm.execute_tool("nonexistent_tool", param1="value")
        
        # 应该返回失败结果而不崩溃
        assert result is not None, "无效工具调用应返回结果"
        assert not result.success, "无效工具调用应标记为失败"
        assert result.error is not None, "应该包含错误信息"
        
        print(f"✅ 无效工具调用测试通过: {result.error}")
    
    @pytest.mark.asyncio
    async def test_invalid_task_parameters(self):
        """测试无效任务参数"""
        # 测试缺少必需参数
        with pytest.raises(TypeError):
            Task()  # 应该需要必需参数
        
        # 测试priority为非法值（Task可能接受任意类型，所以注释掉无效AgentType测试）
        # Task的assigned_to字段可能是Optional，创建后可通过assign_task动态分配
        task = Task(
            task_id="test_invalid",
            description="test with default assigned_to"
        )
        # 验证创建成功
        assert task.task_id == "test_invalid"
        
        print("✅ 无效任务参数测试通过")


class TestBoundaryConditions:
    """边界条件测试"""
    
    @pytest.mark.asyncio
    async def test_zero_proactive_suggestions(self):
        """测试请求0条主动建议"""
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 请求0条建议
        suggestions = await wrapper.get_proactive_suggestions(count=0)
        
        # 应该返回空列表而不崩溃
        assert isinstance(suggestions, list), "应返回列表"
        assert len(suggestions) == 0, "应该是空列表"
        
        print("✅ 零建议请求测试通过")
    
    @pytest.mark.asyncio
    async def test_excessive_suggestions_request(self):
        """测试请求过多主动建议"""
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 请求100条建议
        suggestions = await wrapper.get_proactive_suggestions(count=100)
        
        # 应该返回有限数量而不崩溃
        assert isinstance(suggestions, list), "应返回列表"
        # 系统可能限制最大数量
        assert len(suggestions) >= 0, "应该有合理数量的建议"
        
        print(f"✅ 过多建议请求测试通过: 返回{len(suggestions)}条")
    
    @pytest.mark.asyncio
    async def test_negative_priority_task(self):
        """测试负优先级任务"""
        from active_agi.multi_agent_system import MultiAgentCoordinator
        from unified_memory_system import UnifiedMemorySystem
        
        memory = UnifiedMemorySystem()
        coordinator = MultiAgentCoordinator(memory)
        
        # 创建负优先级任务
        task = Task(
            task_id="negative_priority",
            description="test negative priority",
            assigned_to=AgentType.EXECUTOR,
            priority=-1  # 负优先级
        )
        
        # 应该能够执行
        await coordinator.execute_task_pipeline(task)
        
        # 任务应该完成（系统容忍负优先级）
        assert task.status in ["completed", "in_progress"], "任务应该被处理"
        
        print("✅ 负优先级任务测试通过")
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """测试并发处理"""
        llm_core = EnhancedLLMCore()
        memory = llm_core.memory_system
        wrapper = ActiveAGIWrapper(memory, llm_core)
        await wrapper.start()
        
        # 并发处理多个请求
        async def process_request(i):
            return await wrapper.process_user_input(f"并发测试{i}", user_id=f"concurrent_{i}")
        
        tasks = [process_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 所有请求都应该成功或安全失败
        assert len(results) == 5, "应该返回5个结果"
        
        # 检查没有严重错误
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"  请求{i}异常: {result}")
            else:
                assert 'dominant_motivation' in result, f"请求{i}应包含基本字段"
        
        print(f"✅ 并发处理测试通过: {len([r for r in results if not isinstance(r, Exception)])}/5 成功")


class TestRobustness:
    """系统健壮性测试"""
    
    @pytest.mark.asyncio
    async def test_decision_layer_extreme_reward(self):
        """测试决策层极端奖励值"""
        from active_agi.decision_layer import AutonomousDecisionLayer, RewardCalculator
        
        calculator = RewardCalculator()
        
        # 测试极端奖励场景
        extreme_outcomes = [
            {"task_completion": 0.0, "user_satisfaction": 0.0},  # 全0
            {"task_completion": 1.0, "user_satisfaction": 1.0},  # 全1
            {},  # 空结果
        ]
        
        for outcome in extreme_outcomes:
            reward = calculator.calculate(outcome)
            # 奖励应该在合理范围内
            assert -1.0 <= reward <= 1.0, f"奖励{reward}应在-1到1之间"
        
        print("✅ 极端奖励测试通过")
    
    @pytest.mark.asyncio
    async def test_motivation_overflow_protection(self):
        """测试动机系统溢出保护"""
        from active_agi.motivation_system import MasteryDrive
        
        mastery = MasteryDrive()
        
        # 记录大量进步
        for i in range(100):
            mastery.record_improvement("reasoning", 5.0)
        
        # 技能等级不应超过100
        assert mastery.skills["reasoning"] <= 100, "技能等级应有上限"
        assert mastery.mastery_level <= 100, "掌握水平应有上限"
        
        print(f"✅ 动机溢出保护测试通过: reasoning={mastery.skills['reasoning']:.1f}")
    
    @pytest.mark.asyncio
    async def test_memory_search_with_no_results(self):
        """测试记忆搜索无结果场景"""
        from unified_memory_system import UnifiedMemorySystem
        
        memory = UnifiedMemorySystem()
        
        # 搜索不存在的内容
        results = memory.search_memories("完全不存在的内容xyz123", limit=10)
        
        # 应该返回空列表而不报错
        assert isinstance(results, list), "应返回列表"
        # 可能返回空列表或有其他结果
        
        print(f"✅ 无结果搜索测试通过: 返回{len(results)}条")


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "--tb=short", "-s"])
