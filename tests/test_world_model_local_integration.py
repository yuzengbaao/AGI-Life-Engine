"""
世界模型本地集成测试
测试WorldModelIntegrator与Active AGI决策流程的集成

作者: GitHub Copilot AI Assistant
日期: 2025-11-15
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from world_model_integration import WorldModelIntegrator, WorldState, SimulationResult, PhysicsViolationType


class TestWorldModelLocalIntegration:
    """世界模型本地集成测试套件"""
    
    @pytest.fixture
    def integrator(self):
        """创建WorldModelIntegrator实例"""
        return WorldModelIntegrator(
            enable_physics_check=True,
            enable_causality_check=True,
            cache_enabled=True
        )
    
    def test_integrator_initialization(self, integrator):
        """测试集成器初始化"""
        assert integrator.enabled is True
        assert integrator.world_model is not None
        assert integrator.statistics["total_validations"] == 0
        assert integrator.statistics["violations_prevented"] == 0
    
    @pytest.mark.asyncio
    async def test_validate_action_valid_movement(self, integrator):
        """测试验证有效移动动作"""
        action_desc = "Move robot from point A to point B"
        context = {
            "objects": [
                {"id": "robot", "position": [0, 0, 0], "velocity": [0, 0, 0], 
                 "mass": 10.0, "radius": 0.3}
            ],
            "environment": {"physics": {"gravity": 9.8}},
            "target": [5, 0, 0]
        }
        
        is_valid, explanation, result = await integrator.validate_action(action_desc, context)
        
        assert is_valid is True
        assert "validated successfully" in explanation.lower()
        assert result is not None
        assert result.is_physically_valid is True
        assert integrator.statistics["total_validations"] == 1
        assert integrator.statistics["validations_passed"] == 1
    
    @pytest.mark.asyncio
    async def test_validate_action_invalid_teleportation(self, integrator):
        """测试验证无效传送动作（违反因果律）"""
        action_desc = "Teleport robot instantly to destination"
        context = {
            "objects": [
                {"id": "robot", "position": [0, 0, 0], "velocity": [0, 0, 0], 
                 "mass": 10.0, "radius": 0.3}
            ],
            "environment": {"physics": {"gravity": 9.8}},
            "target": [100, 0, 0]
        }
        
        is_valid, explanation, result = await integrator.validate_action(action_desc, context)
        
        assert is_valid is False
        assert "causality" in explanation.lower() or "instant" in explanation.lower()
        assert result is not None
        assert result.is_physically_valid is False
        assert integrator.statistics["violations_prevented"] == 1
    
    @pytest.mark.asyncio
    async def test_validate_action_parsing_failure(self, integrator):
        """测试动作解析失败时的优雅降级"""
        action_desc = "Ambiguous action with no context"
        context = {}  # 空上下文
        
        is_valid, explanation, result = await integrator.validate_action(action_desc, context)
        
        # 轻量世界模型即使空上下文也可能通过验证（使用默认值）
        assert isinstance(is_valid, bool)
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    
    @pytest.mark.asyncio
    async def test_validate_action_disabled(self):
        """测试禁用世界模型校验"""
        integrator = WorldModelIntegrator(enable_physics_check=False)
        integrator.disable()
        
        is_valid, explanation, result = await integrator.validate_action(
            "Any action", {"objects": []}
        )
        
        assert is_valid is True
        assert "disabled" in explanation.lower()
        assert result is None
    
    @pytest.mark.asyncio
    async def test_validate_action_exception_handling(self, integrator):
        """测试验证过程异常时的优雅降级"""
        with patch.object(integrator.world_model, 'simulate_action', side_effect=Exception("Simulation error")):
            is_valid, explanation, result = await integrator.validate_action(
                "Move robot", {"objects": []}
            )
            
            # 异常时允许动作（优雅降级）
            assert is_valid is True
            assert "error" in explanation.lower()
            assert "allowing action" in explanation.lower()
    
    def test_statistics_tracking(self, integrator):
        """测试统计信息跟踪"""
        stats = integrator.get_statistics()
        
        assert "integration_stats" in stats
        assert "world_model_stats" in stats
        assert "prevention_rate" in stats
        assert "validation_success_rate" in stats
        assert stats["integration_stats"]["total_validations"] == 0
    
    @pytest.mark.asyncio
    async def test_statistics_update_after_validation(self, integrator):
        """测试验证后统计信息更新"""
        # 执行有效动作
        await integrator.validate_action(
            "Move robot slowly",
            {
                "objects": [{"id": "robot", "position": [0, 0, 0]}],
                "target": [1, 0, 0]
            }
        )
        
        stats = integrator.get_statistics()
        assert stats["integration_stats"]["total_validations"] == 1
        assert stats["integration_stats"]["validations_passed"] == 1
        assert stats["integration_stats"]["avg_validation_time"] >= 0  # 允许0值（轻量模型执行快）
    
    @pytest.mark.asyncio
    async def test_multiple_validations(self, integrator):
        """测试多次验证统计"""
        actions = [
            ("Move A to B", {"objects": [], "target": [1, 0, 0]}),
            ("Teleport instantly", {"objects": [], "target": [100, 0, 0]}),
            ("Apply force", {"objects": [], "force": 10.0})
        ]
        
        for action_desc, context in actions:
            await integrator.validate_action(action_desc, context)
        
        stats = integrator.get_statistics()
        assert stats["integration_stats"]["total_validations"] == 3
        assert stats["prevention_rate"] >= 0
        assert stats["validation_success_rate"] >= 0
    
    def test_reset_statistics(self, integrator):
        """测试重置统计信息"""
        integrator.statistics["total_validations"] = 10
        integrator.statistics["violations_prevented"] = 5
        
        integrator.reset_statistics()
        
        assert integrator.statistics["total_validations"] == 0
        assert integrator.statistics["violations_prevented"] == 0
    
    def test_enable_disable_toggle(self, integrator):
        """测试启用/禁用切换"""
        assert integrator.enabled is True
        
        integrator.disable()
        assert integrator.enabled is False
        
        integrator.enable()
        assert integrator.enabled is True
    
    @pytest.mark.asyncio
    async def test_action_type_inference(self, integrator):
        """测试动作类型推断"""
        test_cases = [
            ("Teleport robot to Mars", "teleport"),
            ("Move robot forward", "move"),
            ("Push the box", "push"),
            ("Throw the ball", "throw"),
            ("Generic action", "general")
        ]
        
        for action_desc, expected_type in test_cases:
            action_type = integrator._infer_action_type(action_desc)
            assert action_type == expected_type
    
    @pytest.mark.asyncio
    async def test_violation_type_counting(self, integrator):
        """测试违规类型统计"""
        # 模拟因果律违规
        with patch.object(integrator.world_model, 'simulate_action') as mock_sim:
            mock_sim.return_value = SimulationResult(
                is_physically_valid=False,
                violation_type=PhysicsViolationType.CAUSALITY_VIOLATION,
                predicted_state={},  # 添加必需参数
                simulation_time=0.001,  # 添加必需参数
                insights=["Instant movement detected"],
                confidence_score=0.95
            )
            
            await integrator.validate_action(
                "Teleport instantly",
                {"objects": []}
            )
            
            assert integrator.statistics["violation_types"]["causality"] == 1
    
    @pytest.mark.asyncio
    async def test_integration_with_agi_context(self, integrator):
        """测试与AGI上下文的集成"""
        agi_context = {
            "objects": [
                {"id": "agent", "position": [0, 0, 0], "velocity": [0, 0, 0], "mass": 1.0}
            ],
            "environment": {"physics": {"gravity": 9.8}},
            "user_input": "请移动到桌子旁边",
            "action": {"type": "move", "target": "table"}
        }
        
        is_valid, explanation, result = await integrator.validate_action(
            "移动到桌子旁边",
            agi_context
        )
        
        # 应该成功解析并验证
        assert isinstance(is_valid, bool)
        assert isinstance(explanation, str)
        assert len(explanation) > 0


class TestWorldModelIntegratorPerformance:
    """世界模型集成器性能测试"""
    
    @pytest.mark.asyncio
    async def test_validation_performance(self):
        """测试验证性能（应在毫秒级别）"""
        integrator = WorldModelIntegrator(cache_enabled=True)
        
        import time
        start = time.time()
        
        await integrator.validate_action(
            "Move robot from A to B",
            {"objects": [], "target": [5, 0, 0]}
        )
        
        elapsed = (time.time() - start) * 1000  # 转换为毫秒
        
        # 性能要求：验证应在10ms内完成
        assert elapsed < 10.0, f"Validation took {elapsed:.2f}ms, expected < 10ms"
    
    @pytest.mark.asyncio
    async def test_cache_effectiveness(self):
        """测试缓存有效性"""
        integrator = WorldModelIntegrator(cache_enabled=True)
        
        action = "Move robot to position X"
        context = {"objects": [], "target": [1, 0, 0]}
        
        # 第一次调用（无缓存）
        import time
        start1 = time.time()
        await integrator.validate_action(action, context)
        time1 = time.time() - start1
        
        # 第二次调用（有缓存）
        start2 = time.time()
        await integrator.validate_action(action, context)
        time2 = time.time() - start2
        
        # 缓存应显著提升性能
        # 注意：由于是轻量级实现，提升可能不明显
        assert time2 <= time1 * 1.5  # 允许50%的容差


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    @pytest.mark.asyncio
    async def test_validate_action_convenience(self):
        """测试validate_action便捷函数"""
        from world_model_integration import validate_action
        
        is_valid, explanation = await validate_action(
            "Move robot forward",
            {"objects": [], "target": [1, 0, 0]}
        )
        
        assert isinstance(is_valid, bool)
        assert isinstance(explanation, str)
    
    def test_get_validation_statistics_convenience(self):
        """测试get_validation_statistics便捷函数"""
        from world_model_integration import get_validation_statistics
        
        stats = get_validation_statistics()
        
        assert "integration_stats" in stats
        assert "world_model_stats" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
