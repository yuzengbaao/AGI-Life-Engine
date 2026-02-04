"""
世界模型REST API集成测试
测试WorldModelTool与world_model_framework的API接口交互

作者: GitHub Copilot AI Assistant
日期: 2025-11-15
"""

import pytest
import asyncio
import requests
from unittest.mock import Mock, patch, MagicMock
from enhanced_tools_collection import WorldModelTool, ToolResult


class TestWorldModelRESTIntegration:
    """世界模型REST API集成测试套件"""
    
    @pytest.fixture
    def world_model_tool(self):
        """创建WorldModelTool实例"""
        return WorldModelTool()
    
    def test_world_model_tool_initialization(self, world_model_tool):
        """测试WorldModelTool初始化"""
        assert world_model_tool.name == "world_model"
        assert world_model_tool.category == "世界模型"
        assert "生成虚拟场景" in world_model_tool.description
        assert world_model_tool.base_url == "http://127.0.0.1:8001"
        assert world_model_tool.enabled is True
    
    @patch('requests.Session.get')
    def test_health_check_success(self, mock_get, world_model_tool):
        """测试健康检查成功"""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy", "service": "WorldModelAPI"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = world_model_tool.execute(operation="health")
        
        assert result.success is True
        assert result.data["status"] == "healthy"
        assert result.data["service"] == "WorldModelAPI"
        assert result.execution_time >= 0  # 允许0值（mock环境）
        assert world_model_tool.call_count == 1
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_health_check_connection_error(self, mock_get, world_model_tool):
        """测试健康检查连接失败"""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
        
        result = world_model_tool.execute(operation="health")
        
        assert result.success is False
        assert "无法连接到世界模型服务" in result.error
        assert result.execution_time >= 0  # 允耸0值（mock环境）
        assert world_model_tool.call_count == 1
    
    @patch('requests.Session.post')
    def test_generate_world_success(self, mock_post, world_model_tool):
        """测试生成世界成功"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "world_id": "world_001",
            "world_data": {
                "description": "桌子上放一个红色杯子",
                "objects": [{"type": "table"}, {"type": "cup", "color": "red"}]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = world_model_tool.execute(
            operation="generate",
            prompt="桌子上放一个红色杯子",
            type="text"
        )
        
        assert result.success is True
        assert result.data["world_id"] == "world_001"
        assert "objects" in result.data["world_data"]
        assert len(result.data["world_data"]["objects"]) == 2
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_generate_world_missing_prompt(self, mock_post, world_model_tool):
        """测试生成世界缺少prompt参数"""
        result = world_model_tool.execute(operation="generate")
        
        assert result.success is False
        assert "缺少必需参数: prompt" in result.error
        mock_post.assert_not_called()
    
    @patch('requests.Session.post')
    def test_simulate_world_success(self, mock_post, world_model_tool):
        """测试物理仿真成功"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "world_id": "world_001",
            "updated_state": {
                "simulation_steps": 1,
                "objects": [{"id": "cup", "position": [0, 0, 1]}]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = world_model_tool.execute(
            operation="simulate",
            world_id="world_001",
            actions=[{"type": "move", "object": "cup", "to": {"x": 0, "y": 0, "z": 1}}]
        )
        
        assert result.success is True
        assert result.data["world_id"] == "world_001"
        assert result.data["updated_state"]["simulation_steps"] == 1
        mock_post.assert_called_once()
    
    @patch('requests.Session.post')
    def test_simulate_world_missing_world_id(self, mock_post, world_model_tool):
        """测试物理仿真缺少world_id参数"""
        result = world_model_tool.execute(
            operation="simulate",
            actions=[{"type": "move"}]
        )
        
        assert result.success is False
        assert "缺少必需参数: world_id" in result.error
        mock_post.assert_not_called()
    
    @patch('requests.Session.get')
    def test_observe_world_success(self, mock_get, world_model_tool):
        """测试环境观测成功"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "world_id": "world_001",
            "observations": {
                "object_positions": [{"id": "cup", "x": 0, "y": 0, "z": 1}],
                "collisions": [],
                "forces": []
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        result = world_model_tool.execute(
            operation="observe",
            world_id="world_001"
        )
        
        assert result.success is True
        assert result.data["world_id"] == "world_001"
        assert "observations" in result.data
        assert "object_positions" in result.data["observations"]
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_observe_world_missing_world_id(self, mock_get, world_model_tool):
        """测试环境观测缺少world_id参数"""
        result = world_model_tool.execute(operation="observe")
        
        assert result.success is False
        assert "缺少必需参数: world_id" in result.error
        mock_get.assert_not_called()
    
    def test_unsupported_operation(self, world_model_tool):
        """测试不支持的操作类型"""
        result = world_model_tool.execute(operation="invalid_operation")
        
        assert result.success is False
        assert "不支持的操作" in result.error
        assert "invalid_operation" in result.error
    
    @patch('requests.Session.post')
    def test_generate_world_api_error(self, mock_post, world_model_tool):
        """测试API返回错误"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "error",
            "error": "Internal server error"
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        result = world_model_tool.execute(
            operation="generate",
            prompt="测试场景"
        )
        
        assert result.success is False
        assert result.error == "Internal server error"
    
    def test_tool_statistics_tracking(self, world_model_tool):
        """测试工具统计跟踪"""
        with patch('requests.Session.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"status": "healthy"}
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response
            
            # 执行3次操作
            for _ in range(3):
                world_model_tool.execute(operation="health")
            
            info = world_model_tool.get_info()
            assert info["call_count"] == 3
            assert info["total_time"] >= 0  # 允耸0值（mock环境执行非常快）
            assert info["avg_time"] >= 0
    
    @patch.dict('os.environ', {'WORLD_MODEL_BASE_URL': 'http://custom-host:9000'})
    def test_custom_base_url_from_env(self):
        """测试从环境变量读取自定义base_url"""
        tool = WorldModelTool()
        assert tool.base_url == "http://custom-host:9000"
    
    @patch('requests.Session.post')
    def test_full_workflow_generate_simulate_observe(self, mock_post, world_model_tool):
        """测试完整工作流：生成→仿真→观测"""
        # Step 1: Generate
        mock_post.return_value = Mock(
            json=lambda: {"status": "success", "world_id": "test_world"},
            raise_for_status=Mock()
        )
        gen_result = world_model_tool.execute(
            operation="generate",
            prompt="测试场景"
        )
        assert gen_result.success is True
        world_id = gen_result.data["world_id"]
        
        # Step 2: Simulate
        mock_post.return_value = Mock(
            json=lambda: {"status": "success", "world_id": world_id, "updated_state": {}},
            raise_for_status=Mock()
        )
        sim_result = world_model_tool.execute(
            operation="simulate",
            world_id=world_id,
            actions=[{"type": "move"}]
        )
        assert sim_result.success is True
        
        # Step 3: Observe
        with patch('requests.Session.get') as mock_get:
            mock_get.return_value = Mock(
                json=lambda: {"status": "success", "world_id": world_id, "observations": {}},
                raise_for_status=Mock()
            )
            obs_result = world_model_tool.execute(
                operation="observe",
                world_id=world_id
            )
            assert obs_result.success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
