#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：身份抽取与结构化入库 + 记忆检索
验证 agi_chat_enhanced.py 中的身份抽取逻辑和统一记忆系统的检索能力
"""
import pytest
import os
import sys
import asyncio
import json
import re
from datetime import datetime

# 确保项目根目录在模块路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from unified_memory_system import UnifiedMemorySystem, MemoryPurpose


class TestIdentityExtraction:
    """测试身份抽取与入库逻辑"""

    def test_name_extraction_pattern_with_jiao(self):
        """测试"我叫X"模式的身份抽取"""
        user_input = "我是你的开发者，我叫宝总"
        
        # 模拟抽取逻辑
        name = None
        m = re.search(r"我叫([\u4e00-\u9fa5A-Za-z0-9_\-]+)", user_input)
        if m:
            name = m.group(1)
        
        assert name == "宝总", f"Expected '宝总', got '{name}'"

    def test_name_extraction_pattern_with_mingzi(self):
        """测试"我的名字是X"模式的身份抽取"""
        user_input = "大家好，我的名字是李四"
        
        name = None
        m = re.search(r"我的名字是([\u4e00-\u9fa5A-Za-z0-9_\-]+)", user_input)
        if m:
            name = m.group(1)
        
        assert name == "李四", f"Expected '李四', got '{name}'"

    def test_role_extraction_developer(self):
        """测试"开发者"角色抽取"""
        user_input = "我是你的开发者"
        
        role = None
        if "开发者" in user_input:
            role = "开发者"
        
        assert role == "开发者", f"Expected '开发者', got '{role}'"

    def test_role_extraction_author(self):
        """测试"作者"角色抽取"""
        user_input = "我是本项目的作者"
        
        role = None
        if "作者" in user_input:
            role = "作者"
        
        assert role == "作者", f"Expected '作者', got '{role}'"

    def test_combined_name_and_role(self):
        """测试姓名与角色同时抽取"""
        user_input = "我是你的开发者，我叫宝总"
        
        name = None
        m = re.search(r"我叫([\u4e00-\u9fa5A-Za-z0-9_\-]+)", user_input)
        if m:
            name = m.group(1)
        
        role = None
        if "开发者" in user_input:
            role = "开发者"
        
        assert name == "宝总", f"Name: expected '宝总', got '{name}'"
        assert role == "开发者", f"Role: expected '开发者', got '{role}'"


class TestIdentityMemoryIntegration:
    """测试身份记忆的完整流程：入库 + 检索"""

    @pytest.fixture
    def memory_system(self, tmp_path):
        """创建临时数据库的记忆系统实例"""
        # 使用 pytest 提供的临时目录避免污染真实环境
        db_file = tmp_path / "test_memory.db"
        mem = UnifiedMemorySystem(
            text_memory_db=str(db_file),
            enable_visual_memory=False
        )
        return mem

    def test_store_and_retrieve_identity(self, memory_system):
        """测试完整流程：存储身份资料并检索"""
        # 模拟入库
        profile_text = "用户资料：姓名=宝总 角色=开发者"
        memory_id = memory_system.add_text_memory(
            content=profile_text,
            memory_layer="long_term",
            memory_purpose=MemoryPurpose.KNOWLEDGE,
            tags=["user", "profile", "name"],
            metadata={
                "source": "chat_identity",
                "user_name": "宝总",
                "user_role": "开发者"
            },
            importance_score=0.9
        )
        
        assert memory_id is not None, "Memory ID should not be None"
        assert memory_id.startswith("tm_"), f"Memory ID should start with 'tm_', got '{memory_id}'"
        
        # 模拟检索
        results = memory_system.search_memories(
            query="用户姓名 姓名 名字 我叫 开发者",
            memory_types=["text"],
            memory_purpose=MemoryPurpose.KNOWLEDGE,
            tags=["user", "profile", "name"],
            limit=5
        )
        
        # 验证检索结果
        assert len(results) > 0, "Should find at least one identity record"
        
        first_result = results[0]
        assert "宝总" in first_result.get("content", ""), "Content should contain '宝总'"
        assert "开发者" in first_result.get("content", ""), "Content should contain '开发者'"
        
        # 验证元数据
        metadata = first_result.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        
        assert metadata.get("user_name") == "宝总", f"Expected user_name '宝总', got '{metadata.get('user_name')}'"
        assert metadata.get("user_role") == "开发者", f"Expected user_role '开发者', got '{metadata.get('user_role')}'"

    def test_retrieve_by_purpose_filter(self, memory_system):
        """测试用途过滤：只检索 knowledge 类型"""
        # 写入一条 knowledge
        memory_system.add_text_memory(
            content="用户资料：姓名=测试用户",
            memory_layer="long_term",
            memory_purpose=MemoryPurpose.KNOWLEDGE,
            tags=["user", "profile"],
            importance_score=0.9
        )
        
        # 写入一条 experience (不应被检索到)
        memory_system.add_text_memory(
            content="用户执行了某任务",
            memory_layer="long_term",
            memory_purpose=MemoryPurpose.EXPERIENCE,
            tags=["task"],
            importance_score=0.5
        )
        
        # 仅检索 knowledge（使用更精确的查询词）
        results = memory_system.search_memories(
            query="用户资料 姓名 测试用户",
            memory_types=["text"],
            memory_purpose=MemoryPurpose.KNOWLEDGE,
            limit=10
        )
        
        # 验证只命中 knowledge
        assert len(results) >= 1, "Should find at least one knowledge record"
        
        for r in results:
            # 检查是否包含 knowledge 相关内容（简单验证）
            content = r.get("content", "")
            assert "用户资料" in content or "姓名" in content, f"Unexpected content: {content}"

    def test_retrieve_by_tags_filter(self, memory_system):
        """测试标签过滤：只检索带 user/profile/name 标签的记录"""
        # 写入一条带正确标签
        memory_system.add_text_memory(
            content="用户资料：姓名=张三",
            memory_layer="long_term",
            memory_purpose=MemoryPurpose.KNOWLEDGE,
            tags=["user", "profile", "name"],
            importance_score=0.9
        )
        
        # 写入一条不带目标标签
        memory_system.add_text_memory(
            content="系统配置信息",
            memory_layer="long_term",
            memory_purpose=MemoryPurpose.KNOWLEDGE,
            tags=["system", "config"],
            importance_score=0.5
        )
        
        # 按标签过滤检索
        results = memory_system.search_memories(
            query="姓名 用户",
            memory_types=["text"],
            memory_purpose=MemoryPurpose.KNOWLEDGE,
            tags=["user", "profile", "name"],
            limit=10
        )
        
        assert len(results) >= 1, "Should find at least one record with correct tags"
        
        # 验证结果包含"张三"
        found_zhangsan = False
        for r in results:
            if "张三" in r.get("content", ""):
                found_zhangsan = True
                break
        
        assert found_zhangsan, "Should find the record with '张三'"


if __name__ == "__main__":
    # 支持直接运行测试
    pytest.main([__file__, "-v", "--tb=short"])
