"""
跨会话记忆系统 - 知识图谱核心测试
Cross-Session Memory System - Knowledge Graph Core Tests

版本: 1.0.0
日期: 2025-11-14
阶段: Phase 3 - 知识图谱构建
"""

import pytest
import json
import tempfile
import os
from knowledge_graph import KnowledgeGraph, KnowledgeGraphError
from entity_extractor import Entity, EntityType
from relationship_builder import Relationship, RelationshipType


class TestNodeCRUD:
    """测试节点CRUD操作"""

    def test_add_node(self):
        """测试添加节点"""
        kg = KnowledgeGraph()
        entity = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)

        result = kg.add_node(entity)
        assert result is True
        assert kg.has_node("python")  # 小写存储

    def test_add_duplicate_node(self):
        """测试添加重复节点"""
        kg = KnowledgeGraph()
        entity = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)

        kg.add_node(entity)
        result = kg.add_node(entity)  # 再次添加
        assert result is False  # 默认不覆盖

    def test_add_node_with_overwrite(self):
        """测试覆盖节点"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.8)
        e2 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.9)

        kg.add_node(e1)
        result = kg.add_node(e2, overwrite=True)
        assert result is True

        node = kg.get_node("python")
        assert node["confidence"] == 0.9

    def test_get_node(self):
        """测试获取节点"""
        kg = KnowledgeGraph()
        entity = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(entity)

        node = kg.get_node("Python")
        assert node is not None
        assert node["name"] == "Python"
        assert node["entity_type"] == "technology"

    def test_get_nonexistent_node(self):
        """测试获取不存在节点"""
        kg = KnowledgeGraph()
        node = kg.get_node("Nonexistent")
        assert node is None

    def test_update_node(self):
        """测试更新节点"""
        kg = KnowledgeGraph()
        entity = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(entity)

        result = kg.update_node("Python", {"confidence": 0.95})
        assert result is True

        node = kg.get_node("Python")
        assert node["confidence"] == 0.95

    def test_update_nonexistent_node(self):
        """测试更新不存在节点"""
        kg = KnowledgeGraph()
        result = kg.update_node("Nonexistent", {"confidence": 0.9})
        assert result is False

    def test_remove_node(self):
        """测试删除节点"""
        kg = KnowledgeGraph()
        entity = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(entity)

        result = kg.remove_node("Python")
        assert result is True
        assert not kg.has_node("Python")

    def test_remove_nonexistent_node(self):
        """测试删除不存在节点"""
        kg = KnowledgeGraph()
        result = kg.remove_node("Nonexistent")
        assert result is False

    def test_has_node(self):
        """测试检查节点存在"""
        kg = KnowledgeGraph()
        entity = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(entity)

        assert kg.has_node("Python") is True
        assert kg.has_node("JavaScript") is False


class TestEdgeCRUD:
    """测试边CRUD操作"""

    def test_add_edge(self):
        """测试添加边"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        result = kg.add_edge(rel)
        assert result is True

    def test_add_edge_missing_source(self):
        """测试添加边但源节点不存在"""
        kg = KnowledgeGraph()
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e2)

        rel = Relationship(
            source="Python", target="Django", relation_type=RelationshipType.USES
        )

        with pytest.raises(KnowledgeGraphError, match="Source node"):
            kg.add_edge(rel)

    def test_add_edge_missing_target(self):
        """测试添加边但目标节点不存在"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)

        rel = Relationship(
            source="Python", target="Django", relation_type=RelationshipType.USES
        )

        with pytest.raises(KnowledgeGraphError, match="Target node"):
            kg.add_edge(rel)

    def test_add_duplicate_edge(self):
        """测试添加重复边"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)
        result = kg.add_edge(rel)  # 再次添加
        assert result is False

    def test_get_edge(self):
        """测试获取边"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)

        edge = kg.get_edge("Django", "Python")
        assert edge is not None
        assert edge["relation_type"] == "uses"

    def test_get_nonexistent_edge(self):
        """测试获取不存在边"""
        kg = KnowledgeGraph()
        edge = kg.get_edge("A", "B")
        assert edge is None

    def test_update_edge(self):
        """测试更新边"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)

        result = kg.update_edge("Django", "Python", {"strength": 0.95})
        assert result is True

        edge = kg.get_edge("Django", "Python")
        assert edge["strength"] == 0.95

    def test_remove_edge(self):
        """测试删除边"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)

        result = kg.remove_edge("Django", "Python")
        assert result is True
        assert not kg.has_edge("Django", "Python")

    def test_has_edge(self):
        """测试检查边存在"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)

        assert kg.has_edge("Django", "Python") is True
        assert kg.has_edge("Python", "Django") is False


class TestSerialization:
    """测试序列化/反序列化"""

    def test_to_dict(self):
        """测试转换为字典"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)

        data = kg.to_dict()
        assert "metadata" in data
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 1

    def test_to_json(self):
        """测试转换为JSON"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)

        json_str = kg.to_json()
        assert isinstance(json_str, str)
        assert "Python" in json_str

    def test_from_dict(self):
        """测试从字典加载"""
        kg1 = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg1.add_node(e1)

        data = kg1.to_dict()

        kg2 = KnowledgeGraph()
        kg2.from_dict(data)

        assert kg2.node_count() == 1
        assert kg2.has_node("Python")

    def test_from_json(self):
        """测试从JSON加载"""
        kg1 = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg1.add_node(e1)

        json_str = kg1.to_json()

        kg2 = KnowledgeGraph()
        kg2.from_json(json_str)

        assert kg2.node_count() == 1
        assert kg2.has_node("Python")

    def test_save_and_load_file(self):
        """测试保存和加载文件"""
        kg1 = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg1.add_node(e1)
        kg1.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg1.add_edge(rel)

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            filepath = f.name

        try:
            kg1.save_to_file(filepath)

            # 从文件加载
            kg2 = KnowledgeGraph()
            kg2.load_from_file(filepath)

            assert kg2.node_count() == 2
            assert kg2.edge_count() == 1
            assert kg2.has_edge("Django", "Python")
        finally:
            os.unlink(filepath)


class TestStatistics:
    """测试图统计分析"""

    def test_node_count(self):
        """测试节点计数"""
        kg = KnowledgeGraph()
        assert kg.node_count() == 0

        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        assert kg.node_count() == 1

    def test_edge_count(self):
        """测试边计数"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        assert kg.edge_count() == 0

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)
        assert kg.edge_count() == 1

    def test_get_node_degree(self):
        """测试获取节点度数"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)

        degree = kg.get_node_degree("Python")
        assert degree["in"] == 1
        assert degree["out"] == 0
        assert degree["total"] == 1

    def test_degree_distribution(self):
        """测试度数分布"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)

        dist = kg.degree_distribution()
        assert "in" in dist
        assert "out" in dist
        assert "total" in dist
        assert len(dist["in"]) == 2

    def test_connected_components(self):
        """测试连通分量"""
        kg = KnowledgeGraph()
        # 创建两个独立的组件
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        e3 = Entity(name="Java", entity_type=EntityType.TECHNOLOGY)

        kg.add_node(e1)
        kg.add_node(e2)
        kg.add_node(e3)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)

        components = kg.connected_components()
        assert len(components) == 2  # Python-Django一组, Java一组

    def test_get_statistics(self):
        """测试获取统计信息"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)

        rel = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(rel)

        stats = kg.get_statistics()
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert "avg_degree" in stats
        assert "connected_components" in stats


class TestSubgraph:
    """测试子图提取"""

    def test_get_neighbors(self):
        """测试获取邻居节点"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        e3 = Entity(name="Flask", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)
        kg.add_node(e3)

        r1 = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        r2 = Relationship(
            source="Flask", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(r1)
        kg.add_edge(r2)

        neighbors = kg.get_neighbors("Python", direction="in")
        assert len(neighbors["in"]) == 2
        assert "django" in neighbors["in"]
        assert "flask" in neighbors["in"]

    def test_get_subgraph(self):
        """测试提取子图"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Django", entity_type=EntityType.TECHNOLOGY)
        e3 = Entity(name="Java", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)
        kg.add_node(e2)
        kg.add_node(e3)

        r1 = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        kg.add_edge(r1)

        # 提取Python和Django的子图
        subgraph = kg.get_subgraph(["Python", "Django"])
        assert subgraph.node_count() == 2
        assert subgraph.edge_count() == 1

    def test_extract_ego_graph(self):
        """测试提取ego图"""
        kg = KnowledgeGraph()
        entities = [
            Entity(name="Python", entity_type=EntityType.TECHNOLOGY),
            Entity(name="Django", entity_type=EntityType.TECHNOLOGY),
            Entity(name="Flask", entity_type=EntityType.TECHNOLOGY),
            Entity(name="NumPy", entity_type=EntityType.TECHNOLOGY),
        ]

        for e in entities:
            kg.add_node(e)

        r1 = Relationship(
            source="Django", target="Python", relation_type=RelationshipType.USES
        )
        r2 = Relationship(
            source="Flask", target="Python", relation_type=RelationshipType.USES
        )
        r3 = Relationship(
            source="NumPy", target="Python", relation_type=RelationshipType.PART_OF
        )
        kg.add_edge(r1)
        kg.add_edge(r2)
        kg.add_edge(r3)

        # 以Python为中心提取1跳ego图
        ego = kg.extract_ego_graph("Python", radius=1)
        assert ego.node_count() == 4  # Python + 3个邻居

    def test_extract_ego_graph_nonexistent(self):
        """测试提取不存在节点的ego图"""
        kg = KnowledgeGraph()

        with pytest.raises(KnowledgeGraphError, match="does not exist"):
            kg.extract_ego_graph("Nonexistent")


class TestMiscellaneous:
    """测试其他功能"""

    def test_clear(self):
        """测试清空图谱"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)

        kg.clear()
        assert kg.node_count() == 0

    def test_case_insensitive_node_id(self):
        """测试节点ID大小写不敏感"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        kg.add_node(e1)

        assert kg.has_node("python")
        assert kg.has_node("PYTHON")
        assert kg.has_node("PyThOn")

    def test_multiple_edges_same_nodes(self):
        """测试同节点间多条边"""
        kg = KnowledgeGraph()
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Guido", entity_type=EntityType.PERSON)
        kg.add_node(e1)
        kg.add_node(e2)

        r1 = Relationship(
            source="Python", target="Guido", relation_type=RelationshipType.CREATED_BY
        )
        r2 = Relationship(
            source="Python", target="Guido", relation_type=RelationshipType.RELATED_TO
        )

        kg.add_edge(r1)
        kg.add_edge(r2)

        # 应该有两条不同类型的边
        assert kg.has_edge("Python", "Guido", "created_by")
        assert kg.has_edge("Python", "Guido", "related_to")


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
