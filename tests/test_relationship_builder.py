"""
跨会话记忆系统 - 关系构建器测试
Cross-Session Memory System - Relationship Builder Tests

版本: 1.0.0
日期: 2025-11-14
阶段: Phase 3 - 知识图谱构建
"""

import pytest
import logging
from datetime import datetime, timedelta
from relationship_builder import (
    RelationshipBuilder,
    Relationship,
    RelationshipType,
    RelationshipBuilderError,
)
from entity_extractor import Entity, EntityType

logging.disable(logging.CRITICAL)


class TestRelationship:
    """测试Relationship数据类"""

    def test_relationship_creation(self):
        """测试关系创建"""
        rel = Relationship(
            source="Python",
            target="Programming Language",
            relation_type=RelationshipType.IS_A,
        )
        assert rel.source == "Python"
        assert rel.target == "Programming Language"
        assert rel.relation_type == RelationshipType.IS_A
        assert rel.strength == 1.0
        assert rel.confidence == 1.0

    def test_relationship_with_strength(self):
        """测试带强度的关系"""
        rel = Relationship(
            source="A",
            target="B",
            relation_type=RelationshipType.RELATED_TO,
            strength=0.75,
        )
        assert rel.strength == 0.75

    def test_relationship_invalid_strength(self):
        """测试无效强度"""
        with pytest.raises(ValueError, match="strength must be between"):
            Relationship(
                source="A",
                target="B",
                relation_type=RelationshipType.IS_A,
                strength=1.5,
            )

    def test_relationship_invalid_confidence(self):
        """测试无效置信度"""
        with pytest.raises(ValueError, match="confidence must be between"):
            Relationship(
                source="A",
                target="B",
                relation_type=RelationshipType.IS_A,
                confidence=2.0,
            )

    def test_relationship_equality(self):
        """测试关系相等性"""
        rel1 = Relationship(
            source="Python", target="Language", relation_type=RelationshipType.IS_A
        )
        rel2 = Relationship(
            source="python", target="language", relation_type=RelationshipType.IS_A
        )
        rel3 = Relationship(
            source="Python", target="Language", relation_type=RelationshipType.HAS_A
        )

        assert rel1 == rel2  # 不区分大小写
        assert rel1 != rel3  # 类型不同

    def test_relationship_hash(self):
        """测试关系哈希"""
        rel1 = Relationship(
            source="Python", target="Language", relation_type=RelationshipType.IS_A
        )
        rel2 = Relationship(
            source="python", target="language", relation_type=RelationshipType.IS_A
        )

        rel_set = {rel1, rel2}
        assert len(rel_set) == 1  # 被去重

    def test_relationship_to_dict(self):
        """测试转换为字典"""
        rel = Relationship(
            source="Python",
            target="Language",
            relation_type=RelationshipType.IS_A,
            strength=0.9,
        )
        d = rel.to_dict()

        assert d["source"] == "Python"
        assert d["target"] == "Language"
        assert d["type"] == "is_a"
        assert d["strength"] == 0.9


class TestInitialization:
    """测试初始化"""

    def test_default_initialization(self):
        """测试默认初始化"""
        builder = RelationshipBuilder()
        assert builder.time_decay_rate == 0.1
        assert len(builder.patterns) > 0

    def test_custom_decay_rate(self):
        """测试自定义衰减率"""
        builder = RelationshipBuilder(time_decay_rate=0.2)
        assert builder.time_decay_rate == 0.2

    def test_invalid_decay_rate(self):
        """测试无效衰减率"""
        with pytest.raises(ValueError, match="time_decay_rate must be between"):
            RelationshipBuilder(time_decay_rate=1.5)


class TestPatternExtraction:
    """测试基于模式的关系提取"""

    def test_extract_is_a_relationship(self):
        """测试提取IS_A关系"""
        builder = RelationshipBuilder()
        text = "Python is a language"

        entities = [
            Entity(name="Python", entity_type=EntityType.TECHNOLOGY),
            Entity(name="language", entity_type=EntityType.CONCEPT),
        ]

        rels = builder.extract_relationships(text, entities)

        assert len(rels) > 0
        is_a_rels = [r for r in rels if r.relation_type == RelationshipType.IS_A]
        assert len(is_a_rels) > 0

    def test_extract_has_a_relationship(self):
        """测试提取HAS_A关系"""
        builder = RelationshipBuilder()
        text = "Python has libraries"

        entities = [
            Entity(name="Python", entity_type=EntityType.TECHNOLOGY),
            Entity(name="libraries", entity_type=EntityType.CONCEPT),
        ]

        rels = builder.extract_relationships(text, entities)

        has_a_rels = [r for r in rels if r.relation_type == RelationshipType.HAS_A]
        assert len(has_a_rels) > 0

    def test_extract_created_by_relationship(self):
        """测试提取CREATED_BY关系"""
        builder = RelationshipBuilder()
        text = "Python created by Guido"

        entities = [
            Entity(name="Python", entity_type=EntityType.TECHNOLOGY),
            Entity(name="Guido", entity_type=EntityType.PERSON),
        ]

        rels = builder.extract_relationships(text, entities)

        created_rels = [
            r for r in rels if r.relation_type == RelationshipType.CREATED_BY
        ]
        assert len(created_rels) > 0

    def test_extract_uses_relationship(self):
        """测试提取USES关系"""
        builder = RelationshipBuilder()
        text = "Django uses Python"

        entities = [
            Entity(name="Django", entity_type=EntityType.TECHNOLOGY),
            Entity(name="Python", entity_type=EntityType.TECHNOLOGY),
        ]

        rels = builder.extract_relationships(text, entities)

        uses_rels = [r for r in rels if r.relation_type == RelationshipType.USES]
        assert len(uses_rels) > 0

    def test_extract_empty_text(self):
        """测试空文本"""
        builder = RelationshipBuilder()
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            builder.extract_relationships("", [])


class TestCooccurrenceExtraction:
    """测试基于共现的关系提取"""

    def test_extract_cooccurrence_relationship(self):
        """测试提取共现关系"""
        builder = RelationshipBuilder()
        text = "Python and JavaScript are both popular"

        entities = [
            Entity(name="Python", entity_type=EntityType.TECHNOLOGY),
            Entity(name="JavaScript", entity_type=EntityType.TECHNOLOGY),
        ]

        rels = builder.extract_relationships(text, entities)

        # 应该有RELATED_TO关系(共现)
        related_rels = [
            r for r in rels if r.relation_type == RelationshipType.RELATED_TO
        ]
        assert len(related_rels) > 0

    def test_cooccurrence_strength(self):
        """测试共现关系强度"""
        builder = RelationshipBuilder()
        text = "Python JavaScript"  # 距离很近

        entities = [
            Entity(name="Python", entity_type=EntityType.TECHNOLOGY),
            Entity(name="JavaScript", entity_type=EntityType.TECHNOLOGY),
        ]

        rels = builder.extract_relationships(text, entities)

        if rels:
            # 距离近的应该有较高强度
            assert rels[0].strength > 0.5


class TestStrengthCalculation:
    """测试强度计算"""

    def test_calculate_strength_high_frequency(self):
        """测试高频率的强度"""
        builder = RelationshipBuilder()
        text = "Python Python Python language language"

        strength = builder._calculate_strength("Python", "language", text)
        assert 0.0 <= strength <= 1.0
        assert strength > 0.5  # 高频应该有较高强度

    def test_calculate_strength_close_proximity(self):
        """测试接近距离的强度"""
        builder = RelationshipBuilder()
        text = "Python language"

        strength = builder._calculate_strength("Python", "language", text)
        assert strength > 0.5  # 距离近应该有较高强度


class TestDeduplication:
    """测试去重"""

    def test_deduplicate_identical_relationships(self):
        """测试去除相同关系"""
        rel1 = Relationship(
            source="Python",
            target="Language",
            relation_type=RelationshipType.IS_A,
            strength=0.8,
        )
        rel2 = Relationship(
            source="Python",
            target="Language",
            relation_type=RelationshipType.IS_A,
            strength=0.9,
        )

        builder = RelationshipBuilder()
        result = builder._deduplicate_relationships([rel1, rel2])

        assert len(result) == 1
        assert result[0].strength == 0.9  # 保留强度更高的

    def test_deduplicate_case_insensitive(self):
        """测试大小写不敏感去重"""
        rel1 = Relationship(
            source="Python", target="Language", relation_type=RelationshipType.IS_A
        )
        rel2 = Relationship(
            source="python", target="language", relation_type=RelationshipType.IS_A
        )

        builder = RelationshipBuilder()
        result = builder._deduplicate_relationships([rel1, rel2])

        assert len(result) == 1


class TestTemporalDecay:
    """测试时间衰减"""

    def test_temporal_decay_no_decay(self):
        """测试无衰减(刚创建)"""
        rel = Relationship(
            source="A",
            target="B",
            relation_type=RelationshipType.RELATED_TO,
            strength=1.0,
        )

        builder = RelationshipBuilder(time_decay_rate=0.1)
        decayed = builder.calculate_temporal_decay(rel)

        assert decayed == pytest.approx(1.0, rel=0.01)

    def test_temporal_decay_with_time(self):
        """测试有时间衰减"""
        old_time = datetime.now() - timedelta(days=10)
        rel = Relationship(
            source="A",
            target="B",
            relation_type=RelationshipType.RELATED_TO,
            strength=1.0,
            timestamp=old_time,
        )

        builder = RelationshipBuilder(time_decay_rate=0.1)
        decayed = builder.calculate_temporal_decay(rel)

        assert decayed < 1.0  # 应该有衰减
        assert decayed > 0.0

    def test_temporal_decay_custom_reference(self):
        """测试自定义参考时间"""
        rel_time = datetime(2025, 1, 1)
        ref_time = datetime(2025, 1, 11)  # 10天后

        rel = Relationship(
            source="A",
            target="B",
            relation_type=RelationshipType.RELATED_TO,
            strength=1.0,
            timestamp=rel_time,
        )

        builder = RelationshipBuilder(time_decay_rate=0.1)
        decayed = builder.calculate_temporal_decay(rel, reference_time=ref_time)

        assert decayed < 1.0


class TestRelationshipMerging:
    """测试关系合并"""

    def test_merge_same_relationships(self):
        """测试合并相同关系"""
        rel1 = Relationship(
            source="Python",
            target="Language",
            relation_type=RelationshipType.IS_A,
            strength=0.8,
            confidence=0.9,
        )
        rel2 = Relationship(
            source="Python",
            target="Language",
            relation_type=RelationshipType.IS_A,
            strength=0.9,
            confidence=0.85,
        )

        builder = RelationshipBuilder()
        merged = builder.merge_relationships(rel1, rel2)

        assert merged.strength == 0.9  # 选择更高的
        assert merged.confidence == 0.9

    def test_merge_different_relationships(self):
        """测试合并不同关系失败"""
        rel1 = Relationship(
            source="Python", target="Language", relation_type=RelationshipType.IS_A
        )
        rel2 = Relationship(
            source="Python", target="Tool", relation_type=RelationshipType.IS_A
        )

        builder = RelationshipBuilder()
        with pytest.raises(ValueError, match="Cannot merge"):
            builder.merge_relationships(rel1, rel2)


class TestFiltering:
    """测试过滤"""

    def test_filter_by_type(self):
        """测试按类型过滤"""
        rel1 = Relationship(source="A", target="B", relation_type=RelationshipType.IS_A)
        rel2 = Relationship(
            source="C", target="D", relation_type=RelationshipType.HAS_A
        )

        builder = RelationshipBuilder()
        filtered = builder.filter_relationships(
            [rel1, rel2], relation_types=[RelationshipType.IS_A]
        )

        assert len(filtered) == 1
        assert filtered[0].relation_type == RelationshipType.IS_A

    def test_filter_by_strength(self):
        """测试按强度过滤"""
        rel1 = Relationship(
            source="A",
            target="B",
            relation_type=RelationshipType.IS_A,
            strength=0.9,
        )
        rel2 = Relationship(
            source="C",
            target="D",
            relation_type=RelationshipType.IS_A,
            strength=0.5,
        )

        builder = RelationshipBuilder()
        filtered = builder.filter_relationships([rel1, rel2], min_strength=0.7)

        assert len(filtered) == 1
        assert filtered[0].strength >= 0.7

    def test_filter_by_confidence(self):
        """测试按置信度过滤"""
        rel1 = Relationship(
            source="A",
            target="B",
            relation_type=RelationshipType.IS_A,
            confidence=0.9,
        )
        rel2 = Relationship(
            source="C",
            target="D",
            relation_type=RelationshipType.IS_A,
            confidence=0.6,
        )

        builder = RelationshipBuilder()
        filtered = builder.filter_relationships([rel1, rel2], min_confidence=0.8)

        assert len(filtered) == 1
        assert filtered[0].confidence >= 0.8


class TestBatchExtraction:
    """测试批量提取"""

    def test_build_from_conversations(self):
        """测试从对话构建关系"""
        builder = RelationshipBuilder()

        conversations = [
            {"content": "Python is a language", "role": "user"},
            {"content": "Django uses Python", "role": "assistant"},
        ]

        entities = [
            Entity(name="Python", entity_type=EntityType.TECHNOLOGY),
            Entity(name="language", entity_type=EntityType.CONCEPT),
            Entity(name="Django", entity_type=EntityType.TECHNOLOGY),
        ]

        rels = builder.build_from_conversations(conversations, entities)

        assert len(rels) > 0

    def test_build_from_empty_conversations(self):
        """测试空对话列表"""
        builder = RelationshipBuilder()
        rels = builder.build_from_conversations([], [])
        assert rels == []


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
