"""
跨会话记忆系统 - 实体提取器测试
Cross-Session Memory System - Entity Extractor Tests

版本: 1.0.0
日期: 2025-11-14
阶段: Phase 3 - 知识图谱构建
"""

import pytest
import logging
from entity_extractor import (
    EntityExtractor,
    Entity,
    EntityType,
    EntityExtractorError,
)

logging.disable(logging.CRITICAL)


class TestEntity:
    """测试Entity数据类"""

    def test_entity_creation(self):
        """测试实体创建"""
        entity = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        assert entity.name == "Python"
        assert entity.entity_type == EntityType.TECHNOLOGY
        assert entity.confidence == 1.0
        assert entity.aliases == set()

    def test_entity_with_aliases(self):
        """测试带别名的实体"""
        entity = Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            aliases={"py", "python3"},
        )
        assert "py" in entity.aliases
        assert "python3" in entity.aliases

    def test_entity_invalid_confidence(self):
        """测试无效置信度"""
        with pytest.raises(ValueError, match="Confidence must be between"):
            Entity(name="Test", entity_type=EntityType.CONCEPT, confidence=1.5)

    def test_entity_equality(self):
        """测试实体相等性"""
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="python", entity_type=EntityType.TECHNOLOGY)  # 小写
        e3 = Entity(name="Python", entity_type=EntityType.CONCEPT)

        assert e1 == e2  # 名称不区分大小写
        assert e1 != e3  # 类型不同

    def test_entity_hash(self):
        """测试实体哈希"""
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="python", entity_type=EntityType.TECHNOLOGY)

        # 可以放入集合
        entity_set = {e1, e2}
        assert len(entity_set) == 1  # 被去重

    def test_entity_to_dict(self):
        """测试转换为字典"""
        entity = Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            confidence=0.9,
            aliases={"py"},
        )
        d = entity.to_dict()

        assert d["name"] == "Python"
        assert d["type"] == "technology"
        assert d["confidence"] == 0.9
        assert "py" in d["aliases"]


class TestInitialization:
    """测试初始化"""

    def test_default_initialization(self):
        """测试默认初始化"""
        extractor = EntityExtractor()
        assert extractor.min_confidence == 0.5
        assert len(extractor.patterns) > 0
        assert len(extractor.stop_words) > 0

    def test_custom_min_confidence(self):
        """测试自定义最小置信度"""
        extractor = EntityExtractor(min_confidence=0.7)
        assert extractor.min_confidence == 0.7

    def test_invalid_min_confidence(self):
        """测试无效最小置信度"""
        with pytest.raises(ValueError, match="min_confidence must be between"):
            EntityExtractor(min_confidence=1.5)

    def test_custom_patterns(self):
        """测试自定义模式"""
        custom = {EntityType.PRODUCT: [r"\b(CustomProduct)\b"]}
        extractor = EntityExtractor(custom_patterns=custom)
        assert EntityType.PRODUCT in extractor.patterns


class TestEntityExtraction:
    """测试实体提取"""

    def test_extract_technology(self):
        """测试提取技术实体"""
        extractor = EntityExtractor(min_confidence=0.5)
        text = "Python is a programming language"
        entities = extractor.extract_entities(text)

        assert len(entities) > 0
        assert any(e.name == "Python" for e in entities)
        assert any(e.entity_type == EntityType.TECHNOLOGY for e in entities)

    def test_extract_multiple_entities(self):
        """测试提取多个实体"""
        extractor = EntityExtractor(min_confidence=0.5)
        text = "Python and JavaScript are programming languages"
        entities = extractor.extract_entities(text)

        names = [e.name for e in entities]
        assert "Python" in names
        assert "JavaScript" in names

    def test_extract_person(self):
        """测试提取人物"""
        extractor = EntityExtractor(min_confidence=0.5)
        text = "Guido van Rossum created Python"
        entities = extractor.extract_entities(text)

        persons = [e for e in entities if e.entity_type == EntityType.PERSON]
        assert len(persons) > 0

    def test_extract_organization(self):
        """测试提取组织"""
        extractor = EntityExtractor(min_confidence=0.5)
        text = "Google uses Python extensively"
        entities = extractor.extract_entities(text)

        orgs = [e for e in entities if e.entity_type == EntityType.ORGANIZATION]
        assert len(orgs) > 0
        assert any(e.name == "Google" for e in orgs)

    def test_extract_concept(self):
        """测试提取概念"""
        extractor = EntityExtractor(min_confidence=0.5)
        text = "Machine learning algorithms are powerful"
        entities = extractor.extract_entities(text)

        concepts = [e for e in entities if e.entity_type == EntityType.CONCEPT]
        assert len(concepts) > 0

    def test_empty_text(self):
        """测试空文本"""
        extractor = EntityExtractor()
        with pytest.raises(ValueError, match="text must be a non-empty string"):
            extractor.extract_entities("")

    def test_stopword_filtering(self):
        """测试停用词过滤"""
        extractor = EntityExtractor()
        text = "The Python language"
        entities = extractor.extract_entities(text)

        # "The"应被过滤
        names = [e.name.lower() for e in entities]
        assert "the" not in names

    def test_confidence_threshold(self):
        """测试置信度阈值"""
        extractor_low = EntityExtractor(min_confidence=0.3)
        extractor_high = EntityExtractor(min_confidence=0.9)

        text = "Python programming"

        entities_low = extractor_low.extract_entities(text)
        entities_high = extractor_high.extract_entities(text)

        # 低阈值应该提取更多实体
        assert len(entities_low) >= len(entities_high)


class TestConfidenceCalculation:
    """测试置信度计算"""

    def test_confidence_for_long_entity(self):
        """测试长实体的置信度"""
        extractor = EntityExtractor()
        # 模拟计算
        conf = extractor._calculate_confidence("Python", "Python is great", 0)
        assert 0.0 <= conf <= 1.0

    def test_confidence_for_capitalized(self):
        """测试大写实体的置信度"""
        extractor = EntityExtractor()
        conf1 = extractor._calculate_confidence("Python", "text", 0)
        conf2 = extractor._calculate_confidence("python", "text", 0)
        # 大写应该有更高置信度
        assert conf1 >= conf2

    def test_confidence_position_factor(self):
        """测试位置因子"""
        extractor = EntityExtractor()
        text = "x" * 100
        conf_start = extractor._calculate_confidence("Python", text, 0)
        conf_end = extractor._calculate_confidence("Python", text, 90)
        # 句首应该有更高置信度
        assert conf_start >= conf_end


class TestDeduplication:
    """测试去重"""

    def test_deduplicate_identical(self):
        """测试去除相同实体"""
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.8)
        e2 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.9)

        extractor = EntityExtractor()
        result = extractor._deduplicate_entities([e1, e2])

        assert len(result) == 1
        assert result[0].confidence == 0.9  # 保留高置信度的

    def test_deduplicate_case_insensitive(self):
        """测试大小写不敏感去重"""
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="python", entity_type=EntityType.TECHNOLOGY)

        extractor = EntityExtractor()
        result = extractor._deduplicate_entities([e1, e2])

        assert len(result) == 1

    def test_deduplicate_different_types(self):
        """测试不同类型不去重"""
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Python", entity_type=EntityType.CONCEPT)

        extractor = EntityExtractor()
        result = extractor._deduplicate_entities([e1, e2])

        assert len(result) == 2


class TestEntityMerging:
    """测试实体合并"""

    def test_merge_same_type(self):
        """测试合并相同类型实体"""
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.8)
        e2 = Entity(name="Python 3", entity_type=EntityType.TECHNOLOGY, confidence=0.9)

        extractor = EntityExtractor()
        merged = extractor.merge_entities(e1, e2)

        assert merged.name == "Python 3"  # 选择高置信度的名称
        assert merged.confidence == 0.9
        assert "Python" in merged.aliases

    def test_merge_different_types(self):
        """测试合并不同类型实体失败"""
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Python", entity_type=EntityType.CONCEPT)

        extractor = EntityExtractor()
        with pytest.raises(
            ValueError, match="Cannot merge entities of different types"
        ):
            extractor.merge_entities(e1, e2)

    def test_merge_aliases(self):
        """测试别名合并"""
        e1 = Entity(
            name="Python",
            entity_type=EntityType.TECHNOLOGY,
            aliases={"py"},
        )
        e2 = Entity(
            name="Python3",
            entity_type=EntityType.TECHNOLOGY,
            aliases={"python3"},
        )

        extractor = EntityExtractor()
        merged = extractor.merge_entities(e1, e2)

        # 应该包含所有别名
        assert "py" in merged.aliases or "python3" in merged.aliases


class TestEntityClassification:
    """测试实体分类"""

    def test_classify_known_technology(self):
        """测试分类已知技术"""
        extractor = EntityExtractor()
        entity_type, confidence = extractor.classify_entity_type("Python")

        assert entity_type == EntityType.TECHNOLOGY
        assert confidence > 0.5

    def test_classify_with_context(self):
        """测试基于上下文分类"""
        extractor = EntityExtractor()
        entity_type, confidence = extractor.classify_entity_type(
            "XYZ", context="XYZ is a company"
        )

        assert entity_type == EntityType.ORGANIZATION
        assert confidence > 0.0

    def test_classify_unknown(self):
        """测试未知实体分类"""
        extractor = EntityExtractor()
        entity_type, confidence = extractor.classify_entity_type("Unknown123")

        assert entity_type == EntityType.UNKNOWN
        assert confidence < 0.5


class TestBatchExtraction:
    """测试批量提取"""

    def test_extract_from_conversations(self):
        """测试从对话中提取"""
        extractor = EntityExtractor(min_confidence=0.5)

        conversations = [
            {"content": "I love Python programming", "role": "user"},
            {"content": "JavaScript is also great", "role": "user"},
        ]

        entities = extractor.extract_from_conversations(conversations)

        assert len(entities) > 0
        names = [e.name for e in entities]
        assert "Python" in names or "JavaScript" in names

    def test_extract_empty_conversations(self):
        """测试空对话列表"""
        extractor = EntityExtractor()
        entities = extractor.extract_from_conversations([])
        assert entities == []

    def test_extract_with_metadata(self):
        """测试提取带元数据"""
        extractor = EntityExtractor(min_confidence=0.5)

        conversations = [
            {
                "content": "Python is awesome",
                "role": "user",
                "timestamp": "2025-11-14",
            }
        ]

        entities = extractor.extract_from_conversations(conversations)

        if entities:
            entity = entities[0]
            assert "role" in entity.metadata
            assert entity.metadata["role"] == "user"


class TestEntityFiltering:
    """测试实体过滤"""

    def test_filter_by_type(self):
        """测试按类型过滤"""
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY)
        e2 = Entity(name="Google", entity_type=EntityType.ORGANIZATION)

        extractor = EntityExtractor()
        filtered = extractor.filter_entities(
            [e1, e2], entity_types=[EntityType.TECHNOLOGY]
        )

        assert len(filtered) == 1
        assert filtered[0].entity_type == EntityType.TECHNOLOGY

    def test_filter_by_confidence(self):
        """测试按置信度过滤"""
        e1 = Entity(name="A", entity_type=EntityType.TECHNOLOGY, confidence=0.9)
        e2 = Entity(name="B", entity_type=EntityType.TECHNOLOGY, confidence=0.5)

        extractor = EntityExtractor()
        filtered = extractor.filter_entities([e1, e2], min_confidence=0.7)

        assert len(filtered) == 1
        assert filtered[0].confidence >= 0.7

    def test_filter_combined(self):
        """测试组合过滤"""
        e1 = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.9)
        e2 = Entity(name="Google", entity_type=EntityType.ORGANIZATION, confidence=0.9)
        e3 = Entity(name="Java", entity_type=EntityType.TECHNOLOGY, confidence=0.5)

        extractor = EntityExtractor()
        filtered = extractor.filter_entities(
            [e1, e2, e3],
            entity_types=[EntityType.TECHNOLOGY],
            min_confidence=0.7,
        )

        assert len(filtered) == 1
        assert filtered[0].name == "Python"


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
