"""
跨会话记忆系统 - 实体提取器
Cross-Session Memory System - Entity Extractor

版本: 1.0.0
日期: 2025-11-14
阶段: Phase 3 - 知识图谱构建

负责从对话内容中提取结构化实体,包括人物、地点、组织、概念等。
支持实体消歧、类型分类和置信度评分。
"""

import re
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EntityType(Enum):
    """实体类型枚举"""

    PERSON = "person"  # 人物
    LOCATION = "location"  # 地点
    ORGANIZATION = "organization"  # 组织
    CONCEPT = "concept"  # 概念
    TECHNOLOGY = "technology"  # 技术
    PRODUCT = "product"  # 产品
    EVENT = "event"  # 事件
    UNKNOWN = "unknown"  # 未知


class EntityExtractorError(Exception):
    """实体提取器异常"""


@dataclass
class Entity:
    """实体数据类

    Attributes:
        name: 实体名称
        entity_type: 实体类型
        confidence: 置信度 (0.0-1.0)
        context: 原始上下文
        aliases: 别名集合
        metadata: 附加元数据
    """

    name: str
    entity_type: EntityType
    confidence: float = 1.0
    context: str = ""
    aliases: Set[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.aliases is None:
            self.aliases = set()
        if self.metadata is None:
            self.metadata = {}
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

    def __hash__(self):
        """支持集合操作"""
        return hash((self.name.lower(), self.entity_type))

    def __eq__(self, other):
        """相等比较"""
        if not isinstance(other, Entity):
            return False
        return (
            self.name.lower() == other.name.lower()
            and self.entity_type == other.entity_type
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.entity_type.value,
            "confidence": self.confidence,
            "context": self.context,
            "aliases": list(self.aliases),
            "metadata": self.metadata,
        }


class EntityExtractor:
    """实体提取器

    基于规则和模式匹配的实体识别系统。

    Attributes:
        patterns: 实体识别模式字典
        stop_words: 停用词集合
        min_confidence: 最小置信度阈值

    Example:
        >>> extractor = EntityExtractor()
        >>> text = "Python is a programming language created by Guido"
        >>> entities = extractor.extract_entities(text)
        >>> print(entities[0].name)
        Python
    """

    def __init__(
        self, min_confidence: float = 0.5, custom_patterns: Optional[Dict] = None
    ):
        """初始化提取器

        Args:
            min_confidence: 最小置信度阈值
            custom_patterns: 自定义提取模式

        Raises:
            ValueError: 参数无效
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

        self.min_confidence = min_confidence
        self.patterns = self._init_patterns()
        if custom_patterns:
            self.patterns.update(custom_patterns)

        self.stop_words = self._init_stop_words()

        logger.info(
            "✅ EntityExtractor initialized (min_confidence=%.2f)", min_confidence
        )

    def _init_patterns(self) -> Dict[EntityType, List[str]]:
        """初始化实体识别模式

        Returns:
            实体类型到正则模式的映射
        """
        return {
            EntityType.TECHNOLOGY: [
                r"\b(Python|Java|JavaScript|TypeScript|C\+\+|Go|Rust|Ruby)\b",
                r"\b(React|Vue|Angular|Django|Flask|FastAPI|Express)\b",
                r"\b(TensorFlow|PyTorch|Keras|scikit-learn|NumPy|pandas)\b",
                r"\b(Docker|Kubernetes|AWS|Azure|GCP|Linux|Windows)\b",
                r"\b(SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Redis)\b",
                r"\b(Git|GitHub|GitLab|CI/CD|Jenkins)\b",
            ],
            EntityType.CONCEPT: [
                r"\b(machine learning|deep learning|AI|artificial intelligence)\b",
                r"\b(neural network|algorithm|data structure|design pattern)\b",
                r"\b(microservices|containerization|virtualization)\b",
                r"\b(REST API|GraphQL|WebSocket|HTTP)\b",
                r"\b(agile|scrum|DevOps|continuous integration)\b",
            ],
            EntityType.PERSON: [
                r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",  # 名字格式: John Doe
                r"\b(Guido van Rossum|Linus Torvalds|Elon Musk)\b",  # 知名人物
            ],
            EntityType.ORGANIZATION: [
                r"\b(Google|Microsoft|Apple|Amazon|Meta|Facebook)\b",
                r"\b(OpenAI|DeepMind|Anthropic)\b",
                r"\b(MIT|Stanford|Harvard|Berkeley)\b",
            ],
            EntityType.PRODUCT: [
                r"\b(ChatGPT|GPT-4|Claude|Gemini|LLaMA)\b",
                r"\b(iPhone|Android|Windows|MacOS)\b",
                r"\b(VS Code|PyCharm|IntelliJ|Eclipse)\b",
            ],
        }

    def _init_stop_words(self) -> Set[str]:
        """初始化停用词"""
        return {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "this",
            "that",
        }

    def extract_entities(self, text: str, context: str = "") -> List[Entity]:
        """从文本中提取实体

        Args:
            text: 输入文本
            context: 上下文信息

        Returns:
            实体列表

        Raises:
            ValueError: 文本为空
            EntityExtractorError: 提取失败
        """
        if not text or not isinstance(text, str):
            raise ValueError("text must be a non-empty string")

        try:
            entities = []

            # 对每种实体类型应用模式匹配
            for entity_type, patterns in self.patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity_name = match.group(0)

                        # 过滤停用词
                        if entity_name.lower() in self.stop_words:
                            continue

                        # 计算置信度(基于匹配长度和位置)
                        confidence = self._calculate_confidence(
                            entity_name, text, match.start()
                        )

                        if confidence >= self.min_confidence:
                            entity = Entity(
                                name=entity_name,
                                entity_type=entity_type,
                                confidence=confidence,
                                context=context
                                or text[max(0, match.start() - 20) : match.end() + 20],
                            )
                            entities.append(entity)

            # 去重
            entities = self._deduplicate_entities(entities)

            logger.info("✅ Extracted %d entities from text", len(entities))
            return entities

        except Exception as exc:
            logger.error("❌ Failed to extract entities: %s", exc)
            raise EntityExtractorError(f"Failed to extract entities: {exc}") from exc

    def _calculate_confidence(
        self, entity_name: str, text: str, position: int
    ) -> float:
        """计算实体置信度

        基于多种因素:
        - 实体长度(更长更可信)
        - 大写字母(专有名词)
        - 位置(句首更可信)

        Args:
            entity_name: 实体名称
            text: 原文本
            position: 实体位置

        Returns:
            置信度分数 (0.0-1.0)
        """
        confidence = 0.7  # 基础分

        # 长度因子(3-15字符最优)
        length = len(entity_name)
        if 3 <= length <= 15:
            confidence += 0.1
        elif length > 15:
            confidence += 0.05

        # 大写因子(专有名词)
        if entity_name[0].isupper():
            confidence += 0.1

        # 位置因子(句首更重要)
        if position < len(text) * 0.2:
            confidence += 0.1

        return min(1.0, confidence)

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """实体去重

        相同名称和类型的实体保留置信度最高的。

        Args:
            entities: 实体列表

        Returns:
            去重后的实体列表
        """
        entity_map = {}

        for entity in entities:
            key = (entity.name.lower(), entity.entity_type)
            if key not in entity_map or entity.confidence > entity_map[key].confidence:
                entity_map[key] = entity

        return list(entity_map.values())

    def merge_entities(self, entity1: Entity, entity2: Entity) -> Entity:
        """合并两个相似实体

        Args:
            entity1: 实体1
            entity2: 实体2

        Returns:
            合并后的实体

        Raises:
            ValueError: 实体类型不匹配
        """
        if entity1.entity_type != entity2.entity_type:
            raise ValueError("Cannot merge entities of different types")

        # 选择置信度更高的作为主实体
        if entity1.confidence >= entity2.confidence:
            primary, secondary = entity1, entity2
        else:
            primary, secondary = entity2, entity1

        # 合并别名
        merged_aliases = primary.aliases | secondary.aliases | {secondary.name}

        # 合并元数据
        merged_metadata = {**secondary.metadata, **primary.metadata}

        return Entity(
            name=primary.name,
            entity_type=primary.entity_type,
            confidence=max(primary.confidence, secondary.confidence),
            context=primary.context or secondary.context,
            aliases=merged_aliases,
            metadata=merged_metadata,
        )

    def classify_entity_type(
        self, entity_name: str, context: str = ""
    ) -> Tuple[EntityType, float]:
        """分类实体类型

        Args:
            entity_name: 实体名称
            context: 上下文

        Returns:
            (实体类型, 置信度)
        """
        # 尝试每种模式
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, entity_name, re.IGNORECASE):
                    return entity_type, 0.9

        # 基于上下文推断
        if context:
            context_lower = context.lower()
            if any(
                word in context_lower for word in ["person", "people", "human", "name"]
            ):
                return EntityType.PERSON, 0.6
            elif any(
                word in context_lower for word in ["company", "organization", "corp"]
            ):
                return EntityType.ORGANIZATION, 0.6
            elif any(word in context_lower for word in ["place", "location", "city"]):
                return EntityType.LOCATION, 0.6

        return EntityType.UNKNOWN, 0.3

    def extract_from_conversations(
        self, conversations: List[Dict[str, Any]]
    ) -> List[Entity]:
        """从对话列表中批量提取实体

        Args:
            conversations: 对话消息列表

        Returns:
            提取的所有实体(已去重)

        Raises:
            EntityExtractorError: 提取失败
        """
        if not conversations:
            return []

        try:
            all_entities = []

            for conv in conversations:
                content = conv.get("content", "")
                role = conv.get("role", "")
                timestamp = conv.get("timestamp", "")

                if content:
                    entities = self.extract_entities(content, context=content)

                    # 添加对话元数据
                    for entity in entities:
                        entity.metadata.update(
                            {
                                "role": role,
                                "timestamp": timestamp,
                                "source": "conversation",
                            }
                        )

                    all_entities.extend(entities)

            # 全局去重
            all_entities = self._deduplicate_entities(all_entities)

            logger.info(
                "✅ Extracted %d entities from %d conversations",
                len(all_entities),
                len(conversations),
            )

            return all_entities

        except Exception as exc:
            logger.error("❌ Failed to extract from conversations: %s", exc)
            raise EntityExtractorError(
                f"Failed to extract from conversations: {exc}"
            ) from exc

    def filter_entities(
        self,
        entities: List[Entity],
        entity_types: Optional[List[EntityType]] = None,
        min_confidence: Optional[float] = None,
    ) -> List[Entity]:
        """过滤实体

        Args:
            entities: 实体列表
            entity_types: 保留的实体类型
            min_confidence: 最小置信度

        Returns:
            过滤后的实体列表
        """
        filtered = entities

        if entity_types:
            filtered = [e for e in filtered if e.entity_type in entity_types]

        if min_confidence is not None:
            filtered = [e for e in filtered if e.confidence >= min_confidence]

        return filtered


# 示例演示
if __name__ == "__main__":
    # 创建提取器
    extractor = EntityExtractor(min_confidence=0.6)

    # 测试文本
    text = """
    Python is a programming language created by Guido van Rossum.
    It's widely used for machine learning with TensorFlow and PyTorch.
    Companies like Google and OpenAI use it extensively.
    """

    # 提取实体
    entities = extractor.extract_entities(text)

    print(f"\n=== Extracted {len(entities)} entities ===")
    for entity in entities:
        print(f"\n{entity.name}")
        print(f"  Type: {entity.entity_type.value}")
        print(f"  Confidence: {entity.confidence:.2f}")

    print("\n✅ Demo completed!")
