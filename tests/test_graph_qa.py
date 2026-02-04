"""
跨会话记忆系统 - 图问答系统测试
Cross-Session Memory System - Graph QA Tests

测试覆盖:
1. 问题解析 (QuestionIntent)
2. 实体识别
3. 关系识别
4. 答案提取
5. 推理集成
6. 对话上下文
7. 统计信息
"""

import pytest
from graph_qa import (
    GraphQA,
    QuestionType,
    IntentType,
    QuestionIntent,
    Answer,
    GraphQAError,
)
from knowledge_graph import KnowledgeGraph
from knowledge_reasoner import KnowledgeReasoner
from entity_extractor import Entity, EntityType
from relationship_builder import Relationship, RelationshipType


class TestInitialization:
    """测试初始化"""

    def test_valid_initialization(self):
        """测试正确初始化"""
        kg = KnowledgeGraph()
        qa = GraphQA(kg)

        assert qa.kg == kg
        assert qa.reasoner is not None
        assert isinstance(qa.reasoner, KnowledgeReasoner)
        assert qa.context["history"] == []

    def test_invalid_knowledge_graph(self):
        """测试无效知识图谱"""
        with pytest.raises(GraphQAError):
            GraphQA("invalid")

    def test_custom_reasoner(self):
        """测试自定义推理器"""
        kg = KnowledgeGraph()
        reasoner = KnowledgeReasoner(kg)
        qa = GraphQA(kg, reasoner=reasoner)

        assert qa.reasoner == reasoner


class TestQuestionParsing:
    """测试问题解析"""

    @pytest.fixture
    def sample_qa(self):
        """创建示例问答系统"""
        kg = KnowledgeGraph()

        # 添加实体
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        django = Entity(name="Django", entity_type=EntityType.PRODUCT, confidence=0.9)

        kg.add_node(python)
        kg.add_node(django)

        # 添加关系
        kg.add_edge(
            Relationship(
                source="Python",
                target="Django",
                relation_type=RelationshipType.HAS_A,
                confidence=0.85,
            )
        )

        return GraphQA(kg)

    def test_boolean_question(self, sample_qa):
        """测试是非型问题"""
        intent = sample_qa.parse_question("Python能用于Web开发吗?")
        assert intent.question_type == QuestionType.BOOLEAN

    def test_list_question(self, sample_qa):
        """测试列表型问题"""
        intent = sample_qa.parse_question("Python有哪些框架?")
        assert intent.question_type == QuestionType.LIST

    def test_count_question(self, sample_qa):
        """测试计数型问题"""
        intent = sample_qa.parse_question("Python有多少个框架?")
        assert intent.question_type == QuestionType.COUNT

    def test_definition_question(self, sample_qa):
        """测试定义型问题"""
        intent = sample_qa.parse_question("什么是Python?")
        assert intent.question_type == QuestionType.DEFINITION

    def test_comparison_question(self, sample_qa):
        """测试比较型问题"""
        intent = sample_qa.parse_question("Python和Java的区别是什么?")
        assert intent.question_type == QuestionType.COMPARISON

    def test_reasoning_question(self, sample_qa):
        """测试推理型问题"""
        intent = sample_qa.parse_question("为什么Python适合AI开发?")
        assert intent.question_type == QuestionType.REASONING


class TestEntityExtraction:
    """测试实体提取"""

    @pytest.fixture
    def sample_qa(self):
        """创建示例问答系统"""
        kg = KnowledgeGraph()
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        django = Entity(name="Django", entity_type=EntityType.PRODUCT, confidence=0.9)
        kg.add_node(python)
        kg.add_node(django)
        return GraphQA(kg)

    def test_single_entity_extraction(self, sample_qa):
        """测试单实体提取"""
        intent = sample_qa.parse_question("Python有哪些框架?")
        assert "Python" in intent.entities

    def test_multiple_entity_extraction(self, sample_qa):
        """测试多实体提取"""
        intent = sample_qa.parse_question("Python和Django的关系是什么?")
        assert "Python" in intent.entities
        assert "Django" in intent.entities

    def test_no_entity_extraction(self, sample_qa):
        """测试无实体提取"""
        intent = sample_qa.parse_question("什么是编程语言?")
        assert len(intent.entities) == 0


class TestRelationExtraction:
    """测试关系提取"""

    @pytest.fixture
    def sample_qa(self):
        """创建示例问答系统"""
        kg = KnowledgeGraph()
        return GraphQA(kg)

    def test_has_relation(self, sample_qa):
        """测试has关系"""
        intent = sample_qa.parse_question("Python有什么框架?")
        assert any("has" in r for r in intent.relations)

    def test_uses_relation(self, sample_qa):
        """测试uses关系"""
        intent = sample_qa.parse_question("Python使用什么语法?")
        assert any("uses" in r for r in intent.relations)

    def test_related_relation(self, sample_qa):
        """测试related关系"""
        intent = sample_qa.parse_question("Python相关的技术有哪些?")
        assert any("related" in r for r in intent.relations)


class TestAnswerExtraction:
    """测试答案提取"""

    @pytest.fixture
    def qa_with_data(self):
        """创建包含数据的问答系统"""
        kg = KnowledgeGraph()

        # 添加实体
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        django = Entity(name="Django", entity_type=EntityType.PRODUCT, confidence=0.9)
        flask = Entity(name="Flask", entity_type=EntityType.PRODUCT, confidence=0.9)

        kg.add_node(python)
        kg.add_node(django)
        kg.add_node(flask)

        # 添加关系
        kg.add_edge(
            Relationship(
                source="Python",
                target="Django",
                relation_type=RelationshipType.HAS_A,
                confidence=0.85,
            )
        )
        kg.add_edge(
            Relationship(
                source="Python",
                target="Flask",
                relation_type=RelationshipType.HAS_A,
                confidence=0.8,
            )
        )

        return GraphQA(kg)

    def test_list_answer(self, qa_with_data):
        """测试列表型答案"""
        answer = qa_with_data.answer("Python有哪些框架?")
        assert "Django" in answer.answer or "Flask" in answer.answer
        assert answer.confidence > 0.5

    def test_count_answer(self, qa_with_data):
        """测试计数型答案"""
        answer = qa_with_data.answer("Python有多少个框架?")
        assert "2个" in answer.answer
        assert answer.confidence > 0.5

    def test_boolean_answer_yes(self, qa_with_data):
        """测试是非型答案(肯定)"""
        answer = qa_with_data.answer("Python有框架吗?")
        assert "是" in answer.answer
        assert answer.confidence > 0.5

    def test_boolean_answer_no(self):
        """测试是非型答案(否定)"""
        kg = KnowledgeGraph()
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        kg.add_node(python)
        qa = GraphQA(kg)

        answer = qa.answer("Python有编译器吗?")
        assert "不是" in answer.answer or "找不到" in answer.answer


class TestReasoningIntegration:
    """测试推理集成"""

    @pytest.fixture
    def qa_with_reasoning(self):
        """创建支持推理的问答系统"""
        kg = KnowledgeGraph()

        # 创建传递性关系链: A -> B -> C
        a = Entity(name="A", entity_type=EntityType.CONCEPT, confidence=0.9)
        b = Entity(name="B", entity_type=EntityType.CONCEPT, confidence=0.9)
        c = Entity(name="C", entity_type=EntityType.CONCEPT, confidence=0.9)

        kg.add_node(a)
        kg.add_node(b)
        kg.add_node(c)

        kg.add_edge(
            Relationship(
                source="A", target="B", relation_type=RelationshipType.RELATED_TO, confidence=0.8
            )
        )
        kg.add_edge(
            Relationship(
                source="B", target="C", relation_type=RelationshipType.RELATED_TO, confidence=0.7
            )
        )

        return GraphQA(kg)

    def test_reasoning_enabled(self, qa_with_reasoning):
        """测试启用推理"""
        answer = qa_with_reasoning.answer("为什么A和C相关?", use_reasoning=True)
        # 应该通过推理找到A->B->C的路径
        assert answer.reasoning_chain is not None or answer.confidence > 0

    def test_reasoning_disabled(self, qa_with_reasoning):
        """测试禁用推理"""
        answer = qa_with_reasoning.answer("为什么A和C相关?", use_reasoning=False)
        # 不使用推理,推理链应为空
        assert answer.reasoning_chain is None


class TestDialogueContext:
    """测试对话上下文"""

    @pytest.fixture
    def sample_qa(self):
        """创建示例问答系统"""
        kg = KnowledgeGraph()
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        kg.add_node(python)
        return GraphQA(kg)

    def test_context_update(self, sample_qa):
        """测试上下文更新"""
        sample_qa.answer("Python是什么?")
        assert "Python" in sample_qa.context["current_entities"]

    def test_history_tracking(self, sample_qa):
        """测试历史记录"""
        q1 = "Python是什么?"
        sample_qa.answer(q1)
        assert len(sample_qa.context["history"]) == 1
        assert sample_qa.context["history"][0]["question"] == q1

    def test_context_reset(self, sample_qa):
        """测试上下文重置"""
        sample_qa.answer("Python是什么?")
        sample_qa.reset_context()
        assert len(sample_qa.context["history"]) == 0
        assert len(sample_qa.context["current_entities"]) == 0


class TestStatistics:
    """测试统计信息"""

    @pytest.fixture
    def sample_qa(self):
        """创建示例问答系统"""
        kg = KnowledgeGraph()
        python = Entity(name="Python", entity_type=EntityType.TECHNOLOGY, confidence=0.95)
        kg.add_node(python)
        return GraphQA(kg)

    def test_statistics_tracking(self, sample_qa):
        """测试统计跟踪"""
        initial_stats = sample_qa.get_statistics()
        assert initial_stats["total_questions"] == 0

        sample_qa.answer("Python是什么?")

        updated_stats = sample_qa.get_statistics()
        assert updated_stats["total_questions"] == 1

    def test_success_rate(self, sample_qa):
        """测试成功率计算"""
        # 成功回答
        sample_qa.answer("Python是什么?")
        stats = sample_qa.get_statistics()
        assert stats["successful_answers"] >= 0


class TestAnswerObject:
    """测试答案对象"""

    def test_answer_to_dict(self):
        """测试答案序列化"""
        answer = Answer(
            question="测试问题",
            answer="测试答案",
            confidence=0.85,
            sources=[("A", "related_to", "B")],
        )

        answer_dict = answer.to_dict()
        assert answer_dict["question"] == "测试问题"
        assert answer_dict["answer"] == "测试答案"
        assert answer_dict["confidence"] == 0.85
        assert len(answer_dict["sources"]) == 1


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_graph_query(self):
        """测试空图查询"""
        kg = KnowledgeGraph()
        qa = GraphQA(kg)

        answer = qa.answer("Python是什么?")
        assert "找不到" in answer.answer or answer.confidence < 0.5

    def test_empty_question(self):
        """测试空问题"""
        kg = KnowledgeGraph()
        qa = GraphQA(kg)

        answer = qa.answer("")
        assert answer.confidence >= 0  # 应该不崩溃

    def test_malformed_question(self):
        """测试格式错误的问题"""
        kg = KnowledgeGraph()
        qa = GraphQA(kg)

        answer = qa.answer("???")
        assert answer.confidence >= 0  # 应该不崩溃


class TestIntentTypes:
    """测试意图类型"""

    @pytest.fixture
    def sample_qa(self):
        """创建示例问答系统"""
        kg = KnowledgeGraph()
        return GraphQA(kg)

    def test_query_intent(self, sample_qa):
        """测试查询意图"""
        intent = sample_qa.parse_question("Python是什么?")
        assert intent.intent_type == IntentType.QUERY

    def test_aggregate_intent(self, sample_qa):
        """测试聚合意图"""
        intent = sample_qa.parse_question("所有Python框架有哪些?")
        assert intent.intent_type == IntentType.AGGREGATE

    def test_clarify_intent(self, sample_qa):
        """测试澄清意图"""
        intent = sample_qa.parse_question("你是说Django吗?")
        assert intent.intent_type == IntentType.CLARIFY

    def test_navigate_intent(self, sample_qa):
        """测试导航意图"""
        intent = sample_qa.parse_question("更多关于Python的信息")
        assert intent.intent_type == IntentType.NAVIGATE


class TestConstraintExtraction:
    """测试约束提取"""

    @pytest.fixture
    def sample_qa(self):
        """创建示例问答系统"""
        kg = KnowledgeGraph()
        return GraphQA(kg)

    def test_time_constraint(self, sample_qa):
        """测试时间约束"""
        intent = sample_qa.parse_question("2023年Python有哪些更新?")
        assert "time" in intent.constraints

    def test_limit_constraint(self, sample_qa):
        """测试数量约束"""
        intent = sample_qa.parse_question("前5个Python框架是什么?")
        assert "limit" in intent.constraints


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
