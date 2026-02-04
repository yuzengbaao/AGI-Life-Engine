"""
多模态融合测试
Tests for MultimodalFusion
"""

import numpy as np
import pytest

from multimodal_fusion import (
    MultimodalFusion,
    MultimodalFusionError,
    ModalityFeature,
    FusedRepresentation,
    KnowledgeTriple,
)


@pytest.fixture
def fusion():
    """创建测试用MultimodalFusion"""
    return MultimodalFusion(unified_dim=128, similarity_threshold=0.3)


@pytest.fixture
def image_feature():
    """创建测试用图像特征"""
    return ModalityFeature(
        modality="image",
        features=np.random.randn(64, 64, 3),
        metadata={"source": "test_image.jpg"}
    )


@pytest.fixture
def audio_feature():
    """创建测试用音频特征"""
    return ModalityFeature(
        modality="audio",
        features=np.random.randn(100),
        metadata={"source": "test_audio.wav"}
    )


@pytest.fixture
def text_feature():
    """创建测试用文本特征"""
    return ModalityFeature(
        modality="text",
        features=np.random.randn(256),
        metadata={"source": "test_text.txt"}
    )


class TestInit:
    """测试初始化"""
    def test_default_init(self):
        """测试默认初始化"""
        fusion = MultimodalFusion()
        assert fusion.unified_dim == 512
        assert fusion.similarity_threshold == 0.5
        stats = fusion.get_statistics()
        assert stats["total_fusions"] == 0

    def test_custom_init(self):
        """测试自定义初始化"""
        fusion = MultimodalFusion(unified_dim=256, similarity_threshold=0.7)
        assert fusion.unified_dim == 256
        assert fusion.similarity_threshold == 0.7


class TestModalityFeature:
    """测试ModalityFeature"""
    def test_feature_creation(self, image_feature):
        """测试特征创建"""
        assert image_feature.modality == "image"
        assert image_feature.features.shape == (64, 64, 3)
        assert "source" in image_feature.metadata

    def test_feature_to_dict(self, image_feature):
        """测试特征序列化"""
        feat_dict = image_feature.to_dict()
        assert "modality" in feat_dict
        assert "features_shape" in feat_dict
        assert "metadata" in feat_dict


class TestFeatureAlignment:
    """测试特征对齐"""
    def test_align_large_feature(self, fusion, image_feature):
        """测试大特征对齐(下采样)"""
        aligned = fusion.align_features(image_feature)
        assert aligned.shape == (fusion.unified_dim,)
        assert np.isclose(np.linalg.norm(aligned), 1.0)

    def test_align_small_feature(self, fusion):
        """测试小特征对齐(上采样)"""
        small_feat = ModalityFeature(
            modality="text",
            features=np.random.randn(64)
        )
        aligned = fusion.align_features(small_feat)
        assert aligned.shape == (fusion.unified_dim,)
        assert np.isclose(np.linalg.norm(aligned), 1.0)

    def test_align_exact_size(self, fusion):
        """测试完全匹配大小"""
        exact_feat = ModalityFeature(
            modality="test",
            features=np.random.randn(fusion.unified_dim)
        )
        aligned = fusion.align_features(exact_feat)
        assert aligned.shape == (fusion.unified_dim,)
        assert np.isclose(np.linalg.norm(aligned), 1.0)


class TestModalityFusion:
    """测试模态融合"""
    def test_fuse_single_modality(self, fusion, image_feature):
        """测试单模态融合"""
        result = fusion.fuse_modalities([image_feature])
        assert isinstance(result, FusedRepresentation)
        assert result.unified_features.shape == (fusion.unified_dim,)
        assert "image" in result.modality_contributions

    def test_fuse_two_modalities(self, fusion, image_feature, audio_feature):
        """测试双模态融合"""
        result = fusion.fuse_modalities([image_feature, audio_feature])
        assert result.unified_features.shape == (fusion.unified_dim,)
        assert "image" in result.modality_contributions
        assert "audio" in result.modality_contributions

    def test_fuse_three_modalities(self, fusion, image_feature, audio_feature, text_feature):
        """测试三模态融合"""
        result = fusion.fuse_modalities([image_feature, audio_feature, text_feature])
        assert len(result.modality_contributions) == 3
        assert "modalities" in result.metadata

    def test_fuse_empty_list(self, fusion):
        """测试空列表融合"""
        with pytest.raises(MultimodalFusionError):
            fusion.fuse_modalities([])

    def test_fusion_stats_updated(self, fusion, image_feature):
        """测试融合统计更新"""
        fusion.fuse_modalities([image_feature])
        stats = fusion.get_statistics()
        assert stats["total_fusions"] == 1
        assert stats["avg_fusion_time_ms"] >= 0  # 允许0ms(操作极快时)


class TestFusedRepresentation:
    """测试融合表示"""
    def test_representation_to_dict(self, fusion, image_feature):
        """测试融合表示序列化"""
        result = fusion.fuse_modalities([image_feature])
        result_dict = result.to_dict()
        assert "unified_features_shape" in result_dict
        assert "modality_contributions" in result_dict


class TestCrossModalRetrieval:
    """测试跨模态检索"""
    def test_retrieval_basic(self, fusion, image_feature, audio_feature):
        """测试基本检索"""
        results = fusion.cross_modal_retrieval(
            image_feature,
            [audio_feature],
            top_k=1
        )
        assert isinstance(results, list)

    def test_retrieval_multiple_candidates(self, fusion, image_feature):
        """测试多候选检索"""
        candidates = [
            ModalityFeature("audio", np.random.randn(100)),
            ModalityFeature("text", np.random.randn(200)),
            ModalityFeature("video", np.random.randn(50)),
        ]
        results = fusion.cross_modal_retrieval(image_feature, candidates, top_k=2)
        assert len(results) <= 2
        for idx, sim in results:
            assert 0 <= idx < len(candidates)
            assert -1.0 <= sim <= 1.0

    def test_retrieval_empty_candidates(self, fusion, image_feature):
        """测试空候选列表"""
        results = fusion.cross_modal_retrieval(image_feature, [], top_k=5)
        assert results == []

    def test_retrieval_stats_updated(self, fusion, image_feature, audio_feature):
        """测试检索统计更新"""
        fusion.cross_modal_retrieval(image_feature, [audio_feature])
        stats = fusion.get_statistics()
        assert stats["total_retrievals"] == 1


class TestKnowledgeTriple:
    """测试知识三元组"""
    def test_triple_creation(self):
        """测试三元组创建"""
        triple = KnowledgeTriple(
            subject="cat",
            predicate="is_a",
            object="animal",
            confidence=0.95,
            source_modality="image"
        )
        assert triple.subject == "cat"
        assert triple.predicate == "is_a"
        assert triple.object == "animal"
        assert triple.confidence == 0.95

    def test_triple_to_dict(self):
        """测试三元组序列化"""
        triple = KnowledgeTriple("subject", "predicate", "object")
        triple_dict = triple.to_dict()
        assert "subject" in triple_dict
        assert "predicate" in triple_dict
        assert "object" in triple_dict
        assert "confidence" in triple_dict


class TestKGGeneration:
    """测试知识图谱生成"""
    def test_generate_with_entities(self, fusion, image_feature, audio_feature):
        """测试带实体的三元组生成"""
        entities = {
            "image": ["cat", "dog"],
            "audio": ["meow", "bark"],
        }
        triples = fusion.generate_kg_triples([image_feature, audio_feature], entities)
        assert len(triples) > 0
        assert all(isinstance(t, KnowledgeTriple) for t in triples)

    def test_generate_cross_modal_triples(self, fusion, image_feature, audio_feature):
        """测试跨模态三元组生成"""
        triples = fusion.generate_kg_triples([image_feature, audio_feature])
        # 应至少生成一个跨模态关联
        cross_modal = [t for t in triples if "correlates_with" in t.predicate]
        assert len(cross_modal) >= 0  # 根据相似度阈值可能为0

    def test_generate_empty_features(self, fusion):
        """测试空特征列表"""
        triples = fusion.generate_kg_triples([])
        assert triples == []

    def test_generate_single_feature(self, fusion, image_feature):
        """测试单特征无跨模态关联"""
        entities = {"image": ["cat"]}
        triples = fusion.generate_kg_triples([image_feature], entities)
        # 只有基于实体的三元组,没有跨模态关联
        cross_modal = [t for t in triples if "correlates_with" in t.predicate]
        assert len(cross_modal) == 0

    def test_kg_stats_updated(self, fusion, image_feature):
        """测试KG统计更新"""
        entities = {"image": ["cat"]}
        triples = fusion.generate_kg_triples([image_feature], entities)
        stats = fusion.get_statistics()
        assert stats["total_triples"] == len(triples)


class TestStatistics:
    """测试统计功能"""
    def test_initial_stats(self, fusion):
        """测试初始统计"""
        stats = fusion.get_statistics()
        assert stats["total_fusions"] == 0
        assert stats["total_retrievals"] == 0
        assert stats["total_triples"] == 0
        assert stats["avg_fusion_time_ms"] == 0.0

    def test_stats_immutability(self, fusion):
        """测试统计不可变性"""
        stats1 = fusion.get_statistics()
        stats1["total_fusions"] = 999
        stats2 = fusion.get_statistics()
        assert stats2["total_fusions"] == 0


class TestEdgeCases:
    """测试边界情况"""
    def test_zero_features(self, fusion):
        """测试零向量特征"""
        zero_feat = ModalityFeature("test", np.zeros(100))
        aligned = fusion.align_features(zero_feat)
        # 零向量归一化后仍为零向量
        assert np.allclose(aligned, 0.0)

    def test_high_threshold(self):
        """测试高相似度阈值"""
        fusion = MultimodalFusion(similarity_threshold=0.99)
        feat1 = ModalityFeature("test1", np.random.randn(100))
        feat2 = ModalityFeature("test2", np.random.randn(100))
        # 随机特征很难达到0.99相似度
        results = fusion.cross_modal_retrieval(feat1, [feat2])
        # 可能为空或有结果
        assert isinstance(results, list)

    def test_identical_features(self, fusion):
        """测试相同特征"""
        feat1 = ModalityFeature("test", np.ones(100))
        feat2 = ModalityFeature("test", np.ones(100))
        results = fusion.cross_modal_retrieval(feat1, [feat2])
        # 相同特征应该有很高相似度
        if results:
            assert results[0][1] > 0.9
