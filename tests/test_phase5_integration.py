"""
Phase 5 集成测试
Integration Tests for Phase 5 Multimodal Understanding
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

# Import all processors
from image_processor import ImageProcessor, ImageMetadata
from audio_processor import AudioProcessor, ASREngine
from video_processor import VideoProcessor
from multimodal_fusion import (
    MultimodalFusion,
    ModalityFeature,
    KnowledgeTriple,
)


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def image_processor():
    """创建ImageProcessor"""
    return ImageProcessor()


@pytest.fixture
def audio_processor():
    """创建AudioProcessor"""
    return AudioProcessor(asr_engine=ASREngine.NONE)


@pytest.fixture
def video_processor():
    """创建VideoProcessor"""
    return VideoProcessor(hist_bins=16, scene_threshold=0.4)


@pytest.fixture
def fusion():
    """创建MultimodalFusion"""
    return MultimodalFusion(unified_dim=256, similarity_threshold=0.3)


@pytest.fixture
def sample_image(temp_dir):
    """生成测试图像"""
    pytest.importorskip('PIL')
    from PIL import Image
    img_path = temp_dir / "test_image.jpg"
    img = Image.new('RGB', (320, 240), color=(128, 128, 128))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def sample_audio(temp_dir):
    """生成测试音频"""
    sf = pytest.importorskip('soundfile')
    audio_path = temp_dir / "test_audio.wav"
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    audio_data = np.sin(2 * np.pi * 440 * np.arange(samples) / sample_rate)
    sf.write(audio_path, audio_data, sample_rate)
    return str(audio_path)


@pytest.fixture
def sample_video(temp_dir):
    """生成测试视频"""
    cv2 = pytest.importorskip('cv2')
    video_path = temp_dir / "test_video.avi"
    width, height, fps, frames = 64, 48, 10, 10
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # pylint: disable=no-member
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))  # pylint: disable=no-member
    for i in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (i * 25, 128, 255 - i * 25)
        writer.write(frame)
    writer.release()
    return str(video_path)


class TestImageProcessingIntegration:
    """测试图像处理集成"""

    def test_image_to_fusion_pipeline(self, image_processor, fusion, sample_image):
        """测试图像处理到融合的完整流程"""
        # 处理图像
        result = image_processor.process(sample_image, enable_ocr=False)
        assert result.metadata.width > 0
        
        # 创建特征并融合
        feature = ModalityFeature(
            modality="image",
            features=np.random.randn(128),
            metadata={"source": sample_image}
        )
        fused = fusion.fuse_modalities([feature])
        assert fused.unified_features.shape == (fusion.unified_dim,)
        assert "image" in fused.modality_contributions

    def test_image_metadata_extraction(self, image_processor, sample_image):
        """测试图像元数据提取"""
        result = image_processor.process(sample_image, enable_ocr=False)
        meta = result.metadata
        assert isinstance(meta, ImageMetadata)
        assert meta.width == 320
        assert meta.height == 240
        assert meta.mode == 'RGB'  # RGB has 3 channels


class TestAudioProcessingIntegration:
    """测试音频处理集成"""

    def test_audio_to_fusion_pipeline(self, audio_processor, fusion, sample_audio):
        """测试音频处理到融合的完整流程"""
        # 处理音频
        result = audio_processor.process(sample_audio, enable_asr=False, extract_features=True)
        assert result.features is not None
        
        # 创建特征并融合
        feature = ModalityFeature(
            modality="audio",
            features=result.features.mfcc,
            metadata={"source": sample_audio}
        )
        fused = fusion.fuse_modalities([feature])
        assert "audio" in fused.modality_contributions

    def test_audio_feature_extraction(self, audio_processor, sample_audio):
        """测试音频特征提取"""
        result = audio_processor.process(sample_audio, enable_asr=False, extract_features=True)
        features = result.features
        assert features.mfcc.shape[0] > 0
        assert features.spectral_centroid.shape[0] > 0
        assert features.tempo > 0


class TestVideoProcessingIntegration:
    """测试视频处理集成"""

    def test_video_to_fusion_pipeline(self, video_processor, fusion, sample_video):
        """测试视频处理到融合的完整流程"""
        # 处理视频
        result = video_processor.process(sample_video, enable_keyframes=True, enable_scenes=True)
        assert len(result.keyframes) > 0
        
        # 创建特征并融合
        feature = ModalityFeature(
            modality="video",
            features=np.array(result.keyframes, dtype=float),
            metadata={"source": sample_video}
        )
        fused = fusion.fuse_modalities([feature])
        assert "video" in fused.modality_contributions

    def test_video_keyframe_extraction(self, video_processor, sample_video):
        """测试视频关键帧提取"""
        result = video_processor.process(sample_video)
        assert len(result.keyframes) >= 1
        assert 0 in result.keyframes


class TestMultimodalFusionIntegration:
    """测试多模态融合集成"""

    def test_image_audio_fusion(self, image_processor, audio_processor, fusion, sample_image, sample_audio):
        """测试图像+音频融合"""
        # 处理各模态
        img_result = image_processor.process(sample_image, enable_ocr=False)
        audio_result = audio_processor.process(sample_audio, enable_asr=False, extract_features=True)
        
        # 创建特征
        features = [
            ModalityFeature("image", np.random.randn(100), {"source": sample_image}),
            ModalityFeature("audio", audio_result.features.mfcc, {"source": sample_audio}),
        ]
        
        # 融合
        fused = fusion.fuse_modalities(features)
        assert len(fused.modality_contributions) == 2
        assert "image" in fused.modality_contributions
        assert "audio" in fused.modality_contributions

    def test_image_video_fusion(self, image_processor, video_processor, fusion, sample_image, sample_video):
        """测试图像+视频融合"""
        # 处理各模态
        img_result = image_processor.process(sample_image, enable_ocr=False)
        video_result = video_processor.process(sample_video)
        
        # 创建特征
        features = [
            ModalityFeature("image", np.random.randn(100)),
            ModalityFeature("video", np.array(video_result.keyframes, dtype=float)),
        ]
        
        # 融合
        fused = fusion.fuse_modalities(features)
        assert len(fused.modality_contributions) == 2

    def test_three_modality_fusion(self, image_processor, audio_processor, video_processor,
                                    fusion, sample_image, sample_audio, sample_video):
        """测试三模态融合"""
        # 处理各模态
        image_processor.process(sample_image, enable_ocr=False)
        audio_processor.process(sample_audio, enable_asr=False, extract_features=True)
        video_processor.process(sample_video)
        
        # 创建特征
        features = [
            ModalityFeature("image", np.random.randn(100)),
            ModalityFeature("audio", np.random.randn(100)),
            ModalityFeature("video", np.random.randn(100)),
        ]
        
        # 融合
        fused = fusion.fuse_modalities(features)
        assert len(fused.modality_contributions) == 3


class TestCrossModalRetrieval:
    """测试跨模态检索"""

    def test_image_to_audio_retrieval(self, fusion):
        """测试图像到音频检索"""
        query = ModalityFeature("image", np.random.randn(100))
        candidates = [
            ModalityFeature("audio", np.random.randn(100)),
            ModalityFeature("audio", np.random.randn(100)),
            ModalityFeature("audio", np.random.randn(100)),
        ]
        
        results = fusion.cross_modal_retrieval(query, candidates, top_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_video_to_image_retrieval(self, fusion):
        """测试视频到图像检索"""
        query = ModalityFeature("video", np.random.randn(150))
        candidates = [
            ModalityFeature("image", np.random.randn(200)),
            ModalityFeature("image", np.random.randn(200)),
        ]
        
        results = fusion.cross_modal_retrieval(query, candidates, top_k=1)
        assert isinstance(results, list)


class TestKnowledgeGraphGeneration:
    """测试知识图谱生成"""

    def test_multimodal_kg_generation(self, fusion):
        """测试多模态知识图谱生成"""
        features = [
            ModalityFeature("image", np.random.randn(100)),
            ModalityFeature("audio", np.random.randn(100)),
        ]
        
        entities = {
            "image": ["cat", "dog"],
            "audio": ["meow", "bark"],
        }
        
        triples = fusion.generate_kg_triples(features, entities)
        assert len(triples) > 0
        assert all(isinstance(t, KnowledgeTriple) for t in triples)

    def test_kg_cross_modal_links(self, fusion):
        """测试跨模态知识链接"""
        # 创建相似特征以触发跨模态关联
        base_feat = np.random.randn(100)
        features = [
            ModalityFeature("image", base_feat + np.random.randn(100) * 0.1),
            ModalityFeature("audio", base_feat + np.random.randn(100) * 0.1),
        ]
        
        triples = fusion.generate_kg_triples(features)
        # 可能生成跨模态关联
        assert isinstance(triples, list)


class TestEndToEndWorkflow:
    """测试端到端工作流"""

    def test_full_multimodal_pipeline(self, image_processor, audio_processor, video_processor,
                                       fusion, sample_image, sample_audio, sample_video):
        """测试完整的多模态处理流程"""
        # Step 1: 处理各模态
        img_result = image_processor.process(sample_image, enable_ocr=False)
        audio_result = audio_processor.process(sample_audio, enable_asr=False, extract_features=True)
        video_result = video_processor.process(sample_video)
        
        # 验证各模态结果
        assert img_result.metadata.width > 0
        assert audio_result.metadata.duration_seconds > 0
        assert video_result.metadata.frame_count > 0
        
        # Step 2: 创建特征
        features = [
            ModalityFeature("image", np.random.randn(100)),
            ModalityFeature("audio", audio_result.features.mfcc),
            ModalityFeature("video", np.array(video_result.keyframes, dtype=float)),
        ]
        
        # Step 3: 融合
        fused = fusion.fuse_modalities(features)
        assert fused.unified_features.shape == (fusion.unified_dim,)
        assert len(fused.modality_contributions) == 3
        
        # Step 4: 生成知识图谱
        triples = fusion.generate_kg_triples(features)
        assert isinstance(triples, list)
        
        # Step 5: 检索测试
        query = features[0]
        results = fusion.cross_modal_retrieval(query, features[1:], top_k=2)
        assert isinstance(results, list)

    def test_statistics_tracking(self, image_processor, audio_processor, video_processor, fusion):
        """测试统计追踪"""
        # 获取初始统计
        img_stats = image_processor.get_statistics()
        audio_stats = audio_processor.get_statistics()
        video_stats = video_processor.get_statistics()
        fusion_stats = fusion.get_statistics()
        
        # 验证统计结构
        assert "total_processed" in img_stats
        assert "total_processed" in audio_stats
        assert "total_processed" in video_stats
        assert "total_fusions" in fusion_stats
        assert "total_retrievals" in fusion_stats
        assert "total_triples" in fusion_stats


class TestErrorHandling:
    """测试错误处理"""

    def test_invalid_image_path(self, image_processor):
        """测试无效图像路径"""
        with pytest.raises(Exception):
            image_processor.process("nonexistent.jpg")

    def test_invalid_audio_path(self, audio_processor):
        """测试无效音频路径"""
        with pytest.raises(Exception):
            audio_processor.process("nonexistent.wav")

    def test_invalid_video_path(self, video_processor):
        """测试无效视频路径"""
        with pytest.raises(Exception):
            video_processor.process("nonexistent.mp4")

    def test_empty_fusion(self, fusion):
        """测试空融合"""
        from multimodal_fusion import MultimodalFusionError
        with pytest.raises(MultimodalFusionError):
            fusion.fuse_modalities([])
