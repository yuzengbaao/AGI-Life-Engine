"""
音频处理器测试
Tests for AudioProcessor
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np

from audio_processor import (
    AudioProcessor,
    ASREngine,
    AudioFormat,
    AudioMetadata,
    ASRResult,
    AudioFeatures,
    AudioAnalysisResult,
    AudioProcessorError
)


@pytest.fixture
def audio_processor():
    """创建默认AudioProcessor实例"""
    return AudioProcessor(asr_engine=ASREngine.WHISPER)


@pytest.fixture
def test_audio_path():
    """创建测试音频文件(1秒440Hz正弦波)"""
    import librosa
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = f.name

    # 生成440Hz正弦波(A4音符)
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 保存为WAV
    sf.write(temp_path, y, sr)

    yield temp_path

    # 清理
    try:
        Path(temp_path).unlink()
    except:
        pass


@pytest.fixture
def mock_audio_data():
    """模拟音频数据"""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    return y, sr


class TestAudioProcessorInit:
    """测试AudioProcessor初始化"""

    def test_default_initialization(self):
        """测试默认初始化"""
        processor = AudioProcessor()

        assert processor.asr_engine == ASREngine.WHISPER
        assert processor.sample_rate == 16000
        assert processor.entity_extractor is None

    def test_custom_initialization(self):
        """测试自定义初始化"""
        processor = AudioProcessor(
            asr_engine=ASREngine.VOSK,
            sample_rate=22050
        )

        assert processor.asr_engine == ASREngine.VOSK
        assert processor.sample_rate == 22050

    def test_statistics_initialized(self, audio_processor):
        """测试统计信息初始化"""
        stats = audio_processor.get_statistics()

        assert stats["total_processed"] == 0
        assert stats["asr_count"] == 0
        assert stats["avg_processing_time_ms"] == 0.0


class TestAudioLoading:
    """测试音频加载"""

    def test_load_valid_audio(self, audio_processor, test_audio_path):
        """测试加载有效音频"""
        y, sr = audio_processor.load_audio(test_audio_path)

        assert y is not None
        assert len(y) > 0
        assert sr == 16000

    def test_load_invalid_audio(self, audio_processor):
        """测试加载无效音频"""
        with pytest.raises(AudioProcessorError):
            audio_processor.load_audio("nonexistent.wav")

    def test_audio_resampling(self, test_audio_path):
        """测试音频重采样"""
        processor = AudioProcessor(sample_rate=8000)
        y, sr = processor.load_audio(test_audio_path)

        assert sr == 8000  # 验证重采样到目标采样率


class TestAudioMetadata:
    """测试音频元数据提取"""

    def test_get_metadata(self, audio_processor, test_audio_path):
        """测试获取元数据"""
        y, sr = audio_processor.load_audio(test_audio_path)
        metadata = audio_processor.get_metadata(test_audio_path, y, sr)

        assert isinstance(metadata, AudioMetadata)
        assert metadata.duration_seconds > 0
        assert metadata.sample_rate == 16000
        assert metadata.channels == 1
        assert metadata.size_bytes > 0
        assert len(metadata.hash_md5) == 32

    def test_metadata_hash_consistency(self, audio_processor, test_audio_path):
        """测试元数据hash一致性"""
        y, sr = audio_processor.load_audio(test_audio_path)
        metadata1 = audio_processor.get_metadata(test_audio_path, y, sr)
        metadata2 = audio_processor.get_metadata(test_audio_path, y, sr)

        assert metadata1.hash_md5 == metadata2.hash_md5

    def test_metadata_to_dict(self, audio_processor, test_audio_path):
        """测试元数据转字典"""
        y, sr = audio_processor.load_audio(test_audio_path)
        metadata = audio_processor.get_metadata(test_audio_path, y, sr)
        metadata_dict = metadata.to_dict()

        assert "duration_seconds" in metadata_dict
        assert "sample_rate" in metadata_dict
        assert "hash_md5" in metadata_dict


class TestASR:
    """测试语音识别"""

    def test_asr_whisper_not_installed(self, audio_processor, mock_audio_data):
        """测试Whisper未安装时的降级"""
        y, sr = mock_audio_data

        with patch('audio_processor.whisper', None):
            result = audio_processor.transcribe_audio(y, sr)

        assert isinstance(result, ASRResult)
        assert result.text == ""
        assert result.confidence == 0.0

    def test_asr_disabled(self, mock_audio_data):
        """测试禁用ASR"""
        processor = AudioProcessor(asr_engine=ASREngine.NONE)
        y, sr = mock_audio_data

        result = processor.transcribe_audio(y, sr)

        assert result.text == ""
        assert result.confidence == 0.0

    def test_asr_result_to_dict(self):
        """测试ASR结果转字典"""
        result = ASRResult(
            text="测试文本",
            confidence=0.95,
            language="zh",
            segments=[{"start": 0.0, "end": 1.0, "text": "测试"}]
        )

        result_dict = result.to_dict()
        assert result_dict["text"] == "测试文本"
        assert result_dict["confidence"] == 0.95
        assert len(result_dict["segments"]) == 1


class TestFeatureExtraction:
    """测试特征提取"""

    def test_extract_features(self, audio_processor, mock_audio_data):
        """测试特征提取"""
        y, sr = mock_audio_data
        features = audio_processor.extract_features(y, sr)

        assert isinstance(features, AudioFeatures)
        assert features.mfcc is not None
        assert features.spectral_centroid is not None
        assert features.tempo > 0
        assert features.energy > 0

    def test_mfcc_shape(self, audio_processor, mock_audio_data):
        """测试MFCC形状"""
        y, sr = mock_audio_data
        features = audio_processor.extract_features(y, sr)

        # MFCC默认13个系数
        assert features.mfcc.shape[0] == 13

    def test_features_to_dict(self, audio_processor, mock_audio_data):
        """测试特征转字典"""
        y, sr = mock_audio_data
        features = audio_processor.extract_features(y, sr)
        features_dict = features.to_dict()

        assert "mfcc_shape" in features_dict
        assert "tempo" in features_dict
        assert "energy" in features_dict


class TestEntityExtraction:
    """测试实体提取"""

    def test_extract_entities_from_empty_text(self, audio_processor):
        """测试空文本实体提取"""
        entities = audio_processor.extract_entities_from_asr("")

        assert entities == []

    def test_extract_entities_from_text(self, audio_processor):
        """测试文本实体提取"""
        text = "Apple 公司发布了 iPhone 15"
        entities = audio_processor.extract_entities_from_asr(text)

        assert len(entities) > 0
        # 验证提取到大写单词
        entity_texts = [e["text"] for e in entities]
        assert "Apple" in entity_texts or "iPhone" in entity_texts

    def test_entity_confidence(self, audio_processor):
        """测试实体置信度"""
        text = "Python 是一门编程语言"
        entities = audio_processor.extract_entities_from_asr(text)

        if entities:
            for entity in entities:
                assert "confidence" in entity
                assert 0 <= entity["confidence"] <= 1


class TestFullProcessing:
    """测试完整处理流程"""

    @patch('audio_processor.whisper')
    def test_process_audio_with_asr(self, mock_whisper, audio_processor, test_audio_path):
        """测试启用ASR的音频处理"""
        # Mock Whisper响应
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "这是测试文本",
            "language": "zh",
            "segments": []
        }
        mock_whisper.load_model.return_value = mock_model

        result = audio_processor.process(test_audio_path, enable_asr=True)

        assert isinstance(result, AudioAnalysisResult)
        assert result.metadata is not None
        assert result.asr_result is not None
        assert result.features is not None

    def test_process_audio_without_asr(self, audio_processor, test_audio_path):
        """测试禁用ASR的音频处理"""
        result = audio_processor.process(test_audio_path, enable_asr=False)

        assert isinstance(result, AudioAnalysisResult)
        assert result.metadata is not None
        assert result.asr_result is None
        assert result.features is not None

    def test_process_updates_statistics(self, audio_processor, test_audio_path):
        """测试处理更新统计信息"""
        audio_processor.process(test_audio_path, enable_asr=False)

        stats = audio_processor.get_statistics()
        assert stats["total_processed"] == 1
        assert stats["feature_extraction_count"] == 1
        assert stats["avg_processing_time_ms"] > 0.0

    def test_result_to_dict(self, audio_processor, test_audio_path):
        """测试结果转字典"""
        result = audio_processor.process(test_audio_path, enable_asr=False)
        result_dict = result.to_dict()

        assert "audio_path" in result_dict
        assert "metadata" in result_dict
        assert "features" in result_dict


class TestStatistics:
    """测试统计信息"""

    def test_initial_statistics(self, audio_processor):
        """测试初始统计"""
        stats = audio_processor.get_statistics()

        assert stats["total_processed"] == 0
        assert stats["asr_count"] == 0
        assert stats["avg_processing_time_ms"] == 0.0

    def test_statistics_after_processing(self, audio_processor, test_audio_path):
        """测试处理后统计"""
        audio_processor.process(test_audio_path, enable_asr=False)

        stats = audio_processor.get_statistics()
        assert stats["total_processed"] == 1
        assert stats["feature_extraction_count"] == 1
        assert stats["avg_processing_time_ms"] > 0.0

    def test_statistics_immutability(self, audio_processor):
        """测试统计不可变性"""
        stats1 = audio_processor.get_statistics()
        stats1["total_processed"] = 999

        stats2 = audio_processor.get_statistics()
        assert stats2["total_processed"] == 0


class TestErrorHandling:
    """测试错误处理"""

    def test_invalid_audio_path(self, audio_processor):
        """测试无效音频路径"""
        with pytest.raises(AudioProcessorError):
            audio_processor.process("nonexistent.wav")

    def test_corrupted_audio_handling(self, audio_processor):
        """测试损坏音频处理"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
            f.write(b"not an audio file")

        try:
            with pytest.raises(AudioProcessorError):
                audio_processor.process(temp_path)
        finally:
            try:
                Path(temp_path).unlink()
            except:
                pass
