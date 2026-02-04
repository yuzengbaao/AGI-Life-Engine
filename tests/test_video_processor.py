"""
视频处理器测试
Tests for VideoProcessor
"""

from video_processor import (VideoProcessor, VideoProcessorError,
                             VideoMetadata, VideoAnalysisResult)
import numpy as np
import pytest

video = pytest.importorskip('video_processor')
# pylint: disable=wrong-import-position


@pytest.fixture
def video_processor():
    """创建测试用VideoProcessor"""
    return VideoProcessor(hist_bins=16, scene_threshold=0.4)


@pytest.fixture
def synthetic_video(tmp_path):
    """生成一个简短的合成视频（若cv2不可用则跳过）"""
    cv2 = pytest.importorskip('cv2')

    path = tmp_path / 'test_video.avi'
    width, height, fps, frames = 64, 48, 10, 12

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # pylint: disable=no-member
    writer = cv2.VideoWriter(str(path), fourcc, fps,
                             (width, height))  # pylint: disable=no-member

    # 前6帧红色,后6帧绿色,制造明显场景变化
    for i in range(frames):
        if i < frames // 2:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (0, 0, 255)  # 红色 BGR
        else:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (0, 255, 0)  # 绿色 BGR
        writer.write(frame)
    writer.release()

    return str(path)


class TestInit:
    """测试VideoProcessor初始化"""

    def test_default_init(self):
        """测试默认初始化"""
        processor = VideoProcessor()
        stats = processor.get_statistics()
        assert stats["total_processed"] == 0
        assert stats["avg_processing_time_ms"] == 0.0


class TestLoading:
    """测试视频加载"""

    def test_invalid_path(self, video_processor):  # pylint: disable=redefined-outer-name
        """测试无效路径错误"""
        with pytest.raises(VideoProcessorError):
            video_processor.get_metadata("nonexistent.mp4")


class TestMetadata:
    """测试元数据提取"""
    def test_get_metadata(self, video_processor, synthetic_video):
        """测试获取元数据"""
        m = video_processor.get_metadata(synthetic_video)
        assert isinstance(m, VideoMetadata)
        assert m.width > 0 and m.height > 0
        assert m.fps > 0
        assert m.frame_count > 0
        assert m.duration_seconds >= 0
        assert len(m.hash_md5) == 32

    def test_metadata_to_dict(self, video_processor, synthetic_video):
        """测试元数据序列化"""
        m = video_processor.get_metadata(synthetic_video)
        d = m.to_dict()
        assert "fps" in d and "frame_count" in d


class TestKeyframes:
    """测试关键帧提取"""
    def test_extract_keyframes(self, video_processor, synthetic_video):
        """测试提取关键帧"""
        kf = video_processor.extract_keyframes(synthetic_video)
        assert isinstance(kf, list)
        assert 0 in kf
        # 应至少检测到一次场景变化
        assert len(kf) >= 2


class TestScenes:
    """测试场景检测"""
    def test_detect_scenes(self, video_processor, synthetic_video):
        """测试检测场景边界"""
        scenes = video_processor.detect_scenes(synthetic_video)
        assert isinstance(scenes, list)
        assert len(scenes) >= 2


class TestProcess:
    """测试完整处理流程"""
    def test_process_all(self, video_processor, synthetic_video):
        """测试全功能处理"""
        result = video_processor.process(synthetic_video)
        assert isinstance(result, VideoAnalysisResult)
        assert result.metadata.frame_count > 0
        assert len(result.keyframes) >= 2
        assert len(result.scene_boundaries) >= 2

    def test_stats_updated(self, video_processor, synthetic_video):
        """测试统计更新"""
        video_processor.process(synthetic_video)
        s = video_processor.get_statistics()
        assert s["total_processed"] == 1
        assert s["keyframe_count"] >= 2
        assert s["scene_count"] >= 2
        assert s["avg_processing_time_ms"] > 0


class TestErrors:
    """测试错误处理"""
    def test_no_cv2(self, monkeypatch, tmp_path):
        """测试缺少cv2"""
        import video_processor as vp
        monkeypatch.setattr(vp, 'cv2', None)
        p = vp.VideoProcessor()
        with pytest.raises(vp.VideoProcessorError):
            p.get_metadata(str(tmp_path / 'a.mp4'))

    def test_corrupted_video(self, video_processor, tmp_path):
        """测试损坏视频"""
        bad_file = tmp_path / 'bad.mp4'
        bad_file.write_bytes(b'not a video')
        with pytest.raises(VideoProcessorError):
            video_processor.get_metadata(str(bad_file))


class TestResultSerialization:
    """测试结果序列化"""
    def test_result_to_dict(self, video_processor, synthetic_video):
        """测试结果转字典"""
        result = video_processor.process(synthetic_video)
        d = result.to_dict()
        assert 'video_path' in d
        assert 'metadata' in d
        assert 'keyframes' in d
        assert 'scene_boundaries' in d


class TestStatistics:
    def test_initial_stats(self, video_processor):
        s = video_processor.get_statistics()
        assert s['total_processed'] == 0
        assert s['keyframe_count'] == 0
        assert s['scene_count'] == 0
        assert s['avg_processing_time_ms'] == 0.0

    def test_stats_immutability(self, video_processor):
        s1 = video_processor.get_statistics()
        s1['total_processed'] = 999
        s2 = video_processor.get_statistics()
        assert s2['total_processed'] == 0


class TestKeyframesAdvanced:
    def test_empty_video_keyframes(self, video_processor, tmp_path):
        cv2 = pytest.importorskip('cv2')
        path = tmp_path / 'empty.avi'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(str(path), fourcc, 10, (32, 32))
        writer.release()
        kf = video_processor.extract_keyframes(str(path))
        assert isinstance(kf, list)

    def test_scene_threshold_sensitivity(self, synthetic_video):
        p_low = VideoProcessor(scene_threshold=0.1)
        p_high = VideoProcessor(scene_threshold=0.9)
        kf_low = p_low.extract_keyframes(synthetic_video)
        kf_high = p_high.extract_keyframes(synthetic_video)
        # 低阈值应检测更多关键帧
        assert len(kf_low) >= len(kf_high)


class TestProcessOptions:
    def test_process_no_keyframes(self, video_processor, synthetic_video):
        result = video_processor.process(
            synthetic_video,
            enable_keyframes=False,
            enable_scenes=True)
        assert len(result.keyframes) == 0
        assert len(result.scene_boundaries) >= 0

    def test_process_no_scenes(self, video_processor, synthetic_video):
        result = video_processor.process(
            synthetic_video,
            enable_keyframes=True,
            enable_scenes=False)
        assert len(result.keyframes) >= 0
        assert len(result.scene_boundaries) == 0

    def test_process_nothing(self, video_processor, synthetic_video):
        result = video_processor.process(
            synthetic_video,
            enable_keyframes=False,
            enable_scenes=False)
        assert len(result.keyframes) == 0
        assert len(result.scene_boundaries) == 0


class TestMetadataAdvanced:
    def test_hash_consistency(self, video_processor, synthetic_video):
        m1 = video_processor.get_metadata(synthetic_video)
        m2 = video_processor.get_metadata(synthetic_video)
        assert m1.hash_md5 == m2.hash_md5

    def test_duration_calculation(self, video_processor, synthetic_video):
        m = video_processor.get_metadata(synthetic_video)
        assert m.duration_seconds > 0
        # 验证 duration = frame_count / fps
        assert abs(m.duration_seconds - m.frame_count / m.fps) < 0.1
