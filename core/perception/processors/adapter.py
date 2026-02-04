#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
感知处理器适配器 - Perception Processor Adapter
适配AdvancedVideoProcessor和AdvancedAudioProcessor用于实时流处理

作者: AGI System Team
日期: 2025-11-21
版本: 1.0.0
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import tempfile
import os

from core.perception.processors.video import AdvancedVideoProcessor
from core.perception.processors.audio import AdvancedAudioProcessor, AudioData, AudioTaskType

logger = logging.getLogger(__name__)


class RealtimeVideoAdapter:
    """实时视频处理适配器"""
    
    def __init__(self):
        """初始化视频适配器"""
        self.processor = AdvancedVideoProcessor()
        self.frame_history = []  # 保存最近的帧用于场景分析
        self.max_history = 30  # 保留1秒历史(假设30fps)
        
        logger.info("📹 实时视频适配器初始化完成")
    
    def process_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理单帧
        
        Args:
            frame_data: 包含frame, timestamp, frame_number的字典
            
        Returns:
            处理结果字典
        """
        frame = frame_data['frame']
        timestamp = frame_data['timestamp']
        frame_number = frame_data['frame_number']
        
        try:
            # 添加到历史
            self.frame_history.append(frame)
            if len(self.frame_history) > self.max_history:
                self.frame_history.pop(0)
            
            # 调用底层处理器的单帧处理
            result = self.processor.process_frame(frame)
            
            # 添加时间戳信息
            result['timestamp'] = timestamp
            result['frame_number'] = frame_number
            
            return {
                'success': True,
                'frame_number': frame_number,
                'timestamp': timestamp,
                'detections': result.get('detections', []),
                'scene_analysis': result.get('scene_analysis', {}),
                'summary': self._generate_frame_summary(result)
            }
            
        except Exception as e:
            logger.error(f"❌ 处理帧失败: {e}")
            return {
                'success': False,
                'frame_number': frame_number,
                'timestamp': timestamp,
                'error': str(e)
            }
    
    def _generate_frame_summary(self, result: Dict[str, Any]) -> str:
        """生成帧摘要
        
        Args:
            result: 处理结果
            
        Returns:
            摘要文本
        """
        detections = result.get('detections', [])
        scene = result.get('scene_analysis', {})
        
        summary_parts = []
        
        # 检测对象数量
        if detections:
            obj_counts = {}
            for det in detections:
                label = det.get('label', 'unknown')
                obj_counts[label] = obj_counts.get(label, 0) + 1
            
            obj_str = ', '.join([f"{count}个{label}" for label, count in obj_counts.items()])
            summary_parts.append(f"检测到: {obj_str}")
        
        # 场景复杂度
        complexity = scene.get('complexity', 'unknown')
        if complexity != 'unknown':
            summary_parts.append(f"场景复杂度: {complexity}")
        
        return '; '.join(summary_parts) if summary_parts else "无特殊事件"
    
    def get_scene_summary(self) -> Dict[str, Any]:
        """获取场景摘要(基于历史帧)
        
        Returns:
            场景摘要字典
        """
        if not self.frame_history:
            return {'status': 'no_data'}
        
        try:
            # 使用最新的帧进行分析
            latest_frame = self.frame_history[-1]
            result = self.processor.process_frame(latest_frame)
            
            return {
                'status': 'success',
                'frame_count': len(self.frame_history),
                'scene_analysis': result.get('scene_analysis', {}),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ 获取场景摘要失败: {e}")
            return {'status': 'error', 'error': str(e)}


class RealtimeAudioAdapter:
    """实时音频处理适配器"""
    
    def __init__(self):
        """初始化音频适配器"""
        self.processor = AdvancedAudioProcessor()
        self.audio_history = []  # 保存最近的音频块
        self.max_history = 10  # 保留10块(10秒)
        
        logger.info("🎵 实时音频适配器初始化完成")
    
    def process_audio_chunk(self, audio_data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """处理音频块
        
        Args:
            audio_data_dict: 包含audio, timestamp, chunk_number, sample_rate, channels的字典
            
        Returns:
            处理结果字典
        """
        audio_array = audio_data_dict['audio']
        timestamp = audio_data_dict['timestamp']
        chunk_number = audio_data_dict['chunk_number']
        sample_rate = audio_data_dict['sample_rate']
        channels = audio_data_dict['channels']
        
        try:
            # 添加到历史
            self.audio_history.append(audio_data_dict)
            if len(self.audio_history) > self.max_history:
                self.audio_history.pop(0)
            
            # 转换为AudioData格式
            # 确保是单声道
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            audio_data = AudioData(
                waveform=audio_array.flatten(),
                sample_rate=sample_rate
            )
            
            # 调用底层处理器
            result = self.processor.process_audio(
                audio_data, 
                AudioTaskType.SPEECH_RECOGNITION
            )
            
            return {
                'success': True,
                'chunk_number': chunk_number,
                'timestamp': timestamp,
                'transcription': result.get('transcription', ''),
                'emotion': result.get('emotion', 'neutral'),
                'features': result.get('features', {}),
                'summary': self._generate_audio_summary(result)
            }
            
        except Exception as e:
            logger.error(f"❌ 处理音频块失败: {e}")
            return {
                'success': False,
                'chunk_number': chunk_number,
                'timestamp': timestamp,
                'error': str(e)
            }
    
    def _generate_audio_summary(self, result: Dict[str, Any]) -> str:
        """生成音频摘要
        
        Args:
            result: 处理结果
            
        Returns:
            摘要文本
        """
        summary_parts = []
        
        # 转录文本
        transcription = result.get('transcription', '')
        if transcription:
            summary_parts.append(f"语音: '{transcription}'")
        
        # 情感
        emotion = result.get('emotion', '')
        if emotion and emotion != 'neutral':
            summary_parts.append(f"情感: {emotion}")
        
        # 音频特征
        features = result.get('features', {})
        if 'tempo' in features:
            summary_parts.append(f"节奏: {features['tempo']:.0f}BPM")
        
        return '; '.join(summary_parts) if summary_parts else "无语音内容"
    
    def get_audio_summary(self) -> Dict[str, Any]:
        """获取音频摘要(基于历史块)
        
        Returns:
            音频摘要字典
        """
        if not self.audio_history:
            return {'status': 'no_data'}
        
        try:
            # 统计最近的语音和情感
            transcriptions = []
            emotions = []
            
            for chunk in self.audio_history[-5:]:  # 最近5块
                if 'transcription' in chunk:
                    transcriptions.append(chunk['transcription'])
                if 'emotion' in chunk:
                    emotions.append(chunk['emotion'])
            
            return {
                'status': 'success',
                'chunk_count': len(self.audio_history),
                'recent_transcriptions': transcriptions,
                'recent_emotions': emotions,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ 获取音频摘要失败: {e}")
            return {'status': 'error', 'error': str(e)}


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("感知处理器适配器测试")
    print("=" * 60)
    
    # 测试视频适配器
    print("\n📹 测试视频适配器...")
    video_adapter = RealtimeVideoAdapter()
    
    # 创建测试帧
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame_data = {
        'frame': test_frame,
        'timestamp': datetime.now(),
        'frame_number': 1
    }
    
    print("处理测试帧...")
    result = video_adapter.process_frame(frame_data)
    print(f"结果: {result}")
    
    # 测试音频适配器
    print("\n🎵 测试音频适配器...")
    audio_adapter = RealtimeAudioAdapter()
    
    # 创建测试音频
    test_audio = np.random.randn(16000, 1).astype(np.float32)  # 1秒音频
    audio_data = {
        'audio': test_audio,
        'timestamp': datetime.now(),
        'chunk_number': 1,
        'sample_rate': 16000,
        'channels': 1
    }
    
    print("处理测试音频...")
    result = audio_adapter.process_audio_chunk(audio_data)
    print(f"结果: {result}")
    
    print("\n✅ 测试完成")
