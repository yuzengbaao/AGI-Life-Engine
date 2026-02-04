#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频预处理管道
Audio Preprocessing Pipeline

功能：
1. 音频降噪
2. 特征提取（MFCC、频谱特征）
3. 音频增强
4. 音频分段

Author: AGI System Development Team
Date: 2026-01-26
Version: 1.0.0
"""

import numpy as np
import scipy.signal
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
import logging


class NoiseReductionMethod(Enum):
    """降噪方法"""
    SPECTRAL_SUBTRACTION = "spectral_subtraction"
    WIENER_FILTER = "wiener_filter"
    MOVING_AVERAGE = "moving_average"


class AudioPreprocessor:
    """音频预处理器"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger("AudioPreprocessor")
    
    def normalize(self, audio: np.ndarray, method: str = "peak") -> np.ndarray:
        """
        音频归一化
        
        Args:
            audio: 输入音频
            method: 归一化方法
                - "peak": 峰值归一化
                - "rms": RMS归一化
                
        Returns:
            归一化后的音频
        """
        if method == "peak":
            # 峰值归一化
            peak = np.max(np.abs(audio))
            if peak > 0:
                normalized = audio / peak
            else:
                normalized = audio
        elif method == "rms":
            # RMS归一化
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                normalized = audio / rms
            else:
                normalized = audio
        else:
            normalized = audio
        
        return normalized
    
    def denoise(self, audio: np.ndarray, method: str = "spectral_subtraction",
                noise_frames: int = 10) -> np.ndarray:
        """
        音频降噪
        
        Args:
            audio: 输入音频
            method: 降噪方法
                - "spectral_subtraction": 移动平均（简化版）
                - "moving_average": 移动平均
            noise_frames: 用于估计噪声的帧数
            
        Returns:
            降噪后的音频
        """
        if method == "spectral_subtraction":
            # 使用移动平均滤波作为简单的降噪方法
            denoised = self._moving_average(audio, window_size=5)
        elif method == "moving_average":
            denoised = self._moving_average(audio, window_size=5)
        else:
            denoised = audio
        
        return denoised
    
    def _spectral_subtraction(self, audio: np.ndarray, noise_frames: int = 10) -> np.ndarray:
        """频谱减法降噪（简化版）"""
        # 使用移动平均滤波作为简单的降噪方法
        return self._moving_average(audio, window_size=5)
    
    def _moving_average(self, audio: np.ndarray, window_size: int = 5) -> np.ndarray:
        """移动平均滤波"""
        window = np.ones(window_size) / window_size
        return np.convolve(audio, window, mode='same')
    
    def extract_mfcc(self, audio: np.ndarray, n_mfcc: int = 13, 
                    n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        提取MFCC特征
        
        Args:
            audio: 输入音频
            n_mfcc: MFCC系数数量
            n_fft: FFT窗口大小
            hop_length: 跳跃长度
            
        Returns:
            MFCC特征矩阵 (n_frames, n_mfcc)
        """
        # 计算STFT
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        stft = np.zeros((n_frames, n_fft // 2 + 1))
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft
            frame = audio[start:end]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)), 'constant')
            # 计算功率谱
            frame_fft = np.fft.rfft(frame)
            stft[i] = np.abs(frame_fft)**2
        
        # Mel滤波器组
        mel_filters = self._create_mel_filters(n_fft, n_mels=40)
        mel_spectrum = np.dot(stft, mel_filters.T)
        
        # 对数Mel谱
        log_mel = np.log(mel_spectrum + 1e-10)
        
        # DCT变换得到MFCC
        mfcc = scipy.fft.dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
        
        return mfcc
    
    def _create_mel_filters(self, n_fft: int, n_mels: int = 40) -> np.ndarray:
        """创建Mel滤波器组"""
        # 将频率转换为Mel尺度
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Mel频率点
        mel_points = np.linspace(hz_to_mel(0), hz_to_mel(self.sample_rate / 2), n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # FFT频率点
        fft_bins = np.arange(n_fft // 2 + 1) * self.sample_rate / n_fft
        
        # 创建三角滤波器
        filters = np.zeros((n_mels, n_fft // 2 + 1))
        
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            # 左边上升
            left_slope = (fft_bins - left) / (center - left)
            filters[i] += np.where((fft_bins >= left) & (fft_bins <= center), 
                                  left_slope, 0)
            
            # 右边下降
            right_slope = (right - fft_bins) / (right - center)
            filters[i] += np.where((fft_bins > center) & (fft_bins <= right), 
                                  right_slope, 0)
        
        return filters
    
    def extract_spectral_features(self, audio: np.ndarray, 
                                n_fft: int = 2048) -> Dict[str, float]:
        """
        提取频谱特征
        
        Args:
            audio: 输入音频
            n_fft: FFT窗口大小
            
        Returns:
            频谱特征字典
        """
        # 计算功率谱
        stft = np.abs(np.fft.rfft(audio, n=n_fft))**2
        power_spectrum = np.mean(stft, axis=0)
        
        # 频率轴
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)
        
        features = {}
        
        # 质心频率
        centroid = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
        features['spectral_centroid'] = centroid
        
        # 频谱带宽
        bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * power_spectrum) / 
                          np.sum(power_spectrum))
        features['spectral_bandwidth'] = bandwidth
        
        # 频谱平坦度
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        arithmetic_mean = np.mean(power_spectrum)
        features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
        
        # 频谱滚降点
        cumulative_sum = np.cumsum(power_spectrum)
        total = cumulative_sum[-1]
        rolloff_idx = np.where(cumulative_sum >= 0.85 * total)[0][0]
        features['spectral_rolloff'] = freqs[rolloff_idx]
        
        # 零交叉率
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0)
        features['zero_crossing_rate'] = zero_crossings / len(audio)
        
        # RMS能量
        features['rms_energy'] = np.sqrt(np.mean(audio**2))
        
        return features
    
    def extract_temporal_features(self, audio: np.ndarray, 
                                frame_length: int = 1024) -> Dict[str, float]:
        """
        提取时域特征
        
        Args:
            audio: 输入音频
            frame_length: 帧长度
            
        Returns:
            时域特征字典
        """
        features = {}
        
        # 能量
        features['energy'] = np.sum(audio**2)
        
        # RMS
        features['rms'] = np.sqrt(np.mean(audio**2))
        
        # 过零率
        features['zcr'] = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
        
        # 峰值因子
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio**2))
        features['crest_factor'] = peak / (rms + 1e-10)
        
        return features
    
    def segment_audio(self, audio: np.ndarray, 
                     segment_length: float = 1.0,
                     overlap: float = 0.5) -> List[np.ndarray]:
        """
        音频分段
        
        Args:
            audio: 输入音频
            segment_length: 分段长度（秒）
            overlap: 重叠比例（0-1）
            
        Returns:
            分段后的音频列表
        """
        segment_samples = int(segment_length * self.sample_rate)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        for i in range(0, len(audio) - segment_samples + 1, hop_samples):
            segment = audio[i:i + segment_samples]
            segments.append(segment)
        
        return segments
    
    def preprocess(self, audio: np.ndarray, normalize: bool = True,
                   denoise: bool = True, extract_features: bool = True) -> Dict[str, Any]:
        """
        完整的预处理流程
        
        Args:
            audio: 输入音频
            normalize: 是否归一化
            denoise: 是否降噪
            extract_features: 是否提取特征
            
        Returns:
            预处理结果字典
        """
        result = {
            'original': audio.copy(),
            'processed': audio.copy(),
            'features': {}
        }
        
        processed = audio.copy()
        
        # 降噪
        if denoise:
            processed = self.denoise(processed, method="spectral_subtraction")
        
        # 归一化
        if normalize:
            processed = self.normalize(processed, method="rms")
        
        result['processed'] = processed
        
        # 提取特征
        if extract_features:
            result['features']['spectral'] = self.extract_spectral_features(processed)
            result['features']['temporal'] = self.extract_temporal_features(processed)
            result['features']['mfcc'] = self.extract_mfcc(processed)
        
        return result


def test_audio_preprocessing():
    """测试音频预处理"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("测试音频预处理管道")
    print("=" * 70)
    
    # 创建预处理器
    preprocessor = AudioPreprocessor(sample_rate=16000)
    
    # 生成测试音频（1秒，16kHz）
    duration = 1.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    test_audio = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    
    # 添加噪声
    noise = np.random.normal(0, 0.1, len(test_audio))
    test_audio = test_audio + noise
    
    print(f"\n原始音频形状: {test_audio.shape}")
    print(f"原始音频范围: [{test_audio.min():.3f}, {test_audio.max():.3f}]")
    
    # 测试归一化
    normalized = preprocessor.normalize(test_audio, method="peak")
    print(f"归一化后范围: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # 测试降噪
    denoised = preprocessor.denoise(test_audio, method="spectral_subtraction")
    print(f"降噪后形状: {denoised.shape}")
    
    # 测试特征提取
    spectral_features = preprocessor.extract_spectral_features(test_audio)
    print(f"\n频谱特征:")
    for key, value in spectral_features.items():
        print(f"  {key}: {value:.3f}")
    
    temporal_features = preprocessor.extract_temporal_features(test_audio)
    print(f"\n时域特征:")
    for key, value in temporal_features.items():
        print(f"  {key}: {value:.3f}")
    
    # 测试MFCC提取
    mfcc = preprocessor.extract_mfcc(test_audio, n_mfcc=13)
    print(f"\nMFCC特征形状: {mfcc.shape}")
    print(f"MFCC前5帧的前3个系数:")
    print(mfcc[:5, :3])
    
    # 测试音频分段
    segments = preprocessor.segment_audio(test_audio, segment_length=0.5, overlap=0.5)
    print(f"\n音频分段: {len(segments)} 段")
    print(f"每段长度: {[len(seg) for seg in segments[:3]]}")
    
    # 测试完整预处理流程
    print("\n测试完整预处理流程...")
    result = preprocessor.preprocess(
        test_audio,
        normalize=True,
        denoise=True,
        extract_features=True
    )
    
    print(f"预处理后形状: {result['processed'].shape}")
    print(f"预处理后范围: [{result['processed'].min():.3f}, {result['processed'].max():.3f}]")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    test_audio_preprocessing()
