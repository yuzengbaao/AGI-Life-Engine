#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态感知融合模块
Multimodal Perception Fusion Module

功能：
1. 视觉和听觉信息融合
2. 跨模态特征对齐
3. 注意力机制
4. 多模态决策支持

Author: AGI System Development Team
Date: 2026-01-26
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging


class ModalityType(Enum):
    """模态类型"""
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"


@dataclass
class ModalityData:
    """模态数据"""
    type: ModalityType
    data: Any
    features: Dict[str, Any]
    timestamp: float
    confidence: float = 1.0


class MultimodalFusion:
    """多模态融合器"""
    
    def __init__(self, visual_weight: float = 0.6, audio_weight: float = 0.4):
        self.visual_weight = visual_weight
        self.audio_weight = audio_weight
        self.logger = logging.getLogger("MultimodalFusion")
        
        # 归一化权重
        total = visual_weight + audio_weight
        self.visual_weight = visual_weight / total
        self.audio_weight = audio_weight / total
    
    def fuse_features(self, visual_features: Dict[str, Any], 
                    audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        融合视觉和听觉特征
        
        Args:
            visual_features: 视觉特征
            audio_features: 音频特征
            
        Returns:
            融合后的特征
        """
        fused = {}
        
        # 融合基本统计特征
        if 'mean' in visual_features and 'rms' in audio_features:
            fused['activity_level'] = (
                self.visual_weight * np.mean(visual_features['mean']) +
                self.audio_weight * audio_features['rms']
            )
        
        # 融合动态特征
        if 'edge_density' in visual_features and 'zcr' in audio_features:
            fused['dynamics'] = (
                self.visual_weight * visual_features['edge_density'] +
                self.audio_weight * audio_features['zcr']
            )
        
        # 融合对比度特征
        if 'contrast' in visual_features and 'crest_factor' in audio_features:
            fused['contrast'] = (
                self.visual_weight * visual_features['contrast'] +
                self.audio_weight * audio_features['crest_factor']
            )
        
        return fused
    
    def align_features(self, visual_data: ModalityData, 
                     audio_data: ModalityData,
                     time_window: float = 1.0) -> Tuple[ModalityData, ModalityData]:
        """
        对齐不同模态的时间戳
        
        Args:
            visual_data: 视觉数据
            audio_data: 音频数据
            time_window: 时间窗口（秒）
            
        Returns:
            对齐后的视觉和音频数据
        """
        time_diff = abs(visual_data.timestamp - audio_data.timestamp)
        
        if time_diff > time_window:
            self.logger.warning(f"时间戳差异过大: {time_diff:.3f}s")
        
        return visual_data, audio_data
    
    def compute_attention(self, visual_data: ModalityData, 
                       audio_data: ModalityData) -> Dict[str, float]:
        """
        计算跨模态注意力权重
        
        Args:
            visual_data: 视觉数据
            audio_data: 音频数据
            
        Returns:
            注意力权重字典
        """
        attention = {}
        
        # 基于置信度的注意力
        total_confidence = visual_data.confidence + audio_data.confidence
        attention['visual'] = visual_data.confidence / total_confidence
        attention['audio'] = audio_data.confidence / total_confidence
        
        # 基于特征显著性的注意力
        visual_salience = self._compute_visual_salience(visual_data.features)
        audio_salience = self._compute_audio_salience(audio_data.features)
        
        total_salience = visual_salience + audio_salience
        attention['visual_salience'] = visual_salience / total_salience
        attention['audio_salience'] = audio_salience / total_salience
        
        return attention
    
    def _compute_visual_salience(self, features: Dict[str, Any]) -> float:
        """计算视觉显著性"""
        salience = 0.0
        
        if 'edge_density' in features:
            salience += features['edge_density']
        
        if 'contrast' in features:
            salience += features['contrast'] / 100.0
        
        return salience
    
    def _compute_audio_salience(self, features: Dict[str, Any]) -> float:
        """计算音频显著性"""
        salience = 0.0
        
        if 'rms' in features:
            salience += features['rms']
        
        if 'zcr' in features:
            salience += features['zcr']
        
        return salience
    
    def detect_cross_modal_consistency(self, visual_data: ModalityData,
                                     audio_data: ModalityData) -> float:
        """
        检测跨模态一致性
        
        Args:
            visual_data: 视觉数据
            audio_data: 音频数据
            
        Returns:
            一致性分数 (0-1)
        """
        consistency = 0.0
        
        # 活动水平一致性
        visual_activity = 0.0
        if 'edge_density' in visual_data.features:
            visual_activity = visual_data.features['edge_density']
        
        audio_activity = 0.0
        if 'rms' in audio_data.features:
            audio_activity = audio_data.features['rms']
        
        # 归一化
        visual_activity = min(visual_activity, 1.0)
        audio_activity = min(audio_activity / 2.0, 1.0)
        
        # 计算一致性
        activity_diff = abs(visual_activity - audio_activity)
        consistency += (1.0 - activity_diff) * 0.5
        
        # 动态一致性
        visual_dynamics = 0.0
        if 'contrast' in visual_data.features:
            visual_dynamics = visual_data.features['contrast'] / 100.0
        
        audio_dynamics = 0.0
        if 'crest_factor' in audio_data.features:
            audio_dynamics = audio_data.features['crest_factor'] / 5.0
        
        dynamics_diff = abs(visual_dynamics - audio_dynamics)
        consistency += (1.0 - dynamics_diff) * 0.5
        
        return consistency
    
    def generate_fusion_context(self, visual_data: ModalityData,
                               audio_data: ModalityData) -> Dict[str, Any]:
        """
        生成融合上下文
        
        Args:
            visual_data: 视觉数据
            audio_data: 音频数据
            
        Returns:
            融合上下文字典
        """
        context = {
            'timestamp': (visual_data.timestamp + audio_data.timestamp) / 2,
            'visual': visual_data.features,
            'audio': audio_data.features,
            'fused': self.fuse_features(visual_data.features, audio_data.features),
            'attention': self.compute_attention(visual_data, audio_data),
            'consistency': self.detect_cross_modal_consistency(visual_data, audio_data)
        }
        
        return context
    
    def rank_modalities(self, visual_data: ModalityData,
                       audio_data: ModalityData) -> List[Tuple[str, float]]:
        """
        排序模态重要性
        
        Args:
            visual_data: 视觉数据
            audio_data: 音频数据
            
        Returns:
            排序后的模态列表 [(modality, score)]
        """
        scores = []
        
        # 视觉分数
        visual_score = visual_data.confidence
        if 'edge_density' in visual_data.features:
            visual_score += visual_data.features['edge_density']
        scores.append(('visual', visual_score))
        
        # 音频分数
        audio_score = audio_data.confidence
        if 'rms' in audio_data.features:
            audio_score += audio_data.features['rms']
        scores.append(('audio', audio_score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores


class MultimodalDecisionSupport:
    """多模态决策支持系统"""
    
    def __init__(self, fusion: MultimodalFusion):
        self.fusion = fusion
        self.logger = logging.getLogger("MultimodalDecisionSupport")
    
    def recommend_action(self, fusion_context: Dict[str, Any]) -> str:
        """
        基于融合上下文推荐行动
        
        Args:
            fusion_context: 融合上下文
            
        Returns:
            推荐的行动
        """
        fused = fusion_context['fused']
        consistency = fusion_context['consistency']
        attention = fusion_context['attention']
        
        # 基于活动水平推荐
        if 'activity_level' in fused:
            activity = fused['activity_level']
            
            if activity > 0.8:
                return "high_activity"
            elif activity > 0.5:
                return "moderate_activity"
            else:
                return "low_activity"
        
        # 基于一致性推荐
        if consistency > 0.8:
            return "consistent_perception"
        elif consistency > 0.5:
            return "moderate_consistency"
        else:
            return "inconsistent_perception"
    
    def generate_insight(self, fusion_context: Dict[str, Any]) -> str:
        """
        生成多模态洞察
        
        Args:
            fusion_context: 融合上下文
            
        Returns:
            洞察文本
        """
        insights = []
        
        # 视觉洞察
        visual = fusion_context['visual']
        if 'edge_density' in visual:
            if visual['edge_density'] > 0.5:
                insights.append("视觉场景复杂度高")
            else:
                insights.append("视觉场景相对简单")
        
        # 音频洞察
        audio = fusion_context['audio']
        if 'rms' in audio:
            if audio['rms'] > 0.5:
                insights.append("音频能量较高")
            else:
                insights.append("音频能量较低")
        
        # 一致性洞察
        consistency = fusion_context['consistency']
        if consistency > 0.8:
            insights.append("视听一致性高")
        elif consistency < 0.3:
            insights.append("视听一致性低，可能存在异常")
        
        # 注意力洞察
        attention = fusion_context['attention']
        if attention['visual'] > 0.7:
            insights.append("主要依赖视觉信息")
        elif attention['audio'] > 0.7:
            insights.append("主要依赖听觉信息")
        else:
            insights.append("视听信息平衡")
        
        return "; ".join(insights)


def test_multimodal_fusion():
    """测试多模态融合"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("测试多模态感知融合")
    print("=" * 70)
    
    # 创建融合器
    fusion = MultimodalFusion(visual_weight=0.6, audio_weight=0.4)
    decision_support = MultimodalDecisionSupport(fusion)
    
    # 创建测试数据
    visual_features = {
        'mean': [120, 125, 130],
        'std': [50, 48, 52],
        'edge_density': 0.45,
        'brightness': 125.0,
        'contrast': 35.0
    }
    
    audio_features = {
        'rms': 0.6,
        'zcr': 0.15,
        'crest_factor': 2.5,
        'spectral_centroid': 2000.0,
        'spectral_bandwidth': 1500.0
    }
    
    visual_data = ModalityData(
        type=ModalityType.VISUAL,
        data=None,
        features=visual_features,
        timestamp=time.time(),
        confidence=0.9
    )
    
    audio_data = ModalityData(
        type=ModalityType.AUDIO,
        data=None,
        features=audio_features,
        timestamp=time.time(),
        confidence=0.8
    )
    
    print("\n[测试1] 特征融合")
    print("-" * 70)
    fused = fusion.fuse_features(visual_features, audio_features)
    print("融合后的特征:")
    for key, value in fused.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n[测试2] 注意力计算")
    print("-" * 70)
    attention = fusion.compute_attention(visual_data, audio_data)
    print("注意力权重:")
    for key, value in attention.items():
        print(f"  {key}: {value:.3f}")
    
    print("\n[测试3] 跨模态一致性")
    print("-" * 70)
    consistency = fusion.detect_cross_modal_consistency(visual_data, audio_data)
    print(f"一致性分数: {consistency:.3f}")
    
    print("\n[测试4] 模态排序")
    print("-" * 70)
    ranked = fusion.rank_modalities(visual_data, audio_data)
    print("模态重要性排序:")
    for i, (modality, score) in enumerate(ranked, 1):
        print(f"  {i}. {modality}: {score:.3f}")
    
    print("\n[测试5] 融合上下文生成")
    print("-" * 70)
    context = fusion.generate_fusion_context(visual_data, audio_data)
    print("融合上下文:")
    print(f"  时间戳: {context['timestamp']:.3f}")
    print(f"  一致性: {context['consistency']:.3f}")
    
    print("\n[测试6] 决策支持")
    print("-" * 70)
    action = decision_support.recommend_action(context)
    print(f"推荐行动: {action}")
    
    insight = decision_support.generate_insight(context)
    print(f"多模态洞察: {insight}")
    
    print("\n✅ 测试完成")


if __name__ == "__main__":
    import time
    test_multimodal_fusion()
