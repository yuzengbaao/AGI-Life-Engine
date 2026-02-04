#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双螺旋引擎v2.1 - 智能融合逻辑（分离文件）
实现回归初衷的融合策略

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def intelligent_fusion(
    engine,  # DoubleHelixEngineV2实例
    result_A: Optional[Dict[str, Any]],
    result_B: Optional[Dict[str, Any]],
    state: np.ndarray,
    complementary_analysis: Optional[Any]
) -> Dict[str, Any]:
    """
    智能融合策略（v2.1纠偏版）

    决策流程：
    1. 互补区域识别 → 如果某个系统明显擅长，直接选择
    2. 创造性融合 → 如果强烈分歧，尝试生成新动作
    3. 数值融合 → 兜底策略（线性/非线性/对话）

    Args:
        engine: 引擎实例
        result_A: 系统A结果
        result_B: 系统B结果
        state: 当前状态
        complementary_analysis: 互补分析结果

    Returns:
        融合结果字典
    """

    # === 策略1：互补区域识别优先 ===
    if complementary_analysis is not None:
        from core.complementary_analyzer import SystemPreference
        
        if complementary_analysis.preference == SystemPreference.PREFER_A:
            logger.info(f"[智能融合] 🎯 选择系统A（擅长区域）: {complementary_analysis.reason}")
            engine.stats['fusion_modes_used']['complementary_A'] = engine.stats['fusion_modes_used'].get('complementary_A', 0) + 1
            
            # 🔧 [2026-01-17] 修复：互补选择也产生涌现（系统协同本身是智能的体现）
            conf_A = result_A.get('confidence', 0.5) if result_A else 0.5
            conf_B = result_B.get('confidence', 0.5) if result_B else 0.5
            emergence_from_complementary = abs(conf_A - conf_B) * 0.15  # 差异度产生涌现
            
            return {
                'action': result_A['action'],
                'confidence': result_A['confidence'],
                'method': 'complementary_selection_A',
                'emergence': emergence_from_complementary,  # 🔧 不再硬编码0
                'selected_system': 'A',
                'reason': complementary_analysis.reason,
                # 🆕 涌现行为标志
                'is_creative': False,
                'original_space': True
            }
        
        elif complementary_analysis.preference == SystemPreference.PREFER_B:
            logger.info(f"[智能融合] 🎯 选择系统B（擅长区域）: {complementary_analysis.reason}")
            engine.stats['fusion_modes_used']['complementary_B'] = engine.stats['fusion_modes_used'].get('complementary_B', 0) + 1
            
            # 🔧 [2026-01-17] 修复：互补选择也产生涌现
            conf_A = result_A.get('confidence', 0.5) if result_A else 0.5
            conf_B = result_B.get('confidence', 0.5) if result_B else 0.5
            emergence_from_complementary = abs(conf_A - conf_B) * 0.15
            
            return {
                'action': result_B['action'],
                'confidence': result_B['confidence'],
                'method': 'complementary_selection_B',
                'emergence': emergence_from_complementary,  # 🔧 不再硬编码0
                'selected_system': 'B',
                'reason': complementary_analysis.reason,
                # 🆕 涌现行为标志
                'is_creative': False,
                'original_space': True
            }

    # === 策略2：创造性融合（强烈分歧时） ===
    if engine.creative_fusion is not None and result_A and result_B:
        action_A = result_A['action']
        action_B = result_B['action']
        conf_A = result_A['confidence']
        conf_B = result_B['confidence']
        
        # 使用创造性融合引擎
        creative_result = engine.creative_fusion.fuse(
            action_A, action_B, conf_A, conf_B
        )
        
        if creative_result.is_creative:
            logger.info(f"[智能融合] ✨ 创造性融合: {creative_result.reasoning}")
            engine.stats['fusion_modes_used']['creative'] = engine.stats['fusion_modes_used'].get('creative', 0) + 1
            
            # 计算涌现分数（创造了新动作）
            emergence = 0.5 if creative_result.original_space else 0.8
            
            return {
                'action': creative_result.action,
                'confidence': creative_result.confidence,
                'method': 'creative_fusion',
                'emergence': emergence,
                'is_creative': True,
                'original_space': creative_result.original_space,
                'reason': creative_result.reasoning
            }

    # === 策略3：数值融合（兜底策略） ===
    selected_mode = engine._select_fusion_mode(result_A, result_B)
    
    if selected_mode.value == "dialogue" and engine.dialogue_engine:
        fused_result = engine._fuse_with_dialogue(result_A, result_B)
        engine.stats['fusion_modes_used']['dialogue'] += 1
        engine.stats['dialogue_emergence_total'] += fused_result['emergence']
        logger.debug("[智能融合] 💬 使用对话式融合（兜底）")
        
    elif selected_mode.value == "nonlinear" and engine.nonlinear_fusion:
        fused_result = engine._fuse_with_nonlinear(result_A, result_B)
        engine.stats['fusion_modes_used']['nonlinear'] += 1
        engine.stats['nonlinear_emergence_total'] += fused_result['emergence']
        logger.debug("[智能融合] 🔢 使用非线性融合（兜底）")
        
    else:
        fused_result = engine._fuse_linear(result_A, result_B)
        engine.stats['fusion_modes_used']['linear'] += 1
        logger.debug("[智能融合] ➕ 使用线性融合（兜底）")
    
    return fused_result
