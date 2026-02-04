#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创造性融合引擎 (Creative Fusion Engine)
实现真正的"从分歧中创造新方案"

核心理念：
- 当A和B强烈分歧时，不是简单平均
- 而是分析分歧的语义，生成创造性的复合动作
- 例如：A="attack", B="defend" → Creative="strategic_positioning"

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v1.0 - 系统纠偏版本
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DisagreementLevel(Enum):
    """分歧等级"""
    CONSENSUS = "consensus"        # 一致 (diff < 1)
    MILD = "mild"                 # 轻微 (diff = 1)
    MODERATE = "moderate"         # 中等 (diff = 2)
    STRONG = "strong"             # 强烈 (diff >= 3)


@dataclass
class CreativeFusionResult:
    """创造性融合结果"""
    action: int
    confidence: float
    is_creative: bool  # 是否是创造性动作
    original_space: bool  # 是否在原始动作空间内
    reasoning: str
    disagreement_level: DisagreementLevel
    component_actions: Tuple[int, int]  # (action_A, action_B)


class CreativeFusionEngine:
    """
    创造性融合引擎
    
    核心功能：
    1. 识别分歧类型（对立、正交、互补）
    2. 根据分歧语义生成复合动作
    3. 扩展动作空间而非简单加权
    
    示例：
    - A=0(left), B=1(right) → Creative=4(stop&observe)
    - A=2(forward), B=3(backward) → Creative=5(strategic_retreat)
    """
    
    def __init__(
        self,
        base_action_dim: int = 4,
        enable_expansion: bool = True,
        expansion_dim: int = 8,  # 扩展后的总动作空间
        disagreement_threshold: float = 0.5  # 🆕 分歧阈值（降低以增加创造性触发）
    ):
        self.base_action_dim = base_action_dim
        self.enable_expansion = enable_expansion
        self.expansion_dim = expansion_dim
        self.disagreement_threshold = disagreement_threshold
        
        # 动作语义定义（示例，实际应从配置读取）
        self.action_semantics = {
            0: "move_left",
            1: "move_right", 
            2: "move_forward",
            3: "move_backward"
        }
        
        # 复合动作定义（扩展空间）
        self.composite_actions = {
            4: "stop_and_observe",      # 当左右分歧时
            5: "strategic_retreat",     # 当前后分歧时
            6: "cautious_advance",      # 当一个激进一个保守时
            7: "explore_alternative"    # 当两者都不确定时
        }
        
        # 分歧模式识别
        self.disagreement_patterns = {
            (0, 1): 4,  # left vs right → stop
            (1, 0): 4,  # right vs left → stop
            (2, 3): 5,  # forward vs backward → retreat
            (3, 2): 5,  # backward vs forward → retreat
        }
        
        logger.info(f"[创造性融合] 初始化完成")
        logger.info(f"[创造性融合] 基础空间={base_action_dim}, 扩展空间={expansion_dim}")
        logger.info(f"[创造性融合] 🎯 分歧阈值={disagreement_threshold} (降低以增加创造性触发率)")
    
    def fuse(
        self,
        action_A: int,
        action_B: int,
        conf_A: float,
        conf_B: float,
        context: Optional[Dict[str, Any]] = None
    ) -> CreativeFusionResult:
        """
        创造性融合
        
        Args:
            action_A: 系统A的动作
            action_B: 系统B的动作
            conf_A: 系统A的置信度
            conf_B: 系统B的置信度
            context: 额外上下文
            
        Returns:
            创造性融合结果
        """
        context = context or {}
        
        # 优先检查是否是已知的对立模式
        if (action_A, action_B) in self.disagreement_patterns or (action_B, action_A) in self.disagreement_patterns:
            if self.enable_expansion:
                return self._creative_fusion(action_A, action_B, conf_A, conf_B)
        
        # 1. 计算分歧程度
        disagreement_level = self._assess_disagreement(action_A, action_B)
        
        # 2. 根据分歧程度选择融合策略
        if disagreement_level == DisagreementLevel.CONSENSUS:
            # 一致：直接选择高置信度的
            return self._consensus_fusion(action_A, action_B, conf_A, conf_B)
        
        elif disagreement_level == DisagreementLevel.MILD:
            # 轻微分歧：尝试创造性融合（阈值降低后更容易触发）
            if self.enable_expansion and (action_A, action_B) in self.disagreement_patterns:
                return self._creative_fusion(action_A, action_B, conf_A, conf_B)
            else:
                return self._weighted_fusion(action_A, action_B, conf_A, conf_B)
        
        elif disagreement_level in [DisagreementLevel.MODERATE, DisagreementLevel.STRONG]:
            # 强烈分歧：创造性融合
            if self.enable_expansion and (action_A, action_B) in self.disagreement_patterns:
                return self._creative_fusion(action_A, action_B, conf_A, conf_B)
            else:
                # 无法创造新动作时，选择高置信度的
                return self._confidence_based_selection(action_A, action_B, conf_A, conf_B)
        
        return self._weighted_fusion(action_A, action_B, conf_A, conf_B)
    
    def _assess_disagreement(self, action_A: int, action_B: int) -> DisagreementLevel:
        """评估分歧程度（使用可配置阈值）"""
        diff = abs(action_A - action_B)
        
        if diff == 0:
            return DisagreementLevel.CONSENSUS
        elif diff <= self.disagreement_threshold:  # 🆕 使用阈值
            return DisagreementLevel.MILD
        elif diff <= self.disagreement_threshold * 2:  # 🆕 动态中等阈值
            return DisagreementLevel.MODERATE
        else:
            return DisagreementLevel.STRONG
    
    def _consensus_fusion(
        self, action_A: int, action_B: int, conf_A: float, conf_B: float
    ) -> CreativeFusionResult:
        """一致性融合：两者一致时"""
        action = action_A
        confidence = max(conf_A, conf_B)  # 选择更高的置信度
        
        return CreativeFusionResult(
            action=action,
            confidence=confidence,
            is_creative=False,
            original_space=True,
            reasoning=f"两系统一致选择 action={action}",
            disagreement_level=DisagreementLevel.CONSENSUS,
            component_actions=(action_A, action_B)
        )
    
    def _weighted_fusion(
        self, action_A: int, action_B: int, conf_A: float, conf_B: float
    ) -> CreativeFusionResult:
        """加权融合：轻微分歧时"""
        # 轻微分歧也可能需要创造性融合
        if (action_A, action_B) in self.disagreement_patterns or (action_B, action_A) in self.disagreement_patterns:
            # 即使是轻微分歧，如果是已知的对立模式，也尝试创造
            return self._creative_fusion(action_A, action_B, conf_A, conf_B)
        
        # 基于置信度加权
        if conf_A > conf_B:
            action = action_A
            confidence = conf_A
            reasoning = f"选择高置信度方案A (conf={conf_A:.3f} > {conf_B:.3f})"
        else:
            action = action_B
            confidence = conf_B
            reasoning = f"选择高置信度方案B (conf={conf_B:.3f} > {conf_A:.3f})"
        
        return CreativeFusionResult(
            action=action,
            confidence=confidence,
            is_creative=False,
            original_space=True,
            reasoning=reasoning,
            disagreement_level=DisagreementLevel.MILD,
            component_actions=(action_A, action_B)
        )
    
    def _creative_fusion(
        self, action_A: int, action_B: int, conf_A: float, conf_B: float
    ) -> CreativeFusionResult:
        """创造性融合：强烈分歧时生成新动作"""
        # 查找预定义的复合动作
        composite_action = self.disagreement_patterns.get((action_A, action_B))
        
        if composite_action is not None:
            # 成功生成创造性动作
            confidence = min(conf_A, conf_B) * 0.9  # 保守估计
            action_name = self.composite_actions.get(composite_action, f"composite_{composite_action}")
            
            reasoning = (
                f"检测到强烈分歧: A={self.action_semantics.get(action_A)} "
                f"vs B={self.action_semantics.get(action_B)}, "
                f"生成创造性动作: {action_name}"
            )
            
            logger.info(f"[创造性融合] ✨ {reasoning}")
            
            return CreativeFusionResult(
                action=composite_action,
                confidence=confidence,
                is_creative=True,
                original_space=False,  # 扩展空间
                reasoning=reasoning,
                disagreement_level=DisagreementLevel.STRONG,
                component_actions=(action_A, action_B)
            )
        
        # 无法生成创造性动作，回退到选择模式
        return self._confidence_based_selection(action_A, action_B, conf_A, conf_B)
    
    def _confidence_based_selection(
        self, action_A: int, action_B: int, conf_A: float, conf_B: float
    ) -> CreativeFusionResult:
        """基于置信度的选择：无法创造时回退"""
        if conf_A > conf_B:
            action = action_A
            confidence = conf_A
            reasoning = f"无法创造性融合，选择高置信度A (conf={conf_A:.3f})"
        else:
            action = action_B
            confidence = conf_B
            reasoning = f"无法创造性融合，选择高置信度B (conf={conf_B:.3f})"
        
        return CreativeFusionResult(
            action=action,
            confidence=confidence,
            is_creative=False,
            original_space=True,
            reasoning=reasoning,
            disagreement_level=DisagreementLevel.STRONG,
            component_actions=(action_A, action_B)
        )
    
    def get_action_space_size(self) -> int:
        """获取当前动作空间大小"""
        return self.expansion_dim if self.enable_expansion else self.base_action_dim
    
    def is_extended_action(self, action: int) -> bool:
        """判断是否是扩展动作"""
        return action >= self.base_action_dim
