#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双螺旋决策引擎 v2.0 (Double Helix Decision Engine v2.0)
增强版：集成非线性融合、元学习、辩论式共识

核心特性：
1. 非线性交互：实现1+1>2的真实涌现
2. 元学习优化：自适应调整螺旋参数
3. 辩论式共识：从"融合"进化到"对话"
4. 多模态融合：置信度+熵+响应时间

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v2.0（系统升级版）
"""

import numpy as np
import torch
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

# 导入系统A和B
try:
    from core.seed import TheSeed
except ImportError:
    TheSeed = None

try:
    from core.fractal_intelligence import create_fractal_intelligence
except ImportError:
    create_fractal_intelligence = None

# 导入新组件
try:
    from core.nonlinear_fusion import NonlinearFusionEngine, FusionConfig
except ImportError:
    NonlinearFusionEngine = None
    FusionConfig = None

try:
    from core.meta_learner import MetaLearner, MetaLearningConfig
except ImportError:
    MetaLearner = None
    MetaLearningConfig = None

try:
    from core.dialogue_engine import DialogueEngine
except ImportError:
    DialogueEngine = None

try:
    from core.creative_fusion import CreativeFusionEngine, CreativeFusionResult
except ImportError:
    CreativeFusionEngine = None
    CreativeFusionResult = None

try:
    from core.complementary_analyzer import ComplementaryAnalyzer, ComplementaryAnalysis, SystemPreference
except ImportError:
    ComplementaryAnalyzer = None
    ComplementaryAnalysis = None
    SystemPreference = None

try:
    from core.double_helix_engine_v2_fusion_logic import intelligent_fusion
except ImportError:
    intelligent_fusion = None

logger = logging.getLogger(__name__)


@dataclass
class HelixContext:
    """螺旋上下文"""
    phase: float
    weight_A: float
    weight_B: float
    last_A_output: Optional[np.ndarray]
    last_B_output: Optional[np.ndarray]
    cycle_number: int
    ascent_level: float


@dataclass
class DoubleHelixResult:
    """双螺旋决策结果"""
    action: int
    confidence: float
    weight_A: float
    weight_B: float
    phase: float
    individual_A: Optional[Any]
    individual_B: Optional[Any]
    fusion_method: str
    emergence_score: float
    explanation: str
    response_time_ms: float
    entropy: float = 0.0
    cycle_number: int = 0
    ascent_level: float = 0.0
    # v2新增字段
    dialogue_length: int = 0
    consensus_quality: float = 0.0
    nonlinear_breakdown: Optional[Dict[str, Any]] = None
    complementary_preference: str = 'neutral'  # 🆕 系统偏好：A/B/neutral/creative
    # 🆕 涌现行为验证字段（用于智能观测）
    is_creative: bool = False  # 是否是创造性行为
    original_space: bool = True  # 是否在原始动作空间内
    emergence_quality: float = 0.0  # 涌现质量指标
    # 🔧 P0修复: 添加缺失的字段以支持AGI_Life_Engine.py的访问
    system_a_confidence: Optional[float] = None  # 系统A的置信度
    system_b_confidence: Optional[float] = None  # 系统B的置信度
    reasoning: Optional[str] = None  # 推理过程说明


class FusionMode(Enum):
    """融合模式"""
    LINEAR = "linear"              # 线性加权（原版）
    NONLINEAR = "nonlinear"        # 非线性融合
    DIALOGUE = "dialogue"          # 辩论式共识
    ADAPTIVE = "adaptive"          # 自适应选择


class DoubleHelixEngineV2:
    """
    双螺旋决策引擎 v2.0

    升级特性：
    1. 非线性融合：交互项+互补项+多样性
    2. 元学习：自动优化螺旋参数
    3. 辩论式共识：系统A和B辩论达成共识
    4. 自适应模式：根据场景选择最佳融合方式
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 4,
        device: str = 'cpu',
        # 原有参数
        spiral_radius: float = 0.3,
        phase_shift: float = np.pi,
        phase_speed: float = 0.1,
        cycle_length: int = 10,
        ascent_rate: float = 0.01,
        # v2新增参数
        fusion_mode: FusionMode = FusionMode.ADAPTIVE,
        enable_nonlinear: bool = True,
        enable_meta_learning: bool = True,
        enable_dialogue: bool = False,  # 对话模式较慢，默认关闭
        dialogue_rounds: int = 2,
        adaptive_threshold: float = 0.02,  # 涌现阈值，低于此值启用对话
        # v3新增：动态动作空间
        enable_dynamic_action: bool = True  # 启用动态动作空间扩展
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.enable_dynamic_action = enable_dynamic_action

        # 新增：动态动作空间生成器（Task 12）
        if enable_dynamic_action:
            from core.dynamic_action_space import get_dynamic_action_space, get_hierarchical_action_space
            self.dynamic_action_space = get_dynamic_action_space()
            self.hierarchical_action_space = get_hierarchical_action_space()
            logger.info("[双螺旋v2] 动态动作空间已启用")
        else:
            self.dynamic_action_space = None
            self.hierarchical_action_space = None

        # Task 14: 创造性融合增强
        self.enable_creative_fusion = True
        if self.enable_creative_fusion:
            try:
                from core.creative_fusion_enhanced import get_emergence_detector, get_adaptive_fusion_engine
                self.emergence_detector = get_emergence_detector()
                self.adaptive_fusion_engine = get_adaptive_fusion_engine()
                logger.info("[双螺旋v2] 创造性融合增强已启用")
            except ImportError:
                logger.warning("[双螺旋v2] 无法导入创造性融合增强模块")
                self.emergence_detector = None
                self.adaptive_fusion_engine = None
        else:
            self.emergence_detector = None
            self.adaptive_fusion_engine = None

        # 螺旋参数
        self.spiral_radius = spiral_radius
        self.phase_shift = phase_shift
        self.phase_speed = phase_speed
        self.cycle_length = cycle_length
        self.ascent_rate = ascent_rate

        # v2配置
        self.fusion_mode = fusion_mode
        self.enable_nonlinear = enable_nonlinear
        self.enable_meta_learning = enable_meta_learning
        self.enable_dialogue = enable_dialogue
        self.dialogue_rounds = dialogue_rounds
        self.adaptive_threshold = adaptive_threshold

        # v2.1 纠偏组件
        self.creative_fusion = None
        self.complementary_analyzer = None
        if CreativeFusionEngine is not None:
            self.creative_fusion = CreativeFusionEngine(
                base_action_dim=action_dim,
                enable_expansion=True,
                expansion_dim=action_dim * 2  # 扩展到2倍空间
            )
            logger.info("[双螺旋v2] ✨ 创造性融合引擎已启用")
        
        if ComplementaryAnalyzer is not None:
            self.complementary_analyzer = ComplementaryAnalyzer(
                state_dim=state_dim,
                window_size=100,
                min_samples=10
            )
            logger.info("[双螺旋v2] 🎯 互补分析器已启用")

        # 状态变量
        self.phase = 0.0
        self.decision_count = 0
        self.cycle_number = 1
        self.ascent_level = 0.0

        # 上下文
        self.context = HelixContext(
            phase=0.0,
            weight_A=0.5,
            weight_B=0.5,
            last_A_output=None,
            last_B_output=None,
            cycle_number=1,
            ascent_level=0.0
        )

        # 性能追踪
        self.confidence_history = []
        self.cycle_peaks = []
        self.emergence_history = []

        # 统计
        self.stats = {
            'total_decisions': 0,
            'A_dominant': 0,
            'B_dominant': 0,
            'balanced': 0,
            'avg_emergence': 0.0,
            'avg_confidence': 0.0,
            'cycles_completed': 0,
            # v2新增统计
            'fusion_modes_used': {
                'linear': 0,
                'nonlinear': 0,
                'dialogue': 0
            },
            'meta_optimizations': 0,
            'dialogue_emergence_total': 0.0,
            'nonlinear_emergence_total': 0.0
        }

        # 初始化系统A和B
        self._init_systems()

        # 初始化v2组件
        self._init_v2_components()

        logger.info(f"[双螺旋v2] 引擎初始化完成")
        logger.info(f"[双螺旋v2] 融合模式={fusion_mode.value}")
        logger.info(f"[双螺旋v2] 非线性={enable_nonlinear}, 元学习={enable_meta_learning}, 对话={enable_dialogue}")

    def _init_systems(self):
        """初始化系统A和B"""
        self.seed = None
        if TheSeed:
            try:
                self.seed = TheSeed(
                    state_dim=self.state_dim,
                    action_dim=self.action_dim
                )
                logger.info("[双螺旋v2] 系统A（TheSeed）已启用")
            except Exception as e:
                logger.warning(f"[双螺旋v2] 系统A初始化失败: {e}")

        self.fractal = None
        if create_fractal_intelligence:
            try:
                self.fractal = create_fractal_intelligence(
                    input_dim=self.state_dim,
                    state_dim=self.state_dim,
                    device=self.device
                )
                logger.info("[双螺旋v2] 系统B（分形智能）已启用")
            except Exception as e:
                logger.warning(f"[双螺旋v2] 系统B初始化失败: {e}")

    def _init_v2_components(self):
        """初始化v2新组件"""

        # 1. 非线性融合引擎
        self.nonlinear_fusion = None
        if self.enable_nonlinear and NonlinearFusionEngine:
            try:
                config = FusionConfig(
                    interaction_strength=0.15,
                    complementarity_strength=0.08,
                    diversity_bonus=0.05
                )
                self.nonlinear_fusion = NonlinearFusionEngine(config=config)
                logger.info("[双螺旋v2] 非线性融合引擎已启用")
            except Exception as e:
                logger.warning(f"[双螺旋v2] 非线性融合初始化失败: {e}")

        # 2. 元学习器
        self.meta_learner = None
        if self.enable_meta_learning and MetaLearner:
            try:
                config = MetaLearningConfig(
                    learning_rate=0.01,
                    optimization_interval=20
                )
                self.meta_learner = MetaLearner(config=config, device=self.device)
                logger.info("[双螺旋v2] 元学习器已启用")
            except Exception as e:
                logger.warning(f"[双螺旋v2] 元学习器初始化失败: {e}")

        # 3. 对话引擎
        self.dialogue_engine = None
        if self.enable_dialogue and DialogueEngine:
            try:
                self.dialogue_engine = DialogueEngine()
                logger.info("[双螺旋v2] 对话引擎已启用")
            except Exception as e:
                logger.warning(f"[双螺旋v2] 对话引擎初始化失败: {e}")

    def decide(
        self,
        state: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        last_reward: Optional[float] = None  # 用于更新互补分析器
    ) -> DoubleHelixResult:
        """
        双螺旋决策（v2.1纠偏版）

        核心改进：
        1. 优先识别互补区域 - 谁擅长就用谁
        2. 强烈分歧时创造性融合 - 生成新动作
        3. 最后才是数值融合 - 作为保底策略

        Args:
            state: 当前状态
            context: 额外上下文
            last_reward: 上一步的奖励（用于更新表现）

        Returns:
            决策结果
        """
        start_time = time.time()
        context = context or {}
        self.decision_count += 1
        self.stats['total_decisions'] += 1

        # 步骤0：动态动作空间扩展（Task 38）
        if self.enable_dynamic_action and self.dynamic_action_space is not None:
            novelty_required = context.get('novelty_required', 0.0)
            task_complexity = context.get('task_complexity', 0.5)

            # 高新颖性或高复杂度时扩展动作空间
            if novelty_required > 0.7 or task_complexity > 0.8:
                expanded_actions = self.dynamic_action_space.expand_action_space(
                    context={'novelty_required': novelty_required, 'task_complexity': task_complexity}
                )
                old_dim = self.action_dim
                self.action_dim = expanded_actions.shape[0]
                logger.debug(
                    f"[双螺旋v2] 动作空间扩展: "
                    f"{old_dim}D → {self.action_dim}D "
                    f"(新颖性={novelty_required:.2f}, 复杂度={task_complexity:.2f})"
                )

        # 步骤1：计算相位和权重
        self._update_phase()

        # 步骤2：系统A和B并行决策
        result_A = self._decide_A(state, context)
        result_B = self._decide_B(state, context)

        # 步骤2.5：互补区域分析（新增）
        complementary_analysis = None
        if self.complementary_analyzer is not None:
            complementary_analysis = self.complementary_analyzer.analyze(
                state=state,
                result_A=result_A,
                result_B=result_B
            )

        # 步骤2.6：动作层级提升（Task 38新增）
        if self.enable_dynamic_action and self.hierarchical_action_space is not None:
            from core.dynamic_action_space import ActionLevel
            novelty = context.get('novelty_required', 0.0)

            # 高新颖性时使用高层级动作
            if novelty > 0.8:
                target_level = ActionLevel.META  # 元动作
            elif novelty > 0.6:
                target_level = ActionLevel.ABSTRACT  # 抽象动作
            elif novelty > 0.4:
                target_level = ActionLevel.COMPOSITE  # 复合动作
            else:
                target_level = ActionLevel.PRIMITIVE  # 基础动作

            # 获取高层级动作空间维度
            high_level_dim = self.hierarchical_action_space.get_action_space(target_level)

            if high_level_dim > self.action_dim:
                logger.debug(
                    f"[双螺旋v2] 动作层级提升: "
                    f"从primitive({self.action_dim}D) → {target_level.value}({high_level_dim}D)"
                )
                self.action_dim = high_level_dim

        # 步骤3：智能融合策略选择（新改进）
        if intelligent_fusion is not None:
            # 使用v2.1的智能融合逻辑
            fused_result = intelligent_fusion(
                engine=self,
                result_A=result_A,
                result_B=result_B,
                state=state,
                complementary_analysis=complementary_analysis
            )
            # 🚨 修复：intelligent_fusion已返回有效结果，跳过备用逻辑
            selected_mode = None  # 标记已使用智能融合
        else:
            # 回退到原有逻辑
            selected_mode = self._select_fusion_mode(result_A, result_B)
            fused_result = None  # 需要后续融合

        # 步骤4：执行融合（仅当智能融合未处理时）
        if selected_mode is not None:  # 🚨 仅当未使用智能融合时执行
            if selected_mode == FusionMode.DIALOGUE and self.dialogue_engine:
                fused_result = self._fuse_with_dialogue(result_A, result_B)
                self.stats['fusion_modes_used']['dialogue'] += 1
                self.stats['dialogue_emergence_total'] += fused_result['emergence']
            elif selected_mode == FusionMode.NONLINEAR and self.nonlinear_fusion:
                fused_result = self._fuse_with_nonlinear(result_A, result_B)
                self.stats['fusion_modes_used']['nonlinear'] += 1
                self.stats['nonlinear_emergence_total'] += fused_result['emergence']
            else:
                fused_result = self._fuse_linear(result_A, result_B)
                self.stats['fusion_modes_used']['linear'] += 1

        # 步骤4.5：涌现行为检测（Task 14新增）
        if self.enable_creative_fusion and self.emergence_detector is not None:
            is_emergent, emergence_score, metrics = self.emergence_detector.detect_emergence(
                fused_output=fused_result,
                individual_A=result_A,
                individual_B=result_B,
                context={'task_complexity': context.get('task_complexity', 0.5)}
            )

            if is_emergent:
                logger.info(
                    f"[双螺旋v2] 检测到涌现行为！"
                    f"提升={emergence_score:.2%}, "
                    f"新颖性贡献={metrics['novelty_contribution']:.2%}"
                )
                self.stats['emergence_detected'] = self.stats.get('emergence_detected', 0) + 1

        # 步骤5：更新上下文
        self._update_context(result_A, result_B)

        # 步骤6：检测周期完成和螺旋上升
        self._check_cycle_completion(fused_result['confidence'])

        # 步骤7：元学习记录
        if self.meta_learner:
            current_params = {
                'spiral_radius': self.spiral_radius,
                'phase_speed': self.phase_speed,
                'ascent_rate': self.ascent_rate
            }
            self.meta_learner.record_decision(
                state={
                    'phase': self.context.phase,
                    'weight_A': self.context.weight_A,
                    'weight_B': self.context.weight_B
                },
                fusion_params=current_params,
                reward=fused_result['confidence'],
                emergence=fused_result['emergence']
            )

            # 定期更新参数
            if self.decision_count % 50 == 0:
                suggested_params = self.meta_learner.get_suggested_parameters()
                self._update_parameters_from_meta(suggested_params)

        # 步骤7：更新互补分析器(如果有奖励)
        if last_reward is not None and self.complementary_analyzer is not None:
            # 更新上一次决策的表现
            if fused_result.get('selected_system') == 'A' and result_A:
                self.complementary_analyzer.update_performance(state, 'A', last_reward)
            elif fused_result.get('selected_system') == 'B' and result_B:
                self.complementary_analyzer.update_performance(state, 'B', last_reward)

        # 步骤8：统计
        response_time = (time.time() - start_time) * 1000
        self._update_stats(fused_result)

        # 🆕 从fused_result中提取系统偏好
        selected_system = fused_result.get('selected_system', None)
        if selected_system == 'A':
            complementary_pref = 'A'
        elif selected_system == 'B':
            complementary_pref = 'B'
        elif fused_result.get('is_creative', False):
            complementary_pref = 'creative'
        else:
            complementary_pref = 'neutral'

        # 🔧 P0修复: 提取系统A和B的置信度
        system_a_conf = result_A.get('confidence') if result_A else None
        system_b_conf = result_B.get('confidence') if result_B else None

        return DoubleHelixResult(
            action=fused_result['action'],
            confidence=fused_result['confidence'],
            weight_A=self.context.weight_A,
            weight_B=self.context.weight_B,
            phase=self.context.phase,
            individual_A=result_A,
            individual_B=result_B,
            fusion_method=fused_result['method'],
            emergence_score=fused_result['emergence'],
            explanation=self._generate_explanation_v2(fused_result, selected_mode),
            response_time_ms=response_time,
            entropy=fused_result.get('entropy', 0.0),
            cycle_number=self.context.cycle_number,
            ascent_level=self.context.ascent_level,
            dialogue_length=fused_result.get('dialogue_length', 0),
            consensus_quality=fused_result.get('consensus_quality', 0.0),
            nonlinear_breakdown=fused_result.get('breakdown'),
            complementary_preference=complementary_pref,  # 🆕 系统偏好
            # 🆕 涌现行为验证标志
            is_creative=fused_result.get('is_creative', False),
            original_space=fused_result.get('original_space', True),
            emergence_quality=fused_result.get('emergence', 0.0),  # 使用emergence作为quality
            # 🔧 P0修复: 填充缺失字段
            system_a_confidence=system_a_conf,  # 系统A置信度
            system_b_confidence=system_b_conf,  # 系统B置信度
            reasoning=fused_result.get('reasoning', self._generate_explanation_v2(fused_result, selected_mode))  # 推理过程
        )

    def _select_fusion_mode(
        self,
        result_A: Optional[Dict[str, Any]],
        result_B: Optional[Dict[str, Any]]
    ) -> FusionMode:
        """自适应选择融合模式"""

        if self.fusion_mode == FusionMode.ADAPTIVE:
            # 检查最近涌现表现
            if len(self.emergence_history) >= 10:
                recent_avg_emergence = np.mean(self.emergence_history[-10:])
                # 如果涌现持续很低，启用对话模式
                if recent_avg_emergence < self.adaptive_threshold and self.dialogue_engine:
                    return FusionMode.DIALOGUE
                # 否则使用非线性融合
                elif self.nonlinear_fusion:
                    return FusionMode.NONLINEAR

            # 默认使用非线性
            if self.nonlinear_fusion:
                return FusionMode.NONLINEAR
            else:
                return FusionMode.LINEAR
        else:
            return self.fusion_mode

    def _fuse_with_nonlinear(
        self,
        result_A: Optional[Dict[str, Any]],
        result_B: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """非线性融合"""

        if result_A is None or result_B is None:
            return self._fuse_linear(result_A, result_B)

        fusion_result = self.nonlinear_fusion.fuse(
            result_A, result_B,
            self.context.weight_A,
            self.context.weight_B,
            context={'phase': self.context.phase}
        )

        # 添加螺旋上升加成
        final_confidence = min(1.0, fusion_result['confidence'] + self.context.ascent_level)

        return {
            'action': fusion_result['action'],
            'confidence': final_confidence,
            'method': fusion_result['method'],
            'emergence': fusion_result['emergence'],
            'entropy': self._calculate_entropy(result_A, result_B),
            'breakdown': fusion_result['breakdown'],
            # 🆕 涌现行为标志
            'is_creative': fusion_result['emergence'] > 0.3,  # 涌现>0.3视为创造性
            'original_space': True  # 非线性融合仍在原始空间
        }

    def _fuse_with_dialogue(
        self,
        result_A: Optional[Dict[str, Any]],
        result_B: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """辩论式共识"""

        if result_A is None or result_B is None:
            return self._fuse_linear(result_A, result_B)

        consensus = self.dialogue_engine.engage_dialogue(
            result_A, result_B,
            context={'phase': self.context.phase}
        )

        # 添加螺旋上升加成
        final_confidence = min(1.0, consensus.confidence + self.context.ascent_level)

        return {
            'action': consensus.action,
            'confidence': final_confidence,
            'method': 'dialogue_consensus',
            'emergence': consensus.emergence,
            'entropy': self._calculate_entropy(result_A, result_B),
            'dialogue_length': consensus.dialogue_length,
            'consensus_quality': consensus.consensus_quality,
            'breakdown': consensus.breakdown,
            # 🆕 涌现行为标志
            'is_creative': consensus.emergence > 0.4,  # 对话涌现>0.4视为创造性
            'original_space': True  # 对话融合仍在原始空间
        }

    def _fuse_linear(
        self,
        result_A: Optional[Dict[str, Any]],
        result_B: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """线性融合（原版）"""

        if result_A is None and result_B is None:
            return self._get_fallback()
        elif result_A is None:
            return {
                'action': result_B['action'],
                'confidence': result_B['confidence'],
                'method': 'B_only',
                'emergence': 0.0,
                'entropy': 0.5,
                # 🆕 涌现行为标志
                'is_creative': False,
                'original_space': True
            }
        elif result_B is None:
            return {
                'action': result_A['action'],
                'confidence': result_A['confidence'],
                'method': 'A_only',
                'emergence': 0.0,
                'entropy': 0.5,
                # 🆕 涌现行为标志
                'is_creative': False,
                'original_space': True
            }

        weight_A = self.context.weight_A
        weight_B = self.context.weight_B

        fused_action = int(weight_A * result_A['action'] + weight_B * result_B['action'])
        base_confidence = weight_A * result_A['confidence'] + weight_B * result_B['confidence']
        max_individual_confidence = max(result_A['confidence'], result_B['confidence'])
        real_synergy = base_confidence - max_individual_confidence
        ascent_bonus = self.context.ascent_level
        fused_confidence = min(1.0, base_confidence + ascent_bonus)
        emergence_score = max(0.0, real_synergy)

        if abs(weight_A - weight_B) < 0.1:
            method = 'linear_balanced'
        elif weight_A > weight_B:
            method = 'linear_A_dominant'
        else:
            method = 'linear_B_dominant'

        return {
            'action': fused_action,
            'confidence': fused_confidence,
            'method': method,
            'emergence': emergence_score,
            'entropy': self._calculate_entropy(result_A, result_B),
            # 🆕 涌现行为标志
            'is_creative': False,  # 线性融合不是创造性的
            'original_space': True  # 线性融合在原始空间
        }

    def _update_parameters_from_meta(self, suggested_params: Dict[str, float]):
        """从元学习器更新参数"""

        old_params = {
            'spiral_radius': self.spiral_radius,
            'phase_speed': self.phase_speed,
            'ascent_rate': self.ascent_rate
        }

        self.spiral_radius = suggested_params['spiral_radius']
        self.phase_speed = suggested_params['phase_speed']
        self.ascent_rate = suggested_params['ascent_rate']

        self.stats['meta_optimizations'] += 1

        logger.info(f"[双螺旋v2] 元学习优化 #{self.stats['meta_optimizations']}")
        logger.info(f"[双螺旋v2] 旧参数: {old_params}")
        logger.info(f"[双螺旋v2] 新参数: {suggested_params}")

    # 继承原有方法
    def _update_phase(self):
        """更新相位和权重"""
        self.context.weight_A = 0.5 + self.spiral_radius * np.cos(self.phase)
        self.context.weight_B = 0.5 + self.spiral_radius * np.cos(self.phase + self.phase_shift)
        self.context.weight_A = max(0.0, self.context.weight_A)
        self.context.weight_B = max(0.0, self.context.weight_B)
        total_weight = self.context.weight_A + self.context.weight_B
        if total_weight > 0:
            self.context.weight_A /= total_weight
            self.context.weight_B /= total_weight
        self.context.phase = self.phase
        self.phase += self.phase_speed

    def _decide_A(self, state: np.ndarray, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """系统A决策"""
        if not self.seed:
            return None
        try:
            enhanced_state = self._enhance_state_A(state, context)
            action = self.seed.act(enhanced_state)
            _, uncertainty = self.seed.predict(enhanced_state, action)
            # 🔧 [2026-01-17] 关键修复: seed.predict()返回的是uncertainty(不确定性)
            # 必须转换为confidence: confidence = 1 - uncertainty
            confidence = float(np.clip(1.0 - uncertainty, 0, 1))
            logger.debug(f"[DEBUG-A] uncertainty={uncertainty:.4f} → confidence={confidence:.4f}")
            return {'action': int(action), 'confidence': confidence, 'system': 'A'}
        except Exception as e:
            logger.warning(f"[双螺旋v2] 系统A决策失败: {e}")
            return None

    def _decide_B(self, state: np.ndarray, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """系统B决策"""
        if not self.fractal:
            return None
        try:
            enhanced_state = self._enhance_state_B(state, context)
            state_tensor = torch.FloatTensor(enhanced_state).unsqueeze(0).to(self.device)

            # 🔧 P1修复: 使用decide()方法获取动态置信度，而不是直接调用forward()
            if hasattr(self.fractal, 'decide'):
                # 🔧 [2026-01-17] 修复调试日志，避免访问dict的shape属性
                logger.debug(f"[DEBUG-B0] _decide_B called")
                logger.debug(f"[DEBUG-B0] enhanced_state shape: {enhanced_state.shape}")
                logger.debug(f"[DEBUG-B0] state_tensor shape: {state_tensor.shape}")

                # 使用decide方法，它返回动态计算的confidence
                output, info = self.fractal.decide(state_tensor)

                # 🔧 [2026-01-17] 简化调试日志
                logger.debug(f"[DEBUG-B2] fractal.decide() returned, info keys: {info.keys() if isinstance(info, dict) else 'N/A'}")

                action = output.argmax().item() if output.dim() > 0 else int(output.item())
                confidence_raw = info.get('confidence', 0.5)  # 从decide获取动态confidence

                logger.debug(f"[DEBUG-B2] action: {action}, confidence: {float(confidence_raw):.4f}")

                confidence = confidence_raw
            else:
                logger.warning(f"[DEBUG-B2] fractal.decide() NOT found, using fallback")
                # 回退到原方案
                output = self.fractal.core.forward(state_tensor)
                # 尝试从FractalOutput获取meta信息
                if hasattr(output, 'entropy'):
                    # FractalOutput对象，但没有action字段，需要从output tensor生成
                    action_tensor = output.output if hasattr(output, 'output') else output
                    action = action_tensor.argmax().item() if action_tensor.dim() > 0 else int(action_tensor.item())
                    # 使用self_awareness作为confidence
                    if hasattr(output, 'self_awareness'):
                        confidence = output.self_awareness.mean().item()
                    else:
                        confidence = 0.5
                else:
                    # 纯tensor
                    action = output.argmax().item() if output.dim() > 0 else int(output.item())
                    confidence = 0.5

            logger.info(f"[DEBUG-B3] Returning to double_helix: action={action}, confidence={confidence:.6f}")
            return {'action': int(action), 'confidence': float(confidence), 'system': 'B'}
        except Exception as e:
            logger.error(f"[双螺旋v2] 系统B决策失败: {e}", exc_info=True)
            return None

    def _normalize_state(self, state: Any) -> np.ndarray:
        """
        🆕 [2026-01-17] P0修复：状态输入标准化
        
        将各种输入类型转换为numpy数组
        """
        if isinstance(state, np.ndarray):
            return state.flatten() if state.ndim > 1 else state
        elif isinstance(state, dict):
            # 从字典提取数值
            values = []
            for v in state.values():
                if isinstance(v, (int, float)):
                    values.append(float(v))
                elif isinstance(v, (list, tuple)):
                    values.extend([float(x) for x in v if isinstance(x, (int, float))])
            if not values:
                values = [0.0] * self.state_dim
            arr = np.array(values, dtype=np.float32)
            # 填充或截断到state_dim
            if len(arr) < self.state_dim:
                arr = np.pad(arr, (0, self.state_dim - len(arr)))
            elif len(arr) > self.state_dim:
                arr = arr[:self.state_dim]
            return arr
        elif isinstance(state, (list, tuple)):
            arr = np.array(state, dtype=np.float32).flatten()
            if len(arr) < self.state_dim:
                arr = np.pad(arr, (0, self.state_dim - len(arr)))
            elif len(arr) > self.state_dim:
                arr = arr[:self.state_dim]
            return arr
        else:
            # 单个值，填充为state_dim
            return np.full(self.state_dim, float(state) if isinstance(state, (int, float)) else 0.0, dtype=np.float32)

    def _enhance_state_A(self, state: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """增强系统A的状态"""
        # 🆕 [2026-01-17] P0修复：确保state是正确格式
        state = self._normalize_state(state)
        
        if self.context.last_B_output is not None:
            alpha = 0.7
            beta = 0.3
            enhanced = alpha * state + beta * self.context.last_B_output
            return enhanced
        return state

    def _enhance_state_B(self, state: np.ndarray, context: Dict[str, Any]) -> np.ndarray:
        """增强系统B的状态"""
        # 🆕 [2026-01-17] P0修复：确保state是正确格式
        state = self._normalize_state(state)
        
        if self.context.last_A_output is not None:
            alpha = 0.7
            beta = 0.3
            enhanced = alpha * state + beta * self.context.last_A_output
            return enhanced
        return state

    def _update_context(self, result_A, result_B):
        """更新上下文"""
        if result_A is not None:
            self.context.last_A_output = np.zeros(self.state_dim)
            self.context.last_A_output[result_A['action']] = result_A['confidence']
        if result_B is not None:
            self.context.last_B_output = np.zeros(self.state_dim)
            self.context.last_B_output[result_B['action']] = result_B['confidence']

    def _check_cycle_completion(self, confidence: float):
        """检测周期完成和螺旋上升"""
        self.confidence_history.append(confidence)
        if self.decision_count % self.cycle_length == 0:
            cycle_peak = max(self.confidence_history[-self.cycle_length:])
            self.cycle_peaks.append(cycle_peak)
            if len(self.cycle_peaks) >= 2:
                improvement = self.cycle_peaks[-1] - self.cycle_peaks[-2]
                if improvement > 0:
                    self.ascent_level += self.ascent_rate
                    self.context.ascent_level = self.ascent_level
                    logger.info(f"[双螺旋v2] 周期{self.cycle_number}完成，峰值提升{improvement:.4f}，上升至{self.ascent_level:.4f}")
            self.cycle_number += 1
            self.context.cycle_number = self.cycle_number
            self.stats['cycles_completed'] += 1

    def _calculate_entropy(self, result_A, result_B) -> float:
        """计算熵"""
        if result_A is None or result_B is None:
            return 0.0
        action_diff = abs(result_A['action'] - result_B['action'])
        confidence_diff = abs(result_A['confidence'] - result_B['confidence'])
        entropy = (action_diff / self.action_dim) * 0.5 + (confidence_diff * 0.5)
        return entropy

    def _update_stats(self, result: Dict[str, Any]):
        """更新统计"""
        weight_A = self.context.weight_A
        weight_B = self.context.weight_B
        if abs(weight_A - weight_B) < 0.1:
            self.stats['balanced'] += 1
        elif weight_A > weight_B:
            self.stats['A_dominant'] += 1
        else:
            self.stats['B_dominant'] += 1
        emergence = result['emergence']
        if len(self.emergence_history) > 0:
            self.stats['avg_emergence'] = (
                self.stats['avg_emergence'] * len(self.emergence_history) + emergence
            ) / (len(self.emergence_history) + 1)
        else:
            self.stats['avg_emergence'] = emergence
        self.emergence_history.append(emergence)
        if len(self.confidence_history) > 0:
            self.stats['avg_confidence'] = np.mean(self.confidence_history)

    def _generate_explanation_v2(self, result: Dict[str, Any], mode: Optional[FusionMode]) -> str:
        """生成解释（v2版）"""
        mode_str = mode.value if mode is not None else result.get('method', 'intelligent_fusion')
        explanation = f"双螺旋v2融合 | 模式={mode_str} | 相位={self.context.phase:.2f}"
        explanation += f" | A权重={self.context.weight_A:.2f} B权重={self.context.weight_B:.2f}"

        if result['emergence'] > 0.01:
            explanation += f" | 涌现+{result['emergence']:.4f}"

        if self.context.ascent_level > 0:
            explanation += f" | 上升层级={self.context.ascent_level:.4f}"

        if 'dialogue_length' in result and result['dialogue_length'] > 0:
            explanation += f" | 对话轮数={result['dialogue_length']}"

        if 'consensus_quality' in result and result['consensus_quality'] > 0:
            explanation += f" | 共识质量={result['consensus_quality']:.2f}"

        return explanation

    def _get_fallback(self) -> Dict[str, Any]:
        """兜底决策"""
        return {
            'action': 0,
            'confidence': 0.5,
            'method': 'fallback',
            'emergence': 0.0,
            'entropy': 1.0,
            # 🆕 涌现行为标志
            'is_creative': False,
            'original_space': True
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        base_stats = {
            **self.stats,
            'current_phase': self.context.phase,
            'current_weight_A': self.context.weight_A,
            'current_weight_B': self.context.weight_B,
            'cycle_number': self.context.cycle_number,
            'ascent_level': self.context.ascent_level,
            'cycle_peaks': self.cycle_peaks[-5:] if self.cycle_peaks else [],
            'recent_emergence': self.emergence_history[-10:] if self.emergence_history else [],
            'meta_optimizations': self.stats['meta_optimizations']
        }

        # 添加v2组件统计
        if self.nonlinear_fusion:
            base_stats['nonlinear_fusion'] = self.nonlinear_fusion.get_statistics()

        if self.meta_learner:
            base_stats['meta_learning'] = self.meta_learner.get_statistics()

        if self.dialogue_engine:
            base_stats['dialogue'] = self.dialogue_engine.get_statistics()

        return base_stats


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "双螺旋决策引擎 v2.0 测试")
    print("="*70)

    engine = DoubleHelixEngineV2(
        state_dim=64,
        action_dim=4,
        fusion_mode=FusionMode.ADAPTIVE,
        enable_nonlinear=True,
        enable_meta_learning=True,
        enable_dialogue=True,
        adaptive_threshold=0.02
    )

    print(f"\n[初始化] 双螺旋v2引擎创建成功")
    print(f"[配置] 融合模式=adaptive, 非线性=True, 元学习=True, 对话=True")

    # 执行50次决策
    print(f"\n[测试] 执行50次决策...")
    print("="*70)

    for i in range(50):
        state = np.random.randn(64)
        result = engine.decide(state)

        if (i + 1) % 10 == 0:
            print(f"\n决策 {i+1}/50:")
            print(f"  相位: {result.phase:.2f}")
            print(f"  权重: A={result.weight_A:.2f} B={result.weight_B:.2f}")
            print(f"  融合方法: {result.fusion_method}")
            print(f"  置信度: {result.confidence:.4f}")
            print(f"  涌现分数: {result.emergence_score:.4f}")

            if result.dialogue_length > 0:
                print(f"  对话轮数: {result.dialogue_length}")
                print(f"  共识质量: {result.consensus_quality:.2f}")

    # 显示统计
    print("\n" + "="*70)
    print(" "*25 + "统计信息")
    print("="*70)

    stats = engine.get_statistics()
    print(f"\n总决策数: {stats['total_decisions']}")
    print(f"A主导: {stats['A_dominant']}")
    print(f"B主导: {stats['B_dominant']}")
    print(f"均衡: {stats['balanced']}")
    print(f"平均涌现分数: {stats['avg_emergence']:.4f}")
    print(f"平均置信度: {stats['avg_confidence']:.4f}")
    print(f"完成周期: {stats['cycles_completed']}")
    print(f"上升层级: {stats['ascent_level']:.4f}")

    print(f"\n融合模式分布:")
    for mode, count in stats['fusion_modes_used'].items():
        print(f"  {mode}: {count}次")

    print(f"\n元学习优化: {stats['meta_optimizations']}次")

    if 'nonlinear_fusion' in stats:
        nf_stats = stats['nonlinear_fusion']
        print(f"\n非线性融合统计:")
        print(f"  总融合次数: {nf_stats['total_fusions']}")
        print(f"  平均涌现: {nf_stats['avg_emergence']:.4f}")
        print(f"  涌现率: {nf_stats['emergence_rate']:.2%}")

    if 'dialogue' in stats:
        d_stats = stats['dialogue']
        print(f"\n对话引擎统计:")
        print(f"  总对话数: {d_stats['total_dialogues']}")
        print(f"  平均对话长度: {d_stats['avg_dialogue_length']:.1f}")
        print(f"  共识率: {d_stats['consensus_rate']:.2%}")
        print(f"  平均涌现: {d_stats['avg_emergence']:.4f}")

    print("\n" + "="*70 + "\n")
