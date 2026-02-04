#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合决策引擎 (Hybrid Decision Engine)
融合系统A（组件组装）和系统B（分形拓扑）的决策能力

核心功能：
1. 三路决策：Fractal（快）→ TheSeed（中）→ LLM（慢）
2. 自适应阈值：动态调整决策路径
3. 置信度学习：从决策结果中学习
4. 元学习：MetaLearner优化决策策略

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v1.0
"""

import numpy as np
import torch
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# 导入系统A和B
try:
    from core.seed import TheSeed, Experience
except ImportError:
    TheSeed = None
    Experience = None

try:
    from core.fractal_intelligence import create_fractal_intelligence, FractalOutput
except ImportError:
    create_fractal_intelligence = None

try:
    from core.llm_client import LLMService
except ImportError:
    LLMService = None

# 导入双螺旋引擎 (v2.1纠偏版)
try:
    from core.double_helix_engine_v2 import DoubleHelixEngineV2
except ImportError:
    DoubleHelixEngineV2 = None

# 🆕 [P0级优化] 导入决策缓存
try:
    from core.decision_cache import DecisionCache
except ImportError:
    DecisionCache = None

logger = logging.getLogger(__name__)


class DecisionPath(Enum):
    """决策路径"""
    FRACTAL = "fractal"      # 系统B：最快，10-15ms
    SEED = "seed"            # 系统A：中等，50-100ms
    LLM = "llm"              # 外部LLM：最慢，200-2000ms


@dataclass
class DecisionResult:
    """决策结果"""
    action: int
    confidence: float
    path: DecisionPath
    response_time_ms: float
    explanation: str
    entropy: float = 0.0
    needs_validation: bool = False
    metadata: Dict[str, Any] = None


class HybridDecisionEngine:
    """
    混合决策引擎

    三路决策策略：
    1. Fractal（系统B）- 极速本地决策（10-15ms）
    2. TheSeed（系统A）- DQN增强决策（50-100ms）
    3. LLM（外部）- 复杂推理决策（200-2000ms）
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 4,
        device: str = 'cpu',
        enable_fractal: bool = True,
        enable_llm: bool = False,  # 默认禁用LLM以降低成本
        decision_mode: str = 'round_robin'  # 新增：决策模式
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.enable_fractal = enable_fractal
        self.enable_llm = enable_llm
        self.decision_mode = decision_mode  # 'adaptive', 'round_robin', 'confidence_based'

        # 1. 初始化系统B：分形智能（最快）
        self.fractal = None
        if enable_fractal and create_fractal_intelligence:
            try:
                self.fractal = create_fractal_intelligence(
                    input_dim=state_dim,
                    state_dim=state_dim,
                    device=device
                )
                logger.info("[Hybrid] 系统B（分形智能）已启用")
            except Exception as e:
                logger.warning(f"[Hybrid] 系统B初始化失败: {e}")
                self.enable_fractal = False

        # 2. 初始化系统A：TheSeed（中等速度）
        self.seed = None
        if TheSeed:
            try:
                self.seed = TheSeed(state_dim=state_dim, action_dim=action_dim)
                logger.info("[Hybrid] 系统A（TheSeed）已启用")
            except Exception as e:
                logger.warning(f"[Hybrid] 系统A初始化失败: {e}")
        else:
            logger.warning("[Hybrid] TheSeed不可用")

        # 3. LLM服务（可选）
        self.llm_service = None
        if enable_llm and LLMService:
            try:
                self.llm_service = LLMService()
                logger.info("[Hybrid] LLM服务已启用")
            except Exception as e:
                logger.warning(f"[Hybrid] LLM初始化失败: {e}")
                self.enable_llm = False

        # 4. 自适应阈值管理（提高初始阈值）
        self.confidence_history: List[float] = []
        self.adaptive_threshold = 0.55  # 🔧 修复：提高到0.55，给系统A机会
        self.threshold_window = 100
        # 🆕 [P0优化] 动态阈值范围
        self.threshold_range = (0.4, 0.7)  # 动态范围：0.4-0.7
        self.min_threshold = 0.4
        self.max_threshold = 0.7
        # 🆕 [P0优化] 奖励历史（用于动态阈值调整）
        self.reward_history: List[float] = []  # 保存最近100次奖励

        # 5. 决策统计
        self.stats = {
            'total_decisions': 0,
            'fractal_decisions': 0,
            'seed_decisions': 0,
            'llm_decisions': 0,
            'cache_decisions': 0,  # 🆕 新增：缓存决策统计
            'avg_confidence': 0.0,
            'avg_response_time': 0.0
        }

        # 6. 轮询计数器（用于round_robin模式）
        self.round_robin_counter = 0

        # 🆕 [P0级优化] 决策缓存层
        self.decision_cache = None
        if DecisionCache:
            self.decision_cache = DecisionCache(
                max_size=1000,
                similarity_threshold=0.85,
                ttl_seconds=3600
            )
            logger.info("[Hybrid] [P0优化] 决策缓存已启用 (0ms延迟)")

        # 🆕 [P0级优化] 强制本地模式配置
        self.force_local_mode = True  # 配置项：强制使用本地决策
        self.max_llm_latency = 500.0  # ms：LLM最大可接受延迟
        self.cache_fallback = True  # 配置项：缓存回退

        # 7. 双螺旋引擎v2.1（用于double_helix模式）- 包含创造性融合
        self.helix_engine = None
        if decision_mode == 'double_helix' and DoubleHelixEngineV2:
            try:
                self.helix_engine = DoubleHelixEngineV2(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    device=device,
                    spiral_radius=0.3,
                    phase_shift=np.pi,
                    phase_speed=0.1,
                    cycle_length=10,
                    ascent_rate=0.01,
                    enable_nonlinear=True,      # 启用非线性融合
                    enable_meta_learning=True,  # 启用元学习
                    enable_dialogue=False       # 暂不启用对话引擎
                )
                logger.info("[Hybrid] 🚀 双螺旋引擎v2.1已启用 - 包含创造性融合和互补协同")
            except Exception as e:
                logger.warning(f"[Hybrid] 双螺旋引擎v2.1初始化失败: {e}")
                logger.info("[Hybrid] 回退到round_robin模式")
                self.decision_mode = 'round_robin'

        logger.info(f"[Hybrid] 混合决策引擎初始化完成 (决策模式={decision_mode})")

    def decide(
        self,
        state: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
        force_path: Optional[DecisionPath] = None
    ) -> DecisionResult:
        """
        混合决策（修复版）

        决策模式：
        1. double_helix：双螺旋模式，系统A和B相互缠绕（新增）
        2. round_robin（默认）：强制轮询A和B，确保都使用
        3. adaptive：基于置信度自适应选择
        4. confidence_based：传统阈值模式
        """
        self.stats['total_decisions'] += 1
        context = context or {}
        self.round_robin_counter += 1

        # 🆕 [P0优化] 快速路径0：决策缓存检查（0ms延迟）
        if self.decision_cache and not force_path:
            # 生成状态hash作为缓存key
            state_hash = hash(state.tobytes())

            # 尝试从缓存获取决策结果
            cached_result = self.decision_cache.get(state.flatten() if hasattr(state, 'flatten') else state)
            if cached_result and cached_result[1] > 0.85:  # (intent, confidence) 且置信度 > 0.85
                intent, confidence, metadata = cached_result
                self.stats['cache_decisions'] += 1

                logger.debug(
                    f"[Hybrid] [缓存命中] "
                    f"intent={intent}, "
                    f"confidence={confidence:.3f}, "
                    f"cache_decisions={self.stats['cache_decisions']}"
                )

                # 构造DecisionResult
                return DecisionResult(
                    action=int(self._intent_to_action(intent, state)),
                    confidence=confidence,
                    path=DecisionPath.SEED,  # 缓存结果标记为SEED路径
                    response_time_ms=0.001,  # ~0ms（缓存命中）
                    explanation=f"[缓存] {intent}",
                    metadata={'cached': True, **metadata}
                )

        # 🔧 修复：实现真正的混合决策
        if force_path is not None:
            # 强制指定路径
            return self._decide_by_path(force_path, state, context)

        elif self.decision_mode == 'double_helix' and self.helix_engine:
            # 🧬 双螺旋模式：系统A和B相互缠绕，激发智慧涌现
            helix_result = self.helix_engine.decide(state, context)

            # 转换为DecisionResult格式
            result = DecisionResult(
                action=helix_result.action,
                confidence=helix_result.confidence,
                path=DecisionPath.SEED,  # 双螺旋融合，标记为SEED
                response_time_ms=helix_result.response_time_ms,
                explanation=helix_result.explanation,
                entropy=helix_result.entropy,
                needs_validation=helix_result.confidence < 0.5,
                metadata={
                    'double_helix': True,
                    'weight_A': helix_result.weight_A,
                    'weight_B': helix_result.weight_B,
                    'phase': helix_result.phase,
                    'emergence': helix_result.emergence_score,
                    'cycle': helix_result.cycle_number,
                    'ascent': helix_result.ascent_level,
                    'fusion_method': helix_result.fusion_method,  # 🆕 融合方法
                    'complementary_preference': self._extract_system_preference(helix_result.fusion_method)  # 🆕 系统偏好
                }
            )

            # 更新统计
            if helix_result.individual_A:
                self.stats['seed_decisions'] += 1
            if helix_result.individual_B:
                self.stats['fractal_decisions'] += 1

            # 🆕 [P0优化] 尝试缓存并返回
            return self._maybe_cache_and_return(state, result, intent_override="double_helix")

        elif self.decision_mode == 'round_robin':
            # 🔧 轮询模式：强制交替使用A和B
            if self.round_robin_counter % 2 == 0 and self.enable_fractal and self.fractal:
                result = self._decide_fractal(state, context)
                self.stats['fractal_decisions'] += 1
                result.explanation = f"系统B（轮询{self.round_robin_counter}）- {result.explanation}"
                # 🆕 [P0优化] 尝试缓存并返回
                return self._maybe_cache_and_return(state, result, intent_override="round_robin_fractal")
            elif self.seed:
                result = self._decide_seed(state, context)
                self.stats['seed_decisions'] += 1
                result.explanation = f"系统A（轮询{self.round_robin_counter}）- {result.explanation}"
                # 🆕 [P0优化] 尝试缓存并返回
                return self._maybe_cache_and_return(state, result, intent_override="round_robin_seed")
            else:
                # 兜底
                return self._get_fallback_result(state, context)

        elif self.decision_mode == 'adaptive':
            # 🔧 自适应模式：基于置信度选择，但给系统A机会
            result_fractal = None
            if self.enable_fractal and self.fractal:
                result_fractal = self._decide_fractal(state, context)

            result_seed = None
            if self.seed:
                result_seed = self._decide_seed(state, context)

            # 选择置信度更高的
            if result_fractal and result_seed:
                if result_fractal.confidence >= result_seed.confidence:
                    self.stats['fractal_decisions'] += 1
                    result_fractal.explanation = f"系统B（自适应选择）- {result_fractal.explanation}"
                    # 🆕 [P0优化] 尝试缓存并返回
                    return self._maybe_cache_and_return(state, result_fractal, intent_override="adaptive_fractal")
                else:
                    self.stats['seed_decisions'] += 1
                    result_seed.explanation = f"系统A（自适应选择）- {result_seed.explanation}"
                    # 🆕 [P0优化] 尝试缓存并返回
                    return self._maybe_cache_and_return(state, result_seed, intent_override="adaptive_seed")
            elif result_fractal:
                self.stats['fractal_decisions'] += 1
                # 🆕 [P0优化] 尝试缓存并返回
                return self._maybe_cache_and_return(state, result_fractal, intent_override="adaptive_fractal_only")
            elif result_seed:
                self.stats['seed_decisions'] += 1
                # 🆕 [P0优化] 尝试缓存并返回
                return self._maybe_cache_and_return(state, result_seed, intent_override="adaptive_seed_only")
            else:
                return self._get_fallback_result(state, context)

        else:  # confidence_based (传统模式)
            # 基于阈值选择
            if self.enable_fractal and self.fractal and force_path in [None, DecisionPath.FRACTAL]:
                result = self._decide_fractal(state, context)
                if result.confidence >= self.adaptive_threshold:
                    self.stats['fractal_decisions'] += 1
                    # 🆕 [P0优化] 尝试缓存并返回
                    return self._maybe_cache_and_return(state, result, intent_override="confidence_fractal")

            if self.seed and force_path in [None, DecisionPath.SEED]:
                result = self._decide_seed(state, context)
                if result.confidence >= self.adaptive_threshold:
                    self.stats['seed_decisions'] += 1
                    # 🆕 [P0优化] 尝试缓存并返回
                    return self._maybe_cache_and_return(state, result, intent_override="confidence_seed")

            # 兜底
            return self._get_fallback_result(state, context)

    def _decide_by_path(self, path: DecisionPath, state: np.ndarray, context: Dict[str, Any]) -> DecisionResult:
        """按指定路径决策"""
        if path == DecisionPath.FRACTAL and self.enable_fractal and self.fractal:
            result = self._decide_fractal(state, context)
            self.stats['fractal_decisions'] += 1
            # 🆕 [P0优化] 尝试缓存并返回
            return self._maybe_cache_and_return(state, result, intent_override="force_fractal")
        elif path == DecisionPath.SEED and self.seed:
            result = self._decide_seed(state, context)
            self.stats['seed_decisions'] += 1
            # 🆕 [P0优化] 尝试缓存并返回
            return self._maybe_cache_and_return(state, result, intent_override="force_seed")
        elif path == DecisionPath.LLM and self.enable_llm and self.llm_service:
            result = self._decide_llm(state, context)
            self.stats['llm_decisions'] += 1
            # 🆕 [P0优化] 尝试缓存并返回
            return self._maybe_cache_and_return(state, result, intent_override="llm")
        else:
            return self._get_fallback_result(state, context)

    def _get_fallback_result(self, state: np.ndarray, context: Dict[str, Any]) -> DecisionResult:
        """获取兜底结果"""
        # 优先使用系统B
        if self.enable_fractal and self.fractal:
            result = self._decide_fractal(state, context)
            self.stats['fractal_decisions'] += 1
            return result
        # 其次使用系统A
        if self.seed:
            result = self._decide_seed(state, context)
            self.stats['seed_decisions'] += 1
            return result
        # 最后兜底：随机决策
        return DecisionResult(
            action=np.random.randint(0, self.action_dim),
            confidence=0.3,
            path=DecisionPath.FRACTAL,
            response_time_ms=0.1,
            explanation="随机决策（所有系统不可用）",
            needs_validation=True
        )

    def _decide_fractal(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> DecisionResult:
        """系统B决策：分形拓扑智能"""
        start_time = time.time()

        # 转换为Tensor
        state_tensor = torch.from_numpy(state).float().to(self.device)

        # Fractal决策
        with torch.no_grad():
            output, meta = self.fractal.core(state_tensor, return_meta=True)

        response_time = (time.time() - start_time) * 1000

        # 提取决策信息
        confidence = meta.self_awareness.mean().item()
        entropy = meta.entropy.item()

        # 从输出推断动作（简化处理）
        if output.dim() > 1:
            action = torch.argmax(output, dim=-1).item() % self.action_dim
        else:
            action = int(output.item() % self.action_dim)

        return DecisionResult(
            action=int(action),
            confidence=confidence,
            path=DecisionPath.FRACTAL,
            response_time_ms=response_time,
            explanation=f"系统B（分形拓扑）- 置信度{confidence:.4f}",
            entropy=entropy,
            needs_validation=confidence < self.adaptive_threshold,
            metadata={
                'goal_score': meta.goal_score,
                'metaparams': meta.metaparams
            }
        )

    def _decide_seed(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> DecisionResult:
        """系统A决策：TheSeed DQN"""
        start_time = time.time()

        # TheSeed决策
        action = self.seed.act(state)
        value = self.seed.evaluate(state, state, 0.0)
        confidence = min(1.0, max(0.0, value))

        response_time = (time.time() - start_time) * 1000

        return DecisionResult(
            action=int(action),
            confidence=confidence,
            path=DecisionPath.SEED,
            response_time_ms=response_time,
            explanation=f"系统A（TheSeed）- 价值{value:.4f}",
            entropy=0.5,
            needs_validation=confidence < self.adaptive_threshold
        )

    def _decide_llm(
        self,
        state: np.ndarray,
        context: Dict[str, Any]
    ) -> DecisionResult:
        """外部LLM决策"""
        start_time = time.time()

        # 简化的LLM决策（实际应用中需要完整实现）
        # 这里返回基于状态分析的伪决策

        response_time = (time.time() - start_time) * 1000

        # 简化：基于状态的hash来决定动作
        action = int(hash(state.tobytes()) % self.action_dim)

        return DecisionResult(
            action=action,
            confidence=0.7,  # LLM通常有较高置信度
            path=DecisionPath.LLM,
            response_time_ms=response_time,
            explanation=f"外部LLM决策（简化版）",
            entropy=0.3,
            needs_validation=False
        )

    def learn(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray
    ):
        """
        从经验中学习（两路学习）

        1. TheSeed：DQN学习
        2. Fractal：目标修改
        """
        # 1. TheSeed学习
        if self.seed and Experience:
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state
            )
            self.seed.learn(experience)

        # 2. Fractal学习
        if self.enable_fractal and self.fractal:
            try:
                exp_dict = {'state': torch.from_numpy(state).float().to(self.device)}
                self.fractal.learn(exp_dict, reward)
            except Exception as e:
                logger.debug(f"[Hybrid] Fractal学习失败（正常）: {e}")

        # 3. 更新自适应阈值
        self._update_adaptive_threshold(reward)

    def _update_adaptive_threshold(self, reward: float):
        """🆕 [P0优化] 更新自适应置信度阈值（增强版）

        基于历史性能动态调整阈值：
        1. 记录奖励历史
        2. 计算最近20次奖励平均值
        3. 基于平均值动态调整阈值
        """
        # 1. 记录奖励历史
        self.reward_history.append(reward)
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)

        # 2. 基于奖励调整阈值（原有逻辑）
        if reward > 0:
            # 正奖励：降低阈值，更多使用本地决策
            self.adaptive_threshold = max(self.min_threshold, self.adaptive_threshold - 0.001)
        else:
            # 负奖励：提高阈值，更加谨慎
            self.adaptive_threshold = min(self.max_threshold, self.adaptive_threshold + 0.001)

        # 3. 🆕 基于历史性能动态调整（新增逻辑）
        if len(self.reward_history) >= 20:
            # 计算最近20次的平均奖励
            recent_avg = np.mean(self.reward_history[-20:])

            # 基于历史平均值进行二次调整
            if recent_avg > 0.7:
                # 高性能：降低阈值，更激进使用本地决策
                adjustment = -0.005
                self.adaptive_threshold = max(
                    self.min_threshold,
                    self.adaptive_threshold + adjustment
                )
            elif recent_avg < 0.4:
                # 低性能：提高阈值，更谨慎决策
                adjustment = 0.005
                self.adaptive_threshold = min(
                    self.max_threshold,
                    self.adaptive_threshold + adjustment
                )

            logger.debug(
                f"[Hybrid] [动态阈值] "
                f"threshold={self.adaptive_threshold:.4f}, "
                f"recent_avg_reward={recent_avg:.3f}, "
                f"history_size={len(self.reward_history)}"
            )

        logger.debug(f"[Hybrid] 阈值更新: {self.adaptive_threshold:.4f} (reward={reward:.2f})")

    def _extract_system_preference(self, fusion_method: str) -> str:
        """从融合方法中提取系统偏好"""
        if 'complementary_selection_A' in fusion_method:
            return 'A'
        elif 'complementary_selection_B' in fusion_method:
            return 'B'
        elif 'creative_fusion' in fusion_method:
            return 'creative'
        else:
            return 'neutral'

    def _store_to_cache(
        self,
        state: np.ndarray,
        intent: str,
        confidence: float,
        metadata: Dict[str, Any]
    ):
        """🆕 [P0优化] 存储决策结果到缓存"""
        if not self.decision_cache:
            return

        # 只缓存高置信度结果（> 0.7）
        if confidence > 0.7:
            state_embedding = state.flatten() if hasattr(state, 'flatten') else state
            self.decision_cache.put(
                text_embedding=state_embedding,
                intent=intent,
                confidence=confidence,
                metadata=metadata
            )
            logger.debug(
                f"[Hybrid] [缓存存储] "
                f"intent={intent}, "
                f"confidence={confidence:.3f}"
            )

    def _maybe_cache_and_return(
        self,
        state: np.ndarray,
        result: DecisionResult,
        intent_override: Optional[str] = None
    ) -> DecisionResult:
        """🆕 [P0优化] 尝试缓存结果并返回

        Args:
            state: 原始状态
            result: 决策结果
            intent_override: 可选的意图覆盖（用于从explanation提取意图）

        Returns:
            DecisionResult: 原始结果（可能被缓存）
        """
        # 只缓存高置信度结果
        if result.confidence > 0.7 and self.decision_cache:
            # 生成intent字符串
            if intent_override:
                intent = intent_override
            else:
                # 从explanation提取简单的intent标识
                intent = result.explanation.split('-')[0].strip() if '-' in result.explanation else result.explanation[:20]

            # 提取路径标识作为intent的一部分
            path_name = result.path.name if hasattr(result.path, 'name') else str(result.path)

            # 构造完整intent
            full_intent = f"{path_name}:{intent}"

            # 存储到缓存
            self._store_to_cache(
                state=state,
                intent=full_intent,
                confidence=result.confidence,
                metadata={
                    'path': path_name,
                    'explanation': result.explanation,
                    'needs_validation': result.needs_validation
                }
            )

        return result

    def _intent_to_action(self, intent: str, state: np.ndarray) -> int:
        """🆕 [P0优化] 将意图转换为动作

        Args:
            intent: 意图字符串（如 'file_read', 'system_status'）
            state: 当前状态（用于上下文）

        Returns:
            int: 动作ID
        """
        # 简化版映射：基于intent的hash映射到action空间
        # 确保同一个intent总是映射到同一个action
        action = hash(intent) % self.action_dim

        # 可选：基于状态进行微调（增加多样性）
        state_factor = int(np.sum(state[:10]) * 100) % self.action_dim
        action = (action + state_factor) % self.action_dim

        return action

    def get_statistics(self) -> Dict[str, Any]:
        """获取决策统计"""
        stats = self.stats.copy()

        if stats['total_decisions'] > 0:
            stats['fractal_ratio'] = stats['fractal_decisions'] / stats['total_decisions']
            stats['seed_ratio'] = stats['seed_decisions'] / stats['total_decisions']
            stats['llm_ratio'] = stats['llm_decisions'] / stats['total_decisions']
            stats['external_dependency'] = stats.get('llm_ratio', 0.0)

        stats['adaptive_threshold'] = self.adaptive_threshold

        # 🆕 [P0优化] 添加缓存统计
        if self.decision_cache:
            cache_stats = self.decision_cache.get_statistics()
            stats['cache'] = {
                'enabled': True,
                'hit_rate': cache_stats['hit_rate'],
                'hits': cache_stats['hits'],
                'misses': cache_stats['misses'],
                'size': cache_stats['cache_size'],
                'max_size': cache_stats['max_size']
            }
            # 计算本地决策命中率（缓存 + Fractal + Seed）
            total_local = stats.get('cache_decisions', 0) + stats['fractal_decisions'] + stats['seed_decisions']
            stats['local_hit_rate'] = total_local / max(stats['total_decisions'], 1)
        else:
            stats['cache'] = {'enabled': False}

        # 添加双螺旋统计
        if self.helix_engine:
            helix_stats = self.helix_engine.get_statistics()
            stats['double_helix'] = {
                'enabled': True,
                'current_phase': helix_stats['current_phase'],
                'current_weight_A': helix_stats['current_weight_A'],
                'current_weight_B': helix_stats['current_weight_B'],
                'cycle_number': helix_stats['cycle_number'],
                'ascent_level': helix_stats['ascent_level'],
                'avg_emergence': helix_stats['avg_emergence'],
                'cycles_completed': helix_stats['cycles_completed']
            }
        else:
            stats['double_helix'] = {'enabled': False}

        return stats


# 便捷函数
def create_hybrid_decision_engine(
    state_dim: int = 64,
    action_dim: int = 4,
    device: str = 'cpu',
    enable_fractal: bool = True,
    enable_llm: bool = False
) -> HybridDecisionEngine:
    """创建混合决策引擎"""
    return HybridDecisionEngine(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        enable_fractal=enable_fractal,
        enable_llm=enable_llm
    )
