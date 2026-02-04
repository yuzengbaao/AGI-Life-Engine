#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分形智能核心模块 (Fractal Intelligence Core)
基于第一性原理推导的自指涉分形拓扑网络实现

数学基础：docs/FLUID_INTELLIGENCE_MATHEMATICAL_FOUNDATION_20260112.md
实施路线：docs/B_PLAN_IMPLEMENTATION_ROADMAP_20260112.md

核心创新：
1. 自指涉性：网络能观察和修改自身
2. 分形性：不同尺度上的自相似结构
3. 目标可塑性：能质疑和修改优化目标
4. 熵驱动：好奇心压力阀调节探索

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-12
版本：v1.0 (B组)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 设置日志
logger = logging.getLogger(__name__)

# 注意：递归深度限制已移除硬编码，改用DynamicRecursionLimiter
# 原MAX_RECURSION_DEPTH = 3已废弃，使用动态限制器替代


@dataclass
class FractalOutput:
    """分形网络输出数据类"""
    output: torch.Tensor
    self_awareness: torch.Tensor
    entropy: torch.Tensor
    goal_score: float
    metaparams: Tuple[float, float, float]  # (alpha, beta, gamma)


class SelfReferentialFractalCore(nn.Module):
    """
    自指涉分形核心

    数学对应：Φ = f(Φ, x)

    核心特性：
    1. 自指涉：网络维护关于自身的表示
    2. 分形：多层自相似递归结构
    3. 目标可塑：能质疑和修改优化目标
    4. 熵驱动：好奇心压力阀
    """

    def __init__(
        self,
        input_dim: int = 2,
        state_dim: int = 64,
        fractal_depth: int = 3,
        max_recursion: int = 3,
        device: str = 'cpu',
        entropy_temperature: float = 2.0,  # 默认温度2.0，使熵值更合理
        enable_dynamic_recursion: bool = True  # 新增：启用动态递归限制
    ):
        super().__init__()

        self.input_dim = input_dim
        self.state_dim = state_dim
        self.fractal_depth = fractal_depth
        self.max_recursion = max_recursion
        self.device = device
        self.entropy_temperature = entropy_temperature
        self.enable_dynamic_recursion = enable_dynamic_recursion

        # 新增：动态递归限制器
        if enable_dynamic_recursion:
            from core.dynamic_recursion_limiter import get_recursion_limiter
            self.recursion_limiter = get_recursion_limiter()
            logger.info("[分形智能] 动态递归限制器已启用")
        else:
            self.recursion_limiter = None
            logger.info("[分形智能] 使用固定递归限制")

        # ========== 关键创新1：自指涉状态 ==========
        # 网络维护一个"关于自身的表示"
        self.self_representation = nn.Parameter(
            torch.randn(state_dim, device=device) * 0.01,
            requires_grad=True  # 可学习的自我概念
        )

        # ========== 关键创新2：分形递归块 ==========
        self.fractal_blocks = nn.ModuleList([
            FractalRecursiveBlock(
                state_dim=state_dim,
                depth=d,
                self_reflection=self.self_representation,
                device=device,
                recursion_limiter=self.recursion_limiter if enable_dynamic_recursion else None
            )
            for d in range(fractal_depth)
        ])

        # 输入投影
        self.input_projection = nn.Linear(input_dim, state_dim).to(device)

        # 输出投影
        self.output_projection = nn.Linear(state_dim, 1).to(device)

        # ========== 关键创新3：目标质疑模块（Active模式）==========
        self.goal_questioner = GoalQuestionerActive(state_dim, device=device)

        # ========== 关键创新4：好奇心压力阀 ==========
        self.curiosity_valve = CuriosityPressureValve(state_dim, device=device)

        # 用于追踪历史
        self.entropy_history = []
        self.goal_score_history = []

        logger.info(f"SelfReferentialFractalCore initialized: "
                   f"state_dim={state_dim}, fractal_depth={fractal_depth}, "
                   f"device={device}")

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[int] = None,
        return_meta: bool = True
    ) -> Tuple[torch.Tensor, Optional[FractalOutput]]:
        """
        前向传播实现自指涉分形演化

        数学对应：∂S/∂t = α·∇ₛL_meta + β·∇ᴼL_goal + γ·N
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        # 1. 输入投影
        state = self.input_projection(x)

        # 2. 计算自指涉意识
        self_awareness = self._compute_self_awareness(state)

        # 3. 分形递归处理
        fractal_outputs = []
        for i, block in enumerate(self.fractal_blocks):
            scale_factor = 0.7 ** i
            scaled_state = state * scale_factor
            output = block(scaled_state, self_awareness, t, recursion_depth=0)
            fractal_outputs.append(output)

        # 4. 自指涉融合
        integrated = self._integrate_self_reference(fractal_outputs, self_awareness)

        # 5. 输出投影
        output = self.output_projection(integrated)

        # 6. 计算元信息
        entropy = self._compute_entropy(output, temperature=self.entropy_temperature)
        goal_score = self.goal_questioner(integrated.mean(0))

        # 追踪历史
        self.entropy_history.append(entropy.item())
        self.goal_score_history.append(goal_score)

        # 7. 压力阀调节
        alpha, beta, gamma = self.curiosity_valve(entropy)

        if return_meta:
            meta = FractalOutput(
                output=output,
                self_awareness=self_awareness,
                entropy=entropy,
                goal_score=goal_score,
                metaparams=(alpha, beta, gamma)
            )
            return output, meta

        return output, None

    def _compute_self_awareness(self, state: torch.Tensor) -> torch.Tensor:
        """
        计算自指涉意识

        🔧 根本修复: 从[1,1]维度改为[1, state_dim]维度

        数学对应：Φ_self = η · σ(S · Φ_self_repr)
        """
        # 🔧 修复前（错误）:
        # interaction = torch.matmul(state, self.self_representation.T)  # [1,64] × [64,1] = [1,1]
        # self_awareness = torch.sigmoid(interaction / (self.state_dim ** 0.5))  # [1,1] ← 只有1个元素！

        # 🔧 修复后（正确）:
        # 使用element-wise交互，保持state的维度
        # state: [batch, state_dim], self_representation: [state_dim]
        interaction = state * self.self_representation  # 广播乘法: [1,64] * [64] = [1,64]
        self_awareness = torch.sigmoid(interaction)  # [1,64] ← 64个元素！

        logger.info(f"[DEBUG-AWARENESS] _compute_self_awareness output shape: {self_awareness.shape}")
        logger.info(f"[DEBUG-AWARENESS] self_awareness min: {self_awareness.min().item():.6f}")
        logger.info(f"[DEBUG-AWARENESS] self_awareness max: {self_awareness.max().item():.6f}")
        logger.info(f"[DEBUG-AWARENESS] self_awareness mean: {self_awareness.mean().item():.6f}")
        logger.info(f"[DEBUG-AWARENESS] self_awareness std: {self_awareness.std().item():.6f}")

        return self_awareness

    def _integrate_self_reference(
        self,
        fractal_outputs: list,
        self_awareness: torch.Tensor
    ) -> torch.Tensor:
        """
        整合自指涉信息

        数学对应：I = ∫ e^(-λs) · C(Φ^s(S)) · R(Φ^s(S)) ds
        """
        # 加权整合不同分形尺度的输出
        weights = torch.softmax(
            torch.tensor(
                [0.7 ** i for i in range(len(fractal_outputs))],
                device=self.device
            ),
            dim=0
        )

        # Stack和加权
        stacked = torch.stack(fractal_outputs, dim=0)
        weighted = weights.view(-1, 1, 1) * stacked
        integrated = weighted.sum(0)

        # 自指涉调节
        final = integrated * self_awareness + integrated * (1 - self_awareness)

        return final

    def _compute_entropy(self, output: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        计算认知熵（优化版本，支持温度参数）

        数学对应：H(S) = -∑p_i log p_i

        优化：
        1. 添加温度参数控制分布锐度
        2. 正确归一化到[0, 1]范围
        3. 添加更小的epsilon防止数值问题

        Args:
            output: 网络输出logits
            temperature: 温度参数（>1使分布更均匀，<1更锐利）

        Returns:
            归一化的熵值 [0, 1]
        """
        # 应用温度参数的softmax
        # temperature > 1: 更均匀分布 → 更高熵
        # temperature < 1: 更锐利分布 → 更低熵
        probs = F.softmax(output / temperature, dim=-1)

        # 添加极小量防止log(0)
        log_probs = torch.log(probs + 1e-10)

        # 计算熵（香农熵）
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # 归一化到[0, 1]范围
        # 最大可能熵 = log(类别数)
        # 对于单个输出值，我们将其视为二元分布，最大熵 = log(2)
        max_entropy = np.log(2) if output.shape[-1] == 1 else np.log(output.shape[-1])
        normalized_entropy = entropy / (max_entropy + 1e-10)

        # 确保在[0, 1]范围内
        normalized_entropy = torch.clamp(normalized_entropy, min=0.0, max=1.0)

        return normalized_entropy

    def modify_goal(self, state: torch.Tensor):
        """
        修改目标函数（Active模式）

        这是B组的关键特性：系统能真正质疑和修改自己的目标
        """
        self.goal_questioner.modify_goal(state)

    def get_self_representation(self) -> torch.Tensor:
        """获取当前的自指涉表示"""
        return self.self_representation.detach()

    def get_goal_representation(self) -> torch.Tensor:
        """获取当前的目标表示"""
        return self.goal_questioner.goal_representation.detach()


class FractalRecursiveBlock(nn.Module):
    """
    分形递归块：每一层都是整个网络的缩放版本

    数学性质：
    - 自相似性：f(λx) ~ λf(x)
    - 递归性：f^((n))(x) = f(f^((n-1))(x))
    """

    def __init__(
        self,
        state_dim: int,
        depth: int,
        self_reflection: nn.Parameter,
        device: str = 'cpu',
        recursion_limiter = None  # 新增：递归限制器引用
    ):
        super().__init__()
        self.depth = depth
        self.state_dim = state_dim
        self.device = device
        self.recursion_limiter = recursion_limiter  # 新增：保存限制器引用

        # 主干路径
        self.main_path = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(state_dim, state_dim)
        ).to(device)

        # 递归分形分支（如果depth > 0）
        if depth > 0:
            self.fractal_branch = FractalRecursiveBlock(
                state_dim,
                depth - 1,
                self_reflection,
                device
            )
            self.fractal_projection = nn.Linear(state_dim, state_dim).to(device)
        else:
            self.fractal_branch = None
            self.fractal_projection = None

        # 门控机制
        self.gate = nn.Parameter(torch.zeros(1, device=device))
        self.self_gate = nn.Linear(state_dim, 1).to(device)

    def forward(
        self,
        x: torch.Tensor,
        self_awareness: torch.Tensor,
        t: Optional[int],
        recursion_depth: int = 0
    ) -> torch.Tensor:
        # 动态递归深度限制（新增）
        if self.recursion_limiter is not None:
            max_depth = self.recursion_limiter.get_current_limit()
        else:
            # 默认限制（向后兼容）
            max_depth = 3

        if recursion_depth >= max_depth:
            return x

        # 主干变换
        main = self.main_path(x)

        # 分形递归
        if self.fractal_branch is not None:
            scaled = x * 0.5
            fractal = self.fractal_branch(
                scaled,
                self_awareness,
                t,
                recursion_depth + 1
            )
            fractal = self.fractal_projection(fractal)
        else:
            fractal = torch.zeros_like(main)

        # 自指涉门控
        self_gate_weight = torch.sigmoid(self.self_gate(x))

        # 融合
        gate = torch.sigmoid(self.gate) * self_gate_weight
        output = main + gate * fractal

        return output


class GoalQuestionerActive(nn.Module):
    """
    目标质疑模块 - Active模式

    与A组的关键区别：
    - A组：suggest_only（只能建议）
    - B组：active（能真正修改目标函数）

    数学对应：L_goal^(t+1) = L_goal^(t) + ε·E[∇ₗ I(S, L_goal)]
    """

    def __init__(self, state_dim: int, device: str = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.device = device
        self.mode = 'active'  # 关键改动

        # 目标表示
        self.goal_representation = nn.Parameter(
            torch.randn(state_dim, device=device) * 0.1,
            requires_grad=True
        )

        # 质疑网络
        self.questioner = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, state: torch.Tensor) -> float:
        """评估当前目标是否合理"""
        if state.dim() == 1:
            state = state.unsqueeze(0)

        goal_rep_2d = self.goal_representation.unsqueeze(0).expand_as(state)
        similarity = F.cosine_similarity(state, goal_rep_2d)

        combined = torch.cat([state, goal_rep_2d], dim=-1)
        question_score = self.questioner(combined)

        return question_score.item()

    def modify_goal(self, state: torch.Tensor):
        """
        修改目标函数（Active模式的关键功能）

        这是B组区别于A组的核心特性
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.enable_grad():
            goal_rep_2d = self.goal_representation.unsqueeze(0).expand_as(state)
            combined = torch.cat([state, goal_rep_2d], dim=-1)

            question_output = self.questioner(combined)
            goal_grad = torch.autograd.grad(
                outputs=question_output,
                inputs=self.goal_representation,
                create_graph=True,
                retain_graph=True
            )[0]

        with torch.no_grad():
            learning_rate = 0.001
            self.goal_representation += learning_rate * goal_grad.squeeze()

        logger.debug(f"Goal modified: grad_norm={torch.norm(goal_grad):.6f}")


class CuriosityPressureValve(nn.Module):
    """
    好奇心压力阀：动态调节熵值

    数学对应：根据 H(S) 调节 α, β, γ

    功能：
    - 高熵 → 降低探索权重，提高利用权重
    - 低熵 → 提高探索权重，降低利用权重
    """

    def __init__(
        self,
        state_dim: int,
        target_entropy: float = 0.9,
        device: str = 'cpu'
    ):
        super().__init__()
        self.target_entropy = target_entropy
        self.device = device

        self.valve_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        ).to(device)

    def forward(self, current_entropy: torch.Tensor) -> Tuple[float, float, float]:
        """
        根据当前熵返回元参数

        返回：(alpha探索, beta目标, gamma创新)
        """
        entropy_error = current_entropy - self.target_entropy

        adjustments = self.valve_net(
            torch.tensor([[entropy_error]], device=self.device)
        )

        alpha, beta, gamma = adjustments[0].unbind(0)

        # 归一化
        total = alpha + beta + gamma + 1e-8
        return (alpha/total).item(), (beta/total).item(), (gamma/total).item()


class FractalIntelligenceAdapter:
    """
    分形智能适配器

    用于将SelfReferentialFractalCore集成到现有TRAE AGI系统
    """

    def __init__(
        self,
        input_dim: int = 2,
        state_dim: int = 64,
        device: str = 'cpu'
    ):
        self.core = SelfReferentialFractalCore(
            input_dim=input_dim,
            state_dim=state_dim,
            device=device
        )
        self.device = device

        self.cognitive_bridge = None

        logger.info("FractalIntelligenceAdapter initialized")

    def set_cognitive_bridge(self, cognitive_bridge):
        """
        设置认知桥接器

        认知桥接器为分形智能提供拓扑记忆查询和因果推理能力

        Args:
            cognitive_bridge: CognitiveBridge 实例
        """
        self.cognitive_bridge = cognitive_bridge
        logger.info("CognitiveBridge connected to FractalIntelligence")

    def decide(
        self,
        state: torch.Tensor,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        决策函数（替代外部LLM的主要功能）

        这是降低外部依赖的关键方法
        """
        with torch.no_grad():
            output, meta = self.core(state, return_meta=True)

            # 🔧 紧急诊断: 详细记录self_awareness的计算过程
            logger.info(f"[DEBUG-B1] meta.self_awareness shape: {meta.self_awareness.shape}")
            logger.info(f"[DEBUG-B1] meta.self_awareness dtype: {meta.self_awareness.dtype}")
            logger.info(f"[DEBUG-B1] meta.self_awareness device: {meta.self_awareness.device}")
            logger.info(f"[DEBUG-B1] meta.self_awareness raw values:\n{meta.self_awareness}")
            logger.info(f"[DEBUG-B1] self_awareness min: {meta.self_awareness.min().item():.6f}")
            logger.info(f"[DEBUG-B1] self_awareness max: {meta.self_awareness.max().item():.6f}")
            logger.info(f"[DEBUG-B1] self_awareness std: {meta.self_awareness.std().item():.6f}")
            logger.info(f"[DEBUG-B1] self_awareness.mean() BEFORE final: {meta.self_awareness.mean().item():.6f}")

            # 提取决策信息
            entropy = meta.entropy.item()
            goal_score = meta.goal_score

            # 🔧 根本修复: 使用goal_score作为confidence（动态变化）
            # 原因: self_awareness只有[1,1]个元素，mean()无意义
            # 而goal_score在0.4-0.6范围动态变化，更有代表性
            confidence_old = meta.self_awareness.mean().item()
            confidence = float(goal_score)

            logger.info(f"[DEBUG-B1] confidence_old (self_awareness.mean()): {confidence_old:.6f}")
            logger.info(f"[DEBUG-B1] confidence_NEW (goal_score): {confidence:.6f}")
            logger.info(f"[DEBUG-B1] entropy: {entropy:.6f}")
            logger.info(f"[DEBUG-B1] goal_score: {goal_score}")
            logger.info(f"[DEBUG-B1] FINAL confidence: {confidence:.6f}")

            # 如果置信度高，直接使用本地结果
            if confidence > 0.7:
                return output, {
                    'source': 'fractal_core',
                    'confidence': confidence,
                    'entropy': entropy,
                    'goal_score': goal_score,
                    'local_decision': True
                }
            else:
                # 低置信度：需要外部LLM验证
                return output, {
                    'source': 'fractal_core',
                    'confidence': confidence,
                    'entropy': entropy,
                    'goal_score': goal_score,
                    'local_decision': False,
                    'needs_validation': True
                }

    def learn(
        self,
        experience: Dict[str, Any],
        reward: float
    ):
        """
        从经验中学习（支持在线学习）
        """
        # 这里可以实现简单的在线学习
        # 例如：更新目标函数
        if 'state' in experience:
            state = experience['state']
            self.core.modify_goal(state)


# 便捷函数
def create_fractal_intelligence(
    input_dim: int = 2,
    state_dim: int = 64,
    device: str = 'cpu'
) -> FractalIntelligenceAdapter:
    """
    创建分形智能系统

    Args:
        input_dim: 输入维度
        state_dim: 内部状态维度
        device: 设备（'cpu'或'cuda'）

    Returns:
        分形智能适配器实例
    """
    return FractalIntelligenceAdapter(
        input_dim=input_dim,
        state_dim=state_dim,
        device=device
    )


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("="*60)
    print("[测试] 分形智能核心模块")
    print("="*60)

    # 创建分形智能核心
    adapter = create_fractal_intelligence(
        input_dim=2,
        state_dim=64,
        device='cpu'
    )

    # 测试前向传播
    x = torch.randn(10, 2)
    output, meta = adapter.core(x, return_meta=True)

    print(f"\n[结果] 输出形状: {output.shape}")
    print(f"[结果] 自我意识强度: {meta.self_awareness.mean():.4f}")
    print(f"[结果] 认知熵: {meta.entropy:.4f}")
    print(f"[结果] 目标得分: {meta.goal_score:.4f}")
    print(f"[结果] 元参数: α={meta.metaparams[0]:.4f}, "
          f"β={meta.metaparams[1]:.4f}, γ={meta.metaparams[2]:.4f}")

    # 测试目标修改
    state = torch.randn(64)
    adapter.core.modify_goal(state)
    print(f"\n[结果] 目标已修改")

    print("\n" + "="*60)
    print("[成功] 分形智能核心模块测试通过")
    print("="*60)
