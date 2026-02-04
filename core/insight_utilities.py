#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Insight实用函数库 (Insight Utility Functions)

为系统生成的Insight代码提供缺失的辅助函数实现，
提高代码可执行性和完整性。

创建日期: 2026-01-15
用途: 支持Insight中涉及的记忆重组、因果推理、情感分析等功能
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
import random
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# 记忆相关函数 (Memory-related Functions)
# ============================================================================

def invert_causal_chain(memory: Dict[str, Any]) -> Dict[str, Any]:
    """
    反转记忆的因果链

    Args:
        memory: 记忆字典，应包含'causal_chain'键

    Returns:
        反转因果链后的新记忆
    """
    if 'causal_chain' not in memory:
        logger.warning(f"Memory missing 'causal_chain', returning copy")
        return memory.copy()

    # 反转因果链的顺序
    inverted = memory.copy()
    chain = memory['causal_chain']
    if isinstance(chain, list):
        inverted['causal_chain'] = chain[::-1]
        inverted['inverted'] = True
    else:
        logger.warning(f"Causal chain is not a list: {type(chain)}")

    return inverted


def perturb_attention_weights(memory: Dict[str, Any], scale: float = 0.1) -> Dict[str, Any]:
    """
    扰动记忆的注意力权重

    Args:
        memory: 记忆字典
        scale: 扰动规模

    Returns:
        添加扰动后的新记忆
    """
    perturbed = memory.copy()

    if 'attention_weights' in memory:
        weights = np.array(memory['attention_weights'])
        noise = np.random.normal(0, scale, weights.shape)
        perturbed['attention_weights'] = weights + noise
        perturbed['attention_weights'] = np.clip(
            perturbed['attention_weights'], 0, 1
        ).tolist()
    else:
        # 如果没有注意力权重，添加随机权重
        perturbed['attention_weights'] = np.random.dirichlet(np.ones(5)).tolist()

    perturbed['perturbed'] = True
    return perturbed


def simulate_forward(counterfactual: Dict[str, Any]) -> float:
    """
    前向模拟反事实，返回预测误差

    Args:
        counterfactual: 反事实记忆

    Returns:
        预测误差（0-1之间）
    """
    # 简化版：基于随机性和记忆复杂性计算误差
    error = random.random() * 0.5  # 基础误差

    # 如果有'inverted'标记，增加误差
    if counterfactual.get('inverted', False):
        error += 0.2

    # 如果有'perturbed'标记，增加误差
    if counterfactual.get('perturbed', False):
        error += 0.1

    # 如果有'causal_chain'，根据长度调整
    if 'causal_chain' in counterfactual:
        chain_length = len(counterfactual['causal_chain'])
        error += 0.05 * min(chain_length / 10, 1.0)

    return min(error, 1.0)


def rest_phase_reorganization(
    memory_bank: List[Dict[str, Any]],
    entropy_threshold: float = 0.95
) -> List[Dict[str, Any]]:
    """
    休息阶段记忆重组

    通过高惊讶度记忆的反事实模拟，生成创新种子

    Args:
        memory_bank: 记忆列表
        entropy_threshold: 惊讶度阈值

    Returns:
        重组后的前20%记忆
    """
    if not memory_bank:
        return []

    try:
        # 识别高度不确定（高惊讶）的记忆
        unstable_memories = [
            m for m in memory_bank
            if m.get('surprise', 0) > entropy_threshold
        ]

        if not unstable_memories:
            logger.info(f"No unstable memories found above threshold {entropy_threshold}")
            return []

        # 通过反转因果假设生成反事实变体
        counterfactuals = []
        for m in unstable_memories:
            inverted = invert_causal_chain(m)
            perturbed = perturb_attention_weights(inverted, scale=0.1)
            counterfactuals.append(perturbed)

        # 模拟结果并计算不协调减少潜力
        for cf in counterfactuals:
            prediction_error = simulate_forward(cf)
            cf['motivational_weight'] = -np.log(prediction_error + 1e-8)

        # 将前20%重新整合到工作记忆中
        sorted_seeds = sorted(
            counterfactuals,
            key=lambda x: x.get('motivational_weight', 0),
            reverse=True
        )

        top_n = max(1, int(0.2 * len(sorted_seeds)))
        return sorted_seeds[:top_n]

    except Exception as e:
        logger.error(f"Rest phase reorganization failed: {e}")
        return []


# ============================================================================
# 噪声相关函数 (Noise-related Functions)
# ============================================================================

def noise_guided_rest(
    state: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    噪声引导休息

    在高熵休息期间模拟生成性孵化

    Args:
        state: 状态张量
        temperature: 温度参数

    Returns:
        新状态张量
    """
    noise = torch.randn_like(state) * temperature
    perturbed = state + noise

    # 应用软重置：保留核心轨迹，放大新颖性
    similarity = torch.cosine_similarity(
        state.flatten() if state.dim() > 1 else state,
        perturbed.flatten() if perturbed.dim() > 1 else perturbed,
        dim=0
    )

    mask_value = 1.0 if similarity < 0.8 else 0.0
    mask = torch.tensor(mask_value)

    if state.dim() > 1:
        mask = mask.unsqueeze(-1)

    new_state = state * (1 - mask) + perturbed * mask
    return torch.nn.functional.normalize(new_state.flatten() if new_state.dim() > 1 else new_state, dim=0)


# ============================================================================
# 语义相关函数 (Semantic-related Functions)
# ============================================================================

def semantic_perturb(
    problem_domain: str,
    known_concepts: List[str] = None
) -> str:
    """
    语义扰动

    应用最小概念破坏来打破固着

    Args:
        problem_domain: 问题领域
        known_concepts: 已知概念列表

    Returns:
        扰动提示字符串
    """
    # 无关领域用于强制联想
    disruptors = {
        'quantum physics': ['superposition', 'entanglement', 'tunneling'],
        'mycology': ['hyphae', 'spore dispersal', 'symbiosis'],
        'typography': ['kerning', 'baseline', 'x-height'],
        'choreography': ['proximity', 'rhythm', 'weight sharing'],
        'architecture': ['cantilever', 'load-bearing', 'tensile'],
        'music': ['counterpoint', 'crescendo', 'syncopation']
    }

    domain, features = random.choice(list(disruptors.items()))
    while domain == problem_domain:
        domain, features = random.choice(list(disruptors.items()))

    trigger = random.choice(features)
    return f"Perturb {problem_domain} with '{trigger}' from {domain}: How might they relate?"


def analyze_tone(text: str) -> Dict[str, float]:
    """
    分析文本情感效价

    Args:
        text: 输入文本

    Returns:
        包含'valence'键的字典（-1到+1）
    """
    # 简化版：基于字符哈希的伪情感分析
    hash_val = sum(ord(c) for c in text) % 100 / 50 - 1
    valence = np.tanh(hash_val * 0.5)

    return {'valence': float(valence)}


def semantic_diode(
    input_stream: List[str],
    threshold: float = 0.75,
    hysteresis_window: int = 3
) -> List[str]:
    """
    语义二极管

    通过情感轨迹而非内容过滤认知流

    Args:
        input_stream: 输入文本流
        threshold: 情感变化阈值
        hysteresis_window: 迟滞窗口大小

    Returns:
        过滤后的输出流
    """
    if not input_stream:
        return []

    valence_scores = [analyze_tone(text)['valence'] for text in input_stream]
    gradient = np.diff(valence_scores)

    # 检测显著转变
    significant_shifts = np.abs(gradient) > threshold

    # 应用迟滞
    diode_open = np.convolve(
        significant_shifts.astype(int),
        np.ones(hysteresis_window),
        mode='same'
    ) >= hysteresis_window

    # 输出符合条件的输入
    output = []
    for i, open_state in enumerate(diode_open):
        if i + 1 < len(input_stream) and open_state:
            output.append(input_stream[i + 1])

    return output


# ============================================================================
# 拓扑缺陷相关函数 (Topological Defect Functions)
# ============================================================================

def detect_topological_defect(z: torch.Tensor) -> int:
    """
    检测拓扑缺陷

    Args:
        z: 复数值激活张量

    Returns:
        缺陷数量
    """
    if not z.is_complex():
        # 如果不是复数，添加虚部
        z = z + 1j * torch.zeros_like(z)

    phase = torch.angle(z)
    curl = torch.gradient(phase)[0] if phase.dim() > 0 else torch.tensor(0.0)
    return int((curl > torch.pi).sum().item())


class CurlLayer(nn.Module):
    """旋度层：引入非保守场的旋转分量"""

    def __init__(self, size: int):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(size, size, dtype=torch.complex64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用变换
        W = self.weights + 1j * torch.eye(self.weights.size(0))
        output = torch.matmul(x, W.real) + 1j * torch.matmul(x, W.imag)
        return output


# ============================================================================
# 分形脉冲相关函数 (Fractal Pulse Functions)
# ============================================================================

def fractal_idle_pulse(
    duration: float,
    base_freq: float = 0.1,
    depth: int = 3,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    分形空闲脉冲

    为高熵休息状态生成多尺度扰动信号

    Args:
        duration: 持续时间
        base_freq: 基础频率
        depth: 分形深度
        seed: 随机种子

    Returns:
        (时间数组, 信号数组)
    """
    if seed:
        np.random.seed(seed)

    t = np.linspace(0, duration, int(duration * 100))
    signal = np.zeros_like(t)

    for level in range(1, depth + 1):
        n_pulses = 2 ** level
        pulse_times = np.logspace(
            np.log10(0.1),
            np.log10(max(duration - 0.1, 0.2)),
            n_pulses
        )

        for pt in pulse_times:
            gaussian = np.exp(-((t - pt) * 10 / level) ** 2)
            signal += (1 / level) * gaussian * np.random.randn()

    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    return t, signal


# ============================================================================
# 逆向溯因相关函数 (Reverse Abduction Functions)
# ============================================================================

def kl_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    计算KL散度

    Args:
        p: 分布p
        q: 分布q

    Returns:
        KL散度值
    """
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    return torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8), dim=-1).mean()


def reverse_abduction_step(
    model: nn.Module,
    context: torch.Tensor,
    noise_scale: float = 1.2
) -> Tuple[torch.Tensor, float]:
    """
    逆向溯因步骤

    通过制造内部冲突加速演化

    Args:
        model: 神经网络模型
        context: 上下文张量
        noise_scale: 噪声缩放

    Returns:
        (anti_context, dissonance) 元组
    """
    # 1. 获取正常预测
    with torch.no_grad():
        pred = model(context)
        # 假设模型有generate_rationale方法
        if hasattr(model, 'generate_rationale'):
            pred_rationale = model.generate_rationale(context)
        else:
            pred_rationale = pred

    # 2. 计算梯度
    context_grad = context.clone().detach().requires_grad_(True)
    pred_var = model(context_grad).var()
    pred_var.backward(retain_graph=True)

    # 3. 生成反预测
    noise_direction = torch.sign(context_grad.grad)
    anti_context = context + noise_scale * noise_direction

    # 4. 强制解释
    with torch.no_grad():
        if hasattr(model, 'generate_rationale'):
            explanation = model.generate_rationale(anti_context)
        else:
            explanation = model(anti_context)

    # 5. 计算失调
    dissonance = kl_div(pred_rationale, explanation)

    return anti_context, dissonance.item()


# ============================================================================
# 对抗性直觉相关函数 (Adversarial Intuition Functions)
# ============================================================================

def inject_adversarial_intuition(
    model: nn.Module,
    alpha: float = 0.03,
    backup: bool = True
) -> Dict[str, Any]:
    """
    注入对抗性直觉

    Args:
        model: PyTorch模型
        alpha: 扰动强度
        backup: 是否备份

    Returns:
        注入统计信息
    """
    if backup:
        original_state = {
            name: param.clone()
            for name, param in model.named_parameters()
        }

    stats = {
        'perturbed_params': 0,
        'total_noise': 0.0,
        'backup_created': backup
    }

    try:
        for name, param in model.named_parameters():
            if 'prediction_head' in name and param.requires_grad:
                # 生成悖论噪声
                paradox_noise = alpha * torch.tanh(
                    torch.randn_like(param) * 0.1 +
                    (1 - torch.abs(torch.cos(param)))
                )

                # 限制扰动范围
                paradox_noise = torch.clamp(paradox_noise, -0.1, 0.1)

                param.add_(paradox_noise)
                stats['perturbed_params'] += 1
                stats['total_noise'] += paradox_noise.abs().sum().item()

        # 验证模型未崩溃
        if hasattr(model, 'input_shape'):
            test_input = torch.randn(1, *model.input_shape)
        else:
            test_input = torch.randn(1, 64)

        test_output = model(test_input)
        if torch.isnan(test_output).any() or torch.isinf(test_output).any():
            raise ValueError("Model output contains NaN or Inf after perturbation")

        return stats

    except Exception as e:
        logger.error(f"Adversarial injection failed: {e}")
        if backup and 'original_state' in locals():
            logger.info("Restoring original weights...")
            for name, param in model.named_parameters():
                if name in original_state:
                    param.data.copy_(original_state[name])
        raise


# ============================================================================
# 潜在重组相关函数 (Latent Recombination Functions)
# ============================================================================

def latent_recombination(
    memories: List[np.ndarray],
    noise_scale: float = 0.93
) -> np.ndarray:
    """
    使用受控随机共振重组记忆痕迹

    Args:
        memories: 记忆向量列表
        noise_scale: 噪声规模

    Returns:
        重组后的候选向量（前5个最新颖的）
    """
    if not memories:
        return np.array([])

    # 归一化记忆向量
    M = np.array(memories)
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)

    # 生成结构化噪声
    noise = np.random.normal(0, noise_scale, M_norm.shape)
    noise = 0.5 * noise + 0.5 * np.roll(noise, shift=1, axis=0)

    # 在正交子空间中应用噪声
    perturbed = M_norm + noise - np.outer(
        np.sum(noise @ M_norm.T, axis=1),
        np.mean(M_norm, axis=0)
    )

    # 归一化
    candidates = perturbed / (
        np.linalg.norm(perturbed, axis=1, keepdims=True) + 1e-8
    )

    # 返回前5个最新颖的
    novelty_score = np.min(
        np.abs(candidates @ M_norm.T - 1e-6), axis=1
    )
    top_indices = np.argsort(-novelty_score)[:5]
    return candidates[top_indices]


# ============================================================================
# 测试函数
# ============================================================================

def test_insight_utilities():
    """测试所有实用函数"""
    logger.info("Testing Insight Utilities...")

    # 测试记忆函数
    memory = {
        'causal_chain': ['A', 'B', 'C'],
        'surprise': 0.96,
        'attention_weights': [0.3, 0.3, 0.2, 0.1, 0.1]
    }

    inverted = invert_causal_chain(memory)
    assert inverted['causal_chain'] == ['C', 'B', 'A']
    assert inverted['inverted'] == True

    perturbed = perturb_attention_weights(memory)
    assert 'perturbed' in perturbed

    error = simulate_forward(perturbed)
    assert 0 <= error <= 1

    # 测试记忆重组
    memory_bank = [
        {'surprise': 0.96, 'data': 'test1'},
        {'surprise': 0.94, 'data': 'test2'},
        {'surprise': 0.92, 'data': 'test3'}
    ]

    reorganized = rest_phase_reorganization(memory_bank)
    assert len(reorganized) > 0

    # 测试噪声引导
    state = torch.randn(64)
    new_state = noise_guided_rest(state)
    assert new_state.shape == state.shape

    # 测试语义扰动
    result = semantic_perturb('computing')
    assert 'Perturb' in result
    assert 'computing' in result

    # 测试情感分析
    tone = analyze_tone('test text')
    assert 'valence' in tone
    assert -1 <= tone['valence'] <= 1

    # 测试语义二极管
    stream = ['hello', 'world', 'test', 'stream']
    filtered = semantic_diode(stream)
    assert isinstance(filtered, list)

    # 测试拓扑缺陷检测
    z = torch.randn(10, dtype=torch.complex64)
    defects = detect_topological_defect(z)
    assert isinstance(defects, int)

    # 测试分形脉冲
    t, signal = fractal_idle_pulse(1.0, seed=42)
    assert len(t) == len(signal)

    logger.info("✅ All tests passed!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_insight_utilities()
