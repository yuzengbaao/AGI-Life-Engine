#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
辩论式共识引擎 (Dialogue Consensus Engine)
从"融合"进化到"对话"

核心思想：
1. 系统A和B不是独立决策，而是互相质疑、辩论
2. 通过多轮对话，达成共识
3. 辩论过程本身就是思考过程
4. 最终共识可能超越任何单系统的初始判断

作者：Claude Code (Sonnet 4.5)
创建日期：2026-01-13
版本：v1.0
"""

import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class DialogueStage(Enum):
    """对话阶段"""
    PROPOSAL = "proposal"        # 提案阶段
    CHALLENGE = "challenge"      # 质疑阶段
    DEFENSE = "defense"          # 辩护阶段
    NEGOTIATION = "negotiation"  # 协商阶段
    CONSENSUS = "consensus"      # 共识阶段


@dataclass
class DialogueMessage:
    """对话消息"""
    speaker: str  # 'A' or 'B'
    stage: DialogueStage
    action: int
    confidence: float
    argument: str
    timestamp: float


@dataclass
class ConsensusResult:
    """共识结果"""
    action: int
    confidence: float
    emergence: float
    dialogue_length: int
    stages_completed: List[DialogueStage]
    initial_agreement: float
    final_agreement: float
    consensus_quality: float
    breakdown: Dict[str, Any]


class DialogueEngine:
    """
    辩论式共识引擎

    核心流程：
    1. 提案阶段：系统A和B各自提出方案
    2. 质疑阶段：相互质疑对方的方案
    3. 辩护阶段：各自为自己的方案辩护
    4. 协商阶段：寻找共同点
    5. 共识阶段：达成最终共识
    """

    def __init__(
        self,
        max_rounds: int = 3,
        agreement_threshold: float = 0.8,
        enable_multimodal: bool = True
    ):
        self.max_rounds = max_rounds
        self.agreement_threshold = agreement_threshold
        self.enable_multimodal = enable_multimodal

        # 对话历史
        self.dialogue_history: List[DialogueMessage] = []

        # 统计
        self.stats = {
            'total_dialogues': 0,
            'avg_dialogue_length': 0,
            'consensus_rate': 0,
            'avg_emergence': 0,
            'stage_distribution': {}
        }

        logger.info(f"[辩论引擎] 初始化完成")
        logger.info(f"[辩论引擎] 最大轮数={max_rounds}, 共识阈值={agreement_threshold}")

    def engage_dialogue(
        self,
        result_A: Dict[str, Any],
        result_B: Dict[str, Any],
        state: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """
        进行辩论式对话

        Args:
            result_A: 系统A的决策结果
            result_B: 系统B的决策结果
            state: 当前环境状态
            context: 额外上下文

        Returns:
            共识结果
        """

        start_time = time.time()
        context = context or {}

        # 初始提案
        proposal_A = self._create_proposal(result_A, 'A')
        proposal_B = self._create_proposal(result_B, 'B')

        # 记录提案
        self.dialogue_history.append(proposal_A)
        self.dialogue_history.append(proposal_B)

        # 计算初始一致度
        initial_agreement = self._calculate_agreement(proposal_A, proposal_B)

        # 如果初始一致度高，直接达成共识
        if initial_agreement > self.agreement_threshold:
            logger.debug(f"[辩论引擎] 初始一致度高({initial_agreement:.2f})，直接共识")
            return self._form_immediate_consensus(
                proposal_A, proposal_B,
                initial_agreement,
                start_time
            )

        # 开始辩论
        stages_completed = []
        current_A = proposal_A
        current_B = proposal_B

        for round_num in range(self.max_rounds):
            # 阶段1：质疑
            challenge_A = self._create_challenge(current_A, current_B, 'A')
            challenge_B = self._create_challenge(current_B, current_A, 'B')
            self.dialogue_history.extend([challenge_A, challenge_B])
            stages_completed.append(DialogueStage.CHALLENGE)

            # 阶段2：辩护
            defense_A = self._create_defense(current_A, challenge_B, 'A')
            defense_B = self._create_defense(current_B, challenge_A, 'B')
            self.dialogue_history.extend([defense_A, defense_B])
            stages_completed.append(DialogueStage.DEFENSE)

            # 阶段3：协商
            negotiation = self._negotiate(
                defense_A, defense_B,
                initial_agreement
            )
            self.dialogue_history.append(negotiation)
            stages_completed.append(DialogueStage.NEGOTIATION)

            # 检查是否可以达成共识
            agreement_after_negotiation = self._calculate_agreement(
                defense_A, defense_B
            )

            if agreement_after_negotiation > self.agreement_threshold:
                break

        # 阶段4：达成共识
        consensus = self._reach_consensus(
            proposal_A, proposal_B,
            self.dialogue_history,
            initial_agreement
        )
        stages_completed.append(DialogueStage.CONSENSUS)

        # 计算涌现
        emergence = self._calculate_dialogue_emergence(
            consensus, proposal_A, proposal_B
        )

        # 计算共识质量
        consensus_quality = self._evaluate_consensus_quality(
            consensus, emergence,
            len(stages_completed)
        )

        # 更新统计
        self._update_stats(
            len(self.dialogue_history),
            emergence,
            consensus_quality > 0.7,
            stages_completed
        )

        return ConsensusResult(
            action=consensus['action'],
            confidence=consensus['confidence'],
            emergence=emergence,
            dialogue_length=len(self.dialogue_history),
            stages_completed=stages_completed,
            initial_agreement=initial_agreement,
            final_agreement=consensus.get('agreement', 0.0),
            consensus_quality=consensus_quality,
            breakdown={
                'proposal_A': proposal_A,
                'proposal_B': proposal_B,
                'dialogue_history': self.dialogue_history.copy(),
                'final_consensus': consensus
            }
        )

    def _create_proposal(
        self,
        result: Dict[str, Any],
        speaker: str
    ) -> DialogueMessage:
        """创建初始提案"""

        action = result.get('action', 0)
        confidence = result.get('confidence', 0.5)

        # 生成论据
        argument = self._generate_proposal_argument(
            action, confidence, speaker
        )

        return DialogueMessage(
            speaker=speaker,
            stage=DialogueStage.PROPOSAL,
            action=action,
            confidence=confidence,
            argument=argument,
            timestamp=time.time()
        )

    def _create_challenge(
        self,
        my_proposal: DialogueMessage,
        their_proposal: DialogueMessage,
        speaker: str
    ) -> DialogueMessage:
        """创建质疑"""

        # 质疑策略：寻找对方提案的弱点
        action_diff = abs(my_proposal.action - their_proposal.action)
        conf_diff = abs(my_proposal.confidence - their_proposal.confidence)

        # 如果对方置信度低，质疑其不确定性
        if their_proposal.confidence < 0.6:
            argument = f"对方提案action={their_proposal.action}置信度过低({their_proposal.confidence:.2f})，存在较大不确定性"
            adjusted_confidence = my_proposal.confidence * 1.1
        # 如果动作差异大，质疑其合理性
        elif action_diff > 1:
            argument = f"对方提案action={their_proposal.action}与我方提案差异显著({action_diff})，需要进一步论证"
            adjusted_confidence = my_proposal.confidence * 1.05
        else:
            argument = f"对对方提案action={their_proposal.action}持保留意见"
            adjusted_confidence = my_proposal.confidence

        adjusted_confidence = min(1.0, adjusted_confidence)

        return DialogueMessage(
            speaker=speaker,
            stage=DialogueStage.CHALLENGE,
            action=my_proposal.action,
            confidence=adjusted_confidence,
            argument=argument,
            timestamp=time.time()
        )

    def _create_defense(
        self,
        my_proposal: DialogueMessage,
        their_challenge: DialogueMessage,
        speaker: str
    ) -> DialogueMessage:
        """创建辩护"""

        # 辩护策略：强化自身提案的合理性
        if their_challenge.stage == DialogueStage.CHALLENGE:
            # 根据质疑内容进行辩护
            if "置信度过低" in their_challenge.argument:
                argument = f"我方提案action={my_proposal.action}基于充分的考虑，置信度{my_proposal.confidence:.2f}反映了决策的审慎性"
                adjusted_confidence = my_proposal.confidence * 1.05
            elif "差异显著" in their_challenge.argument:
                argument = f"我方提案action={my_proposal.action}有独特的考量，差异恰恰体现了视角的多样性"
                adjusted_confidence = my_proposal.confidence * 1.03
            else:
                argument = f"坚持我方提案action={my_proposal.action}的合理性"
                adjusted_confidence = my_proposal.confidence
        else:
            argument = f"我方提案action={my_proposal.action}是最优选择"
            adjusted_confidence = my_proposal.confidence

        adjusted_confidence = min(1.0, adjusted_confidence)

        return DialogueMessage(
            speaker=speaker,
            stage=DialogueStage.DEFENSE,
            action=my_proposal.action,
            confidence=adjusted_confidence,
            argument=argument,
            timestamp=time.time()
        )

    def _negotiate(
        self,
        defense_A: DialogueMessage,
        defense_B: DialogueMessage,
        initial_agreement: float
    ) -> DialogueMessage:
        """协商阶段：寻找共同点"""

        # 计算当前一致度
        current_agreement = self._calculate_agreement(defense_A, defense_B)

        # 协商策略
        if defense_A.action == defense_B.action:
            # 动作相同，直接达成共识
            argument = f"双方均认为action={defense_A.action}为最优，达成一致"
            consensus_action = defense_A.action
            consensus_confidence = (defense_A.confidence + defense_B.confidence) / 2
        elif abs(defense_A.action - defense_B.action) == 1:
            # 动作相近，寻找折中方案
            consensus_action = int((defense_A.action + defense_B.action) / 2)
            consensus_confidence = (defense_A.confidence + defense_B.confidence) / 2 * 1.1
            argument = f"双方提案相近({defense_A.action} vs {defense_B.action})，建议折中方案action={consensus_action}"
        else:
            # 动作差异大，选择置信度更高的
            if defense_A.confidence > defense_B.confidence:
                consensus_action = defense_A.action
                argument = f"采纳系统A提案action={defense_A.action}（置信度更高）"
            else:
                consensus_action = defense_B.action
                argument = f"采纳系统B提案action={defense_B.action}（置信度更高）"
            consensus_confidence = max(defense_A.confidence, defense_B.confidence)

        consensus_confidence = min(1.0, consensus_confidence)

        return DialogueMessage(
            speaker='CONSENSUS',
            stage=DialogueStage.NEGOTIATION,
            action=consensus_action,
            confidence=consensus_confidence,
            argument=argument,
            timestamp=time.time()
        )

    def _reach_consensus(
        self,
        proposal_A: DialogueMessage,
        proposal_B: DialogueMessage,
        dialogue_history: List[DialogueMessage],
        initial_agreement: float
    ) -> Dict[str, Any]:
        """达成最终共识"""

        # 找到最后的协商消息
        negotiation_msgs = [
            m for m in dialogue_history
            if m.stage == DialogueStage.NEGOTIATION
        ]

        if negotiation_msgs:
            last_negotiation = negotiation_msgs[-1]
            consensus_action = last_negotiation.action
            consensus_confidence = last_negotiation.confidence

            # 计算最终一致度
            final_agreement = self._calculate_agreement(
                proposal_A, proposal_B
            ) + 0.2  # 对话提高了理解
        else:
            # 如果没有协商消息，使用原始提案
            if proposal_A.confidence > proposal_B.confidence:
                consensus_action = proposal_A.action
                consensus_confidence = proposal_A.confidence
            else:
                consensus_action = proposal_B.action
                consensus_confidence = proposal_B.confidence

            final_agreement = initial_agreement

        # 对话增强：多轮对话本身提升了决策质量
        dialogue_enhancement = min(0.1, len(dialogue_history) * 0.01)
        consensus_confidence = min(1.0, consensus_confidence + dialogue_enhancement)

        return {
            'action': consensus_action,
            'confidence': consensus_confidence,
            'agreement': final_agreement,
            'dialogue_enhancement': dialogue_enhancement
        }

    def _calculate_agreement(
        self,
        msg_A: DialogueMessage,
        msg_B: DialogueMessage
    ) -> float:
        """计算两个消息的一致度"""

        # 动作一致度
        action_agreement = 1.0 if msg_A.action == msg_B.action else max(0, 1 - abs(msg_A.action - msg_B.action) / 4)

        # 置信度一致度
        conf_diff = abs(msg_A.confidence - msg_B.confidence)
        conf_agreement = 1.0 - conf_diff

        # 综合一致度
        overall_agreement = 0.6 * action_agreement + 0.4 * conf_agreement

        return overall_agreement

    def _calculate_dialogue_emergence(
        self,
        consensus: Dict[str, Any],
        proposal_A: DialogueMessage,
        proposal_B: DialogueMessage
    ) -> float:
        """计算对话涌现分数"""

        # 最终共识置信度
        final_confidence = consensus['confidence']

        # 初始最大置信度
        max_initial_confidence = max(proposal_A.confidence, proposal_B.confidence)

        # 涌现 = 最终置信度 - 初始最大置信度
        emergence = final_confidence - max_initial_confidence

        # 如果涌现为正，说明对话产生了协同效应
        return max(0.0, emergence)

    def _evaluate_consensus_quality(
        self,
        consensus: Dict[str, Any],
        emergence: float,
        stages_count: int
    ) -> float:
        """评估共识质量"""

        # 质量维度
        confidence = consensus['confidence']
        agreement = consensus.get('agreement', 0.5)
        enhancement = consensus.get('dialogue_enhancement', 0.0)

        # 综合质量分数
        quality = (
            0.4 * confidence +
            0.3 * agreement +
            0.2 * (1.0 if emergence > 0 else 0.5) +
            0.1 * min(1.0, stages_count / 5.0)
        )

        return quality

    def _form_immediate_consensus(
        self,
        proposal_A: DialogueMessage,
        proposal_B: DialogueMessage,
        initial_agreement: float,
        start_time: float
    ) -> ConsensusResult:
        """快速达成共识（初始一致度高）"""

        # 使用加权平均
        consensus_action = int(
            (proposal_A.action * proposal_A.confidence +
             proposal_B.action * proposal_B.confidence) /
            (proposal_A.confidence + proposal_B.confidence)
        )

        consensus_confidence = (
            proposal_A.confidence + proposal_B.confidence
        ) / 2 * 1.05  # 一致性奖励
        consensus_confidence = min(1.0, consensus_confidence)

        emergence = consensus_confidence - max(
            proposal_A.confidence,
            proposal_B.confidence
        )

        return ConsensusResult(
            action=consensus_action,
            confidence=consensus_confidence,
            emergence=max(0.0, emergence),
            dialogue_length=2,  # 只有2个提案
            stages_completed=[DialogueStage.PROPOSAL, DialogueStage.CONSENSUS],
            initial_agreement=initial_agreement,
            final_agreement=initial_agreement,
            consensus_quality=0.9,  # 高一致度 = 高质量
            breakdown={
                'proposal_A': proposal_A,
                'proposal_B': proposal_B,
                'dialogue_history': [proposal_A, proposal_B],
                'final_consensus': {
                    'action': consensus_action,
                    'confidence': consensus_confidence
                }
            }
        )

    def _generate_proposal_argument(
        self,
        action: int,
        confidence: float,
        speaker: str
    ) -> str:
        """生成提案论据"""

        if confidence > 0.8:
            certainty = "高度确定"
        elif confidence > 0.6:
            certainty = "较为确定"
        else:
            certainty = "有一定把握"

        return f"系统{speaker}建议采取action={action}，{certainty}(置信度={confidence:.2f})"

    def _update_stats(
        self,
        dialogue_length: int,
        emergence: float,
        is_high_quality: bool,
        stages: List[DialogueStage]
    ):
        """更新统计"""

        self.stats['total_dialogues'] += 1

        # 平均对话长度
        if self.stats['total_dialogues'] == 1:
            self.stats['avg_dialogue_length'] = dialogue_length
        else:
            self.stats['avg_dialogue_length'] = (
                self.stats['avg_dialogue_length'] * 0.9 +
                dialogue_length * 0.1
            )

        # 平均涌现
        if self.stats['total_dialogues'] == 1:
            self.stats['avg_emergence'] = emergence
        else:
            self.stats['avg_emergence'] = (
                self.stats['avg_emergence'] * 0.9 +
                emergence * 0.1
            )

        # 共识率
        if is_high_quality:
            self.stats['consensus_rate'] = (
                self.stats['consensus_rate'] * 0.9 +
                1.0 * 0.1
            )

        # 阶段分布
        for stage in stages:
            stage_name = stage.value
            if stage_name not in self.stats['stage_distribution']:
                self.stats['stage_distribution'][stage_name] = 0
            self.stats['stage_distribution'][stage_name] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

    def reset(self):
        """重置引擎"""

        self.dialogue_history.clear()
        self.stats = {
            'total_dialogues': 0,
            'avg_dialogue_length': 0,
            'consensus_rate': 0,
            'avg_emergence': 0,
            'stage_distribution': {}
        }

        logger.info("[辩论引擎] 已重置")


# 测试代码
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*18 + "辩论式共识引擎测试")
    print("="*70)

    engine = DialogueEngine(
        max_rounds=3,
        agreement_threshold=0.8
    )

    print(f"\n[初始化] 辩论引擎创建成功")
    print(f"[配置] 最大轮数=3, 共识阈值=0.8")

    # 测试场景1：高度一致的提案
    print(f"\n[场景1] 高度一致的提案")
    print("="*70)

    result_A1 = {'action': 1, 'confidence': 0.85}
    result_B1 = {'action': 1, 'confidence': 0.82}

    consensus1 = engine.engage_dialogue(result_A1, result_B1)

    print(f"系统A提案: action={result_A1['action']}, conf={result_A1['confidence']:.2f}")
    print(f"系统B提案: action={result_B1['action']}, conf={result_B1['confidence']:.2f}")
    print(f"\n共识结果:")
    print(f"  最终动作: {consensus1.action}")
    print(f"  最终置信度: {consensus1.confidence:.4f}")
    print(f"  涌现分数: {consensus1.emergence:.4f}")
    print(f"  对话长度: {consensus1.dialogue_length}")
    print(f"  初始一致度: {consensus1.initial_agreement:.4f}")
    print(f"  最终一致度: {consensus1.final_agreement:.4f}")
    print(f"  共识质量: {consensus1.consensus_quality:.4f}")
    print(f"  完成阶段: {[s.value for s in consensus1.stages_completed]}")

    # 测试场景2：有差异的提案
    print(f"\n[场景2] 有差异的提案（需要辩论）")
    print("="*70)

    result_A2 = {'action': 1, 'confidence': 0.75}
    result_B2 = {'action': 2, 'confidence': 0.70}

    # 重置对话历史
    engine.dialogue_history.clear()

    consensus2 = engine.engage_dialogue(result_A2, result_B2)

    print(f"系统A提案: action={result_A2['action']}, conf={result_A2['confidence']:.2f}")
    print(f"系统B提案: action={result_B2['action']}, conf={result_B2['confidence']:.2f}")
    print(f"\n共识结果:")
    print(f"  最终动作: {consensus2.action}")
    print(f"  最终置信度: {consensus2.confidence:.4f}")
    print(f"  涌现分数: {consensus2.emergence:.4f}")
    print(f"  对话长度: {consensus2.dialogue_length}")
    print(f"  初始一致度: {consensus2.initial_agreement:.4f}")
    print(f"  最终一致度: {consensus2.final_agreement:.4f}")
    print(f"  共识质量: {consensus2.consensus_quality:.4f}")
    print(f"  完成阶段: {[s.value for s in consensus2.stages_completed]}")

    # 显示对话历史（简版）
    print(f"\n对话过程:")
    for i, msg in enumerate(consensus2.breakdown['dialogue_history'][:10]):
        print(f"  {i+1}. [{msg.speaker}][{msg.stage.value}] action={msg.action}, conf={msg.confidence:.3f}")
        print(f"     \"{msg.argument[:50]}...\"")

    # 测试场景3：显著差异的提案
    print(f"\n[场景3] 显著差异的提案（多轮辩论）")
    print("="*70)

    result_A3 = {'action': 0, 'confidence': 0.9}
    result_B3 = {'action': 3, 'confidence': 0.85}

    # 重置对话历史
    engine.dialogue_history.clear()

    consensus3 = engine.engage_dialogue(result_A3, result_B3)

    print(f"系统A提案: action={result_A3['action']}, conf={result_A3['confidence']:.2f}")
    print(f"系统B提案: action={result_B3['action']}, conf={result_B3['confidence']:.2f}")
    print(f"\n共识结果:")
    print(f"  最终动作: {consensus3.action}")
    print(f"  最终置信度: {consensus3.confidence:.4f}")
    print(f"  涌现分数: {consensus3.emergence:.4f}")
    print(f"  对话长度: {consensus3.dialogue_length}")
    print(f"  初始一致度: {consensus3.initial_agreement:.4f}")
    print(f"  最终一致度: {consensus3.final_agreement:.4f}")
    print(f"  共识质量: {consensus3.consensus_quality:.4f}")

    # 显示最终统计
    print("\n" + "="*70)
    print(" "*25 + "统计信息")
    print("="*70)

    stats = engine.get_statistics()
    print(f"\n总对话数: {stats['total_dialogues']}")
    print(f"平均对话长度: {stats['avg_dialogue_length']:.1f}")
    print(f"共识率: {stats['consensus_rate']:.2%}")
    print(f"平均涌现: {stats['avg_emergence']:.4f}")

    print(f"\n阶段分布:")
    for stage, count in stats['stage_distribution'].items():
        print(f"  {stage}: {count}次")

    print("\n" + "="*70 + "\n")
