#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组件别名映射 (Component Aliases)
===============================

功能：统一架构图名称和实际文件名/类名
解决类名不一致导致的导入问题

作者：Claude Code (Sonnet 4.5)
日期：2026-01-26
版本：1.0.0

使用方法：
    from core.component_aliases import *
    # 现在可以使用架构图中的名称了
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============================================================
# 瓶颈系统 (Bottleneck Systems)
# ============================================================

# 瓶颈1: 超深度推理引擎
# 架构图名称: UltraDeepReasoningEngine
# 实际文件: core/deep_reasoning_engine.py
# 实际类名: UltraDeepReasoningEngine
from core.deep_reasoning_engine import UltraDeepReasoningEngine

# 瓶颈2: 自主目标系统
# 架构图名称: AutonomousGoalSystem
# 实际文件: core/autonomous_goal_system.py
# 实际类名: AutonomousGoalGenerator
from core.autonomous_goal_system import AutonomousGoalGenerator as AutonomousGoalSystem

# 瓶颈3: 跨域迁移系统
# 架构图名称: CrossDomainTransferSystem
# 实际文件: core/cross_domain_transfer.py
# 实际类名: CrossDomainTransferSystem
from core.cross_domain_transfer import CrossDomainTransferSystem


# ============================================================
# LLM优先架构 (LLM-First Architecture)
# ============================================================

# LLM优先对话引擎
# 架构图名称: LLMFirstDialogueEngine
# 实际文件: core/llm_first_dialogue_v2.py
# 实际类名: LLMFirstDialogueEngineV2 (还有DialogueHistoryManager)
from core.llm_first_dialogue_v2 import (
    LLMFirstDialogueEngineV2 as LLMFirstDialogueEngine,
    DialogueHistoryManager
)

# 幻觉感知LLM引擎
# 架构图名称: HallucinationAwareLLMEngine
# 实际文件: core/hallucination_aware_llm.py
# 实际类名: HallucinationAwareLLMEngine
from core.hallucination_aware_llm import HallucinationAwareLLMEngine

# 认知桥接层
# 架构图名称: CognitiveBridge
# 实际文件: core/cognitive_bridge.py
# 实际类名: CognitiveBridge
from core.cognitive_bridge import CognitiveBridge


# ============================================================
# 分形AGI组件 (M1-M4 Fractal AGI)
# ============================================================

# M1: 元学习器
# 架构图名称: MetaLearner
# 实际文件: core/meta_learner.py
# 实际类名: MetaLearner
from core.meta_learner import MetaLearner, MetaLearningConfig

# M2: 目标提问器
# 架构图名称: GoalQuestioner
# 实际文件: core/goal_questioner.py
# 实际类名: GoalQuestioner
from core.goal_questioner import GoalQuestioner

# M3: 自我修改引擎
# 架构图名称: SelfModifyingEngine
# 实际文件: core/self_modifying_engine.py
# 实际类名: SelfModifyingEngine
from core.self_modifying_engine import SelfModifyingEngine

# M4: 递归自我记忆
# 架构图名称: RecursiveSelfMemory
# 实际文件: core/recursive_self_memory.py
# 实际类名: RecursiveSelfMemory
from core.recursive_self_memory import RecursiveSelfMemory


# ============================================================
# 记忆系统 (Memory Systems)
# ============================================================

# 经验记忆
# 架构图名称: ExperienceMemory
# 实际文件: core/experience_manager.py
# 实际类名: ExperienceManager
from core.experience_manager import ExperienceManager as ExperienceMemory

# 工作记忆
# 架构图名称: WorkingMemory
# 实际文件: core/working_memory.py
# 实际类名: ShortTermWorkingMemory
from core.working_memory import ShortTermWorkingMemory as WorkingMemory

# 拓扑记忆
# 架构图名称: TopologyMemory
# 实际文件: core/memory/topology_memory.py
# 实际类名: TopologicalMemoryCore
from core.memory.topology_memory import TopologicalMemoryCore as TopologyMemory


# ============================================================
# 双螺旋引擎 (Double Helix Engine V2)
# ============================================================

from core.double_helix_engine_v2 import (
    DoubleHelixEngineV2,
    FusionMode,
    DoubleHelixResult,
    HelixContext
)

# 双螺旋依赖组件
from core.seed import TheSeed
from core.fractal_intelligence import create_fractal_intelligence


# ============================================================
# 其他核心组件
# ============================================================

# 世界模型
from core.bayesian_world_model import BayesianWorldModel as WorldModel

# 因果推理
from core.causal_reasoning import CausalReasoningEngine

# 组件协调器
from agi_consciousness_coordinator import (
    AGIConsciousnessCoordinator,
    get_coordinator,
    ConsciousnessLayer,
    ResourcePriority
)

# 意图对话桥接
from intent_dialogue_bridge import (
    IntentDialogueBridge,
    get_intent_bridge,
    Intent,
    IntentState,
    IntentDepth
)


# ============================================================
# 导出列表
# ============================================================

__all__ = [
    # 瓶颈系统
    'UltraDeepReasoningEngine',
    'AutonomousGoalSystem',
    'CrossDomainTransferSystem',

    # LLM优先架构
    'LLMFirstDialogueEngine',
    'DialogueHistoryManager',
    'HallucinationAwareLLMEngine',
    'CognitiveBridge',

    # 分形AGI M1-M4
    'MetaLearner',
    'MetaLearningConfig',
    'GoalQuestioner',
    'SelfModifyingEngine',
    'RecursiveSelfMemory',

    # 记忆系统
    'ExperienceMemory',
    'WorkingMemory',
    'TopologyMemory',

    # 双螺旋引擎
    'DoubleHelixEngineV2',
    'FusionMode',
    'DoubleHelixResult',
    'HelixContext',
    'TheSeed',
    'create_fractal_intelligence',

    # 其他核心组件
    'WorldModel',
    'CausalReasoningEngine',
    'AGIConsciousnessCoordinator',
    'get_coordinator',
    'ConsciousnessLayer',
    'ResourcePriority',

    # 意图桥接
    'IntentDialogueBridge',
    'get_intent_bridge',
    'Intent',
    'IntentState',
    'IntentDepth',
]


# ============================================================
# 辅助函数
# ============================================================

def get_component_info():
    """
    获取所有组件的映射信息

    Returns:
        dict: 架构图名称 -> (实际模块路径, 实际类名)
    """
    return {
        # 瓶颈系统
        'UltraDeepReasoningEngine': ('core.deep_reasoning_engine', 'UltraDeepReasoningEngine'),
        'AutonomousGoalSystem': ('core.autonomous_goal_system', 'AutonomousGoalGenerator'),
        'CrossDomainTransferSystem': ('core.cross_domain_transfer', 'CrossDomainTransferSystem'),

        # LLM优先架构
        'LLMFirstDialogueEngine': ('core.llm_first_dialogue_v2', 'LLMFirstDialogueEngineV2'),
        'HallucinationAwareLLMEngine': ('core.hallucination_aware_llm', 'HallucinationAwareLLMEngine'),
        'CognitiveBridge': ('core.cognitive_bridge', 'CognitiveBridge'),

        # 分形AGI
        'MetaLearner': ('core.meta_learner', 'MetaLearner'),
        'GoalQuestioner': ('core.goal_questioner', 'GoalQuestioner'),
        'SelfModifyingEngine': ('core.self_modifying_engine', 'SelfModifyingEngine'),
        'RecursiveSelfMemory': ('core.recursive_self_memory', 'RecursiveSelfMemory'),

        # 记忆系统
        'ExperienceMemory': ('core.experience_manager', 'ExperienceManager'),
        'WorkingMemory': ('core.working_memory', 'ShortTermWorkingMemory'),
        'TopologyMemory': ('core.memory.topology_memory', 'TopologicalMemoryCore'),

        # 双螺旋
        'DoubleHelixEngineV2': ('core.double_helix_engine_v2', 'DoubleHelixEngineV2'),
        'TheSeed': ('core.seed', 'TheSeed'),
        'FractalIntelligence': ('core.fractal_intelligence', 'create_fractal_intelligence'),
    }


def print_component_mappings():
    """打印所有组件映射关系（用于调试）"""
    info = get_component_info()

    print("=" * 80)
    print("组件映射关系表")
    print("=" * 80)

    categories = {
        '瓶颈系统': ['UltraDeepReasoningEngine', 'AutonomousGoalSystem', 'CrossDomainTransferSystem'],
        'LLM优先架构': ['LLMFirstDialogueEngine', 'HallucinationAwareLLMEngine', 'CognitiveBridge'],
        '分形AGI (M1-M4)': ['MetaLearner', 'GoalQuestioner', 'SelfModifyingEngine', 'RecursiveSelfMemory'],
        '记忆系统': ['ExperienceMemory', 'WorkingMemory', 'TopologyMemory'],
        '双螺旋引擎': ['DoubleHelixEngineV2', 'TheSeed', 'FractalIntelligence'],
    }

    for category, components in categories.items():
        print(f"\n【{category}】")
        for arch_name in components:
            if arch_name in info:
                module_path, actual_class = info[arch_name]
                print(f"  {arch_name}")
                print(f"    → {module_path}.{actual_class}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 修复Windows编码
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # 测试所有导入
    print("测试组件别名导入...\n")

    try:
        # 测试瓶颈系统
        print("✅ 瓶颈系统:")
        print(f"   UltraDeepReasoningEngine: {UltraDeepReasoningEngine}")
        print(f"   AutonomousGoalSystem: {AutonomousGoalSystem}")
        print(f"   CrossDomainTransferSystem: {CrossDomainTransferSystem}")

        # 测试分形AGI
        print("\n✅ 分形AGI:")
        print(f"   MetaLearner: {MetaLearner}")
        print(f"   GoalQuestioner: {GoalQuestioner}")
        print(f"   SelfModifyingEngine: {SelfModifyingEngine}")
        print(f"   RecursiveSelfMemory: {RecursiveSelfMemory}")

        # 测试记忆系统
        print("\n✅ 记忆系统:")
        print(f"   ExperienceMemory: {ExperienceMemory}")
        print(f"   WorkingMemory: {WorkingMemory}")
        print(f"   TopologyMemory: {TopologyMemory}")

        print("\n✅ 所有组件别名导入成功！")

        # 打印映射关系
        print("\n")
        print_component_mappings()

    except Exception as e:
        print(f"\n❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
