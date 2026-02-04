#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
短期工作记忆（Short-term Working Memory）
======================================

功能：打破思想循环，维持推理连贯性
基于：Miller's Magic Number 7±2

版本: 1.0.0
"""

import time
import random
import hashlib
from typing import List, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class Thought:
    """思想单元"""
    content: str
    timestamp: float
    concept_id: str
    action: str
    context: Dict[str, Any]

    def __str__(self):
        return f"({self.action}) -> {self.concept_id}"


class ShortTermWorkingMemory:
    """
    短期工作记忆系统

    核心功能：
    1. 维持活跃思想（容量限制）
    2. 检测思想循环
    3. 打破循环（发散思维）
    4. 追踪推理上下文
    """

    def __init__(self, capacity: int = 7, loop_threshold: int = 3):
        """
        初始化短期工作记忆

        Args:
            capacity: 记忆容量（默认7，符合认知科学）
            loop_threshold: 检测循环的最小重复次数
        """
        self.capacity = capacity
        self.loop_threshold = loop_threshold

        # 记忆存储
        self.active_thoughts = deque(maxlen=capacity)  # 活跃思想（FIFO）
        self.thought_history = []  # 完整历史（用于检测长期循环）

        # 上下文追踪
        self.current_context = {}
        self.context_changes = []

        # 🆕 [2026-01-15] 概念冷却机制：防止重复概念立即出现
        self.concept_cooldown = {}  # {concept_id: remaining_ticks}

        # 统计信息
        self.stats = {
            'total_thoughts': 0,
            'loops_detected': 0,
            'loops_broken': 0,
            'divergent_thoughts': 0,
            'concepts_cooled': 0  # 新增：冷却的概念计数
        }
        
        # 🆕 [2026-01-17] 简单键值存储（兼容性接口）
        self._kv_store = {}

    # 🆕 [2026-01-17] 兼容性方法：简单键值存储
    def add(self, key: str, value: Any) -> Optional['Thought']:
        """
        兼容性方法：简单键值存储
        
        Args:
            key: 存储键
            value: 存储值（任意类型）
            
        Returns:
            思想对象（如果同时添加到思想流）
        """
        self._kv_store[key] = value
        
        # 同时添加到思想流
        if isinstance(value, dict):
            action = value.get('action', 'store')
            concept = value.get('concept', str(key))
        else:
            action = 'store'
            concept = str(value)[:50]  # 截断过长内容
        
        return self.add_thought(action, concept, context={'key': key, 'value': value})
    
    def get(self, key: str) -> Optional[Any]:
        """
        兼容性方法：简单键值检索
        
        Args:
            key: 检索键
            
        Returns:
            存储的值，不存在返回None
        """
        return self._kv_store.get(key)

    def add_thought(self, action: str, concept: str,
                    context: Optional[Dict] = None) -> Thought:
        """
        添加新思想

        🔧 [2026-01-16] P0修复：在创建思想前检查冷却状态，避免过度触发
        🔧 [2026-01-16] P1修复：检查并执行动作持久性要求

        Args:
            action: 动作类型（analyze, explore, create等）
            concept: 概念ID或内容
            context: 上下文信息

        Returns:
            Thought: 思想对象（可能被修改以打破循环）
        """
        # 🆕 [2026-01-16] P1修复：检查是否有强制动作要求
        if self.active_thoughts:
            last_thought = self.active_thoughts[-1]

            # 检查是否有强制动作标记
            if 'forced_action' in last_thought.context and 'force_duration' in last_thought.context:
                forced_action = last_thought.context['forced_action']
                force_duration = last_thought.context['force_duration']

                # 如果仍在强制期内，覆盖当前动作
                if force_duration > 0 and action != forced_action:
                    print(f"  [WorkingMemory] [FORCE-ACTION] 动作强制: {action} → {forced_action} (剩余: {force_duration})")
                    action = forced_action

                    # 更新剩余持续时间
                    last_thought.context['force_duration'] = force_duration - 1

        # 🆕 [2026-01-16] P0修复：预先检查概念是否在冷却期
        original_concept_id = self._generate_concept_id(concept)
        concept_id = original_concept_id
        if concept_id in self.concept_cooldown and self.concept_cooldown[concept_id] > 0:
            # 🔧 [2026-01-16] P0修复v2：循环验证直到找到真正的新概念
            max_attempts = 10
            for attempt in range(max_attempts):
                new_concept = self._generate_divergent_concept()
                new_concept_id = self._generate_concept_id(new_concept)

                # 验证新概念不在冷却期且与原概念不同
                if (new_concept_id != original_concept_id and
                    (new_concept_id not in self.concept_cooldown or self.concept_cooldown[new_concept_id] <= 0)):
                    concept = new_concept
                    concept_id = new_concept_id
                    # 🔧 [2026-01-30] P0 FIX: 降低日志频率（每10次打印一次）
                    if (attempt + 1) % 10 == 0 or attempt == max_attempts - 1:
                        print(f"  [WorkingMemory] [COOLDOWN] {original_concept_id} → 尝试{attempt+1}")
                    self.stats['concepts_cooled'] += 1
                    break
            else:
                # 所有尝试都失败，使用紧急生成
                concept = f"Emergency_{random.randint(0, 0xFFFFFF):08x}"
                concept_id = self._generate_concept_id(concept)
                # 🔧 [2026-01-30] P0 FIX: 紧急生成时才打印日志
                print(f"  [WorkingMemory] [EMERGENCY] 概念耗尽，生成紧急概念: {concept_id}")
                # 🆕 [2026-01-17] 为紧急生成的概念也添加冷却期，避免元循环
                self.concept_cooldown[concept_id] = 3  # 3步冷却
                self.stats['concepts_cooled'] += 1

        # 创建思想对象
        thought = Thought(
            content=concept,
            timestamp=time.time(),
            concept_id=concept_id,
            action=action,
            context=context or self.current_context
        )

        # 检测循环
        is_looping, loop_info = self._detect_loop(thought)

        if is_looping:
            self.stats['loops_detected'] += 1
            print(f"  [WorkingMemory] [LOOP] 检测到循环: {loop_info}")

            # 打破循环
            modified_thought = self._break_loop(thought)
            self.stats['loops_broken'] += 1

            # 添加到记忆
            self.active_thoughts.append(modified_thought)
            self.thought_history.append(modified_thought)
            self.stats['total_thoughts'] += 1

            return modified_thought
        else:
            # 正常添加
            self.active_thoughts.append(thought)
            self.thought_history.append(thought)
            self.stats['total_thoughts'] += 1

            return thought

    def _generate_concept_id(self, content: str) -> str:
        """
        生成概念ID（减少哈希冲突）

        🔧 [2026-01-16] P2修复：使用更长的哈希（前12位而非8位），降低冲突概率
        """
        # 使用哈希确保相同内容产生相同ID
        hash_obj = hashlib.md5(content.encode())
        hash_hex = hash_obj.hexdigest()

        # 🆕 使用前12位（原为8位），冲突概率从1/4B降至1/16T
        return f"C{hash_hex[:12]}"

    def _detect_loop(self, new_thought: Thought) -> tuple[bool, str]:
        """
        检测思想循环

        🔧 [2026-01-16] P0修复：移除冷却期检查（已在add_thought中预先处理）
        🔧 [2026-01-16] P0修复v2：新增长期循环检测，防止逃避检测

        Returns:
            (是否循环, 循环信息)
        """
        if len(self.active_thoughts) < self.loop_threshold:
            return False, ""

        # 检查最近N次思想是否重复
        recent = list(self.active_thoughts)[-self.loop_threshold:]

        # 检查概念ID是否相同
        concept_ids = [t.concept_id for t in recent]

        # 简单重复检测
        if len(set(concept_ids)) == 1 and concept_ids[0] == new_thought.concept_id:
            return True, f"简单重复: {new_thought.concept_id}"

        # 🆕 长期循环检测（检查历史中同一概念出现频率）
        if len(self.thought_history) >= 10:
            recent_history = list(self.thought_history)[-20:]
            history_concept_ids = [t.concept_id for t in recent_history]

            # 统计新概念在历史中的出现频率
            target_count = history_concept_ids.count(new_thought.concept_id)
            if target_count >= 10:  # 最近20次中出现10次以上
                return True, f"长期循环: {new_thought.concept_id} (频率: {target_count}/20)"

        # 检查动作是否陷入循环
        actions = [t.action for t in recent + [new_thought]]
        if len(set(actions)) == 1 and actions[0] in ['analyze', 'rest']:
            return True, f"动作循环: {actions[0]}"

        # 检查是否在两个状态间震荡
        if len(self.active_thoughts) >= 4:
            last_four = list(self.active_thoughts)[-4:]
            actions_four = [t.action for t in last_four]
            if actions_four == ['analyze', 'explore', 'analyze', 'explore']:
                return True, "震荡循环"

        return False, ""

    def _break_loop(self, thought: Thought) -> Thought:
        """
        打破思想循环

        🔧 P1修复: 增强动作多样性，彻底打破explore循环
        🆕 [2026-01-15] P2修复: 添加概念冷却机制，防止循环概念立即重复

        策略：
        1. 改变动作类型（增强版）
        2. 生成新概念
        3. 将旧概念标记为冷却状态
        4. 注入随机性
        5. 添加动作持久性标记
        """
        print(f"  [WorkingMemory] [BREAK] 打破循环: {thought.action} -> ", end="")

        # 记录需要冷却的概念（打破循环前的概念）
        old_concept = thought.concept_id

        # 🔧 P1修复: 增强的动作映射（增加多样性）
        action_map = {
            'analyze': ['create', 'integrate', 'rest'],  # 分析 → 创建/整合/休息
            'explore': ['analyze', 'integrate', 'create'],  # 探索 → 分析/整合/创建（强制远离explore）
            'create': ['analyze', 'integrate', 'explore'],
            'integrate': ['analyze', 'create', 'explore'],
            'rest': ['analyze', 'create', 'explore']
        }

        # 如果动作循环，切换动作
        if thought.action in action_map:
            old_action = thought.action
            # 🔧 P1修复: 随机选择一个不同的动作，增加多样性
            alternative_actions = action_map[old_action]
            new_action = random.choice(alternative_actions)
            thought.action = new_action
            print(f"动作切换: {old_action} → {thought.action}")

            # 🔧 P1修复: 添加动作持久性标记，防止立即切回
            thought.context['forced_action'] = new_action
            thought.context['force_duration'] = random.randint(3, 5)  # 强制保持3-5步

        # 策略2: 生成新概念
        if self._should_generate_new_concept(thought):
            thought.concept_id = self._generate_divergent_concept()
            thought.content = f"Novel_{thought.concept_id}"
            print(f"概念切换: {old_concept} → {thought.concept_id}")
            self.stats['divergent_thoughts'] += 1

        # 🔧 [2026-01-16] P0修复：将旧概念标记为冷却状态（3个tick，降低以减少过度触发）
        self._cooldown_concept(old_concept, cooldown_ticks=3)

        # 策略4: 添加"打破循环"标记
        thought.context['loop_break'] = True
        thought.context['previous_loop'] = self._get_loop_pattern()

        return thought

    def _should_generate_new_concept(self, thought: Thought) -> bool:
        """决定是否需要生成新概念"""
        # 如果概念重复率高，生成新概念
        if len(self.active_thoughts) < 3:
            return False

        recent_concepts = [t.concept_id for t in list(self.active_thoughts)[-3:]]
        unique_ratio = len(set(recent_concepts)) / len(recent_concepts)

        # 如果唯一率<50%，生成新概念
        return unique_ratio < 0.5

    def _generate_divergent_concept(self) -> str:
        """
        生成发散概念（避免重复）

        🔧 [2026-01-20] 优化：扩大历史概念池（50 → 100 → 2000），在性能和内存间取得平衡
        🔧 [2026-01-16] P1修复：优先选择低频概念，使用时间戳确保唯一性
        """
        # 策略0：优先选择历史中少见的概念
        if self.thought_history and len(self.thought_history) >= 5:
            recent_history = list(self.thought_history)[-2000:]  # 🆕 50→100→2000

            # 统计概念频率
            concept_counts = {}
            for t in recent_history:
                concept_counts[t.concept_id] = concept_counts.get(t.concept_id, 0) + 1

            # 按频率升序排序，选择最少见的（避免重复高频概念）
            rare_concepts = sorted(concept_counts.items(), key=lambda x: x[1])[:5]

            if rare_concepts:
                selected_concept_id = rare_concepts[0][0]
                # 检查是否在冷却期
                if selected_concept_id not in self.concept_cooldown or self.concept_cooldown[selected_concept_id] <= 0:
                    return f"RecallRare_{selected_concept_id}"

        # 策略1：从历史中选择不同动作类型的概念
        if self.thought_history and len(self.thought_history) >= 5:
            recent_history = list(self.thought_history)[-2000:]  # 🆕 50→100→2000

            # 统计动作频率
            action_counts = {}
            for t in recent_history:
                action_counts[t.action] = action_counts.get(t.action, 0) + 1

            # 选择最少见的动作
            rare_actions = sorted(action_counts.items(), key=lambda x: x[1])[:2]
            if rare_actions:
                target_action = rare_actions[0][0]

                # 从使用该动作的历史思想中选择
                candidates = [t for t in recent_history if t.action == target_action]
                if candidates:
                    selected = random.choice(candidates)
                    return f"SwitchAction_{selected.concept_id}"

        # 🆕 策略1.5：语义变体生成（2026-01-20 新增）
        if self.thought_history and len(self.thought_history) >= 10:
            recent_history = list(self.thought_history)[-2000:]  # 🆕 100→2000

            # 随机选择一个历史概念作为基础
            base_thought = random.choice(recent_history)
            base_concept = base_thought.content

            # 生成语义变体（添加修饰词、变换视角）
            semantic_variants = [
                f"Reflect_{base_concept}",      # 反思变体
                f"Explore_{base_concept}",      # 探索变体
                f"Deep_{base_concept}",         # 深度变体
                f"Meta_{base_concept}",         # 元认知变体
                f"Anti_{base_concept}",         # 反向变体
            ]

            # 随机选择一个变体
            variant = random.choice(semantic_variants)

            # 检查变体是否在冷却期
            variant_id = self._generate_concept_id(variant)
            if variant_id not in self.concept_cooldown or self.concept_cooldown[variant_id] <= 0:
                # 添加时间戳后缀确保唯一性
                timestamp_suffix = int(time.time() * 1000) & 0xFFF
                return f"{variant}_{timestamp_suffix:03x}"

        # 策略2：生成全新概念（使用时间戳确保唯一性）
        # 🆕 [2026-01-20] 优化：增强时间戳随机性，避免冲突

        # 使用高精度时间戳（微秒级）
        high_precision_time = int(time.time() * 1_000_000)  # 微秒

        # 添加多个随机源增加熵
        random_sources = [
            random.randint(0, 0xFFFFFF),      # 随机数1（24位）
            random.randint(0, 0xFFFF),        # 随机数2（16位）
            hash(str(high_precision_time)) & 0xFFFF,  # 时间戳哈希
            id(object()) & 0xFFF,             # 对象ID
        ]

        # 混合随机源
        mixed_entropy = 0
        for i, source in enumerate(random_sources):
            mixed_entropy ^= (source << (i * 8))  # 异或混合

        # 组合时间戳和随机熵
        timestamp_part = high_precision_time & 0xFFFFFFF  # 28位时间戳
        entropy_part = mixed_entropy & 0xFFFFFFF           # 28位随机熵

        return f"Novel_{timestamp_part:08x}_{entropy_part:08x}"

    def _get_loop_pattern(self) -> str:
        """获取当前循环模式"""
        if len(self.active_thoughts) < 2:
            return "unknown"

        recent = list(self.active_thoughts)[-5:]
        actions = [t.action for t in recent]
        return " → ".join(actions)

    def get_context_summary(self) -> Dict[str, Any]:
        """获取当前上下文摘要"""
        return {
            'active_thoughts_count': len(self.active_thoughts),
            'current_action': self.active_thoughts[-1].action if self.active_thoughts else None,
            'recent_concepts': [t.concept_id for t in list(self.active_thoughts)[-3:]],
            'diversity': self._calculate_diversity(),
            'stats': self.stats
        }

    def _calculate_diversity(self) -> float:
        """计算思想多样性"""
        if len(self.active_thoughts) < 2:
            return 1.0

        concepts = [t.concept_id for t in self.active_thoughts]
        unique = len(set(concepts))

        return unique / len(concepts)

    def get_thought_chain(self, n: int = 10) -> List[str]:
        """获取最近的思想链"""
        recent = list(self.active_thoughts)[-n:]
        return [str(t) for t in recent]

    # 🆕 [2026-01-15] 概念冷却机制方法

    def _cooldown_concept(self, concept_id: str, cooldown_ticks: int = 5):
        """
        将概念标记为冷却状态，防止立即重复

        🔧 [2026-01-20] 优化：增加默认冷却时间（3 → 5），减少概念重复触发
        🔧 [2026-01-16] P0修复：降低默认冷却时间（5 → 3），减少过度触发

        Args:
            concept_id: 概念ID
            cooldown_ticks: 冷却tick数量（默认5，上次为3，最初为5）
        """
        self.concept_cooldown[concept_id] = cooldown_ticks
        self.stats['concepts_cooled'] += 1

    def _force_concept_switch(self, thought: Thought):
        """
        强制切换概念（用于冷却期检测）

        Args:
            thought: 需要修改的思想对象
        """
        # 生成新概念
        old_concept = thought.concept_id
        thought.concept_id = self._generate_divergent_concept()
        thought.content = f"Switched_{thought.concept_id}"

        print(f"  [WorkingMemory] [FORCE] 概念强制切换: {old_concept} → {thought.concept_id}")
        self.stats['divergent_thoughts'] += 1

    def tick_cooldown(self):
        """
        更新所有概念的冷却状态（每个tick调用一次）

        将所有在冷却中的概念的剩余tick数减1，移除已冷却完成的概念
        """
        to_remove = []
        for concept_id in self.concept_cooldown:
            if self.concept_cooldown[concept_id] > 0:
                self.concept_cooldown[concept_id] -= 1
                if self.concept_cooldown[concept_id] <= 0:
                    to_remove.append(concept_id)

        # 移除已冷却完成的概念
        for concept_id in to_remove:
            del self.concept_cooldown[concept_id]

    def get_cooldown_status(self) -> Dict[str, int]:
        """获取当前冷却状态"""
        return self.concept_cooldown.copy()

    def clear(self):
        """清空工作记忆"""
        self.active_thoughts.clear()
        print(f"  [WorkingMemory] [CLEAR] 工作记忆已清空")

    def __repr__(self):
        return f"ShortTermWorkingMemory(capacity={self.capacity}, " \
               f"active={len(self.active_thoughts)}, " \
               f"diversity={self._calculate_diversity():.2f})"


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("短期工作记忆测试")
    print("=" * 60)

    # 创建工作记忆
    wm = ShortTermWorkingMemory(capacity=7, loop_threshold=5)  # \ud83d\udd27 2026-01-17: 3\u21925

    # 测试1: 正常添加
    print("\n[测试1] 正常添加思想")
    for i in range(5):
        thought = wm.add_thought("analyze", f"Concept_{i}")
        print(f"  添加: {thought}")

    print(f"  多样性: {wm._calculate_diversity():.2f}")

    # 测试2: 触发循环
    print("\n[测试2] 触发循环检测")
    for i in range(5):
        thought = wm.add_thought("analyze", f"Concept_{5}")  # 重复相同概念
        print(f"  添加: {thought}")

    # 测试3: 查看摘要
    print("\n[测试3] 上下文摘要")
    summary = wm.get_context_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # 测试4: 思想链
    print("\n[测试4] 思想链")
    chain = wm.get_thought_chain(10)
    for i, thought in enumerate(chain, 1):
        print(f"  {i}. {thought}")
