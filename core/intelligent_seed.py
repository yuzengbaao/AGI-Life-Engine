
# ============ 增强的Seed包装器 ============
# 文件: core/intelligent_seed.py

import time
import random
from core.working_memory import ShortTermWorkingMemory

class IntelligentSeed:
    """
    增强的Seed模块
    集成短期工作记忆，打破思想循环
    """

    def __init__(self, base_seed, capacity=7):
        self.base_seed = base_seed
        self.working_memory = ShortTermWorkingMemory(capacity=capacity)
        self.upgrade_enabled = True

    def generate_thought_chain(self, base_guidance):
        """生成增强的思想链"""
        if not self.upgrade_enabled:
            return base_guidance.get('thought_chain', '')

        # 获取基础思想链
        base_chain = base_guidance.get('thought_chain', '')
        if not base_chain:
            return ''

        # 解析思想链
        thoughts = base_chain.split(' => ')
        processed_thoughts = []

        for thought_str in thoughts:
            thought_str = thought_str.strip()
            if not thought_str:
                continue

            # 解析动作和概念
            action = 'unknown'
            concept = ''

            if '->' in thought_str:
                parts = thought_str.split('->')
                if len(parts) >= 1:
                    action_part = parts[0].strip()
                    if action_part.startswith('(') and ')' in action_part:
                        action = action_part[1:action_part.index(')')]

                if len(parts) >= 2:
                    concept = parts[1].strip()

            # 添加到工作记忆
            thought_obj = self.working_memory.add_thought(action, concept)
            processed_thoughts.append(str(thought_obj))

        # 检测到循环打破？
        if len(processed_thoughts) != len(thoughts):
            # 循环被打破了
            print(f"   [智能Seed] [BREAK] 思想循环已打破")

        # 重新组装思想链
        enhanced_chain = ' => '.join(processed_thoughts)
        return enhanced_chain

    def get_intelligence_status(self):
        """获取智能状态"""
        return self.working_memory.get_context_summary()
