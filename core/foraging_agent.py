"""
Foraging Agent - 主动学习与探索模块
基于信息觅食理论 (Information Foraging Theory) 实现

功能:
1. 当好奇心高 (>0.7) 时主动探索未知领域
2. 识别知识空白 (Knowledge Gaps)
3. 提出探索性问题和实验
4. 优化探索路径 (Exploration vs. Exploitation)
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import time

logger = logging.getLogger(__name__)

class ForagingAgent:
    """
    信息觅食智能体
    
    核心机制:
    - Patch Selection: 选择最有价值的知识领域探索
    - Information Scent: 评估信息价值线索
    - Optimal Foraging: 平衡探索/利用
    """
    
    def __init__(self, 
                 curiosity_threshold: float = 0.7,
                 exploration_budget: int = 10,
                 min_info_gain: float = 0.3):
        """
        初始化觅食智能体
        
        Args:
            curiosity_threshold: 触发主动探索的好奇心阈值
            exploration_budget: 每次探索允许的最大步数
            min_info_gain: 最小信息增益阈值
        """
        self.curiosity_threshold = curiosity_threshold
        self.exploration_budget = exploration_budget
        self.min_info_gain = min_info_gain
        
        # 知识领域追踪
        self.knowledge_patches: Dict[str, Dict] = {}  # {patch_id: {value, visits, info_gain}}
        self.exploration_history: List[Dict] = []
        self.current_patch: Optional[str] = None
        self.steps_in_patch = 0
        
        # 统计
        self.total_explorations = 0
        self.successful_explorations = 0
        
    def should_explore(self, curiosity: float, entropy: float) -> bool:
        """
        判断是否应该触发主动探索
        
        Args:
            curiosity: 当前好奇心水平 (0-1)
            entropy: 当前系统熵 (0-1)
            
        Returns:
            是否触发探索
        """
        # 高好奇心触发
        if curiosity > self.curiosity_threshold:
            return True
            
        # 或者熵过低（陷入重复）也需要探索
        if entropy < 0.2:
            logger.info(f"[ForagingAgent] 🔍 Low entropy ({entropy:.2f}) triggers exploration")
            return True
            
        return False
    
    def identify_knowledge_gaps(self, 
                                knowledge_graph: Any,
                                memory_system: Any) -> List[Dict[str, Any]]:
        """
        识别知识空白区域
        
        策略:
        1. 孤立节点: 连接数<2的概念
        2. 断裂区域: 两个密集子图之间缺少桥接
        3. 未验证假设: 从未被测试的推理路径
        
        Returns:
            知识空白列表 [{gap_type, location, priority}]
        """
        gaps = []
        
        try:
            # 1. 检测孤立节点
            if hasattr(knowledge_graph, 'graph'):
                G = knowledge_graph.graph
                for node in G.nodes():
                    degree = G.degree(node)
                    if degree < 2:
                        gaps.append({
                            'type': 'isolated_node',
                            'location': node,
                            'priority': 0.8,
                            'description': f'Concept "{node}" has only {degree} connections'
                        })
            
            # 2. 检测未探索的领域（基于记忆访问频率）
            if hasattr(memory_system, 'get_access_stats'):
                stats = memory_system.get_access_stats()
                for concept, access_count in stats.items():
                    if access_count == 0:
                        gaps.append({
                            'type': 'unexplored_concept',
                            'location': concept,
                            'priority': 0.6,
                            'description': f'Concept "{concept}" never accessed'
                        })
            
            # 3. 检测推理死胡同（高错误率路径）
            # 这需要执行历史，暂时用占位符
            
            # 排序：按优先级降序
            gaps.sort(key=lambda x: x['priority'], reverse=True)
            
        except Exception as e:
            logger.error(f"[ForagingAgent] ❌ Error identifying gaps: {e}")
            
        return gaps[:10]  # 返回Top 10
    
    def select_exploration_target(self, 
                                  gaps: List[Dict],
                                  current_context: str = "") -> Optional[Dict]:
        """
        选择探索目标 (Patch Selection)
        
        使用信息价值评估:
        - Value = Priority × (1 - Visit_Frequency) × Context_Relevance
        
        Args:
            gaps: 知识空白列表
            current_context: 当前上下文（用于计算相关性）
            
        Returns:
            选中的探索目标
        """
        if not gaps:
            return None
        
        # 计算每个gap的价值
        scored_gaps = []
        for gap in gaps:
            patch_id = gap['location']
            
            # 访问频率惩罚
            visits = self.knowledge_patches.get(patch_id, {}).get('visits', 0)
            visit_penalty = 1.0 / (1.0 + visits)
            
            # 上下文相关性（简化：基于字符串相似度）
            relevance = 1.0  # 默认全相关
            if current_context:
                relevance = 0.5 + 0.5 * (
                    1.0 if current_context.lower() in patch_id.lower() else 0.3
                )
            
            value = gap['priority'] * visit_penalty * relevance
            
            scored_gaps.append({
                **gap,
                'value': value
            })
        
        # 选择最高价值
        scored_gaps.sort(key=lambda x: x['value'], reverse=True)
        selected = scored_gaps[0]
        
        logger.info(f"[ForagingAgent] 🎯 Selected exploration target: {selected['location']} "
                   f"(value={selected['value']:.2f})")
        
        return selected
    
    def generate_exploration_actions(self, target: Dict) -> List[Dict[str, Any]]:
        """
        生成探索行动序列
        
        根据gap类型生成不同的探索策略:
        - isolated_node: 寻找可能的连接
        - unexplored_concept: 深入分析其定义和用途
        - reasoning_dead_end: 尝试替代推理路径
        
        Args:
            target: 探索目标
            
        Returns:
            行动列表 [{action_type, params}]
        """
        actions = []
        gap_type = target['type']
        location = target['location']
        
        if gap_type == 'isolated_node':
            actions = [
                {
                    'action_type': 'search_relations',
                    'params': {'concept': location},
                    'description': f'Search for possible relations of {location}'
                },
                {
                    'action_type': 'query_llm',
                    'params': {
                        'prompt': f'What are the key characteristics and applications of {location}? '
                                 f'How does it relate to similar concepts?'
                    },
                    'description': f'Ask LLM about {location}'
                },
                {
                    'action_type': 'create_hypothesis',
                    'params': {'concept': location},
                    'description': f'Generate testable hypotheses about {location}'
                }
            ]
        
        elif gap_type == 'unexplored_concept':
            actions = [
                {
                    'action_type': 'deep_dive',
                    'params': {'concept': location},
                    'description': f'Deep analysis of {location}'
                },
                {
                    'action_type': 'analogy_search',
                    'params': {'concept': location},
                    'description': f'Find analogous concepts to {location}'
                }
            ]
        
        else:
            # 默认通用探索
            actions = [
                {
                    'action_type': 'investigate',
                    'params': {'target': location},
                    'description': f'General investigation of {location}'
                }
            ]
        
        return actions[:self.exploration_budget]
    
    def execute_foraging(self, 
                        curiosity: float,
                        entropy: float,
                        knowledge_graph: Any,
                        memory_system: Any,
                        current_context: str = "") -> Optional[Dict[str, Any]]:
        """
        执行主动觅食流程
        
        完整流程:
        1. 判断是否需要探索
        2. 识别知识空白
        3. 选择目标
        4. 生成行动
        5. 记录结果
        
        Returns:
            探索结果 {target, actions, expected_gain}
        """
        # 1. 判断触发
        if not self.should_explore(curiosity, entropy):
            return None
        
        logger.info(f"[ForagingAgent] 🚀 Triggered foraging (curiosity={curiosity:.2f}, entropy={entropy:.2f})")
        
        # 2. 识别空白
        gaps = self.identify_knowledge_gaps(knowledge_graph, memory_system)
        
        if not gaps:
            logger.warning("[ForagingAgent] ⚠️ No knowledge gaps identified")
            return None
        
        logger.info(f"[ForagingAgent] 📋 Identified {len(gaps)} knowledge gaps")
        
        # 3. 选择目标
        target = self.select_exploration_target(gaps, current_context)
        
        if not target:
            return None
        
        # 4. 生成行动
        actions = self.generate_exploration_actions(target)
        
        # 5. 更新状态
        patch_id = target['location']
        if patch_id not in self.knowledge_patches:
            self.knowledge_patches[patch_id] = {
                'value': target.get('value', 0),
                'visits': 0,
                'info_gain': []
            }
        
        self.knowledge_patches[patch_id]['visits'] += 1
        self.current_patch = patch_id
        self.steps_in_patch = 0
        self.total_explorations += 1
        
        result = {
            'timestamp': time.time(),
            'target': target,
            'actions': actions,
            'expected_gain': target.get('value', 0),
            'status': 'initiated'
        }
        
        self.exploration_history.append(result)
        
        logger.info(f"[ForagingAgent] ✅ Foraging plan created with {len(actions)} actions")
        
        return result
    
    def record_exploration_result(self, 
                                 patch_id: str,
                                 info_gain: float,
                                 success: bool):
        """
        记录探索结果
        
        Args:
            patch_id: 探索的知识领域
            info_gain: 实际信息增益
            success: 是否成功
        """
        if patch_id in self.knowledge_patches:
            self.knowledge_patches[patch_id]['info_gain'].append(info_gain)
        
        if success and info_gain >= self.min_info_gain:
            self.successful_explorations += 1
        
        logger.info(f"[ForagingAgent] 📊 Exploration result recorded: "
                   f"{patch_id} (gain={info_gain:.2f}, success={success})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_rate = (self.successful_explorations / max(1, self.total_explorations))
        
        return {
            'total_explorations': self.total_explorations,
            'successful_explorations': self.successful_explorations,
            'success_rate': success_rate,
            'explored_patches': len(self.knowledge_patches),
            'current_patch': self.current_patch,
            'history_length': len(self.exploration_history)
        }
