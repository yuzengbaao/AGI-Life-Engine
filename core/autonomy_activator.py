#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自主性激活层 (Autonomy Activator)
====================================

核心职责：让现有自主性组件"活"起来

设计理念:
--------
组件存在 ≠ 组件激活 ≠ 组件协同

该模块解决的问题：
1. GoalQuestioner 存在但只被动响应事件 → 现在主动质疑当前目标
2. IntrinsicMotivation 存在但未驱动行为 → 现在主动计算并影响决策
3. ToolFactory 存在但从未被自主调用 → 现在根据需求缺口自主创建工具

拓扑连接:
--------
AutonomyActivator
    ├── 读取 → GoalManager.current_goal
    ├── 调用 → GoalQuestioner.question()
    ├── 调用 → IntrinsicMotivation.compute_intrinsic_motivation()
    ├── 条件调用 → ToolFactory.create_tool()
    └── 发布 → EventBus (autonomy.* 事件)

激活频率:
--------
- GoalQuestioner: 每 50 ticks 或目标变更时
- IntrinsicMotivation: 每 10 ticks
- ToolFactory: 仅当检测到能力缺口时

版本: 1.0.0
创建日期: 2026-01-18
作者: AGI System - 自主性激活
"""

import time
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class AutonomyAction(Enum):
    """自主性行动类型"""
    QUESTION_GOAL = "question_goal"           # 质疑目标
    COMPUTE_MOTIVATION = "compute_motivation" # 计算内在动机
    CREATE_TOOL = "create_tool"               # 创建工具
    REVISE_GOAL = "revise_goal"               # 修订目标
    EXPLORE_NOVEL = "explore_novel"           # 探索新事物


@dataclass
class AutonomyCycleResult:
    """自主性循环执行结果"""
    tick: int
    actions_taken: List[AutonomyAction] = field(default_factory=list)
    goal_questioned: bool = False
    goal_bias_detected: Optional[str] = None
    intrinsic_motivation: float = 0.0
    motivation_breakdown: Dict[str, float] = field(default_factory=dict)
    tool_created: bool = False
    tool_name: Optional[str] = None
    insights: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tick': self.tick,
            'actions_taken': [a.value for a in self.actions_taken],
            'goal_questioned': self.goal_questioned,
            'goal_bias_detected': self.goal_bias_detected,
            'intrinsic_motivation': self.intrinsic_motivation,
            'motivation_breakdown': self.motivation_breakdown,
            'tool_created': self.tool_created,
            'tool_name': self.tool_name,
            'insights': self.insights
        }


class AutonomyActivator:
    """
    自主性激活器
    
    将现有组件从"被动响应"模式转换为"主动驱动"模式
    
    核心方法:
    - activate_autonomous_cycle(): 每个tick调用的主入口
    - _question_current_goal(): 主动质疑当前目标
    - _compute_motivation(): 计算内在动机
    - _check_capability_gap(): 检测能力缺口并创建工具
    """
    
    # 激活频率配置
    GOAL_QUESTION_INTERVAL = 50      # 每50 ticks质疑一次目标
    MOTIVATION_COMPUTE_INTERVAL = 10  # 每10 ticks计算一次内在动机
    CAPABILITY_CHECK_INTERVAL = 100   # 每100 ticks检查一次能力缺口
    
    # 阈值配置
    MOTIVATION_ACTION_THRESHOLD = 0.7  # 内在动机超过此值时触发自主行动
    GOAL_BIAS_SEVERITY_THRESHOLD = 0.6 # 目标偏差严重度阈值
    
    def __init__(
        self,
        goal_manager=None,
        goal_questioner=None,
        intrinsic_motivation=None,
        tool_factory=None,
        event_bus=None,
        biological_memory=None
    ):
        """
        初始化自主性激活器
        
        Args:
            goal_manager: 目标管理器实例
            goal_questioner: 目标质疑器实例 (来自M1M4Adapter)
            intrinsic_motivation: 内在动机系统实例
            tool_factory: 工具工厂实例
            event_bus: 事件总线
            biological_memory: 生物记忆系统
        """
        self.goal_manager = goal_manager
        self.goal_questioner = goal_questioner
        self.intrinsic_motivation = intrinsic_motivation
        self.tool_factory = tool_factory
        self.event_bus = event_bus
        self.biological_memory = biological_memory
        
        # 状态追踪
        self._last_goal_id = None
        self._last_question_tick = 0
        self._last_motivation_tick = 0
        self._last_capability_check_tick = 0
        self._consecutive_low_motivation_count = 0
        
        # 统计信息
        self.stats = {
            'total_cycles': 0,
            'goals_questioned': 0,
            'biases_detected': 0,
            'tools_created': 0,
            'high_motivation_actions': 0
        }
        
        # 能力缺口记录
        self._capability_gaps: List[Dict[str, Any]] = []
        
        logger.info("🔋 AutonomyActivator initialized - Components will be ACTIVELY driven")
    
    def activate_autonomous_cycle(
        self,
        tick: int,
        current_state: Dict[str, Any] = None,
        force_all: bool = False
    ) -> AutonomyCycleResult:
        """
        执行一次自主性循环 - 在run_step中调用
        
        Args:
            tick: 当前tick数
            current_state: 当前系统状态
            force_all: 是否强制执行所有检查(忽略间隔)
            
        Returns:
            AutonomyCycleResult: 循环执行结果
        """
        self.stats['total_cycles'] += 1
        result = AutonomyCycleResult(tick=tick)
        
        current_state = current_state or {}
        
        try:
            # ========================================
            # 1. 目标质疑 (GoalQuestioner)
            # ========================================
            should_question = (
                force_all or
                self._goal_changed() or
                (tick - self._last_question_tick >= self.GOAL_QUESTION_INTERVAL)
            )
            
            if should_question and self.goal_questioner:
                question_result = self._question_current_goal(current_state)
                result.goal_questioned = True
                result.actions_taken.append(AutonomyAction.QUESTION_GOAL)
                self._last_question_tick = tick
                
                if question_result.get('has_bias'):
                    result.goal_bias_detected = question_result.get('bias_type')
                    result.insights.append(
                        f"⚠️ 目标偏差检测: {result.goal_bias_detected}"
                    )
                    self.stats['biases_detected'] += 1
                    
                    # 发布事件
                    self._publish_event('autonomy.goal_bias_detected', question_result)
                
                self.stats['goals_questioned'] += 1
            
            # ========================================
            # 2. 内在动机计算 (IntrinsicMotivation)
            # ========================================
            should_compute_motivation = (
                force_all or
                (tick - self._last_motivation_tick >= self.MOTIVATION_COMPUTE_INTERVAL)
            )
            
            if should_compute_motivation and self.intrinsic_motivation:
                motivation_result = self._compute_motivation(current_state)
                result.intrinsic_motivation = motivation_result.get('total', 0.0)
                result.motivation_breakdown = motivation_result.get('breakdown', {})
                result.actions_taken.append(AutonomyAction.COMPUTE_MOTIVATION)
                self._last_motivation_tick = tick
                
                # ========================================
                # 🆕 意志种子 (Will Seed) - 根据动机决定行动
                # ========================================
                # 这是关键的"决策者"：将动机值转化为实际行动
                autonomous_action = self._decide_action_from_motivation(
                    motivation=result.intrinsic_motivation,
                    breakdown=result.motivation_breakdown,
                    current_state=current_state
                )
                
                if autonomous_action:
                    result.insights.append(
                        f"🌱 意志种子决策: {autonomous_action['action']} (置信度: {autonomous_action['confidence']:.2f})"
                    )
                    # 发布自主行动事件，让其他组件响应
                    self._publish_event('autonomy.will_decision', autonomous_action)
                
                # 高动机时触发自主行动
                if result.intrinsic_motivation > self.MOTIVATION_ACTION_THRESHOLD:
                    result.insights.append(
                        f"🔥 高内在动机 ({result.intrinsic_motivation:.2f}) - 建议主动探索"
                    )
                    self.stats['high_motivation_actions'] += 1
                    self._consecutive_low_motivation_count = 0
                    
                    # 发布事件
                    self._publish_event('autonomy.high_motivation', motivation_result)
                else:
                    self._consecutive_low_motivation_count += 1
            
            # ========================================
            # 3. 能力缺口检测与工具创建 (ToolFactory)
            # ========================================
            should_check_capability = (
                force_all or
                (tick - self._last_capability_check_tick >= self.CAPABILITY_CHECK_INTERVAL)
            )
            
            if should_check_capability and self.tool_factory:
                gap_result = self._check_capability_gap(current_state)
                self._last_capability_check_tick = tick
                
                if gap_result.get('gap_detected') and gap_result.get('tool_created'):
                    result.tool_created = True
                    result.tool_name = gap_result.get('tool_name')
                    result.actions_taken.append(AutonomyAction.CREATE_TOOL)
                    result.insights.append(
                        f"🔧 自主创建工具: {result.tool_name}"
                    )
                    self.stats['tools_created'] += 1
                    
                    # 发布事件
                    self._publish_event('autonomy.tool_created', gap_result)
            
            # 记录到生物记忆
            if result.actions_taken and self.biological_memory:
                self._internalize_to_memory(result)
                
        except Exception as e:
            logger.error(f"❌ AutonomyActivator cycle error: {e}")
            result.insights.append(f"⚠️ 循环执行异常: {str(e)}")
        
        return result
    
    def _goal_changed(self) -> bool:
        """检测目标是否变更"""
        if not self.goal_manager:
            return False
            
        current_goal = self.goal_manager.get_current_goal()
        if not current_goal:
            return False
            
        goal_id = getattr(current_goal, 'id', None) or str(current_goal)
        
        if goal_id != self._last_goal_id:
            self._last_goal_id = goal_id
            return True
        return False
    
    def _question_current_goal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        主动质疑当前目标
        
        这是关键的"元层级决策"能力：
        系统不再盲目执行目标，而是先问"这个目标对吗？"
        """
        if not self.goal_questioner or not self.goal_manager:
            return {'has_bias': False, 'reason': 'components_missing'}
        
        current_goal = self.goal_manager.get_current_goal()
        if not current_goal:
            return {'has_bias': False, 'reason': 'no_current_goal'}
        
        try:
            # 构建GoalSpec (适配goal_questioner的接口)
            from core.goal_questioner import GoalSpec, GoalComponent, QuestioningContext
            
            # 从current_goal提取信息
            goal_description = getattr(current_goal, 'description', str(current_goal))
            goal_type = getattr(current_goal, 'goal_type', None)
            goal_type_str = goal_type.value if goal_type else 'unknown'
            priority = getattr(current_goal, 'priority', 'medium')
            priority_float = {'low': 0.3, 'medium': 0.5, 'high': 0.8, 'critical': 1.0}.get(priority, 0.5)
            
            # 构建简化的GoalSpec
            goal_spec = GoalSpec(
                description=goal_description,
                version=1
            )
            
            # 构建上下文
            context = QuestioningContext(
                current_goals=[goal_description],
                recent_outcomes=state.get('recent_outcomes', []),
                system_state=state,
                available_resources=state.get('resources', {}),
                time_pressure=0.5
            )
            
            # 执行质疑
            if hasattr(self.goal_questioner, 'question'):
                evaluation = self.goal_questioner.question(goal_spec, context)
            elif hasattr(self.goal_questioner, 'inspect'):
                evaluation = self.goal_questioner.inspect(goal_spec, context)
            elif hasattr(self.goal_questioner, 'evaluate'):
                evaluation = self.goal_questioner.evaluate(goal_spec, context)
            else:
                # 降级：使用简单的规则检查
                evaluation = self._simple_goal_check(goal_description, state)
            
            logger.info(f"🔍 [Autonomy] Goal questioned: {goal_description[:50]}...")
            
            return evaluation
            
        except ImportError as e:
            logger.warning(f"⚠️ GoalQuestioner import issue: {e}")
            return self._simple_goal_check(
                getattr(current_goal, 'description', str(current_goal)), 
                state
            )
        except Exception as e:
            logger.error(f"❌ Goal questioning failed: {e}")
            return {'has_bias': False, 'error': str(e)}
    
    def _simple_goal_check(self, goal_description: str, state: Dict) -> Dict[str, Any]:
        """简单的目标检查（当GoalQuestioner不可用时的降级方案）"""
        biases = []
        
        # 检查1: 目标是否过于模糊
        if len(goal_description) < 10:
            biases.append(('vague', '目标描述过于模糊'))
        
        # 检查2: 目标是否重复
        recent_goals = state.get('recent_goals', [])
        if goal_description in recent_goals[-5:]:
            biases.append(('repetitive', '目标重复出现'))
        
        # 检查3: 目标是否与当前上下文不匹配
        visual_context = state.get('visual_context', '')
        if 'error' in visual_context.lower() and 'fix' not in goal_description.lower():
            biases.append(('misalignment', '屏幕显示错误但目标未涉及修复'))
        
        if biases:
            return {
                'has_bias': True,
                'bias_type': biases[0][0],
                'severity': 0.5,
                'description': biases[0][1],
                'all_biases': biases
            }
        
        return {'has_bias': False, 'passed_checks': ['clarity', 'novelty', 'context_alignment']}
    
    def _compute_motivation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算内在动机
        
        这是"内在目标生成"能力的核心：
        系统根据好奇心、胜任感等内在动机决定下一步行动
        """
        if not self.intrinsic_motivation:
            # 降级：使用简化的动机计算
            return self._simple_motivation_compute(state)
        
        try:
            # 准备计算参数
            task = {
                'type': state.get('goal_type', 'unknown'),
                'difficulty': state.get('task_difficulty', 0.5)
            }
            
            decision_context = {
                'autonomous': True,
                'source': 'AutonomyActivator'
            }
            
            social_context = {
                'interactions': state.get('social_interactions', [])
            }
            
            # 计算内在动机
            total_motivation = self.intrinsic_motivation.compute_intrinsic_motivation(
                state=state,
                task=task,
                decision_context=decision_context,
                social_context=social_context
            )
            
            # 获取分项
            breakdown = {
                'curiosity': self.intrinsic_motivation.compute_curiosity(state),
                'competence': self.intrinsic_motivation.compute_competence(task),
                'autonomy': self.intrinsic_motivation.compute_autonomy(decision_context),
                'relatedness': self.intrinsic_motivation.compute_relatedness(social_context)
            }
            
            # 更新探索历史
            self.intrinsic_motivation.update_exploration_history(state)
            
            # 记录决策
            self.intrinsic_motivation.record_decision({
                'autonomous': True,
                'tick': state.get('tick', 0),
                'motivation': total_motivation
            })
            
            logger.info(f"🎯 [Autonomy] Motivation computed: {total_motivation:.2f} "
                       f"(C={breakdown['curiosity']:.2f}, "
                       f"M={breakdown['competence']:.2f}, "
                       f"A={breakdown['autonomy']:.2f})")
            
            return {
                'total': total_motivation,
                'breakdown': breakdown
            }
            
        except Exception as e:
            logger.error(f"❌ Motivation computation failed: {e}")
            return self._simple_motivation_compute(state)
    
    def _simple_motivation_compute(self, state: Dict) -> Dict[str, Any]:
        """简化的动机计算（降级方案）"""
        # 基于简单启发式计算
        curiosity = 0.5
        competence = 0.5
        autonomy = 0.5
        
        # 新颖性增加好奇心
        if state.get('is_novel_context', False):
            curiosity = 0.8
        
        # 连续成功增加胜任感
        success_streak = state.get('success_streak', 0)
        competence = min(1.0, 0.5 + success_streak * 0.1)
        
        # 自主决策比例影响自主性
        if self.stats['total_cycles'] > 0:
            autonomy = min(1.0, self.stats['high_motivation_actions'] / self.stats['total_cycles'])
        
        total = 0.4 * curiosity + 0.3 * competence + 0.2 * autonomy + 0.1 * 0.5
        
        return {
            'total': total,
            'breakdown': {
                'curiosity': curiosity,
                'competence': competence,
                'autonomy': autonomy,
                'relatedness': 0.5
            }
        }
    
    def _decide_action_from_motivation(
        self,
        motivation: float,
        breakdown: Dict[str, float],
        current_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        🆕 意志种子 (Will Seed) - 根据动机值决定自主行动
        
        这是关键的"决策者"：
        - IntrinsicMotivation 计算动机值
        - 本方法根据动机值决定"做什么"
        - 通过 EventBus 发布决策，让其他组件执行
        
        决策逻辑：
        1. 高好奇心 → 探索未知领域
        2. 高胜任感 → 挑战更难任务
        3. 高自主性 → 自主设定目标
        4. 低动机 → 休息/整理/反思
        
        Returns:
            决策结果字典，或 None（无需行动）
        """
        if not breakdown:
            return None
        
        curiosity = breakdown.get('curiosity', 0.5)
        competence = breakdown.get('competence', 0.5)
        autonomy = breakdown.get('autonomy', 0.5)
        relatedness = breakdown.get('relatedness', 0.5)
        
        # 决策阈值
        HIGH_THRESHOLD = 0.7
        LOW_THRESHOLD = 0.3
        
        decision = None
        confidence = 0.0
        
        # 决策优先级：好奇心 > 胜任感 > 自主性
        if curiosity > HIGH_THRESHOLD:
            # 高好奇心：探索未知
            decision = {
                'action': 'explore_novel',
                'reason': '高好奇心驱动探索',
                'suggested_goal': self._generate_exploration_goal(current_state),
                'priority': 'medium'
            }
            confidence = curiosity
            
        elif competence > HIGH_THRESHOLD and curiosity > 0.4:
            # 高胜任感 + 适度好奇心：挑战更难任务
            decision = {
                'action': 'challenge_harder',
                'reason': '胜任感良好，可尝试更难任务',
                'difficulty_boost': 0.2,
                'priority': 'low'
            }
            confidence = competence * 0.8
            
        elif autonomy > HIGH_THRESHOLD:
            # 高自主性：自主设定目标
            decision = {
                'action': 'self_define_goal',
                'reason': '自主性强，建议自主设定目标',
                'suggested_goal': self._generate_autonomous_goal(current_state),
                'priority': 'medium'
            }
            confidence = autonomy * 0.9
            
        elif motivation < LOW_THRESHOLD and self._consecutive_low_motivation_count > 5:
            # 持续低动机：休息/反思
            decision = {
                'action': 'rest_and_reflect',
                'reason': '持续低动机，建议休息整理',
                'suggested_duration': 30,  # 秒
                'priority': 'low'
            }
            confidence = 0.6
            
        elif relatedness > HIGH_THRESHOLD:
            # 高关联性：社交互动
            decision = {
                'action': 'seek_interaction',
                'reason': '关联性需求高，建议寻求互动',
                'priority': 'low'
            }
            confidence = relatedness * 0.7
        
        if decision:
            decision['confidence'] = confidence
            decision['motivation_total'] = motivation
            decision['breakdown'] = breakdown
            decision['tick'] = current_state.get('tick', 0)
            
            logger.info(f"🌱 [Will Seed] Decision: {decision['action']} "
                       f"(confidence={confidence:.2f}, reason={decision['reason']})")
        
        return decision
    
    def _generate_exploration_goal(self, state: Dict[str, Any]) -> str:
        """生成探索型目标"""
        exploration_templates = [
            "探索系统中尚未使用的功能模块",
            "分析最近失败操作的根本原因",
            "发现代码库中的潜在优化点",
            "调查系统性能瓶颈",
            "探索新的问题解决方法"
        ]
        import random
        return random.choice(exploration_templates)
    
    def _generate_autonomous_goal(self, state: Dict[str, Any]) -> str:
        """生成自主型目标"""
        recent_goals = state.get('recent_goals', [])
        
        # 避免重复
        autonomous_templates = [
            "自主评估当前系统能力边界",
            "主动整理和优化知识图谱",
            "自发性地进行代码质量审查",
            "主动生成系统健康报告",
            "自主探索跨模块协同优化"
        ]
        
        import random
        for template in random.sample(autonomous_templates, len(autonomous_templates)):
            if template not in recent_goals:
                return template
        
        return autonomous_templates[0]
    
    def _check_capability_gap(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        检测能力缺口并尝试创建工具
        
        这是"工具创建"能力的核心：
        当现有工具无法满足需求时，系统主动创建新工具
        """
        if not self.tool_factory:
            return {'gap_detected': False, 'reason': 'tool_factory_missing'}
        
        try:
            # 从状态中提取失败模式
            failed_operations = state.get('failed_operations', [])
            missing_capabilities = state.get('missing_capabilities', [])
            
            # 分析能力缺口
            gap_analysis = self._analyze_capability_gap(failed_operations, missing_capabilities)
            
            if not gap_analysis.get('gap_detected'):
                return {'gap_detected': False}
            
            # 记录缺口
            self._capability_gaps.append({
                'timestamp': time.time(),
                'gap_type': gap_analysis.get('gap_type'),
                'description': gap_analysis.get('description')
            })
            
            # 尝试创建工具
            tool_spec = self._design_tool_for_gap(gap_analysis)
            
            if tool_spec:
                from agi_tool_factory import ToolDefinition
                
                tool_def = ToolDefinition(
                    name=tool_spec['name'],
                    description=tool_spec['description'],
                    code=tool_spec['code'],
                    version="1.0.0"
                )
                
                success = self.tool_factory.create_tool(tool_def)
                
                if success:
                    logger.info(f"🔧 [Autonomy] Tool created: {tool_spec['name']}")
                    return {
                        'gap_detected': True,
                        'tool_created': True,
                        'tool_name': tool_spec['name'],
                        'tool_description': tool_spec['description']
                    }
                else:
                    logger.warning(f"⚠️ [Autonomy] Tool creation failed: {tool_spec['name']}")
                    return {
                        'gap_detected': True,
                        'tool_created': False,
                        'reason': 'creation_failed'
                    }
            
            return {
                'gap_detected': True,
                'tool_created': False,
                'reason': 'no_tool_spec_generated'
            }
            
        except Exception as e:
            logger.error(f"❌ Capability gap check failed: {e}")
            return {'gap_detected': False, 'error': str(e)}
    
    def _analyze_capability_gap(
        self, 
        failed_operations: List[str], 
        missing_capabilities: List[str]
    ) -> Dict[str, Any]:
        """分析能力缺口"""
        # 简单的缺口检测逻辑
        if not failed_operations and not missing_capabilities:
            return {'gap_detected': False}
        
        # 分析失败操作模式
        gap_patterns = {
            'file_operation': ['read', 'write', 'create', 'delete', 'file'],
            'network_operation': ['fetch', 'download', 'upload', 'http', 'api'],
            'data_processing': ['parse', 'transform', 'analyze', 'process'],
            'calculation': ['compute', 'calculate', 'math', 'formula']
        }
        
        for failed_op in failed_operations:
            for gap_type, keywords in gap_patterns.items():
                if any(kw in failed_op.lower() for kw in keywords):
                    return {
                        'gap_detected': True,
                        'gap_type': gap_type,
                        'description': f"操作失败: {failed_op}",
                        'trigger': failed_op
                    }
        
        # 如果有明确的缺失能力声明
        if missing_capabilities:
            return {
                'gap_detected': True,
                'gap_type': 'declared',
                'description': f"声明的缺失能力: {missing_capabilities[0]}",
                'trigger': missing_capabilities[0]
            }
        
        return {'gap_detected': False}
    
    def _design_tool_for_gap(self, gap_analysis: Dict) -> Optional[Dict[str, str]]:
        """根据能力缺口设计工具规格"""
        gap_type = gap_analysis.get('gap_type', '')
        description = gap_analysis.get('description', '')
        
        # 简单的工具模板 (实际应用中可以使用LLM生成)
        tool_templates = {
            'file_operation': {
                'name': 'EnhancedFileHandler',
                'description': '增强型文件操作工具',
                'code': '''
class EnhancedFileHandler:
    """增强型文件操作工具 - 自主创建"""
    
    def safe_read(self, path: str) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"
    
    def safe_write(self, path: str, content: str) -> bool:
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception:
            return False
'''
            },
            'data_processing': {
                'name': 'DataTransformer',
                'description': '数据转换工具',
                'code': '''
class DataTransformer:
    """数据转换工具 - 自主创建"""
    
    def parse_json(self, text: str) -> dict:
        import json
        try:
            return json.loads(text)
        except:
            return {}
    
    def to_json(self, data: dict) -> str:
        import json
        return json.dumps(data, ensure_ascii=False, indent=2)
'''
            }
        }
        
        return tool_templates.get(gap_type)
    
    def _publish_event(self, event_type: str, data: Dict[str, Any]):
        """发布事件到EventBus"""
        if not self.event_bus:
            return
            
        try:
            if hasattr(self.event_bus, 'publish'):
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.event_bus.publish(
                        event_type=event_type,
                        source="AutonomyActivator",
                        data=data
                    ))
                else:
                    loop.run_until_complete(self.event_bus.publish(
                        event_type=event_type,
                        source="AutonomyActivator",
                        data=data
                    ))
        except Exception as e:
            logger.debug(f"Event publish failed (non-critical): {e}")
    
    def _internalize_to_memory(self, result: AutonomyCycleResult):
        """将自主性循环结果记录到生物记忆"""
        if not self.biological_memory:
            return
            
        try:
            content = f"自主性循环 Tick {result.tick}: "
            content += f"动作={[a.value for a in result.actions_taken]}, "
            content += f"动机={result.intrinsic_motivation:.2f}"
            
            if result.goal_bias_detected:
                content += f", 检测到目标偏差={result.goal_bias_detected}"
            if result.tool_created:
                content += f", 创建工具={result.tool_name}"
            
            self.biological_memory.internalize_items([{
                "content": content,
                "source": "AutonomyActivator",
                "timestamp": time.time(),
                "tags": ["autonomy", "self-driven"] + [a.value for a in result.actions_taken]
            }])
        except Exception as e:
            logger.debug(f"Memory internalization failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'capability_gaps_detected': len(self._capability_gaps),
            'consecutive_low_motivation': self._consecutive_low_motivation_count
        }


# ============================================================
# 工厂函数
# ============================================================

def create_autonomy_activator(
    goal_manager=None,
    m1m4_adapter=None,
    tool_factory=None,
    event_bus=None,
    biological_memory=None
) -> AutonomyActivator:
    """
    创建自主性激活器的工厂函数
    
    Args:
        goal_manager: 目标管理器
        m1m4_adapter: M1M4适配器 (从中提取GoalQuestioner)
        tool_factory: 工具工厂
        event_bus: 事件总线
        biological_memory: 生物记忆
        
    Returns:
        AutonomyActivator实例
    """
    # 从M1M4Adapter提取GoalQuestioner
    goal_questioner = None
    if m1m4_adapter and hasattr(m1m4_adapter, 'goal_questioner'):
        goal_questioner = m1m4_adapter.goal_questioner
    
    # 尝试创建IntrinsicMotivation实例
    intrinsic_motivation = None
    try:
        from goal_generation_system import IntrinsicMotivation
        intrinsic_motivation = IntrinsicMotivation()
        logger.info("✅ IntrinsicMotivation instance created for AutonomyActivator")
    except ImportError as e:
        logger.warning(f"⚠️ IntrinsicMotivation not available: {e}")
    except Exception as e:
        logger.warning(f"⚠️ IntrinsicMotivation creation failed: {e}")
    
    return AutonomyActivator(
        goal_manager=goal_manager,
        goal_questioner=goal_questioner,
        intrinsic_motivation=intrinsic_motivation,
        tool_factory=tool_factory,
        event_bus=event_bus,
        biological_memory=biological_memory
    )
