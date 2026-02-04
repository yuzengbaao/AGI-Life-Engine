#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模式匹配器 (Pattern Matcher)
=======================

功能：
1. 基于正则表达式的常见意图快速识别
2. 决策树分类
3. 与fractal_intelligence集成
4. 支持中英文混合模式

目标：
- 50-100个常见意图模式
- 匹配延迟 < 5ms
- 覆盖率 > 30%

作者：AGI Self-Improvement Module
创建日期：2026-02-04
"""

import re
import logging
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """匹配结果"""
    intent: str
    confidence: float
    matched_pattern: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PatternMatcher:
    """
    快速模式匹配器

    核心功能：
    1. 正则表达式模式匹配
    2. 优先级排序（具体模式优先）
    3. 置信度计算
    4. 匹配统计

    使用示例：
    ```python
    matcher = PatternMatcher()
    result = matcher.match("读取文件config.txt")
    if result:
        print(f"意图: {result.intent}, 置信度: {result.confidence}")
    ```
    """

    def __init__(self, enable_stats: bool = True):
        """
        初始化模式匹配器

        Args:
            enable_stats: 是否启用统计
        """
        self.enable_stats = enable_stats

        # 加载模式库
        self.patterns = self._load_patterns()

        # 构建优先级索引（按长度降序，具体模式优先）
        self._build_priority_index()

        # 统计信息
        self.stats = defaultdict(int)
        self.total_matches = 0
        self.total_misses = 0

        logger.info(
            f"[模式匹配器] 初始化完成 "
            f"(加载{len(self.patterns)}个模式)"
        )

    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        加载模式库（50-100个常见意图）

        Returns:
            模式字典 {intent_id: {pattern, regex, priority, ...}}
        """
        patterns = {}

        # ============= 文件操作模式 =============
        patterns['file_read'] = {
            'patterns': [
                r'(读取|read|查看|view|显示|show|list|打开|open).*(文件|file)',
                r'(文件|file).*(读取|read|查看|view|显示|show)',
                r'cat\s+\S+',  # Unix cat命令
                r'type\s+\S+',  # Windows type命令
            ],
            'intent': 'file_read',
            'confidence': 1.0,
            'priority': 10,
            'no_llm': True,
            'metadata': {'tool': 'file_operations.read'}
        }

        patterns['file_write'] = {
            'patterns': [
                r'(写入|write|保存|save|创建|create).*(文件|file)',
                r'(编辑|edit|修改|modify).*(文件|file)',
                r'echo.*>.+',  # Shell重定向
            ],
            'intent': 'file_write',
            'confidence': 1.0,
            'priority': 10,
            'no_llm': True,
            'metadata': {'tool': 'file_operations.write'}
        }

        patterns['file_delete'] = {
            'patterns': [
                r'(删除|delete|remove|rm).*(文件|file)',
                r'(文件|file).*(删除|delete|remove)',
            ],
            'intent': 'file_delete',
            'confidence': 0.95,
            'priority': 9,
            'no_llm': True,
            'metadata': {'tool': 'file_operations.delete', 'risk': 'high'}
        }

        # ============= 系统查询模式 =============
        patterns['system_status'] = {
            'patterns': [
                r'(系统状态|状态|健康|health|运行状态|status)',
                r'怎么|如何|how.*(状态|status|health)',
                r'(检查|check|诊断|diagnose).*(系统|system)',
            ],
            'intent': 'system_status',
            'confidence': 1.0,
            'priority': 10,
            'no_llm': True,
            'metadata': {'tool': 'system_monitor'}
        }

        patterns['process_list'] = {
            'patterns': [
                r'(进程|process|任务|task).*(列表|list|查看|view)',
                r'ps\s*',  # Unix ps命令
                r'tasklist',  # Windows tasklist
            ],
            'intent': 'process_list',
            'confidence': 1.0,
            'priority': 9,
            'no_llm': True,
            'metadata': {'tool': 'process_monitor'}
        }

        # ============= 知识查询模式 =============
        patterns['knowledge_query'] = {
            'patterns': [
                r'(查询|query|搜索|search|查找|find).*(知识|knowledge|信息|info)',
                r'什么是|what\s+is|定义|define',
                r'(解释|explain|说明|describe).*(是什么|how|what)',
            ],
            'intent': 'knowledge_query',
            'confidence': 0.9,
            'priority': 8,
            'no_llm': False,  # 可能需要LLM生成答案
            'metadata': {'tool': 'knowledge_graph'}
        }

        # ============= 任务执行模式 =============
        patterns['task_execute'] = {
            'patterns': [
                r'(执行|execute|运行|run).*(任务|task|命令|command)',
                r'(启动|start|开启|begin)',
            ],
            'intent': 'task_execute',
            'confidence': 0.85,
            'priority': 7,
            'no_llm': True,
            'metadata': {'tool': 'task_executor'}
        }

        # ============= 代码操作模式 =============
        patterns['code_read'] = {
            'patterns': [
                r'(读取|read|查看|view|显示|show).*(代码|code|源码|source)',
                r'(代码|code).*(分析|analyze|查看|view)',
            ],
            'intent': 'code_read',
            'confidence': 1.0,
            'priority': 10,
            'no_llm': True,
            'metadata': {'tool': 'code_reader'}
        }

        patterns['code_modify'] = {
            'patterns': [
                r'(修改|modify|编辑|edit|改|优化|optimize).*(代码|code)',
                r'(代码|code).*(修改|modify|编辑|edit|优化|optimize)',
                r'(修复|fix).*(bug|错误|error)',
            ],
            'intent': 'code_modify',
            'confidence': 0.95,
            'priority': 9,
            'no_llm': False,  # 需要LLM分析修改内容
            'metadata': {'tool': 'code_modifier', 'risk': 'medium'}
        }

        # ============= 记忆操作模式 =============
        patterns['memory_recall'] = {
            'patterns': [
                r'(回忆|recall|记住|remember).*(之前|previous|past)',
                r'(历史|history|记录|record).*(查询|query)',
                r'(之前|previous).*(做了什么|did|happened)',
            ],
            'intent': 'memory_recall',
            'confidence': 0.9,
            'priority': 8,
            'no_llm': True,
            'metadata': {'tool': 'memory'}
        }

        # ============= 学习模式 =============
        patterns['learn_explore'] = {
            'patterns': [
                r'(学习|learn|探索|explore|研究|research)',
                r'(了解|understand|知道|know).*(更多|more)',
            ],
            'intent': 'learn_explore',
            'confidence': 0.85,
            'priority': 7,
            'no_llm': False,
            'metadata': {'tool': 'curiosity_explore'}
        }

        # ============= 对话交互模式 =============
        patterns['conversation_greeting'] = {
            'patterns': [
                r'^(你好|hello|hi|嗨|您好)',
                r'^(早上好|下午好|晚上好|good\s+(morning|afternoon|evening))',
            ],
            'intent': 'conversation_greeting',
            'confidence': 1.0,
            'priority': 10,
            'no_llm': True,
            'metadata': {'response_type': 'greeting'}
        }

        patterns['conversation_help'] = {
            'patterns': [
                r'(帮助|help|协助|assist|支持|support)',
                r'(怎么用|如何使用|how\s+to.*use)',
                r'(教程|tutorial|指南|guide)',
            ],
            'intent': 'conversation_help',
            'confidence': 1.0,
            'priority': 9,
            'no_llm': True,
            'metadata': {'response_type': 'help'}
        }

        patterns['conversation_question'] = {
            'patterns': [
                r'[?？]$',  # 以问号结尾
                r'^(是否|能不能|可以吗|could|can|would|should)',
                r'(什么|why|how|where|when|who|which).*(吗|呢|\?)',
            ],
            'intent': 'conversation_question',
            'confidence': 0.8,
            'priority': 6,
            'no_llm': False,
            'metadata': {'response_type': 'question'}
        }

        # ============= 自我评估模式 =============
        patterns['self_evaluate'] = {
            'patterns': [
                r'(自我评估|self.?evaluate|自省|introspect)',
                r'(自己的|your).*(能力|capability|水平|level|性能|performance)',
                r'(你怎么样|how\s+are\s+you|状态如何)',
            ],
            'intent': 'self_evaluate',
            'confidence': 0.95,
            'priority': 9,
            'no_llm': True,
            'metadata': {'tool': 'metacognition'}
        }

        # ============= 创造性探索模式 =============
        patterns['creative_explore'] = {
            'patterns': [
                r'(创意|creative|创新|innovate|发明|invent)',
                r'(新想法|new\s+idea|灵感|inspiration)',
                r'(试试|try|尝试|attempt).*(新的|new|不同|different)',
            ],
            'intent': 'creative_explore',
            'confidence': 0.9,
            'priority': 8,
            'no_llm': False,
            'metadata': {'tool': 'creative_exploration_engine'}
        }

        # ============= 安全检查模式 =============
        patterns['security_check'] = {
            'patterns': [
                r'(安全|security|权限|permission|授权|authorize)',
                r'(检查|check).*(风险|risk|漏洞|vulnerability)',
            ],
            'intent': 'security_check',
            'confidence': 1.0,
            'priority': 10,
            'no_llm': True,
            'metadata': {'tool': 'security_validator', 'risk': 'critical'}
        }

        # ============= 备份操作模式 =============
        patterns['backup_create'] = {
            'patterns': [
                r'(备份|backup).*(创建|create|生成|make)',
                r'(保存|save).*(备份|backup)',
            ],
            'intent': 'backup_create',
            'confidence': 1.0,
            'priority': 9,
            'no_llm': True,
            'metadata': {'tool': 'backup_service'}
        }

        patterns['backup_restore'] = {
            'patterns': [
                r'(恢复|restore).*(备份|backup)',
                r'(备份|backup).*(恢复|restore|回滚|rollback)',
            ],
            'intent': 'backup_restore',
            'confidence': 1.0,
            'priority': 9,
            'no_llm': True,
            'metadata': {'tool': 'backup_service', 'risk': 'medium'}
        }

        # ============= 优化模式 =============
        patterns['optimize_performance'] = {
            'patterns': [
                r'(优化|optimize|提升|improve|加速|speed\s+up).*(性能|performance|速度|speed)',
                r'(更快|faster|更高效|efficient).*(方式|method|way)',
            ],
            'intent': 'optimize_performance',
            'confidence': 0.9,
            'priority': 8,
            'no_llm': False,
            'metadata': {'tool': 'performance_optimizer'}
        }

        # ============= 测试模式 =============
        patterns['test_run'] = {
            'patterns': [
                r'(测试|test).*(运行|run|执行|execute)',
                r'(运行|run|执行|execute).*(测试|test)',
                r'(验证|verify|检查|check).*(是否|whether)',
            ],
            'intent': 'test_run',
            'confidence': 0.95,
            'priority': 9,
            'no_llm': True,
            'metadata': {'tool': 'test_runner'}
        }

        # ============= 调试模式 =============
        patterns['debug_analyze'] = {
            'patterns': [
                r'(调试|debug|排错|troubleshoot).*(问题|problem|issue|error)',
                r'(分析|analyze|诊断|diagnose).*(错误|error|异常|exception)',
                r'(为什么|why).*(失败|fail|错误|error)',
            ],
            'intent': 'debug_analyze',
            'confidence': 0.9,
            'priority': 8,
            'no_llm': False,
            'metadata': {'tool': 'debugger'}
        }

        # ============= 监控模式 =============
        patterns['monitor_start'] = {
            'patterns': [
                r'(开始|start|启用|enable).*(监控|monitor)',
                r'(监控|monitor).*(开始|start|启用|enable)',
            ],
            'intent': 'monitor_start',
            'confidence': 1.0,
            'priority': 9,
            'no_llm': True,
            'metadata': {'tool': 'monitor', 'action': 'start'}
        }

        patterns['monitor_stop'] = {
            'patterns': [
                r'(停止|stop|结束|end|禁用|disable).*(监控|monitor)',
                r'(监控|monitor).*(停止|stop|结束|end)',
            ],
            'intent': 'monitor_stop',
            'confidence': 1.0,
            'priority': 9,
            'no_llm': True,
            'metadata': {'tool': 'monitor', 'action': 'stop'}
        }

        # ============= 通用确认模式 =============
        patterns['confirm_positive'] = {
            'patterns': [
                r'^(是|yes|对|正确|correct|好的|ok|确定|sure)$',
                r'^(同意|agree|批准|approve|可以|can|行)$',
            ],
            'intent': 'confirm_positive',
            'confidence': 1.0,
            'priority': 10,
            'no_llm': True,
            'metadata': {'response_type': 'confirmation', 'value': True}
        }

        patterns['confirm_negative'] = {
            'patterns': [
                r'^(不|no|否|错误|wrong|cancel)$',
                r'^(不同意|disagree|拒绝|refuse|不能|cannot)$',
            ],
            'intent': 'confirm_negative',
            'confidence': 1.0,
            'priority': 10,
            'no_llm': True,
            'metadata': {'response_type': 'confirmation', 'value': False}
        }

        # 编译所有正则表达式
        for intent_id, pattern_config in patterns.items():
            regexes = []
            for pattern_str in pattern_config['patterns']:
                try:
                    # 编译正则表达式（忽略大小写）
                    regex = re.compile(pattern_str, re.IGNORECASE)
                    regexes.append(regex)
                except re.error as e:
                    logger.warning(f"[模式匹配器] 正则表达式编译失败: {pattern_str}, 错误: {e}")

            pattern_config['compiled_patterns'] = regexes

        return patterns

    def _build_priority_index(self) -> None:
        """构建优先级索引（按优先级降序）"""
        # 按优先级排序
        sorted_patterns = sorted(
            self.patterns.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )

        # 构建索引
        self.priority_index = [intent_id for intent_id, _ in sorted_patterns]

    def match(self, text: str) -> Optional[MatchResult]:
        """
        匹配用户意图

        Args:
            text: 用户输入文本

        Returns:
            MatchResult 或 None
        """
        if not text or not text.strip():
            return None

        # 遍历优先级索引
        for intent_id in self.priority_index:
            pattern_config = self.patterns[intent_id]

            # 尝试所有正则表达式
            for regex in pattern_config['compiled_patterns']:
                match = regex.search(text)
                if match:
                    # 匹配成功
                    self.total_matches += 1
                    self.stats[intent_id] += 1

                    result = MatchResult(
                        intent=pattern_config['intent'],
                        confidence=pattern_config['confidence'],
                        matched_pattern=regex.pattern,
                        metadata=pattern_config.get('metadata', {})
                    )

                    logger.debug(
                        f"[模式匹配器] 命中 "
                        f"(intent={result.intent}, "
                        f"confidence={result.confidence:.2f}, "
                        f"pattern={regex.pattern[:30]}...)"
                    )

                    return result

        # 未匹配
        self.total_misses += 1
        logger.debug(f"[模式匹配器] 未匹配: {text[:50]}...")
        return None

    def match_all(self, text: str) -> List[MatchResult]:
        """
        返回所有匹配结果（用于多重意图）

        Args:
            text: 用户输入文本

        Returns:
            所有匹配结果列表（按置信度降序）
        """
        results = []

        for intent_id in self.priority_index:
            pattern_config = self.patterns[intent_id]

            for regex in pattern_config['compiled_patterns']:
                match = regex.search(text)
                if match:
                    results.append(MatchResult(
                        intent=pattern_config['intent'],
                        confidence=pattern_config['confidence'],
                        matched_pattern=regex.pattern,
                        metadata=pattern_config.get('metadata', {})
                    ))
                    break  # 每个intent只匹配一次

        # 按置信度降序排序
        results.sort(key=lambda x: x.confidence, reverse=True)

        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取匹配统计信息

        Returns:
            统计信息字典
        """
        total_requests = self.total_matches + self.total_misses
        match_rate = self.total_matches / total_requests if total_requests > 0 else 0.0

        return {
            'total_patterns': len(self.patterns),
            'total_matches': self.total_matches,
            'total_misses': self.total_misses,
            'match_rate': match_rate,
            'top_intents': dict(sorted(
                self.stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])  # 前10个最常匹配的意图
        }

    def add_pattern(
        self,
        intent_id: str,
        patterns: List[str],
        intent: str,
        confidence: float = 0.9,
        priority: int = 5,
        no_llm: bool = False,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        动态添加新模式

        Args:
            intent_id: 意图ID
            patterns: 正则表达式列表
            intent: 意图名称
            confidence: 置信度
            priority: 优先级（1-10，10最高）
            no_llm: 是否不需要LLM
            metadata: 额外元数据
        """
        # 编译正则表达式
        compiled_patterns = []
        for pattern_str in patterns:
            try:
                regex = re.compile(pattern_str, re.IGNORECASE)
                compiled_patterns.append(regex)
            except re.error as e:
                logger.error(f"正则表达式编译失败: {pattern_str}, 错误: {e}")
                continue

        # 添加到模式库
        self.patterns[intent_id] = {
            'patterns': patterns,
            'compiled_patterns': compiled_patterns,
            'intent': intent,
            'confidence': confidence,
            'priority': priority,
            'no_llm': no_llm,
            'metadata': metadata or {}
        }

        # 重建优先级索引
        self._build_priority_index()

        logger.info(f"[模式匹配器] 已添加新模式: {intent_id} ({intent})")


# ==================== 便捷函数 ====================

_matcher_instance: Optional[PatternMatcher] = None


def get_pattern_matcher() -> PatternMatcher:
    """获取或创建模式匹配器单例"""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = PatternMatcher()
    return _matcher_instance
