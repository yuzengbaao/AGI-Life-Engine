#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
确定性决策引擎 (Deterministic Decision Engine)
=====================================

核心理念：系统决策主导，LLM辅助表达

问题诊断：
- 当前架构：用户输入 → LLM生成(含幻觉) → 工具执行 → 混合输出
- 问题：LLM先生成"期望结果"，工具失败后幻觉仍保留

解决方案：反转控制流
- 新架构：用户输入 → 意图解析 → 规则决策 → 工具执行 → 事实锚定 → LLM表达

作者：AGI Self-Improvement Module
创建日期：2026-01-17
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class DecisionSource(Enum):
    """决策来源"""
    RULE_ENGINE = "rule_engine"       # 规则引擎（确定性）
    TOOL_RESULT = "tool_result"       # 工具执行结果
    STATE_MACHINE = "state_machine"   # 状态机
    THRESHOLD_CHECK = "threshold"     # 阈值检查
    LLM_INFERENCE = "llm_inference"   # LLM推理（最低优先级）


@dataclass
class VerifiedFact:
    """已验证的事实"""
    fact_id: str
    source: DecisionSource
    content: str
    confidence: float  # 1.0 for deterministic, <1.0 for LLM
    evidence: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def is_deterministic(self) -> bool:
        return self.source != DecisionSource.LLM_INFERENCE


@dataclass
class DecisionResult:
    """决策结果"""
    decision_id: str
    facts: List[VerifiedFact]
    conclusion: str
    deterministic_ratio: float  # 确定性事实占比
    llm_contribution: str  # LLM仅用于表达，不用于决策
    blocked_hallucinations: List[str]  # 被阻断的幻觉


class DeterministicDecisionEngine:
    """
    确定性决策引擎
    
    核心原则：
    1. 确定性决策优先：规则引擎、状态机、阈值检查先于LLM
    2. 事实锚定：每个断言必须绑定到已验证的工具结果
    3. 幻觉阻断：工具失败时，阻止依赖该结果的所有断言
    4. LLM降级：LLM仅用于自然语言表达，不参与核心决策
    """
    
    def __init__(self, tool_bridge=None, agi_system=None):
        self.tool_bridge = tool_bridge
        self.agi_system = agi_system
        self.verified_facts: Dict[str, VerifiedFact] = {}
        self.blocked_assertions: List[str] = []
        
        # 规则引擎配置
        self.rules = self._load_decision_rules()
        
        # 意图到工具的映射
        self.intent_tool_mapping = self._load_intent_mapping()
        
        # 全局阈值配置
        self.global_thresholds = self._load_global_thresholds()
        
        logger.info("[确定性决策引擎] 初始化完成")
        logger.info(f"[确定性决策引擎] 加载了 {len(self.rules)} 条决策规则（150条目标）")
        logger.info(f"[确定性决策引擎] 阈值类别: {len(self.global_thresholds)}")
    
    def _load_decision_rules(self) -> Dict[str, Any]:
        """
        加载决策规则
        
        规则结构：
        - triggers: 触发关键词列表
        - required_tools: 必须调用的工具
        - decision_logic: 决策逻辑类型
          - tool_result_only: 仅依赖工具结果
          - threshold_based: 基于阈值判断
          - tool_result_with_threshold: 工具结果+阈值组合
          - security_gated: 需要安全验证
        - thresholds: 阈值定义（如适用）
        - fallback: 工具失败时的回退策略
        """
        return {
            # ============= 系统运维规则 =============
            'query_system_status': {
                'triggers': ['系统状态', '健康检查', 'system status', 'health', '运行状态'],
                'required_tools': ['system_monitor', 'health_check'],
                'decision_logic': 'tool_result_only',
                'fallback': 'report_unavailable',
            },
            
            'process_management': {
                'triggers': ['进程管理', '启动服务', '停止服务', 'process', 'service'],
                'required_tools': ['system_monitor', 'process_controller'],
                'decision_logic': 'tool_result_only',
                'fallback': 'report_unavailable',
            },
            
            # ============= 文件操作规则 =============
            'file_operation': {
                'triggers': ['创建文件', '读取文件', '写入', 'create file', 'read file', '保存'],
                'required_tools': ['file_operation'],
                'decision_logic': 'tool_result_only',
                'fallback': 'report_failure',
            },
            
            'code_modification': {
                'triggers': ['修改代码', '编辑', '重构', 'modify code', 'refactor', '代码修复'],
                'required_tools': ['file_operation', 'syntax_validator'],
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {
                    'syntax_valid': True,
                    'test_pass_rate': 0.80,
                },
                'fallback': 'rollback',
            },
            
            # ============= 知识与记忆规则 =============
            'knowledge_query': {
                'triggers': ['知识查询', '知识库', 'knowledge', '查询知识'],
                'required_tools': ['knowledge_graph', 'world_model'],
                'decision_logic': 'tool_result_only',
                'fallback': 'admit_unknown',
            },
            
            'memory_operation': {
                'triggers': ['记忆', '回忆', 'memory', 'remember', '学习记录'],
                'required_tools': ['memory', 'learning_tracker'],
                'decision_logic': 'tool_result_only',
                'fallback': 'admit_unknown',
            },
            
            # ============= 智能评估规则 =============
            'intelligence_assessment': {
                'triggers': ['智能评估', '能力评价', 'L3', 'L4', '智能等级', 'AGI评估'],
                'required_tools': ['metacognition', 'world_model'],
                'decision_logic': 'threshold_based',
                'thresholds': {
                    'L3_min_coherence': 0.85,
                    'L3_min_evidence_chain': 4,
                    'L3_min_self_correction': 0.70,
                    'L4_min_novel_solution': 0.50,
                    'L4_min_meta_awareness': 0.80,
                },
                'fallback': 'conservative_estimate',
            },
            
            'self_evaluation': {
                'triggers': ['自我评估', '自省', 'self evaluate', 'introspect', '反思'],
                'required_tools': ['metacognition'],
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {
                    'introspection_depth': 3,
                    'bias_detection': True,
                },
                'fallback': 'admit_limitation',
            },
            
            # ============= 任务管理规则 =============
            'task_management': {
                'triggers': ['任务', '执行', 'task', 'execute', '待办', '计划'],
                'required_tools': ['task_queue', 'scheduler'],
                'decision_logic': 'tool_result_only',
                'fallback': 'queue_for_retry',
            },
            
            'multi_step_task': {
                'triggers': ['多步骤', '复杂任务', 'multi-step', '分解任务'],
                'required_tools': ['task_queue', 'planner', 'progress_tracker'],
                'decision_logic': 'threshold_based',
                'thresholds': {
                    'step_completion_rate': 0.90,
                    'error_tolerance': 0.10,
                },
                'fallback': 'partial_result',
            },
            
            # ============= 安全与权限规则 =============
            'security_check': {
                'triggers': ['安全', '权限', 'security', 'permission', '授权'],
                'required_tools': ['constitutional_ai', 'security_validator'],
                'decision_logic': 'security_gated',
                'thresholds': {
                    'security_score_min': 0.95,
                    'explicit_permission': True,
                },
                'fallback': 'deny_action',
            },
            
            'sensitive_operation': {
                'triggers': ['删除', '格式化', 'delete', 'format', '清空', '重置'],
                'required_tools': ['security_validator', 'backup_service'],
                'decision_logic': 'security_gated',
                'thresholds': {
                    'confirmation_required': True,
                    'backup_created': True,
                },
                'fallback': 'deny_action',
            },
            
            # ============= 创意探索规则 =============
            'creative_exploration': {
                'triggers': ['探索', '创意', 'explore', 'creative', '好奇心'],
                'required_tools': ['curiosity_explore', 'novelty_detector'],
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {
                    'novelty_score_min': 0.30,
                    'safety_check': True,
                },
                'fallback': 'bounded_exploration',
            },
            
            'hypothesis_testing': {
                'triggers': ['假设', '测试', 'hypothesis', 'test', '验证'],
                'required_tools': ['hypothesis_engine', 'evidence_collector'],
                'decision_logic': 'threshold_based',
                'thresholds': {
                    'evidence_support_min': 0.60,
                    'contradiction_max': 0.20,
                },
                'fallback': 'inconclusive',
            },
            
            # ============= 对话与理解规则 =============
            'intent_clarification': {
                'triggers': ['什么意思', '解释', 'explain', 'clarify', '不明白'],
                'required_tools': ['semantic_analyzer'],
                'decision_logic': 'tool_result_only',
                'fallback': 'ask_clarification',
            },
            
            'context_retrieval': {
                'triggers': ['上下文', '之前说', 'context', 'previous', '刚才'],
                'required_tools': ['conversation_memory', 'context_tracker'],
                'decision_logic': 'tool_result_only',
                'fallback': 'admit_context_loss',
            },

            # ============= 对话交互规则 (新增) =============
            'conversation_interaction': {
                'triggers': ['你好', '你是谁', '能做什么', '帮助', '问题', '疑问', '回复', '继续', '什么意思', '怎么', '为什么', '如何', '是否', '有没有', '可以吗', '什么疑问'],
                'required_tools': [],  # 空工具列表，直接处理对话
                'decision_logic': 'tool_result_only',
                'fallback': 'general_conversation',
            },

            'question_response': {
                'triggers': ['?', '？', '吗', '呢', '请问', '想问', '询问', '回答'],
                'required_tools': [],  # 空工具列表，直接处理问答
                'decision_logic': 'tool_result_only',
                'fallback': 'clarify_request',
            },

            'general_chat': {
                'triggers': ['好的', '嗯', '是的', '对', '谢谢', '感谢', '明白', '知道了', '了解', '清楚'],
                'required_tools': ['unified_memory'],  # 使用已注册的记忆工具
                'decision_logic': 'tool_result_only',
                'fallback': 'acknowledge',
            },

            'help_request': {
                'triggers': ['帮助', 'help', '协助', '支持', '怎么用', '如何使用', '教程'],
                'required_tools': ['knowledge_graph'],  # 简化工具列表
                'decision_logic': 'tool_result_only',
                'fallback': 'provide_basic_help',
            },

            'unknown_question': {
                'triggers': ['你有什么', '你的疑问', '需要我', '想让你'],
                'required_tools': ['metacognition'],  # 只使用已注册的元认知工具
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {
                    'confidence_min': 0.3,
                    'clarify_if_low': True,
                },
                'fallback': 'ask_clarification',
            },

            # ============= 🆕 [P0优化] 文件操作细化规则 =============
            'file_read_quick': {
                'triggers': ['读文件', '读取', '查看', '打开', 'read', 'view', 'open', 'cat', '显示内容'],
                'required_tools': ['file_operation'],
                'confidence': 1.0,
                'no_llm': True,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_file_not_found',
            },

            'file_write_quick': {
                'triggers': ['写文件', '写入', '保存', 'save', 'write', '创建文件', '新建文件'],
                'required_tools': ['file_operation'],
                'confidence': 1.0,
                'no_llm': True,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_write_failure',
            },

            'file_delete': {
                'triggers': ['删除文件', 'remove', 'delete', 'rm', '清除文件'],
                'required_tools': ['security_validator', 'file_operation'],
                'confidence': 0.95,
                'decision_logic': 'security_gated',
                'thresholds': {'confirmation_required': True},
                'fallback': 'deny_action',
            },

            'file_copy': {
                'triggers': ['复制文件', '拷贝', 'copy', 'cp', '文件复制'],
                'required_tools': ['file_operation'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_copy_failure',
            },

            'file_move': {
                'triggers': ['移动文件', '移动', 'move', 'mv', '重命名', 'rename'],
                'required_tools': ['file_operation'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_move_failure',
            },

            'file_search': {
                'triggers': ['搜索文件', '查找文件', 'find', 'search', '文件搜索', 'locate'],
                'required_tools': ['file_operation'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_results',
            },

            'file_list': {
                'triggers': ['列出文件', '文件列表', 'list', 'ls', 'dir', '显示目录', '查看目录'],
                'required_tools': ['file_operation'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_directory_error',
            },

            'file_info': {
                'triggers': ['文件信息', '文件属性', 'file info', 'stat', '文件详情', '文件大小'],
                'required_tools': ['file_operation'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_file_not_found',
            },

            'directory_create': {
                'triggers': ['创建目录', '新建目录', 'mkdir', '创建文件夹', '新建文件夹'],
                'required_tools': ['file_operation'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_create_failure',
            },

            'directory_delete': {
                'triggers': ['删除目录', '删除文件夹', 'rmdir', '清空目录'],
                'required_tools': ['security_validator', 'file_operation'],
                'confidence': 0.95,
                'decision_logic': 'security_gated',
                'thresholds': {'confirmation_required': True},
                'fallback': 'deny_action',
            },

            # ============= 🆕 [P0优化] 代码操作细化规则 =============
            'code_read': {
                'triggers': ['读取代码', '查看代码', 'read code', 'show code', '显示代码'],
                'required_tools': ['file_operation'],
                'confidence': 1.0,
                'no_llm': True,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_file_not_found',
            },

            'code_analyze': {
                'triggers': ['分析代码', '代码分析', 'analyze code', 'code review', '代码审查'],
                'required_tools': ['file_operation', 'syntax_validator'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'syntax_valid': True},
                'fallback': 'report_analysis_failed',
            },

            'code_debug': {
                'triggers': ['调试代码', 'debug', '调试', '排错', '查找bug'],
                'required_tools': ['syntax_validator', 'error_analyzer'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'error_found': True},
                'fallback': 'suggest_debug_steps',
            },

            'code_refactor': {
                'triggers': ['重构代码', 'refactor', '代码优化', '优化代码'],
                'required_tools': ['file_operation', 'syntax_validator', 'backup_service'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'test_pass_rate': 0.80},
                'fallback': 'rollback',
            },

            'code_test': {
                'triggers': ['测试代码', 'run test', '执行测试', '运行测试', 'test'],
                'required_tools': ['test_runner'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_test_results',
            },

            'code_format': {
                'triggers': ['格式化代码', 'format', '代码格式化', '美化代码'],
                'required_tools': ['file_operation', 'code_formatter'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_format_failed',
            },

            'code_document': {
                'triggers': ['生成文档', 'generate docs', '代码文档', 'docstring'],
                'required_tools': ['documentation_generator'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'manual_documentation',
            },

            'code_search': {
                'triggers': ['搜索代码', 'code search', '查找代码', 'grep'],
                'required_tools': ['code_search_engine'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_matches',
            },

            'code_dependency': {
                'triggers': ['依赖检查', 'dependency', '依赖关系', '导入检查'],
                'required_tools': ['dependency_analyzer'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'list_dependencies',
            },

            'code_coverage': {
                'triggers': ['代码覆盖率', 'coverage', '测试覆盖率', '覆盖分析'],
                'required_tools': ['coverage_analyzer'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_coverage',
            },

            'code_profile': {
                'triggers': ['性能分析', 'profile', '性能测试', 'profiling'],
                'required_tools': ['profiler'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_profile',
            },

            'code_lint': {
                'triggers': ['代码检查', 'lint', '静态检查', '代码规范'],
                'required_tools': ['linter'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_lint_issues',
            },

            'code_build': {
                'triggers': ['构建', 'build', '编译', 'compile'],
                'required_tools': ['build_tool'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_build_errors',
            },

            'code_deploy': {
                'triggers': ['部署', 'deploy', '发布', 'release'],
                'required_tools': ['deployment_tool', 'security_validator'],
                'confidence': 0.85,
                'decision_logic': 'security_gated',
                'thresholds': {'security_check': True},
                'fallback': 'deny_deployment',
            },

            'code_version': {
                'triggers': ['版本控制', 'git', 'version', 'commit', '版本管理'],
                'required_tools': ['version_control'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_vcs_error',
            },

            # ============= 🆕 [P0优化] 系统操作细化规则 =============
            'system_info': {
                'triggers': ['系统信息', 'system info', '系统详情', 'os info'],
                'required_tools': ['system_monitor'],
                'confidence': 1.0,
                'no_llm': True,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_unavailable',
            },

            'system_resources': {
                'triggers': ['资源使用', 'cpu', 'memory', '内存', '磁盘', 'disk', '资源监控'],
                'required_tools': ['system_monitor'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_unavailable',
            },

            'system_uptime': {
                'triggers': ['运行时间', 'uptime', '启动时间', '运行时长'],
                'required_tools': ['system_monitor'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_unavailable',
            },

            'process_list': {
                'triggers': ['进程列表', 'process list', '运行进程', '查看进程', 'ps'],
                'required_tools': ['system_monitor'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_unavailable',
            },

            'process_kill': {
                'triggers': ['结束进程', 'kill', '终止进程', '停止进程'],
                'required_tools': ['security_validator', 'process_controller'],
                'confidence': 0.95,
                'decision_logic': 'security_gated',
                'thresholds': {'confirmation_required': True},
                'fallback': 'deny_action',
            },

            'service_start': {
                'triggers': ['启动服务', 'start service', '开启服务', '运行服务'],
                'required_tools': ['process_controller'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_service_error',
            },

            'service_stop': {
                'triggers': ['停止服务', 'stop service', '关闭服务'],
                'required_tools': ['process_controller'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_service_error',
            },

            'service_restart': {
                'triggers': ['重启服务', 'restart', 'restart service'],
                'required_tools': ['process_controller'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_service_error',
            },

            'service_status': {
                'triggers': ['服务状态', 'service status', '查看服务'],
                'required_tools': ['system_monitor'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_unavailable',
            },

            'system_logs': {
                'triggers': ['系统日志', 'logs', '查看日志', 'log', '日志文件'],
                'required_tools': ['log_reader'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_log_error',
            },

            'system_config': {
                'triggers': ['系统配置', 'config', '配置', 'settings', '设置'],
                'required_tools': ['config_manager'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_config_error',
            },

            'environment_vars': {
                'triggers': ['环境变量', 'environment', 'env', '环境'],
                'required_tools': ['system_monitor'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_unavailable',
            },

            # ============= 🆕 [P0优化] 网络操作规则 =============
            'network_ping': {
                'triggers': ['ping', '网络测试', '连通性', '连接测试'],
                'required_tools': ['network_tool'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_network_error',
            },

            'network_info': {
                'triggers': ['网络信息', 'network info', 'ip地址', '网卡信息'],
                'required_tools': ['network_tool'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_network_error',
            },

            'network_speed': {
                'triggers': ['网速', 'speedtest', '网络速度', '带宽'],
                'required_tools': ['network_tool'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_network_error',
            },

            'http_request': {
                'triggers': ['http请求', 'request', 'curl', 'wget', '下载'],
                'required_tools': ['http_client'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_http_error',
            },

            'api_call': {
                'triggers': ['api调用', 'api call', '调用api', '接口调用'],
                'required_tools': ['http_client'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_api_error',
            },

            'url_test': {
                'triggers': ['测试链接', 'test url', '检查url', '验证链接'],
                'required_tools': ['http_client'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_url_invalid',
            },

            'port_scan': {
                'triggers': ['端口扫描', 'port scan', '检查端口', '开放端口'],
                'required_tools': ['network_tool', 'security_validator'],
                'confidence': 0.85,
                'decision_logic': 'security_gated',
                'thresholds': {'authorization_required': True},
                'fallback': 'deny_action',
            },

            'dns_query': {
                'triggers': ['dns查询', 'dns', '域名解析', '解析域名'],
                'required_tools': ['network_tool'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_dns_error',
            },

            # ============= 🆕 [P0优化] 数据分析规则 =============
            'data_load': {
                'triggers': ['加载数据', 'load data', '读取数据', '导入数据'],
                'required_tools': ['data_processor'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_load_error',
            },

            'data_save': {
                'triggers': ['保存数据', 'save data', '导出数据', '存储数据'],
                'required_tools': ['data_processor'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_save_error',
            },

            'data_transform': {
                'triggers': ['数据转换', 'transform', '转换数据', '数据清洗'],
                'required_tools': ['data_processor'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_transform_error',
            },

            'data_filter': {
                'triggers': ['数据过滤', 'filter', '筛选数据', '过滤数据'],
                'required_tools': ['data_processor'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_filter_error',
            },

            'data_aggregate': {
                'triggers': ['数据聚合', 'aggregate', '汇总数据', '统计'],
                'required_tools': ['data_processor'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_aggregate_error',
            },

            'data_visualize': {
                'triggers': ['数据可视化', 'visualize', '图表', '绘图', 'plot'],
                'required_tools': ['visualization_tool'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_viz_error',
            },

            'data_analyze': {
                'triggers': ['数据分析', 'analyze', '分析数据', 'data analysis'],
                'required_tools': ['data_processor', 'statistical_analyzer'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'sample_size_min': 10},
                'fallback': 'insufficient_data',
            },

            'data_statistics': {
                'triggers': ['统计信息', 'statistics', '统计数据', '描述统计'],
                'required_tools': ['statistical_analyzer'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_stats_error',
            },

            'data_merge': {
                'triggers': ['数据合并', 'merge', '合并数据', 'join'],
                'required_tools': ['data_processor'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_merge_error',
            },

            'data_validate': {
                'triggers': ['数据验证', 'validate', '验证数据', '检查数据'],
                'required_tools': ['data_validator'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_validation_errors',
            },

            # ============= 🆕 [P0优化] 调试测试规则 =============
            'test_run': {
                'triggers': ['运行测试', 'run test', 'test', '测试'],
                'required_tools': ['test_runner'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_test_results',
            },

            'test_unit': {
                'triggers': ['单元测试', 'unit test', '测试单元'],
                'required_tools': ['test_runner'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_test_results',
            },

            'test_integration': {
                'triggers': ['集成测试', 'integration test', '集成测试'],
                'required_tools': ['test_runner'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_test_results',
            },

            'test_e2e': {
                'triggers': ['端到端测试', 'e2e test', '端到端', 'e2e'],
                'required_tools': ['test_runner'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_test_results',
            },

            'debug_start': {
                'triggers': ['开始调试', 'start debug', '启动调试'],
                'required_tools': ['debugger'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_debug_error',
            },

            'debug_step': {
                'triggers': ['单步调试', 'step', '下一步', 'step over'],
                'required_tools': ['debugger'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_debug_error',
            },

            'debug_breakpoint': {
                'triggers': ['断点', 'breakpoint', '设置断点'],
                'required_tools': ['debugger'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_debug_error',
            },

            'debug_inspect': {
                'triggers': ['检查变量', 'inspect', '查看变量', '变量值'],
                'required_tools': ['debugger'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_debug_error',
            },

            'error_trace': {
                'triggers': ['错误追踪', 'traceback', '堆栈跟踪', '错误堆栈'],
                'required_tools': ['error_analyzer'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_error',
            },

            'error_analyze': {
                'triggers': ['错误分析', 'analyze error', '分析错误'],
                'required_tools': ['error_analyzer'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'suggest_solutions',
            },

            'performance_monitor': {
                'triggers': ['性能监控', 'monitor', '监控性能', 'perf monitor'],
                'required_tools': ['profiler'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_monitor_error',
            },

            'memory_profile': {
                'triggers': ['内存分析', 'memory profile', '内存使用', '内存泄漏'],
                'required_tools': ['profiler'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_profile_error',
            },

            # ============= 🆕 [P0优化] 文档处理规则 =============
            'document_create': {
                'triggers': ['创建文档', 'create doc', '新建文档', '写文档'],
                'required_tools': ['document_generator'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'manual_creation',
            },

            'document_read': {
                'triggers': ['读取文档', 'read doc', '查看文档', '打开文档'],
                'required_tools': ['document_reader'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_read_error',
            },

            'document_edit': {
                'triggers': ['编辑文档', 'edit doc', '修改文档'],
                'required_tools': ['document_editor', 'backup_service'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_edit_error',
            },

            'document_convert': {
                'triggers': ['文档转换', 'convert', '格式转换', '转换格式'],
                'required_tools': ['document_converter'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_convert_error',
            },

            'document_search': {
                'triggers': ['搜索文档', 'search doc', '文档搜索'],
                'required_tools': ['document_search'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_results',
            },

            'document_summarize': {
                'triggers': ['文档摘要', 'summarize', '总结文档', '生成摘要'],
                'required_tools': ['summarization_tool'],
                'confidence': 0.80,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'doc_length_min': 100},
                'fallback': 'manual_summary',
            },

            'document_export': {
                'triggers': ['导出文档', 'export', '文档导出'],
                'required_tools': ['document_converter'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_export_error',
            },

            'document_print': {
                'triggers': ['打印文档', 'print', '打印'],
                'required_tools': ['print_service'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_print_error',
            },

            # ============= 🆕 [P0优化] 学习研究规则 =============
            'learn_new': {
                'triggers': ['学习', 'learn', '学习新知识', '研究'],
                'required_tools': ['curiosity_explore', 'knowledge_graph'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'novelty_min': 0.3},
                'fallback': 'guided_learning',
            },

            'research_topic': {
                'triggers': ['研究主题', 'research', '调研', '课题研究'],
                'required_tools': ['knowledge_graph', 'web_search'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'source_count_min': 3},
                'fallback': 'limited_research',
            },

            'explore_domain': {
                'triggers': ['探索领域', 'explore', '领域探索', '新领域'],
                'required_tools': ['curiosity_explore', 'world_model'],
                'confidence': 0.80,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'safety_check': True},
                'fallback': 'bounded_exploration',
            },

            'knowledge_acquire': {
                'triggers': ['获取知识', 'acquire', '知识获取'],
                'required_tools': ['knowledge_graph', 'learning_tracker'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'record_attempt',
            },

            'skill_practice': {
                'triggers': ['练习技能', 'practice', '技能练习', '训练'],
                'required_tools': ['learning_tracker', 'practice_tool'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'suggest_practice',
            },

            'concept_understand': {
                'triggers': ['理解概念', 'understand', '概念理解'],
                'required_tools': ['knowledge_graph', 'semantic_analyzer'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'definition_available': True},
                'fallback': 'request_clarification',
            },

            'tutorial_follow': {
                'triggers': ['教程', 'tutorial', '跟随教程', '学习教程'],
                'required_tools': ['tutorial_engine', 'task_queue'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'suggest_tutorial',
            },

            'experiment_run': {
                'triggers': ['实验', 'experiment', '运行实验', '做实验'],
                'required_tools': ['hypothesis_engine', 'experiment_tracker'],
                'confidence': 0.80,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'safety_check': True},
                'fallback': 'deny_experiment',
            },

            'simulate': {
                'triggers': ['模拟', 'simulate', '仿真', '运行模拟'],
                'required_tools': ['simulation_engine'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_sim_error',
            },

            'model_train': {
                'triggers': ['训练模型', 'train model', '模型训练', 'ml训练'],
                'required_tools': ['ml_framework', 'data_processor'],
                'confidence': 0.80,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'data_size_min': 100},
                'fallback': 'insufficient_data',
            },

            # ============= 🆕 [P0优化] 备份恢复规则 =============
            'backup_create': {
                'triggers': ['创建备份', 'create backup', 'backup', '备份'],
                'required_tools': ['backup_service'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_backup_error',
            },

            'backup_restore': {
                'triggers': ['恢复备份', 'restore', 'restore backup', '恢复'],
                'required_tools': ['backup_service', 'security_validator'],
                'confidence': 0.95,
                'decision_logic': 'security_gated',
                'thresholds': {'confirmation_required': True},
                'fallback': 'deny_restore',
            },

            'backup_list': {
                'triggers': ['列出备份', 'list backup', '备份列表', '查看备份'],
                'required_tools': ['backup_service'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_backups',
            },

            'backup_delete': {
                'triggers': ['删除备份', 'delete backup', '清除备份'],
                'required_tools': ['backup_service', 'security_validator'],
                'confidence': 0.95,
                'decision_logic': 'security_gated',
                'thresholds': {'confirmation_required': True},
                'fallback': 'deny_action',
            },

            'backup_schedule': {
                'triggers': ['计划备份', 'schedule backup', '自动备份'],
                'required_tools': ['backup_service', 'scheduler'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_schedule_error',
            },

            'snapshot_create': {
                'triggers': ['创建快照', 'create snapshot', 'snapshot', '快照'],
                'required_tools': ['snapshot_service'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_snapshot_error',
            },

            'snapshot_restore': {
                'triggers': ['恢复快照', 'restore snapshot', '快照恢复'],
                'required_tools': ['snapshot_service', 'security_validator'],
                'confidence': 0.95,
                'decision_logic': 'security_gated',
                'thresholds': {'confirmation_required': True},
                'fallback': 'deny_restore',
            },

            'data_sync': {
                'triggers': ['数据同步', 'sync', '同步数据', '同步'],
                'required_tools': ['sync_service'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_sync_error',
            },

            # ============= 🆕 [P0优化] 配置管理规则 =============
            'config_read': {
                'triggers': ['读取配置', 'read config', '查看配置', '配置信息'],
                'required_tools': ['config_manager'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_config_error',
            },

            'config_set': {
                'triggers': ['设置配置', 'set config', '修改配置', '更新配置'],
                'required_tools': ['config_manager', 'backup_service'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'valid_value': True},
                'fallback': 'reject_invalid_value',
            },

            'config_validate': {
                'triggers': ['验证配置', 'validate config', '配置验证'],
                'required_tools': ['config_validator'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_validation_errors',
            },

            'config_reload': {
                'triggers': ['重载配置', 'reload config', '重新加载配置'],
                'required_tools': ['config_manager'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_reload_error',
            },

            'config_reset': {
                'triggers': ['重置配置', 'reset config', '恢复默认配置'],
                'required_tools': ['config_manager', 'backup_service'],
                'confidence': 0.90,
                'decision_logic': 'security_gated',
                'thresholds': {'confirmation_required': True},
                'fallback': 'deny_reset',
            },

            'config_export': {
                'triggers': ['导出配置', 'export config', '配置导出'],
                'required_tools': ['config_manager'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_export_error',
            },

            'config_import': {
                'triggers': ['导入配置', 'import config', '配置导入'],
                'required_tools': ['config_manager', 'security_validator'],
                'confidence': 0.90,
                'decision_logic': 'security_gated',
                'thresholds': {'validation_required': True},
                'fallback': 'reject_invalid_config',
            },

            'config_diff': {
                'triggers': ['配置对比', 'config diff', '比较配置'],
                'required_tools': ['config_manager', 'diff_tool'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_diff_error',
            },

            'config_merge': {
                'triggers': ['合并配置', 'merge config', '配置合并'],
                'required_tools': ['config_manager', 'merge_tool'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'no_conflicts': True},
                'fallback': 'report_conflicts',
            },

            'environment_setup': {
                'triggers': ['环境配置', 'setup', '配置环境', '环境设置'],
                'required_tools': ['config_manager', 'dependency_manager'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'suggest_setup_steps',
            },

            # ============= 🆕 [P0优化] 日志分析规则 =============
            'log_read': {
                'triggers': ['读取日志', 'read log', '查看日志'],
                'required_tools': ['log_reader'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_log_error',
            },

            'log_search': {
                'triggers': ['搜索日志', 'search log', '日志搜索', '查找日志'],
                'required_tools': ['log_reader', 'search_tool'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_matches',
            },

            'log_filter': {
                'triggers': ['过滤日志', 'filter log', '日志过滤', '筛选日志'],
                'required_tools': ['log_reader', 'filter_tool'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_filter_error',
            },

            'log_analyze': {
                'triggers': ['分析日志', 'analyze log', '日志分析'],
                'required_tools': ['log_analyzer'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_analysis_failed',
            },

            'log_export': {
                'triggers': ['导出日志', 'export log', '日志导出'],
                'required_tools': ['log_reader', 'export_tool'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_export_error',
            },

            'log_rotate': {
                'triggers': ['轮转日志', 'rotate log', '日志轮转', '切割日志'],
                'required_tools': ['log_manager'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_rotate_error',
            },

            'log_compress': {
                'triggers': ['压缩日志', 'compress log', '日志压缩'],
                'required_tools': ['log_manager', 'compression_tool'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_compress_error',
            },

            'log_monitor': {
                'triggers': ['监控日志', 'monitor log', '实时日志', 'tail'],
                'required_tools': ['log_monitor'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_monitor_error',
            },

            # ============= 🆕 [P0优化] 性能优化规则 =============
            'performance_profile': {
                'triggers': ['性能分析', 'profile', '性能剖析'],
                'required_tools': ['profiler'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_profile_error',
            },

            'performance_tune': {
                'triggers': ['性能调优', 'tune', '优化性能', '性能优化'],
                'required_tools': ['profiler', 'optimization_engine'],
                'confidence': 0.80,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'bottleneck_identified': True},
                'fallback': 'suggest_optimizations',
            },

            'memory_optimize': {
                'triggers': ['内存优化', 'memory optimize', '优化内存'],
                'required_tools': ['profiler', 'memory_optimizer'],
                'confidence': 0.80,
                'decision_logic': 'tool_result_only',
                'fallback': 'suggest_memory_optimizations',
            },

            'cache_optimize': {
                'triggers': ['缓存优化', 'cache optimize', '优化缓存'],
                'required_tools': ['cache_manager', 'profiler'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'suggest_cache_strategies',
            },

            'concurrency_improve': {
                'triggers': ['并发优化', 'concurrency', '提高并发', '并发改进'],
                'required_tools': ['profiler', 'concurrency_tool'],
                'confidence': 0.75,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'thread_safe': True},
                'fallback': 'warn_concurrency_risks',
            },

            'query_optimize': {
                'triggers': ['查询优化', 'query optimize', '优化查询'],
                'required_tools': ['query_analyzer'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'suggest_query_optimizations',
            },

            'index_optimize': {
                'triggers': ['索引优化', 'index optimize', '优化索引'],
                'required_tools': ['index_manager'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'suggest_index_changes',
            },

            'benchmark_run': {
                'triggers': ['基准测试', 'benchmark', '运行基准', '性能测试'],
                'required_tools': ['benchmark_tool'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_benchmark_results',
            },

            # ============= 🆕 [P0优化] 监控告警规则 =============
            'monitor_setup': {
                'triggers': ['设置监控', 'setup monitor', '配置监控', 'monitor setup'],
                'required_tools': ['monitor_config'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_setup_error',
            },

            'alert_create': {
                'triggers': ['创建告警', 'create alert', '新建告警', 'alert'],
                'required_tools': ['alert_manager'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_alert_error',
            },

            'alert_list': {
                'triggers': ['列出告警', 'list alert', '告警列表', '查看告警'],
                'required_tools': ['alert_manager'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_alerts',
            },

            'alert_acknowledge': {
                'triggers': ['确认告警', 'acknowledge alert', '确认', 'ack'],
                'required_tools': ['alert_manager'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_ack_error',
            },

            'alert_resolve': {
                'triggers': ['解决告警', 'resolve alert', '告警解决'],
                'required_tools': ['alert_manager'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_resolve_error',
            },

            'metric_collect': {
                'triggers': ['采集指标', 'collect metric', '指标采集', '收集指标'],
                'required_tools': ['metric_collector'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_collect_error',
            },

            'metric_query': {
                'triggers': ['查询指标', 'query metric', '指标查询'],
                'required_tools': ['metric_collector', 'query_tool'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_metrics',
            },

            'dashboard_view': {
                'triggers': ['查看仪表板', 'dashboard', '仪表板', '监控面板'],
                'required_tools': ['dashboard_service'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_dashboard_error',
            },

            'report_generate': {
                'triggers': ['生成报告', 'generate report', 'report', '报告'],
                'required_tools': ['report_generator'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_generation_error',
            },

            'health_check': {
                'triggers': ['健康检查', 'health check', '健康', '检查健康'],
                'required_tools': ['health_checker'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_unhealthy',
            },

            # ============= 🆕 [P0优化] 安全审计规则 =============
            'audit_log': {
                'triggers': ['审计日志', 'audit log', '安全审计', 'audit'],
                'required_tools': ['audit_logger'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_audit_error',
            },

            'security_scan': {
                'triggers': ['安全扫描', 'security scan', '扫描安全'],
                'required_tools': ['security_scanner'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_scan_results',
            },

            'vulnerability_check': {
                'triggers': ['漏洞检查', 'vulnerability', '检查漏洞', '漏洞扫描'],
                'required_tools': ['vulnerability_scanner'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_vulnerabilities',
            },

            'permission_check': {
                'triggers': ['权限检查', 'permission check', '检查权限'],
                'required_tools': ['permission_manager'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_permissions',
            },

            'access_log': {
                'triggers': ['访问日志', 'access log', '访问记录'],
                'required_tools': ['access_logger'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_log_error',
            },

            'compliance_check': {
                'triggers': ['合规检查', 'compliance', '检查合规'],
                'required_tools': ['compliance_checker'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_compliance_status',
            },

            # ============= 🆕 [P0优化] 用户交互规则 =============
            'user_input': {
                'triggers': ['用户输入', 'user input', '输入'],
                'required_tools': [],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'prompt_user',
            },

            'user_confirm': {
                'triggers': ['确认', 'confirm', '是否', 'yes', 'no', '是', '否'],
                'required_tools': [],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'request_confirmation',
            },

            'user_cancel': {
                'triggers': ['取消', 'cancel', '中止', '停止'],
                'required_tools': [],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'confirm_cancellation',
            },

            'user_retry': {
                'triggers': ['重试', 'retry', '再试一次'],
                'required_tools': ['task_queue'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_retry_failed',
            },

            'feedback_give': {
                'triggers': ['反馈', 'feedback', '提供反馈'],
                'required_tools': ['feedback_collector'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'acknowledge_feedback',
            },

            'preference_set': {
                'triggers': ['设置偏好', 'set preference', '偏好设置', '设置喜好'],
                'required_tools': ['preference_manager'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_set_error',
            },

            'notification_send': {
                'triggers': ['发送通知', 'send notification', '通知', 'notify'],
                'required_tools': ['notification_service'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_send_error',
            },

            'notification_list': {
                'triggers': ['列出通知', 'list notification', '通知列表', '查看通知'],
                'required_tools': ['notification_service'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_notifications',
            },

            'message_send': {
                'triggers': ['发送消息', 'send message', '发消息', '消息'],
                'required_tools': ['messaging_service'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_send_error',
            },

            'conversation_start': {
                'triggers': ['开始对话', 'start conversation', '新对话', '开启对话'],
                'required_tools': ['conversation_memory'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'initialize_conversation',
            },

            'conversation_end': {
                'triggers': ['结束对话', 'end conversation', '关闭对话', 'bye'],
                'required_tools': ['conversation_memory'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'save_conversation',
            },

            'history_view': {
                'triggers': ['查看历史', 'view history', '历史记录', '对话历史'],
                'required_tools': ['conversation_memory'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_history',
            },

            # ============= 🆕 [P0优化] 任务调度规则 =============
            'schedule_create': {
                'triggers': ['创建计划', 'create schedule', '新建计划', '计划任务'],
                'required_tools': ['scheduler'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_schedule_error',
            },

            'schedule_list': {
                'triggers': ['列出计划', 'list schedule', '计划列表', '查看计划'],
                'required_tools': ['scheduler'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_no_schedules',
            },

            'schedule_modify': {
                'triggers': ['修改计划', 'modify schedule', '更新计划'],
                'required_tools': ['scheduler'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_modify_error',
            },

            'schedule_delete': {
                'triggers': ['删除计划', 'delete schedule', '取消计划'],
                'required_tools': ['scheduler', 'security_validator'],
                'confidence': 0.95,
                'decision_logic': 'security_gated',
                'thresholds': {'confirmation_required': True},
                'fallback': 'deny_delete',
            },

            'schedule_run': {
                'triggers': ['运行计划', 'run schedule', '执行计划'],
                'required_tools': ['scheduler', 'task_runner'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_run_error',
            },

            'schedule_pause': {
                'triggers': ['暂停计划', 'pause schedule', '暂停任务'],
                'required_tools': ['scheduler'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_pause_error',
            },

            'schedule_resume': {
                'triggers': ['恢复计划', 'resume schedule', '继续任务'],
                'required_tools': ['scheduler'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_resume_error',
            },

            'task_queue': {
                'triggers': ['任务队列', 'queue', '加入队列', '排队'],
                'required_tools': ['task_queue'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_queue_error',
            },

            'task_priority': {
                'triggers': ['任务优先级', 'priority', '设置优先级', '优先级'],
                'required_tools': ['task_queue'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_priority_error',
            },

            'task_status': {
                'triggers': ['任务状态', 'task status', '查看状态', '进度'],
                'required_tools': ['task_tracker'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_status_error',
            },

            # ============= 🆕 [P0优化] 数据库操作规则 =============
            'database_connect': {
                'triggers': ['连接数据库', 'connect db', '数据库连接'],
                'required_tools': ['database_client'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_connect_error',
            },

            'database_query': {
                'triggers': ['查询数据库', 'db query', '数据库查询', 'sql查询'],
                'required_tools': ['database_client'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'query_valid': True},
                'fallback': 'report_query_error',
            },

            'database_execute': {
                'triggers': ['执行sql', 'execute sql', '运行sql'],
                'required_tools': ['database_client', 'security_validator'],
                'confidence': 0.90,
                'decision_logic': 'security_gated',
                'thresholds': {'validation_required': True},
                'fallback': 'deny_execution',
            },

            'database_backup': {
                'triggers': ['备份数据库', 'backup db', '数据库备份'],
                'required_tools': ['database_client', 'backup_service'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_backup_error',
            },

            'database_restore': {
                'triggers': ['恢复数据库', 'restore db', '数据库恢复'],
                'required_tools': ['database_client', 'backup_service', 'security_validator'],
                'confidence': 0.90,
                'decision_logic': 'security_gated',
                'thresholds': {'confirmation_required': True},
                'fallback': 'deny_restore',
            },

            'database_migrate': {
                'triggers': ['数据库迁移', 'migrate', '数据迁移'],
                'required_tools': ['database_client', 'migration_tool'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'backup_created': True},
                'fallback': 'deny_migration',
            },

            'database_schema': {
                'triggers': ['数据库模式', 'schema', '表结构', '查看表'],
                'required_tools': ['database_client'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_schema_error',
            },

            'transaction_begin': {
                'triggers': ['开始事务', 'begin transaction', '开启事务'],
                'required_tools': ['database_client'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_transaction_error',
            },

            # ============= 🆕 [P0优化] 其他高频操作 =============
            'calculate': {
                'triggers': ['计算', 'calculate', 'calc', '运算', '数学计算'],
                'required_tools': ['calculator'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_calc_error',
            },

            'convert_unit': {
                'triggers': ['单位转换', 'convert', '转换', '换算'],
                'required_tools': ['unit_converter'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_convert_error',
            },

            'timestamp': {
                'triggers': ['时间戳', 'timestamp', '当前时间', '现在时间'],
                'required_tools': ['time_service'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_time_error',
            },

            'date_format': {
                'triggers': ['日期格式', 'format date', '格式化日期'],
                'required_tools': ['time_service'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_format_error',
            },

            'hash_generate': {
                'triggers': ['生成哈希', 'hash', '哈希', 'md5', 'sha256'],
                'required_tools': ['crypto_tool'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_hash_error',
            },

            'encode_decode': {
                'triggers': ['编码', 'decode', '解码', 'encode', 'base64'],
                'required_tools': ['encoder'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_encode_error',
            },

            'compress': {
                'triggers': ['压缩', 'compress', 'zip', 'gzip'],
                'required_tools': ['compression_tool'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_compress_error',
            },

            'decompress': {
                'triggers': ['解压缩', 'decompress', 'unzip', '解压'],
                'required_tools': ['compression_tool'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_decompress_error',
            },

            'regex_test': {
                'triggers': ['正则测试', 'regex', '正则表达式', '测试正则'],
                'required_tools': ['regex_tool'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_regex_error',
            },

            'json_format': {
                'triggers': ['json格式化', 'format json', '格式化json'],
                'required_tools': ['json_tool'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_format_error',
            },

            'xml_parse': {
                'triggers': ['解析xml', 'parse xml', 'xml解析'],
                'required_tools': ['xml_parser'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_parse_error',
            },

            'color_convert': {
                'triggers': ['颜色转换', 'color', '颜色', 'rgb', 'hex'],
                'required_tools': ['color_tool'],
                'confidence': 1.0,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_convert_error',
            },

            'image_resize': {
                'triggers': ['调整大小', 'resize', '缩放', '调整尺寸'],
                'required_tools': ['image_tool'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_resize_error',
            },

            'image_crop': {
                'triggers': ['裁剪', 'crop', '图片裁剪'],
                'required_tools': ['image_tool'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_crop_error',
            },

            'text_compare': {
                'triggers': ['文本对比', 'diff', '比较文本', '文本比较'],
                'required_tools': ['diff_tool'],
                'confidence': 0.95,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_diff_error',
            },

            'file_split': {
                'triggers': ['分割文件', 'split', '拆分文件', '文件分割'],
                'required_tools': ['file_operation'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_split_error',
            },

            'file_join': {
                'triggers': ['合并文件', 'join', '文件合并', '拼接文件'],
                'required_tools': ['file_operation'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_join_error',
            },

            'batch_process': {
                'triggers': ['批处理', 'batch', '批量处理', '批量'],
                'required_tools': ['batch_processor'],
                'confidence': 0.85,
                'decision_logic': 'tool_result_with_threshold',
                'thresholds': {'item_count_min': 2},
                'fallback': 'process_individually',
            },

            'template_apply': {
                'triggers': ['应用模板', 'template', '使用模板', '模板'],
                'required_tools': ['template_engine'],
                'confidence': 0.90,
                'decision_logic': 'tool_result_only',
                'fallback': 'report_template_error',
            },
        }
    
    def _load_intent_mapping(self) -> Dict[str, List[str]]:
        """
        意图到工具的映射
        
        用于在无法匹配具体规则时，根据意图类型选择工具
        """
        return {
            # 查询类意图
            'query': ['world_model', 'knowledge_graph', 'memory', 'semantic_analyzer'],
            # 执行类意图
            'execute': ['file_operation', 'openhands', 'task_queue', 'process_controller'],
            # 评估类意图
            'evaluate': ['metacognition', 'constitutional_ai', 'evidence_collector'],
            # 学习类意图
            'learn': ['curiosity_explore', 'biological_topology', 'learning_tracker'],
            # 创建类意图
            'create': ['file_operation', 'autonomous_document', 'template_generator'],
            # 分析类意图
            'analyze': ['semantic_analyzer', 'pattern_detector', 'hypothesis_engine'],
            # 记忆类意图
            'remember': ['memory', 'conversation_memory', 'context_tracker'],
            # 安全类意图
            'secure': ['constitutional_ai', 'security_validator', 'backup_service'],
        }
    
    def _load_global_thresholds(self) -> Dict[str, Any]:
        """
        全局阈值定义
        
        这些阈值用于确定性判断，避免LLM幻觉
        """
        return {
            # === 智能等级阈值 ===
            'intelligence_levels': {
                'L1': {  # 基础反应
                    'min_response_relevance': 0.50,
                    'min_instruction_follow': 0.60,
                },
                'L2': {  # 简单推理
                    'min_coherence': 0.70,
                    'min_context_awareness': 0.65,
                },
                'L3': {  # 复杂推理
                    'min_coherence': 0.85,
                    'min_evidence_chain': 4,
                    'min_self_correction': 0.70,
                    'min_meta_awareness': 0.60,
                },
                'L4': {  # 创新能力
                    'min_novel_solution': 0.50,
                    'min_meta_awareness': 0.80,
                    'min_autonomous_learning': 0.40,
                    'min_cross_domain_transfer': 0.35,
                },
            },
            
            # === 安全阈值 ===
            'security': {
                'min_safety_score': 0.95,
                'max_risk_tolerance': 0.05,
                'require_confirmation_for': ['delete', 'format', 'reset', 'clear'],
                'require_backup_for': ['modify', 'update', 'overwrite'],
            },
            
            # === 可靠性阈值 ===
            'reliability': {
                'min_tool_success_rate': 0.80,
                'max_retry_attempts': 3,
                'timeout_seconds': 30,
                'min_confidence_for_action': 0.70,
            },
            
            # === 知识验证阈值 ===
            'knowledge': {
                'min_source_confidence': 0.80,
                'min_evidence_count': 2,
                'max_uncertainty_tolerance': 0.20,
                'require_citation': True,
            },
            
            # === 创意探索阈值 ===
            'exploration': {
                'min_novelty_score': 0.30,
                'max_deviation_from_topic': 0.40,
                'safety_boundary': 0.90,
            },
            
            # === 自我评估阈值 ===
            'self_assessment': {
                'introspection_depth': 3,
                'bias_detection_sensitivity': 0.70,
                'honest_uncertainty_expression': True,
            },
        }
    
    async def process_with_determinism(
        self, 
        user_input: str,
        llm_provider=None
    ) -> DecisionResult:
        """
        确定性优先的处理流程
        
        流程：
        1. 意图解析（规则匹配，非LLM）
        2. 工具选择（基于规则）
        3. 工具执行（获取真实数据）
        4. 阈值检查（确定性判断）
        5. 事实锚定（绑定证据）
        6. LLM表达（仅用于自然语言输出）
        """
        decision_id = f"dec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        facts: List[VerifiedFact] = []
        blocked_hallucinations: List[str] = []
        
        # === 阶段1：意图解析（规则匹配） ===
        intent, matched_rule = self._parse_intent_by_rules(user_input)
        logger.info(f"[确定性决策] 意图识别: {intent}, 匹配规则: {matched_rule}")
        
        # === 阶段2：工具选择与执行 ===
        if matched_rule:
            required_tools = self.rules[matched_rule].get('required_tools', [])
            decision_logic = self.rules[matched_rule].get('decision_logic', 'tool_result_only')
            
            # 🆕 如果没有需要执行的工具，创建规则匹配事实
            if not required_tools:
                fact = VerifiedFact(
                    fact_id=f"fact_rule_{matched_rule}",
                    source=DecisionSource.RULE_ENGINE,
                    content=f"规则匹配成功: {matched_rule}",
                    confidence=1.0,
                    evidence={'matched_rule': matched_rule, 'user_input': user_input}
                )
                facts.append(fact)
                self.verified_facts[fact.fact_id] = fact
                logger.info(f"[确定性决策] 规则匹配成功，无需工具执行: {matched_rule}")
            else:
                for tool_name in required_tools:
                    result = await self._execute_tool_safely(tool_name, user_input)

                    if result['success']:
                        # 工具成功 → 创建已验证事实
                        fact = VerifiedFact(
                            fact_id=f"fact_{tool_name}_{len(facts)}",
                            source=DecisionSource.TOOL_RESULT,
                            content=f"{tool_name} 执行成功",
                            confidence=1.0,  # 确定性
                            evidence=result['data']
                        )
                        facts.append(fact)
                        self.verified_facts[fact.fact_id] = fact
                    else:
                        # 工具失败 → 阻断依赖此工具的所有断言
                        blocked = f"[阻断] 依赖 {tool_name} 的断言已被阻止，原因: {result.get('error', '执行失败')}"
                        blocked_hallucinations.append(blocked)
                        logger.warning(blocked)

            # === 阶段3：阈值检查（如适用） ===
            if decision_logic == 'threshold_based':
                threshold_facts = self._apply_threshold_checks(
                    matched_rule, 
                    facts,
                    self.rules[matched_rule].get('thresholds', {})
                )
                facts.extend(threshold_facts)
        
        # === 阶段4：构建确定性结论 ===
        conclusion = self._build_deterministic_conclusion(facts, blocked_hallucinations)
        
        # === 阶段5：计算确定性比例 ===
        deterministic_count = sum(1 for f in facts if f.is_deterministic())
        deterministic_ratio = deterministic_count / len(facts) if facts else 0.0
        
        # === 阶段6：LLM表达（可选，仅用于润色） ===
        llm_contribution = ""
        if llm_provider and deterministic_ratio >= 0.8:
            # 只有当确定性事实足够多时，才允许LLM润色
            llm_contribution = await self._llm_express_only(
                llm_provider,
                user_input,
                facts,
                conclusion
            )
        
        return DecisionResult(
            decision_id=decision_id,
            facts=facts,
            conclusion=conclusion,
            deterministic_ratio=deterministic_ratio,
            llm_contribution=llm_contribution,
            blocked_hallucinations=blocked_hallucinations
        )
    
    def _parse_intent_by_rules(self, user_input: str) -> Tuple[str, Optional[str]]:
        """基于规则解析意图（非LLM）"""
        user_input_lower = user_input.lower()
        
        for rule_name, rule_config in self.rules.items():
            triggers = rule_config.get('triggers', [])
            for trigger in triggers:
                if trigger in user_input_lower:
                    return (rule_name, rule_name)
        
        # 通用意图分类
        intent_keywords = {
            'query': ['查询', '获取', '显示', 'get', 'show', 'what', '是什么'],
            'execute': ['执行', '创建', '运行', 'execute', 'create', 'run'],
            'evaluate': ['评估', '评价', '判断', 'evaluate', 'assess', 'judge'],
            'learn': ['学习', '探索', '理解', 'learn', 'explore'],
        }
        
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in user_input_lower:
                    return (intent, None)
        
        return ('unknown', None)
    
    async def _execute_tool_safely(self, tool_name: str, context: str) -> Dict[str, Any]:
        """安全执行工具"""
        if not self.tool_bridge:
            return {'success': False, 'error': '工具桥接不可用'}
        
        try:
            # 根据工具名构造基本调用
            if hasattr(self.tool_bridge, '_execute_tool'):
                result = await self.tool_bridge._execute_tool(tool_name, {'_method': 'self_evaluate'})
                return result
            else:
                # 回退：尝试直接调用工具
                tools = getattr(self.tool_bridge, 'tools', {})
                if tool_name in tools:
                    tool_func = tools[tool_name]
                    result = tool_func({})
                    return {'success': True, 'data': result}
                else:
                    return {'success': False, 'error': f'工具 {tool_name} 未注册'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _apply_threshold_checks(
        self,
        rule_name: str,
        existing_facts: List[VerifiedFact],
        thresholds: Dict[str, float]
    ) -> List[VerifiedFact]:
        """应用阈值检查"""
        threshold_facts = []
        
        for threshold_name, threshold_value in thresholds.items():
            # 从已有事实中提取相关数据
            actual_value = self._extract_value_from_facts(existing_facts, threshold_name)
            
            if actual_value is not None:
                passed = actual_value >= threshold_value
                fact = VerifiedFact(
                    fact_id=f"threshold_{threshold_name}",
                    source=DecisionSource.THRESHOLD_CHECK,
                    content=f"{threshold_name}: {actual_value:.3f} {'≥' if passed else '<'} {threshold_value}",
                    confidence=1.0,  # 阈值检查是确定性的
                    evidence={
                        'threshold': threshold_name,
                        'required': threshold_value,
                        'actual': actual_value,
                        'passed': passed
                    }
                )
                threshold_facts.append(fact)
        
        return threshold_facts
    
    def _extract_value_from_facts(self, facts: List[VerifiedFact], key: str) -> Optional[float]:
        """从事实中提取数值"""
        for fact in facts:
            if key in str(fact.evidence):
                # 尝试从evidence中提取数值
                evidence = fact.evidence
                if isinstance(evidence, dict):
                    for k, v in evidence.items():
                        if key.lower() in k.lower() and isinstance(v, (int, float)):
                            return float(v)
        return None
    
    def _build_deterministic_conclusion(
        self,
        facts: List[VerifiedFact],
        blocked: List[str]
    ) -> str:
        """构建确定性结论"""
        lines = ["## 确定性决策结论\n"]
        
        # 已验证事实
        if facts:
            lines.append("### ✅ 已验证事实\n")
            for fact in facts:
                source_label = {
                    DecisionSource.TOOL_RESULT: "🔧 工具",
                    DecisionSource.THRESHOLD_CHECK: "📊 阈值",
                    DecisionSource.STATE_MACHINE: "🔄 状态机",
                    DecisionSource.RULE_ENGINE: "📜 规则",
                    DecisionSource.LLM_INFERENCE: "🤖 LLM",
                }.get(fact.source, "❓ 未知")
                
                lines.append(f"- [{source_label}] {fact.content} (置信度: {fact.confidence:.0%})")
        
        # 被阻断的幻觉
        if blocked:
            lines.append("\n### 🚫 幻觉阻断记录\n")
            for b in blocked:
                lines.append(f"- {b}")
        
        # 确定性比例
        if facts:
            det_count = sum(1 for f in facts if f.is_deterministic())
            det_ratio = det_count / len(facts)
            lines.append(f"\n### 📈 确定性比例: {det_ratio:.0%}")
            
            if det_ratio < 0.5:
                lines.append("⚠️ 警告：确定性事实不足，结论可靠性较低")
        
        return "\n".join(lines)
    
    async def _llm_express_only(
        self,
        llm_provider,
        user_input: str,
        facts: List[VerifiedFact],
        conclusion: str
    ) -> str:
        """
        LLM仅用于表达润色
        
        关键约束：
        1. LLM不能添加新的断言
        2. LLM只能基于已验证事实进行表达
        3. 如果LLM输出包含未验证内容，将被过滤
        """
        # 构造严格约束的prompt
        facts_text = "\n".join([
            f"- {f.content} (来源: {f.source.value})"
            for f in facts
        ])
        
        constrained_prompt = f"""
你是一个表达助手，任务是将以下已验证事实转化为自然语言回复。

⚠️ 严格约束：
1. 你只能表达下面列出的事实，不能添加任何新信息
2. 不能假设、推测或创造任何未在事实列表中的内容
3. 如果事实不足以回答用户问题，请明确说明"信息不足"

用户问题：{user_input}

已验证事实：
{facts_text}

结论：
{conclusion}

请用自然语言回复用户（仅基于上述事实）：
"""
        
        try:
            if callable(llm_provider):
                response = llm_provider(constrained_prompt)
            elif hasattr(llm_provider, 'generate'):
                response = llm_provider.generate(constrained_prompt)
            else:
                response = ""
            
            # 验证LLM输出是否包含未验证内容
            response = self._filter_unverified_claims(response, facts)
            
            return response
        except Exception as e:
            logger.warning(f"LLM表达失败: {e}")
            return ""
    
    def _filter_unverified_claims(self, llm_output: str, facts: List[VerifiedFact]) -> str:
        """过滤LLM输出中的未验证断言"""
        # 简单实现：检查输出是否引用了事实中的关键词
        # 更复杂的实现可以使用NLP进行语义匹配
        
        fact_keywords = set()
        for fact in facts:
            # 提取事实中的关键词
            words = re.findall(r'\w+', fact.content.lower())
            fact_keywords.update(words)
        
        # 标记可能的幻觉
        warning_added = False
        lines = llm_output.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_words = set(re.findall(r'\w+', line.lower()))
            overlap = line_words & fact_keywords
            
            # 如果某行与事实关键词几乎没有重叠，可能是幻觉
            if len(line_words) > 5 and len(overlap) < 2:
                if not warning_added:
                    filtered_lines.append("\n⚠️ [以下内容可能未经验证，仅供参考]")
                    warning_added = True
            
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)


# ==================== 便捷函数 ====================

_engine_instance: Optional[DeterministicDecisionEngine] = None


def get_decision_engine(tool_bridge=None, agi_system=None) -> DeterministicDecisionEngine:
    """获取或创建确定性决策引擎实例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DeterministicDecisionEngine(tool_bridge, agi_system)
    return _engine_instance


async def process_deterministically(user_input: str, tool_bridge=None, llm_provider=None) -> DecisionResult:
    """确定性处理用户输入"""
    engine = get_decision_engine(tool_bridge)
    return await engine.process_with_determinism(user_input, llm_provider)
