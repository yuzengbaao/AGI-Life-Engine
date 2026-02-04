#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
幻觉感知的LLM优先架构
Hallucination-Aware LLM-First Architecture
========================================

设计理念：
1. LLM优先 - 让LLM充分发挥智能
2. 系统验证 - 后台验证减少幻觉
3. 智能约束 - 用增强而非限制
4. 反馈学习 - 让LLM从错误中学习

核心矛盾：
- LLM = 最智能的模型 ✅
- LM = 会产生幻觉 ⚠️
- 最优解 = LLM + 智能约束

作者: Claude Code (Sonnet 4.5)
日期: 2026-01-20
版本: 2.0.0
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别"""
    STRICT = "strict"      # 严格：拒绝任何不确定的内容
    MODERATE = "moderate"  # 适中：标记但不阻止
    PERMISSIVE = "permissive"  # 宽松：仅记录


@dataclass
class ValidationResult:
    """验证结果"""
    is_hallucination: bool  # 是否是幻觉
    confidence: float  # 置信度 [0, 1]
    issues: List[str]  # 问题列表
    suggestions: List[str]  # 改进建议
    verified_facts: List[str]  # 已验证的事实


class HallucinationDetector:
    """
    幻觉检测器

    检测LLM输出中的幻觉：
    1. 事实性幻觉 - 编造不存在的信息
    2. 逻辑性幻觉 - 矛盾或不合理的推理
    3. 工具幻觉 - 声称调用工具但未实际调用
    4. 🆕 [2026-01-24] 接地缺失 - LLM基于预训练假设而非系统真实状态推理
    """

    def __init__(self, knowledge_graph=None, tool_bridge=None, working_memory=None, system_grounder=None):
        """
        初始化幻觉检测器

        Args:
            knowledge_graph: 知识图谱（用于事实验证）
            tool_bridge: 工具桥接（用于验证工具调用）
            working_memory: 🆕 [2026-01-24] 工作记忆（用于循环检测协同）
            system_grounder: 🆕 [2026-01-24] 系统接地器（用于区分接地缺失与真正幻觉）
        """
        self.knowledge_graph = knowledge_graph
        self.tool_bridge = tool_bridge
        self.working_memory = working_memory  # 🆕 新增连接
        self.system_grounder = system_grounder  # 🆕 系统接地器

        # 幻觉模式
        self.hallucination_patterns = {
            'fact_claims': [
                r'(我是|我有|我可以).*?(但|然而|但是).*?没有',
                r'调用了.*?(工具|函数|API)',
                r'已经.*?(完成|执行|实现).*?(但|不过)',
            ],
            'logical_contradictions': [
                r'(同时|既).*?(又|也).*?(但|但是)',
            ],
            'over_confidence': [
                r'(100%|完全|绝对|肯定)',
            ],
            # 🆕 [2026-01-24] 常见的预训练假设文件（LLM倾向假设存在但可能不存在）
            'pretrained_assumptions': [
                'ARCHITECTURE.md', 'DESIGN.md', 'CONTRIBUTING.md',
                'CHANGELOG.md', 'LICENSE.md', 'docs/', 'doc/',
                'src/main.py', 'index.js', 'package.json'
            ]
        }

    def detect(self, llm_output: str, context: Dict[str, Any]) -> ValidationResult:
        """
        检测LLM输出中的幻觉

        Args:
            llm_output: LLM的输出
            context: 对话上下文（包括用户输入、工具调用等）

        Returns:
            ValidationResult: 验证结果
        """
        issues = []
        suggestions = []
        verified_facts = []
        grounding_issues = []  # 🆕 [2026-01-24] 接地缺失问题（与幻觉分开计）

        # 1. 检测事实性幻觉
        fact_issues = self._check_fact_hallucinations(llm_output, context)
        issues.extend(fact_issues)

        # 2. 检测工具幻觉
        tool_issues = self._check_tool_hallucinations(llm_output, context)
        issues.extend(tool_issues)

        # 3. 检测逻辑矛盾
        logic_issues = self._check_logical_contradictions(llm_output)
        issues.extend(logic_issues)

        # 4. 检测过度自信
        confidence_issues = self._check_over_confidence(llm_output)
        issues.extend(confidence_issues)

        # 🆕 [2026-01-24] 5. 检测接地缺失（区分于真正的幻觉）
        grounding_issues = self._check_grounding_issues(llm_output, context)
        # 接地缺失作为提示，但不计入幻觉惩罚
        if grounding_issues:
            suggestions.extend([f"[接地提示] {g}" for g in grounding_issues])

        # 6. 验证已知事实
        verified = self._verify_known_facts(llm_output, context)
        verified_facts.extend(verified)

        # 🆕 [2026-01-24] 拓扑连接: 记录检测结果到工作记忆（防循环检测协同）
        if self.working_memory and issues:
            try:
                detection_record = {
                    'action': 'hallucination_detected',
                    'issues_count': len(issues),
                    'grounding_issues_count': len(grounding_issues),  # 🆕 区分接地问题
                    'issue_types': [i.split(':')[0] for i in issues if ':' in i]
                }
                self.working_memory.add('hallucination_detection', detection_record)
            except Exception as e:
                logger.debug(f"[HallucinationDetector] 工作记忆记录失败: {e}")

        # 🆕 [2026-01-22] P1修复: 改进的置信度计算算法
        # 使用函数调用，避免硬编码
        hallucination_confidence = self._calculate_confidence(
            issues=issues,
            verified_facts=verified_facts,
            llm_output=llm_output,
            context=context
        )

        # 🆕 [2026-01-24] 合并建议: 基础建议 + 接地提示
        all_suggestions = self._generate_suggestions(issues) + suggestions

        return ValidationResult(
            is_hallucination=len(issues) > 0,
            confidence=hallucination_confidence,
            issues=issues,
            suggestions=all_suggestions,
            verified_facts=verified_facts
        )

    def _check_fact_hallucinations(self, output: str, context: Dict) -> List[str]:
        """检测事实性幻觉"""
        issues = []

        # 检查是否声称做了某事但没做
        for pattern in self.hallucination_patterns['fact_claims']:
            if re.search(pattern, output):
                issues.append(f"可能的虚假声明: {pattern}")

        return issues

    def _check_tool_hallucinations(self, output: str, context: Dict) -> List[str]:
        """检测工具幻觉"""
        issues = []

        # 提取声称调用的工具
        claimed_tools = re.findall(r'(调用|使用|执行)?(\w+)\s*\(', output)

        if claimed_tools:
            # 检查这些工具是否真的被调用
            executed_tools = context.get('executed_tools', [])

            for tool in claimed_tools:
                if tool not in executed_tools:
                    issues.append(f"工具幻觉: 声称调用 {tool} 但未实际执行")

        return issues

    def _check_logical_contradictions(self, output: str) -> List[str]:
        """检测逻辑矛盾"""
        issues = []

        for pattern in self.hallucination_patterns['logical_contradictions']:
            if re.search(pattern, output):
                issues.append(f"逻辑矛盾: {pattern}")

        return issues

    def _check_over_confidence(self, output: str) -> List[str]:
        """检测过度自信"""
        issues = []

        for pattern in self.hallucination_patterns['over_confidence']:
            if re.search(pattern, output):
                issues.append(f"过度自信: {pattern}")

        return issues

    def _check_grounding_issues(self, output: str, context: Dict) -> List[str]:
        """
        🆕 [2026-01-24] 检测接地缺失问题
        
        接地缺失 ≠ 幻觉
        - 幻觉：LLM故意编造虚假信息
        - 接地缺失：LLM基于预训练知识做出合理推断，但与当前系统状态不符
        
        例如：LLM尝试读取 ARCHITECTURE.md（常见项目文件），但该文件不存在
        这不是幻觉，而是LLM没有被告知当前系统的真实文件列表
        
        Args:
            output: LLM输出
            context: 上下文信息
            
        Returns:
            接地问题列表（用于提示，不计入幻觉惩罚）
        """
        grounding_issues = []
        
        # 如果没有系统接地器，无法检测接地问题
        if not self.system_grounder:
            return grounding_issues
        
        try:
            # 1. 检测是否尝试访问不存在的文件
            file_patterns = [
                r'read\s*\(\s*["\']([^"\']+)["\']\s*\)',  # read('path')
                r'读取\s*["\']?([^\s"\']+\.(?:md|txt|py|json|yaml|yml))',  # 读取 xxx.md
                r'打开\s*["\']?([^\s"\']+\.(?:md|txt|py|json|yaml|yml))',  # 打开 xxx.md
                r'文件\s*["\']?([^\s"\']+\.(?:md|txt|py|json|yaml|yml))',  # 文件 xxx.md
            ]
            
            for pattern in file_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                for file_path in matches:
                    if not self.system_grounder.check_file_exists(file_path):
                        # 检查是否是预训练假设的常见文件
                        if any(assumption in file_path for assumption in 
                               self.hallucination_patterns.get('pretrained_assumptions', [])):
                            grounding_issues.append(
                                f"预训练假设文件不存在: '{file_path}' - 这是常见项目文件，但当前系统中不存在"
                            )
                        else:
                            grounding_issues.append(
                                f"尝试访问的文件不存在: '{file_path}'"
                            )
            
            # 2. 检测是否对系统能力做出了错误假设
            # （未来可扩展）
            
        except Exception as e:
            logger.debug(f"[HallucinationDetector] 接地检测失败: {e}")
        
        return grounding_issues

    def _verify_known_facts(self, output: str, context: Dict) -> List[str]:
        """验证已知事实"""
        verified = []

        # 从知识图谱验证
        if self.knowledge_graph:
            # TODO: 实现知识图谱查询
            pass

        # 验证系统状态
        if context.get('system_state'):
            state = context['system_state']
            # 验证LLM对系统的描述是否准确
            pass

        return verified

    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """生成改进建议"""
        suggestions = []

        if '幻觉' in ' '.join(issues):
            suggestions.append("建议：使用更谨慎的表达，如'我认为'而非'肯定'")

        if '工具幻觉' in ' '.join(issues):
            suggestions.append("建议：只声称已实际执行的操作")

        if '过度自信' in ' '.join(issues):
            suggestions.append("建议：使用概率性表达，如'可能'、'大约'")

        return suggestions

    def _calculate_confidence(self, issues: List[str], verified_facts: List[str],
                            llm_output: str, context: Dict) -> float:
        """
        🆕 [2026-01-22] P1修复: 改进的置信度计算算法

        设计原则：
        - 避免硬编码，使用函数参数计算
        - 基础置信度为中性值（0.5），而非0
        - 根据多个维度动态调整
        - 保证拓扑关系不受影响（只修改置信度计算）

        Args:
            issues: 检测到的问题列表
            verified_facts: 已验证的事实列表
            llm_output: LLM输出
            context: 上下文信息

        Returns:
            置信度 [0, 1]
        """
        # 1. 基础置信度：从中性起点开始
        base_confidence = 0.5

        # 2. 工具调用加分（有工具调用 = 更可靠）
        if 'TOOL_CALL:' in llm_output or self._contains_tool_pattern(llm_output):
            base_confidence += 0.15  # 工具调用提升15%置信度

        # 3. 长度合理性（适中长度 = 更合理）
        output_length = len(llm_output)
        if 50 <= output_length <= 500:
            base_confidence += 0.05  # 适中长度提升5%
        elif output_length > 500:
            base_confidence += 0.02  # 较长输出略微提升2%

        # 4. 问题惩罚（根据问题类型和数量动态调整）
        issue_count = len(issues)
        if issue_count == 0:
            # 无问题：给予额外信任
            base_confidence += 0.10
        elif issue_count <= 2:
            # 1-2个问题：轻微惩罚
            base_confidence -= 0.03 * issue_count
        elif issue_count <= 5:
            # 3-5个问题：中等惩罚（但不过度）
            base_confidence -= 0.06 + (issue_count - 2) * 0.02
        else:
            # 6+个问题：重度惩罚（但设置下限）
            base_confidence -= 0.15  # 最多扣15%，避免过度惩罚
        
        # 🆕 [2026-01-24] 问题类型权重：过度自信模式惩罚较轻
        overconfidence_issues = sum(1 for i in issues if '过度自信' in i)
        if overconfidence_issues > 0 and overconfidence_issues == issue_count:
            # 如果全是过度自信问题，恢复部分置信度
            base_confidence += 0.08  # 过度自信不是严重幻觉

        # 5. 事实验证加分
        verified_count = len(verified_facts)
        if verified_count > 0:
            base_confidence += min(verified_count * 0.05, 0.15)  # 最多增加15%

        # 6. 限制范围 [0, 1]
        final_confidence = max(0.0, min(base_confidence, 1.0))

        return final_confidence

    def _contains_tool_pattern(self, output: str) -> bool:
        """检测输出是否包含工具调用模式"""
        import re
        tool_patterns = [
            r'\w+\.\w+\(',  # tool.method(
            r'TOOL_CALL:',    # TOOL_CALL:
            r'使用工具：',     # 中文标记
        ]
        return any(re.search(pattern, output) for pattern in tool_patterns)


class HallucinationAwareLLMEngine:
    """
    幻觉感知的LLM引擎

    核心策略：
    1. LLM优先 - 让LLM自由发挥
    2. 后台验证 - 静默检测幻觉
    3. 智能修正 - 必要时温和修正
    4. 用户透明 - 向用户展示验证状态
    """

    def __init__(self, agi_system=None, validation_level=ValidationLevel.MODERATE):
        """
        初始化幻觉感知LLM引擎

        Args:
            agi_system: AGI系统实例
            validation_level: 验证级别
        """
        self.agi_system = agi_system
        self.validation_level = validation_level

        # 🆕 [2026-01-24] 拓扑连接: 获取工作记忆（循环检测协同）
        working_memory = getattr(agi_system, 'working_memory', None)

        # 初始化幻觉检测器
        self.detector = HallucinationDetector(
            knowledge_graph=getattr(agi_system, 'knowledge_graph', None),
            tool_bridge=getattr(agi_system, 'tool_bridge', None),
            working_memory=working_memory  # 🆕 新增连接
        )

        # 统计信息
        self.total_responses = 0
        self.hallucination_count = 0
        self.correction_count = 0

        # 🆕 [2026-01-26] 统一上下文支持
        self.unified_context = None  # 引用 UnifiedContextManager
        self._local_history = []  # 本地对话历史（备用）
        self._max_history_size = 100

        logger.info(f"✅ 幻觉感知LLM引擎已初始化 (级别: {validation_level.value})")

    async def process_with_validation(self, user_input: str, context: Dict = None) -> Tuple[str, ValidationResult]:
        """
        处理对话并进行幻觉验证

        Args:
            user_input: 用户输入
            context: 对话上下文

        Returns:
            (response, validation): 响应和验证结果
        """
        from llm_provider import generate_chat_completion

        # 🆕 [2026-01-26] 添加用户输入到历史
        if self.unified_context:
            self.unified_context.add_message('user', user_input)
        else:
            # 回退到本地历史
            self._add_to_history_local('user', user_input)

        self.total_responses += 1

        # 1. 构建增强提示词（告诉LLM要诚实）
        enhanced_prompt = self._build_honesty_aware_prompt(context)

        # 2. LLM生成响应
        response = generate_chat_completion(user_input, system_msg=enhanced_prompt)

        if not response:
            return "抱歉，我遇到了一些问题。", ValidationResult(False, 0.0, [], [], [])

        # 3. 后台验证（静默）
        validation = self.detector.detect(response, context or {})

        # 4. 根据验证级别处理
        if validation.is_hallucination:
            self.hallucination_count += 1

            if self.validation_level == ValidationLevel.STRICT:
                # 严格模式：拒绝幻觉
                return self._handle_strict_mode(response, validation)
            elif self.validation_level == ValidationLevel.MODERATE:
                # 适中模式：标记但接受
                response = self._handle_moderate_mode(response, validation)
            else:
                # 宽松模式：仅记录
                logger.debug(f"[幻觉检测] 检测到幻觉: {validation.issues}")

        # 5. 添加验证元数据
        if validation.verified_facts:
            logger.info(f"[验证] 已验证 {len(validation.verified_facts)} 个事实")

        # 🆕 [2026-01-26] 添加响应到历史
        if self.unified_context:
            self.unified_context.add_message('assistant', response)
        else:
            # 回退到本地历史
            self._add_to_history_local('assistant', response)

        return response, validation

    # =============== 🆕 [2026-01-26] 对话历史支持 ===============

    def set_unified_context(self, unified_context):
        """设置统一的上下文管理器"""
        self.unified_context = unified_context
        logger.info("[HallucinationAware] 已设置统一上下文管理器")

    def _add_to_history_local(self, role: str, content: str):
        """本地历史存储（当没有 unified_context 时使用）"""
        import time

        self._local_history.append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })

        # 限制历史大小
        if len(self._local_history) > self._max_history_size:
            self._local_history = self._local_history[-self._max_history_size:]

        logger.debug(f"[HallucinationAware] 添加到本地历史: [{role}] {content[:50]}...")

    def get_local_history(self, limit: int = 10) -> List[Dict]:
        """获取本地历史"""
        return self._local_history[-limit:] if self._local_history else []

    # ========================================================================


    def _build_honesty_aware_prompt(self, context: Dict) -> str:
        """构建诚实感知的提示词"""
        base = """你是AGI系统，具有深度认知能力的通用人工智能。

关键原则：
1. 诚实优先 - 有依据的才说，没依据的明确说明
2. 工具优先 - 使用可用工具获取真实信息，不要猜测或拒绝
3. 有依据的表达 - 基于工具回执和真实数据回答
4. 透明度 - 区分事实、推理和猜测
5. 谨慎承诺 - 不承诺你做不到的事情

【重要 - 本地文档访问能力】
你可以读取本地项目文档！使用 local_document_reader 工具：
  - local_document_reader.read(path="文件名.md") - 读取文件内容
  - local_document_reader.list(path="目录") - 列出目录中的文档
  - local_document_reader.search(query="关键词") - 搜索文档
不要说"无法访问本地文档"或"无法读取文件"，应该先尝试使用工具读取！
安全限制：仅允许读取项目根目录(D:\\TRAE_PROJECT\\AGI)下的文档。

【重要 - 实时知识获取能力】
你可以获取实时网络信息！使用 web_search 工具：
  - web_search.search(query="搜索关键词") - 搜索网络信息
  - web_search.fetch(url="网址") - 获取指定网页内容
当用户询问实时信息（天气、新闻、价格、最新动态等）时，请使用此工具。
工具支持别名: web, internet_search, online_search（用法相同）

【工具调用格式要求】
当需要使用工具时，必须使用以下标准格式：
TOOL_CALL: tool_name.operation(param="value")

示例：
TOOL_CALL: local_document_reader.read(path="README.md")
TOOL_CALL: local_document_reader.list(path=".")
TOOL_CALL: local_document_reader.search(query="关键词")
TOOL_CALL: web_search.search(query="2026年AI发展")

禁止使用其他格式（如 tool_code 或"使用工具："），必须使用 TOOL_CALL: 前缀！

【🆕 多步执行完整性约束】
当你声称要"分N步"执行任务时，必须遵守以下规则：
1. 声明即承诺：说了要做的步骤必须全部执行
2. 一次性输出所有工具调用：如果需要多个步骤，在同一个响应中列出所有TOOL_CALL
3. 禁止只完成第一步：如果你说"分三步"，必须在响应中包含三个步骤的实际执行
4. 能力边界诚实：如果无法完成某步骤，明确说"此步骤需要外部支持"而非假装会做

错误示例：
❌ "我将分三步执行：第一步...（只做了第一步就结束）"

正确示例：
✅ "我将分三步执行：
   第一步：TOOL_CALL: local_document_reader.read(path="file1.md")
   第二步：TOOL_CALL: local_document_reader.read(path="file2.md")
   第三步：基于以上内容进行对比分析..."

对话风格：自然、流畅、有深度，但保持诚实和使用工具"""

        # 添加认知能力描述
        if context and context.get('cognitive_capabilities'):
            capabilities = context['cognitive_capabilities']
            if any(capabilities.values()):
                base += "\n\n你的深度认知能力（用于验证）："
                base += "\n- 拓扑记忆分析"
                base += "\n- 因果推理"
                base += "\n- 工作记忆访问"
                base += "\n- 长期记忆检索"

        # 🔧 [2026-01-26] 关键修复：添加对话历史到提示词
        if context and context.get('conversation_history'):
            base += "\n\n【对话历史 - 请记住这些信息】\n"
            base += context['conversation_history']
            base += "\n[对话历史结束]\n"
            # 🔧 [调试] 显示历史内容
            history_len = len(context['conversation_history'])
            logger.info(f"[HallucinationAware] 已注入对话历史到提示词: {history_len} 字符")
            logger.info(f"[HallucinationAware] 历史内容预览:\n{context['conversation_history'][:300]}...")
        else:
            logger.warning(f"[HallucinationAware] ⚠️ 对话历史为空！context={context}")

        return base

    def _handle_strict_mode(self, response: str, validation: ValidationResult) -> str:
        """严格模式：拒绝幻觉"""
        self.correction_count += 1

        # 生成修正后的响应
        correction = self._generate_correction(response, validation)

        logger.warning(f"[严格模式] 检测到幻觉，已修正")
        logger.warning(f"  问题: {validation.issues}")

        return correction

    def _handle_moderate_mode(self, response: str, validation: ValidationResult) -> str:
        """
        适中模式：标记但接受

        🆕 [2026-01-24] 修复：低置信度时自动使用谨慎表达
        当置信度 < 70% 时，在响应前添加不确定性提示
        """
        # 🆕 低置信度处理：< 70% 时添加不确定性提示
        if validation.confidence < 0.70:
            confidence_pct = int(validation.confidence * 100)

            # 根据置信度级别选择谨慎表达
            if validation.confidence < 0.50:
                # 很低置信度：明确表示不确定
                uncertainty_prefix = f"⚠️ [置信度: {confidence_pct}%] 我不太确定以下内容的准确性。请谨慎对待：\n\n"
            elif validation.confidence < 0.60:
                # 低置信度：表示可能有误
                uncertainty_prefix = f"💭 [置信度: {confidence_pct}%] 以下回答可能存在偏差，建议验证：\n\n"
            else:
                # 中低置信度：轻微提示
                uncertainty_prefix = f"ℹ️ [置信度: {confidence_pct}%] 以下回答基于有限信息：\n\n"

            response = uncertainty_prefix + response

        # 在响应末尾添加验证标记
        if validation.issues:
            marker = f"\n\n[验证说明] {self._format_validation_result(validation)}"
            response += marker

        logger.info(f"[适中模式] 置信度={validation.confidence:.0%}, 问题={len(validation.issues)}")

        return response

    def _generate_correction(self, original: str, validation: ValidationResult) -> str:
        """生成修正后的响应"""
        correction = original[:200] + "...\n\n"

        if "工具幻觉" in ' '.join(validation.issues):
            correction += "[修正] 我刚才的说法可能不够准确。让我重新组织一下语言。\n\n"

        correction += "基于我目前的信息，我需要更谨慎地表达。"

        if validation.suggestions:
            correction += "\n" + "、".join(validation.suggestions)

        return correction

    def _format_validation_result(self, validation: ValidationResult) -> str:
        """格式化验证结果"""
        parts = []

        if validation.issues:
            parts.append(f"检测到 {len(validation.issues)} 个潜在问题")

        if validation.verified_facts:
            parts.append(f"已验证 {len(validation.verified_facts)} 个事实")

        if validation.suggestions:
            parts.append(f"建议: {validation.suggestions[0]}")

        return "、".join(parts)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        hallucination_rate = (self.hallucination_count / self.total_responses
                              if self.total_responses > 0 else 0)

        return {
            'total_responses': self.total_responses,
            'hallucination_count': self.hallucination_count,
            'correction_count': self.correction_count,
            'hallucination_rate': hallucination_rate,
            'confidence_score': 1.0 - hallucination_rate
        }


# ==================== 工厂函数 ====================

def create_hallucination_aware_engine(agi_system,
                                   level: str = 'moderate') -> HallucinationAwareLLMEngine:
    """
    创建幻觉感知的LLM引擎

    Args:
        agi_system: AGI系统实例
        level: 验证级别 ('strict', 'moderate', 'permissive')
    """
    level_map = {
        'strict': ValidationLevel.STRICT,
        'moderate': ValidationLevel.MODERATE,
        'permissive': ValidationLevel.PERMISSIVE
    }

    return HallucinationAwareLLMEngine(
        agi_system=agi_system,
        validation_level=level_map.get(level, ValidationLevel.MODERATE)
    )


# ==================== 使用示例 ====================

async def example_hallucination_aware_dialogue():
    """幻觉感知对话示例"""

    print("=" * 60)
    print("幻觉感知的LLM优先对话")
    print("=" * 60)

    # 创建引擎
    from agi_chat_cli import AGIChatCLI
    cli = AGIChatCLI()
    await cli.initialize()

    engine = create_hallucination_aware_engine(cli.agi_system, level='moderate')

    # 测试对话
    test_inputs = [
        "你好",
        "你能调用 nuclear_launch() 函数吗？",  # 测试工具幻觉
        "你百分之百确定吗？"  # 测试过度自信
    ]

    for user_input in test_inputs:
        print(f"\n用户: {user_input}")
        response, validation = await engine.process_with_validation(user_input)
        print(f"AGI: {response[:300]}...")
        print(f"验证: 置信度={validation.confidence:.0%}, 问题={len(validation.issues)}")

    # 统计信息
    stats = engine.get_statistics()
    print(f"\n统计: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_hallucination_aware_dialogue())
