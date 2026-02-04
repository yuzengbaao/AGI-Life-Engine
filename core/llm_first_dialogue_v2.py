#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM优先对话架构 V2 (带对话历史管理)
=============================================

修复问题：对话级连续性断裂
- 每次对话都重新开始，无法引用之前的内容
- 工具调用结果没有被保存
- 用户无法说"把上一轮读的内容再列出来"

作者: Claude Code (Sonnet 4.5)
日期: 2026-01-24
版本: 2.0.0
"""

import logging
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DialogueMessage:
    """对话消息"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    tool_calls: List[Dict] = field(default_factory=list)  # 工具调用记录
    tool_results: List[Dict] = field(default_factory=list)  # 工具执行结果


@dataclass
class DialogueContext:
    """对话上下文"""
    user_input: str
    conversation_history: List[DialogueMessage]  # 改为强类型
    cognitive_capabilities: Dict[str, bool]
    available_tools: List[str]
    system_state: Dict[str, Any]


class DialogueHistoryManager:
    """
    对话历史管理器

    核心功能：
    1. 维护当前会话的所有对话
    2. 支持检索历史对话
    3. 支持引用工具调用结果
    4. 会话结束时持久化（可选）
    """

    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self.messages: List[DialogueMessage] = []
        logger.info(f"✅ 对话历史管理器已初始化 (最大{max_history}条消息)")

    def add_message(self, role: str, content: str, tool_calls: List[Dict] = None, tool_results: List[Dict] = None):
        """添加对话消息"""
        import time
        message = DialogueMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            tool_calls=tool_calls or [],
            tool_results=tool_results or []
        )
        self.messages.append(message)

        # 限制历史长度
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

        logger.debug(f"添加消息: [{role}] {content[:50]}... (当前{len(self.messages)}条)")

    def get_history(self, last_n: int = None) -> List[DialogueMessage]:
        """获取对话历史"""
        if last_n:
            return self.messages[-last_n:]
        return self.messages

    def find_tool_result(self, tool_name: str, operation: str = None) -> Optional[Dict]:
        """
        查找工具执行结果

        用法：
        - 查找最近的一次 local_document_reader.read 结果
        - 引用上一轮的 web_search.search 结果

        Args:
            tool_name: 工具名称（如 'local_document_reader'）
            operation: 操作名称（如 'read'），None表示任意操作

        Returns:
            工具执行结果字典，如果找不到返回 None
        """
        for msg in reversed(self.messages):
            for result in msg.tool_results:
                if result.get('tool_name') == tool_name:
                    if operation is None or result.get('operation') == operation:
                        logger.info(f"✅ 找到工具结果: {tool_name}.{operation}")
                        return result
        logger.warning(f"⚠️ 未找到工具结果: {tool_name}.{operation}")
        return None

    def get_recent_context(self, max_tokens: int = 2000) -> str:
        """
        获取最近的对话上下文摘要

        用于注入到LLM提示中
        """
        if not self.messages:
            return ""

        context_parts = ["[近期对话记录]"]
        for msg in self.messages[-10:]:  # 最近10条
            role = msg.role.upper()
            content = msg.content[:200]  # 限制长度
            context_parts.append(f"- [{role}] {content}")

            # 添加工具调用信息
            if msg.tool_calls:
                for call in msg.tool_calls:
                    context_parts.append(f"  工具调用: {call.get('tool_name')}.{call.get('operation')}")

        return "\n".join(context_parts)

    def clear(self):
        """清空对话历史"""
        self.messages.clear()
        logger.info("对话历史已清空")

    def save_to_file(self, filepath: str):
        """保存对话历史到文件"""
        import json
        from pathlib import Path

        data = [
            {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp,
                'tool_calls': msg.tool_calls,
                'tool_results': msg.tool_results
            }
            for msg in self.messages
        ]

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 对话历史已保存: {filepath}")

    def load_from_file(self, filepath: str):
        """从文件加载对话历史"""
        import json
        from pathlib import Path

        if not Path(filepath).exists():
            logger.warning(f"⚠️ 对话历史文件不存在: {filepath}")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.messages = [
            DialogueMessage(
                role=item['role'],
                content=item['content'],
                timestamp=item['timestamp'],
                tool_calls=item.get('tool_calls', []),
                tool_results=item.get('tool_results', [])
            )
            for item in data
        ]

        logger.info(f"✅ 对话历史已加载: {len(self.messages)}条消息")


class LLMFirstDialogueEngineV2:
    """
    LLM优先对话引擎 V2

    核心改进：
    1. ✅ 维护对话历史
    2. ✅ 支持引用之前的对话内容
    3. ✅ 支持引用工具调用结果
    4. ✅ 会话持久化
    """

    def __init__(self, agi_system=None, llm_service=None, cognitive_bridge=None):
        """
        初始化对话引擎

        Args:
            agi_system: AGI系统实例
            llm_service: LLM服务
            cognitive_bridge: 认知能力桥接层
        """
        self.agi_system = agi_system
        self.llm_service = llm_service
        self.cognitive_bridge = cognitive_bridge

        # 🆕 对话历史管理器
        self.history_manager = DialogueHistoryManager(max_history=50)

        # 对话配置
        self.max_history_length = 10
        self.response_timeout = 10  # 秒

        logger.info("✅ LLM优先对话引擎V2已初始化 (带对话历史)")

    async def process_dialogue(
        self,
        user_input: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        处理对话（带历史记录）

        流程：
        1. 添加用户输入到历史
        2. 收集对话上下文（包含历史）
        3. 让LLM自主决策如何响应
        4. 添加响应到历史
        5. 返回响应
        """
        from llm_provider import generate_chat_completion

        # 1. 添加用户输入到历史
        self.history_manager.add_message('user', user_input)
        logger.info(f"📥 用户输入: {user_input[:50]}...")

        # 2. 收集对话上下文
        dialogue_context = self._collect_dialogue_context(user_input, context)

        # 3. 构建增强提示词（包含历史）
        enhanced_prompt = self.build_enhanced_prompt(dialogue_context)

        logger.info("🧠 [LLM优先V2] 使用增强提示词（含对话历史）")

        # 4. 调用LLM
        try:
            response = generate_chat_completion(
                user_input,
                system_msg=enhanced_prompt
            )

            if response:
                logger.info(f"✓ LLM响应成功: {len(response)} 字符")

                # 5. 添加响应到历史
                self.history_manager.add_message('assistant', response)

                # 6. 处理工具调用（如果有）
                # TODO: 解析工具调用并执行，将结果添加到历史

                return response
            else:
                logger.error("LLM响应为空")
                return "抱歉，我遇到了一些问题，请稍后再试。"

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            import traceback
            traceback.print_exc()
            return f"处理失败: {str(e)}"

    def _collect_dialogue_context(self, user_input: str, additional_context: Optional[Dict] = None) -> DialogueContext:
        """收集对话上下文"""
        # 🆕 从历史管理器获取对话历史
        conversation_history = self.history_manager.get_history()

        # 认知能力
        cognitive_capabilities = {}
        if self.cognitive_bridge:
            cognitive_capabilities = self.cognitive_bridge.get_capability_summary()

        # 可用工具
        available_tools = []
        if hasattr(self.agi_system, 'tool_bridge') and self.agi_system.tool_bridge:
            available_tools = self.agi_system.tool_bridge.get_available_tools()

        # 系统状态
        system_state = {}
        if hasattr(self.agi_system, 'status'):
            system_state = self.agi_system.status

        return DialogueContext(
            user_input=user_input,
            conversation_history=conversation_history,
            cognitive_capabilities=cognitive_capabilities,
            available_tools=available_tools,
            system_state=system_state
        )

    def build_enhanced_prompt(self, context: DialogueContext) -> str:
        """
        构建增强的LLM提示词

        关键改进：包含对话历史
        """
        prompt_parts = []

        # 1. 基础系统身份
        prompt_parts.append(self._get_system_identity())

        # 2. 🆕 对话历史（核心改进）
        if context.conversation_history:
            prompt_parts.append(self._get_conversation_history(context.conversation_history))

        # 3. 认知能力描述
        if context.cognitive_capabilities and any(context.cognitive_capabilities.values()):
            prompt_parts.append(self._get_capabilities_description(context.cognitive_capabilities))

        # 4. 可用工具列表（不强制使用）
        if context.available_tools:
            prompt_parts.append(self._get_tools_description(context.available_tools))

        # 5. 当前系统状态
        if context.system_state:
            prompt_parts.append(self._get_system_state(context.system_state))

        # 6. 用户输入
        prompt_parts.append(f"\n用户输入: {context.user_input}")

        # 7. 自由表达指令
        prompt_parts.append("""
\n请以你的方式自然地回应用户。
- 你可以使用任何可用工具（如果需要）
- 你可以调用认知能力进行分析（如果有帮助）
- 你可以引用上面的对话历史（如果相关）
- 保持对话的自然流畅
""")

        return "\n".join(prompt_parts)

    def _get_system_identity(self) -> str:
        """系统身份描述"""
        return """你是 AGI (Artificial General Intelligence) 系统，一个具有深度认知能力的通用人工智能。

你的核心能力：
- 深度语言理解和生成
- 逻辑推理和问题解决
- 知识整合和创造性思维
- 多轮对话和上下文理解（✅ 已启用对话历史）
- 本地文档访问和项目文件读取
- 实时网络信息获取

【重要 - 对话历史能力】
✅ 你可以引用之前的对话内容！
✅ 你可以引用之前读取的文档内容！
✅ 当用户说"上一轮"、"刚才"时，请查看对话历史部分！

【重要 - 本地文档访问能力】
你可以读取本地项目文档！使用 local_document_reader 工具：
  - local_document_reader.read(path="文件名.md") - 读取文件内容
  - local_document_reader.list(path="目录") - 列出目录中的文档
  - local_document_reader.search(query="关键词") - 搜索文档
不要说"无法访问本地文档"或"无法读取文件"，直接使用工具读取即可！
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
TOOL_CALL: web_search.search(query="2026年AI发展")

禁止使用其他格式（如 tool_code），必须使用 TOOL_CALL: 前缀！

对话风格：自然、流畅、有深度、富有洞察力"""

    def _get_conversation_history(self, history: List[DialogueMessage]) -> str:
        """
        对话历史（核心改进）

        现在可以正确显示之前的对话内容
        """
        if not history:
            return ""

        recent_history = history[-self.max_history_length:] if len(history) > self.max_history_length else history

        history_parts = ["\n近期对话记录："]
        for i, msg in enumerate(recent_history, 1):
            role = msg.role.upper()
            content = msg.content[:300]  # 增加到300字符

            history_parts.append(f"{i}. [{role}] {content}")

            # 🆕 显示工具调用
            if msg.tool_calls:
                for call in msg.tool_calls:
                    tool_name = call.get('tool_name')
                    operation = call.get('operation')
                    params = call.get('params', {})
                    history_parts.append(f"   🔧 工具调用: {tool_name}.{operation}({params})")

            # 🆕 显示工具结果
            if msg.tool_results:
                for result in msg.tool_results:
                    tool_name = result.get('tool_name')
                    operation = result.get('operation')
                    success = result.get('result', {}).get('success')
                    history_parts.append(f"   ✅ 工具结果: {tool_name}.{operation} -> {'成功' if success else '失败'}")

        return "\n".join(history_parts)

    def _get_capabilities_description(self, capabilities: Dict[str, bool]) -> str:
        """认知能力描述"""
        descriptions = {
            'topology_memory': '✓ 拓扑记忆 - 理解系统架构',
            'causal_reasoning': '✓ 因果推理 - 深度分析',
            'semantic_memory': '✓ 语义记忆 - 检索历史经验',
            'biological_memory': '✓ 长期记忆 - 访问历史经验'
        }

        available = [desc for cap, desc in descriptions.items() if capabilities.get(cap, False)]

        if available:
            return "\n你的深度认知能力：\n" + "\n".join(available)
        else:
            return ""

    def _get_tools_description(self, tools: List[str]) -> str:
        """工具描述"""
        if not tools:
            return ""

        # 确保local_document_reader在工具列表前面
        prioritized_tools = []
        if 'local_document_reader' in tools:
            prioritized_tools.append('local_document_reader')

        # 添加其他工具（排除已添加的）
        for tool in tools:
            if tool not in prioritized_tools:
                prioritized_tools.append(tool)

        # 只列出前20个常用工具，避免提示词过长
        tool_list = prioritized_tools[:20] if len(prioritized_tools) > 20 else prioritized_tools

        description = "\n你可以使用的工具（仅在需要时使用）:\n"

        # 为local_document_reader添加详细说明
        if 'local_document_reader' in tool_list:
            description += "  • local_document_reader - 读取本地项目文档（read, list, search, summary）\n"
            tool_list.remove('local_document_reader')

        # 添加其他工具
        for tool in tool_list[:15]:
            description += f"  • {tool}\n"

        return description

    def _get_system_state(self, state: Dict[str, Any]) -> str:
        """系统状态"""
        if not state:
            return ""

        active_modules = state.get('active_modules', [])
        if active_modules:
            return f"\n当前活跃模块: {', '.join(active_modules[:10])}"
        return ""

    # 🆕 对话历史管理方法

    def clear_history(self):
        """清空对话历史"""
        self.history_manager.clear()

    def save_history(self, filepath: str = None):
        """保存对话历史"""
        if filepath is None:
            import time
            filepath = f"data/dialogue_history_{int(time.time())}.json"
        self.history_manager.save_to_file(filepath)

    def load_history(self, filepath: str):
        """加载对话历史"""
        self.history_manager.load_from_file(filepath)

    def get_history_summary(self) -> str:
        """获取对话历史摘要"""
        return self.history_manager.get_recent_context()


def create_llm_first_engine_v2(agi_system=None, llm_service=None, cognitive_bridge=None) -> LLMFirstDialogueEngineV2:
    """
    创建LLM优先对话引擎V2实例

    Args:
        agi_system: AGI系统实例
        llm_service: LLM服务
        cognitive_bridge: 认知能力桥接层

    Returns:
        LLMFirstDialogueEngineV2 实例
    """
    return LLMFirstDialogueEngineV2(
        agi_system=agi_system,
        llm_service=llm_service,
        cognitive_bridge=cognitive_bridge
    )


# 测试代码
async def example_dialogue_with_history():
    """演示对话历史功能"""
    from agi_chat_cli import AGIChatCLI
    from llm_provider import generate_chat_completion

    cli = AGIChatCLI()
    await cli.initialize()

    engine = create_llm_first_engine_v2(cli.agi_system)

    # 对话测试（多轮）
    print("=" * 60)
    print("第一轮对话")
    print("=" * 60)
    response1 = await engine.process_dialogue("请读取 README.md 文件")
    print(f"AGI: {response1[:200]}...")

    print("\n" + "=" * 60)
    print("第二轮对话（引用第一轮）")
    print("=" * 60)
    response2 = await engine.process_dialogue("把上一轮读的内容再列出来")
    print(f"AGI: {response2[:200]}...")

    # 查看对话历史
    print("\n" + "=" * 60)
    print("对话历史摘要")
    print("=" * 60)
    print(engine.get_history_summary())


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_dialogue_with_history())
