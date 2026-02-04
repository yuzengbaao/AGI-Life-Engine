#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一上下文管理器 (Unified Context Manager)
==========================================

设计理念：
1. 统一所有引擎的对话历史（LLM优先 + 幻觉感知）
2. 统一持久化到 dialogue_controller
3. 统一会话ID管理
4. 整合意图历史和相关记忆

作者: Claude Code (Sonnet 4.5)
日期: 2026-01-26
版本: 1.0.0
"""

import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class UnifiedContextManager:
    """
    统一的上下文管理器

    职责：
    1. 管理所有引擎的对话历史（LLM优先 + 幻觉感知）
    2. 统一持久化到 dialogue_controller
    3. 统一会话ID管理
    4. 整合意图历史和相关记忆
    """

    def __init__(
        self,
        dialogue_controller=None,
        session_id: str = "cli_session_default",
        llm_first_engine=None,
        intent_bridge=None,
        agi_system=None
    ):
        """
        初始化统一上下文管理器

        Args:
            dialogue_controller: 对话控制器
            session_id: 会话ID
            llm_first_engine: LLM优先对话引擎
            intent_bridge: 意图桥接器
            agi_system: AGI系统实例
        """
        self.dialogue_controller = dialogue_controller
        self.session_id = session_id
        self.llm_first_engine = llm_first_engine
        self.intent_bridge = intent_bridge
        self.agi_system = agi_system

        # 对话历史（本地缓存）
        self._local_history = []
        self._max_history_size = 100

        logger.info(f"✅ 统一上下文管理器已初始化 (session={session_id})")

    def add_message(self, role: str, content: str, metadata: Dict = None):
        """
        添加消息到所有历史存储

        Args:
            role: 角色 ('user', 'assistant', 'system')
            content: 消息内容
            metadata: 额外元数据
        """
        timestamp = time.time()

        # 1. 添加到本地缓存
        self._local_history.append({
            'role': role,
            'content': content,
            'timestamp': timestamp,
            'metadata': metadata or {}
        })

        # 限制本地历史大小
        if len(self._local_history) > self._max_history_size:
            self._local_history = self._local_history[-self._max_history_size:]

        # 2. 添加到 dialogue_controller
        if self.dialogue_controller:
            try:
                self.dialogue_controller.add_message(self.session_id, role, content)
                logger.debug(f"[UnifiedContext] 消息已添加到 dialogue_controller: [{role}] {content[:50]}...")
            except Exception as e:
                logger.warning(f"[UnifiedContext] 添加到 dialogue_controller 失败: {e}")

        # 3. 添加到 LLMFirstDialogueEngine（如果存在）
        if self.llm_first_engine:
            try:
                self.llm_first_engine._add_to_history(role, content)
                logger.debug(f"[UnifiedContext] 消息已添加到 LLMFirstDialogueEngine: [{role}]")
            except Exception as e:
                logger.warning(f"[UnifiedContext] 添加到 LLMFirstDialogueEngine 失败: {e}")

        # 4. 持久化到 CONVERSATION_HISTORY.md（通过 LLMFirstDialogueEngine）
        if self.llm_first_engine:
            try:
                self.llm_first_engine._persist_history()
                logger.debug(f"[UnifiedContext] 对话历史已持久化到文件")
            except Exception as e:
                logger.warning(f"[UnifiedContext] 持久化对话历史失败: {e}")

    def get_full_context(self, user_input: str = None, limit: int = 10) -> str:
        """
        获取完整上下文（包括对话历史、意图历史、相关记忆）

        Args:
            user_input: 当前用户输入（用于检索相关记忆）
            limit: 对话历史条数限制

        Returns:
            完整的上下文字符串
        """
        context_parts = []

        # 1. 获取对话历史
        history = self.get_recent_history(limit=limit)
        if history:
            context_parts.append("\n[对话历史]\n")
            for item in history:
                context_parts.append(f"{item['role']}: {item['content']}\n")

        # 2. 获取意图历史（如果有未完成的意图）
        if self.intent_bridge and hasattr(self.intent_bridge, 'intent_history'):
            try:
                recent_intents = self.intent_bridge.intent_history[-5:]  # 最近5个意图
                pending_intents = [
                    i for i in recent_intents
                    if i.state.value in ['pending', 'analyzing', 'confirming', 'executing']
                ]

                if pending_intents:
                    context_parts.append("\n[未完成的意图]\n")
                    for intent in pending_intents:
                        context_parts.append(f"- {intent.raw_input[:100]}... (状态: {intent.state.value})\n")
                    logger.debug(f"[UnifiedContext] 添加了 {len(pending_intents)} 个未完成意图到上下文")
            except Exception as e:
                logger.warning(f"[UnifiedContext] 获取意图历史失败: {e}")

        # 3. 获取相关记忆（如果提供了用户输入）
        if user_input and self.agi_system and hasattr(self.agi_system, 'experience_memory'):
            try:
                relevant_memories = self.agi_system.experience_memory.retrieve_relevant(
                    query=user_input,
                    limit=3
                )
                if relevant_memories:
                    context_parts.append("\n[相关记忆]\n")
                    for mem in relevant_memories[:3]:
                        content_preview = mem.get('content', mem.get('metadata', {}))
                        if isinstance(content_preview, dict):
                            content_preview = str(content_preview)
                        context_parts.append(f"- {str(content_preview)[:100]}...\n")
                    logger.debug(f"[UnifiedContext] 添加了 {len(relevant_memories)} 个相关记忆到上下文")
            except Exception as e:
                logger.debug(f"[UnifiedContext] 获取相关记忆失败: {e}")

        return "\n".join(context_parts)

    def get_recent_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的对话历史

        Args:
            limit: 返回的条数限制

        Returns:
            对话历史列表
        """
        # 优先从 dialogue_controller 获取（最权威）
        if self.dialogue_controller:
            try:
                history = self.dialogue_controller.get_recent_history(self.session_id, limit=limit)
                if history:
                    return history
            except Exception as e:
                logger.warning(f"[UnifiedContext] 从 dialogue_controller 获取历史失败: {e}")

        # 回退到本地缓存
        return self._local_history[-limit:] if self._local_history else []

    def switch_session(self, new_session_id: str):
        """
        切换到新会话

        Args:
            new_session_id: 新会话ID
        """
        old_session_id = self.session_id
        self.session_id = new_session_id

        # 清空本地历史（不同会话不应该共享历史）
        self._local_history.clear()

        logger.info(f"[UnifiedContext] 会话已切换: {old_session_id} -> {new_session_id}")

    def clear_history(self):
        """清空当前会话的所有历史"""
        self._local_history.clear()

        # 同时清空其他存储
        if self.dialogue_controller:
            try:
                # TODO: 实现 dialogue_controller 的清空方法
                pass
            except Exception as e:
                logger.warning(f"[UnifiedContext] 清空 dialogue_controller 历史失败: {e}")

        if self.llm_first_engine:
            try:
                self.llm_first_engine.clear_history()
            except Exception as e:
                logger.warning(f"[UnifiedContext] 清空 LLMFirstDialogueEngine 历史失败: {e}")

        logger.info(f"[UnifiedContext] 会话历史已清空: {self.session_id}")

    def get_status(self) -> Dict[str, Any]:
        """获取上下文管理器状态"""
        history = self.get_recent_history(limit=1000)

        return {
            'session_id': self.session_id,
            'local_history_count': len(self._local_history),
            'total_history_count': len(history),
            'has_dialogue_controller': self.dialogue_controller is not None,
            'has_llm_first_engine': self.llm_first_engine is not None,
            'has_intent_bridge': self.intent_bridge is not None,
            'has_agi_system': self.agi_system is not None,
        }

    def __repr__(self):
        return f"UnifiedContextManager(session={self.session_id}, history={len(self._local_history)} items)"
