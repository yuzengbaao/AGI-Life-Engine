#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DialogueContext:
    user_input: str
    conversation_history: List[Dict[str, str]]
    cognitive_capabilities: Dict[str, bool]
    available_tools: List[str]
    system_state: Dict[str, Any]

class LLMFirstDialogueEngine:
    def __init__(self, agi_system=None, llm_service=None, cognitive_bridge=None):
        self.agi_system = agi_system
        self.llm_service = llm_service
        self.cognitive_bridge = cognitive_bridge
        self.max_history_length = 10
        self.response_timeout = 10
        self._conversation_history = []
        self._max_history_size = 50
        self._load_history()
        logger.info("LLMFirstDialogueEngine Initialized (Persistent History)")

    def build_enhanced_prompt(self, context: DialogueContext) -> str:
        prompt_parts = []
        prompt_parts.append(self._get_system_identity())
        
        if context.cognitive_capabilities:
            prompt_parts.append(self._get_capabilities_description(context.cognitive_capabilities))
            
        if context.available_tools:
            prompt_parts.append(self._get_tools_description(context.available_tools))
            
        if context.system_state:
            prompt_parts.append(self._get_system_state(context.system_state))
            
        # IMPORTANT: Inject History
        if context.conversation_history:
            history_str = self._format_conversation_history(context.conversation_history)
            prompt_parts.append(history_str)
            print(f"   [DEBUG] History Injected: {len(context.conversation_history)} messages")
        else:
            print("   [DEBUG] No history to inject.")

        prompt_parts.append(f"\nUser Input: {context.user_input}")
        
        # Add freedom instruction
        prompt_parts.append("\nPlease respond naturally. You may use tools if needed.")
        
        final_prompt = "\n".join(prompt_parts)
        # Debug Log
        if "Recent Conversation" in final_prompt:
             print("   [DEBUG] History marker CONFIRMED in final prompt.")
        return final_prompt

    def _get_system_identity(self) -> str:
        return "You are AGI, an advanced AI system. You have memory of past conversations. If the user refers to 'last time' or 'previously', check the history."

    def _get_capabilities_description(self, capabilities: Dict[str, bool]) -> str:
        return "Capabilities: " + str(capabilities)

    def _get_tools_description(self, tools: List[str]) -> str:
        return "Available Tools: " + ", ".join(tools)

    def _get_system_state(self, state: Dict[str, Any]) -> str:
        return "System State: Active"

    def _get_conversation_history(self) -> List[Dict[str, str]]:
        return self._conversation_history

    def _add_to_history(self, role: str, content: str):
        import time
        self._conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })
        if len(self._conversation_history) > self._max_history_size:
            self._conversation_history = self._conversation_history[-self._max_history_size:]
        self._persist_history()
        print(f"   [DEBUG] History Saved: {len(self._conversation_history)} items")

    def _get_history_file_path(self) -> str:
        from pathlib import Path
        # D:/TRAE_PROJECT/AGI/core/../data/CONVERSATION_HISTORY.md
        script_dir = Path(__file__).parent.parent.resolve()
        return str(script_dir / "data" / "CONVERSATION_HISTORY.md")

    def _persist_history(self):
        from pathlib import Path
        from datetime import datetime
        try:
            history_file = Path(self._get_history_file_path())
            history_file.parent.mkdir(parents=True, exist_ok=True)
            with open(history_file, 'w', encoding='utf-8') as f:
                f.write("# Conversation History\n\n")
                for msg in self._conversation_history:
                    f.write(f"## {msg['role'].upper()}\n{msg['content']}\n\n")
        except Exception as e:
            logger.warning(f"Failed to persist history: {e}")

    def _load_history(self):
        from pathlib import Path
        import re
        try:
            history_file = Path(self._get_history_file_path())
            if not history_file.exists():
                return
            
            with open(history_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple parsing
            parts = content.split("## ")
            self._conversation_history = []
            for part in parts:
                if not part.strip() or part.startswith("Conversation History"): continue
                lines = part.strip().split("\n", 1)
                if len(lines) == 2:
                    role = lines[0].strip().lower()
                    text = lines[1].strip()
                    self._conversation_history.append({'role': role, 'content': text})
            
            print(f"   [DEBUG] Loaded {len(self._conversation_history)} history items from disk.")
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")

    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        if not history: return ""
        parts = ["\n[Recent Conversation History]"]
        for msg in history[-self.max_history_length:]:
            parts.append(f"{msg['role'].upper()}: {msg['content']}")
        return "\n".join(parts)

    async def process_dialogue(self, user_input: str, context: Optional[Dict] = None) -> str:
        print(f"   [DEBUG] Processing Dialogue: {user_input}")

        # 🔧 [2026-01-26] 关键修复：使用传入的context中的对话历史
        conversation_history_to_use = self._conversation_history

        # 如果context中有conversation_history，优先使用（从UnifiedContextManager来的）
        if context and 'conversation_history' in context and context['conversation_history']:
            # 将字符串格式的对话历史转换为字典列表格式
            # UnifiedContextManager返回的是字符串格式
            try:
                # 解析对话历史字符串
                history_text = context['conversation_history']
                # 简单的解析：假设格式是 "ROLE: content\nROLE: content..."
                lines = history_text.split('\n')
                parsed_history = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('[') or line.startswith('#'):
                        continue
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            role = parts[0].strip().lower()
                            content = parts[1].strip()
                            if role in ['user', 'assistant', 'system']:
                                parsed_history.append({'role': role, 'content': content})

                if parsed_history:
                    conversation_history_to_use = parsed_history
                    print(f"   [DEBUG] 使用UnifiedContextManager的历史: {len(parsed_history)} 条消息")
            except Exception as e:
                logger.warning(f"解析对话历史失败: {e}，使用内部历史")

        # 1. Add User Input
        self._add_to_history('user', user_input)

        # 2. Collect Context
        ctx = DialogueContext(
            user_input=user_input,
            conversation_history=conversation_history_to_use,  # 🔧 使用解析后的历史
            cognitive_capabilities=context.get('cognitive_capabilities', {}) if context else {},
            available_tools=context.get('available_tools', []) if context else [],
            system_state=context.get('system_state', {}) if context else {}
        )

        # 3. Build Prompt
        enhanced_prompt = self.build_enhanced_prompt(ctx)

        # 4. Call LLM (CORRECTED)
        try:
            if self.llm_service:
                response = self.llm_service.chat_completion(
                    system_prompt=enhanced_prompt,
                    user_prompt=user_input
                )
            else:
                response = "Error: LLM Service not available."
            
            # 5. Add Response
            self._add_to_history('assistant', response)
            return response
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            return f"Error: {e}"

def create_llm_first_engine(agi_system) -> LLMFirstDialogueEngine:
    llm_service = getattr(agi_system, 'llm_service', None)
    if not llm_service:
        from core.llm_client import LLMService
        llm_service = LLMService()
    return LLMFirstDialogueEngine(agi_system, llm_service)
