#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
会话上下文恢复器 (Session Context Restorer)
=============================================

解决问题：会话隔离导致任务需要多轮重复
解决方案：启动时自动加载持久化的上下文和任务状态

版本: 1.0.0
日期: 2026-01-24
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class SessionContextRestorer:
    """
    会话上下文恢复器
    
    功能：
    1. 启动时自动加载最近的对话上下文
    2. 恢复未完成的任务状态
    3. 提供历史记忆摘要给LLM
    """
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        self.project_root = Path(project_root)
        self.memory_dir = self.project_root / "memory"
        self.data_dir = self.project_root / "data"
        
        # 持久化文件路径
        self.consciousness_file = self.data_dir / "consciousness.json"
        self.metacognition_file = self.memory_dir / "metacognition_history.json"
        self.insights_file = self.memory_dir / "long_term_insights.md"
        self.session_history_file = self.data_dir / "session_history.json"
        
        # 上下文恢复时间窗口（小时）
        self.context_window_hours = 24
        
        logger.info("🔄 会话上下文恢复器已初始化")
    
    def restore_context(self) -> Dict[str, Any]:
        """
        恢复会话上下文
        
        返回包含：
        - 最近的对话摘要
        - 未完成的任务
        - 相关的长期洞察
        - 当前系统状态
        """
        context = {
            "restored_at": datetime.now().isoformat(),
            "previous_session": None,
            "active_goals": [],
            "recent_insights": [],
            "working_memory": [],
            "attention_focus": None,
            "restoration_success": False
        }
        
        try:
            # 1. 恢复全局工作区状态
            if self.consciousness_file.exists():
                with open(self.consciousness_file, 'r', encoding='utf-8') as f:
                    consciousness = json.load(f)
                    context["previous_session"] = {
                        "timestamp": consciousness.get("timestamp"),
                        "attention": consciousness.get("attention"),
                        "cognitive_state": consciousness.get("cognitive_state")
                    }
                    context["active_goals"] = consciousness.get("goals", [])
                    context["working_memory"] = consciousness.get("thoughts", [])
                    context["attention_focus"] = consciousness.get("attention")
                    logger.info(f"✅ 恢复了 {len(context['active_goals'])} 个活跃目标")
            
            # 2. 加载最近的元认知洞察
            if self.metacognition_file.exists():
                with open(self.metacognition_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    # 获取最近24小时的洞察
                    cutoff = datetime.now() - timedelta(hours=self.context_window_hours)
                    recent = []
                    for entry in reversed(history[-20:]):  # 最多20条
                        if "insight" in entry:
                            recent.append({
                                "insight": entry["insight"],
                                "timestamp": entry.get("timestamp"),
                                "intelligence_index": entry.get("intelligence_index")
                            })
                    context["recent_insights"] = recent[:5]  # 最近5条
                    logger.info(f"✅ 恢复了 {len(context['recent_insights'])} 条最近洞察")
            
            # 3. 加载长期洞察摘要
            if self.insights_file.exists():
                with open(self.insights_file, 'r', encoding='utf-8') as f:
                    insights_text = f.read()
                    # 提取最近的洞察（最后500字符）
                    context["long_term_summary"] = insights_text[-500:] if len(insights_text) > 500 else insights_text
            
            # 4. 加载会话历史（如果存在）
            if self.session_history_file.exists():
                with open(self.session_history_file, 'r', encoding='utf-8') as f:
                    session_history = json.load(f)
                    if session_history:
                        last_session = session_history[-1]
                        context["last_session_summary"] = last_session.get("summary", "无摘要")
                        context["last_session_tasks"] = last_session.get("pending_tasks", [])
            
            context["restoration_success"] = True
            logger.info("✅ 会话上下文恢复完成")
            
        except Exception as e:
            logger.error(f"❌ 上下文恢复失败: {e}")
            context["restoration_error"] = str(e)
        
        return context
    
    def save_session_state(self, summary: str, pending_tasks: List[str] = None):
        """
        保存当前会话状态（在会话结束时调用）
        """
        try:
            # 加载现有历史
            history = []
            if self.session_history_file.exists():
                with open(self.session_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            
            # 添加当前会话
            session_entry = {
                "timestamp": datetime.now().isoformat(),
                "summary": summary,
                "pending_tasks": pending_tasks or []
            }
            history.append(session_entry)
            
            # 只保留最近50个会话
            history = history[-50:]
            
            # 保存
            os.makedirs(self.session_history_file.parent, exist_ok=True)
            with open(self.session_history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 会话状态已保存")
            
        except Exception as e:
            logger.error(f"❌ 会话状态保存失败: {e}")
    
    def generate_context_prompt(self) -> str:
        """
        生成用于注入LLM的上下文恢复提示
        """
        context = self.restore_context()
        
        if not context["restoration_success"]:
            return ""
        
        prompt_parts = ["[会话上下文恢复]"]
        
        # 上一次会话信息
        if context.get("previous_session"):
            prev = context["previous_session"]
            prompt_parts.append(f"- 上次会话状态: {prev.get('cognitive_state', '未知')}")
            prompt_parts.append(f"- 上次关注焦点: {prev.get('attention', '未知')}")
        
        # 活跃目标
        if context.get("active_goals"):
            prompt_parts.append(f"- 未完成目标 ({len(context['active_goals'])}个):")
            for goal in context["active_goals"][:3]:
                prompt_parts.append(f"  - {goal.get('goal', '未知目标')}")
        
        # 最近洞察
        if context.get("recent_insights"):
            prompt_parts.append("- 最近洞察:")
            for insight in context["recent_insights"][:2]:
                prompt_parts.append(f"  - {insight.get('insight', '')[:100]}...")
        
        # 上次会话摘要
        if context.get("last_session_summary"):
            prompt_parts.append(f"- 上次会话摘要: {context['last_session_summary'][:200]}")
        
        # 待处理任务
        if context.get("last_session_tasks"):
            prompt_parts.append("- 待处理任务:")
            for task in context["last_session_tasks"][:3]:
                prompt_parts.append(f"  - {task}")
        
        return "\n".join(prompt_parts)


# 全局实例
_context_restorer = None

def get_context_restorer() -> SessionContextRestorer:
    """获取全局上下文恢复器实例"""
    global _context_restorer
    if _context_restorer is None:
        _context_restorer = SessionContextRestorer()
    return _context_restorer


def restore_session_context() -> str:
    """
    快捷函数：恢复会话上下文并返回提示
    
    在对话引擎启动时调用此函数，将返回的字符串注入系统提示
    """
    return get_context_restorer().generate_context_prompt()


if __name__ == "__main__":
    # 测试
    logging.basicConfig(level=logging.INFO)
    restorer = SessionContextRestorer()
    context = restorer.restore_context()
    print(json.dumps(context, ensure_ascii=False, indent=2))
    print("\n--- 生成的上下文提示 ---")
    print(restorer.generate_context_prompt())
