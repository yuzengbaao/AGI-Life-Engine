#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证：身份抽取与结构化入库 + 检索命中
"""
import asyncio
import json
import os
import sys

# 保证项目根目录在模块搜索路径
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agi_chat_enhanced import AGIEnhancedChat
from unified_memory_system import MemoryPurpose


async def main():
    chat = AGIEnhancedChat()
    await chat.initialize()

    user_input = "我是你的开发者，我叫宝总"
    agi_response = "好的，我会记住。"

    # 触发持久化(含身份抽取)
    await chat._store_conversation_to_memory(user_input, agi_response)

    # 检索：按智能标签与用途过滤
    mem = chat.memory_system
    results = mem.search_memories(
        query="用户姓名 姓名 名字 我叫 开发者",
        memory_types=["text"],
        memory_purpose=MemoryPurpose.KNOWLEDGE,
        tags=["user", "profile", "name"],
        limit=5,
    )

    print("\n=== 检索结果 ===")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        print(f"{i}. id={r.get('memory_id')} score={r.get('importance_score')} time={r.get('timestamp')}")
        print(f"   content= {r.get('content')}")
        print(f"   meta= {meta}")


if __name__ == "__main__":
    asyncio.run(main())
