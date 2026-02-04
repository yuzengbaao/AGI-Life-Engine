#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI 真实能力验证脚本
用于验证 EventDrivenSystem 和 ConsciousnessEngine 的真实能力修复情况
"""

import asyncio
import os
import sys
import time
import logging
import shutil
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from active_agi.event_driven_system import EventDrivenSystem, EventHandler, Event, EventType, EventPriority
from active_agi.learning_event_handler import LearningEventHandler
from active_agi.consciousness_engine import ContinuousConsciousness
from unified_memory_system import UnifiedMemorySystem, MemoryPurpose

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Verification")

# 模拟持续学习框架，用于验证连接
class MockLearningFramework:
    def __init__(self):
        self.learned_events = []
        self.memory_system = None

    async def learn_from_experience(self, input_text: str, outcome: Dict[str, Any]) -> None:
        logger.info(f"🔥 [VERIFICATION SUCCESS] Learning triggered: {input_text[:50]}...")
        self.learned_events.append({
            'input': input_text,
            'outcome': outcome,
            'timestamp': time.time()
        })

async def verify_real_abilities():
    print("="*60)
    print("🔍 AGI 真实能力修复验证程序")
    print("="*60)

    # 1. 初始化记忆系统
    print("\n[1/4] 初始化统一记忆系统...")
    memory_system = UnifiedMemorySystem(enable_visual_memory=False)
    
    # 插入一条测试记忆
    test_memory_id = memory_system.add_text_memory(
        content="验证记忆: AGI系统必须具备真实的感知和学习能力，而不是随机模拟。",
        memory_purpose=MemoryPurpose.KNOWLEDGE,
        tags=["verification", "important"],
        importance_score=0.9
    )
    print(f"✅ 插入测试记忆 ID: {test_memory_id}")

    # 2. 初始化事件系统和学习处理器
    print("\n[2/4] 初始化事件驱动系统 (带真实监控)...")
    learning_framework = MockLearningFramework()
    learning_handler = LearningEventHandler(learning_framework)
    
    event_system = EventDrivenSystem()
    event_system.event_bus.register_handler(learning_handler)
    
    # 启动事件系统
    print("🚀 启动事件驱动循环...")
    loop_task = asyncio.create_task(event_system.event_loop(check_interval=0.5))
    print("✅ 事件系统已启动")

    # 3. 验证真实文件监控和学习触发
    print("\n[3/4] 验证真实文件监控和学习闭环...")
    
    # 创建一个被监控的文件
    monitor_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "learning_state.json")
    if not os.path.exists(monitor_file):
        with open(monitor_file, 'w') as f:
            f.write("{}")
    
    # 触发文件修改
    print(f"📝 修改文件以触发事件: {monitor_file}")
    with open(monitor_file, 'w') as f:
        f.write('{"updated": true, "time": "%s"}' % time.time())
    
    # 等待事件处理
    print("⏳ 等待事件捕获和处理 (5秒)...")
    await asyncio.sleep(5)
    
    if learning_framework.learned_events:
        print(f"✅ 成功捕获并学习了 {len(learning_framework.learned_events)} 个事件!")
        print(f"   最后事件内容: {learning_framework.learned_events[-1]['input']}")
    else:
        print("❌ 未能捕获文件变更事件 (可能需要检查 monitor_files 配置)")

    # 4. 验证意识引擎的真实记忆回顾
    print("\n[4/4] 验证意识引擎的真实记忆回顾...")
    consciousness = ContinuousConsciousness(memory_system=memory_system)
    
    # 运行一次记忆回顾
    print("🧠 运行记忆回顾 (Strategy: important)...")
    
    # 强制调用私有方法进行测试
    # 注意: 真实运行中是在 _consciousness_loop 中调用的
    # 这里我们需要模拟 _recall_memories 的行为，或者直接调用它
    memories = await consciousness._recall_memories()
    
    found_test_memory = False
    if memories:
        print(f"✅ 成功回顾了 {len(memories)} 条记忆:")
        for m in memories:
            print(f"   - [{m.get('strategy')}] {m.get('content')}")
            if "AGI系统必须具备真实的感知" in m.get('content', ''):
                found_test_memory = True
    else:
        print("⚠️ 未回顾到任何记忆 (可能是随机策略未选中或数据库为空)")

    if found_test_memory:
        print("✅ 成功验证: 意识引擎读取了刚刚插入的真实记忆!")
    
    # 停止系统
    await event_system.stop()
    print("\n" + "="*60)
    print("🎉 验证完成!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(verify_real_abilities())
