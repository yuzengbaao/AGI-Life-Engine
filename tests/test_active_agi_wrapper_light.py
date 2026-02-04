#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import pytest

from active_agi.multi_agent_system import Task, AgentType
from active_agi.motivation_system import MotivationSystem


def test_task_instantiation_assigned_to_ok():
    """Task 应支持 assigned_to 字段用于指派 Agent。"""
    t = Task(
        task_id="t1",
        description="demo",
        assigned_to=AgentType.EXECUTOR,
        priority=1,
    )
    assert t.assigned_to == AgentType.EXECUTOR
    assert t.priority == 1
    assert t.status == "pending"


@pytest.mark.asyncio
async def test_generate_motivated_actions_is_async():
    """验证 generate_motivated_actions 需要使用 await 调用，且返回列表。"""
    motivation = MotivationSystem(memory_system=None, llm_core=None)
    actions = await motivation.generate_motivated_actions()
    assert isinstance(actions, list)
    # 元素结构基本校验（可能为空，但类型应为列表）
    if actions:
        a0 = actions[0]
        assert isinstance(a0, dict)
        assert 'motivation' in a0
