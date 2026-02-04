#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 AGI_Life_Engine 中的自修改工具集成
"""
import sys
import asyncio
sys.path.insert(0, 'd:/TRAE_PROJECT/AGI')

print("=" * 60)
print("AGI_Life_Engine Self-Modification Integration Test")
print("=" * 60)

# 模拟一个简化的 AGI 系统来测试工具桥接
from tool_execution_bridge import ToolExecutionBridge

class MockAGISystem:
    """模拟 AGI 系统"""
    def __init__(self):
        self.self_modifier = None
        self.component_coordinator = None
        
        # 初始化 self_modifier
        from core.self_modifying_engine import SelfModifyingEngine
        self.self_modifier = SelfModifyingEngine()

# 创建模拟系统
mock_agi = MockAGISystem()

# 创建桥接层，传入 AGI 系统
bridge = ToolExecutionBridge(agi_system=mock_agi)

print("\n1. 验证 AGI 系统引用:")
print(f"   bridge.agi_system: {bridge.agi_system is not None}")
print(f"   bridge.agi_system.self_modifier: {hasattr(bridge.agi_system, 'self_modifier') and bridge.agi_system.self_modifier is not None}")

print("\n2. 测试自修改工具通过 AGI 系统访问:")
handler = bridge.tools.get('self_modification')
if handler:
    result = handler({'operation': 'status'})
    print(f"   成功: {result.get('success')}")
    engine_type = result.get('data', {}).get('engine_type', 'N/A')
    print(f"   引擎类型: {engine_type}")
else:
    print("   ❌ 工具未注册")

print("\n3. 测试完整回归流程:")
if handler:
    result = handler({
        'operation': 'regression_test',
        'target_module': 'core.math_utils'
    })
    print(f"   成功: {result.get('success')}")
    if result.get('success'):
        data = result.get('data', {})
        print(f"   propose: {data.get('propose')}")
        print(f"   sandbox: {data.get('sandbox')}")
        print(f"   apply: {data.get('apply')}")
        print(f"   rollback: {data.get('rollback')}")
        print(f"   总体: {data.get('overall')}")
    else:
        print(f"   错误: {result.get('error')}")

print("\n4. 测试 TOOL_CALL 解析和执行:")
async def test_tool_call():
    test_response = '''
    我将检查自修改引擎的状态。
    
    TOOL_CALL: self_modification(operation="status")
    
    状态检查完成。
    '''
    result = await bridge.process_response(test_response)
    print(f"   has_tool_calls: {result.get('has_tool_calls')}")
    if result.get('tool_results'):
        for tr in result['tool_results']:
            print(f"   工具: {tr.get('tool_name')}")
            print(f"   结果: {tr.get('result', {}).get('success')}")
    return result

asyncio.run(test_tool_call())

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
