#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试自修改工具是否正确注册和工作"""
import sys
sys.path.insert(0, 'd:/TRAE_PROJECT/AGI')

from tool_execution_bridge import ToolExecutionBridge

# 创建桥接层实例
bridge = ToolExecutionBridge()

print("=" * 60)
print("Self-Modification Tool Integration Test")
print("=" * 60)

# 检查工具是否注册
print("\n1. 检查工具注册状态:")
for tool_name in ['self_modification', 'self_modify', 'self_modifier', 'code_patch']:
    registered = tool_name in bridge.tools
    print(f"   {tool_name}: {'✅ 已注册' if registered else '❌ 未注册'}")

# 检查工具能力元数据
print("\n2. 检查工具能力元数据:")
if 'self_modification' in bridge.tool_capabilities:
    cap = bridge.tool_capabilities['self_modification']
    print(f"   描述: {cap.get('description', 'N/A')}")
    print(f"   操作数: {len(cap.get('operations', {}))}")
    print(f"   操作列表: {list(cap.get('operations', {}).keys())}")
else:
    print("   ❌ 未找到self_modification能力元数据")

# 测试工具调用
print("\n3. 测试工具调用 - status:")
handler = bridge.tools.get('self_modification')
if handler:
    result = handler({'operation': 'status'})
    print(f"   成功: {result.get('success')}")
    print(f"   数据: {result.get('data', {})}")
else:
    print("   ❌ 未找到处理器")

# 测试 analyze 操作
print("\n4. 测试工具调用 - analyze:")
if handler:
    result = handler({'operation': 'analyze', 'target_module': 'core.math_utils'})
    print(f"   成功: {result.get('success')}")
    if result.get('success'):
        print(f"   分析结果: {result.get('data', {}).get('analysis', {})}")
    else:
        print(f"   错误: {result.get('error')}")

# 测试 regression_test 操作
print("\n5. 测试工具调用 - regression_test:")
if handler:
    result = handler({'operation': 'regression_test', 'target_module': 'core.math_utils'})
    print(f"   成功: {result.get('success')}")
    if result.get('success'):
        data = result.get('data', {})
        print(f"   结果: propose={data.get('propose')}, sandbox={data.get('sandbox')}, "
              f"apply={data.get('apply')}, rollback={data.get('rollback')}")
        print(f"   总体: {data.get('overall')}")
    else:
        print(f"   错误: {result.get('error')}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
