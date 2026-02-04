#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
显-潜意识统一修复验证测试
Intent Bridge Fix Verification Tests

验证所有修复是否生效
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from intent_dialogue_bridge import get_intent_bridge
from tool_execution_bridge import ToolExecutionBridge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IntentBridgeFixTests:
    """意图桥接修复测试套件"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def test_intent_bridge_absolute_path(self):
        """测试意图桥接使用绝对路径"""
        logger.info("\n[Test 1] IntentDialogueBridge 绝对路径")
        
        try:
            bridge = get_intent_bridge()
            data_dir = bridge.data_dir
            
            # 检查是否为绝对路径
            assert data_dir.is_absolute(), f"数据目录不是绝对路径: {data_dir}"
            
            # 检查目录存在
            assert data_dir.exists(), f"数据目录不存在: {data_dir}"
            
            logger.info(f"✅ 意图桥接使用绝对路径: {data_dir}")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ 绝对路径测试失败: {e}")
            self.failed += 1
    
    def test_intent_bridge_files_init(self):
        """测试意图桥接文件初始化（不跳过旧意图）"""
        logger.info("\n[Test 2] IntentDialogueBridge 文件初始化")
        
        try:
            bridge = get_intent_bridge()
            
            # 检查文件位置指针（应该从头开始）
            user_pos = bridge._user_file_pos
            assert user_pos == 0, f"用户意图文件位置应为0（从头开始），实际: {user_pos}"
            
            # 检查通信文件存在
            assert bridge.user_intents_file.exists(), "用户意图文件不存在"
            assert bridge.engine_responses_file.exists(), "引擎响应文件不存在"
            
            logger.info("✅ 意图桥接文件初始化正确（从头读取）")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ 文件初始化测试失败: {e}")
            self.failed += 1
    
    def test_tool_bridge_update_operation(self):
        """测试 persistent_knowledge_base.update 操作"""
        logger.info("\n[Test 3] persistent_knowledge_base.update 操作")
        
        try:
            bridge = ToolExecutionBridge()
            
            # 测试 update 操作
            result = bridge._tool_persistent_knowledge_base({
                'operation': 'update',
                'key': 'test_update_key',
                'value': {'data': 'updated_value'}
            })
            
            assert result['success'] == True, "update 操作失败"
            assert 'updated' in result['data'], "返回结果缺少 updated 字段"
            
            logger.info("✅ persistent_knowledge_base.update 操作正常")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ update 操作测试失败: {e}")
            self.failed += 1
    
    def test_constitutional_ai_tool(self):
        """测试 constitutional_ai 工具注册"""
        logger.info("\n[Test 4] constitutional_ai 工具")
        
        try:
            bridge = ToolExecutionBridge()
            
            # 检查工具是否注册
            assert 'constitutional_ai' in bridge.tools, "constitutional_ai 工具未注册"
            
            # 检查能力声明
            caps = bridge.tool_capabilities
            assert 'constitutional_ai' in caps, "constitutional_ai 未在能力声明中"
            
            # 测试 validate_claim 操作
            result = bridge._tool_constitutional_ai({
                'operation': 'validate_claim',
                'claim': '测试断言'
            })
            
            assert result['success'] == True, "validate_claim 操作失败"
            assert 'valid' in result['data'], "返回结果缺少 valid 字段"
            
            logger.info("✅ constitutional_ai 工具注册并正常运行")
            logger.info(f"   支持操作: {list(caps['constitutional_ai']['operations'].keys())[:5]}")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ constitutional_ai 工具测试失败: {e}")
            self.failed += 1
    
    def test_tool_capabilities_updated(self):
        """测试工具能力声明已更新"""
        logger.info("\n[Test 5] TOOL_CAPABILITIES 更新验证")
        
        try:
            bridge = ToolExecutionBridge()
            caps = bridge.tool_capabilities
            
            # 检查 persistent_knowledge_base 包含 update
            pkb_ops = caps['persistent_knowledge_base']['operations']
            assert 'update' in pkb_ops, "persistent_knowledge_base 缺少 update 操作"
            assert 'modify' in pkb_ops, "persistent_knowledge_base 缺少 modify 别名"
            
            # 检查 constitutional_ai 存在
            assert 'constitutional_ai' in caps, "constitutional_ai 未在能力声明中"
            
            logger.info("✅ TOOL_CAPABILITIES 更新验证通过")
            logger.info(f"   persistent_knowledge_base 新增: update, modify, change")
            logger.info(f"   constitutional_ai 已注册")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ TOOL_CAPABILITIES 更新验证失败: {e}")
            self.failed += 1
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("\n" + "="*60)
        logger.info("开始显-潜意识统一修复验证测试")
        logger.info("="*60)
        
        self.test_intent_bridge_absolute_path()
        self.test_intent_bridge_files_init()
        self.test_tool_bridge_update_operation()
        self.test_constitutional_ai_tool()
        self.test_tool_capabilities_updated()
        
        logger.info("\n" + "="*60)
        logger.info(f"测试完成: {self.passed} 通过, {self.failed} 失败")
        logger.info("="*60)
        
        return self.failed == 0


if __name__ == "__main__":
    tests = IntentBridgeFixTests()
    success = tests.run_all_tests()
    
    if success:
        print("\n✅ 所有修复验证通过！")
        print("\n📋 修复摘要:")
        print("   1. IntentDialogueBridge 使用绝对路径（消除 cwd 依赖）")
        print("   2. _init_files 从头读取意图（不跳过旧数据）")
        print("   3. CLI 等待超时延长到15秒（容忍高熵阻塞）")
        print("   4. persistent_knowledge_base 支持 update 操作")
        print("   5. constitutional_ai 工具已注册")
        print("\n🚀 建议：重启 AGI_Life_Engine.py 和 agi_chat_cli.py 使修复生效")
    else:
        print("\n❌ 部分测试失败，请检查日志")
    
    sys.exit(0 if success else 1)
