#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
潜意识桥接修复验证测试
Subconscious Bridge Repair Verification Tests

验证所有新增的潜意识操作是否正常工作
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tool_execution_bridge import ToolExecutionBridge
from subconscious_config import SubconsciousConfig, get_subconscious_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SubconsciousBridgeTests:
    """潜意识桥接测试套件"""
    
    def __init__(self):
        self.bridge = ToolExecutionBridge()
        self.config = get_subconscious_config()
        self.passed = 0
        self.failed = 0
    
    def test_persistent_knowledge_base_query(self):
        """测试 persistent_knowledge_base.query（别名）"""
        logger.info("\n[Test 1] persistent_knowledge_base.query")
        
        try:
            # 先存储一些数据
            self.bridge._tool_persistent_knowledge_base({
                'operation': 'put',
                'key': 'test_concept',
                'value': {'description': 'test data for query'}
            })
            
            # 使用 query 别名搜索
            result = self.bridge._tool_persistent_knowledge_base({
                'operation': 'query',
                'query': 'test_concept'
            })
            
            assert result['success'] == True, "查询操作失败"
            assert 'data' in result, "缺少数据字段"
            assert result['data']['count'] >= 1, "未找到测试数据"
            
            logger.info("✅ persistent_knowledge_base.query 测试通过")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ persistent_knowledge_base.query 测试失败: {e}")
            self.failed += 1
    
    def test_persistent_knowledge_base_reorganize(self):
        """测试 persistent_knowledge_base.reorganize_by_association"""
        logger.info("\n[Test 2] persistent_knowledge_base.reorganize_by_association")
        
        try:
            result = self.bridge._tool_persistent_knowledge_base({
                'operation': 'reorganize_by_association'
            })
            
            assert result['success'] == True, "重组操作失败"
            assert 'data' in result, "缺少数据字段"
            assert 'message' in result['data'], "缺少消息字段"
            
            logger.info("✅ persistent_knowledge_base.reorganize 测试通过")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ persistent_knowledge_base.reorganize 测试失败: {e}")
            self.failed += 1
    
    def test_biological_topology_sample_activation(self):
        """测试 biological_topology.sample_activation"""
        logger.info("\n[Test 3] biological_topology.sample_activation")
        
        try:
            # 先创建一些关联
            self.bridge._tool_biological_topology({
                'operation': 'associate',
                'concept_a': 'concept_1',
                'concept_b': 'concept_2',
                'strength': 0.9
            })
            
            # 采样激活节点
            result = self.bridge._tool_biological_topology({
                'operation': 'sample_activation',
                'context_hash': 'test_context',
                'threshold': 0.8
            })
            
            assert result['success'] == True, "采样操作失败"
            assert 'data' in result, "缺少数据字段"
            assert 'sampled_nodes' in result['data'], "缺少采样节点"
            assert 'interrupt_flag' in result['data'], "缺少中断标志"
            
            logger.info(f"✅ biological_topology.sample_activation 测试通过")
            logger.info(f"   采样节点数: {len(result['data']['sampled_nodes'])}")
            logger.info(f"   中断标志: {result['data']['interrupt_flag']}")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ biological_topology.sample_activation 测试失败: {e}")
            self.failed += 1
    
    def test_biological_topology_strengthen_paths(self):
        """测试 biological_topology.strengthen_paths"""
        logger.info("\n[Test 4] biological_topology.strengthen_paths")
        
        try:
            # 创建关联
            self.bridge._tool_biological_topology({
                'operation': 'associate',
                'concept_a': 'concept_A',
                'concept_b': 'concept_B',
                'strength': 0.5
            })
            
            # 强化路径
            result = self.bridge._tool_biological_topology({
                'operation': 'strengthen_paths',
                'concepts': ['concept_A'],
                'factor': 0.2
            })
            
            assert result['success'] == True, "强化操作失败"
            assert 'data' in result, "缺少数据字段"
            assert 'strengthened_count' in result['data'], "缺少强化计数"
            
            logger.info(f"✅ biological_topology.strengthen_paths 测试通过")
            logger.info(f"   强化路径数: {result['data']['strengthened_count']}")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ biological_topology.strengthen_paths 测试失败: {e}")
            self.failed += 1
    
    def test_curiosity_explore_launch_autonomous_inquiry(self):
        """测试 curiosity_explore.launch_autonomous_inquiry"""
        logger.info("\n[Test 5] curiosity_explore.launch_autonomous_inquiry")
        
        try:
            result = self.bridge._tool_curiosity_explore({
                'operation': 'launch_autonomous_inquiry',
                'depth': 5,
                'timeout': 30.0,
                'focus': '测试探索领域'
            })
            
            assert result['success'] == True, "自主探索启动失败"
            assert 'data' in result, "缺少数据字段"
            assert 'inquiry_id' in result['data'], "缺少探索ID"
            assert result['data']['status'] == 'launched', "探索状态不正确"
            
            logger.info(f"✅ curiosity_explore.launch_autonomous_inquiry 测试通过")
            logger.info(f"   探索ID: {result['data']['inquiry_id']}")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ curiosity_explore.launch_autonomous_inquiry 测试失败: {e}")
            self.failed += 1
    
    def test_tool_capabilities_updated(self):
        """测试工具能力声明是否已更新"""
        logger.info("\n[Test 6] TOOL_CAPABILITIES 更新验证")
        
        try:
            caps = self.bridge.tool_capabilities
            
            # 检查 persistent_knowledge_base
            pkb_ops = caps['persistent_knowledge_base']['operations']
            assert 'query' in pkb_ops, "persistent_knowledge_base 缺少 query 操作"
            assert 'reorganize_by_association' in pkb_ops, "persistent_knowledge_base 缺少 reorganize_by_association 操作"
            
            # 检查 biological_topology
            bt_ops = caps['biological_topology']['operations']
            assert 'sample_activation' in bt_ops, "biological_topology 缺少 sample_activation 操作"
            assert 'strengthen_paths' in bt_ops, "biological_topology 缺少 strengthen_paths 操作"
            
            # 检查 curiosity_explore
            ce_ops = caps['curiosity_explore']['operations']
            assert 'launch_autonomous_inquiry' in ce_ops, "curiosity_explore 缺少 launch_autonomous_inquiry 操作"
            
            logger.info("✅ TOOL_CAPABILITIES 更新验证通过")
            logger.info(f"   persistent_knowledge_base 操作数: {len(pkb_ops)}")
            logger.info(f"   biological_topology 操作数: {len(bt_ops)}")
            logger.info(f"   curiosity_explore 操作数: {len(ce_ops)}")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ TOOL_CAPABILITIES 更新验证失败: {e}")
            self.failed += 1
    
    def test_subconscious_config(self):
        """测试潜意识配置开关"""
        logger.info("\n[Test 7] SubconsciousConfig 功能测试")
        
        try:
            # 测试功能检查
            assert self.config.should_allow_memory_sampling() == True, "记忆采样应默认启用"
            assert self.config.should_allow_autonomous_inquiry() == True, "自主探索应默认启用"
            assert self.config.should_allow_preemption() == False, "任务抢占应默认禁用"
            
            # 测试配置导出
            config_dict = self.config.to_dict()
            assert 'feature_flags' in config_dict, "配置缺少功能标志"
            assert 'protection_params' in config_dict, "配置缺少保护参数"
            
            logger.info("✅ SubconsciousConfig 功能测试通过")
            logger.info(f"   {self.config.get_status_summary()}")
            self.passed += 1
        except Exception as e:
            logger.error(f"❌ SubconsciousConfig 功能测试失败: {e}")
            self.failed += 1
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("\n" + "="*60)
        logger.info("开始潜意识桥接修复验证测试")
        logger.info("="*60)
        
        self.test_persistent_knowledge_base_query()
        self.test_persistent_knowledge_base_reorganize()
        self.test_biological_topology_sample_activation()
        self.test_biological_topology_strengthen_paths()
        self.test_curiosity_explore_launch_autonomous_inquiry()
        self.test_tool_capabilities_updated()
        self.test_subconscious_config()
        
        logger.info("\n" + "="*60)
        logger.info(f"测试完成: {self.passed} 通过, {self.failed} 失败")
        logger.info("="*60)
        
        return self.failed == 0


if __name__ == "__main__":
    tests = SubconsciousBridgeTests()
    success = tests.run_all_tests()
    
    sys.exit(0 if success else 1)
