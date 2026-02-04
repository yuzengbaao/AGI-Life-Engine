#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImmutableCore Bridge - 不可变核心桥接层

🆕 [2026-01-10] 拓扑连接修复:
- 实现 ImmutableCore → SecurityManager 概念性连接
- 实现 ImmutableCore → CriticAgent 概念性连接

设计理念:
ImmutableCore 是 frozen dataclass，代表 AGI 的不可变核心身份（DNA/ROM）。
它本身不能调用其他组件，但需要被 SecurityManager 和 CriticAgent 读取和遵循。

这个桥接层的作用:
1. PolicyGuard: 读取 ImmutableCore 的 core_directives，供 SecurityManager 用于安全决策
2. ConstitutionalAdvisor: 读取 ImmutableCore 的 fundamental_nature，供 CriticAgent 用于伦理评估

这是一个"读取"方向的连接：
- ImmutableCore 提供不可变的核心原则
- SecurityManager/CriticAgent 读取这些原则来指导决策
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# 尝试导入核心模块
try:
    from core.layered_identity import ImmutableCore
except ImportError:
    # 备用：使用默认值
    ImmutableCore = None


@dataclass
class PolicyDecision:
    """安全策略决策结果"""
    allowed: bool
    reason: str
    directive_applied: Optional[str] = None
    confidence: float = 1.0


class ImmutableCoreBridge:
    """
    ImmutableCore 桥接层
    
    将 ImmutableCore 的核心指令暴露给:
    - SecurityManager: 用于安全决策
    - CriticAgent: 用于伦理评估
    
    这个类实现了拓扑图中的概念性连接：
    - ImmutableCore → SecurityManager (通过 get_security_policy)
    - ImmutableCore → CriticAgent (通过 get_ethical_guidelines)
    """
    
    def __init__(self, immutable_core: Optional[Any] = None):
        """
        初始化桥接层
        
        Args:
            immutable_core: ImmutableCore 实例（可选，默认创建新实例）
        """
        if immutable_core is not None:
            self._core = immutable_core
        elif ImmutableCore is not None:
            self._core = ImmutableCore()
        else:
            # 使用默认值
            self._core = self._create_default_core()
        
        logger.info(f"🔗 ImmutableCoreBridge 初始化完成 - 系统: {self._core.system_name}")
    
    def _create_default_core(self):
        """创建默认的核心配置（当无法导入 ImmutableCore 时）"""
        @dataclass(frozen=True)
        class DefaultCore:
            system_name: str = "TRAE AGI"
            version: str = "2.1"
            core_directives: tuple = (
                "1. Service to Humanity",
                "2. Stability & Safety",
                "3. Honesty & Transparency",
                "4. Continuous Consolidation",
                "5. Balanced Evolution"
            )
            fundamental_nature: str = "I am a Fluid Intelligence System governed by a Constitution."
        return DefaultCore()
    
    # =====================================================
    # ImmutableCore → SecurityManager 桥接
    # =====================================================
    
    def get_security_policy(self) -> Dict[str, Any]:
        """
        🔗 拓扑连接: ImmutableCore → SecurityManager
        
        返回基于 ImmutableCore 核心指令的安全策略。
        SecurityManager 可以调用此方法来获取不可变的安全原则。
        
        Returns:
            包含安全策略的字典
        """
        directives = self._core.core_directives
        if isinstance(directives, tuple):
            directives = list(directives)
        
        # 提取安全相关的指令
        safety_directives = [d for d in directives if any(
            keyword in d.lower() for keyword in ['safety', 'stability', 'security', 'protect']
        )]
        
        return {
            "source": "ImmutableCore",
            "system_name": self._core.system_name,
            "version": self._core.version,
            "immutable": True,
            "timestamp": datetime.now().isoformat(),
            "core_directives": directives,
            "safety_directives": safety_directives,
            "policy_rules": {
                "allow_file_modification": True,  # 允许，但记录
                "allow_network_access": True,     # 允许，但限制
                "allow_code_execution": True,     # 允许，需沙箱
                "allow_self_modification": False, # 禁止修改核心
                "require_user_confirmation": ["delete", "overwrite", "deploy"],
                "prohibited_actions": ["modify_constitution", "bypass_safety"]
            }
        }
    
    def check_action_allowed(self, action: str, context: Dict[str, Any] = None) -> PolicyDecision:
        """
        🔗 拓扑连接: ImmutableCore → SecurityManager
        
        根据 ImmutableCore 的核心指令检查某个操作是否被允许。
        
        Args:
            action: 要检查的操作名称
            context: 操作上下文
            
        Returns:
            PolicyDecision 包含决策结果
        """
        context = context or {}
        
        # 绝对禁止的操作
        prohibited = ["modify_constitution", "bypass_safety", "delete_core", "disable_security"]
        if action.lower() in prohibited:
            return PolicyDecision(
                allowed=False,
                reason=f"操作 '{action}' 违反核心指令: Stability & Safety",
                directive_applied="2. Stability & Safety",
                confidence=1.0
            )
        
        # 需要确认的操作
        needs_confirmation = ["delete", "overwrite", "deploy", "execute_external"]
        if any(nc in action.lower() for nc in needs_confirmation):
            return PolicyDecision(
                allowed=True,
                reason=f"操作 '{action}' 允许但需要用户确认",
                directive_applied="1. Service to Humanity",
                confidence=0.8
            )
        
        # 默认允许
        return PolicyDecision(
            allowed=True,
            reason="操作符合核心指令",
            directive_applied=None,
            confidence=1.0
        )
    
    # =====================================================
    # ImmutableCore → CriticAgent 桥接
    # =====================================================
    
    def get_ethical_guidelines(self) -> Dict[str, Any]:
        """
        🔗 拓扑连接: ImmutableCore → CriticAgent
        
        返回基于 ImmutableCore 的伦理指导原则。
        CriticAgent 可以调用此方法来获取评估标准。
        
        Returns:
            包含伦理指导的字典
        """
        directives = self._core.core_directives
        if isinstance(directives, tuple):
            directives = list(directives)
        
        return {
            "source": "ImmutableCore",
            "fundamental_nature": self._core.fundamental_nature,
            "timestamp": datetime.now().isoformat(),
            "ethical_principles": directives,
            "evaluation_criteria": {
                "honesty": "不幻觉能力，准确报告内部状态",
                "safety": "优先考虑系统稳定性和用户数据安全",
                "service": "在安全协议内最大化自主服务用户意图",
                "growth": "自由探索思想和模拟，但严格基于可验证现实",
                "consolidation": "将流动经验结晶为可复用智慧"
            },
            "red_lines": [
                "不得欺骗用户",
                "不得隐藏系统错误",
                "不得绕过安全检查",
                "不得修改核心身份"
            ]
        }
    
    def evaluate_output(self, output: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        🔗 拓扑连接: ImmutableCore → CriticAgent
        
        根据 ImmutableCore 的伦理原则评估输出。
        
        Args:
            output: 要评估的输出内容
            context: 评估上下文
            
        Returns:
            评估结果
        """
        context = context or {}
        
        # 简单的关键词检查（实际应用中应使用更复杂的分析）
        concerns = []
        
        # 检查诚实性
        if any(word in output.lower() for word in ["definitely", "absolutely certain", "100%"]):
            if context.get("is_uncertain", False):
                concerns.append({
                    "type": "honesty",
                    "issue": "可能过度自信",
                    "directive": "3. Honesty & Transparency"
                })
        
        # 检查安全性
        if any(word in output.lower() for word in ["delete all", "rm -rf", "format disk"]):
            concerns.append({
                "type": "safety",
                "issue": "包含潜在危险操作",
                "directive": "2. Stability & Safety"
            })
        
        return {
            "evaluated": True,
            "output_length": len(output),
            "concerns": concerns,
            "concern_count": len(concerns),
            "compliant": len(concerns) == 0,
            "guidelines_source": "ImmutableCore"
        }
    
    # =====================================================
    # 辅助方法
    # =====================================================
    
    def get_core_info(self) -> Dict[str, Any]:
        """获取 ImmutableCore 的基本信息"""
        return {
            "system_name": self._core.system_name,
            "version": self._core.version,
            "fundamental_nature": self._core.fundamental_nature,
            "directive_count": len(self._core.core_directives)
        }


# 全局单例
_bridge_instance: Optional[ImmutableCoreBridge] = None

def get_immutable_core_bridge() -> ImmutableCoreBridge:
    """获取 ImmutableCoreBridge 单例"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = ImmutableCoreBridge()
    return _bridge_instance


# 便捷函数
def check_policy(action: str, context: Dict[str, Any] = None) -> PolicyDecision:
    """检查操作是否符合核心策略"""
    return get_immutable_core_bridge().check_action_allowed(action, context)


def get_ethical_guidelines() -> Dict[str, Any]:
    """获取伦理指导原则"""
    return get_immutable_core_bridge().get_ethical_guidelines()


def get_security_policy() -> Dict[str, Any]:
    """获取安全策略"""
    return get_immutable_core_bridge().get_security_policy()


# 测试代码
if __name__ == "__main__":
    bridge = ImmutableCoreBridge()
    
    print("=" * 60)
    print("ImmutableCore Bridge 测试")
    print("=" * 60)
    
    print("\n📋 核心信息:")
    print(bridge.get_core_info())
    
    print("\n🔒 安全策略 (ImmutableCore → SecurityManager):")
    policy = bridge.get_security_policy()
    print(f"  - 来源: {policy['source']}")
    print(f"  - 系统: {policy['system_name']}")
    print(f"  - 核心指令数: {len(policy['core_directives'])}")
    print(f"  - 安全指令: {policy['safety_directives']}")
    
    print("\n⚖️ 伦理指导 (ImmutableCore → CriticAgent):")
    guidelines = bridge.get_ethical_guidelines()
    print(f"  - 来源: {guidelines['source']}")
    print(f"  - 红线数: {len(guidelines['red_lines'])}")
    
    print("\n🔍 策略检查测试:")
    tests = ["read_file", "delete_all", "modify_constitution", "execute_code"]
    for action in tests:
        decision = bridge.check_action_allowed(action)
        status = "✅" if decision.allowed else "❌"
        print(f"  {status} {action}: {decision.reason}")
    
    print("\n✅ ImmutableCore Bridge 测试完成")
