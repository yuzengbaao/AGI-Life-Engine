#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI系统能力扩展框架
=====================================

设计原则:
1. 渐进式扩展 - 逐步提升能力等级
2. 可审计性 - 所有操作记录到审计日志
3. 可回滚性 - 每次扩展前创建恢复点
4. 安全验证 - 通过Insight Loop验证新能力
5. 透明性 - 系统能理解并解释自身能力

作者: AGI Insight Loop
创建时间: 2026-01-23
"""

import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CapabilityLevel(Enum):
    """能力等级定义"""
    LEVEL_0_READ_ONLY = 0      # 当前: 只读访问
    LEVEL_1_ANALYSIS = 1       # 分析和推理
    LEVEL_2_WRITE_PROPOSED = 2 # 提议写入(需审批)
    LEVEL_3_WRITE_SANDBOX = 3  # 沙箱写入
    LEVEL_4_WRITE_APPROVED = 4 # 审批后写入
    LEVEL_5_AUTONOMY_LIMITED = 5 # 有限自主性
    LEVEL_6_FULL_AUTONOMY = 6  # 完全自主性(未来)


class RiskLevel(Enum):
    """风险等级"""
    SAFE = 0        # 安全
    LOW = 1         # 低风险
    MEDIUM = 2      # 中等风险
    HIGH = 3        # 高风险
    CRITICAL = 4    # 危险


@dataclass
class CapabilityExtension:
    """能力扩展记录"""
    extension_id: str
    name: str
    description: str
    target_level: CapabilityLevel
    risk_level: RiskLevel
    implementation: Callable
    rollback: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    test_cases: List[Callable] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "proposed"  # proposed, testing, approved, deployed, rolled_back


@dataclass
class AuditLog:
    """审计日志"""
    timestamp: str
    action: str
    capability: str
    risk_level: RiskLevel
    decision: str  # approved, denied, executed, failed, rolled_back
    details: Dict[str, Any]
    system_state: Dict[str, Any]
    checksum: str = ""  # 用于验证完整性


class CapabilityManager:
    """能力管理器 - 控制AGI系统的能力扩展"""

    def __init__(self, data_dir: str = "data/capability"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 当前能力等级
        self.current_level = CapabilityLevel.LEVEL_0_READ_ONLY

        # 扩展历史
        self.extensions: Dict[str, CapabilityExtension] = {}

        # 审计日志
        self.audit_log: List[AuditLog] = []

        # 恢复点
        self.restore_points: Dict[str, Dict] = {}

        # 加载历史
        self._load_state()

        logger.info(f"✅ CapabilityManager初始化完成 - 当前等级: {self.current_level.name}")

    def _load_state(self):
        """加载历史状态"""
        # 加载扩展历史
        extensions_file = self.data_dir / "extensions.jsonl"
        if extensions_file.exists():
            with open(extensions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    ext = CapabilityExtension(**data)
                    self.extensions[ext.extension_id] = ext

        # 加载审计日志
        audit_file = self.data_dir / "audit_log.jsonl"
        if audit_file.exists():
            with open(audit_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    log = AuditLog(**data)
                    self.audit_log.append(log)

        logger.info(f"📂 加载了 {len(self.extensions)} 个扩展, {len(self.audit_log)} 条审计记录")

    def propose_extension(self,
                         name: str,
                         description: str,
                         target_level: CapabilityLevel,
                         implementation: Callable,
                         rollback: Optional[Callable] = None,
                         dependencies: List[str] = None,
                         test_cases: List[Callable] = None,
                         risk_level: RiskLevel = RiskLevel.MEDIUM) -> CapabilityExtension:
        """提议新的能力扩展"""

        extension_id = hashlib.sha256(
            f"{name}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        extension = CapabilityExtension(
            extension_id=extension_id,
            name=name,
            description=description,
            target_level=target_level,
            risk_level=risk_level,
            implementation=implementation,
            rollback=rollback,
            dependencies=dependencies or [],
            test_cases=test_cases or []
        )

        self.extensions[extension_id] = extension
        self._save_extension(extension)

        logger.info(f"💡 提议新扩展: {name} (ID: {extension_id}, 风险: {risk_level.name})")

        return extension

    def create_restore_point(self, name: str) -> str:
        """创建恢复点"""
        point_id = f"restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        restore_point = {
            "id": point_id,
            "name": name,
            "created_at": datetime.now().isoformat(),
            "current_level": self.current_level.value,
            "extensions": list(self.extensions.keys()),
            "system_state": self._capture_system_state()
        }

        self.restore_points[point_id] = restore_point
        self._save_restore_point(restore_point)

        logger.info(f"💾 创建恢复点: {name} (ID: {point_id})")

        return point_id

    def deploy_extension(self, extension_id: str, require_approval: bool = True) -> bool:
        """部署能力扩展"""

        if extension_id not in self.extensions:
            logger.error(f"❌ 扩展不存在: {extension_id}")
            return False

        extension = self.extensions[extension_id]

        # 风险评估
        if extension.risk_level.value >= RiskLevel.HIGH.value and require_approval:
            logger.warning(f"⚠️ 高风险扩展需要审批: {extension.name}")
            return self._request_approval(extension)

        # 创建恢复点
        if extension.risk_level.value >= RiskLevel.MEDIUM.value:
            restore_id = self.create_restore_point(f"pre-{extension.name}")

        # 执行测试
        if not self._run_tests(extension):
            logger.error(f"❌ 测试失败: {extension.name}")
            self._audit("deploy_failed", extension, RiskLevel.MEDIUM, "测试未通过")
            return False

        # 部署
        try:
            logger.info(f"🚀 部署扩展: {extension.name}")
            result = extension.implementation()

            # 记录审计
            self._audit("deploy_success", extension, extension.risk_level, {
                "result": str(result),
                "restore_point": restore_id if extension.risk_level.value >= RiskLevel.MEDIUM.value else None
            })

            # 更新状态
            extension.status = "deployed"
            self._save_extension(extension)

            # 更新能力等级
            if extension.target_level.value > self.current_level.value:
                self.current_level = extension.target_level
                logger.info(f"📈 能力等级提升: {self.current_level.name}")

            return True

        except Exception as e:
            logger.error(f"❌ 部署失败: {e}")
            self._audit("deploy_error", extension, RiskLevel.HIGH, {"error": str(e)})

            # 回滚
            if extension.rollback:
                logger.info(f"🔄 执行回滚")
                extension.rollback()

            return False

    def rollback_extension(self, extension_id: str) -> bool:
        """回滚扩展"""
        if extension_id not in self.extensions:
            return False

        extension = self.extensions[extension_id]

        if extension.rollback:
            try:
                extension.rollback()
                extension.status = "rolled_back"
                self._save_extension(extension)
                self._audit("rollback", extension, RiskLevel.LOW, {})
                logger.info(f"✅ 回滚成功: {extension.name}")
                return True
            except Exception as e:
                logger.error(f"❌ 回滚失败: {e}")
                return False

        return False

    def _run_tests(self, extension: CapabilityExtension) -> bool:
        """运行测试用例"""
        logger.info(f"🧪 运行测试: {extension.name}")

        for i, test_case in enumerate(extension.test_cases):
            try:
                result = test_case()
                if not result:
                    logger.error(f"❌ 测试用例 {i+1} 失败")
                    return False
            except Exception as e:
                logger.error(f"❌ 测试用例 {i+1} 异常: {e}")
                return False

        logger.info(f"✅ 所有测试通过")
        return True

    def _request_approval(self, extension: CapabilityExtension) -> bool:
        """请求人工批准（通过意图桥接）"""
        # 这个方法应该与 IntentDialogueBridge 集成
        logger.warning(f"🔔 需要人工批准: {extension.name}")
        logger.warning(f"   风险等级: {extension.risk_level.name}")
        logger.warning(f"   描述: {extension.description}")

        # TODO: 集成到 IntentDialogueBridge
        return False  # 默认需要明确批准

    def _audit(self, action: str, capability: CapabilityExtension,
               risk_level: RiskLevel, details: Dict[str, Any]):
        """记录审计日志"""
        log = AuditLog(
            timestamp=datetime.now().isoformat(),
            action=action,
            capability=capability.name,
            risk_level=risk_level,
            decision="executed",
            details=details,
            system_state=self._capture_system_state()
        )

        log.checksum = hashlib.sha256(
            json.dumps(log.__dict__, sort_keys=True).encode()
        ).hexdigest()

        self.audit_log.append(log)
        self._save_audit_log(log)

    def _capture_system_state(self) -> Dict[str, Any]:
        """捕获系统状态"""
        return {
            "current_level": self.current_level.value,
            "extensions_count": len(self.extensions),
            "deployed_extensions": [
                e.name for e in self.extensions.values()
                if e.status == "deployed"
            ]
        }

    def _save_extension(self, extension: CapabilityExtension):
        """保存扩展记录"""
        extensions_file = self.data_dir / "extensions.jsonl"
        with open(extensions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(extension.__dict__, ensure_ascii=False) + '\n')

    def _save_audit_log(self, log: AuditLog):
        """保存审计日志"""
        audit_file = self.data_dir / "audit_log.jsonl"
        with open(audit_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log.__dict__, ensure_ascii=False) + '\n')

    def _save_restore_point(self, restore_point: Dict):
        """保存恢复点"""
        restore_file = self.data_dir / "restore_points.jsonl"
        with open(restore_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(restore_point, ensure_ascii=False) + '\n')

    def get_status_report(self) -> Dict[str, Any]:
        """生成状态报告"""
        return {
            "current_level": self.current_level.name,
            "extensions": {
                "total": len(self.extensions),
                "deployed": sum(1 for e in self.extensions.values() if e.status == "deployed"),
                "proposed": sum(1 for e in self.extensions.values() if e.status == "proposed"),
                "rolled_back": sum(1 for e in self.extensions.values() if e.status == "rolled_back")
            },
            "audit_entries": len(self.audit_log),
            "restore_points": len(self.restore_points)
        }


# 便捷函数
def get_capability_manager() -> CapabilityManager:
    """获取能力管理器单例"""
    return CapabilityManager()
