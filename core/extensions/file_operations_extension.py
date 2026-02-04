#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI系统能力扩展 - 文件操作模块
=====================================

功能:
1. 安全的文件写入
2. 审计追踪
3. 沙箱验证
4. 自动回滚

作者: AGI Capability Framework
创建时间: 2026-01-23
"""

import os
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from core.capability_framework import CapabilityManager, CapabilityLevel, RiskLevel

logger = logging.getLogger(__name__)


class SecureFileOperations:
    """安全的文件操作类"""

    def __init__(self,
                 allowed_paths: List[str] = None,
                 audit_log_path: str = "data/capability/file_operations.log"):
        self.allowed_paths = [Path(p).resolve() for p in (allowed_paths or ["D:/TRAE_PROJECT/AGI"])]
        self.audit_log_path = Path(audit_log_path)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        # 操作历史（用于回滚）
        self.operation_history: List[Dict] = []

        logger.info(f"✅ SecureFileOperations初始化 - 允许路径: {len(self.allowed_paths)}个")

    def _is_path_allowed(self, path: Path) -> bool:
        """检查路径是否在允许范围内"""
        try:
            resolved = path.resolve()
            for allowed in self.allowed_paths:
                # 检查是否在允许的路径或其子目录下
                try:
                    resolved.relative_to(allowed)
                    return True
                except ValueError:
                    continue
            return False
        except Exception as e:
            logger.error(f"路径检查失败: {e}")
            return False

    def _assess_risk(self, path: Path, operation: str) -> RiskLevel:
        """评估操作风险"""
        # 检查文件扩展名
        dangerous_extensions = ['.exe', '.bat', '.sh', '.cmd', '.scr']
        if path.suffix.lower() in dangerous_extensions:
            return RiskLevel.CRITICAL

        # 检查系统目录
        system_keywords = ['system32', 'windows', 'program files']
        if any(kw in str(path).lower() for kw in system_keywords):
            return RiskLevel.HIGH

        # 检查是否覆盖核心文件
        core_files = ['agi_chat_cli.py', 'AGI_Life_Engine.py', 'intent_dialogue_bridge.py']
        if path.name in core_files and operation in ['write', 'delete']:
            return RiskLevel.HIGH

        return RiskLevel.MEDIUM

    def _create_backup(self, path: Path) -> Optional[Path]:
        """创建备份"""
        if not path.exists():
            return None

        backup_dir = path.parent / ".backups"
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{path.name}.{timestamp}.bak"

        try:
            shutil.copy2(path, backup_path)
            logger.info(f"💾 创建备份: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"❌ 备份失败: {e}")
            return None

    def write_file(self,
                   path: str,
                   content: str,
                   create_backup: bool = True,
                   require_approval: bool = True) -> Dict[str, Any]:
        """
        安全写入文件

        Args:
            path: 文件路径
            content: 文件内容
            create_backup: 是否创建备份
            require_approval: 是否需要审批

        Returns:
            操作结果字典
        """

        target_path = Path(path).resolve()

        # 1. 路径检查
        if not self._is_path_allowed(target_path):
            result = {
                "success": False,
                "error": "路径不在允许范围内",
                "path": str(target_path)
            }
            self._audit_log("write_denied", target_path, result)
            return result

        # 2. 风险评估
        risk_level = self._assess_risk(target_path, "write")

        if risk_level.value >= RiskLevel.HIGH.value and require_approval:
            result = {
                "success": False,
                "error": f"高风险操作需要审批: {risk_level.name}",
                "path": str(target_path),
                "risk_level": risk_level.name,
                "requires_approval": True
            }
            self._audit_log("write_approval_required", target_path, result)
            return result

        # 3. 创建备份
        backup_path = None
        if create_backup and target_path.exists():
            backup_path = self._create_backup(target_path)
            if not backup_path:
                return {
                    "success": False,
                    "error": "备份创建失败",
                    "path": str(target_path)
                }

        # 4. 执行写入
        try:
            # 确保目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)

            # 写入内容
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # 计算校验和
            checksum = hashlib.sha256(content.encode()).hexdigest()

            result = {
                "success": True,
                "path": str(target_path),
                "size": len(content),
                "checksum": checksum,
                "backup": str(backup_path) if backup_path else None,
                "timestamp": datetime.now().isoformat()
            }

            # 记录操作
            self.operation_history.append({
                "operation": "write",
                "path": str(target_path),
                "backup": str(backup_path) if backup_path else None,
                "checksum": checksum,
                "timestamp": datetime.now().isoformat()
            })

            self._audit_log("write_success", target_path, result)
            logger.info(f"✅ 文件写入成功: {target_path}")

            return result

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "path": str(target_path)
            }

            self._audit_log("write_error", target_path, error_result)
            logger.error(f"❌ 文件写入失败: {e}")

            return error_result

    def read_file(self, path: str) -> Dict[str, Any]:
        """读取文件（补充现有能力）"""
        target_path = Path(path).resolve()

        if not self._is_path_allowed(target_path):
            return {
                "success": False,
                "error": "路径不在允许范围内"
            }

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                "success": True,
                "path": str(target_path),
                "content": content,
                "size": len(content)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def delete_file(self,
                    path: str,
                    create_backup: bool = True,
                    require_approval: bool = True) -> Dict[str, Any]:
        """删除文件（高风险操作）"""
        target_path = Path(path).resolve()

        if not self._is_path_allowed(target_path):
            return {
                "success": False,
                "error": "路径不在允许范围内"
            }

        # 删除总是需要审批
        if require_approval:
            return {
                "success": False,
                "error": "删除操作需要明确审批",
                "requires_approval": True
            }

        # 创建备份
        backup_path = None
        if create_backup and target_path.exists():
            backup_path = self._create_backup(target_path)

        try:
            target_path.unlink()

            result = {
                "success": True,
                "path": str(target_path),
                "backup": str(backup_path) if backup_path else None
            }

            self._audit_log("delete_success", target_path, result)
            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def rollback_last_operation(self) -> bool:
        """回滚最后一次操作"""
        if not self.operation_history:
            logger.warning("⚠️ 没有可回滚的操作")
            return False

        last_op = self.operation_history[-1]

        if last_op["operation"] == "write":
            backup_path = last_op.get("backup")
            if backup_path:
                try:
                    backup = Path(backup_path)
                    if backup.exists():
                        target = Path(last_op["path"])
                        shutil.copy2(backup, target)
                        logger.info(f"✅ 回滚成功: {last_op['path']}")
                        return True
                except Exception as e:
                    logger.error(f"❌ 回滚失败: {e}")
                    return False

        return False

    def _audit_log(self, action: str, path: Path, details: Dict):
        """记录审计日志"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "path": str(path),
            "details": details
        }

        with open(self.audit_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')


# 创建全局实例
_file_ops_instance = None

def get_secure_file_operations() -> SecureFileOperations:
    """获取文件操作实例"""
    global _file_ops_instance
    if _file_ops_instance is None:
        _file_ops_instance = SecureFileOperations()
    return _file_ops_instance
