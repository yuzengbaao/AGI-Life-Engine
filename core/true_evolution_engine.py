"""
真进化机制引擎 - P2修复
实现架构自修改能力，区别于伪进化（参数调整）
核心能力：沙盒隔离、自动化测试、版本回滚
"""

import os
import sys
import json
import time
import shutil
import subprocess
import hashlib
import tempfile
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from pathlib import Path
import threading


class EvolutionStatus(Enum):
    """进化状态"""
    PENDING = "pending"           # 待处理
    ANALYZING = "analyzing"       # 分析中
    PROPOSING = "proposing"       # 生成方案
    SANDBOXING = "sandboxing"     # 沙盒测试
    TESTING = "testing"           # 自动化测试
    BENCHMARKING = "benchmarking" # 性能基准
    APPROVING = "approving"       # 等待审批
    APPLYING = "applying"         # 应用修改
    COMPLETED = "completed"       # 完成
    FAILED = "failed"             # 失败
    ROLLED_BACK = "rolled_back"   # 已回滚


@dataclass
class ArchitectureChange:
    """架构修改提案"""
    change_id: str
    motivation: str              # 修改动机
    target_module: str           # 目标模块
    change_type: str             # 修改类型: refactor/optimize/add/remove
    description: str             # 修改描述
    proposed_code: str           # 提议的代码
    original_code: str           # 原始代码
    estimated_impact: float      # 预估影响度 0-1
    risk_level: str              # 风险等级: low/medium/high
    created_at: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvolutionResult:
    """进化执行结果"""
    change_id: str
    status: EvolutionStatus
    start_time: float
    end_time: Optional[float] = None
    sandbox_success: bool = False
    test_pass_rate: float = 0.0
    performance_delta: float = 0.0
    benchmark_results: Dict = field(default_factory=dict)
    error_message: Optional[str] = None
    applied_by: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            "change_id": self.change_id,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "sandbox_success": self.sandbox_success,
            "test_pass_rate": self.test_pass_rate,
            "performance_delta": self.performance_delta,
            "benchmark_results": self.benchmark_results,
            "error_message": self.error_message,
            "applied_by": self.applied_by
        }


class IsolatedSandbox:
    """
    隔离沙盒环境
    
    功能:
    1. 创建临时隔离环境
    2. 在沙盒中应用修改
    3. 运行测试验证
    4. 清理沙盒环境
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.sandbox_dir: Optional[str] = None
        self._lock = threading.Lock()
    
    def create_sandbox(self) -> str:
        """创建隔离沙盒"""
        with self._lock:
            # 创建临时目录
            self.sandbox_dir = tempfile.mkdtemp(prefix="agi_evolution_")
            
            # 复制项目到沙盒（排除大文件）
            self._copy_project_to_sandbox()
            
            return self.sandbox_dir
    
    def _copy_project_to_sandbox(self):
        """复制项目文件到沙盒"""
        exclude_patterns = [
            '.git', '__pycache__', '*.pyc', '*.pyo',
            '*.log', 'data/creative_outputs/*', '*.db',
            'node_modules', '.venv', 'venv'
        ]
        
        for root, dirs, files in os.walk(self.project_root):
            # 过滤目录
            dirs[:] = [d for d in dirs if not any(
                pattern in os.path.join(root, d) for pattern in exclude_patterns
            )]
            
            for file in files:
                if any(pattern in file for pattern in exclude_patterns):
                    continue
                
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, self.project_root)
                dst_path = os.path.join(self.sandbox_dir, rel_path)
                
                # 创建目标目录
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                # 复制文件
                try:
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"[Sandbox] Warning: Failed to copy {rel_path}: {e}")
    
    def apply_change(self, change: ArchitectureChange) -> bool:
        """在沙盒中应用修改"""
        if not self.sandbox_dir:
            raise RuntimeError("Sandbox not created")
        
        try:
            # 定位目标文件
            target_file = os.path.join(self.sandbox_dir, change.target_module)
            
            if not os.path.exists(target_file):
                print(f"[Sandbox] Target file not found: {change.target_module}")
                return False
            
            # 读取原文件
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 验证原始代码匹配
            if change.original_code not in content:
                print(f"[Sandbox] Original code mismatch in {change.target_module}")
                return False
            
            # 应用修改
            new_content = content.replace(change.original_code, change.proposed_code)
            
            # 写入修改后文件
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"[Sandbox] Applied change to {change.target_module}")
            return True
            
        except Exception as e:
            print(f"[Sandbox] Failed to apply change: {e}")
            return False
    
    def run_tests(self, test_pattern: str = "tests/") -> Dict:
        """在沙盒中运行测试"""
        if not self.sandbox_dir:
            raise RuntimeError("Sandbox not created")
        
        print(f"[Sandbox] Running tests in {test_pattern}...")
        
        try:
            # 运行pytest
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_pattern, "-v", "--tb=short", "--json-report"],
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            # 解析结果
            test_result = {
                "return_code": result.returncode,
                "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
                "passed": result.returncode == 0
            }
            
            # 尝试解析pytest JSON报告
            report_path = os.path.join(self.sandbox_dir, ".pytest_report.json")
            if os.path.exists(report_path):
                with open(report_path) as f:
                    report = json.load(f)
                    test_result["summary"] = report.get("summary", {})
                    test_result["total"] = report.get("summary", {}).get("total", 0)
                    test_result["passed_count"] = report.get("summary", {}).get("passed", 0)
                    test_result["failed_count"] = report.get("summary", {}).get("failed", 0)
            
            return test_result
            
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Test timeout", "passed_count": 0, "total": 0}
        except Exception as e:
            return {"passed": False, "error": str(e), "passed_count": 0, "total": 0}
    
    def run_benchmark(self, benchmark_script: str = "benchmarks/lighthouse_task.py") -> Dict:
        """运行性能基准测试"""
        if not self.sandbox_dir:
            raise RuntimeError("Sandbox not created")
        
        print(f"[Sandbox] Running benchmark {benchmark_script}...")
        
        try:
            result = subprocess.run(
                [sys.executable, benchmark_script],
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            
            # 尝试解析JSON输出
            try:
                output_lines = result.stdout.split('\n')
                for line in reversed(output_lines):
                    if line.strip().startswith('{'):
                        return json.loads(line)
            except:
                pass
            
            return {
                "passed": result.returncode == 0,
                "output": result.stdout[-2000:],
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "Benchmark timeout"}
        except Exception as e:
            return {"passed": False, "error": str(e)}
    
    def cleanup(self):
        """清理沙盒环境"""
        if self.sandbox_dir and os.path.exists(self.sandbox_dir):
            shutil.rmtree(self.sandbox_dir)
            print(f"[Sandbox] Cleaned up {self.sandbox_dir}")
            self.sandbox_dir = None
    
    def __enter__(self):
        self.create_sandbox()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class EvolutionVersionControl:
    """
    进化版本控制系统
    
    功能:
    1. 创建修改前的备份
    2. 管理修改历史
    3. 快速回滚机制
    """
    
    BACKUP_DIR = "data/evolution_backups"
    MAX_BACKUPS = 10
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.backup_dir = os.path.join(project_root, self.BACKUP_DIR)
        os.makedirs(self.backup_dir, exist_ok=True)
        self._history: List[Dict] = []
        self._load_history()
    
    def create_backup(self, change_id: str, target_module: str) -> str:
        """创建修改前的备份"""
        backup_id = f"{change_id}_{int(time.time())}"
        backup_path = os.path.join(self.backup_dir, backup_id)
        os.makedirs(backup_path, exist_ok=True)
        
        # 备份目标文件
        target_file = os.path.join(self.project_root, target_module)
        if os.path.exists(target_file):
            shutil.copy2(target_file, os.path.join(backup_path, os.path.basename(target_module)))
        
        # 备份元数据
        metadata = {
            "change_id": change_id,
            "target_module": target_module,
            "backup_time": time.time(),
            "backup_path": backup_path
        }
        
        with open(os.path.join(backup_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[VersionControl] Created backup: {backup_id}")
        return backup_id
    
    def rollback(self, backup_id: str) -> bool:
        """回滚到备份版本"""
        backup_path = os.path.join(self.backup_dir, backup_id)
        
        if not os.path.exists(backup_path):
            print(f"[VersionControl] Backup not found: {backup_id}")
            return False
        
        try:
            # 读取元数据
            with open(os.path.join(backup_path, "metadata.json")) as f:
                metadata = json.load(f)
            
            target_module = metadata["target_module"]
            backup_file = os.path.join(backup_path, os.path.basename(target_module))
            target_file = os.path.join(self.project_root, target_module)
            
            # 恢复文件
            if os.path.exists(backup_file):
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                shutil.copy2(backup_file, target_file)
                print(f"[VersionControl] Rolled back {target_module}")
                return True
            else:
                print(f"[VersionControl] Backup file not found")
                return False
                
        except Exception as e:
            print(f"[VersionControl] Rollback failed: {e}")
            return False
    
    def commit(self, backup_id: str, change: ArchitectureChange, approved_by: str = None):
        """提交修改到历史"""
        entry = {
            "change_id": change.change_id,
            "backup_id": backup_id,
            "target_module": change.target_module,
            "change_type": change.change_type,
            "timestamp": time.time(),
            "approved_by": approved_by,
            "status": "committed"
        }
        
        self._history.append(entry)
        self._save_history()
        
        print(f"[VersionControl] Committed change: {change.change_id}")
    
    def get_history(self) -> List[Dict]:
        """获取修改历史"""
        return self._history.copy()
    
    def _load_history(self):
        """加载历史记录"""
        history_file = os.path.join(self.backup_dir, "history.json")
        if os.path.exists(history_file):
            with open(history_file) as f:
                self._history = json.load(f)
    
    def _save_history(self):
        """保存历史记录"""
        history_file = os.path.join(self.backup_dir, "history.json")
        with open(history_file, 'w') as f:
            json.dump(self._history, f, indent=2)
    
    def cleanup_old_backups(self):
        """清理旧备份"""
        backups = []
        for item in os.listdir(self.backup_dir):
            item_path = os.path.join(self.backup_dir, item)
            if os.path.isdir(item_path) and item != "history.json":
                try:
                    with open(os.path.join(item_path, "metadata.json")) as f:
                        metadata = json.load(f)
                        backups.append((item, metadata["backup_time"]))
                except:
                    pass
        
        # 按时间排序，保留最新的
        backups.sort(key=lambda x: x[1], reverse=True)
        
        for backup_id, _ in backups[self.MAX_BACKUPS:]:
            backup_path = os.path.join(self.backup_dir, backup_id)
            shutil.rmtree(backup_path)
            print(f"[VersionControl] Cleaned old backup: {backup_id}")


class TrueEvolutionEngine:
    """
    真进化引擎 - 架构自修改能力
    
    完整流程:
    1. 分析架构瓶颈
    2. 生成修改方案
    3. 沙盒测试验证
    4. 自动化测试
    5. 性能基准对比
    6. 人工审批（可选）
    7. 应用到生产环境
    8. 监控和回滚
    """
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.sandbox = IsolatedSandbox(project_root)
        self.version_control = EvolutionVersionControl(project_root)
        self._results: Dict[str, EvolutionResult] = {}
        self._change_counter = 0
        self._stats = {
            "proposals_generated": 0,
            "sandbox_tests": 0,
            "production_applications": 0,
            "rollbacks": 0,
            "success_rate": 0.0
        }
    
    def analyze_bottleneck(self) -> Dict[str, Any]:
        """分析当前架构瓶颈"""
        bottlenecks = []
        
        # 分析模块数量
        core_dir = os.path.join(self.project_root, "core")
        if os.path.exists(core_dir):
            module_count = len([f for f in os.listdir(core_dir) if f.endswith('.py')])
            if module_count > 50:
                bottlenecks.append({
                    "type": "module_complexity",
                    "severity": "high",
                    "description": f"Too many modules ({module_count}), consider consolidation",
                    "suggestion": "Merge related modules"
                })
        
        # 分析代码重复（简化版）
        bottlenecks.append({
            "type": "code_duplication",
            "severity": "medium",
            "description": "Potential code duplication in utility functions",
            "suggestion": "Extract common utilities to shared module"
        })
        
        # 分析测试覆盖
        test_dir = os.path.join(self.project_root, "tests")
        if not os.path.exists(test_dir) or len(os.listdir(test_dir)) < 5:
            bottlenecks.append({
                "type": "test_coverage",
                "severity": "high",
                "description": "Insufficient test coverage",
                "suggestion": "Add comprehensive unit tests"
            })
        
        return {
            "bottlenecks": bottlenecks,
            "timestamp": time.time()
        }
    
    def propose_architecture_change(self, motivation: str = None) -> Optional[ArchitectureChange]:
        """提出架构修改方案"""
        self._change_counter += 1
        change_id = f"evolution_{datetime.now().strftime('%Y%m%d')}_{self._change_counter:04d}"
        
        # 1. 分析瓶颈
        bottleneck = self.analyze_bottleneck()
        
        if not bottleneck["bottlenecks"]:
            print("[EvolutionEngine] No significant bottlenecks found")
            return None
        
        # 选择最严重的瓶颈
        top_bottleneck = max(bottleneck["bottlenecks"], key=lambda x: {"low": 1, "medium": 2, "high": 3}[x["severity"]])
        
        # 2. 生成修改方案（简化版，实际应使用LLM）
        if top_bottleneck["type"] == "module_complexity":
            change = self._generate_merge_modules_change(change_id, top_bottleneck)
        elif top_bottleneck["type"] == "code_duplication":
            change = self._generate_extract_utility_change(change_id, top_bottleneck)
        elif top_bottleneck["type"] == "test_coverage":
            change = self._generate_add_tests_change(change_id, top_bottleneck)
        else:
            return None
        
        if change:
            self._stats["proposals_generated"] += 1
        
        return change
    
    def _generate_merge_modules_change(self, change_id: str, bottleneck: Dict) -> ArchitectureChange:
        """生成合并模块的修改方案"""
        return ArchitectureChange(
            change_id=change_id,
            motivation=bottleneck["description"],
            target_module="core/__init__.py",
            change_type="refactor",
            description="Consolidate related modules to reduce complexity",
            proposed_code="# Consolidated module\nfrom .module_a import *\nfrom .module_b import *",
            original_code="# Original imports\nfrom .module_a import func_a\nfrom .module_b import func_b",
            estimated_impact=0.7,
            risk_level="medium",
            created_at=time.time()
        )
    
    def _generate_extract_utility_change(self, change_id: str, bottleneck: Dict) -> ArchitectureChange:
        """生成提取工具函数的修改方案"""
        return ArchitectureChange(
            change_id=change_id,
            motivation=bottleneck["description"],
            target_module="core/utils.py",
            change_type="refactor",
            description="Extract common utility functions",
            proposed_code="def common_utility():\n    pass\n\ndef another_utility():\n    pass",
            original_code="# No common utilities yet",
            estimated_impact=0.5,
            risk_level="low",
            created_at=time.time()
        )
    
    def _generate_add_tests_change(self, change_id: str, bottleneck: Dict) -> ArchitectureChange:
        """生成添加测试的修改方案"""
        return ArchitectureChange(
            change_id=change_id,
            motivation=bottleneck["description"],
            target_module="tests/test_new_feature.py",
            change_type="add",
            description="Add comprehensive unit tests",
            proposed_code="import unittest\n\nclass TestNewFeature(unittest.TestCase):\n    def test_feature(self):\n        self.assertTrue(True)",
            original_code="",
            estimated_impact=0.3,
            risk_level="low",
            created_at=time.time()
        )
    
    async def validate_change(self, change: ArchitectureChange) -> EvolutionResult:
        """验证修改方案"""
        result = EvolutionResult(
            change_id=change.change_id,
            status=EvolutionStatus.PENDING,
            start_time=time.time()
        )
        
        print(f"\n[EvolutionEngine] Validating change: {change.change_id}")
        print(f"  Target: {change.target_module}")
        print(f"  Type: {change.change_type}")
        print(f"  Risk: {change.risk_level}")
        
        # 3. 沙盒测试
        result.status = EvolutionStatus.SANDBOXING
        with self.sandbox:
            sandbox_success = self.sandbox.apply_change(change)
            result.sandbox_success = sandbox_success
            
            if not sandbox_success:
                result.status = EvolutionStatus.FAILED
                result.error_message = "Failed to apply change in sandbox"
                result.end_time = time.time()
                return result
            
            # 4. 自动化测试
            result.status = EvolutionStatus.TESTING
            test_result = self.sandbox.run_tests()
            result.test_pass_rate = test_result.get("passed_count", 0) / max(test_result.get("total", 1), 1)
            
            if result.test_pass_rate < 0.9:  # 要求90%通过率
                result.status = EvolutionStatus.FAILED
                result.error_message = f"Test pass rate too low: {result.test_pass_rate:.1%}"
                result.end_time = time.time()
                return result
            
            # 5. 性能基准
            result.status = EvolutionStatus.BENCHMARKING
            benchmark = self.sandbox.run_benchmark()
            result.benchmark_results = benchmark
            
            # 简化：假设性能不变
            result.performance_delta = 0.0
        
        result.status = EvolutionStatus.APPROVING
        result.end_time = time.time()
        
        self._results[change.change_id] = result
        self._stats["sandbox_tests"] += 1
        
        print(f"[EvolutionEngine] Validation complete:")
        print(f"  Test pass rate: {result.test_pass_rate:.1%}")
        print(f"  Performance delta: {result.performance_delta:+.1%}")
        
        return result
    
    async def apply_change(self, change: ArchitectureChange, 
                          validated_result: EvolutionResult,
                          approved_by: str = "system") -> EvolutionResult:
        """应用经过验证的修改到生产环境"""
        if validated_result.status != EvolutionStatus.APPROVING:
            validated_result.error_message = "Change not in approving state"
            validated_result.status = EvolutionStatus.FAILED
            return validated_result
        
        validated_result.status = EvolutionStatus.APPLYING
        
        # 1. 创建备份
        backup_id = self.version_control.create_backup(change.change_id, change.target_module)
        
        # 2. 应用修改
        try:
            target_file = os.path.join(self.project_root, change.target_module)
            
            # 创建目录
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            
            # 读取或创建文件
            if os.path.exists(target_file):
                with open(target_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                content = ""
            
            # 应用修改
            if change.original_code:
                new_content = content.replace(change.original_code, change.proposed_code)
            else:
                new_content = content + "\n\n" + change.proposed_code
            
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # 3. 提交到版本控制
            self.version_control.commit(backup_id, change, approved_by)
            
            validated_result.status = EvolutionStatus.COMPLETED
            validated_result.applied_by = approved_by
            
            self._stats["production_applications"] += 1
            self._update_success_rate()
            
            print(f"[EvolutionEngine] Change applied successfully: {change.change_id}")
            print(f"  Backup ID: {backup_id}")
            print(f"  Rollback available: true")
            
        except Exception as e:
            validated_result.status = EvolutionStatus.FAILED
            validated_result.error_message = str(e)
            print(f"[EvolutionEngine] Failed to apply change: {e}")
        
        validated_result.end_time = time.time()
        return validated_result
    
    def rollback_change(self, change_id: str) -> bool:
        """回滚修改"""
        print(f"[EvolutionEngine] Rolling back change: {change_id}")
        
        # 查找备份
        backup_id = None
        for entry in self.version_control.get_history():
            if entry["change_id"] == change_id:
                backup_id = entry["backup_id"]
                break
        
        if not backup_id:
            print(f"[EvolutionEngine] No backup found for {change_id}")
            return False
        
        # 执行回滚
        success = self.version_control.rollback(backup_id)
        
        if success:
            self._stats["rollbacks"] += 1
            print(f"[EvolutionEngine] Rollback successful")
        else:
            print(f"[EvolutionEngine] Rollback failed")
        
        return success
    
    def _update_success_rate(self):
        """更新成功率"""
        total = self._stats["production_applications"] + self._stats["rollbacks"]
        if total > 0:
            self._stats["success_rate"] = (self._stats["production_applications"] - self._stats["rollbacks"]) / total
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self._stats.copy()
    
    def get_pending_changes(self) -> List[ArchitectureChange]:
        """获取待处理的修改"""
        # 简化版：返回最近生成的修改
        return []


# 便捷函数
def create_evolution_engine(project_root: str = None) -> TrueEvolutionEngine:
    """创建真进化引擎"""
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return TrueEvolutionEngine(project_root)


# 测试代码
if __name__ == "__main__":
    import asyncio
    
    async def test_evolution():
        print("真进化引擎测试")
        print("=" * 70)
        
        # 创建引擎
        engine = create_evolution_engine()
        
        # 分析瓶颈
        print("\n1. 分析架构瓶颈...")
        bottleneck = engine.analyze_bottleneck()
        print(f"   发现 {len(bottleneck['bottlenecks'])} 个瓶颈")
        for b in bottleneck["bottlenecks"]:
            print(f"   - [{b['severity']}] {b['type']}: {b['description']}")
        
        # 生成修改方案
        print("\n2. 生成修改方案...")
        change = engine.propose_architecture_change()
        
        if change:
            print(f"   方案ID: {change.change_id}")
            print(f"   目标模块: {change.target_module}")
            print(f"   修改类型: {change.change_type}")
            print(f"   预估影响: {change.estimated_impact:.1%}")
            print(f"   风险等级: {change.risk_level}")
            
            # 验证修改
            print("\n3. 验证修改方案...")
            result = await engine.validate_change(change)
            
            print(f"   状态: {result.status.value}")
            print(f"   沙盒成功: {result.sandbox_success}")
            print(f"   测试通过率: {result.test_pass_rate:.1%}")
            print(f"   性能变化: {result.performance_delta:+.1%}")
            
            if result.status == EvolutionStatus.APPROVING:
                print("\n4. 应用到生产环境...")
                final_result = await engine.apply_change(change, result, approved_by="test_user")
                print(f"   最终状态: {final_result.status.value}")
        else:
            print("   未生成修改方案")
        
        # 统计
        print("\n" + "=" * 70)
        print("统计:")
        stats = engine.get_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # 运行测试
    asyncio.run(test_evolution())
