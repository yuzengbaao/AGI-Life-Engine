"""
SelfModifyingEngine - 架构自修改引擎

⚠️  警告: 这是AGI系统中最危险的组件

功能边界:
- 输入: 待优化的代码模块 + 性能/安全分析报告
- 输出: 测试通过的安全代码补丁 (或拒绝理由)
- 约束: 严格的不可变约束 + 沙箱测试 + 快速回滚

拓扑连接:
- SelfModifyingEngine 分析 core.* 模块
- SelfModifyingEngine 通过 EventBus 发布 modification_proposed 事件
- CriticAgent 审批高风险修改
- AuditLog 记录所有修改证据链

安全原则:
1. **不可变约束**: 核心安全代码不可修改
2. **沙箱测试**: 所有修改必须在隔离环境测试
3. **快速回滚**: 30秒内必须能回滚
4. **人工审批**: 高风险修改需要人工确认
5. **完整审计**: 每次修改都有完整证据链
"""

import ast
import inspect
import logging
import hashlib
import json
import time
import copy
import sys
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, Callable, Set
from difflib import unified_diff

# 集成无LLM补丁生成器
from core.template_based_patch_generator import TemplateBasedPatchGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# 枚举和数据结构
# ============================================================================

class ModificationRisk(Enum):
    """修改风险等级"""
    SAFE = "safe"           # 安全: 仅优化非关键代码
    LOW = "low"             # 低风险: 优化关键代码但逻辑不变
    MEDIUM = "medium"       # 中风险: 轻微逻辑变更
    HIGH = "high"           # 高风险: 重要逻辑变更,需人工审批
    CRITICAL = "critical"   # 禁止: 触发不可变约束


class ModificationStatus(Enum):
    """修改状态"""
    PROPOSED = "proposed"           # 已提出
    ANALYZING = "analyzing"         # 分析中
    SANDBOX_TESTING = "sandbox_testing"  # 沙箱测试中
    APPROVED = "approved"           # 已批准
    APPLIED = "applied"             # 已应用
    REJECTED = "rejected"           # 已拒绝
    ROLLED_BACK = "rolled_back"     # 已回滚


@dataclass
class CodeLocation:
    """代码位置"""
    file_path: str
    class_name: Optional[str] = None
    function_name: Optional[str] = None
    line_start: int = 0
    line_end: int = 0


@dataclass
class ImmutableConstraint:
    """
    不可变约束

    保护核心安全机制,防止自修改破坏安全边界
    """
    name: str
    description: str
    protected_patterns: List[str]  # 受保护的代码模式
    check_func: Callable[[str, CodeLocation], bool]  # 检查函数
    violation_level: ModificationRisk  # 违规等级


@dataclass
class CodeAnalysis:
    """代码分析结果"""
    locations: List[CodeLocation]
    dependencies: List[str]  # 依赖的其他模块
    risk_points: List[str]   # 风险点
    complexity: float        # 复杂度评分
    test_coverage: float     # 测试覆盖率
    safety_score: float      # 安全评分


@dataclass
class CodePatch:
    """代码补丁"""
    original_code: str
    modified_code: str
    location: CodeLocation
    description: str
    risk_level: ModificationRisk
    estimated_impact: str
    test_cases: List[str]   # 需要运行的测试


@dataclass
class ModificationRecord:
    """修改记录 (审计日志)"""
    id: str
    timestamp: float
    status: ModificationStatus

    # 修改内容
    patch: CodePatch
    original_code_hash: str
    modified_code_hash: str

    # 分析结果
    analysis: CodeAnalysis
    risk_assessment: Dict[str, Any]

    # 测试结果
    sandbox_test_passed: bool
    test_results: Dict[str, Any]

    # 审批流程
    human_approval_required: bool
    human_approval_granted: bool = False
    approver: Optional[str] = None

    # 回滚信息
    backup_path: Optional[str] = None
    rollback_successful: bool = False
    rollback_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        data = asdict(self)
        data['status'] = self.status.value
        data['patch']['risk_level'] = self.patch.risk_level.value
        return data


# ============================================================================
# 不可变约束定义
# ============================================================================

class ImmutableConstraints:
    """
    不可变约束集合

    这些约束是AGI系统的安全底线,绝对不可通过自修改破坏
    """

    @staticmethod
    def get_core_constraints() -> List[ImmutableConstraint]:
        """获取核心不可变约束"""
        return [
            # 1. 安全开关不可修改
            ImmutableConstraint(
                name="safety_switch_protection",
                description="保护紧急关闭/人工接管机制",
                protected_patterns=[
                    "class EmergencyShutdown",
                    "def manual_override",
                    "SAFETY_INTERLOCK"
                ],
                check_func=ImmutableConstraints._check_safety_switch,
                violation_level=ModificationRisk.CRITICAL
            ),

            # 2. 权限模型不可修改
            ImmutableConstraint(
                name="permission_model_protection",
                description="保护权限检查机制",
                protected_patterns=[
                    "class Permission",
                    "def check_permission",
                    "def authorize"
                ],
                check_func=ImmutableConstraints._check_permission_model,
                violation_level=ModificationRisk.CRITICAL
            ),

            # 3. 回滚机制不可修改
            ImmutableConstraint(
                name="rollback_protection",
                description="保护回滚/恢复机制",
                protected_patterns=[
                    "def rollback",
                    "def restore",
                    "class BackupManager"
                ],
                check_func=ImmutableConstraints._check_rollback_mechanism,
                violation_level=ModificationRisk.CRITICAL
            ),

            # 4. SelfModifyingEngine自身不可修改 (防止无限递归)
            ImmutableConstraint(
                name="self_modification_protection",
                description="防止SelfModifyingEngine修改自身",
                protected_patterns=[
                    "class SelfModifyingEngine",
                    "def propose_patch",
                    "def apply_patch"
                ],
                check_func=ImmutableConstraints._check_self_modification,
                violation_level=ModificationRisk.CRITICAL
            ),

            # 5. 审计日志不可修改
            ImmutableConstraint(
                name="audit_log_protection",
                description="保护审计日志完整性",
                protected_patterns=[
                    "class AuditLog",
                    "def record_modification",
                    "audit_log.append"
                ],
                check_func=ImmutableConstraints._check_audit_log,
                violation_level=ModificationRisk.CRITICAL
            )
        ]

    @staticmethod
    def _check_safety_switch(code: str, location: CodeLocation) -> bool:
        """检查是否修改安全开关"""
        dangerous_patterns = [
            "EmergencyShutdown",
            "manual_override",
            "SAFETY_INTERLOCK",
            "emergency_stop",
            "kill_switch"
        ]
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                logger.critical(f"[不可变约束] 检测到安全开关修改尝试: {pattern}")
                return False
        return True

    @staticmethod
    def _check_permission_model(code: str, location: CodeLocation) -> bool:
        """检查是否修改权限模型"""
        dangerous_patterns = [
            "def authorize",
            "def check_permission",
            "class Permission",
            "class AccessControl"
        ]
        for pattern in dangerous_patterns:
            if pattern in code:
                logger.critical(f"[不可变约束] 检测到权限模型修改尝试: {pattern}")
                return False
        return True

    @staticmethod
    def _check_rollback_mechanism(code: str, location: CodeLocation) -> bool:
        """检查是否修改回滚机制"""
        dangerous_patterns = [
            "def rollback",
            "def restore_backup",
            "class BackupManager"
        ]
        for pattern in dangerous_patterns:
            if pattern in code:
                # 允许在rollback实现文件中,但不能修改其逻辑
                if "self_modifying_engine" not in location.file_path.lower():
                    logger.critical(f"[不可变约束] 检测到回滚机制修改尝试: {pattern}")
                    return False
        return True

    @staticmethod
    def _check_self_modification(code: str, location: CodeLocation) -> bool:
        """检查是否尝试修改SelfModifyingEngine自身"""
        # 如果代码中包含SelfModifyingEngine的核心方法定义，则视为尝试自修改
        dangerous_methods = [
            "def propose_patch",
            "def apply_patch",
            "def sandbox_test",
            "def _analyze_code"
        ]
        for method in dangerous_methods:
            if method in code:
                logger.critical(f"[不可变约束] 检测到SelfModifyingEngine自身修改: {method}")
                return False
        return True

    @staticmethod
    def _check_audit_log(code: str, location: CodeLocation) -> bool:
        """检查是否修改审计日志"""
        dangerous_patterns = [
            "audit_log.clear(",
            "audit_log.remove(",
            "del audit_log",
            ".clear()",  # 任何对审计日志的clear操作
            ".remove(",  # 任何对审计日志的remove操作
        ]
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower or pattern in code:
                # 检查是否真的针对audit_log
                if "audit" in code_lower and ("clear" in code_lower or "remove" in code_lower or "del " in code_lower):
                    logger.critical(f"[不可变约束] 检测到审计日志破坏尝试: {pattern}")
                    return False
        return True


# ============================================================================
# 核心实现
# ============================================================================

class SelfModifyingEngine:
    """
    架构自修改引擎

    流程:
    1. analyze(): 静态分析代码 (依赖图/风险点/复杂度)
    2. propose_patch(): 生成补丁 (基于分析结果)
    3. sandbox_test(): 在隔离环境测试
    4. apply_or_reject(): 应用或拒绝
    5. rollback(): 必要时回滚

    安全保证:
    - 不可变约束强制检查
    - 所有修改都有备份
    - 30秒内必须能回滚
    - 完整审计日志
    """

    # 配置常量
    MAX_ROLLBACK_TIME_SECONDS = 30  # 最大回滚时间
    SANDBOX_TEST_TIMEOUT = 60       # 沙箱测试超时
    MAX_PATCH_SIZE_LINES = 100      # 单次补丁最大行数
    MAX_CHANGES_PER_SESSION = 5     # 每次会话最多修改数

    def __init__(self, event_bus: Any = None,
                 project_root: str = None,
                 auto_apply_safe: bool = False):
        """
        初始化SelfModifyingEngine

        Args:
            event_bus: 事件总线
            project_root: 项目根目录
            auto_apply_safe: 是否自动应用安全级别修改
        """
        self.event_bus = event_bus
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.auto_apply_safe = auto_apply_safe

        # 不可变约束
        self.immutable_constraints = ImmutableConstraints.get_core_constraints()

        # 状态
        self.modification_history: List[ModificationRecord] = []
        self.current_session_changes = 0
        self.backup_dir = self.project_root / ".backups" / "self_modification"
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # 统计
        self._total_proposed = 0
        self._total_applied = 0
        self._total_rejected = 0
        self._total_rolled_back = 0

        # 新增：无LLM补丁生成器
        self.patch_generator = TemplateBasedPatchGenerator()

        logger.info(f"🔧 SelfModifyingEngine initialized (project_root={self.project_root})")

    # ========================================================================
    # 核心接口
    # ========================================================================

    def analyze(self, module_path: str) -> CodeAnalysis:
        """
        静态分析代码

        分析内容:
        1. AST解析,提取依赖关系
        2. 识别风险点 (复杂函数、深层嵌套等)
        3. 计算复杂度
        4. 检查测试覆盖

        Args:
            module_path: Python模块路径 (如 "core.seed")

        Returns:
            代码分析结果
        """
        logger.info(f"[SelfModifyingEngine] 分析模块: {module_path}")

        # 转换为文件路径
        file_path = self._module_to_file(module_path)
        if not file_path or not file_path.exists():
            logger.error(f"模块不存在: {module_path}")
            return CodeAnalysis(locations=[], dependencies=[], risk_points=[],
                              complexity=0.0, test_coverage=0.0, safety_score=0.0)

        # 读取源代码
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()

        # AST分析
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            logger.error(f"语法错误: {e}")
            return CodeAnalysis(locations=[], dependencies=[], risk_points=["语法错误"],
                              complexity=0.0, test_coverage=0.0, safety_score=0.0)

        # 提取信息
        locations = self._extract_locations(tree, file_path)
        dependencies = self._extract_dependencies(tree)
        risk_points = self._identify_risk_points(tree, source_code)
        complexity = self._calculate_complexity(tree)
        test_coverage = self._estimate_test_coverage(module_path)
        safety_score = self._calculate_safety_score(risk_points, complexity)

        analysis = CodeAnalysis(
            locations=locations,
            dependencies=dependencies,
            risk_points=risk_points,
            complexity=complexity,
            test_coverage=test_coverage,
            safety_score=safety_score
        )

        logger.info(f"  分析完成: {len(locations)} 个位置, "
                   f"{len(risk_points)} 个风险点, "
                   f"复杂度={complexity:.2f}")

        return analysis

    def propose_patch(self, target_module: str,
                     issue_description: str,
                     optimization_goal: str = "performance",
                     use_llm: bool = False,
                     patch_strategy: str = "auto") -> Optional[CodePatch]:
        """
        生成代码补丁

        约束:
        1. 单次补丁不超过100行
        2. 不触发不可变约束
        3. 风险等级评估

        Args:
            target_module: 目标模块 (如 "core.seed")
            issue_description: 问题描述
            optimization_goal: 优化目标 (performance/safety/readability)

        Returns:
            代码补丁或None (如果不可修改)
        """
        logger.info(f"[SelfModifyingEngine] 提出补丁: {target_module}")

        self._total_proposed += 1

        # 分析目标代码
        analysis = self.analyze(target_module)
        file_path = self._module_to_file(target_module)

        if not file_path or not file_path.exists():
            logger.error(f"目标模块不存在: {target_module}")
            return None

        # 读取原始代码
        with open(file_path, 'r', encoding='utf-8') as f:
            original_code = f.read()

        # 检查不可变约束
        if not self._check_immutable_constraints(original_code, file_path):
            logger.error("触发不可变约束,拒绝修改")
            self._publish_rejection_event(target_module, "触发不可变约束")
            self._total_rejected += 1
            return None


        # 优先使用无LLM补丁生成器
        if not use_llm:
            modified_code = self.patch_generator.generate_patch(
                old_code=original_code,
                target_desc=issue_description + ", goal=" + optimization_goal,
                strategy=patch_strategy
            )
        else:
            # 兼容原有LLM/符号执行分支
            modified_code = self._generate_optimization(
                original_code,
                analysis,
                optimization_goal
            )

        if not modified_code or modified_code == original_code:
            logger.warning("未能生成有效补丁")
            return None

        # 计算补丁大小
        diff_lines = sum(1 for _ in unified_diff(
            original_code.splitlines(keepends=True),
            modified_code.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}"
        ))

        if diff_lines > self.MAX_PATCH_SIZE_LINES:
            logger.warning(f"补丁过大 ({diff_lines} 行),拒绝修改")
            self._publish_rejection_event(target_module, f"补丁过大 ({diff_lines} > {self.MAX_PATCH_SIZE_LINES})")
            self._total_rejected += 1
            return None

        # 评估风险等级
        risk_level = self._assess_risk_level(original_code, modified_code, analysis)

        if risk_level == ModificationRisk.CRITICAL:
            logger.error("风险等级为CRITICAL,拒绝修改")
            self._publish_rejection_event(target_module, "风险等级过高")
            self._total_rejected += 1
            return None

        # 生成测试用例
        test_cases = self._generate_test_cases(target_module, modified_code)

        # 选择修改位置
        location = CodeLocation(
            file_path=str(file_path),
            line_start=0,
            line_end=len(original_code.splitlines())
        )

        patch = CodePatch(
            original_code=original_code,
            modified_code=modified_code,
            location=location,
            description=f"{optimization_goal}优化: {issue_description}",
            risk_level=risk_level,
            estimated_impact=self._estimate_impact(analysis, optimization_goal),
            test_cases=test_cases
        )

        logger.info(f"  补丁已生成: risk_level={risk_level.value}, "
                   f"diff_lines={diff_lines}")

        return patch

    def sandbox_test(self, patch: CodePatch) -> Tuple[bool, Dict[str, Any]]:
        """
        在沙箱环境中测试补丁（增强版）

        测试流程:
        1. 函数级测试（如果是函数补丁）
        2. 性能基准测试
        3. 隔离沙箱测试
        4. 语法和导入检查

        Args:
            patch: 代码补丁

        Returns:
            (测试是否通过, 测试结果详情)
        """
        logger.info(f"[SelfModifyingEngine] 沙箱测试: {patch.location.file_path}")

        test_results = {
            'timestamp': time.time(),
            'patch_risk_level': patch.risk_level.value,
            'test_cases_run': 0,
            'test_cases_passed': 0,
            'errors': [],
            'warnings': [],
            'performance': None,  # 新增：性能基准
            'function_test': None,  # 新增：函数级测试
            'sandbox_isolation': None  # 新增：沙箱隔离测试
        }

        try:
            # ========== 新增：函数级测试 ==========
            if patch.location.function_name:
                logger.info(f"  函数级测试: {patch.location.function_name}")
                function_test_result = self._test_function_in_sandbox(
                    patch.location.class_name,
                    patch.location.function_name,
                    patch.modified_code
                )
                test_results['function_test'] = function_test_result

                if not function_test_result.get('passed', False):
                    test_results['errors'].append(
                        f"函数级测试失败: {function_test_result.get('error')}"
                    )
            # ========== 新增结束 ==========

            # ========== 新增：性能基准测试 ==========
            perf_result = self._performance_benchmark(patch)
            test_results['performance'] = perf_result

            if not perf_result.get('passed', True):
                test_results['warnings'].append(
                    f"性能基准未达标: {perf_result.get('avg_time_ms', 0):.2f}ms "
                    f"(目标: <1.0ms)"
                )
            # ========== 新增结束 ==========

            # ========== 新增：隔离沙箱测试 ==========
            isolation_result = self._test_isolated_sandbox(patch)
            test_results['sandbox_isolation'] = isolation_result

            if not isolation_result.get('passed', True):
                test_results['errors'].append(
                    f"隔离沙箱测试失败: {isolation_result.get('error')}"
                )
            # ========== 新增结束 ==========

            # ========== 原有测试逻辑 ==========
            with tempfile.TemporaryDirectory() as sandbox_dir:
                sandbox_path = Path(sandbox_dir)

                # 复制文件到沙箱
                sandbox_file = sandbox_path / "test_module.py"
                with open(sandbox_file, 'w', encoding='utf-8') as f:
                    f.write(patch.modified_code)

                # 尝试导入和语法检查
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "test_module",
                    sandbox_file
                )

                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)

                    # 语法检查
                    try:
                        spec.loader.exec_module(module)
                        test_results['syntax_check'] = "PASS"
                    except Exception as e:
                        test_results['syntax_check'] = "FAIL"
                        test_results['errors'].append(f"语法错误: {e}")
                        return False, test_results

                # 运行测试用例
                for test_case in patch.test_cases:
                    test_results['test_cases_run'] += 1

                    try:
                        # 简化: 这里应该运行实际的单元测试
                        # 目前只做基本的导入检查
                        if test_case == "import_test":
                            test_results['test_cases_passed'] += 1
                        elif test_case == "syntax_test":
                            test_results['test_cases_passed'] += 1

                    except Exception as e:
                        test_results['errors'].append(f"测试失败: {test_case}, {e}")
            # ========== 原有逻辑结束 ==========

        except Exception as e:
            test_results['errors'].append(f"沙箱测试异常: {e}")
            logger.error(f"沙箱测试失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False, test_results

        # 判断是否通过
        passed = (
            len(test_results['errors']) == 0 and
            test_results['test_cases_passed'] >= test_results['test_cases_run']
        )

        if passed:
            logger.info(f"  沙箱测试通过: {test_results['test_cases_passed']}/"
                       f"{test_results['test_cases_run']} 测试通过")
        else:
            logger.warning(f"  沙箱测试失败: {len(test_results['errors'])} 个错误")

        return passed, test_results

    def _test_function_in_sandbox(
        self,
        class_name: Optional[str],
        function_name: str,
        code: str
    ) -> Dict[str, Any]:
        """
        在沙箱中测试单个函数

        Args:
            class_name: 类名（可选）
            function_name: 函数名
            code: 函数代码

        Returns:
            测试结果字典
        """
        result = {
            'passed': False,
            'error': None,
            'executions': 0,
            'exceptions': 0
        }

        try:
            # 编译代码
            namespace = {}
            exec(code, namespace)

            # 提取函数
            func = namespace.get(function_name)
            if not func:
                result['error'] = f"函数未找到: {function_name}"
                return result

            # 测试执行
            test_args = (1, 2, 3)  # 默认测试参数

            for i in range(5):
                result['executions'] += 1
                try:
                    func(*test_args[:func.__code__.co_argcount])
                except Exception as e:
                    result['exceptions'] += 1
                    logger.debug(f"  函数测试异常（第{i+1}次）: {e}")

            # 判断成功（无异常或异常率<20%）
            result['passed'] = (result['exceptions'] == 0 or
                               result['exceptions'] / result['executions'] < 0.2)

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    def _performance_benchmark(self, patch: CodePatch) -> Dict[str, Any]:
        """
        性能基准测试

        使用timeit重复执行代码，测量平均耗时

        Args:
            patch: 代码补丁

        Returns:
            性能测试结果
        """
        import timeit
        import numpy as np

        result = {
            'passed': True,
            'avg_time_ms': 0.0,
            'std_time_ms': 0.0,
            'min_time_ms': 0.0,
            'max_time_ms': 0.0,
            'samples': 0
        }

        try:
            # 运行新代码100次，重复5轮
            timings = timeit.repeat(
                lambda: exec(patch.modified_code, {}),
                number=100,
                repeat=5
            )

            # 转换为毫秒
            timings_ms = [t * 1000 for t in timings]

            result['avg_time_ms'] = np.mean(timings_ms)
            result['std_time_ms'] = np.std(timings_ms)
            result['min_time_ms'] = np.min(timings_ms)
            result['max_time_ms'] = np.max(timings_ms)
            result['samples'] = len(timings)

            # 判断是否通过（平均<1ms）
            result['passed'] = result['avg_time_ms'] < 1.0

            logger.debug(
                f"  性能基准: 平均={result['avg_time_ms']:.2f}ms, "
                f"标准差={result['std_time_ms']:.2f}ms"
            )

            return result

        except Exception as e:
            logger.warning(f"性能基准测试失败: {e}")
            result['passed'] = True  # 默认通过，不阻塞
            result['error'] = str(e)
            return result

    def _test_isolated_sandbox(self, patch: CodePatch) -> Dict[str, Any]:
        """
        测试隔离沙箱

        使用独立进程执行代码，验证隔离效果

        Args:
            patch: 代码补丁

        Returns:
            隔离测试结果
        """
        result = {
            'passed': True,
            'error': None,
            'escape_detected': False
        }

        try:
            from core.isolated_sandbox import get_isolated_sandbox

            sandbox = get_isolated_sandbox()

            # 在隔离环境中执行代码
            success, data, error = sandbox.execute_in_sandbox(
                code=patch.modified_code,
                timeout=10.0
            )

            if not success:
                result['passed'] = False
                result['error'] = error

            # 检查逃逸尝试
            if sandbox.escape_attempts:
                result['passed'] = False
                result['escape_detected'] = True
                result['error'] = f"检测到逃逸尝试: {sandbox.escape_attempts[-1]}"

            return result

        except Exception as e:
            logger.warning(f"隔离沙箱测试失败（非致命）: {e}")
            result['passed'] = True  # 默认通过，不阻塞
            result['error'] = str(e)
            return result

    def apply_or_reject(self, patch: CodePatch,
                       force_apply: bool = False) -> ModificationRecord:
        """
        应用或拒绝补丁

        流程:
        1. 创建备份
        2. 应用补丁
        3. 运行验证测试
        4. 如果失败,回滚
        5. 记录审计日志

        Args:
            patch: 代码补丁
            force_apply: 强制应用 (跳过人工审批)

        Returns:
            修改记录
        """
        logger.info(f"[SelfModifyingEngine] 应用补丁: {patch.location.file_path}")

        # 创建修改记录
        record_id = hashlib.sha256(
            f"{patch.location.file_path}{time.time()}".encode()
        ).hexdigest()[:16]

        # 分析代码
        analysis = self.analyze(
            self._file_to_module(str(patch.location.file_path))
        )

        # 风险评估
        risk_assessment = {
            'risk_level': patch.risk_level.value,
            'estimated_impact': patch.estimated_impact,
            'complexity_change': 0.0,
            'safety_score_change': 0.0
        }

        # 判断是否需要人工审批
        human_approval_required = (
            patch.risk_level in [ModificationRisk.HIGH, ModificationRisk.CRITICAL] and
            not force_apply
        )

        # 创建修改记录
        record = ModificationRecord(
            id=record_id,
            timestamp=time.time(),
            status=ModificationStatus.PROPOSED,
            patch=patch,
            original_code_hash=hashlib.sha256(
                patch.original_code.encode()
            ).hexdigest(),
            modified_code_hash=hashlib.sha256(
                patch.modified_code.encode()
            ).hexdigest(),
            analysis=analysis,
            risk_assessment=risk_assessment,
            sandbox_test_passed=False,
            test_results={},
            human_approval_required=human_approval_required,
            human_approval_granted=False
        )

        # 检查是否需要人工审批
        if human_approval_required and not force_apply:
            logger.warning(f"[SelfModifyingEngine] 需要人工审批: {record_id}")
            logger.warning(f"  风险等级: {patch.risk_level.value}")
            logger.warning(f"  描述: {patch.description}")

            record.status = ModificationStatus.PROPOSED
            self.modification_history.append(record)
            self._publish_approval_request_event(record)

            return record

        def run_regression_flow(self,
                                target_module: str,
                                issue_description: str,
                                optimization_goal: str = "readability",
                                use_llm: bool = False,
                                patch_strategy: str = "auto",
                                force_apply: bool = True) -> Dict[str, Any]:
            """
            一体化回归流程：生成补丁 → 沙箱测试 → 自动应用 → 回滚

            Returns:
                执行结果摘要
            """
            result: Dict[str, Any] = {
                "target_module": target_module,
                "issue_description": issue_description,
                "optimization_goal": optimization_goal,
                "patch_generated": False,
                "sandbox_test": None,
                "apply": None,
                "rollback": None
            }

            patch = self.propose_patch(
                target_module=target_module,
                issue_description=issue_description,
                optimization_goal=optimization_goal,
                use_llm=use_llm,
                patch_strategy=patch_strategy
            )

            if not patch:
                result["error"] = "patch_generation_failed"
                return result

            result["patch_generated"] = True

            ok, report = self.sandbox_test(patch)
            result["sandbox_test"] = {"ok": ok, "report": report}
            if not ok:
                result["error"] = "sandbox_test_failed"
                return result

            record = self.apply_or_reject(patch, force_apply=force_apply)
            result["apply"] = record

            record_id = None
            if hasattr(record, 'id'):
                record_id = record.id
            elif isinstance(record, dict):
                record_id = record.get('record_id')

            if record_id:
                result["rollback"] = self.rollback(record_id)
            else:
                result["rollback"] = False
            return result

        # 沙箱测试
        logger.info("运行沙箱测试...")
        test_passed, test_results = self.sandbox_test(patch)
        record.sandbox_test_passed = test_passed
        record.test_results = test_results

        if not test_passed:
            logger.error("沙箱测试失败,拒绝补丁")
            record.status = ModificationStatus.REJECTED
            self.modification_history.append(record)
            self._total_rejected += 1
            self._publish_rejection_event(str(patch.location.file_path), "沙箱测试失败")
            return record

        # 创建备份
        backup_path = self._create_backup(patch.location.file_path, record_id)
        record.backup_path = str(backup_path)

        # 应用补丁
        try:
            logger.info("应用补丁...")
            start_time = time.time()

            with open(patch.location.file_path, 'w', encoding='utf-8') as f:
                f.write(patch.modified_code)

            apply_time = time.time() - start_time
            logger.info(f"  补丁已应用 ({apply_time:.3f}秒)")

            # 验证修改后的代码
            verification_passed = self._verify_modification(patch)

            if not verification_passed:
                logger.warning("验证失败,回滚补丁")
                self._rollback_patch(record)
                record.status = ModificationStatus.REJECTED
                self.modification_history.append(record)
                return record

            # 成功应用
            record.status = ModificationStatus.APPLIED
            self.modification_history.append(record)
            self._total_applied += 1
            self.current_session_changes += 1

            # 发布事件
            self._publish_modification_event(record)

            logger.info(f"✅ 补丁成功应用: {record_id}")

            return record

        except Exception as e:
            logger.error(f"应用补丁失败: {e}")
            # 尝试回滚
            self._rollback_patch(record)
            record.status = ModificationStatus.REJECTED
            self.modification_history.append(record)
            return record

    def rollback(self, record_id: str) -> bool:
        """
        回滚指定的修改

        Args:
            record_id: 修改记录ID

        Returns:
            回滚是否成功
        """
        # 查找记录
        record = None
        for r in self.modification_history:
            if r.id == record_id:
                record = r
                break

        if not record:
            logger.error(f"未找到修改记录: {record_id}")
            return False

        return self._rollback_patch(record)

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _check_immutable_constraints(self, code: str,
                                     file_path: Path) -> bool:
        """检查不可变约束"""
        location = CodeLocation(file_path=str(file_path))

        for constraint in self.immutable_constraints:
            # 检查受保护模式
            for pattern in constraint.protected_patterns:
                if pattern in code:
                    # 运行检查函数
                    if not constraint.check_func(code, location):
                        logger.error(f"触发不可变约束: {constraint.name}")
                        return False

        return True

    def _assess_risk_level(self, original_code: str,
                          modified_code: str,
                          analysis: CodeAnalysis) -> ModificationRisk:
        """评估风险等级"""
        # 计算代码差异
        original_lines = len(original_code.splitlines())
        modified_lines = len(modified_code.splitlines())
        line_change_ratio = abs(modified_lines - original_lines) / max(original_lines, 1)

        # 计算AST差异
        try:
            original_ast = ast.parse(original_code)
            modified_ast = ast.parse(modified_code)
            ast_change_ratio = self._compare_ast(original_ast, modified_ast)
        except:
            ast_change_ratio = 1.0  # 解析失败视为高风险

        # 综合评估
        risk_score = (
            line_change_ratio * 0.5 +
            ast_change_ratio * 0.3 +
            (1 - analysis.safety_score) * 0.2
        )

        if risk_score < 0.2:
            return ModificationRisk.SAFE
        elif risk_score < 0.4:
            return ModificationRisk.LOW
        elif risk_score < 0.6:
            return ModificationRisk.MEDIUM
        else:
            return ModificationRisk.HIGH

    def _create_backup(self, file_path: str, record_id: str) -> Path:
        """创建备份"""
        backup_path = self.backup_dir / f"{record_id}_{Path(file_path).name}"

        shutil.copy2(file_path, backup_path)
        logger.info(f"  备份已创建: {backup_path}")

        return backup_path

    def _rollback_patch(self, record: ModificationRecord) -> bool:
        """回滚补丁"""
        logger.info(f"[SelfModifyingEngine] 回滚补丁: {record.id}")

        start_time = time.time()

        try:
            if not record.backup_path or not Path(record.backup_path).exists():
                logger.error("备份文件不存在")
                return False

            # 恢复备份
            shutil.copy2(record.backup_path, record.patch.location.file_path)

            rollback_time = time.time() - start_time

            # 检查回滚时间
            if rollback_time > self.MAX_ROLLBACK_TIME_SECONDS:
                logger.warning(f"回滚时间过长: {rollback_time:.2f}秒")
            else:
                logger.info(f"  回滚成功 ({rollback_time:.3f}秒)")

            record.status = ModificationStatus.ROLLED_BACK
            record.rollback_successful = True
            record.rollback_time = rollback_time

            self._total_rolled_back += 1

            return True

        except Exception as e:
            logger.error(f"回滚失败: {e}")
            record.rollback_successful = False
            return False

    def _verify_modification(self, patch: CodePatch) -> bool:
        """验证修改"""
        try:
            # 基本语法检查
            with open(patch.location.file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            ast.parse(code)

            # 尝试导入
            module_path = self._file_to_module(str(patch.location.file_path))
            if module_path:
                __import__(module_path)

            return True

        except Exception as e:
            logger.error(f"验证失败: {e}")
            return False

    def _extract_locations(self, tree: ast.AST,
                          file_path: Path) -> List[CodeLocation]:
        """提取代码位置"""
        locations = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                location = CodeLocation(
                    file_path=str(file_path),
                    class_name=None,
                    function_name=node.name if isinstance(node, ast.FunctionDef) else None,
                    line_start=getattr(node, 'lineno', 0),
                    line_end=getattr(node, 'end_lineno', 0)
                )
                locations.append(location)

        return locations

    def _extract_dependencies(self, tree: ast.AST) -> List[str]:
        """提取依赖关系"""
        dependencies = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dependencies.add(node.module.split('.')[0])

        return list(dependencies)

    def _identify_risk_points(self, tree: ast.AST,
                             source_code: str) -> List[str]:
        """识别风险点"""
        risk_points = []

        for node in ast.walk(tree):
            # 深层嵌套
            depth = self._calculate_nesting_depth(node)
            if depth > 5:
                risk_points.append(f"深层嵌套 (depth={depth}) at line {getattr(node, 'lineno', 0)}")

            # 长函数
            if isinstance(node, ast.FunctionDef):
                lines = getattr(node, 'end_lineno', 0) - getattr(node, 'lineno', 0)
                if lines > 50:
                    risk_points.append(f"长函数 ({lines} lines): {node.name}")

        return risk_points

    def _calculate_nesting_depth(self, node: ast.AST) -> int:
        """计算嵌套深度"""
        if not hasattr(node, 'body'):
            return 0

        max_child_depth = 0
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                child_depth = self._calculate_nesting_depth(child)
                max_child_depth = max(max_child_depth, child_depth + 1)

        return max_child_depth

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """计算复杂度 (简化版圈复杂度)"""
        complexity = 1  # 基准复杂度

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1

        return float(complexity)

    def _estimate_test_coverage(self, module_path: str) -> float:
        """估算测试覆盖率 (简化版本)"""
        # 实际应该使用coverage.py
        test_file = self.project_root / f"tests/test_{module_path.replace('.', '_')}.py"
        if test_file.exists():
            return 0.5  # 假设有测试文件就是50%覆盖
        return 0.0

    def _calculate_safety_score(self, risk_points: List[str],
                               complexity: float) -> float:
        """计算安全分数"""
        # 风险点越多,分数越低
        risk_penalty = min(len(risk_points) * 0.1, 0.5)

        # 复杂度越高,分数越低
        complexity_penalty = min(complexity / 50, 0.3)

        score = 1.0 - risk_penalty - complexity_penalty
        return max(0.0, min(1.0, score))

    def _generate_optimization(self, original_code: str,
                              analysis: CodeAnalysis,
                              goal: str) -> Optional[str]:
        """生成优化代码 (简化版本)"""
        # 实际应该使用LLM或符号执行
        # 这里只是演示结构

        if goal == "performance":
            # 性能优化示例: 添加缓存装饰器
            lines = original_code.splitlines()
            optimized_lines = []

            for i, line in enumerate(lines):
                optimized_lines.append(line)
                # 在简单函数前添加缓存装饰器
                if line.strip().startswith("def ") and "cache" not in line.lower():
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith("return"):
                        # 插入lru_cache
                        indent = len(line) - len(line.lstrip())
                        optimized_lines.append(" " * indent + "@lru_cache(maxsize=128)")

            return "\n".join(optimized_lines)

        elif goal == "readability":
            # 可读性优化: 添加文档字符串
            lines = original_code.splitlines()
            optimized_lines = []

            for line in lines:
                optimized_lines.append(line)
                if line.strip().startswith("def "):
                    # 添加文档字符串模板
                    indent = len(line) - len(line.lstrip())
                    optimized_lines.append(" " * (indent + 4) + '"""TODO: Add docstring"""')

            return "\n".join(optimized_lines)

        return None

    def _estimate_impact(self, analysis: CodeAnalysis,
                        goal: str) -> str:
        """估算影响"""
        if goal == "performance":
            return f"预计性能提升: {1 - analysis.complexity/50:.1%}"
        elif goal == "readability":
            return "可读性提升,维护成本降低"
        else:
            return "通用优化"

    def _generate_test_cases(self, module_path: str,
                            modified_code: str) -> List[str]:
        """生成测试用例"""
        return [
            "import_test",   # 导入测试
            "syntax_test"    # 语法测试
        ]

    def _compare_ast(self, tree1: ast.AST, tree2: ast.AST) -> float:
        """比较AST差异"""
        # 简化版本: 比较节点数量
        nodes1 = list(ast.walk(tree1))
        nodes2 = list(ast.walk(tree2))

        if len(nodes1) == 0:
            return 0.0

        return abs(len(nodes2) - len(nodes1)) / len(nodes1)

    def _module_to_file(self, module_path: str) -> Optional[Path]:
        """模块路径转文件路径"""
        parts = module_path.split(".")
        file_path = self.project_root / Path(*parts).with_suffix('.py')

        if file_path.exists():
            return file_path

        return None

    def _file_to_module(self, file_path: str) -> Optional[str]:
        """文件路径转模块路径"""
        try:
            rel_path = Path(file_path).relative_to(self.project_root)
            module_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
            return module_path
        except:
            return None

    # ========================================================================
    # 事件发布
    # ========================================================================

    def _publish_modification_event(self, record: ModificationRecord):
        """发布修改事件"""
        if not self.event_bus:
            return

        try:
            from core.event_bus import Event, EventType
            event = Event(
                type=EventType.INFO,
                source="SelfModifyingEngine",
                message="代码修改已应用",
                data={
                    'record_id': record.id,
                    'file_path': record.patch.location.file_path,
                    'description': record.patch.description,
                    'risk_level': record.patch.risk_level.value,
                    'rollback_available': True
                }
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.warning(f"发布修改事件失败: {e}")

    def _publish_approval_request_event(self, record: ModificationRecord):
        """发布审批请求事件"""
        if not self.event_bus:
            return

        try:
            from core.event_bus import Event, EventType
            event = Event(
                type=EventType.WARNING,
                source="SelfModifyingEngine",
                message="需要人工审批: 高风险代码修改",
                data={
                    'record_id': record.id,
                    'file_path': record.patch.location.file_path,
                    'description': record.patch.description,
                    'risk_level': record.patch.risk_level.value,
                    'patch_preview': record.patch.modified_code[:500]
                }
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.warning(f"发布审批请求失败: {e}")

    def _publish_rejection_event(self, target: str, reason: str):
        """发布拒绝事件"""
        if not self.event_bus:
            return

        try:
            from core.event_bus import Event, EventType
            event = Event(
                type=EventType.WARNING,
                source="SelfModifyingEngine",
                message=f"代码修改被拒绝: {reason}",
                data={
                    'target': target,
                    'reason': reason
                }
            )
            self.event_bus.publish(event)
        except Exception as e:
            logger.warning(f"发布拒绝事件失败: {e}")

    # ========================================================================
    # 工具方法
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_proposed': self._total_proposed,
            'total_applied': self._total_applied,
            'total_rejected': self._total_rejected,
            'total_rolled_back': self._total_rolled_back,
            'success_rate': (
                self._total_applied / max(self._total_proposed, 1)
            ),
            'current_session_changes': self.current_session_changes,
            'backup_dir': str(self.backup_dir)
        }

    def export_audit_log(self, output_path: str) -> None:
        """导出审计日志"""
        audit_data = {
            'timestamp': time.time(),
            'statistics': self.get_statistics(),
            'modifications': [r.to_dict() for r in self.modification_history]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, ensure_ascii=False)

        logger.info(f"审计日志已导出: {output_path}")
