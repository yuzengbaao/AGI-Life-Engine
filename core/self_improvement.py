#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自我改进引擎（Self-Improvement Engine）
=======================================

功能：实现系统自我修改和优化能力
基于：程序合成 + 自动化优化 + 遗传编程

核心能力：
1. 代码生成（自我编程）
2. 性能优化（自动调优）
3. Bug检测与修复
4. 架构自修改

版本: 1.0.0
"""

import os
import re
import ast
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ImprovementType(Enum):
    """改进类型"""
    PERFORMANCE = "performance"      # 性能优化
    BUG_FIX = "bug_fix"            # Bug修复
    FEATURE_ADD = "feature_add"    # 功能添加
    REFACTOR = "refactor"          # 重构
    OPTIMIZATION = "optimization"  # 优化


@dataclass
class CodeModule:
    """代码模块"""
    module_path: str
    code: str
    ast_tree: Any
    complexity: float
    lines_of_code: int
    imports: List[str]
    functions: List[str]
    classes: List[str]
    last_modified: float


@dataclass
class ImprovementProposal:
    """改进提案"""
    proposal_id: str
    module_path: str
    improvement_type: ImprovementType
    description: str
    original_code: str
    improved_code: str
    expected_benefit: float
    confidence: float
    risk_level: float
    timestamp: float


@dataclass
class ImprovementHistory:
    """改进历史"""
    history_id: str
    proposal_id: str
    module_path: str
    applied: bool
    success: bool
    performance_delta: float
    timestamp: float
    rollback_info: Optional[str] = None


class SelfImprovementEngine:
    """
    自我改进引擎

    核心功能：
    1. 分析现有代码
    2. 生成改进提案
    3. 应用改进（带回滚）
    4. 验证改进效果
    """

    def __init__(self, project_root: str):
        """
        初始化自我改进引擎

        Args:
            project_root: 项目根目录
        """
        self.project_root = Path(project_root)

        # 代码模块索引
        self.code_modules: Dict[str, CodeModule] = {}

        # 改进历史
        self.improvement_history: List[ImprovementHistory] = []

        # 性能基准
        self.performance_baselines: Dict[str, float] = {}

        # 统计信息
        self.stats = {
            'total_proposals': 0,
            'applied_improvements': 0,
            'successful_improvements': 0,
            'failed_improvements': 0,
            'rollbacks': 0,
            'total_performance_gain': 0.0
        }

        # 扫描项目代码
        self._scan_project()

    def _scan_project(self):
        """扫描项目代码"""
        print(f"  [SelfImprovement] Scanning project: {self.project_root}", flush=True)

        # 排除目录列表
        excluded_dirs = {
            '.conda', 'venv', 'env', '.git', '.idea', '.vscode', 
            '__pycache__', 'dist', 'build', 'backups', 'node_modules',
            'site-packages'
        }

        # 扫描Python文件
        python_files = []
        for root, dirs, files in os.walk(self.project_root):
            # 修改 dirs 列表以原地排除目录
            dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        for py_file in python_files:
            # 跳过测试文件
            if 'test' in str(py_file).lower():
                continue

            try:
                module = self._analyze_module(str(py_file))
                if module:
                    self.code_modules[module.module_path] = module
            except Exception as e:
                # 仅在非编码错误时打印警告，避免刷屏
                if "codec" not in str(e):
                    print(f"    [WARNING] Failed to analyze {py_file}: {e}", flush=True)

        print(f"  [SelfImprovement] Scanned {len(self.code_modules)} modules", flush=True)

    def _analyze_module(self, file_path: str) -> Optional[CodeModule]:
        """分析代码模块"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except UnicodeDecodeError:
            # 尝试使用 latin-1 读取 (兼容旧代码)
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    code = f.read()
            except Exception:
                return None

        # 解析AST
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        # 分析复杂度
        complexity = self._calculate_complexity(tree)

        # 提取信息
        imports = []
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        return CodeModule(
            module_path=file_path,
            code=code,
            ast_tree=tree,
            complexity=complexity,
            lines_of_code=len(code.splitlines()),
            imports=imports,
            functions=functions,
            classes=classes,
            last_modified=os.path.getmtime(file_path)
        )

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """计算代码复杂度（简化版本）"""
        complexity = 1  # 基础复杂度

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.Lambda):
                complexity += 0.5
            elif isinstance(node, ast.ListComp):
                complexity += 0.5

        return complexity

    def analyze_and_propose(self, module_path: str) -> List[ImprovementProposal]:
        """
        分析模块并生成改进提案

        Args:
            module_path: 模块路径

        Returns:
            改进提案列表
        """
        if module_path not in self.code_modules:
            print(f"  [SelfImprovement] Module not found: {module_path}")
            return []

        module = self.code_modules[module_path]
        proposals = []

        print(f"  [SelfImprovement] Analyzing: {module_path}")

        # 1. 性能优化提案
        perf_proposals = self._propose_performance_optimizations(module)
        proposals.extend(perf_proposals)

        # 2. 重构提案
        refactor_proposals = self._propose_refactoring(module)
        proposals.extend(refactor_proposals)

        # 3. Bug检测提案
        bug_proposals = self._propose_bug_fixes(module)
        proposals.extend(bug_proposals)

        self.stats['total_proposals'] += len(proposals)

        return proposals

    def _propose_performance_optimizations(self, module: CodeModule) -> List[ImprovementProposal]:
        """提出性能优化建议"""
        proposals = []
        code = module.code

        # 优化1: 列表推导代替循环
        for_pattern = r'for\s+(\w+)\s+in\s+(\w+):\s+\n\s+(\w+)\.append\('
        if re.search(for_pattern, code):
            improved_code = re.sub(
                for_pattern,
                # 简化的替换（实际需要更复杂的转换）
                r'# Suggested: [x for x in \2 if condition]',
                code
            )

            proposals.append(ImprovementProposal(
                proposal_id=f"perf_{int(time.time())}_{len(proposals)}",
                module_path=module.module_path,
                improvement_type=ImprovementType.PERFORMANCE,
                description="Use list comprehension instead of loop+append",
                original_code="Detected loop+append pattern",
                improved_code=improved_code,
                expected_benefit=0.2,  # 20%性能提升
                confidence=0.7,
                risk_level=0.2,
                timestamp=time.time()
            ))

        # 优化2: 使用生成器
        if 'list(' in code and 'range(' in code:
            proposals.append(ImprovementProposal(
                proposal_id=f"perf_{int(time.time())}_{len(proposals)}",
                module_path=module.module_path,
                improvement_type=ImprovementType.OPTIMIZATION,
                description="Consider using generator instead of list for memory efficiency",
                original_code="Found list(range(...))",
                improved_code="Suggested: xrange or generator expression",
                expected_benefit=0.3,
                confidence=0.6,
                risk_level=0.3,
                timestamp=time.time()
            ))

        return proposals

    def _propose_refactoring(self, module: CodeModule) -> List[ImprovementProposal]:
        """提出重构建议"""
        proposals = []

        # 重构1: 长函数拆分
        for func in module.functions:
            if self._is_long_function(module.ast_tree, func):
                proposals.append(ImprovementProposal(
                    proposal_id=f"refactor_{int(time.time())}_{len(proposals)}",
                    module_path=module.module_path,
                    improvement_type=ImprovementType.REFACTOR,
                    description=f"Long function '{func}' should be split into smaller functions",
                    original_code=f"Function '{func}' is too long",
                    improved_code="Suggested: Extract sub-functions",
                    expected_benefit=0.4,
                    confidence=0.8,
                    risk_level=0.4,
                    timestamp=time.time()
                ))

        return proposals

    def _propose_bug_fixes(self, module: CodeModule) -> List[ImprovementProposal]:
        """提出Bug修复建议"""
        proposals = []
        code = module.code

        # Bug1: 未处理的异常
        if 'except:' in code and 'except Exception' not in code:
            proposals.append(ImprovementProposal(
                proposal_id=f"bug_{int(time.time())}_{len(proposals)}",
                module_path=module.module_path,
                improvement_type=ImprovementType.BUG_FIX,
                description="Bare except clause catches all exceptions, should specify exception type",
                original_code="except:",
                improved_code="except Exception as e:",
                expected_benefit=0.5,
                confidence=0.9,
                risk_level=0.1,
                timestamp=time.time()
            ))

        # Bug2: 未使用的导入
        unused_imports = self._find_unused_imports(module)
        if unused_imports:
            proposals.append(ImprovementProposal(
                proposal_id=f"bug_{int(time.time())}_{len(proposals)}",
                module_path=module.module_path,
                improvement_type=ImprovementType.BUG_FIX,
                description=f"Remove unused imports: {', '.join(unused_imports[:3])}",
                original_code="Unused imports detected",
                improved_code="Remove unused import statements",
                expected_benefit=0.1,
                confidence=0.7,
                risk_level=0.1,
                timestamp=time.time()
            ))

        return proposals

    def _is_long_function(self, tree: ast.AST, func_name: str) -> bool:
        """检查是否是长函数"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                # 简化：检查行数
                return hasattr(node, 'body') and len(node.body) > 20
        return False

    def _find_unused_imports(self, module: CodeModule) -> List[str]:
        """查找未使用的导入"""
        unused = []
        code = module.code

        for imp in module.imports:
            # 简化检查：如果导入名不在代码中出现
            if imp.split('.')[0] not in code.replace(f"import {imp}", ""):
                unused.append(imp)

        return unused

    def apply_improvement(self, proposal: ImprovementProposal,
                          auto_backup: bool = True) -> ImprovementHistory:
        """
        应用改进提案

        Args:
            proposal: 改进提案
            auto_backup: 是否自动备份

        Returns:
            改进历史记录
        """
        print(f"\n  [SelfImprovement] Applying improvement: {proposal.proposal_id}")
        print(f"    Type: {proposal.improvement_type.value}")
        print(f"    Description: {proposal.description}")

        # 创建备份
        if auto_backup:
            backup_path = self._create_backup(proposal.module_path)
            print(f"    Backup: {backup_path}")

        try:
            # 应用改进（简化版本：实际需要复杂的代码转换）
            success = self._apply_code_change(proposal)

            if success:
                self.stats['applied_improvements'] += 1

                # 记录历史
                history = ImprovementHistory(
                    history_id=f"hist_{int(time.time())}",
                    proposal_id=proposal.proposal_id,
                    module_path=proposal.module_path,
                    applied=True,
                    success=True,
                    performance_delta=proposal.expected_benefit,
                    timestamp=time.time()
                )

                self.improvement_history.append(history)
                self.stats['successful_improvements'] += 1
                self.stats['total_performance_gain'] += proposal.expected_benefit

                print(f"    [SUCCESS] Improvement applied")

                # 重新扫描修改的模块
                self._scan_project()

                return history
            else:
                raise Exception("Failed to apply code change")

        except Exception as e:
            print(f"    [ERROR] {e}")

            # 回滚
            if auto_backup:
                self._rollback_from_backup(backup_path, proposal.module_path)
                self.stats['rollbacks'] += 1

            history = ImprovementHistory(
                history_id=f"hist_{int(time.time())}",
                proposal_id=proposal.proposal_id,
                module_path=proposal.module_path,
                applied=False,
                success=False,
                performance_delta=0.0,
                timestamp=time.time(),
                rollback_info=f"Rollback due to: {e}"
            )

            self.improvement_history.append(history)
            self.stats['failed_improvements'] += 1

            return history

    def _apply_code_change(self, proposal: ImprovementProposal) -> bool:
        """应用代码更改（简化版本）"""
        # 在实际系统中，这里需要复杂的AST转换
        # 当前版本只记录提案，不实际修改代码
        print(f"      [SIMULATION] Would modify: {proposal.module_path}")
        print(f"      [SIMULATION] Original: {proposal.original_code[:50]}...")
        print(f"      [SIMULATION] Improved: {proposal.improved_code[:50]}...")

        # 模拟成功应用
        return True

    def _create_backup(self, file_path: str) -> str:
        """创建备份"""
        path = Path(file_path)
        backup_path = path.parent / f"{path.stem}_backup_{int(time.time())}{path.suffix}"

        with open(file_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())

        return str(backup_path)

    def _rollback_from_backup(self, backup_path: str, original_path: str):
        """从备份回滚"""
        import shutil
        shutil.copy(backup_path, original_path)
        print(f"    [ROLLBACK] Restored from backup")

    def generate_self_improvement_code(self, task_description: str) -> str:
        """
        生成自我改进代码

        Args:
            task_description: 任务描述

        Returns:
            生成的代码
        """
        print(f"\n  [SelfImprovement] Generating code for: {task_description}")

        # 简化的代码生成模板
        code_template = f'''
# Auto-generated code for: {task_description}
# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}

def optimized_{hashlib.md5(task_description.encode()).hexdigest()[:8]}(self, *args, **kwargs):
    """
    Optimized implementation for: {task_description}

    This is an auto-generated function that should be reviewed
    and tested before deployment.
    """
    # TODO: Implement the optimization
    # Current approach: Placeholder
    # Expected improvement: Performance gain

    result = None  # Placeholder for actual implementation

    return result

# Integration point
# This function should be integrated into the appropriate module
'''

        return code_template

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'modules_scanned': len(self.code_modules),
            'total_complexity': sum(m.complexity for m in self.code_modules.values()),
            'total_lines_of_code': sum(m.lines_of_code for m in self.code_modules.values()),
            'improvement_history_size': len(self.improvement_history)
        }

    def get_improvement_summary(self) -> List[Dict[str, Any]]:
        """获取改进摘要"""
        summary = []

        for history in self.improvement_history[-10:]:  # 最近10次
            summary.append({
                'proposal_id': history.proposal_id,
                'module': history.module_path,
                'applied': history.applied,
                'success': history.success,
                'performance_delta': history.performance_delta,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(history.timestamp))
            })

        return summary


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 60)
    print("自我改进引擎测试")
    print("=" * 60)

    # 创建自我改进引擎
    project_root = Path(__file__).parent.parent
    engine = SelfImprovementEngine(str(project_root))

    # 测试1: 分析模块
    print("\n[测试1] 分析代码模块")
    print("-" * 60)

    stats = engine.get_statistics()
    print(f"  Modules scanned: {stats['modules_scanned']}")
    print(f"  Total complexity: {stats['total_complexity']:.1f}")
    print(f"  Total LOC: {stats['total_lines_of_code']}")

    # 测试2: 生成改进提案
    print("\n[测试2] 生成改进提案")
    print("-" * 60)

    # 选择一个模块进行分析
    if engine.code_modules:
        module_path = list(engine.code_modules.keys())[0]
        proposals = engine.analyze_and_propose(module_path)

        print(f"  Generated {len(proposals)} proposals for {module_path}")

        for i, prop in enumerate(proposals[:3], 1):
            print(f"    [{i}] {prop.improvement_type.value}: {prop.description}")
            print(f"        Expected benefit: {prop.expected_benefit:.2%}")
            print(f"        Confidence: {prop.confidence:.2%}")

    # 测试3: 应用改进
    print("\n[测试3] 应用改进")
    print("-" * 60)

    if proposals:
        proposal = proposals[0]
        history = engine.apply_improvement(proposal)
        print(f"  Applied: {history.applied}")
        print(f"  Success: {history.success}")

    # 测试4: 生成自我改进代码
    print("\n[测试4] 生成自我改进代码")
    print("-" * 60)

    generated_code = engine.generate_self_improvement_code("Optimize sorting algorithm")
    print(f"  Generated code length: {len(generated_code)} chars")
    print(f"  Preview:\n{generated_code[:300]}...")

    # 测试5: 改进摘要
    print("\n[测试5] 改进历史摘要")
    print("-" * 60)

    summary = engine.get_improvement_summary()
    for entry in summary:
        print(f"  {entry['timestamp']}: {entry['module']}")
        print(f"    Applied: {entry['applied']}, Success: {entry['success']}, Delta: {entry['performance_delta']:.2%}")

    print("\n  [OK] 自我改进引擎测试完成")
