#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组件依赖图谱映射器 (Component Dependency Mapper)
=================================================

架构感知层第一组件：分析系统组件间的依赖关系

功能：
- 扫描系统代码结构
- 分析import依赖关系
- 构建依赖图
- 检测循环依赖
- 识别关键路径
- 生成依赖报告

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

import os
import re
import ast
import json
import time
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path


class DependencyType(Enum):
    """依赖类型"""
    INTERNAL = "internal"      # 内部模块依赖
    EXTERNAL = "external"      # 外部库依赖
    STANDARD = "standard"      # 标准库依赖
    DYNAMIC = "dynamic"        # 动态导入依赖


class ComponentType(Enum):
    """组件类型"""
    LAYER = "layer"            # 层级组件（如Layer 0-6）
    MODULE = "module"          # 模块组件
    CLASS = "class"            # 类组件
    FUNCTION = "function"      # 函数组件
    UTILITY = "utility"        # 工具组件


@dataclass
class DependencyNode:
    """依赖节点"""
    name: str
    path: str
    component_type: ComponentType
    layer: Optional[int] = None  # 所属层级（0-6）
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)
    size_lines: int = 0
    complexity: float = 0.0  # 0.0-1.0


@dataclass
class DependencyEdge:
    """依赖边"""
    from_node: str
    to_node: str
    dependency_type: DependencyType
    strength: float  # 0.0-1.0 (依赖强度)
    import_count: int = 0
    line_numbers: List[int] = field(default_factory=list)


@dataclass
class CircularDependency:
    """循环依赖"""
    cycle: List[str]  # 循环路径
    severity: str  # low, medium, high, critical
    impact: str


@dataclass
class CriticalPath:
    """关键路径"""
    path: List[str]
    importance: float  # 0.0-1.0
    bottleneck_risk: float  # 0.0-1.0


@dataclass
class DependencyAnalysis:
    """依赖分析结果"""
    total_components: int
    total_dependencies: int
    internal_dependencies: int
    external_dependencies: int
    circular_dependencies: List[CircularDependency]
    critical_paths: List[CriticalPath]
    dependency_depth: Dict[str, int]
    layer_violations: List[str]  # 违反分层架构的依赖
    orphan_components: List[str]  # 未被任何组件依赖的孤立组件
    god_components: List[str]  # 被过多组件依赖的核心组件
    analysis_timestamp: float


class ComponentDependencyMapper:
    """
    组件依赖图谱映射器

    核心功能：
    1. 扫描系统代码，分析import依赖
    2. 构建组件依赖图
    3. 检测架构问题（循环依赖、层级违规等）
    4. 识别关键路径和核心组件
    5. 生成依赖分析报告
    """

    def __init__(self, project_root: str):
        """
        初始化依赖映射器

        Args:
            project_root: 项目根目录
        """
        self.project_root = Path(project_root)

        # 依赖图谱
        self.nodes: Dict[str, DependencyNode] = {}
        self.edges: List[DependencyEdge] = []

        # 配置
        self.exclude_dirs = {
            '__pycache__', '.git', 'venv', 'env', '.venv',
            'node_modules', 'dist', 'build', '.pytest_cache',
            'data', 'logs', 'workspace'
        }

        self.exclude_files = {
            '*.pyc', '*.pyo', '*.pyd', '__pycache__',
            '.gitignore', '*.md', '*.txt'
        }

        # AGI系统层级定义
        self.layer_structure = {
            0: ["core/immutable_core.py"],  # 不变核心
            1: ["core/memory", "core/knowledge"],  # 记忆与知识
            2: ["core/working_memory", "core/meta_cognitive"],  # 认知处理
            3: ["core/agents"],  # 智能体层
            4: ["core/evolution"],  # 进化层
            5: ["core/skills"],  # 技能层
            6: ["AGI_Life_Engine.py", "core/bridges"],  # 应用层
        }

    def analyze(self, include_external: bool = False) -> DependencyAnalysis:
        """
        执行完整的依赖分析

        Args:
            include_external: 是否包含外部依赖

        Returns:
            DependencyAnalysis: 完整的依赖分析结果
        """
        print(f"\n{'='*60}")
        print(f"[ArchitectureAwareness] 组件依赖图谱分析")
        print(f"{'='*60}")
        print(f"项目根目录: {self.project_root}")

        start_time = time.time()

        # 1. 扫描Python文件
        print(f"\n[步骤 1/6] 扫描Python文件...")
        python_files = self._scan_python_files()
        print(f"  发现 {len(python_files)} 个Python文件")

        # 2. 解析依赖关系
        print(f"\n[步骤 2/6] 解析依赖关系...")
        self._parse_dependencies(python_files)
        print(f"  解析了 {len(self.nodes)} 个组件")
        print(f"  发现 {len(self.edges)} 条依赖关系")

        # 3. 检测循环依赖
        print(f"\n[步骤 3/6] 检测循环依赖...")
        circular_deps = self._detect_circular_dependencies()
        print(f"  发现 {len(circular_deps)} 个循环依赖")

        # 4. 计算依赖深度
        print(f"\n[步骤 4/6] 计算依赖深度...")
        dependency_depth = self._calculate_dependency_depth()
        print(f"  最大依赖深度: {max(dependency_depth.values()) if dependency_depth else 0}")

        # 5. 识别关键路径
        print(f"\n[步骤 5/6] 识别关键路径...")
        critical_paths = self._identify_critical_paths()
        print(f"  识别 {len(critical_paths)} 条关键路径")

        # 6. 检测架构违规
        print(f"\n[步骤 6/6] 检测架构违规...")
        layer_violations = self._detect_layer_violations()
        orphan_components = self._identify_orphan_components()
        god_components = self._identify_god_components()
        print(f"  层级违规: {len(layer_violations)}")
        print(f"  孤立组件: {len(orphan_components)}")
        print(f"  核心组件: {len(god_components)}")

        # 统计依赖类型
        internal_deps = sum(1 for e in self.edges if e.dependency_type == DependencyType.INTERNAL)
        external_deps = sum(1 for e in self.edges if e.dependency_type == DependencyType.EXTERNAL)

        duration = time.time() - start_time

        # 构建分析结果
        analysis = DependencyAnalysis(
            total_components=len(self.nodes),
            total_dependencies=len(self.edges),
            internal_dependencies=internal_deps,
            external_dependencies=external_deps,
            circular_dependencies=circular_deps,
            critical_paths=critical_paths[:10],  # 只保留前10条
            dependency_depth=dependency_depth,
            layer_violations=layer_violations,
            orphan_components=orphan_components,
            god_components=god_components,
            analysis_timestamp=time.time()
        )

        # 打印分析报告
        self._print_analysis_report(analysis, duration)

        return analysis

    def _scan_python_files(self) -> List[Path]:
        """扫描所有Python文件"""
        python_files = []

        for root, dirs, files in os.walk(self.project_root):
            # 排除特定目录
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    python_files.append(file_path)

        return python_files

    def _parse_dependencies(self, python_files: List[Path]):
        """解析Python文件的依赖关系"""
        for file_path in python_files:
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 解析AST
                tree = ast.parse(content, filename=str(file_path))

                # 创建节点
                relative_path = file_path.relative_to(self.project_root)
                node_name = str(relative_path).replace('/', '.').replace('\\', '.').replace('.py', '')

                # 确定组件类型和层级
                component_type, layer = self._classify_component(file_path)

                node = DependencyNode(
                    name=node_name,
                    path=str(relative_path),
                    component_type=component_type,
                    layer=layer,
                    size_lines=len(content.splitlines()),
                    complexity=self._calculate_complexity(content)
                )

                # 解析imports
                for node_ast in ast.walk(tree):
                    if isinstance(node_ast, ast.Import):
                        for alias in node_ast.names:
                            node.imports.append(alias.name)
                    elif isinstance(node_ast, ast.ImportFrom):
                        if node_ast.module:
                            node.imports.append(node_ast.module)

                self.nodes[node_name] = node

                # 创建依赖边
                for imp in node.imports:
                    dep_type = self._classify_dependency(imp)
                    edge = DependencyEdge(
                        from_node=node_name,
                        to_node=imp,
                        dependency_type=dep_type,
                        strength=0.5,  # 默认强度
                        import_count=1
                    )
                    self.edges.append(edge)

            except Exception as e:
                # 解析失败，跳过该文件
                continue

        # 构建反向依赖关系
        for edge in self.edges:
            if edge.to_node in self.nodes:
                if edge.from_node not in self.nodes[edge.to_node].imported_by:
                    self.nodes[edge.to_node].imported_by.append(edge.from_node)

    def _classify_component(self, file_path: Path) -> Tuple[ComponentType, Optional[int]]:
        """分类组件并确定所属层级"""
        path_str = str(file_path.relative_to(self.project_root))

        # 确定层级
        layer = None
        for layer_num, patterns in self.layer_structure.items():
            for pattern in patterns:
                if pattern in path_str:
                    layer = layer_num
                    break

        # 确定组件类型
        if 'agent' in path_str.lower():
            component_type = ComponentType.CLASS
        elif 'core' in path_str.lower():
            component_type = ComponentType.MODULE
        elif 'util' in path_str.lower():
            component_type = ComponentType.UTILITY
        else:
            component_type = ComponentType.MODULE

        return component_type, layer

    def _classify_dependency(self, import_name: str) -> DependencyType:
        """分类依赖类型"""
        # 标准库
        standard_libs = {'os', 'sys', 're', 'json', 'time', 'datetime', 'pathlib', 'collections'}
        if import_name.split('.')[0] in standard_libs:
            return DependencyType.STANDARD

        # 内部模块
        if import_name.startswith('core.') or import_name.startswith('AGI'):
            return DependencyType.INTERNAL

        # 外部库
        return DependencyType.EXTERNAL

    def _calculate_complexity(self, content: str) -> float:
        """计算代码复杂度（简化版）"""
        lines = content.splitlines()
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]

        if not code_lines:
            return 0.0

        # 简单指标：代码行数 + 嵌套层次
        complexity = min(len(code_lines) / 1000.0, 1.0)

        # 检测嵌套
        max_indent = 0
        for line in code_lines:
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)
        complexity += min(max_indent / 100.0, 0.5)

        return min(complexity, 1.0)

    def _detect_circular_dependencies(self) -> List[CircularDependency]:
        """检测循环依赖"""
        circular_deps = []
        visited = set()
        rec_stack = set()
        path = []

        # 只分析内部依赖
        internal_edges = [
            (e.from_node, e.to_node)
            for e in self.edges
            if e.dependency_type == DependencyType.INTERNAL
            and e.to_node in self.nodes
        ]

        # 构建邻接表
        graph = defaultdict(list)
        for from_node, to_node in internal_edges:
            graph[from_node].append(to_node)

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # 找到循环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]

                    # 评估严重程度
                    severity = "low"
                    if len(cycle) <= 2:
                        severity = "critical"
                    elif len(cycle) <= 3:
                        severity = "high"
                    elif len(cycle) <= 5:
                        severity = "medium"

                    circular_deps.append(CircularDependency(
                        cycle=cycle,
                        severity=severity,
                        impact=f"{'严重' if severity in ['high', 'critical'] else '轻微'}影响可维护性"
                    ))

            path.pop()
            rec_stack.remove(node)
            return False

        for node in list(graph.keys()):
            if node not in visited:
                dfs(node)

        return circular_deps

    def _calculate_dependency_depth(self) -> Dict[str, int]:
        """计算每个节点的依赖深度"""
        depth = {}

        # 构建邻接表（只考虑内部依赖）
        graph = defaultdict(list)
        for edge in self.edges:
            if edge.dependency_type == DependencyType.INTERNAL and edge.to_node in self.nodes:
                graph[edge.from_node].append(edge.to_node)

        def calculate_depth(node):
            if node in depth:
                return depth[node]

            if node not in graph or not graph[node]:
                depth[node] = 0
                return 0

            max_child_depth = 0
            for child in graph[node]:
                child_depth = calculate_depth(child)
                max_child_depth = max(max_child_depth, child_depth)

            depth[node] = max_child_depth + 1
            return depth[node]

        for node in self.nodes:
            calculate_depth(node)

        return depth

    def _identify_critical_paths(self) -> List[CriticalPath]:
        """识别关键路径（被最多组件依赖的路径）"""
        # 统计每个节点的被依赖次数
        dependency_count = defaultdict(int)
        for edge in self.edges:
            if edge.dependency_type == DependencyType.INTERNAL and edge.to_node in self.nodes:
                dependency_count[edge.to_node] += 1

        # 按依赖次数排序
        sorted_nodes = sorted(dependency_count.items(), key=lambda x: x[1], reverse=True)

        # 构建关键路径
        critical_paths = []
        for node, count in sorted_nodes[:10]:  # 前10个
            if count > 0:
                importance = min(count / 20.0, 1.0)  # 归一化
                bottleneck_risk = self.nodes[node].complexity

                # 追溯路径
                path = [node]
                current = node
                visited = {node}

                for _ in range(5):  # 最多追溯5层
                    # 找到当前节点依赖的最关键节点
                    dependencies = [
                        e for e in self.edges
                        if e.from_node == current
                        and e.dependency_type == DependencyType.INTERNAL
                        and e.to_node in self.nodes
                        and e.to_node not in visited
                    ]

                    if not dependencies:
                        break

                    # 选择被依赖次数最多的
                    dep_counts = {e.to_node: dependency_count.get(e.to_node, 0) for e in dependencies}
                    next_node = max(dep_counts.items(), key=lambda x: x[1])[0]

                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

                critical_paths.append(CriticalPath(
                    path=path,
                    importance=importance,
                    bottleneck_risk=bottleneck_risk
                ))

        return critical_paths

    def _detect_layer_violations(self) -> List[str]:
        """检测层级架构违规（下层依赖上层）"""
        violations = []

        for edge in self.edges:
            if edge.dependency_type != DependencyType.INTERNAL:
                continue

            from_node = self.nodes.get(edge.from_node)
            to_node = self.nodes.get(edge.to_node)

            if not from_node or not to_node:
                continue

            if from_node.layer is not None and to_node.layer is not None:
                # 下层不应该依赖上层
                if from_node.layer < to_node.layer:
                    violations.append(
                        f"层级违规: {edge.from_node} (Layer {from_node.layer}) "
                        f"-> {edge.to_node} (Layer {to_node.layer})"
                    )

        return violations

    def _identify_orphan_components(self) -> List[str]:
        """识别孤立组件（未被任何组件依赖）"""
        orphans = []

        for node_name, node in self.nodes.items():
            if not node.imported_by:
                # 排除主入口文件
                if not any(x in node_name for x in ['AGI_Life_Engine', 'main', '__init__']):
                    orphans.append(node_name)

        return orphans

    def _identify_god_components(self) -> List[str]:
        """识别核心组件（被过多组件依赖）"""
        gods = []

        for node_name, node in self.nodes.items():
            # 被超过10个组件依赖视为核心组件
            if len(node.imported_by) > 10:
                gods.append(f"{node_name} (被{len(node.imported_by)}个组件依赖)")

        return gods

    def _print_analysis_report(self, analysis: DependencyAnalysis, duration: float):
        """打印分析报告"""
        print(f"\n{'─'*60}")
        print(f"[依赖分析报告]")
        print(f"{'─'*60}")

        print(f"\n📊 统计概览:")
        print(f"  • 总组件数: {analysis.total_components}")
        print(f"  • 总依赖数: {analysis.total_dependencies}")
        print(f"  • 内部依赖: {analysis.internal_dependencies}")
        print(f"  • 外部依赖: {analysis.external_dependencies}")
        print(f"  • 分析耗时: {duration:.2f}秒")

        if analysis.circular_dependencies:
            print(f"\n⚠️  循环依赖 ({len(analysis.circular_dependencies)}个):")
            for i, dep in enumerate(analysis.circular_dependencies[:5], 1):
                print(f"  {i}. {' -> '.join(dep.cycle)}")
                print(f"     严重度: {dep.severity} | 影响: {dep.impact}")
        else:
            print(f"\n✅ 未发现循环依赖")

        if analysis.critical_paths:
            print(f"\n🔥 关键路径 (Top {min(5, len(analysis.critical_paths))}):")
            for i, path in enumerate(analysis.critical_paths[:5], 1):
                print(f"  {i}. 重要性: {path.importance:.2%} | 风险: {path.bottleneck_risk:.2%}")
                print(f"     路径: {' -> '.join(path.path[:3])}...")

        if analysis.layer_violations:
            print(f"\n❌ 层级违规 ({len(analysis.layer_violations)}个):")
            for violation in analysis.layer_violations[:5]:
                print(f"  • {violation}")
        else:
            print(f"\n✅ 无层级违规")

        if analysis.orphan_components:
            print(f"\n👻 孤立组件 ({len(analysis.orphan_components)}个):")
            for orphan in analysis.orphan_components[:5]:
                print(f"  • {orphan}")

        if analysis.god_components:
            print(f"\n👑 核心组件 ({len(analysis.god_components)}个):")
            for god in analysis.god_components[:5]:
                print(f"  • {god}")

        print(f"\n{'='*60}")

        # 关键输出：架构健康度评估
        health_score = 1.0
        if analysis.circular_dependencies:
            health_score -= 0.2 * len(analysis.circular_dependencies)
        if analysis.layer_violations:
            health_score -= 0.1 * len(analysis.layer_violations)

        health_score = max(0.0, min(1.0, health_score))

        if health_score > 0.8:
            print(f"[ArchitectureAwareness] ✅ 架构健康度: {health_score:.2%} (优秀)")
        elif health_score > 0.6:
            print(f"[ArchitectureAwareness] ⚠️  架构健康度: {health_score:.2%} (良好)")
        elif health_score > 0.4:
            print(f"[ArchitectureAwareness] ⚠️  架构健康度: {health_score:.2%} (一般)")
        else:
            print(f"[ArchitectureAwareness] 🔴 架构健康度: {health_score:.2%} (需改进)")

    def export_graph(self, output_path: str):
        """导出依赖图到JSON文件"""
        graph_data = {
            "nodes": [
                {
                    "name": node.name,
                    "path": node.path,
                    "type": node.component_type.value,
                    "layer": node.layer,
                    "size": node.size_lines,
                    "complexity": node.complexity
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "from": edge.from_node,
                    "to": edge.to_node,
                    "type": edge.dependency_type.value,
                    "strength": edge.strength
                }
                for edge in self.edges
                if edge.dependency_type == DependencyType.INTERNAL
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        print(f"[ArchitectureAwareness] 📁 依赖图已导出: {output_path}")


# ============ 使用示例 ============

if __name__ == "__main__":
    print("="*60)
    print("组件依赖图谱映射器测试")
    print("="*60)

    mapper = ComponentDependencyMapper(
        project_root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    # 执行分析
    analysis = mapper.analyze()

    # 导出图谱
    mapper.export_graph("data/architecture/dependency_graph.json")

    print("\n✅ 依赖分析完成！")
