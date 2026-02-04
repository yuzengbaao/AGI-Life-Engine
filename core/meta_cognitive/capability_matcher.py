#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
能力匹配分析器 (Capability Matcher)
======================================

元认知层第二组件：评估系统能力与任务的匹配程度

功能：
- 能力与任务匹配分析（我能解决这个问题吗？）
- 能力边界检测（我的局限在哪里？）
- 缺失能力识别（我缺少什么能力？）
- 能力相似度搜索（我有类似的能力吗？）

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class MatchLevel(Enum):
    """匹配程度"""
    PERFECT = "perfect"      # 完美匹配（直接有对应能力）
    GOOD = "good"           # 良好匹配（有相似能力）
    PARTIAL = "partial"     # 部分匹配（需要组合能力）
    POOR = "poor"          # 匹配度低（勉强能做）
    NONE = "none"          # 无匹配（无法完成）


@dataclass
class CapabilityProfile:
    """能力画像"""
    name: str
    category: str  # cognitive, tool, domain, knowledge
    strength: float  # 0.0-1.0
    versatility: float  # 0.0-1.0 (通用性)
    dependencies: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    use_cases: List[str] = field(default_factory=list)


@dataclass
class MatchResult:
    """匹配结果"""
    task: str
    match_level: MatchLevel
    matching_capabilities: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)
    capability_gaps: List[str] = field(default_factory=list)
    suggested_alternatives: List[str] = field(default_factory=list)
    confidence: float = 0.0
    workarounds: List[str] = field(default_factory=list)


class CapabilityMatcher:
    """
    能力匹配分析器

    核心功能：
    1. 维护系统能力注册表
    2. 分析任务需求与系统能力的匹配度
    3. 识别能力边界
    4. 提供替代方案和工作建议
    """

    def __init__(self):
        """初始化能力匹配器"""
        # 系统能力注册表
        self.capabilities = self._initialize_capabilities()

        # 能力依赖关系
        self.dependency_graph = self._build_dependency_graph()

        # 能力相似度矩阵
        self.similarity_matrix = self._build_similarity_matrix()

    def _initialize_capabilities(self) -> Dict[str, CapabilityProfile]:
        """初始化系统能力注册表"""
        return {
            # 认知能力
            "text_understanding": CapabilityProfile(
                name="文本理解",
                category="cognitive",
                strength=0.9,
                versatility=0.95,
                use_cases=["文档阅读", "信息提取", "文本分析"]
            ),
            "code_analysis": CapabilityProfile(
                name="代码分析",
                category="cognitive",
                strength=0.85,
                versatility=0.8,
                dependencies=["text_understanding"],
                use_cases=["代码审查", "bug分析", "重构建议"]
            ),
            "logical_reasoning": CapabilityProfile(
                name="逻辑推理",
                category="cognitive",
                strength=0.75,
                versatility=0.85,
                use_cases=["任务规划", "问题分解", "因果推理"]
            ),
            "pattern_recognition": CapabilityProfile(
                name="模式识别",
                category="cognitive",
                strength=0.8,
                versatility=0.9,
                use_cases=["数据分析", "异常检测", "分类"]
            ),

            # 工具能力
            "file_operations": CapabilityProfile(
                name="文件操作",
                category="tool",
                strength=0.95,
                versatility=0.7,
                use_cases=["读写文件", "目录遍历", "路径操作"]
            ),
            "web_search": CapabilityProfile(
                name="网络搜索",
                category="tool",
                strength=0.9,
                versatility=0.85,
                dependencies=["text_understanding"],
                use_cases=["信息检索", "资料收集", "实时查询"]
            ),
            "command_execution": CapabilityProfile(
                name="命令执行",
                category="tool",
                strength=0.9,
                versatility=0.8,
                use_cases=["运行脚本", "系统调用", "工具链操作"]
            ),
            "code_execution": CapabilityProfile(
                name="代码执行",
                category="tool",
                strength=0.85,
                versatility=0.75,
                dependencies=["command_execution"],
                limitations=["沙箱限制", "无网络访问"],
                use_cases=["Python代码", "数据处理", "算法验证"]
            ),

            # 领域知识
            "mathematics": CapabilityProfile(
                name="数学知识",
                category="knowledge",
                strength=0.7,
                versatility=0.9,
                use_cases=["基础计算", "统计分析", "简单优化"]
            ),
            "programming": CapabilityProfile(
                name="编程知识",
                category="knowledge",
                strength=0.9,
                versatility=0.95,
                use_cases=["多语言开发", "架构设计", "调试"]
            ),
            "data_science": CapabilityProfile(
                name="数据科学",
                category="knowledge",
                strength=0.75,
                versatility=0.8,
                dependencies=["mathematics", "programming"],
                use_cases=["数据处理", "可视化", "建模"]
            ),

            # 缺失的高级能力（用于对比）
            "3d_geometry": CapabilityProfile(
                name="3D几何处理",
                category="knowledge",
                strength=0.0,  # 不具备
                versatility=0.0,
                limitations=["完全不具备点云处理能力", "无法进行3D重建"]
            ),
            "quantum_physics": CapabilityProfile(
                name="量子物理",
                category="knowledge",
                strength=0.0,  # 不具备
                versatility=0.0,
                limitations=["不具备量子力学知识", "无法进行量子计算"]
            ),
            "molecular_biology": CapabilityProfile(
                name="分子生物学",
                category="knowledge",
                strength=0.0,  # 不具备
                versatility=0.0,
                limitations=["不具备生物学知识", "无法进行蛋白质分析"]
            ),
        }

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """构建能力依赖关系图"""
        return {
            "code_analysis": ["text_understanding"],
            "web_search": ["text_understanding"],
            "code_execution": ["command_execution"],
            "data_science": ["mathematics", "programming"],
        }

    def _build_similarity_matrix(self) -> Dict[Tuple[str, str], float]:
        """构建能力相似度矩阵"""
        return {
            ("text_understanding", "pattern_recognition"): 0.7,
            ("code_analysis", "programming"): 0.9,
            ("data_science", "mathematics"): 0.8,
            ("pattern_recognition", "data_science"): 0.6,
            ("logical_reasoning", "code_analysis"): 0.5,
        }

    def match(self, task: str, context: Optional[Dict] = None) -> MatchResult:
        """
        匹配任务需求与系统能力

        Args:
            task: 任务描述
            context: 额外上下文

        Returns:
            MatchResult: 匹配结果
        """
        print(f"\n{'='*60}")
        print(f"[MetaCognitive] 能力匹配分析")
        print(f"{'='*60}")
        print(f"任务描述: {task}")

        # 1. 提取任务需求
        required_capabilities = self._extract_required_capabilities(task)

        # 2. 查找匹配能力
        matching = self._find_matching_capabilities(required_capabilities)

        # 3. 识别缺失能力
        missing = self._identify_missing_capabilities(required_capabilities)

        # 4. 评估匹配度
        match_level = self._assess_match_level(matching, missing)

        # 5. 计算置信度
        confidence = self._calculate_confidence(match_level, matching, missing)

        # 6. 生成替代方案
        alternatives = self._generate_alternatives(task, missing)

        # 7. 生成工作建议
        workarounds = self._generate_workarounds(task, missing)

        # 构建结果
        result = MatchResult(
            task=task,
            match_level=match_level,
            matching_capabilities=list(matching.keys()),
            missing_capabilities=list(missing),
            capability_gaps=self._analyze_capability_gaps(missing),
            suggested_alternatives=alternatives,
            confidence=confidence,
            workarounds=workarounds
        )

        # 输出匹配结果
        self._print_match_result(result)

        return result

    def _extract_required_capabilities(self, task: str) -> Set[str]:
        """提取任务所需能力"""
        task_lower = task.lower()
        required = set()

        # 🔧 [2026-01-16] 修复false positive: 识别系统内部任务
        # 系统内部任务模式（不需要特殊领域能力）
        system_task_patterns = [
            r"wait for.*loop",
            r"generating new directive",
            r"idle\.",
            r"waiting for",
            r"\(resting\)",
            r"\(idle\)",
            r"system maintenance",
            r"triggering evolution",
            r"spinning up",
        ]

        for pattern in system_task_patterns:
            if re.search(pattern, task_lower):
                # 系统内部任务，只需要基础能力
                return {"text_understanding", "logical_reasoning"}

        # 关键词到能力的映射
        capability_keywords = {
            "text_understanding": ["text", "read", "document", "string"],
            "code_analysis": ["code", "function", "class", "algorithm", "program"],
            "logical_reasoning": ["plan", "analyze", "evaluate", "reason", "logic"],
            "pattern_recognition": ["pattern", "recognize", "classify", "detect"],
            "file_operations": ["file", "save", "load", "write", "read"],
            "web_search": ["search", "web", "internet", "lookup"],
            "command_execution": ["command", "execute", "run", "bash"],
            "code_execution": ["python", "execute code", "run code"],
            "mathematics": ["math", "calculate", "statistics", "optimize"],
            "programming": ["develop", "implement", "design"],
            "data_science": ["data", "analyze", "visualize", "model"],
            "3d_geometry": ["3d", "point cloud", "mesh", "geometry"],
            "quantum_physics": ["quantum", "entanglement", "wave function"],
            "molecular_biology": ["protein", "dna", "gene", "molecule"],
        }

        # 检测关键词（使用更精确的单词边界匹配）
        for capability, keywords in capability_keywords.items():
            for kw in keywords:
                # 使用单词边界匹配，避免子串误匹配
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, task_lower, re.IGNORECASE):
                    required.add(capability)
                    break  # 找到一个匹配就跳出

        # 如果没有检测到特定能力，默认需要文本理解和逻辑推理
        if not required:
            required = {"text_understanding", "logical_reasoning"}

        return required

    def _find_matching_capabilities(self, required: Set[str]) -> Dict[str, CapabilityProfile]:
        """查找匹配的能力"""
        matching = {}

        for req_cap in required:
            # 直接匹配
            if req_cap in self.capabilities:
                cap_profile = self.capabilities[req_cap]
                if cap_profile.strength > 0:
                    matching[req_cap] = cap_profile

            # 相似能力匹配
            for cap_name, cap_profile in self.capabilities.items():
                if cap_profile.strength > 0:
                    similarity = self.similarity_matrix.get((req_cap, cap_name), 0)
                    if similarity > 0.6:
                        matching[f"{req_cap} (via {cap_name})"] = cap_profile

        return matching

    def _identify_missing_capabilities(self, required: Set[str]) -> Set[str]:
        """识别缺失的能力"""
        missing = set()

        for req_cap in required:
            # 检查是否有直接匹配
            if req_cap in self.capabilities and self.capabilities[req_cap].strength > 0:
                continue

            # 检查是否有相似能力
            has_similar = False
            for cap_name, cap_profile in self.capabilities.items():
                if cap_profile.strength > 0:
                    similarity = self.similarity_matrix.get((req_cap, cap_name), 0)
                    if similarity > 0.6:
                        has_similar = True
                        break

            if not has_similar:
                missing.add(req_cap)

        return missing

    def _assess_match_level(self, matching: Dict, missing: Set) -> MatchLevel:
        """评估匹配等级"""
        if len(missing) == 0 and len(matching) > 0:
            return MatchLevel.PERFECT
        elif len(missing) == 0:
            return MatchLevel.GOOD
        elif len(missing) <= 2 and len(matching) > 0:
            return MatchLevel.PARTIAL
        elif len(matching) > 0:
            return MatchLevel.POOR
        else:
            return MatchLevel.NONE

    def _calculate_confidence(self, level: MatchLevel, matching: Dict, missing: Set) -> float:
        """计算匹配置信度"""
        base_confidence = {
            MatchLevel.PERFECT: 0.95,
            MatchLevel.GOOD: 0.8,
            MatchLevel.PARTIAL: 0.6,
            MatchLevel.POOR: 0.4,
            MatchLevel.NONE: 0.1,
        }

        confidence = base_confidence[level]

        # 根据匹配能力的强度调整
        if matching:
            avg_strength = sum(c.strength for c in matching.values()) / len(matching)
            confidence = confidence * 0.7 + avg_strength * 0.3

        # 根据缺失能力数量调整
        if missing:
            confidence -= 0.1 * len(missing)

        return max(0.0, min(1.0, confidence))

    def _analyze_capability_gaps(self, missing: Set[str]) -> List[str]:
        """分析能力缺口"""
        gaps = []

        for cap in missing:
            if cap in self.capabilities:
                cap_profile = self.capabilities[cap]
                gaps.append(f"{cap_profile.name}: {', '.join(cap_profile.limitations)}")
            else:
                gaps.append(f"{cap}: 系统完全不具备此能力")

        return gaps

    def _generate_alternatives(self, task: str, missing: Set[str]) -> List[str]:
        """生成替代方案"""
        alternatives = []

        if "3d_geometry" in missing:
            alternatives.append("使用外部库如Open3D或PCL处理3D数据")
            alternatives.append("将3D问题降维为2D处理")

        if "quantum_physics" in missing:
            alternatives.append("搜索量子计算相关文档和资料")
            alternatives.append("咨询量子物理领域专家")

        if "molecular_biology" in missing:
            alternatives.append("使用生物信息学数据库如PDB")
            alternatives.append("借助专业生物分析工具")

        if not alternatives:
            alternatives.append("将任务分解为更小的子任务")
            alternatives.append("寻求外部专业知识或工具")

        return alternatives

    def _generate_workarounds(self, task: str, missing: Set[str]) -> List[str]:
        """生成工作建议"""
        workarounds = []

        if len(missing) == 1:
            workarounds.append(f"主要缺失能力: {list(missing)[0]}")
            workarounds.append("建议: 先学习相关知识或寻找替代工具")
        elif len(missing) > 1:
            workarounds.append(f"缺失{len(missing)}项核心能力，任务难度较大")
            workarounds.append("建议: 分阶段实施，先完成能力范围内的部分")

        return workarounds

    def _print_match_result(self, result: MatchResult):
        """打印匹配结果"""
        print(f"\n{'─'*60}")
        print(f"[匹配结果]")
        print(f"{'─'*60}")
        print(f"匹配等级: {result.match_level.value}")
        print(f"置信度:   {result.confidence:.2f}")
        print(f"匹配能力: {len(result.matching_capabilities)}项")
        print(f"缺失能力: {len(result.missing_capabilities)}项")

        if result.matching_capabilities:
            print(f"\n✅ 已匹配能力:")
            for cap in result.matching_capabilities:
                profile = self.capabilities.get(cap.replace(" (via", " (via").split()[0], None)
                if profile:
                    print(f"  • {profile.name} (强度: {profile.strength:.2f})")

        if result.missing_capabilities:
            print(f"\n❌ 缺失能力:")
            for cap in result.missing_capabilities:
                print(f"  • {cap}")

        if result.capability_gaps:
            print(f"\n⚠️ 能力缺口:")
            for gap in result.capability_gaps:
                print(f"  • {gap}")

        if result.suggested_alternatives:
            print(f"\n💡 替代方案:")
            for i, alt in enumerate(result.suggested_alternatives, 1):
                print(f"  {i}. {alt}")

        if result.workarounds:
            print(f"\n🔧 工作建议:")
            for advice in result.workarounds:
                print(f"  • {advice}")

        print(f"\n{'='*60}")

        # 关键输出：能力边界认知
        if result.match_level in [MatchLevel.POOR, MatchLevel.NONE]:
            print(f"[MetaCognitive] ⚠️ 系统能力边界检测: 该任务超出能力范围")
            print(f"[MetaCognitive] 📊 能力匹配度: {result.confidence:.2%}")
            print(f"[MetaCognitive] 🚫 建议: 寻求外部工具或专业知识支持")
        else:
            print(f"[MetaCognitive] ✅ 系统能力充分: 可以尝试处理该任务")
            print(f"[MetaCognitive] 📊 能力匹配度: {result.confidence:.2%}")


# ============ 使用示例 ============

if __name__ == "__main__":
    print("="*60)
    print("能力匹配分析器测试")
    print("="*60)

    matcher = CapabilityMatcher()

    # 测试1: 匹配任务
    print("\n[测试1] 匹配任务")
    result1 = matcher.match("分析Python代码并生成优化建议")

    # 测试2: 部分匹配任务
    print("\n[测试2] 部分匹配任务")
    result2 = matcher.match("读取CSV文件并进行数据可视化")

    # 测试3: 不匹配任务
    print("\n[测试3] 不匹配任务")
    result3 = matcher.match("分析3D点云数据并提取表面法向量")

    # 测试4: 完全不匹配任务
    print("\n[测试4] 完全不匹配任务")
    result4 = matcher.match("解释量子纠缠的物理机制")
