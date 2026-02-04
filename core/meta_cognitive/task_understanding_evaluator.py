#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务理解深度评估器 (Task Understanding Depth Evaluator)
==========================================================

元认知层第一组件：评估系统对任务的理解深度

功能：
- 评估任务理解深度（我真的理解了吗？）
- 识别知识缺口（我不知道什么？）
- 评估置信度（我的理解有多少把握？）
- 判断任务可行性（我能解决这个问题吗？）

Version: 1.0.0
Author: AGI Evolution Team
Date: 2026-01-16
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class UnderstandingLevel(Enum):
    """理解深度等级"""
    SURFACE = "surface"      # 表层理解（仅知道字面意思）
    SHALLOW = "shallow"      # 浅层理解（知道基本概念）
    MODERATE = "moderate"    # 中等理解（理解主要关系）
    DEEP = "deep"           # 深度理解（理解底层原理）
    EXPERT = "expert"       # 专家理解（可创新延伸）


@dataclass
class TaskAnalysis:
    """任务分析结果"""
    task_description: str
    understanding_level: UnderstandingLevel
    confidence: float  # 0.0-1.0
    can_solve: bool
    knowledge_gaps: List[str] = field(default_factory=list)
    missing_capabilities: List[str] = field(default_factory=list)
    complexity_indicators: List[str] = field(default_factory=list)
    suggested_approach: Optional[str] = None
    estimated_difficulty: str = "unknown"  # trivial, easy, medium, hard, expert


class TaskUnderstandingEvaluator:
    """
    任务理解深度评估器

    核心功能：
    1. 分析任务描述，提取关键要素
    2. 评估系统对该任务的理解深度
    3. 识别知识缺口和缺失能力
    4. 判断任务可行性
    """

    def __init__(self, knowledge_graph=None, memory_system=None):
        """
        初始化评估器

        Args:
            knowledge_graph: 知识图谱引用
            memory_system: 记忆系统引用
        """
        self.knowledge_graph = knowledge_graph
        self.memory_system = memory_system

        # 定义能力边界（系统当前具备的能力）
        self.capability_registry = {
            "text_processing": True,
            "code_analysis": True,
            "file_operations": True,
            "web_search": True,
            "basic_math": True,
            "data_analysis": True,
            "3d_geometry": False,  # 不具备
            "quantum_physics": False,  # 不具备
            "advanced_calculus": False,  # 不具备
            "molecular_biology": False,  # 不具备
        }

        # 定义领域关键词
        self.domain_keywords = {
            "quantum": ["quantum", "entanglement", "superposition", "wave function", "schrodinger"],
            "3d_geometry": ["point cloud", "mesh", "3d reconstruction", "stereoscopic", "depth map"],
            "calculus": ["derivative", "integral", "differential equation", "gradient", "optimization"],
            "biology": ["protein", "dna", "gene", "molecule", "cell", "metabolism"],
            "physics": ["mechanics", "thermodynamics", "electromagnetism", "relativity"],
            "machine_learning": ["neural network", "training", "inference", "model", "algorithm"],
        }

    def evaluate(self, task: str, context: Optional[Dict] = None) -> TaskAnalysis:
        """
        评估任务理解深度

        Args:
            task: 任务描述
            context: 额外上下文信息

        Returns:
            TaskAnalysis: 详细的任务分析结果
        """
        print(f"\n{'='*60}")
        print(f"[MetaCognitive] 任务理解深度评估")
        print(f"{'='*60}")
        print(f"任务描述: {task}")

        # 1. 提取任务特征
        features = self._extract_task_features(task)

        # 2. 评估理解深度
        understanding_level = self._assess_understanding_level(task, features)

        # 3. 评估置信度
        confidence = self._assess_confidence(task, features, understanding_level)

        # 4. 识别知识缺口
        knowledge_gaps = self._identify_knowledge_gaps(task, features)

        # 5. 识别缺失能力
        missing_capabilities = self._identify_missing_capabilities(task, features)

        # 6. 判断可行性
        can_solve = self._assess_feasibility(knowledge_gaps, missing_capabilities)

        # 7. 评估复杂度
        complexity = self._assess_complexity(features)

        # 8. 生成建议方法
        suggested_approach = self._suggest_approach(task, features) if can_solve else None

        # 构建分析结果
        analysis = TaskAnalysis(
            task_description=task,
            understanding_level=understanding_level,
            confidence=confidence,
            can_solve=can_solve,
            knowledge_gaps=knowledge_gaps,
            missing_capabilities=missing_capabilities,
            complexity_indicators=features.get("complexity_indicators", []),
            suggested_approach=suggested_approach,
            estimated_difficulty=complexity
        )

        # 输出评估结果
        self._print_evaluation(analysis)

        return analysis

    def _extract_task_features(self, task: str) -> Dict[str, Any]:
        """提取任务特征"""
        features = {
            "domains": [],
            "complexity_indicators": [],
            "keywords": [],
            "has_numbers": False,
            "has_code": False,
            "has_file_ops": False,
            "estimated_steps": 1,
            "task_type": "normal",  # normal, idle, waiting, maintenance, system
        }

        task_lower = task.lower()

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
                features["task_type"] = "idle"
                # idle任务不进行领域检测，直接返回
                return features

        # 检测领域（增强上下文理解）
        for domain, keywords in self.domain_keywords.items():
            # 使用更精确的匹配：单词边界检测
            for kw in keywords:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, task_lower, re.IGNORECASE):
                    features["domains"].append(domain)
                    break  # 找到一个匹配就跳出

        # 检测复杂度指标
        complexity_patterns = {
            "multi_step": ["and then", "after that", "followed by", "subsequently"],
            "conditional": ["if", "when", "depending on", "based on"],
            "iterative": ["repeat", "iterate", "loop", "for each"],
            "optimization": ["optimize", "minimize", "maximize", "best"],
            "analysis": ["analyze", "evaluate", "assess", "compare"],
            "creation": ["create", "design", "invent", "develop"],
        }

        for complexity, patterns in complexity_patterns.items():
            if any(pattern in task_lower for pattern in patterns):
                features["complexity_indicators"].append(complexity)

        # 检测关键词
        words = re.findall(r'\b\w+\b', task)
        features["keywords"] = list(set(words))
        features["has_numbers"] = bool(re.search(r'\d+', task))
        features["has_code"] = bool(re.search(r'code|function|class|algorithm', task_lower))
        features["has_file_ops"] = bool(re.search(r'file|read|write|save|load', task_lower))

        # 估算步数
        if len(features["complexity_indicators"]) > 0:
            features["estimated_steps"] = min(1 + len(features["complexity_indicators"]), 5)

        return features

    def _assess_understanding_level(self, task: str, features: Dict) -> UnderstandingLevel:
        """评估理解深度"""
        # 🔧 [2026-01-16] 修复false positive: 特殊处理idle/waiting任务
        if features.get("task_type") == "idle":
            # 系统内部idle任务，完全理解
            return UnderstandingLevel.EXPERT

        # 检查是否涉及未知领域
        unknown_domains = set(features["domains"]) - {"text_processing", "code_analysis", "file_operations", "web_search", "basic_math", "data_analysis"}

        if len(unknown_domains) > 0:
            # 涉及未知领域
            if len(features["domains"]) == 0:
                return UnderstandingLevel.SURFACE
            else:
                return UnderstandingLevel.SHALLOW

        # 检查任务复杂度
        complexity = len(features["complexity_indicators"])

        if complexity == 0:
            return UnderstandingLevel.SHALLOW
        elif complexity <= 2:
            return UnderstandingLevel.MODERATE
        elif complexity <= 4:
            return UnderstandingLevel.DEEP
        else:
            return UnderstandingLevel.EXPERT

    def _assess_confidence(self, task: str, features: Dict, level: UnderstandingLevel) -> float:
        """评估置信度"""
        base_confidence = {
            UnderstandingLevel.SURFACE: 0.3,
            UnderstandingLevel.SHALLOW: 0.5,
            UnderstandingLevel.MODERATE: 0.7,
            UnderstandingLevel.DEEP: 0.85,
            UnderstandingLevel.EXPERT: 0.95,
        }

        confidence = base_confidence[level]

        # 🔧 [2026-01-16] 修复false positive: idle任务给予高置信度
        if features.get("task_type") == "idle":
            return 0.98  # 系统完全理解内部任务

        # 如果涉及未知领域，降低置信度
        unknown_domains = set(features["domains"]) - {"text_processing", "code_analysis", "file_operations", "web_search", "basic_math", "data_analysis"}
        if len(unknown_domains) > 0:
            confidence -= 0.3 * len(unknown_domains)

        # 如果任务模糊，降低置信度
        if len(task) < 20:
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))

    def _identify_knowledge_gaps(self, task: str, features: Dict) -> List[str]:
        """识别知识缺口"""
        gaps = []

        # 🔧 [2026-01-16] 修复false positive: idle任务无知识缺口
        if features.get("task_type") == "idle":
            return gaps  # 空列表，无知识缺口

        # 检查领域缺口
        unknown_domains = set(features["domains"]) - {"text_processing", "code_analysis", "file_operations", "web_search", "basic_math", "data_analysis"}

        domain_names = {
            "quantum": "量子物理",
            "3d_geometry": "3D几何与点云处理",
            "calculus": "微积分",
            "biology": "分子生物学",
            "physics": "高级物理",
        }

        for domain in unknown_domains:
            if domain in domain_names:
                gaps.append(f"缺少{domain_names[domain]}领域知识")

        # 检查概念缺口
        if features["has_numbers"] and "calculus" not in features["domains"]:
            gaps.append("缺少数学建模知识")

        return gaps

    def _identify_missing_capabilities(self, task: str, features: Dict) -> List[str]:
        """识别缺失能力"""
        missing = []

        # 🔧 [2026-01-16] 修复false positive: idle任务无缺失能力
        if features.get("task_type") == "idle":
            return missing  # 空列表，无缺失能力

        # 检查是否需要特定领域能力
        for domain in features["domains"]:
            if not self.capability_registry.get(domain, False):
                missing.append(f"缺少{domain}处理能力")

        # 检查是否需要特定工具
        if "point cloud" in task.lower():
            missing.append("缺少3D点云处理工具")

        if "protein" in task.lower() or "dna" in task.lower():
            missing.append("缺少生物信息学分析工具")

        return missing

    def _assess_feasibility(self, knowledge_gaps: List[str], missing_capabilities: List[str]) -> bool:
        """评估任务可行性"""
        # 如果有缺失能力，无法完成
        if len(missing_capabilities) > 0:
            return False

        # 如果知识缺口太多，置信度低
        if len(knowledge_gaps) > 2:
            return False

        return True

    def _assess_complexity(self, features: Dict) -> str:
        """评估任务复杂度"""
        complexity_score = len(features["complexity_indicators"])

        if complexity_score == 0:
            return "easy"
        elif complexity_score <= 2:
            return "medium"
        elif complexity_score <= 4:
            return "hard"
        else:
            return "expert"

    def _suggest_approach(self, task: str, features: Dict) -> str:
        """建议处理方法"""
        steps = []

        # 基于复杂度指标生成建议
        if "multi_step" in features["complexity_indicators"]:
            steps.append("1. 分解任务为多个子步骤")

        if "analysis" in features["complexity_indicators"]:
            steps.append("2. 使用PlannerAgent生成详细分析计划")

        if features["has_file_ops"]:
            steps.append("3. 使用ExecutorAgent执行文件操作")

        if "conditional" in features["complexity_indicators"]:
            steps.append("4. 根据条件动态调整策略")

        return "\n".join(steps) if steps else "1. 直接执行任务"

    def _print_evaluation(self, analysis: TaskAnalysis):
        """打印评估结果"""
        print(f"\n{'─'*60}")
        print(f"[评估结果]")
        print(f"{'─'*60}")
        print(f"理解深度: {analysis.understanding_level.value}")
        print(f"置信度:   {analysis.confidence:.2f}")
        print(f"可行性:   {'✅ 可行' if analysis.can_solve else '❌ 不可行'}")
        print(f"复杂度:   {analysis.estimated_difficulty}")

        if analysis.knowledge_gaps:
            print(f"\n知识缺口:")
            for gap in analysis.knowledge_gaps:
                print(f"  • {gap}")

        if analysis.missing_capabilities:
            print(f"\n缺失能力:")
            for cap in analysis.missing_capabilities:
                print(f"  • {cap}")

        if analysis.complexity_indicators:
            print(f"\n复杂度指标:")
            for indicator in analysis.complexity_indicators:
                print(f"  • {indicator}")

        if analysis.suggested_approach:
            print(f"\n建议方法:")
            for line in analysis.suggested_approach.split('\n'):
                print(f"  {line}")

        print(f"\n{'='*60}")

        # 关键输出：系统自我认知
        if not analysis.can_solve:
            print(f"[MetaCognitive] ⚠️ 系统自我评估: 该任务超出当前能力边界")
            print(f"[MetaCognitive] 💡 建议: {self._generate_fallback_suggestion(analysis)}")
        else:
            print(f"[MetaCognitive] ✅ 系统自我评估: 该任务在能力范围内")

    def _generate_fallback_suggestion(self, analysis: TaskAnalysis) -> str:
        """生成回退建议"""
        if analysis.missing_capabilities:
            return f"需要获取{'或'.join(analysis.missing_capabilities)}后再尝试"

        if analysis.knowledge_gaps:
            return f"需要学习{analysis.knowledge_gaps[0]}相关知识"

        return "建议将任务分解为更小的子任务"


# ============ 使用示例 ============

if __name__ == "__main__":
    print("="*60)
    print("任务理解深度评估器测试")
    print("="*60)

    evaluator = TaskUnderstandingEvaluator()

    # 测试1: 简单任务（在能力范围内）
    print("\n[测试1] 简单任务")
    result1 = evaluator.evaluate("读取文件hello.txt并统计行数")

    # 测试2: 复杂任务（在能力范围内）
    print("\n[测试2] 复杂任务")
    result2 = evaluator.evaluate("分析项目中所有Python文件的代码质量，生成优化建议报告")

    # 测试3: 超出能力范围的任务
    print("\n[测试3] 超出能力范围")
    result3 = evaluator.evaluate("分析3D点云数据的几何特征，提取表面法向量")

    # 测试4: 量子物理任务（超出知识范围）
    print("\n[测试4] 量子物理任务")
    result4 = evaluator.evaluate("解释量子纠缠的物理机制及其在量子计算中的应用")
