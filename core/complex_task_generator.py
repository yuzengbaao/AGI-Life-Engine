"""
复杂任务生成器 - P1修复
解决推理深度始终停留在shallow档的问题
生成具有挑战性的复杂任务，触发deep推理
"""

import random
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum


class TaskComplexity(Enum):
    """任务复杂度等级"""
    SHALLOW = 0.3    # 简单任务（监控、查询）
    MEDIUM = 0.6     # 中等任务（分析、设计）
    DEEP = 0.9       # 复杂任务（创造、实现）


class TaskType(Enum):
    """任务类型"""
    CREATIVE_TOOL = "creative_tool"      # 创造性工具
    DEEP_ANALYSIS = "deep_analysis"      # 深度分析
    CROSS_DOMAIN = "cross_domain"        # 跨域迁移
    SYSTEM_DESIGN = "system_design"      # 系统设计
    RESEARCH_INVESTIGATION = "research"  # 调查研究


@dataclass
class ComplexTask:
    """复杂任务定义"""
    id: str
    name: str
    description: str
    task_type: TaskType
    complexity: float
    success_criteria: Dict[str, Any]
    expected_outputs: List[str]
    reasoning_depth_required: str  # shallow/medium/deep
    estimated_duration_minutes: int
    domain: str
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type.value,
            "complexity": self.complexity,
            "success_criteria": self.success_criteria,
            "expected_outputs": self.expected_outputs,
            "reasoning_depth_required": self.reasoning_depth_required,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "domain": self.domain
        }


class ComplexTaskGenerator:
    """
    复杂任务生成器
    
    核心功能:
    1. 分析当前能力缺口
    2. 生成复杂度>0.5的任务
    3. 动态选择任务类型
    4. 明确的验收标准
    """
    
    # 任务模板库
    TASK_TEMPLATES = {
        TaskType.CREATIVE_TOOL: [
            {
                "name": "自动化工具开发",
                "description": "设计并实现一个{domain}领域的自动化工具，解决{problem}。工具应包含命令行接口、配置文件支持和日志记录功能。",
                "complexity": 0.75,
                "success_criteria": {
                    "has_implementation": True,
                    "has_cli_interface": True,
                    "has_config_support": True,
                    "execution_success": True,
                    "test_pass_rate": 0.8
                },
                "expected_outputs": ["tool.py", "config.yaml", "README.md"],
                "duration": 120
            },
            {
                "name": "数据处理管道",
                "description": "创建一个数据处理管道，能够从{data_source}读取数据，进行{transform}转换，并输出到{output_format}。",
                "complexity": 0.7,
                "success_criteria": {
                    "can_read_input": True,
                    "can_transform": True,
                    "can_write_output": True,
                    "handles_errors": True
                },
                "expected_outputs": ["pipeline.py", "test_data.json"],
                "duration": 90
            }
        ],
        TaskType.DEEP_ANALYSIS: [
            {
                "name": "模式识别分析",
                "description": "分析{data_source}中的模式，提出{count}个假设并验证。使用统计方法或机器学习方法，提供可视化结果。",
                "complexity": 0.65,
                "success_criteria": {
                    "hypothesis_count": 3,
                    "evidence_provided": True,
                    "conclusion_drawn": True,
                    "visualization_created": True
                },
                "expected_outputs": ["analysis_report.md", "visualization.png"],
                "duration": 100
            },
            {
                "name": "性能瓶颈诊断",
                "description": "诊断{system}的性能瓶颈，识别前3个瓶颈点，提出优化方案并验证效果。",
                "complexity": 0.7,
                "success_criteria": {
                    "bottlenecks_identified": 3,
                    "optimization_proposed": True,
                    "improvement_measured": True
                },
                "expected_outputs": ["diagnosis_report.md", "optimized_code.py"],
                "duration": 110
            }
        ],
        TaskType.CROSS_DOMAIN: [
            {
                "name": "概念迁移应用",
                "description": "将{domain_a}的{concept}迁移到{domain_b}，设计实现方案并验证可行性。需要理解两个领域的核心差异。",
                "complexity": 0.85,
                "success_criteria": {
                    "analogy_clear": True,
                    "implementation_feasible": True,
                    "value_demonstrated": True,
                    "limitations_documented": True
                },
                "expected_outputs": ["migration_plan.md", "prototype.py"],
                "duration": 150
            },
            {
                "name": "跨领域创新",
                "description": "结合{domain_a}和{domain_b}的技术，设计一个创新解决方案，解决{problem}。",
                "complexity": 0.8,
                "success_criteria": {
                    "innovation_novel": True,
                    "technical_feasible": True,
                    "prototype_working": True
                },
                "expected_outputs": ["innovation_proposal.md", "prototype/"],
                "duration": 140
            }
        ],
        TaskType.SYSTEM_DESIGN: [
            {
                "name": "子系统架构设计",
                "description": "为{system}设计一个子系统的架构，包括组件划分、接口定义、数据流设计和错误处理策略。",
                "complexity": 0.75,
                "success_criteria": {
                    "architecture_documented": True,
                    "interfaces_defined": True,
                    "data_flow_clear": True,
                    "scalability_considered": True
                },
                "expected_outputs": ["architecture.md", "interface_spec.yaml"],
                "duration": 130
            }
        ],
        TaskType.RESEARCH_INVESTIGATION: [
            {
                "name": "技术调研报告",
                "description": "调研{technology}的最新进展，对比至少3个相关方案，给出推荐意见和实施建议。",
                "complexity": 0.6,
                "success_criteria": {
                    "sources_reviewed": 5,
                    "comparison_table": True,
                    "recommendation_given": True,
                    "implementation_roadmap": True
                },
                "expected_outputs": ["research_report.md"],
                "duration": 90
            }
        ]
    }
    
    # 变量填充库
    DOMAIN_POOL = [
        "数据分析", "机器学习", "自然语言处理", "计算机视觉",
        "自动化测试", "系统监控", "日志处理", "配置管理",
        "API开发", "数据库优化", "缓存策略", "并发处理"
    ]
    
    PROBLEM_POOL = [
        "重复性任务自动化", "数据处理效率低", "系统监控盲区",
        "配置管理混乱", "日志分析困难", "测试覆盖不足",
        "性能瓶颈定位", "错误处理不完善"
    ]
    
    DATA_SOURCE_POOL = [
        "系统日志", "用户行为数据", "性能指标", "错误日志",
        "配置文件", "API调用记录", "数据库查询日志"
    ]
    
    TECHNOLOGY_POOL = [
        "向量数据库", "大语言模型微调", "图神经网络",
        "强化学习", "自动化机器学习", "联邦学习",
        "边缘计算", "Serverless架构"
    ]
    
    def __init__(self):
        self._task_counter = 0
        self._capability_history: List[Dict] = []
        self._stats = {
            "tasks_generated": 0,
            "by_type": {t.value: 0 for t in TaskType},
            "avg_complexity": 0.0
        }
    
    def analyze_capability_gap(self) -> Dict[str, float]:
        """
        分析当前能力缺口
        
        Returns:
            {"creative_tool": 0.3, "deep_analysis": 0.7, ...}
        """
        # 基于历史任务完成情况计算能力缺口
        gaps = {t.value: 0.5 for t in TaskType}  # 默认中等缺口
        
        if not self._capability_history:
            return gaps
        
        # 统计各类型任务的完成率
        type_stats = {t.value: {"total": 0, "success": 0} for t in TaskType}
        
        for record in self._capability_history:
            task_type = record.get("task_type")
            if task_type in type_stats:
                type_stats[task_type]["total"] += 1
                if record.get("success", False):
                    type_stats[task_type]["success"] += 1
        
        # 计算缺口（完成率越低，缺口越大）
        for task_type, stats in type_stats.items():
            if stats["total"] > 0:
                success_rate = stats["success"] / stats["total"]
                gaps[task_type] = 1.0 - success_rate
            else:
                gaps[task_type] = 0.7  # 无历史记录时较高缺口
        
        return gaps
    
    def select_task_type(self, capability_gaps: Dict[str, float]) -> TaskType:
        """基于能力缺口选择任务类型"""
        # 按缺口大小排序
        sorted_gaps = sorted(capability_gaps.items(), key=lambda x: x[1], reverse=True)
        
        # 前3个缺口类型中随机选择
        top_types = [t for t, _ in sorted_gaps[:3]]
        selected = random.choice(top_types)
        
        return TaskType(selected)
    
    def generate_complex_task(self, 
                            context: Optional[Dict] = None,
                            force_type: Optional[TaskType] = None) -> ComplexTask:
        """
        生成复杂任务
        
        Args:
            context: 当前上下文信息
            force_type: 强制指定任务类型（可选）
        
        Returns:
            ComplexTask对象
        """
        self._task_counter += 1
        
        # 1. 分析能力缺口
        capability_gaps = self.analyze_capability_gap()
        
        # 2. 选择任务类型
        task_type = force_type if force_type else self.select_task_type(capability_gaps)
        
        # 3. 选择模板
        templates = self.TASK_TEMPLATES.get(task_type, [])
        if not templates:
            task_type = TaskType.CREATIVE_TOOL  # 回退到默认类型
            templates = self.TASK_TEMPLATES[task_type]
        
        template = random.choice(templates)
        
        # 4. 填充变量
        variables = self._extract_variables(task_type, context)
        description = template["description"].format(**variables)
        
        # 5. 构建任务
        task = ComplexTask(
            id=f"complex_task_{self._task_counter:04d}",
            name=template["name"],
            description=description,
            task_type=task_type,
            complexity=template["complexity"],
            success_criteria=template["success_criteria"],
            expected_outputs=template["expected_outputs"],
            reasoning_depth_required="deep" if template["complexity"] > 0.7 else "medium",
            estimated_duration_minutes=template["duration"],
            domain=variables.get("domain", "general")
        )
        
        # 6. 更新统计
        self._stats["tasks_generated"] += 1
        self._stats["by_type"][task_type.value] += 1
        self._update_avg_complexity(template["complexity"])
        
        return task
    
    def _extract_variables(self, task_type: TaskType, 
                          context: Optional[Dict]) -> Dict[str, str]:
        """提取变量值用于填充模板"""
        variables = {}
        
        if context:
            # 从上下文提取
            variables["domain"] = context.get("domain", random.choice(self.DOMAIN_POOL))
            variables["problem"] = context.get("problem", random.choice(self.PROBLEM_POOL))
        else:
            # 随机选择
            variables["domain"] = random.choice(self.DOMAIN_POOL)
            variables["problem"] = random.choice(self.PROBLEM_POOL)
        
        # 类型特定变量
        if task_type == TaskType.CREATIVE_TOOL:
            variables["data_source"] = random.choice(self.DATA_SOURCE_POOL)
            variables["transform"] = random.choice(["清洗", "聚合", "转换", "过滤"])
            variables["output_format"] = random.choice(["JSON", "CSV", "数据库", "报告"])
        
        elif task_type == TaskType.DEEP_ANALYSIS:
            variables["data_source"] = random.choice(self.DATA_SOURCE_POOL)
            variables["count"] = str(random.randint(2, 5))
            variables["system"] = context.get("system", "当前系统") if context else "当前系统"
        
        elif task_type == TaskType.CROSS_DOMAIN:
            domains = random.sample(self.DOMAIN_POOL, 2)
            variables["domain_a"] = domains[0]
            variables["domain_b"] = domains[1]
            variables["concept"] = random.choice(["缓存策略", "索引机制", "工作流编排", "并发模型"])
        
        elif task_type == TaskType.RESEARCH_INVESTIGATION:
            variables["technology"] = random.choice(self.TECHNOLOGY_POOL)
        
        elif task_type == TaskType.SYSTEM_DESIGN:
            variables["system"] = context.get("system", "AGI核心") if context else "AGI核心"
        
        return variables
    
    def _update_avg_complexity(self, new_complexity: float):
        """更新平均复杂度"""
        n = self._stats["tasks_generated"]
        current_avg = self._stats["avg_complexity"]
        self._stats["avg_complexity"] = (current_avg * (n - 1) + new_complexity) / n
    
    def record_task_completion(self, task_id: str, task_type: str, 
                              success: bool, actual_complexity: float):
        """记录任务完成情况"""
        self._capability_history.append({
            "task_id": task_id,
            "task_type": task_type,
            "success": success,
            "actual_complexity": actual_complexity,
            "timestamp": time.time()
        })
    
    def get_stats(self) -> Dict:
        """获取生成统计"""
        return self._stats.copy()
    
    def get_complexity_distribution(self) -> Dict[str, int]:
        """获取任务复杂度分布"""
        distribution = {"shallow": 0, "medium": 0, "deep": 0}
        
        for template_list in self.TASK_TEMPLATES.values():
            for template in template_list:
                c = template["complexity"]
                if c < 0.5:
                    distribution["shallow"] += 1
                elif c < 0.75:
                    distribution["medium"] += 1
                else:
                    distribution["deep"] += 1
        
        return distribution


# 便捷函数
def create_complex_task_generator() -> ComplexTaskGenerator:
    """创建复杂任务生成器"""
    return ComplexTaskGenerator()


# 测试代码
if __name__ == "__main__":
    import time
    
    generator = ComplexTaskGenerator()
    
    print("复杂任务生成器测试:")
    print("=" * 70)
    
    # 生成5个不同类型的任务
    for i in range(5):
        task = generator.generate_complex_task()
        
        print(f"\n任务 {i+1}: {task.name}")
        print(f"  ID: {task.id}")
        print(f"  类型: {task.task_type.value}")
        print(f"  复杂度: {task.complexity}")
        print(f"  推理深度: {task.reasoning_depth_required}")
        print(f"  预计耗时: {task.estimated_duration_minutes}分钟")
        print(f"  描述: {task.description[:60]}...")
        print(f"  成功标准: {list(task.success_criteria.keys())}")
        print(f"  预期产出: {task.expected_outputs}")
    
    print("\n" + "=" * 70)
    print("统计:")
    stats = generator.get_stats()
    print(f"  生成任务数: {stats['tasks_generated']}")
    print(f"  平均复杂度: {stats['avg_complexity']:.2f}")
    print(f"  按类型分布: {stats['by_type']}")
    
    print("\n复杂度分布:")
    dist = generator.get_complexity_distribution()
    print(f"  Shallow (<0.5): {dist['shallow']}")
    print(f"  Medium (0.5-0.75): {dist['medium']}")
    print(f"  Deep (>0.75): {dist['deep']}")
