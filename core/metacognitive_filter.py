"""
元认知智能过滤模块 - P0修复
解决元认知调查空转、假阳性率高的问题
"""

import time
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class MetaEvaluationRecord:
    """元认知评估记录"""
    task_hash: str
    timestamp: float
    complexity: float
    decision: str
    had_actionable_insight: bool


class MetaCognitiveFilter:
    """
    元认知智能过滤器
    
    过滤策略:
    1. 复杂度阈值: 只评估复杂度>=0.3的任务
    2. 冷却期: 同一任务类型5分钟内不重复评估
    3. 重复检测: 相似度>80%的任务视为重复
    4. 白名单: 监控类任务跳过评估
    """
    
    # 配置参数
    MIN_COMPLEXITY = 0.3           # 最低复杂度阈值
    COOLDOWN_SECONDS = 300         # 5分钟冷却期
    SIMILARITY_THRESHOLD = 0.8     # 相似度阈值
    MAX_HISTORY = 100              # 最大历史记录数
    
    # 监控类任务白名单（这些任务跳过元认知评估）
    MONITORING_PATTERNS = [
        "监控", "monitor", "观察", "observe", "检查", "check",
        "统计", "statistic", "分析状态", "analyze status",
        "日志", "log", "报告", "report", "评估", "evaluate"
    ]
    
    def __init__(self):
        self._evaluation_history: deque = deque(maxlen=self.MAX_HISTORY)
        self._last_eval_time: Dict[str, float] = {}  # 按任务类型记录
        self._stats = {
            "total_requests": 0,
            "filtered_by_complexity": 0,
            "filtered_by_cooldown": 0,
            "filtered_by_duplicate": 0,
            "filtered_by_whitelist": 0,
            "actual_evaluations": 0,
            "actionable_insights": 0
        }
    
    def should_evaluate(self, task: str, context: Dict) -> Tuple[bool, str]:
        """
        判断是否应该进行元认知评估
        
        Returns:
            (should_evaluate, reason)
        """
        self._stats["total_requests"] += 1
        
        # 1. 白名单检查 - 监控类任务跳过
        if self._is_monitoring_task(task):
            self._stats["filtered_by_whitelist"] += 1
            return False, "monitoring_task_whitelist"
        
        # 2. 复杂度检查
        complexity = context.get("complexity", self._estimate_complexity(task))
        if complexity < self.MIN_COMPLEXITY:
            self._stats["filtered_by_complexity"] += 1
            return False, f"complexity_too_low({complexity:.2f}<{self.MIN_COMPLEXITY})"
        
        # 3. 冷却期检查
        task_type = context.get("goal_type", "default")
        current_time = time.time()
        last_time = self._last_eval_time.get(task_type, 0)
        if current_time - last_time < self.COOLDOWN_SECONDS:
            self._stats["filtered_by_cooldown"] += 1
            return False, f"cooldown_active({int(current_time-last_time)}s<{self.COOLDOWN_SECONDS}s)"
        
        # 4. 重复检测
        task_hash = self._compute_task_hash(task)
        if self._is_similar_task(task_hash, task):
            self._stats["filtered_by_duplicate"] += 1
            return False, "similar_task_recently_evaluated"
        
        # 通过所有过滤，应该评估
        self._stats["actual_evaluations"] += 1
        self._last_eval_time[task_type] = current_time
        return True, "passed_all_filters"
    
    def record_result(self, task: str, context: Dict, 
                      decision: str, had_insight: bool):
        """记录评估结果到历史"""
        record = MetaEvaluationRecord(
            task_hash=self._compute_task_hash(task),
            timestamp=time.time(),
            complexity=context.get("complexity", 0.5),
            decision=decision,
            had_actionable_insight=had_insight
        )
        self._evaluation_history.append(record)
        
        if had_insight:
            self._stats["actionable_insights"] += 1
    
    def _is_monitoring_task(self, task: str) -> bool:
        """检查是否为监控类任务"""
        task_lower = task.lower()
        return any(pattern.lower() in task_lower for pattern in self.MONITORING_PATTERNS)
    
    def _estimate_complexity(self, task: str) -> float:
        """
        简单启发式估计任务复杂度
        
        复杂度因素:
        - 任务长度（长任务通常更复杂）
        - 关键词（分析/设计/实现等）
        - 步骤数量指示词
        """
        complexity = 0.5  # 基础复杂度
        
        # 根据长度调整
        length = len(task)
        if length < 20:
            complexity -= 0.2
        elif length > 100:
            complexity += 0.1
        elif length > 200:
            complexity += 0.2
        
        # 关键词检测
        high_complexity_keywords = ["设计", "实现", "创建", "分析", "优化", 
                                    "design", "implement", "create", "analyze", "optimize"]
        low_complexity_keywords = ["检查", "查看", "获取", "更新", "记录",
                                   "check", "view", "get", "update", "log"]
        
        task_lower = task.lower()
        for keyword in high_complexity_keywords:
            if keyword in task_lower:
                complexity += 0.1
        for keyword in low_complexity_keywords:
            if keyword in task_lower:
                complexity -= 0.1
        
        # 限制在0-1范围
        return max(0.0, min(1.0, complexity))
    
    def _compute_task_hash(self, task: str) -> str:
        """计算任务哈希（用于重复检测）"""
        # 简化任务文本后哈希
        simplified = self._simplify_task(task)
        return hashlib.md5(simplified.encode()).hexdigest()[:16]
    
    def _simplify_task(self, task: str) -> str:
        """简化任务文本用于比较"""
        # 转换为小写，移除标点，标准化空格
        simplified = task.lower()
        for char in "，。！？；：""''（）【】":
            simplified = simplified.replace(char, " ")
        simplified = " ".join(simplified.split())  # 标准化空格
        return simplified
    
    def _is_similar_task(self, task_hash: str, task: str) -> bool:
        """检查是否有相似任务最近被评估过"""
        current_time = time.time()
        simplified_current = self._simplify_task(task)
        
        for record in self._evaluation_history:
            # 检查时间窗口（只比较最近冷却期内的）
            if current_time - record.timestamp > self.COOLDOWN_SECONDS:
                continue
            
            # 哈希完全匹配
            if record.task_hash == task_hash:
                return True
        
        return False
    
    def get_stats(self) -> Dict:
        """获取过滤统计信息"""
        stats = self._stats.copy()
        total = stats["total_requests"]
        if total > 0:
            stats["filter_rate"] = (total - stats["actual_evaluations"]) / total
            stats["false_positive_estimate"] = 1.0 - (stats["actionable_insights"] / max(stats["actual_evaluations"], 1))
        else:
            stats["filter_rate"] = 0.0
            stats["false_positive_estimate"] = 0.0
        return stats
    
    def reset_stats(self):
        """重置统计"""
        self._stats = {
            "total_requests": 0,
            "filtered_by_complexity": 0,
            "filtered_by_cooldown": 0,
            "filtered_by_duplicate": 0,
            "filtered_by_whitelist": 0,
            "actual_evaluations": 0,
            "actionable_insights": 0
        }


# 全局过滤器实例（单例）
_meta_filter = None

def get_meta_filter() -> MetaCognitiveFilter:
    """获取全局元认知过滤器"""
    global _meta_filter
    if _meta_filter is None:
        _meta_filter = MetaCognitiveFilter()
    return _meta_filter


def should_evaluate_meta(task: str, context: Dict = None) -> Tuple[bool, str]:
    """
    便捷函数：判断是否应该进行元认知评估
    
    Example:
        should_eval, reason = should_evaluate_meta("实现一个排序算法", {"complexity": 0.7})
        if should_eval:
            report = meta_cognitive_layer.evaluate(...)
    """
    if context is None:
        context = {}
    return get_meta_filter().should_evaluate(task, context)


# 测试代码
if __name__ == "__main__":
    filter = MetaCognitiveFilter()
    
    # 测试用例
    test_cases = [
        ("监控CPU使用率", {"goal_type": "monitor", "complexity": 0.1}, False),  # 白名单
        ("检查日志文件", {"goal_type": "check", "complexity": 0.2}, False),     # 白名单+复杂度
        ("实现快速排序算法", {"goal_type": "implement", "complexity": 0.8}, True),  # 高复杂度
        ("设计数据库架构", {"goal_type": "design", "complexity": 0.9}, True),       # 高复杂度
        ("获取当前时间", {"goal_type": "query", "complexity": 0.1}, False),        # 低复杂度
    ]
    
    print("元认知过滤器测试:")
    print("-" * 60)
    
    for task, context, expected in test_cases:
        result, reason = filter.should_evaluate(task, context)
        status = "✅" if result == expected else "❌"
        print(f"{status} 任务: {task[:30]}")
        print(f"   复杂度: {context['complexity']}, 类型: {context['goal_type']}")
        print(f"   结果: {'评估' if result else '跳过'}, 原因: {reason}")
        print()
    
    print("-" * 60)
    print(f"统计: {filter.get_stats()}")
