#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Observation Logger - 预测性编码数据收集系统
Purpose: Systematic data collection for predictive coding research
Author: AGI System Development Team
Date: 2026-01-29
"""

import json
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import threading


@dataclass
class PredictionCycle:
    """单次预测-验证-修正循环的完整数据"""
    timestamp: float
    generation: int
    session_id: str

    # 预测阶段 (T时刻)
    prediction: str
    self_definition_before: str
    confidence_before: float

    # 现实阶段 (T+1时刻)
    actual_user_input: str

    # 误差计算
    prediction_error: float
    cognitive_dissonance: str  # 误差原因分析

    # 修正阶段 (T+2时刻)
    correction: str
    new_self_definition: str
    confidence_after: float

    # 元数据
    llm_provider: str
    processing_time_ms: float


class ObservationLogger:
    """
    预测性编码观测日志系统

    核心功能:
    1. 记录完整的预测-验证-修正循环
    2. 持久化到 JSONL 文件 (便于后续分析)
    3. 提供统计分析接口
    4. 支持实时监控
    """

    def __init__(self, log_dir: str = "data/observations"):
        """
        初始化观测日志系统

        Args:
            log_dir: 日志存储目录
        """
        self.log_dir = log_dir
        self.lock = threading.Lock()

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)

        # 日志文件路径
        self.prediction_cycles_file = os.path.join(log_dir, "prediction_cycles.jsonl")
        self.summary_file = os.path.join(log_dir, "summary.json")

        # 统计数据
        self._stats = {
            "total_cycles": 0,
            "total_prediction_error": 0.0,
            "avg_prediction_error": 0.0,
            "cycles_by_confidence": {},  # {0.8: count, ...}
            "start_time": time.time(),
            "last_update": time.time()
        }

        # 加载历史统计
        self._load_summary()

    def log_prediction_cycle(self, cycle: PredictionCycle) -> None:
        """
        记录一次完整的预测-验证-修正循环

        Args:
            cycle: PredictionCycle 对象
        """
        with self.lock:
            try:
                # 1. 写入 JSONL 文件 (append 模式)
                with open(self.prediction_cycles_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(asdict(cycle), ensure_ascii=False) + '\n')

                # 2. 更新统计
                self._update_stats(cycle)

                # 3. 定期保存摘要 (每 10 次更新一次)
                if cycle.generation % 10 == 0:
                    self._save_summary()

                # 4. 实时日志输出
                self._print_cycle_summary(cycle)

            except Exception as e:
                print(f"⚠️ [ObservationLogger] 记录失败: {e}")

    def _update_stats(self, cycle: PredictionCycle) -> None:
        """更新统计数据"""
        self._stats["total_cycles"] += 1
        self._stats["total_prediction_error"] += cycle.prediction_error
        self._stats["avg_prediction_error"] = (
            self._stats["total_prediction_error"] / self._stats["total_cycles"]
        )

        # 按置信度分组统计
        conf_key = round(cycle.confidence_before, 2)
        self._stats["cycles_by_confidence"][conf_key] = (
            self._stats["cycles_by_confidence"].get(conf_key, 0) + 1
        )

        self._stats["last_update"] = time.time()

    def _save_summary(self) -> None:
        """保存摘要统计"""
        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                json.dump(self._stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ [ObservationLogger] 保存摘要失败: {e}")

    def _load_summary(self) -> None:
        """加载历史摘要"""
        try:
            if os.path.exists(self.summary_file):
                with open(self.summary_file, 'r', encoding='utf-8') as f:
                    self._stats = json.load(f)
        except Exception as e:
            print(f"⚠️ [ObservationLogger] 加载摘要失败: {e}")

    def _print_cycle_summary(self, cycle: PredictionCycle) -> None:
        """打印循环摘要 (实时监控)"""
        print(f"\n{'='*80}")
        print(f"⚡ PREDICTION CYCLE #{cycle.generation}")
        print(f"{'='*80}")
        print(f"📊 Session: {cycle.session_id}")
        print(f"🔮 Prediction: {cycle.prediction[:100]}...")
        print(f"👤 Reality: {cycle.actual_user_input[:100]}...")
        print(f"❌ Error: {cycle.prediction_error:.3f}")
        print(f"🧠 Cognitive Dissonance: {cycle.cognitive_dissonance[:100]}...")
        print(f"✨ New Definition: {cycle.new_self_definition[:100]}...")
        print(f"⏱️ Processing Time: {cycle.processing_time_ms:.0f}ms")
        print(f"{'='*80}\n")

    def get_statistics(self) -> Dict[str, Any]:
        """获取当前统计数据"""
        with self.lock:
            return self._stats.copy()

    def get_recent_cycles(self, n: int = 10) -> List[PredictionCycle]:
        """
        获取最近的 N 次循环记录

        Args:
            n: 获取数量

        Returns:
            List of PredictionCycle
        """
        cycles = []

        try:
            with open(self.prediction_cycles_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 取最后 N 行
            for line in lines[-n:]:
                data = json.loads(line.strip())
                cycles.append(PredictionCycle(**data))

        except Exception as e:
            print(f"⚠️ [ObservationLogger] 获取历史记录失败: {e}")

        return cycles

    def export_for_analysis(self, output_file: str = None) -> str:
        """
        导出数据用于分析 (CSV 或 JSON)

        Args:
            output_file: 输出文件路径 (可选)

        Returns:
            导出的文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"export_{timestamp}.json")

        try:
            # 读取所有循环数据
            cycles = []
            with open(self.prediction_cycles_file, 'r', encoding='utf-8') as f:
                for line in f:
                    cycles.append(json.loads(line.strip()))

            # 导出
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cycles, f, ensure_ascii=False, indent=2)

            print(f"✅ [ObservationLogger] 导出 {len(cycles)} 条记录到 {output_file}")
            return output_file

        except Exception as e:
            print(f"⚠️ [ObservationLogger] 导出失败: {e}")
            return None

    def analyze_trends(self) -> Dict[str, Any]:
        """
        分析长期趋势

        Returns:
            趋势分析字典
        """
        try:
            cycles = self.get_recent_cycles(n=1000)  # 获取最近 1000 条

            if not cycles:
                return {"error": "No data available"}

            # 预测误差趋势
            errors = [c.prediction_error for c in cycles]
            avg_error = sum(errors) / len(errors)

            # 置信度趋势
            confidences_before = [c.confidence_before for c in cycles]
            confidences_after = [c.confidence_after for c in cycles]
            avg_conf_before = sum(confidences_before) / len(confidences_before)
            avg_conf_after = sum(confidences_after) / len(confidences_after)

            # 自我定义长度变化 (复杂度指标)
            definition_lengths = [len(c.new_self_definition) for c in cycles]
            avg_def_length = sum(definition_lengths) / len(definition_lengths)

            # 处理时间趋势
            processing_times = [c.processing_time_ms for c in cycles]
            avg_time = sum(processing_times) / len(processing_times)

            return {
                "total_cycles_analyzed": len(cycles),
                "prediction_error": {
                    "avg": avg_error,
                    "min": min(errors),
                    "max": max(errors),
                    "trend": "decreasing" if len(errors) > 10 and errors[-1] < errors[0] else "stable"
                },
                "confidence": {
                    "avg_before": avg_conf_before,
                    "avg_after": avg_conf_after,
                    "change": avg_conf_after - avg_conf_before
                },
                "complexity": {
                    "avg_definition_length": avg_def_length,
                    "trend": "increasing" if len(definition_lengths) > 10 and definition_lengths[-1] > definition_lengths[0] else "stable"
                },
                "performance": {
                    "avg_processing_time_ms": avg_time
                }
            }

        except Exception as e:
            return {"error": str(e)}


# 全局单例
_instance: Optional[ObservationLogger] = None


def get_observation_logger() -> ObservationLogger:
    """获取全局观测日志实例"""
    global _instance
    if _instance is None:
        _instance = ObservationLogger()
    return _instance


if __name__ == "__main__":
    # 测试代码
    logger = get_observation_logger()

    # 模拟一次预测循环
    test_cycle = PredictionCycle(
        timestamp=time.time(),
        generation=1,
        session_id="test_session",
        prediction="用户会问关于量子物理的问题",
        self_definition_before="我是量子物理专家",
        confidence_before=0.9,
        actual_user_input="你喜欢吃红烧肉吗？",
        prediction_error=1.0,
        cognitive_dissonance="预测与现实完全不符",
        correction="调整自我定义以包含更广泛的对话范围",
        new_self_definition="我是全能助手，可以聊任何话题",
        confidence_after=0.7,
        llm_provider="dashscope",
        processing_time_ms=1250.0
    )

    logger.log_prediction_cycle(test_cycle)

    # 打印统计
    stats = logger.get_statistics()
    print(f"\n📊 Statistics:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 分析趋势
    trends = logger.analyze_trends()
    print(f"\n📈 Trends:")
    print(json.dumps(trends, indent=2, ensure_ascii=False))
