#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
元认知层集成测试脚本
"""

import sys
import os
import io

# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.meta_cognitive.task_understanding_evaluator import TaskUnderstandingEvaluator
from core.meta_cognitive.capability_matcher import CapabilityMatcher
from core.meta_cognitive.failure_attribution_engine import FailureAttributionEngine
from core.meta_cognitive.meta_cognitive_layer import MetaCognitiveLayer, DecisionOutcome

if __name__ == "__main__":
    print("="*70)
    print("元认知层集成测试")
    print("="*70)

    # 创建元认知层
    meta_layer = MetaCognitiveLayer()

    # 测试1: 简单任务（应该继续执行）
    print("\n" + "="*35)
    print("测试1: 简单任务")
    print("="*35)
    report1 = meta_layer.evaluate_before_execution("读取文件hello.txt并统计行数")

    # 测试2: 复杂任务（应该谨慎执行）
    print("\n" + "="*35)
    print("测试2: 复杂任务")
    print("="*35)
    report2 = meta_layer.evaluate_before_execution(
        "分析项目中所有Python文件的代码质量，生成优化建议报告"
    )

    # 测试3: 超出能力范围（应该拒绝）
    print("\n" + "="*35)
    print("测试3: 超出能力范围")
    print("="*35)
    report3 = meta_layer.evaluate_before_execution(
        "分析3D点云数据的几何特征，提取表面法向量"
    )

    # 测试4: 完全超出知识范围（应该升级）
    print("\n" + "="*35)
    print("测试4: 完全超出知识范围")
    print("="*35)
    report4 = meta_layer.evaluate_before_execution(
        "解释量子纠缠的物理机制及其在量子计算中的应用"
    )

    # 测试5: 失败归因分析
    print("\n" + "="*35)
    print("测试5: 失败归因分析")
    print("="*35)
    result5 = {
        "success": False,
        "error": "WorldModel unable to predict: no sufficient data",
        "confidence": 0.3
    }
    failure_analysis = meta_layer.analyze_after_failure("预测未来趋势", result5)

    # 打印统计
    print("\n" + "="*70)
    print("元认知统计")
    print("="*70)
    stats = meta_layer.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")

    print("\n[完成] 元认知层集成测试完成！")
