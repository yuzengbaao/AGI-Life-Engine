#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试运行脚本

运行单元测试并生成覆盖率报告

使用方法:
    python scripts/run_unit_tests.py              # 运行所有测试
    python scripts/run_unit_tests.py --cov         # 生成覆盖率报告
    python scripts/run_unit_tests.py --fast        # 快速运行（跳过慢测试）

作者: AGI System
日期: 2026-02-04
"""

import sys
import subprocess
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests(
    coverage=False,
    fast=False,
    verbose=False,
    pattern=None,
    marker=None,
):
    """
    运行测试套件

    Args:
        coverage: 是否生成覆盖率报告
        fast: 快速模式（跳过慢测试）
        verbose: 详细输出
        pattern: 只运行匹配模式的测试
        marker: 只运行特定标记的测试
    """
    # 构建pytest命令
    cmd = ["python", "-m", "pytest"]

    # 添加详细输出
    if verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")

    # 快速模式：跳过慢测试
    if fast:
        cmd.extend(["-m", "not slow"])

    # 添加模式过滤
    if pattern:
        cmd.extend(["-k", pattern])

    # 添加标记过滤
    if marker:
        cmd.extend(["-m", marker])

    # 添加覆盖率
    if coverage:
        cmd.extend([
            "--cov=.",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=0",  # 不设置最低覆盖率要求
            "--cov-branch",
        ])

    # 运行测试
    print("🧪 开始运行测试...")
    print(f"命令: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    # 输出结果摘要
    print()
    print("=" * 60)
    if result.returncode == 0:
        print("✅ 测试通过！")
    else:
        print("❌ 测试失败")
    print("=" * 60)

    if coverage:
        print("\n📊 覆盖率报告已生成:")
        print("   - HTML: htmlcov/index.html")
        print("   - XML: coverage.xml")

    return result.returncode


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="运行AGI系统单元测试"
    )
    parser.add_argument(
        "--cov",
        "--coverage",
        action="store_true",
        help="生成覆盖率报告"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="快速模式（跳过慢测试）"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "-k",
        "--pattern",
        type=str,
        help="只运行匹配模式的测试"
    )
    parser.add_argument(
        "-m",
        "--marker",
        type=str,
        help="只运行特定标记的测试"
    )

    args = parser.parse_args()

    # 运行测试
    return run_tests(
        coverage=args.cov,
        fast=args.fast,
        verbose=args.verbose,
        pattern=args.pattern,
        marker=args.marker,
    )


if __name__ == "__main__":
    sys.exit(main())
