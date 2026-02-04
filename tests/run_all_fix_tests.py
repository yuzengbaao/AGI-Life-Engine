#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI 修复验证测试套件主运行器
AGI Fix Verification Test Suite Runner

运行所有修复验证测试并生成综合报告。

修复内容：
1. 修复3658行：真实语义向量替代随机数
2. 修复3545行：智能好奇心计算（对数增长+动机驱动）
3. 修复motivation.py:149：快速探索触发（80→30）

使用方法:
    python tests/run_all_fix_tests.py
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRunner:
    """测试运行器"""

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tests_dir = os.path.join(self.base_dir, 'tests')

        self.test_files = [
            'test_fix_1_real_semantic_vector.py',
            'test_fix_2_adaptive_curiosity.py',
            'test_fix_3_fast_exploration.py',
        ]

        self.test_names = [
            '修复1: 真实语义向量',
            '修复2: 智能好奇心计算',
            '修复3: 快速探索触发',
        ]

        self.results = []

    def print_header(self):
        """打印标题"""
        print("\n" + "="*70)
        print(" "*10 + "🔬 AGI 系统修复验证测试套件")
        print("="*70)
        print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"项目路径: {self.base_dir}")
        print(f"\n待运行测试: {len(self.test_files)}")

        print("\n测试列表:")
        for i, name in enumerate(self.test_names, 1):
            print(f"  {i}. {name}")

        print("\n" + "-"*70)

    def run_test(self, test_file, test_name):
        """运行单个测试"""
        test_path = os.path.join(self.tests_dir, test_file)

        if not os.path.exists(test_path):
            return {
                'name': test_name,
                'file': test_file,
                'success': False,
                'error': f'文件不存在: {test_path}',
                'duration': 0
            }

        print(f"\n🔄 运行: {test_name}")
        print(f"   文件: {test_file}")

        start_time = time.time()

        try:
            # 运行测试
            result = subprocess.run(
                [sys.executable, test_path],
                cwd=self.tests_dir,
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )

            duration = time.time() - start_time

            success = result.returncode == 0

            return {
                'name': test_name,
                'file': test_file,
                'success': success,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': duration
            }

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                'name': test_name,
                'file': test_file,
                'success': False,
                'error': '超时 (120秒)',
                'duration': duration
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'name': test_name,
                'file': test_file,
                'success': False,
                'error': str(e),
                'duration': duration
            }

    def run_all_tests(self):
        """运行所有测试"""
        self.print_header()

        total_duration = 0

        for test_file, test_name in zip(self.test_files, self.test_names):
            result = self.run_test(test_file, test_name)
            self.results.append(result)
            total_duration += result['duration']

            # 打印即时结果
            if result['success']:
                print(f"   ✅ 通过 ({result['duration']:.2f}秒)")
            else:
                print(f"   ❌ 失败 ({result['duration']:.2f}秒)")
                if 'error' in result:
                    print(f"   错误: {result['error']}")

        self.print_summary(total_duration)

    def print_summary(self, total_duration):
        """打印总结"""
        print("\n" + "="*70)
        print(" "*20 + "📊 测试套件总结报告")
        print("="*70)

        # 统计
        total = len(self.results)
        passed = sum(1 for r in self.results if r['success'])
        failed = total - passed

        print(f"\n总测试数: {total}")
        print(f"✅ 通过: {passed}")
        print(f"❌ 失败: {failed}")
        print(f"成功率: {passed/total*100:.1f}%")
        print(f"总耗时: {total_duration:.2f}秒")

        # 详细结果
        print("\n" + "-"*70)
        print("详细结果:")
        print("-"*70)

        for i, result in enumerate(self.results, 1):
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']

            print(f"\n{i}. {result['name']}")
            print(f"   状态: {status}")
            print(f"   耗时: {duration:.2f}秒")

            if not result['success']:
                if 'error' in result:
                    print(f"   错误: {result['error']}")
                if result.get('stderr'):
                    # 只显示最后几行错误
                    stderr_lines = result['stderr'].strip().split('\n')
                    if len(stderr_lines) > 5:
                        print(f"   错误输出 (最后5行):")
                        for line in stderr_lines[-5:]:
                            print(f"     {line}")
                    else:
                        print(f"   错误输出:")
                        for line in stderr_lines:
                            print(f"     {line}")

        # 最终结论
        print("\n" + "="*70)
        if failed == 0:
            print("🎉 所有修复验证通过！")
            print("\n修复摘要:")
            print("  ✅ 修复1: 神经符号验证现在使用真实语义向量")
            print("  ✅ 修复2: 好奇心计算从二值跳变改为对数增长")
            print("  ✅ 修复3: 探索响应速度提升2.7倍")
            print("\n系统状态: 从 '深度睡眠' 苏醒到 '清醒活跃'")
            print("\n下一步: 重启 AGI_Life_Engine.py 以应用修复")
        else:
            print(f"⚠️ {failed} 个测试失败，请检查修复是否正确应用")
            print("\n故障排查:")
            print("  1. 确认所有修复已正确应用")
            print("  2. 检查依赖库是否正确安装 (sentence-transformers)")
            print("  3. 查看上面的详细错误信息")

        print("\n" + "="*70)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 保存报告到文件
        self.save_report()

    def save_report(self):
        """保存报告到文件"""
        report_dir = os.path.join(self.base_dir, 'tests', 'reports')
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(report_dir, f'fix_verification_report_{timestamp}.txt')

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("AGI 系统修复验证测试报告\n")
                f.write("="*70 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # 详细结果
                for i, result in enumerate(self.results, 1):
                    f.write(f"\n{i}. {result['name']}\n")
                    f.write(f"   状态: {'PASS' if result['success'] else 'FAIL'}\n")
                    f.write(f"   耗时: {result['duration']:.2f}秒\n")

                    if not result['success'] and 'error' in result:
                        f.write(f"   错误: {result['error']}\n")

                print(f"\n📄 报告已保存: {report_file}")
        except Exception as e:
            print(f"\n⚠️ 无法保存报告: {e}")


def main():
    """主函数"""
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    runner = TestRunner()
    runner.run_all_tests()

    # 返回码
    failed = sum(1 for r in runner.results if not r['success'])
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
