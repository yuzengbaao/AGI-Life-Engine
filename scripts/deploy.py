#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数学组件部署脚本
Deployment Script for Math Component

自动化部署流程：
1. 检查系统环境
2. 安装依赖
3. 验证安装
4. 运行测试
5. 生成报告

Version: 1.0.0
Date: 2025-11-15
"""

import os
import sys
import subprocess
import platform
import json
from pathlib import Path
from typing import List, Tuple, Dict


class DeploymentManager:
    """部署管理器"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.platform = platform.system()
        self.results = {}
        
    def print_header(self, title: str):
        """打印标题"""
        print("\n" + "="*80)
        print(f"  {title}")
        print("="*80)
    
    def print_step(self, step: str, status: str = ""):
        """打印步骤"""
        if status:
            print(f"\n[{status}] {step}")
        else:
            print(f"\n>>> {step}")
    
    def run_command(self, cmd: List[str], check: bool = True) -> Tuple[bool, str]:
        """运行命令"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                cwd=str(self.project_root)
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, e.stderr
        except Exception as e:
            return False, str(e)
    
    def check_python_version(self) -> bool:
        """检查Python版本"""
        self.print_step("检查Python版本...")
        
        major, minor = sys.version_info.major, sys.version_info.minor
        print(f"  当前Python版本: {major}.{minor}.{sys.version_info.micro}")
        
        if major < 3 or (major == 3 and minor < 8):
            print("  ❌ Python版本过低，需要3.8或更高")
            self.results['python_version'] = False
            return False
        
        print("  ✅ Python版本符合要求")
        self.results['python_version'] = True
        return True
    
    def check_dependencies(self) -> bool:
        """检查依赖包"""
        self.print_step("检查必需依赖...")
        
        required_packages = [
            'numpy',
            'sympy',
            'scipy',
            'torch',
            'pytest'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package} (未安装)")
                missing.append(package)
        
        if missing:
            print(f"\n  缺少以下依赖: {', '.join(missing)}")
            self.results['dependencies'] = False
            return False
        
        print("\n  ✅ 所有依赖已安装")
        self.results['dependencies'] = True
        return True
    
    def install_dependencies(self) -> bool:
        """安装依赖"""
        self.print_step("安装项目依赖...")
        
        # 检查requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            print("  ⚠️ requirements.txt不存在，跳过")
            return True
        
        # 使用pip安装
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        success, output = self.run_command(cmd, check=False)
        
        if success:
            print("  ✅ 依赖安装成功")
            self.results['install_dependencies'] = True
            return True
        else:
            print(f"  ❌ 依赖安装失败:\n{output}")
            self.results['install_dependencies'] = False
            return False
    
    def verify_math_component(self) -> bool:
        """验证数学组件"""
        self.print_step("验证数学组件...")
        
        try:
            # 导入核心模块
            from math_component.core import MathCore
            from math_component.engines import (
                SymbolicEngine,
                NumericalEngine,
                PhysicsMathEngine,
                GeometryEngine,
                MathLearningEngine
            )
            from math_component.integration import AGIMathBridge, AGIMathTool
            
            print("  ✅ 所有模块导入成功")
            
            # 快速功能测试
            core = MathCore()
            symbolic = SymbolicEngine(core)
            
            result = symbolic.differentiate("x**2", "x")
            if result.get('success'):
                print("  ✅ 符号计算功能正常")
            else:
                print("  ❌ 符号计算功能异常")
                self.results['math_component'] = False
                return False
            
            self.results['math_component'] = True
            return True
            
        except Exception as e:
            print(f"  ❌ 验证失败: {e}")
            self.results['math_component'] = False
            return False
    
    def run_tests(self, test_file: str = None) -> bool:
        """运行测试"""
        self.print_step("运行测试套件...")
        
        if test_file:
            test_path = self.project_root / test_file
            if not test_path.exists():
                print(f"  ❌ 测试文件不存在: {test_file}")
                return False
            
            # 运行指定测试
            cmd = [sys.executable, str(test_path)]
            success, output = self.run_command(cmd, check=False)
            
            if success:
                print(f"  ✅ 测试通过: {test_file}")
                self.results['tests'] = True
                return True
            else:
                print(f"  ❌ 测试失败:\n{output}")
                self.results['tests'] = False
                return False
        else:
            # 运行所有pytest
            cmd = [sys.executable, "-m", "pytest", "-v"]
            success, output = self.run_command(cmd, check=False)
            
            print(output)
            
            if success:
                print("  ✅ 所有测试通过")
                self.results['tests'] = True
                return True
            else:
                print("  ❌ 部分测试失败")
                self.results['tests'] = False
                return False
    
    def create_config(self) -> bool:
        """创建配置文件"""
        self.print_step("生成配置文件...")
        
        config = {
            "math_component": {
                "cache_enabled": True,
                "cache_size": 1000,
                "precision": 1e-10,
                "symbolic_timeout": 30,
                "numerical_tolerance": 1e-6
            },
            "engines": {
                "symbolic": {"enabled": True},
                "numerical": {"enabled": True},
                "physics": {"enabled": True, "device": "cuda"},
                "geometry": {"enabled": True},
                "learning": {"enabled": True}
            },
            "plugins": {
                "auto_load": True,
                "plugin_dirs": ["math_component/plugins"]
            },
            "logging": {
                "level": "INFO",
                "file": "math_component.log"
            }
        }
        
        config_file = self.project_root / "config.json"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"  ✅ 配置文件已生成: {config_file}")
            self.results['config'] = True
            return True
        except Exception as e:
            print(f"  ❌ 配置文件生成失败: {e}")
            self.results['config'] = False
            return False
    
    def generate_report(self) -> str:
        """生成部署报告"""
        self.print_header("部署报告")
        
        total = len(self.results)
        passed = sum(1 for v in self.results.values() if v)
        
        print(f"\n总计: {passed}/{total} 项通过\n")
        
        for step, result in self.results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {step}")
        
        # 保存报告
        report_file = self.project_root / "deployment_report.json"
        report = {
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "python_version": self.python_version,
            "platform": self.platform,
            "results": self.results,
            "summary": {
                "total": total,
                "passed": passed,
                "success_rate": passed / total if total > 0 else 0
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存: {report_file}")
        
        return "SUCCESS" if passed == total else "PARTIAL"
    
    def deploy(self, skip_tests: bool = False, install: bool = True):
        """执行完整部署流程"""
        self.print_header("Math Component 部署工具")
        
        print(f"项目根目录: {self.project_root}")
        print(f"Python版本: {self.python_version}")
        print(f"操作系统: {self.platform}")
        
        # 1. 检查Python版本
        if not self.check_python_version():
            print("\n❌ 部署失败: Python版本不符合要求")
            return False
        
        # 2. 安装依赖
        if install:
            if not self.check_dependencies():
                self.install_dependencies()
        
        # 3. 验证组件
        if not self.verify_math_component():
            print("\n❌ 部署失败: 组件验证失败")
            return False
        
        # 4. 运行测试
        if not skip_tests:
            # 运行AGI集成测试
            self.run_tests("test_agi_math_integration.py")
        
        # 5. 生成配置
        self.create_config()
        
        # 6. 生成报告
        status = self.generate_report()
        
        if status == "SUCCESS":
            print("\n" + "="*80)
            print("  🎉 部署成功！Math Component已准备就绪")
            print("="*80)
            return True
        else:
            print("\n" + "="*80)
            print("  ⚠️ 部署完成（部分步骤失败）")
            print("="*80)
            return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Math Component 部署工具")
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="跳过测试阶段"
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="不自动安装依赖"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        help="项目根目录路径"
    )
    
    args = parser.parse_args()
    
    deployer = DeploymentManager(args.project_root)
    success = deployer.deploy(
        skip_tests=args.skip_tests,
        install=not args.no_install
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
