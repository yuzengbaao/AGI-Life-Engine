#!/usr/bin/env python3
"""
AGI AUTONOMOUS CORE V6.2 - 多文件项目生成器 V2（优化版）

改进:
1. 修复编码问题（移除emoji）
2. 简化上下文（减少token消耗）
3. 模块化生成策略（每个文件150-250行）
4. 进度持久化
5. 更好的错误处理
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 导入 V6.2 核心组件
import sys
sys.path.insert(0, str(Path(__file__).parent))

from AGI_AUTONOMOUS_CORE_V6_2 import V62Generator, DeepSeekLLM


class MultiFileProjectGeneratorV2:
    """多文件项目生成器 V2 - 优化版"""

    def __init__(self):
        # 初始化 LLM
        self.llm = DeepSeekLLM()
        self.core = V62Generator(llm=self.llm)

        # 输出目录
        self.output_dir = Path("output/multi_file_project_v2")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "generation_progress.json"

        # 统计信息
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_lines': 0,
            'total_time_ms': 0
        }

    def plan_modular_project(self, project_description: str) -> Dict[str, Any]:
        """
        规划模块化项目结构

        策略: 将大型项目拆分为多个中小型模块（每个150-250行）
        """
        print(f"\n[规划] 分析项目需求...")
        print(f"[规划] 项目描述: {project_description[:100]}...")

        # 模块化项目结构
        project_plan = {
            "name": "data_processing_tool",
            "description": project_description,
            "modules": [
                {
                    "name": "main.py",
                    "description": "Main entry point with CLI",
                    "methods": [
                        "main() - Entry point",
                        "parse_args() - CLI parsing"
                    ],
                    "lines_target": 150,
                    "dependencies": []
                },
                {
                    "name": "config.py",
                    "description": "Configuration management",
                    "methods": [
                        "load_config() - Load from file",
                        "validate_config() - Validate settings",
                        "ConfigManager class"
                    ],
                    "lines_target": 100,
                    "dependencies": []
                },
                {
                    "name": "utils/helpers.py",
                    "description": "Helper utilities",
                    "methods": [
                        "read_file() - File I/O",
                        "write_file() - File I/O",
                        "setup_logger() - Logging",
                        "format_bytes() - Format utilities"
                    ],
                    "lines_target": 150,
                    "dependencies": []
                },
                {
                    "name": "core/validator.py",
                    "description": "Data validation logic",
                    "methods": [
                        "validate_schema() - Schema validation",
                        "check_types() - Type checking",
                        "DataValidator class"
                    ],
                    "lines_target": 180,
                    "dependencies": ["utils/helpers.py", "config.py"]
                },
                {
                    "name": "core/processor.py",
                    "description": "Data processing engine",
                    "methods": [
                        "clean_data() - Data cleaning",
                        "transform_data() - Transform",
                        "aggregate_data() - Aggregation",
                        "DataProcessor class"
                    ],
                    "lines_target": 220,
                    "dependencies": ["core/validator.py", "utils/helpers.py"]
                },
                {
                    "name": "core/reporter.py",
                    "description": "Report generation",
                    "methods": [
                        "generate_excel_report() - Excel output",
                        "generate_pdf_report() - PDF output",
                        "ReportGenerator class"
                    ],
                    "lines_target": 200,
                    "dependencies": ["config.py", "utils/helpers.py"]
                }
            ],
            "docs": [
                {
                    "name": "README.md",
                    "content": self._generate_readme_content()
                },
                {
                    "name": "requirements.txt",
                    "content": "pandas\nopenpyxl\nreportlab\npyyaml\n"
                }
            ]
        }

        total_lines = sum(m['lines_target'] for m in project_plan['modules'])
        print(f"[规划] 规划了 {len(project_plan['modules'])} 个模块，总计 {total_lines} 行")
        return project_plan

    async def generate_module(
        self,
        module: Dict[str, Any],
        project_context: Dict[str, Any],
        module_index: int,
        total_modules: int
    ) -> Dict[str, Any]:
        """
        生成单个模块

        优化: 简化上下文，专注当前模块
        """
        module_name = module['name']
        print(f"\n[生成 {module_index}/{total_modules}] {module_name}")
        print(f"  目标: {module['description']}")
        print(f"  预期: {module['lines_target']} 行")

        # 构建简化的生成提示
        prompt = self._build_simplified_prompt(module, project_context)

        # 方法列表
        methods = module['methods']

        # 输出文件路径
        output_file = self.output_dir / module_name

        # 创建父目录
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 记录开始时间
        start_time = datetime.now()

        try:
            # 调用 V6.2 生成
            result = await self.core.generate(
                project_desc=prompt,
                methods=methods,
                filename=str(output_file)
            )

            # 计算实际行数
            actual_lines = 0
            if result['success'] and output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    actual_lines = len(f.readlines())

            duration_ms = result.get('duration_ms', 0)

            # 打印结果（ASCII字符，避免编码错误）
            if result['success']:
                print(f"  [OK] 成功: {actual_lines} 行, {duration_ms/1000:.1f}秒")
                self.stats['successful'] += 1
                self.stats['total_lines'] += actual_lines
            else:
                print(f"  [FAIL] 失败")
                self.stats['failed'] += 1

            self.stats['total_time_ms'] += duration_ms

            return {
                "module_name": module_name,
                "success": result['success'],
                "target_lines": module['lines_target'],
                "actual_lines": actual_lines,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"  [ERROR] 异常: {e}")
            self.stats['failed'] += 1
            return {
                "module_name": module_name,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _build_simplified_prompt(self, module: Dict[str, Any], project_context: Dict[str, Any]) -> str:
        """
        构建简化的生成提示

        优化: 减少上下文，节省token
        """
        prompt = f"""# {module['name']}

## Purpose
{module['description']}

## Methods to Implement
{chr(10).join(f"- {m}" for m in module['methods'])}

## Dependencies
{', '.join(module['dependencies']) if module['dependencies'] else 'None'}

## Requirements
1. Clean, production-ready Python code
2. Comprehensive docstrings (Google style)
3. Type hints for all functions
4. Error handling
5. PEP 8 compliant
6. Include usage examples

Generate complete code for {module['name']}."""

        return prompt

    async def generate_project(self, project_description: str) -> Dict[str, Any]:
        """生成完整项目"""
        print("=" * 80)
        print("AGI V6.2 多文件项目生成器 V2（优化版）")
        print("=" * 80)

        # 步骤1: 规划项目
        project_plan = self.plan_modular_project(project_description)
        self.stats['total_files'] = len(project_plan['modules'])

        # 步骤2: 生成模块
        print(f"\n[生成] 开始生成 {len(project_plan['modules'])} 个模块...")

        generated_modules = []
        for i, module in enumerate(project_plan['modules'], 1):
            context = {
                'project_name': project_plan['name'],
                'description': project_plan['description'],
                'generated_modules': generated_modules
            }

            result = await self.generate_module(
                module=module,
                project_context=context,
                module_index=i,
                total_modules=len(project_plan['modules'])
            )

            generated_modules.append(result)

            # 保存进度
            self._save_progress({
                'plan': project_plan,
                'generated_modules': generated_modules,
                'stats': self.stats
            })

        # 步骤3: 生成文档
        print(f"\n[文档] 生成项目文档...")
        for doc in project_plan['docs']:
            doc_file = self.output_dir / doc['name']
            doc_file.write_text(doc['content'], encoding='utf-8')
            print(f"  [OK] {doc['name']}")

        # 步骤4: 生成报告
        report = {
            'success': self.stats['successful'] == self.stats['total_files'],
            'project_name': project_plan['name'],
            'total_modules': self.stats['total_files'],
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'total_lines': self.stats['total_lines'],
            'total_duration_ms': self.stats['total_time_ms'],
            'output_directory': str(self.output_dir),
            'modules': generated_modules,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "=" * 80)
        print("生成完成!")
        print("=" * 80)
        print(f"项目名称: {report['project_name']}")
        print(f"模块总数: {report['total_modules']}")
        print(f"成功: {report['successful']}")
        print(f"失败: {report['failed']}")
        print(f"代码行数: {report['total_lines']} 行")
        print(f"总耗时: {report['total_duration_ms']/1000:.1f} 秒")
        print(f"输出目录: {report['output_directory']}")
        print("=" * 80)

        return report

    def _save_progress(self, progress: Dict[str, Any]):
        """保存生成进度"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [WARN] 保存进度失败: {e}")

    def _generate_readme_content(self) -> str:
        """生成 README 内容"""
        return """# Data Processing Tool

A powerful Python tool for processing, validating, and transforming data files.

## Features

- Read data from CSV/Excel files
- Data validation and cleaning
- Data transformation and aggregation
- Generate reports (Excel/PDF)
- Command-line interface
- Configuration file support
- Comprehensive logging

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --input data.csv --output result.xlsx
```

### With Configuration File

```bash
python main.py --config config.yaml --input data.csv
```

### Generate PDF Report

```bash
python main.py --input data.csv --output report.pdf --format pdf
```

## Configuration

Create a `config.yaml` file:

```yaml
input:
  encoding: utf-8
  sheet_name: 0

output:
  format: xlsx
  include_index: false

processing:
  remove_duplicates: true
  fill_missing: forward
```

## Project Structure

```
.
├── main.py              # Entry point
├── config.py            # Configuration management
├── utils/
│   └── helpers.py       # Helper utilities
├── core/
│   ├── validator.py     # Data validation
│   ├── processor.py     # Data processing
│   └── reporter.py      # Report generation
├── README.md
└── requirements.txt
```

## License

MIT License
"""

    def print_summary(self):
        """打印项目摘要"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)

            print("\n[摘要] 项目生成摘要:")
            for module in progress.get('generated_modules', []):
                status = "[OK]" if module.get('success') else "[FAIL]"
                print(f"  {status} {module.get('module_name', 'unknown')}: "
                      f"{module.get('actual_lines', 0)} 行")


async def main():
    """主函数"""
    generator = MultiFileProjectGeneratorV2()

    # 示例项目：数据处理工具
    project_description = """
创建一个数据处理工具，具有以下功能：
1. 从CSV或Excel文件读取数据
2. 数据验证和清洗（处理缺失值、异常值）
3. 数据转换和聚合
4. 生成处理报告（Excel/PDF）
5. 命令行界面，支持多种操作模式
6. 配置文件支持
7. 日志记录
8. 错误处理和通知
"""

    try:
        result = await generator.generate_project(project_description)

        # 保存最终报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = Path("output") / f"multi_file_generation_v2_{timestamp}.json"

        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n[报告] 已保存到: {report_file}")

        # 打印摘要
        generator.print_summary()

    except KeyboardInterrupt:
        print("\n[中断] 用户取消操作")
        generator.print_summary()
    except Exception as e:
        print(f"\n[错误] 生成失败: {e}")
        raise


if __name__ == "__main__":
    # 设置Windows控制台编码（避免中文乱码）
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

    asyncio.run(main())
