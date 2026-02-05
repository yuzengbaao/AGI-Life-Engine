#!/usr/bin/env python3
"""
AGI AUTONOMOUS CORE V6.2 - 多文件生成测试

测试 V6.2 系统生成多文件工具项目的能力
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# 导入 V6.2 核心组件
import sys
sys.path.insert(0, str(Path(__file__).parent))

from AGI_AUTONOMOUS_CORE_V6_2 import V62Generator, DeepSeekLLM


class MultiFileProjectGenerator:
    """多文件项目生成器"""

    def __init__(self):
        # 初始化 LLM
        self.llm = DeepSeekLLM()
        self.core = V62Generator(llm=self.llm)
        self.output_dir = Path("output/multi_file_project")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.output_dir / "generation_progress.json"

    def plan_project(self, project_description: str) -> dict:
        """规划项目结构"""
        print(f"\n[规划] 分析项目需求...")
        print(f"[规划] 项目描述: {project_description}")

        # 简化版：针对工具/脚本项目的标准结构
        project_plan = {
            "name": "data_processing_tool",
            "description": project_description,
            "files": [
                {
                    "name": "main.py",
                    "description": "Main entry point - command-line interface and orchestration",
                    "dependencies": [],
                    "lines_estimate": 120
                },
                {
                    "name": "config.py",
                    "description": "Configuration management - load and validate settings",
                    "dependencies": [],
                    "lines_estimate": 80
                },
                {
                    "name": "utils/helpers.py",
                    "description": "Helper functions - file I/O, string utilities, logging setup",
                    "dependencies": [],
                    "lines_estimate": 100
                },
                {
                    "name": "core/processor.py",
                    "description": "Core data processing logic - validation, transformation, aggregation",
                    "dependencies": ["utils/helpers.py", "config.py"],
                    "lines_estimate": 180
                },
                {
                    "name": "core/handlers.py",
                    "description": "Event handlers - error handling, logging, notifications",
                    "dependencies": ["config.py", "utils/helpers.py"],
                    "lines_estimate": 120
                },
            ],
            "docs": [
                {
                    "name": "README.md",
                    "description": "Complete documentation - installation, usage, examples"
                },
                {
                    "name": "requirements.txt",
                    "description": "Python dependencies list"
                }
            ]
        }

        print(f"[规划] 规划了 {len(project_plan['files'])} 个代码文件")
        print(f"[规划] 规划了 {len(project_plan['docs'])} 个文档文件")
        return project_plan

    async def generate_file(
        self,
        file_plan: dict,
        project_context: dict,
        file_index: int,
        total_files: int
    ) -> dict:
        """生成单个文件"""
        filename = file_plan['name']
        print(f"\n[生成 {file_index}/{total_files}] {filename}")
        print(f"  描述: {file_plan['description']}")

        # 构建项目上下文
        context = {
            "project_name": project_context['name'],
            "project_description": project_context['description'],
            "current_file": filename,
            "dependencies": file_plan.get('dependencies', []),
            "previous_files": project_context.get('generated_files', []),
            "file_purpose": file_plan['description']
        }

        # 构建生成提示
        prompt = self._build_generation_prompt(context)

        # 使用 V6.2 生成代码
        output_file = self.output_dir / filename

        # 创建父目录
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 提取要生成的方法列表
        # 对于工具/脚本，我们生成主要的功能模块
        methods = self._extract_methods_from_description(file_plan)

        # 调用 V6.2 核心
        result = await self.core.generate(
            project_desc=prompt,
            methods=methods,
            filename=str(output_file)
        )

        # 计算生成的行数
        lines_generated = 0
        if result['success']:
            with open(output_file, 'r', encoding='utf-8') as f:
                lines_generated = len(f.readlines())

        return {
            "filename": filename,
            "success": result['success'],
            "lines": lines_generated,
            "duration_ms": result.get('duration_ms', 0),
            "timestamp": datetime.now().isoformat()
        }

    def _build_generation_prompt(self, context: dict) -> str:
        """构建代码生成提示"""
        prompt = f"""
# Project: {context['project_name']}

## Project Description
{context['project_description']}

## Current File
{context['current_file']}

## Purpose
{context['file_purpose']}

## Dependencies
{', '.join(context['dependencies']) if context['dependencies'] else 'None'}

## Project Structure
"""
        # 添加已生成的文件信息
        if context['previous_files']:
            prompt += "\n### Previously Generated Files\n"
            for prev_file in context['previous_files']:
                prompt += f"- {prev_file['filename']}: {prev_file.get('purpose', '')}\n"

        prompt += f"""

## Requirements
1. Write clean, production-ready Python code
2. Include comprehensive docstrings
3. Add type hints for all functions
4. Include error handling
5. Follow PEP 8 style guidelines
6. Make it maintainable and extensible
7. Include usage examples in docstrings

Generate complete, runnable code for {context['current_file']}.
"""
        return prompt

    def _extract_methods_from_description(self, file_plan: dict) -> list:
        """从文件描述中提取要生成的方法/类列表"""
        filename = file_plan['name']
        description = file_plan['description']

        # 根据文件类型和描述生成合适的方法列表
        if 'main.py' in filename:
            return [
                'main() - Entry point',
                'parse_arguments() - CLI argument parsing',
                'orchestrate() - Main orchestration logic',
                'print_summary() - Summary output'
            ]
        elif 'config.py' in filename:
            return [
                'load_config() - Load configuration from file',
                'validate_config() - Validate configuration settings',
                'get_default_config() - Get default configuration',
                'ConfigManager class - Configuration management'
            ]
        elif 'helpers.py' in filename:
            return [
                'read_file() - Read file contents',
                'write_file() - Write to file',
                'setup_logger() - Setup logging',
                'format_timestamp() - Format timestamp',
                'sanitize_filename() - Sanitize filename'
            ]
        elif 'processor.py' in filename:
            return [
                'validate_data() - Validate input data',
                'transform_data() - Transform data',
                'aggregate_data() - Aggregate data',
                'filter_data() - Filter data',
                'DataProcessor class - Main processor'
            ]
        elif 'handlers.py' in filename:
            return [
                'handle_error() - Error handling',
                'log_event() - Event logging',
                'send_notification() - Send notifications',
                'ErrorHandler class - Error handler'
            ]
        else:
            # 默认方法列表
            return [
                'initialize() - Initialize',
                'process() - Main processing',
                'cleanup() - Cleanup'
            ]

    async def generate_project(self, project_description: str) -> dict:
        """生成完整项目"""
        print("=" * 80)
        print("AGI V6.2 多文件项目生成器")
        print("=" * 80)

        # 步骤1: 规划项目
        project_plan = self.plan_project(project_description)

        # 步骤2: 按依赖顺序生成文件
        print(f"\n[生成] 开始生成 {len(project_plan['files'])} 个文件...")

        generated_files = []
        for i, file_plan in enumerate(project_plan['files'], 1):
            context = {
                'name': project_plan['name'],
                'description': project_plan['description'],
                'generated_files': generated_files
            }

            result = await self.generate_file(
                file_plan=file_plan,
                project_context=context,
                file_index=i,
                total_files=len(project_plan['files'])
            )

            if result['success']:
                generated_files.append({
                    'filename': result['filename'],
                    'purpose': file_plan['description']
                })
                print(f"  ✅ 成功: {result['lines']} 行, {result['duration_ms']/1000:.1f}秒")
            else:
                print(f"  ❌ 失败")

            # 保存进度
            self._save_progress({
                'plan': project_plan,
                'generated_files': generated_files,
                'current_file': i,
                'total_files': len(project_plan['files'])
            })

        # 步骤3: 生成文档
        print(f"\n[文档] 生成项目文档...")
        for doc in project_plan['docs']:
            print(f"  - {doc['name']}")

        # 步骤4: 生成报告
        total_lines = sum(f.get('lines', 0) for f in generated_files)
        total_time = sum(f.get('duration_ms', 0) for f in generated_files)

        report = {
            'success': True,
            'project_name': project_plan['name'],
            'files_generated': len(generated_files),
            'total_lines': total_lines,
            'total_duration_ms': total_time,
            'output_directory': str(self.output_dir),
            'generated_files': generated_files,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "=" * 80)
        print("生成完成!")
        print("=" * 80)
        print(f"项目名称: {report['project_name']}")
        print(f"生成文件: {report['files_generated']} 个")
        print(f"代码行数: {report['total_lines']} 行")
        print(f"总耗时: {report['total_duration_ms']/1000:.1f} 秒")
        print(f"输出目录: {report['output_directory']}")
        print("=" * 80)

        return report

    def _save_progress(self, progress: dict):
        """保存生成进度"""
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)


async def main():
    """主函数"""
    generator = MultiFileProjectGenerator()

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

    result = await generator.generate_project(project_description)

    # 保存最终报告
    report_file = Path("output") / f"multi_file_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存到: {report_file}")


if __name__ == "__main__":
    asyncio.run(main())
