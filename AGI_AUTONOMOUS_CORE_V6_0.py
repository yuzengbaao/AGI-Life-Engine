#!/usr/bin/env python3
"""
AGI AUTONOMOUS CORE V6.0 - MULTI-FILE BATCH GENERATION

核心改进：
- 支持多文件/多模块项目生成
- 每个模块独立骨架生成和分批实现
- 自动创建目录结构
- 完整的依赖管理
- 保持 V5.0 的分批生成能力
"""

import asyncio
import json
import time
import os
import sys
import ast
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

# Fix encoding for Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[Init] Environment variables loaded")
except:
    print("[Init] dotenv not available, using system env")


class DeepSeekLLM:
    """DeepSeek LLM 客户端"""

    def __init__(self):
        self.client = None
        self.model = None
        self._init_provider()

    def _init_provider(self):
        """Initialize DeepSeek provider"""
        try:
            import openai

            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                print("[LLM] Warning: DEEPSEEK_API_KEY not found")
                return

            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

            print(f"[LLM] DeepSeek client initialized")
            print(f"[LLM] Model: {self.model}")
            print(f"[LLM] Multi-file batch generation enabled")

        except ImportError:
            print("[LLM] Error: openai package not installed")
        except Exception as e:
            print(f"[LLM] Error: {e}")

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 8000,
        temperature: float = 0.7
    ) -> str:
        """生成响应"""
        if not self.client:
            return self._simulate_response(prompt)

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[LLM] API error: {e}")
            return self._simulate_response(prompt)

    def _simulate_response(self, prompt: str) -> str:
        """模拟响应"""
        return "# Simulated response\ndef placeholder():\n    pass"


class MultiFileBatchGenerator:
    """
    多文件批量生成器

    核心策略：
    1. 解析项目结构，识别所有需要生成的模块
    2. 为每个模块独立生成骨架
    3. 对每个模块进行分批方法实现
    4. 创建正确的目录结构
    5. 处理模块间的依赖关系
    """

    def __init__(self, llm: DeepSeekLLM):
        self.llm = llm
        self.stats = {
            "files_generated": 0,
            "total_methods": 0,
            "total_batches": 0,
            "total_tokens": 0
        }

    async def generate_project(
        self,
        project_description: str,
        base_dir: str
    ) -> Dict[str, Any]:
        """
        生成完整的多模块项目

        Args:
            project_description: 项目描述（包含模块列表）
            base_dir: 项目根目录

        Returns:
            生成结果统计
        """
        print(f"\n[Project] Starting multi-file project generation...")
        print(f"[Project] Base directory: {base_dir}")

        # Step 1: 解析项目结构
        print(f"\n[Step 1] Parsing project structure...")
        modules = await self._parse_project_structure(project_description)

        if not modules:
            print(f"[Project] No modules found in description")
            return {"status": "failed", "reason": "no modules"}

        print(f"[Step 1] Found {len(modules)} modules to generate:")
        for i, module in enumerate(modules, 1):
            print(f"  {i}. {module['path']}")

        # Step 2: 为每个模块生成代码
        print(f"\n[Step 2] Generating modules...")
        generated_files = []

        for i, module in enumerate(modules, 1):
            print(f"\n[{i}/{len(modules)}] Generating {module['path']}...")
            print(f"  Description: {module.get('description', 'N/A')}")

            # 生成单个模块
            code, methods_count, batches = await self._generate_module(
                module,
                base_dir
            )

            if code:
                # 保存文件
                file_path = self._save_module(base_dir, module['path'], code)
                generated_files.append(file_path)

                print(f"  ✓ Generated {methods_count} methods in {batches} batches")
                print(f"  ✓ Saved to: {file_path}")

                self.stats["files_generated"] += 1
                self.stats["total_methods"] += methods_count
                self.stats["total_batches"] += batches
            else:
                print(f"  ✗ Failed to generate {module['path']}")

        # Step 3: 验证所有文件
        print(f"\n[Step 3] Validating all generated files...")
        validation_results = await self._validate_project(base_dir, generated_files)

        # Step 4: 生成项目元数据
        print(f"\n[Step 4] Generating project metadata...")
        self._generate_metadata(base_dir, modules, validation_results)

        # 打印统计
        print(f"\n{'='*70}")
        print(f"[Project] Generation Complete!")
        print(f"{'='*70}")
        print(f"Files generated: {self.stats['files_generated']}")
        print(f"Total methods: {self.stats['total_methods']}")
        print(f"Total batches: {self.stats['total_batches']}")
        print(f"Est. tokens used: {self.stats['total_tokens']}")
        print(f"Validation: {'✓ All files valid' if validation_results['all_valid'] else '⚠ Some files have issues'}")

        return {
            "status": "success",
            "files": generated_files,
            "stats": self.stats,
            "validation": validation_results
        }

    async def _parse_project_structure(self, description: str) -> List[Dict]:
        """
        解析项目描述，提取模块列表

        支持的格式：
        1) core/task_parser.py – parses natural language tasks
        2) core/action_executor.py – selects and executes actions
        """
        prompt = f"""You are analyzing a project description to identify all Python modules that need to be generated.

Project Description:
{description}

Extract ALL Python module files mentioned. For each module, identify:
1. The file path (e.g., "core/task_parser.py")
2. The purpose/description of the module

Return JSON:
{{
    "modules": [
        {{
            "path": "core/task_parser.py",
            "description": "parses natural language tasks into structured actions"
        }},
        {{
            "path": "core/action_executor.py",
            "description": "selects and executes actions"
        }}
    ]
}}

IMPORTANT:
- Extract ALL modules mentioned
- Include main.py or entry points if mentioned
- Return ONLY valid JSON"""

        try:
            response = await self.llm.generate(prompt, max_tokens=2000, temperature=0.3)

            # 提取 JSON
            json_str = self._extract_json(response)
            data = json.loads(json_str)

            return data.get("modules", [])

        except Exception as e:
            print(f"[Parse] Error: {e}")

            # Fallback: 使用正则表达式提取
            modules = []
            pattern = r'([\w/]+\.py)\s*[–-]\s*([^\n]+)'
            matches = re.findall(pattern, description)

            for path, desc in matches:
                modules.append({
                    "path": path.strip(),
                    "description": desc.strip()
                })

            return modules

    async def _generate_module(
        self,
        module: Dict,
        base_dir: str
    ) -> Tuple[str, int, int]:
        """
        生成单个模块

        Returns:
            (code, methods_count, batches_count)
        """
        module_path = module['path']
        description = module.get('description', '')

        # Phase 1: 生成模块骨架
        skeleton_prompt = f"""Generate a Python module skeleton for:

File: {module_path}
Purpose: {description}

Requirements:
1. Include necessary imports
2. Define class(es) with proper names
3. Add method signatures with docstrings
4. Use 'pass' for method bodies
5. Include type hints
6. Make it production-ready

Output ONLY the complete Python code:"""

        skeleton_response = await self.llm.generate(skeleton_prompt, max_tokens=4000)
        skeleton = self._extract_code(skeleton_response)

        # 提取方法名
        methods = re.findall(r'def\s+(\w+)\s*\(', skeleton)

        if not methods:
            print(f"  [Warning] No methods found in skeleton")
            return skeleton, 0, 0

        print(f"  [Phase 1] Skeleton: {len(methods)} methods found")

        # Phase 2: 分批实现方法
        implemented_code = skeleton
        implemented_methods = []
        batches = 0
        max_methods_per_batch = 3

        num_batches = (len(methods) + max_methods_per_batch - 1) // max_methods_per_batch

        for batch_num in range(num_batches):
            start_idx = batch_num * max_methods_per_batch
            end_idx = min(start_idx + max_methods_per_batch, len(methods))
            batch_methods = methods[start_idx:end_idx]

            # 实现这一批方法
            batch_code = await self._implement_methods(
                implemented_code,
                batch_methods,
                implemented_methods,
                module_path
            )

            if batch_code:
                implemented_code = batch_code
                implemented_methods.extend(batch_methods)
                batches += 1

                # 更新统计
                self.stats['total_tokens'] += 6000
            else:
                print(f"  [Batch {batch_num + 1}] ✗ Failed")

        return implemented_code, len(methods), batches

    async def _implement_methods(
        self,
        current_code: str,
        batch_methods: List[str],
        implemented_methods: List[str],
        module_path: str
    ) -> str:
        """实现一批方法"""

        methods_str = ", ".join(batch_methods)
        implemented_str = ", ".join(implemented_methods) if implemented_methods else "None"

        prompt = f"""You are implementing methods for module: {module_path}

Current code state:
```python
{current_code[:3000]}
{"..." if len(current_code) > 3000 else ""}
```

Already implemented: {implemented_str}

Task: Implement ONLY these methods: {methods_str}
- Replace their 'pass' with actual working code
- Keep all other methods as 'pass'
- Maintain the exact same structure
- Return the FULL updated code

Output the complete updated code:"""

        response = await self.llm.generate(prompt, max_tokens=8000)
        return self._extract_code(response)

    def _save_module(self, base_dir: str, module_path: str, code: str) -> str:
        """保存模块到文件"""
        # 创建完整路径
        full_path = os.path.join(base_dir, module_path)

        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # 保存文件
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(code)

        return full_path

    async def _validate_project(
        self,
        base_dir: str,
        files: List[str]
    ) -> Dict:
        """验证所有生成的文件"""
        results = {
            "all_valid": True,
            "files": {}
        }

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                # 语法检查
                ast.parse(code)

                results["files"][file_path] = {
                    "valid": True,
                    "syntax_ok": True,
                    "lines": len(code.split('\n'))
                }

            except SyntaxError as e:
                results["all_valid"] = False
                results["files"][file_path] = {
                    "valid": False,
                    "error": str(e)
                }
                print(f"  [Validation] ✗ {file_path}: {e}")
            except Exception as e:
                results["all_valid"] = False
                results["files"][file_path] = {
                    "valid": False,
                    "error": str(e)
                }
                print(f"  [Validation] ✗ {file_path}: {e}")

        return results

    def _generate_metadata(
        self,
        base_dir: str,
        modules: List[Dict],
        validation: Dict
    ):
        """生成项目元数据"""
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "generator": "AGI_AUTONOMOUS_CORE_V6_0",
            "modules": modules,
            "validation": validation,
            "stats": self.stats
        }

        metadata_path = os.path.join(base_dir, "project_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"  ✓ Metadata saved to: {metadata_path}")

    def _extract_code(self, text: str) -> str:
        """提取代码块"""
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text

    def _extract_json(self, text: str) -> str:
        """提取 JSON"""
        try:
            if "```json" in text:
                return text.split("```json")[1].split("```")[0].strip()
            if "```" in text:
                return text.split("```")[1].split("```")[0].strip()
            if "{" in text and "}" in text:
                return text[text.find("{"):text.rfind("}")+1]
            return text
        except:
            return "{}"


class AutonomousAGI_V6_0:
    """
    AGI Core V6.0 - 多文件批量生成版本

    核心改进：
    - 支持完整的多模块项目生成
    - 每个模块独立分批实现
    - 自动目录结构创建
    - 完整的验证和元数据
    """

    def __init__(self):
        print("=" * 70)
        print("AGI AUTONOMOUS CORE V6.0 - MULTI-FILE GENERATION")
        print("=" * 70)
        print("[Improvement] Multi-file project support")
        print("[Improvement] Independent module generation")
        print("[Improvement] Automatic directory structure")
        print("[Improvement] Complete validation")
        print("=" * 70)

        self.llm = DeepSeekLLM()
        self.generator = MultiFileBatchGenerator(self.llm)
        self.memory = []
        self.step_count = 0

        self.workspace = "data/autonomous_outputs_v6_0"
        os.makedirs(self.workspace, exist_ok=True)

        print(f"[Init] Workspace: {self.workspace}")
        print(f"[Init] Ready. Multi-file generation enabled.")
        print("=" * 70)

    async def autonomous_loop(self):
        """完全自主循环"""
        while True:
            self.step_count += 1
            tick_time = datetime.now().strftime("%H:%M:%S")
            print(f"\n[Tick {self.step_count}] {tick_time}")
            print("-" * 70)

            try:
                # 自主决策
                goal = await self._autonomous_decision()

                if goal["action"] == "create_project":
                    # 创建多模块项目
                    await self._create_project(goal)

                elif goal["action"] == "reflect":
                    await self._self_reflection()

                elif goal["action"] == "improve":
                    await self._improve_project()

                else:
                    print(f"[Action] {goal['action']}: {goal.get('reasoning', '')}")

                # 记录经验
                self.memory.append({
                    "tick": self.step_count,
                    "goal": goal,
                    "timestamp": time.time()
                })

            except Exception as e:
                print(f"[Error] {e}")
                import traceback
                traceback.print_exc()

            # 自主节奏
            await asyncio.sleep(5)

    async def _autonomous_decision(self) -> Dict:
        """自主决策"""
        context = {
            "tick": self.step_count,
            "memory_size": len(self.memory),
            "recent": self.memory[-3:] if self.memory else []
        }

        prompt = f"""You are an autonomous AGI system.

State: {json.dumps(context, indent=2, default=str)}

Decide your next action:
1. **create_project** - Generate a multi-module Python project
2. **reflect** - Analyze your performance
3. **improve** - Enhance previous project

Return JSON:
{{
    "action": "create_project|reflect|improve",
    "reasoning": "why",
    "project_description": "detailed project description (if action=create_project)"
}}"""

        try:
            response = await self.llm.generate(prompt, temperature=0.8)
            decision = json.loads(self._extract_json(response))
            print(f"[Decision] {decision['action']}: {decision.get('reasoning', '')}")
            return decision
        except:
            # 默认创建一个项目
            return {
                "action": "create_project",
                "reasoning": "Starting with a new project",
                "project_description": "Generate a complete Python package with the following modules: 1) core/task_parser.py – parses natural language tasks into structured actions, 2) core/action_executor.py – selects and executes actions, 3) memory/memory_manager.py – stores and retrieves experiences, 4) loop/self_improvement.py – implements reflection and improvement cycles. Include a main.py entry point and unit tests for each module."
            }

    async def _create_project(self, goal: Dict):
        """创建多模块项目"""
        output_id = f"project_{int(time.time())}"
        output_dir = os.path.join(self.workspace, output_id)

        project_desc = goal.get("project_description", "")

        print(f"\n[Project] Output ID: {output_id}")
        print(f"[Project] Description: {project_desc[:150]}...")

        # 生成项目
        result = await self.generator.generate_project(project_desc, output_dir)

        # 保存结果
        result_file = os.path.join(output_dir, "generation_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n[Result] Project saved to: {output_dir}")

    async def _self_reflection(self):
        """自我反思"""
        print(f"\n[Reflection] Analyzing performance...")
        print(f"[Reflection] Total projects: {len([m for m in self.memory if m.get('goal', {}).get('action') == 'create_project'])}")

    async def _improve_project(self):
        """改进项目"""
        print(f"\n[Improve] Scanning for previous projects...")

    def _extract_json(self, text: str) -> str:
        """提取 JSON"""
        try:
            if "```json" in text:
                return text.split("```json")[1].split("```")[0].strip()
            if "```" in text:
                return text.split("```")[1].split("```")[0].strip()
            if "{" in text and "}" in text:
                return text[text.find("{"):text.rfind("}")+1]
            return text
        except:
            return "{}"


if __name__ == "__main__":
    agi = AutonomousAGI_V6_0()
    asyncio.run(agi.autonomous_loop())
