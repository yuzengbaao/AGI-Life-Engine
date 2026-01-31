#!/usr/bin/env python3
"""
AGI AUTONOMOUS CORE V6.1 - MULTI-BASE MODEL SUPPORT

核心改进：
- 支持多种基座模型（DeepSeek, Zhipu, Kimi, Qwen, Gemini）
- 多实例并行运行
- 基座模型性能对比
- 动态基座模型切换
"""

import asyncio
import json
import time
import os
import sys
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from enum import Enum

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


class BaseModel(Enum):
    """支持的基座模型"""
    DEEPSEEK = "deepseek"
    ZHIPU = "zhipu"
    KIMI = "kimi"
    QWEN = "qwen"
    GEMINI = "gemini"


class BaseLLM:
    """通用 LLM 客户端基类"""

    def __init__(self, model_type: BaseModel):
        self.model_type = model_type
        self.client = None
        self.model = None
        self.base_url = None
        self._init_provider()

    def _init_provider(self):
        """初始化对应的 LLM provider"""
        providers = {
            BaseModel.DEEPSEEK: self._init_deepseek,
            BaseModel.ZHIPU: self._init_zhipu,
            BaseModel.KIMI: self._init_kimi,
            BaseModel.QWEN: self._init_qwen,
            BaseModel.GEMINI: self._init_gemini,
        }

        init_func = providers.get(self.model_type)
        if init_func:
            init_func()
        else:
            print(f"[LLM] Unsupported model type: {self.model_type}")

    def _init_deepseek(self):
        """初始化 DeepSeek"""
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
            self.base_url = "https://api.deepseek.com/v1"

            print(f"[LLM] DeepSeek client initialized")
            print(f"[LLM] Model: {self.model}")

        except ImportError:
            print("[LLM] Error: openai package not installed")
        except Exception as e:
            print(f"[LLM] Error: {e}")

    def _init_zhipu(self):
        """初始化智谱 GLM"""
        try:
            import openai

            api_key = os.getenv("ZHIPU_API_KEY")
            if not api_key:
                print("[LLM] Warning: ZHIPU_API_KEY not found")
                return

            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4"
            )
            self.model = os.getenv("ZHIPU_MODEL", "glm-4-plus")
            self.base_url = "https://open.bigmodel.cn/api/paas/v4"

            print(f"[LLM] Zhipu GLM client initialized")
            print(f"[LLM] Model: {self.model}")

        except ImportError:
            print("[LLM] Error: openai package not installed")
        except Exception as e:
            print(f"[LLM] Error: {e}")

    def _init_kimi(self):
        """初始化 Moonshot Kimi"""
        try:
            import openai

            api_key = os.getenv("KIMI_API_KEY")
            if not api_key:
                print("[LLM] Warning: KIMI_API_KEY not found")
                return

            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.moonshot.cn/v1"
            )
            self.model = os.getenv("KIMI_MODEL", "moonshot-v1-128k")
            self.base_url = "https://api.moonshot.cn/v1"

            print(f"[LLM] Kimi client initialized")
            print(f"[LLM] Model: {self.model}")

        except ImportError:
            print("[LLM] Error: openai package not installed")
        except Exception as e:
            print(f"[LLM] Error: {e}")

    def _init_qwen(self):
        """初始化千问 Qwen"""
        try:
            import openai

            api_key = os.getenv("QWEN_API_KEY")
            if not api_key:
                print("[LLM] Warning: QWEN_API_KEY not found")
                return

            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = os.getenv("QWEN_MODEL", "qwen-plus")
            self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

            print(f"[LLM] Qwen client initialized")
            print(f"[LLM] Model: {self.model}")

        except ImportError:
            print("[LLM] Error: openai package not installed")
        except Exception as e:
            print(f"[LLM] Error: {e}")

    def _init_gemini(self):
        """初始化 Gemini"""
        try:
            import openai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("[LLM] Warning: GEMINI_API_KEY not found")
                return

            # Gemini 通过 OpenAI 兼容接口或使用 Google AI SDK
            # 这里使用 OpenAI SDK 的反向代理或直接调用
            self.client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            self.model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

            print(f"[LLM] Gemini client initialized")
            print(f"[LLM] Model: {self.model}")

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


class MultiModelBatchGenerator:
    """多模型批量生成器"""

    def __init__(self, llm: BaseLLM, model_type: BaseModel):
        self.llm = llm
        self.model_type = model_type
        self.stats = {
            "model": model_type.value,
            "files_generated": 0,
            "total_methods": 0,
            "total_batches": 0,
            "total_tokens": 0,
            "start_time": time.time()
        }

    async def generate_project(
        self,
        project_description: str,
        base_dir: str
    ) -> Dict[str, Any]:
        """生成完整的多模块项目"""
        print(f"\n[Project] Starting multi-file project generation...")
        print(f"[Project] Base model: {self.model_type.value}")
        print(f"[Project] Base directory: {base_dir}")

        # Step 1: 解析项目结构
        print(f"\n[Step 1] Parsing project structure...")
        modules = await self._parse_project_structure(project_description)

        if not modules:
            print(f"[Project] No modules found in description")
            return {"status": "failed", "reason": "no modules"}

        print(f"[Step 1] Found {len(modules)} modules to generate")

        # Step 2: 为每个模块生成代码
        print(f"\n[Step 2] Generating modules...")
        generated_files = []

        for i, module in enumerate(modules, 1):
            print(f"\n[{i}/{len(modules)}] Generating {module['path']}...")

            code, methods_count, batches = await self._generate_module(
                module,
                base_dir
            )

            if code:
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

        self.stats["end_time"] = time.time()
        self.stats["duration"] = self.stats["end_time"] - self.stats["start_time"]

        return {
            "status": "success",
            "files": generated_files,
            "stats": self.stats,
            "validation": validation_results
        }

    async def _parse_project_structure(self, description: str) -> List[Dict]:
        """解析项目描述，提取模块列表"""
        prompt = f"""Extract ALL Python module files from this project description:

{description}

Return JSON:
{{
    "modules": [
        {{"path": "core/task_parser.py", "description": "parses natural language tasks"}},
        {{"path": "core/action_executor.py", "description": "executes actions"}}
    ]
}}"""

        try:
            response = await self.llm.generate(prompt, max_tokens=2000, temperature=0.3)
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
    ) -> tuple:
        """生成单个模块"""
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
        full_path = os.path.join(base_dir, module_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(code)

        return full_path

    async def _validate_project(
        self,
        base_dir: str,
        files: List[str]
    ) -> Dict:
        """验证所有生成的文件"""
        import ast

        results = {
            "all_valid": True,
            "files": {}
        }

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()

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
            "generator": f"AGI_AUTONOMOUS_CORE_V6_1_{self.model_type.value.upper()}",
            "base_model": self.model_type.value,
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
                return text.split("```\")[1].split("```")[0].strip()
            if "{" in text and "}" in text:
                return text[text.find("{"):text.rfind("}")+1]
            return text
        except:
            return "{}"


class AutonomousAGI_V6_1:
    """AGI Core V6.1 - 多基座模型支持"""

    def __init__(self, model_type: BaseModel, instance_id: Optional[str] = None):
        self.model_type = model_type
        self.instance_id = instance_id or f"inst_{model_type.value}_{int(time.time())}"

        print("=" * 70)
        print(f"AGI AUTONOMOUS CORE V6.1 - {model_type.value.upper()}")
        print("=" * 70)
        print(f"[Instance] ID: {self.instance_id}")
        print(f"[Model] {model_type.value}")

        self.llm = BaseLLM(model_type)
        self.generator = MultiModelBatchGenerator(self.llm, model_type)
        self.memory = []
        self.step_count = 0

        self.workspace = f"data/autonomous_outputs_v6_1/{model_type.value}"
        os.makedirs(self.workspace, exist_ok=True)

        print(f"[Init] Workspace: {self.workspace}")
        print(f"[Init] Ready. Base model: {model_type.value}")
        print("=" * 70)

    async def autonomous_loop(self, max_ticks: int = 5):
        """自主循环（带限制）"""
        while self.step_count < max_ticks:
            self.step_count += 1
            tick_time = datetime.now().strftime("%H:%M:%S")
            print(f"\n[Tick {self.step_count}] {tick_time}")
            print("-" * 70)

            try:
                goal = await self._autonomous_decision()

                if goal["action"] == "create_project":
                    await self._create_project(goal)
                elif goal["action"] == "reflect":
                    await self._self_reflection()
                else:
                    print(f"[Action] {goal['action']}: {goal.get('reasoning', '')}")

                self.memory.append({
                    "tick": self.step_count,
                    "goal": goal,
                    "timestamp": time.time()
                })

            except Exception as e:
                print(f"[Error] {e}")
                import traceback
                traceback.print_exc()

            await asyncio.sleep(5)

    async def _autonomous_decision(self) -> Dict:
        """自主决策"""
        context = {
            "tick": self.step_count,
            "memory_size": len(self.memory),
            "base_model": self.model_type.value,
            "recent": self.memory[-3:] if self.memory else []
        }

        prompt = f"""You are an autonomous AGI system powered by {self.model_type.value.upper()}.

State: {json.dumps(context, indent=2, default=str)}

Decide your next action:
1. **create_project** - Generate a multi-module Python project
2. **reflect** - Analyze your performance

Return JSON:
{{
    "action": "create_project|reflect",
    "reasoning": "why",
    "project_description": "detailed description (if action=create_project)"
}}"""

        try:
            response = await self.llm.generate(prompt, temperature=0.8)
            decision = json.loads(self._extract_json(response))
            print(f"[Decision] {decision['action']}: {decision.get('reasoning', '')}")
            return decision
        except:
            return {
                "action": "create_project",
                "reasoning": "Starting with a new project",
                "project_description": "Generate a complete Python package with the following modules: 1) core/task_parser.py – parses natural language tasks, 2) core/action_executor.py – executes actions, 3) memory/memory_manager.py – stores experiences, 4) main.py – entry point. Include unit tests."
            }

    async def _create_project(self, goal: Dict):
        """创建多模块项目"""
        output_id = f"project_{int(time.time())}"
        output_dir = os.path.join(self.workspace, output_id)

        project_desc = goal.get("project_description", "")

        print(f"\n[Project] Output ID: {output_id}")
        print(f"[Project] Description: {project_desc[:150]}...")

        result = await self.generator.generate_project(project_desc, output_dir)

        result_file = os.path.join(output_dir, "generation_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        print(f"\n[Result] Project saved to: {output_dir}")

    async def _self_reflection(self):
        """自我反思"""
        print(f"\n[Reflection] Analyzing performance...")
        print(f"[Reflection] Total projects: {len([m for m in self.memory if m.get('goal', {}).get('action') == 'create_project'])}")

    def _extract_json(self, text: str) -> str:
        """提取 JSON"""
        try:
            if "```json" in text:
                return text.split("```json")[1].split("```")[0].strip()
            if "```" in text:
                return text.split("```\")[1].split("```")[0].strip()
            if "{" in text and "}" in text:
                return text[text.find("{"):text.rfind("}")+1]
            return text
        except:
            return "{}"


async def run_multi_instance_comparison():
    """运行多实例对比"""
    print("\n" + "=" * 70)
    print("MULTI-INSTANCE BASE MODEL COMPARISON")
    print("=" * 70)

    # 检查可用的 API KEY
    available_models = []
    if os.getenv("DEEPSEEK_API_KEY"):
        available_models.append(BaseModel.DEEPSEEK)
    if os.getenv("ZHIPU_API_KEY"):
        available_models.append(BaseModel.ZHIPU)
    if os.getenv("KIMI_API_KEY"):
        available_models.append(BaseModel.KIMI)
    if os.getenv("QWEN_API_KEY"):
        available_models.append(BaseModel.QWEN)
    if os.getenv("GEMINI_API_KEY"):
        available_models.append(BaseModel.GEMINI)

    print(f"\n[Available] {len(available_models)} models found:")
    for model in available_models:
        print(f"  - {model.value}")

    if not available_models:
        print("\n[Error] No API keys found. Please set environment variables.")
        return

    # 创建多个实例
    instances = []
    for model in available_models:
        instance = AutonomousAGI_V6_1(model)
        instances.append(instance)

    # 并行运行
    print("\n[Start] Running all instances in parallel...")
    tasks = [inst.autonomous_loop(max_ticks=3) for inst in instances]
    await asyncio.gather(*tasks)

    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AGI Autonomous Core V6.1")
    parser.add_argument(
        "--model",
        type=str,
        choices=["deepseek", "zhipu", "kimi", "qwen", "gemini", "all"],
        default="deepseek",
        help="Base model to use"
    )

    args = parser.parse_args()

    if args.model == "all":
        asyncio.run(run_multi_instance_comparison())
    else:
        model_type = BaseModel(args.model)
        agi = AutonomousAGI_V6_1(model_type)
        asyncio.run(agi.autonomous_loop())
