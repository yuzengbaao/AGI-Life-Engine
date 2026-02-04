"""
AGI Evolution Implementation
实现了 core/evolution_specs.py 定义的四大支柱接口。
这是构建"本质智能"的实体层。
"""

import asyncio
import json
import logging
import random
import os
import shutil
import time
from typing import Dict, Any, List, Optional, Union

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, models
import numpy as np

import psutil

from core.evolution_specs import ISandboxCompiler, INeuralMemory, IValueNetwork, IWorldModel
from core.llm_client import LLMService
from core.seed import TheSeed, Experience  # Import The Seed

logger = logging.getLogger("EvolutionSystem")

# --- Q-Learning Parameters ---
ALPHA = 0.1  # Learning Rate
GAMMA = 0.9  # Discount Factor
# 🆕 [2026-01-17] P0修复：增加EPSILON以增加动作多样性
EPSILON = 0.35  # Exploration Rate (原0.2 → 0.35, 增加动作多样性)
# 🆕 [2026-01-17] P0修复：调整动作顺序，降低explore的默认优先级
ACTIONS = ["analyze", "create", "integrate", "explore", "rest"]
# 🆕 [2026-01-17] P0修复：动作冷却计数器
ACTION_COOLDOWN = {"explore": 0, "analyze": 0, "create": 0, "rest": 0, "integrate": 0}
MAX_CONSECUTIVE_SAME_ACTION = 3  # 同一动作最多连续执行3次

from core.research.lab import ResearchLab, ShadowRunner

# 🆕 [2026-01-09] 全局变量：追踪重推演时间，防止无限循环
_last_re_inference_tick = {}  # {seed_id: tick_count}
# 🔧 [2026-01-17] 优化：冷却期从5降到3，更快响应深度不足
_RE_INFERENCE_COOLDOWN = 3  # 冷却期：3个tick

class SandboxCompiler(ISandboxCompiler):
    """
    [Pillar 1] 自修改运行时实现
    """
    def __init__(self, shadow_runner: ShadowRunner = None, llm_service: LLMService = None):
        self.shadow_runner = shadow_runner
        self.llm = llm_service
        self.last_verification: Optional[Dict[str, Any]] = None
        self.optimization_history: set = set() # Track optimized files to avoid loops
        self.last_auto_optimization_time = 0
        self.AUTO_OPTIMIZATION_COOLDOWN = 300 # 5 minutes cooldown

    def _select_optimization_target(self) -> Optional[str]:
        """
        [Phase 3.3] Autonomously select a target for optimization using Attention Mechanism.
        Strategy: Weighted selection based on file complexity (size) and staleness (time).
        Global Awareness: Scans all eligible files.
        Avoid Local Optima: Uses probabilistic selection (weighted random) rather than deterministic max.
        
        [Phase 3.4 - Meta-Evolution] impl.py is now eligible for self-optimization,
        but requires mandatory shadow testing verification before any changes are applied.
        """
        try:
            candidates = []
            core_path = os.path.join(os.getcwd(), "core")
            current_time = time.time()
            
            # 1. Scan for candidates
            for root, _, files in os.walk(core_path):
                for file in files:
                    if file.endswith(".py"):
                        full_path = os.path.join(root, file)
                        
                        # Filter exclusions
                        if full_path in self.optimization_history:
                            continue
                        if file == "__init__.py" or file.endswith("_specs.py"):
                            continue
                        
                        # [PHASE 3.4 META-EVOLUTION] impl.py is now ALLOWED for self-optimization
                        # Previous constraint (removed): if file == "impl.py": continue
                        # New behavior: impl.py can be selected, but verify_in_sandbox() 
                        # will enforce mandatory shadow testing before any changes are applied.
                        # The shadow testing mechanism in ShadowRunner provides isolation.
                        if file == "impl.py":
                            # Mark as requiring extra verification (handled by verify_in_sandbox)
                            logger.info("   🧬 [Meta-Evolution] impl.py eligible for self-optimization (shadow test required)")
                            
                        # 2. Calculate Attention Metrics
                        try:
                            stats = os.stat(full_path)
                            size = stats.st_size
                            mtime = stats.st_mtime
                            age = current_time - mtime
                            
                            # Attention Score Formula:
                            # Base importance = Size (Complexity proxy)
                            # Urgency multiplier = Age (Staleness proxy)
                            # We use log of size to dampen the effect of massive files slightly, 
                            # but linear age to encourage updating old code.
                            # Ensure size > 0
                            size_score = max(1, size)
                            age_days = age / 86400.0
                            
                            # Score = log(Size) * (1 + Age_Days)
                            # This balances "Big Important Files" vs "Old Neglected Files"
                            import math
                            attention_score = math.log(size_score) * (1.0 + age_days * 0.5)
                            
                            candidates.append({
                                "path": full_path,
                                "score": attention_score,
                                "name": file
                            })
                        except Exception:
                            continue
            
            if not candidates:
                return None
                
            # 3. Weighted Selection (Probabilistic) to avoid local optima
            # Normalize scores for probability
            total_score = sum(c["score"] for c in candidates)
            if total_score == 0:
                 return random.choice([c["path"] for c in candidates])
                 
            probabilities = [c["score"] / total_score for c in candidates]
            paths = [c["path"] for c in candidates]
            
            # Select one based on distribution
            selected = random.choices(paths, weights=probabilities, k=1)[0]
            
            # Log the selection logic for transparency (Global Awareness)
            logger.info(f"   🎯 Attention Mechanism: Scanned {len(candidates)} candidates.")
            # Find the selected item's score for logging
            sel_idx = paths.index(selected)
            logger.info(f"      Selected: {candidates[sel_idx]['name']} (Score: {candidates[sel_idx]['score']:.2f}, Prob: {probabilities[sel_idx]:.2%})")
            
            return selected
            
        except Exception as e:
            logger.error(f"Error selecting optimization target: {e}")
            return None

    def _scan_project_structure(self) -> str:
        """
        [Global Awareness] Scan the project structure to provide context for the LLM.
        This prevents hallucinations like importing non-existent 'core.utils'.
        """
        structure = []
        try:
            core_path = os.path.join(os.getcwd(), "core")
            for root, _, files in os.walk(core_path):
                rel_root = os.path.relpath(root, os.getcwd())
                for file in files:
                    if file.endswith(".py"):
                        # Convert file path to module path for clarity
                        # e.g., core/system_tools.py -> core.system_tools
                        mod_path = os.path.join(rel_root, file).replace(os.sep, ".").replace(".py", "")
                        structure.append(f"- {rel_root}/{file} (Module: {mod_path})")
        except Exception as e:
            return f"Error scanning structure: {e}"
        return "\n".join(sorted(structure))

    async def analyze_bottleneck(self, system_metrics: Dict[str, Any]) -> str:
        # 模拟分析：检查是否有模块运行时间过长
        # logger.info("[Self-Modifying] Analyzing system bottlenecks...")
        # Check for command execution failures
        if system_metrics.get("last_action_success") is False:
            return "execution_failure_recovery"
            
        if system_metrics.get("avg_latency", 0) > 2.0:
            return "planner_agent"
        return "none"

    async def generate_optimized_code(self, module_name: str, requirements: str) -> str:
        import re
        logger.info(f"[Self-Modifying] Generating optimized code for {module_name}...")
        
        if not self.llm:
            logger.warning("  ⚠️ LLM Service not available for code generation.")
            return f"# Optimized code for {module_name}\n# Logic improved based on {requirements}"

        # 1. Read existing code
        current_code = ""
        try:
            # module_name might be a file path or module dot path
            if os.path.exists(module_name):
                with open(module_name, 'r', encoding='utf-8') as f:
                    current_code = f.read()
            else:
                # Try to resolve module path if needed
                pass
        except Exception as e:
            logger.warning(f"  ⚠️ Could not read module {module_name}: {e}")

        # 2. Get Global Context (The "Map")
        project_structure = self._scan_project_structure()

        base_prompt = f"""
You are an Expert Python Optimizer with GLOBAL AWARENESS of the project.
Task: Optimize the following Python module based on specific requirements.

Module Path: {module_name}
Requirements: {requirements}

--- GLOBAL PROJECT STRUCTURE (Use this to resolve imports) ---
{project_structure}
------------------------------------------------------------

Current Code:
```python
{current_code}
```

Instructions:
1. Return the COMPLETE updated code.
2. Maintain existing functionality unless explicitly asked to change it.
3. Improve performance, readability, or add features as requested.
4. **CRITICAL**: Only import modules listed in the GLOBAL PROJECT STRUCTURE above. DO NOT invent modules like 'core.utils'. Use 'core.system_tools' or others as appropriate.
5. Ensure the code is syntactically correct.
6. IMPORTANT: Return ONLY the code in a single ```python``` block.
7. DO NOT include any Markdown formatting (like **bold**) or explanations outside the code block.
"""
        max_retries = 2
        attempt = 0
        last_code = ""
        last_error = ""

        while attempt <= max_retries:
            try:
                loop = asyncio.get_running_loop()
                
                if attempt == 0:
                    current_prompt = base_prompt
                else:
                    logger.info(f"   🔄 Attempting to fix syntax error (Attempt {attempt+1})...")
                    current_prompt = f"""
You are a Python Syntax Repair Specialist.
The previous code generation attempt failed with a SyntaxError.

Error: {last_error}

Failed Code Snippet:
```python
{last_code}
```

Task: Fix the syntax error and return the COMPLETE, CORRECTED code.
Return ONLY the code in a ```python``` block.
"""

                response = await loop.run_in_executor(
                    None,
                    lambda: self.llm.chat_completion(system_prompt="You are a Senior Python Architect.", user_prompt=current_prompt)
                )
                
                # Clean code
                code = response
                # Robust extraction using regex (Improved)
                # Match ```python ... ``` or just ``` ... ```
                # re.DOTALL makes . match newlines
                match = re.search(r"```(?:python)?\s*(.*?)```", code, re.DOTALL | re.IGNORECASE)
                if match:
                    code = match.group(1).strip()
                elif "```" in code:
                    # Fallback simple split if regex fails for some reason
                    parts = code.split("```")
                    if len(parts) > 1:
                        code = parts[1]
                        if code.strip().lower().startswith("python"):
                            code = code.strip()[6:]
                        code = code.strip()
                
                # Remove rogue backticks just in case
                code = code.replace("`", "")
                
                # Verify syntax
                try:
                    compile(code, "<string>", "exec")
                    return code
                except SyntaxError as e:
                    last_error = str(e)
                    last_code = code # Save for next iteration prompt
                    logger.warning(f"❌ Generated code has syntax error: {e}")
                    attempt += 1
                    
            except Exception as e:
                logger.error(f"❌ Code generation failed: {e}")
                return ""

        logger.error("❌ Failed to generate valid code after retries.")
        return ""

    async def generate_test_cases(self, module_name: str, original_code: str, new_code: str) -> List[Dict]:
        """
        Generate verification test cases for the optimized code.
        """
        if not self.llm:
            return []

        # Calculate import path from file path (Robust Path Normalization)
        # 1. Normalize separators
        clean_path = module_name.replace("\\", "/")
        project_root = os.getcwd().replace("\\", "/")
        
        # 2. Make relative if absolute
        if clean_path.lower().startswith(project_root.lower()):
            clean_path = clean_path[len(project_root):].lstrip("/")
        
        # 3. Handle drive letters just in case (e.g. D:/...)
        if ":" in clean_path:
            # If we still have a colon, it implies an absolute path that didn't match project_root
            # Try to find 'core/' or similar
            if "core/" in clean_path:
                 clean_path = "core/" + clean_path.split("core/", 1)[1]
        
        # 4. Convert to dot notation
        import_path = clean_path.replace("/", ".")
        if import_path.endswith(".py"):
            import_path = import_path[:-3]

        prompt = f"""
You are a QA Engineer.
Task: Generate a Python test script to verify the integrity and improvement of a modified module.

Module File: {module_name}
Likely Import Path: {import_path}

Original Code (Snippet):
{original_code[:500]}...

New Code (Snippet):
{new_code[:500]}...

Requirements:
1. The test script must verify basic functionality (regression testing).
2. The test script must verify the IMPROVEMENT (e.g. check if a function is faster, or if a new feature exists).
3. The script should print "SUCCESS" if passed, or raise Exception/exit(1) if failed.
4. IMPORT INSTRUCTION: Import the module using `import {import_path} as target` (or similar) and use `target.function_name`.
5. IMPORTANT: Return ONLY the test script code in a single ```python``` block.
6. DO NOT include any conversational text or Markdown outside the code block.
"""
        try:
            import re
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.chat_completion(system_prompt="You are a QA Engineer.", user_prompt=prompt)
            )
            
            # Clean code
            test_script = response
            match = re.search(r"```(?:python)?\s*(.*?)```", test_script, re.DOTALL | re.IGNORECASE)
            if match:
                test_script = match.group(1).strip()
            elif "```" in test_script:
                test_script = test_script.split("```")[1]
                if test_script.strip().startswith("python"):
                    test_script = test_script.strip()[6:]
                test_script = test_script.strip()

            return [{
                "module_path": module_name,
                "test_code": test_script
            }]
        except Exception:
            return []

    def _extract_traceback(self, output: str) -> str:
        start = output.find("Traceback (most recent call last):")
        if start == -1:
            return ""
        return output[start:]

    def _extract_exit_code(self, output: str) -> Optional[int]:
        try:
            import re
            m = re.search(r"退出代码:\s*(\d+)", output)
            if m:
                return int(m.group(1))
            m = re.search(r"exit code:\s*(\d+)", output, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
        except Exception:
            return None
        return None

    def _diagnose_output(self, output: str) -> str:
        tb = self._extract_traceback(output)
        if not tb:
            return ""
        try:
            from core.system_tools import SystemTools
            tools = SystemTools(work_dir=os.getcwd())
            return tools.analyze_traceback(tb)
        except Exception:
            return ""

    async def verify_in_sandbox(self, code: str, test_cases: List[Dict]) -> bool:
        logger.info("[Self-Modifying] Verifying code in sandbox...")
        self.last_verification = None

        if not self.shadow_runner:
            logger.warning("  ⚠️ ShadowRunner not configured; skipping verification.")
            return True

        module_rel_path: Optional[str] = None
        module_to_test: Optional[str] = None
        test_scripts: List[str] = []

        for tc in (test_cases or []):
            if module_rel_path is None:
                module_rel_path = tc.get("module_path") or tc.get("target_file") or tc.get("file_path")
            if module_to_test is None:
                module_to_test = tc.get("module_to_test") or tc.get("module")

            scripts = tc.get("test_scripts")
            if isinstance(scripts, list):
                test_scripts.extend([s for s in scripts if isinstance(s, str) and s.strip()])
            else:
                s = tc.get("test_code") or tc.get("code") or tc.get("script")
                if isinstance(s, str) and s.strip():
                    test_scripts.append(s)

        if not module_rel_path:
            self.last_verification = {"ok": False, "stage": "params", "reason": "missing_module_path"}
            logger.error("  ❌ Missing module path in test_cases; cannot verify.")
            return False

        # Normalize paths to handle absolute paths correctly (Robust V4 - Native OS Handling)
        try:
            # 1. Ensure we have an absolute path to the target file
            # Handles both relative input ("core/utils.py") and absolute input ("D:/.../core/utils.py")
            if not os.path.isabs(module_rel_path):
                 target_abs_path = os.path.abspath(module_rel_path)
            else:
                 target_abs_path = os.path.normpath(module_rel_path)

            # 2. Compute relative path from Project Root (CWD)
            # This is critical for ShadowRunner to place it correctly inside the sandbox without breaking out.
            project_root = os.getcwd()
            try:
                clean_rel_path = os.path.relpath(target_abs_path, start=project_root)
            except ValueError:
                # Happens on Windows if drives are different (e.g., C: vs D:)
                logger.error(f"  ❌ Cannot optimize file on different drive: {target_abs_path}")
                return False

            if clean_rel_path.startswith(".."):
                logger.error(f"  ❌ Cannot optimize file outside project root: {clean_rel_path}")
                return False

            # 3. Prepare Module Name for Import
            # Convert native path separators to dots (e.g. "core\utils.py" -> "core.utils")
            module_name_candidate = os.path.splitext(clean_rel_path)[0]
            module_to_test = module_name_candidate.replace(os.sep, ".")
            
            # Use the clean relative path for ShadowRunner keys
            module_rel_path = clean_rel_path
            
            logger.info(f"  🔍 Path Normalized: File='{module_rel_path}' -> Module='{module_to_test}'")

        except Exception as e:
            logger.error(f"  ❌ Path normalization failed: {e}")
            return False


        shadow_path: Optional[str] = None
        try:
            logger.info(f"  🌑 Shadow env target: {module_rel_path} ({module_to_test})")
            # Enable full_context to ensure relative imports (like .video) work correctly
            shadow_path = self.shadow_runner.create_shadow_env({module_rel_path: code}, full_context=True)

            ok, dry_output = self.shadow_runner.dry_run(shadow_path, module_to_test)
            if not ok:
                diagnosis = self._diagnose_output(dry_output)
                self.last_verification = {
                    "ok": False,
                    "stage": "dry_run",
                    "module": module_to_test,
                    "output": dry_output,
                    "diagnosis": diagnosis,
                }
                logger.error(f"  ❌ Dry Run failed: {module_to_test}")
                if diagnosis:
                    logger.error(f"  👁️ Diagnosis: {diagnosis}")
                return False

            if not test_scripts:
                self.last_verification = {"ok": True, "stage": "dry_run_only", "module": module_to_test}
                logger.info("  ✅ Verification passed (Dry Run only).")
                return True

            env = self.shadow_runner._get_shadow_env_vars(shadow_path)
            for idx, test_code in enumerate(test_scripts, start=1):
                test_name = f"test_shadow_{idx}.py"
                test_path = os.path.join(shadow_path, test_name)
                with open(test_path, "w", encoding="utf-8") as f:
                    f.write(test_code)

                output = self.shadow_runner.execute_script(test_name, cwd=shadow_path, env=env)
                exit_code = self._extract_exit_code(output)
                if exit_code is None:
                    exit_code = 1

                if exit_code != 0:
                    diagnosis = self._diagnose_output(output)
                    self.last_verification = {
                        "ok": False,
                        "stage": "tests",
                        "module": module_to_test,
                        "test": test_name,
                        "output": output,
                        "diagnosis": diagnosis,
                    }
                    logger.error(f"  ❌ Test failed: {test_name}")
                    if diagnosis:
                        logger.error(f"  👁️ Diagnosis: {diagnosis}")
                    return False

            self.last_verification = {"ok": True, "stage": "tests", "module": module_to_test}
            logger.info("  ✅ Verification passed (Dry Run + Tests).")
            return True
        finally:
            if shadow_path:
                self.shadow_runner.cleanup(shadow_path)

    async def hot_swap_module(self, module_path: str, new_code: str) -> bool:
        logger.warning(f"[Self-Modifying] HOT SWAPPING module: {module_path}")
        if not module_path:
            return False

        abs_path = module_path
        if not os.path.isabs(abs_path):
            abs_path = os.path.abspath(os.path.join(os.getcwd(), abs_path))
        else:
            abs_path = os.path.abspath(abs_path)

        project_root = os.path.abspath(os.getcwd())
        if not abs_path.startswith(project_root):
            logger.error(f"[Self-Modifying] HOT SWAP denied (outside project): {abs_path}")
            return False

        backup_path = f"{abs_path}.bak_{int(time.time())}"
        try:
            if os.path.exists(abs_path):
                shutil.copy2(abs_path, backup_path)

            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_code)
            return True
        except Exception as e:
            logger.error(f"[Self-Modifying] HOT SWAP failed: {e}")
            try:
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, abs_path)
            except Exception:
                pass
            return False

    async def verify_and_hot_swap(self, module_path: str, new_code: str, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Verify and apply changes to a module.
        
        [Phase 3.4 - Meta-Evolution Safety]
        For impl.py (self-modification), additional safety checks are enforced:
        1. ShadowRunner MUST be available (no skip allowed)
        2. Both dry_run AND test scripts are required
        3. Backup is created before any changes
        """
        # [META-EVOLUTION SAFETY] Check if this is self-modification
        is_self_modification = False
        if module_path:
            normalized_path = module_path.replace("\\", "/")
            if "impl.py" in normalized_path or normalized_path.endswith("impl.py"):
                is_self_modification = True
                logger.warning("  ⚠️ [META-EVOLUTION] Detected self-modification of impl.py")
                
                # Enforce stricter requirements for self-modification
                if not self.shadow_runner:
                    logger.error("  ❌ [META-EVOLUTION] ShadowRunner is REQUIRED for impl.py modification - DENIED")
                    return {"ok": False, "stage": "meta_evolution_check", "reason": "shadow_runner_required_for_impl"}
                
                if not test_cases:
                    logger.error("  ❌ [META-EVOLUTION] Test cases are REQUIRED for impl.py modification - DENIED")
                    return {"ok": False, "stage": "meta_evolution_check", "reason": "test_cases_required_for_impl"}
                
                logger.info("  🧬 [META-EVOLUTION] Safety checks passed, proceeding with shadow verification...")
        
        verified = await self.verify_in_sandbox(new_code, test_cases)
        if not verified:
            return {"ok": False, "stage": "verify", "verification": self.last_verification}

        swapped = await self.hot_swap_module(module_path, new_code)
        if not swapped:
            return {"ok": False, "stage": "hot_swap", "verification": self.last_verification}

        # Log meta-evolution success
        if is_self_modification:
            logger.info("  🎉 [META-EVOLUTION] impl.py successfully self-optimized via shadow testing!")

        return {"ok": True, "stage": "hot_swap", "verification": self.last_verification}


from chromadb.utils import embedding_functions
import numpy as np

class MockEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, input: List[str]) -> List[List[float]]:
        # Return random embeddings of dimension 384 (standard for MiniLM)
        return np.random.rand(len(input), 384).tolist()

class LocalSentenceTransformerEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model):
        self.model = model
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.model.encode(input).tolist()

from core.evolution.dynamics import EvolutionaryDynamics
from core.research.lab import ResearchLab
from core.memory.topology_memory import TopologicalMemoryCore

class NeuralMemory(INeuralMemory):
    """
    [Pillar 2] 在线神经可塑性实现 (ChromaDB + SentenceTransformer)
    """
    def __init__(self, persistence_path: str = "./memory_db", llm_service: LLMService = None):
        self.persistence_path = persistence_path
        self.llm = llm_service
        os.makedirs(persistence_path, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=persistence_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 尝试加载本地模型，如果不存在则回退到在线加载或 Mock
        model_path = os.path.join(os.getcwd(), "models", "all-MiniLM-L6-v2")
        
        try:
            if os.path.exists(model_path) and os.listdir(model_path):
                logger.info(f"📂 Loading embedding model from local path: {model_path}")
                # 尝试手动组装模型以应对缺失子文件夹的情况
                try:
                    word_embedding_model = models.Transformer(model_path)
                    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
                    st_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                except Exception as inner_e:
                    logger.warning(f"⚠️ Manual assembly failed ({inner_e}), trying standard load...")
                    st_model = SentenceTransformer(model_path)
                
                self.embedder = LocalSentenceTransformerEmbeddingFunction(st_model)
                self.use_mock = False
            else:
                logger.info("🌐 Attempting to download embedding model from HuggingFace...")
                st_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedder = LocalSentenceTransformerEmbeddingFunction(st_model)
                self.use_mock = False
        except Exception as e:
            logger.warning(f"⚠️ Failed to load SentenceTransformer: {e}")
            logger.warning("⚠️ Switching to Mock Embeddings (Random) for offline mode.")
            self.embedder = MockEmbeddingFunction()
            self.use_mock = True

        self.collection = self.chroma_client.get_or_create_collection(
            name="episodic_memory",
            embedding_function=self.embedder
        )
        
        self.semantic_collection = self.chroma_client.get_or_create_collection(
            name="semantic_memory",
            embedding_function=self.embedder
        )
        
        # [Fractal Memory Core]
        # The "Mycelium Seed" that grows topologically
        self.topological_core = TopologicalMemoryCore(max_degree=32, diffusion_gain=0.9)
        
        self.short_term_buffer = []
        logger.info(f"🧠 Neural Memory initialized at {persistence_path}")

    def trigger_fractal_expansion(self, node_idx: int, metadata: Dict[str, Any]) -> str:
        """
        Trigger a fractal split at the given node index.
        Returns a description of the event.
        """
        try:
            subgraph = self.topological_core.fractal_expand(node_idx, init_params=None)
            logger.info(f"✨ FRACTAL EXPANSION: Created subgraph at Node {node_idx}. Subgraph ID: {id(subgraph)}")
            return f"Expanded Node {node_idx} into Subgraph"
        except Exception as e:
            logger.error(f"❌ Fractal expansion failed: {e}")
            return "Expansion Failed"

    async def forget_and_consolidate(self):
        """
        Executes the 'Sleep Cycle':
        1. Forgetting: Prune weak memories based on vitality.
        2. Abstraction: Cluster dense memory regions into principles.
        """
        logger.info("[NeuralMemory] Starting Sleep Cycle (Forgetting & Abstraction)...")
        
        # 1. Fetch all memories (simulated scan, in real Chroma we'd use get())
        # Since Chroma get() without IDs is limited, we iterate via a known list or metadata query
        # Here we simulate the logic for the concept.
        
        try:
            # Get a batch of recent memories to check for abstraction
            result = self.collection.get(limit=100, include=["metadatas", "documents", "embeddings"])
            ids = result['ids']
            metadatas = result['metadatas']
            documents = result['documents']
            
            if not ids:
                return

            current_time = time.time()
            ids_to_delete = []
            clusters = {} # naive clustering by first matching ID
            
            for i, meta in enumerate(metadatas):
                # A. Vitality Calculation (Forgetting)
                creation_time = meta.get("timestamp", current_time)
                age_hours = (current_time - creation_time) / 3600.0
                access_count = meta.get("access_count", 0)
                initial_importance = meta.get("score", 0.5) # Use score as importance
                
                vitality = EvolutionaryDynamics.calculate_memory_vitality(
                    initial_importance, access_count, age_hours
                )
                
                # Pruning Threshold (e.g., 0.2)
                if vitality < 0.2:
                    ids_to_delete.append(ids[i])
                    continue
                    
                # B. Abstraction Density Check
                # Simple clustering: check distance to other embeddings in this batch
                # (Real implementation would use DB query)
                # Here we just mark high-vitality items as candidates
                if vitality > 0.8:
                    clusters[ids[i]] = documents[i]

            # Execute Forgetting
            if ids_to_delete:
                logger.info(f"🗑️ Pruning {len(ids_to_delete)} weak memories (Forgetting).")
                self.collection.delete(ids=ids_to_delete)
                
            # Execute Abstraction (Simulated)
            # If we found high-value candidates, we try to "crystallize" them
            if len(clusters) > 3 and self.llm:
                logger.info(f"💎 Found {len(clusters)} high-value memories. Attempting Abstraction...")
                
                # Combine texts for LLM
                context_text = "\n".join(list(clusters.values())[:5]) # Limit to 5 for prompt
                
                sys_prompt = "你是AGI的抽象思维模块。你的任务是将具体的记忆压缩为抽象的法则。"
                user_prompt = f"请阅读以下具体记忆，并提炼出一条普适的法则：\n{context_text}"
                
                # Async call simulation
                loop = asyncio.get_running_loop()
                principle = await loop.run_in_executor(
                    None, 
                    lambda: self.llm.chat_completion(sys_prompt, user_prompt)
                )
                
                # Store Principle
                self.semantic_collection.add(
                    documents=[principle],
                    metadatas=[{"type": "crystallized_principle", "timestamp": time.time()}],
                    ids=[f"abs_{time.time()}"]
                )
                logger.info(f"✨ Crystallized Principle: {principle[:50]}...")
                
        except Exception as e:
            logger.error(f"❌ Sleep cycle failed: {e}")

    async def add_short_term(self, context: Dict[str, Any]) -> List[float]:
        """Add short-term memory and return its embedding for The Seed"""
        text = f"Context: {str(context)}"
        # In a real system, this would call an embedding model
        # Here we simulate an embedding for The Seed
        simulated_embedding = np.random.randn(64).tolist()
        
        self.short_term_buffer.append(context)
        
        # Also store in Chroma if needed
        # self.chroma_collection.add(...)
        
        return simulated_embedding

    async def capture_episodic_memory(self, context: Dict[str, Any]):
        """Capture memory and immediately store in vector DB (simulating rapid consolidation)"""
        try:
            # Convert context dict to string representation for embedding
            text_content = f"Goal: {context.get('goal')} | Action: {context.get('action')} | Result: {context.get('result')}"
            
            # Clean metadata to remove None values
            metadata = {
                "step": context.get("step") if context.get("step") is not None else 0,
                "score": float(context.get("score")) if context.get("score") is not None else 0.0
            }
            
            self.collection.add(
                documents=[text_content],
                metadatas=[metadata],
                ids=[f"mem_{metadata['step']}_{time.time()}"]
            )
            self.short_term_buffer.append(context)
        except Exception as e:
            logger.error(f"❌ Failed to capture memory: {e}")

    async def consolidate_memory_nocturnal(self):
        """
        Consolidate short-term episodic memories into long-term semantic memory.
        Uses LLM to find patterns between RECENT events and RELEVANT PAST memories.
        """
        if not self.short_term_buffer:
            return

        logger.info(f"[NeuralPlasticity] Consolidating {len(self.short_term_buffer)} memories...")
        
        # 1. Prepare Recent Memory Text
        recent_memory_text = "\n".join([
            f"- Goal: {m.get('goal')}, Action: {m.get('action')}, Result: {m.get('result')}" 
            for m in self.short_term_buffer
        ])
        
        # 2. Retrieve Relevant Historical Context (Associative Memory)
        # We query the semantic memory using the recent context to find related past insights.
        try:
            query_result = self.semantic_collection.query(
                query_texts=[recent_memory_text],
                n_results=3 # Get top 3 related past insights
            )
            related_history = query_result['documents'][0] if query_result['documents'] else []
            history_text = "\n".join([f"- Past Insight: {doc}" for doc in related_history])
        except Exception as e:
            logger.warning(f"⚠️ Failed to retrieve history: {e}")
            history_text = "No relevant history found."
        
        if self.llm:
            try:
                # 3. Generate Grounded Insight via LLM
                system_prompt = "你是AGI的'海马体'（记忆整合中枢）。你的任务是基于【最近的经历】与【相关的历史记忆】，寻找深层关联，并提炼普适规则。"
                user_prompt = f"""
                【最近的经历】(Short-term):
                {recent_memory_text}
                
                【相关的历史记忆】(Long-term Association):
                {history_text}
                
                任务：
                1. 对比：最近的经历是否验证、反驳或扩展了历史记忆？
                2. 归纳：基于这两者的联系，提炼一条更具普适性的规则。
                
                原则：
                - 严禁凭空捏造。
                - 必须建立在"历史"与"现在"的关联之上。
                - 如果没有明显关联，仅总结当前事实。
                
                输出格式 JSON:
                {{
                    "association_analysis": "历史记忆提到...而这次...",
                    "refined_principle": "提炼出的普适规则"
                }}
                """
                
                # Use executor to run synchronous LLM call in async context
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.llm.chat_completion(system_prompt, user_prompt)
                )
                
                # Parse JSON (Simple robust parsing)
                try:
                    if "```json" in response:
                        response = response.split("```json")[1].split("```")[0].strip()
                    elif "{" in response:
                        response = "{" + response.split("{", 1)[1].split("}", 1)[0] + "}"
                    
                    data = json.loads(response)
                    analysis = data.get("association_analysis", "")
                    principle = data.get("refined_principle", response)
                except:
                    analysis = "Parsing failed"
                    principle = response
                
                # Calculate Intelligence/Compression Metrics
                raw_size = len(recent_memory_text.encode('utf-8'))
                compressed_size = len(principle.encode('utf-8'))
                compression_ratio = raw_size / max(compressed_size, 1)
                
                # 4. Store in Semantic Memory with Compression Metadata
                self.semantic_collection.add(
                    documents=[f"Principle: {principle}\nBasis: {analysis}"],
                    metadatas=[{
                        "source": "associative_consolidation", 
                        "timestamp": time.time(), 
                        "type": "pattern_matching",
                        "compression_ratio": compression_ratio,
                        "raw_size": raw_size,
                        "compressed_size": compressed_size
                    }],
                    ids=[f"sem_{time.time()}"]
                )
                logger.info(f"✨ Associated Principle: {principle[:50]}... | 🔗 Basis: {analysis[:30]}...")
                logger.info(f"📉 Intelligence Compression: Ratio={compression_ratio:.2f}x (Raw: {raw_size}b -> Compressed: {compressed_size}b)")
                
            except Exception as e:
                logger.error(f"❌ Consolidation failed: {e}")
        
        # 5. Clear Buffer
        self.short_term_buffer.clear()

    async def retrieve_intuition(self, stimulus: str) -> float:
        """
        Retrieve 'intuition' by checking if we have seen similar situations before.
        Returns a confidence score (0.0 - 1.0) based on distance to nearest neighbor.
        """
        try:
            results = self.collection.query(
                query_texts=[stimulus],
                n_results=1
            )
            
            if not results['distances'][0]:
                return 0.0
                
            # Chroma returns distance. Convert to similarity/confidence.
            # Cosine distance: 0 is identical, 1 is orthogonal.
            # We want 1.0 for identical.
            distance = results['distances'][0][0]
            confidence = max(0.0, 1.0 - distance)
            return confidence
            
        except Exception as e:
            logger.error(f"❌ Intuition retrieval failed: {e}")
            return 0.0

    async def update_synaptic_weights(self, loss_metric: float):
        # logger.info(f"[NeuralPlasticity] Online weight update. Loss: {loss_metric}")
        pass


class ValueNetwork(IValueNetwork):
    """
    [Pillar 3] 内在价值函数实现 (Q-Learning)
    """
    def __init__(self, q_table_path="data/q_table.json"):
        self.q_table_path = q_table_path
        self.q_table = {} # Format: {"state_hash": {"action": q_value}}
        self.last_state = None
        self.last_action = None
        self.action_history = [] # For boredom/repetition detection
        self.state_history = {} # [Topological Awareness] Track state visitation frequency
        self._load_q_table()
        logger.info("❤️ Value Network (Q-Learning) initialized.")

    def _load_q_table(self):
        if os.path.exists(self.q_table_path):
            try:
                with open(self.q_table_path, 'r') as f:
                    self.q_table = json.load(f)
            except:
                self.q_table = {}

    def _save_q_table(self):
        os.makedirs(os.path.dirname(self.q_table_path), exist_ok=True)
        with open(self.q_table_path, 'w') as f:
            json.dump(self.q_table, f)

    def _get_state_hash(self, knowledge_state: Dict) -> str:
        """Simplify state to a hashable string. E.g., 'high_entropy' or 'low_health'"""
        # Simple heuristic state
        score = knowledge_state.get("score", 0.0)
        entropy = knowledge_state.get("entropy", 0.5)
        
        state_parts = []
        if score > 0.8: state_parts.append("success")
        elif score < 0.2: state_parts.append("fail")
        else: state_parts.append("neutral")
        
        if entropy > 0.7: state_parts.append("high_entropy")
        elif entropy < 0.3: state_parts.append("low_entropy")
        else: state_parts.append("med_entropy")
        
        return "_".join(state_parts)

    def evaluate(self, context: Dict[str, Any]) -> Any:
        """
        Comprehensive value assessment of the current context.
        Returns an object with .score attribute.
        """
        # Create a simple object to hold the score and other metrics
        class ValueAssessment:
            def __init__(self, score, entropy, survival_drive):
                self.score = score
                self.entropy = entropy
                self.survival_drive = survival_drive
            
            def __repr__(self):
                return f"ValueAssessment(score={self.score:.2f}, entropy={self.entropy:.2f})"

        # Extract score from context or calculate it
        score = float(context.get("score", 0.0))
        
        # Fix: Check top-level intuition_confidence first (passed by EvolutionController)
        ic = context.get("intuition_confidence")
        if ic is None:
            ic = context.get("seed_guidance", {}).get("intuition_confidence", 0.5)

        # [Topological Awareness] Dynamic Habituation (Boredom)
        # 1. Calculate State Hash
        temp_state = {"score": score, "entropy": 1.0 - ic} # Temporary state for hashing
        state_hash = self._get_state_hash(temp_state)
        
        # 2. Update Visitation History
        self.state_history[state_hash] = self.state_history.get(state_hash, 0) + 1
        visit_count = self.state_history[state_hash]
        
        # 3. Calculate Dynamic Penalty (Logarithmic Decay)
        # As we visit the same state more, the "novelty" wears off.
        # 🔧 [2026-01-09] 限制习惯化惩罚上限,防止永久低置信度
        import math
        habituation_penalty = min(0.5, math.log1p(visit_count) * 0.1)  # 从0.8改为0.5
        
        # 🆕 [2026-01-09] 访问次数过多后停止惩罚增长
        if visit_count > 50:  # 访问50次后惩罚固定在0.5
            habituation_penalty = 0.5
        
        # 4. Apply Penalty to Confidence (Simulating Boredom)
        # Higher penalty -> Lower Confidence -> Higher Entropy -> Curiosity Trigger
        ic = max(0.0, ic - habituation_penalty)

        pseudo_state = {
            "intuition_confidence": ic,
            "score": score
        }
        entropy = self.calculate_entropy(pseudo_state)
        survival = self.get_survival_drive()
        
        return ValueAssessment(score, entropy, survival)

    def calculate_entropy(self, knowledge_state: Dict) -> float:
        """
        🆕 [2026-01-24] 多维度熵计算 (修复版)
        
        修复问题：原公式 `1.0 - confidence` 过于简化，导致：
        - 休息状态被误判为低熵
        - 系统陷入僵化循环
        - 创见生成停止
        
        新公式考虑：
        1. 基础熵 = 1 - 置信度
        2. 状态重复惩罚（访问次数过多 → 增加熵）
        3. 低分惩罚（执行效果差 → 增加熵）
        4. 动作循环惩罚（重复同一动作 → 增加熵）
        """
        confidence = knowledge_state.get("intuition_confidence", 0.5)
        score = knowledge_state.get("score", 0.5)
        
        # 1. 基础熵 = 1 - 置信度
        base_entropy = 1.0 - confidence
        
        # 2. 状态重复惩罚：如果状态访问次数过多，增加熵值（打破僵化）
        state_hash = self._get_state_hash(knowledge_state)
        visit_count = self.state_history.get(state_hash, 0)
        repetition_penalty = min(0.3, visit_count * 0.02)  # 最多增加0.3
        
        # 3. 低分惩罚：如果执行效果差，增加熵值（触发探索）
        low_score_penalty = max(0, (0.5 - score) * 0.4)  # 分数<0.5时增加熵
        
        # 4. 动作循环检测：最近5次同一动作则增加熵
        action_loop_penalty = 0
        if hasattr(self, 'action_history') and len(self.action_history) >= 5:
            recent = list(self.action_history)[-5:]
            if len(set(recent)) == 1:  # 最近5次同一动作
                action_loop_penalty = 0.3
                logger.debug(f"[ValueNetwork] 检测到动作循环: {recent[0]}×5, 增加熵惩罚0.3")
        
        # 综合熵值
        entropy = base_entropy + repetition_penalty + low_score_penalty + action_loop_penalty
        
        # 限制在 [0, 1] 范围
        final_entropy = min(1.0, max(0.0, entropy))
        
        # 调试日志（仅当熵值显著偏离基础值时）
        if abs(final_entropy - base_entropy) > 0.1:
            logger.debug(
                f"[ValueNetwork] 熵值校正: base={base_entropy:.2f} + rep={repetition_penalty:.2f} "
                f"+ low_score={low_score_penalty:.2f} + loop={action_loop_penalty:.2f} = {final_entropy:.2f}"
            )
        
        return final_entropy

    def evaluate_curiosity_reward(self, new_information: Dict) -> float:
        return len(str(new_information)) * 0.01

    def reset_entropy_state(self):
        """
        🆕 [2026-01-17] P0紧急修复：重置熵值状态

        功能：
        1. 清空状态访问历史（消除习惯化惩罚）
        2. 重置动作历史（打破动作循环）
        3. 降低熵值到基线（通过重置内部状态）

        用途：
        - 当系统陷入病态高熵锁定时调用
        - 由Entropy Regulator在强制重置时触发
        - 作为最后的降熵手段
        """
        logger.warning("[ValueNetwork] 🔄 执行熵值状态重置（P0修复）")

        # 1. 清空状态访问历史（消除习惯化惩罚）
        old_state_count = len(self.state_history)
        self.state_history.clear()
        logger.info(f"   - 清空状态历史: {old_state_count}个状态 → 0个")

        # 2. 重置动作历史（打破动作循环）
        old_action_count = len(self.action_history)
        self.action_history.clear()
        logger.info(f"   - 清空动作历史: {old_action_count}个动作 → 0个")

        # 3. 重置Q-table中的熵值相关状态（可选，保留学习成果）
        # 不清空整个Q-table，只重置与熵值相关的部分
        entropy_states = [k for k in self.q_table.keys() if 'entropy' in k]
        for state in entropy_states:
            del self.q_table[state]
        logger.info(f"   - 清除熵值状态: {len(entropy_states)}个Q-table条目")

        logger.info("[ValueNetwork] ✅ 熵值状态重置完成，系统应该返回到低熵状态")

        return {
            "states_cleared": old_state_count,
            "actions_cleared": old_action_count,
            "q_entries_cleared": len(entropy_states)
        }

    def get_survival_drive(self, system_health: float = None) -> float:
        """
        Calculate survival drive based on system health.
        If system_health is not provided, it is calculated from system resources (CPU/RAM).
        High Drive = Low Health (Need to recover).
        """
        if system_health is None:
            try:
                # Get CPU and Memory usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                # Normalize to 0.0 - 1.0 (1.0 is max load)
                load_factor = (cpu_percent + memory_percent) / 200.0
                
                # Health is inverse of load
                system_health = max(0.0, 1.0 - load_factor)
            except Exception:
                system_health = 0.8 # Default healthy
                
        return 1.0 - system_health

    def select_action_based_on_value(self, possible_actions: List[str] = None, current_state: Dict = None) -> str:
        """
        🆕 [2026-01-17] P0修复：增强动作选择逻辑，防止动作循环

        改进：
        1. 检测连续重复动作并惩罚
        2. 强制动作多样性
        3. 增加随机探索概率
        """
        if not possible_actions:
            possible_actions = ACTIONS
        
        if current_state is None:
            current_state = {"score": 0.5, "entropy": 0.5}

        # Get current state hash
        state_hash = self._get_state_hash(current_state)
        
        # 🆕 P0修复：检测动作循环
        if len(self.action_history) >= MAX_CONSECUTIVE_SAME_ACTION:
            recent_actions = self.action_history[-MAX_CONSECUTIVE_SAME_ACTION:]
            if len(set(recent_actions)) == 1:  # 全部相同
                repeated_action = recent_actions[0]
                # 从候选动作中移除重复动作
                filtered_actions = [a for a in possible_actions if a != repeated_action]
                if filtered_actions:
                    possible_actions = filtered_actions
                    logger.info(f"[ValueNetwork] 🔄 动作多样性: 排除重复动作 '{repeated_action}'")

        # Epsilon-greedy selection (增加到35%)
        if random.random() < EPSILON:
            selected = random.choice(possible_actions)
            self.action_history.append(selected)
            # 保持历史长度合理
            if len(self.action_history) > 20:
                self.action_history = self.action_history[-20:]
            return selected
        
        # Exploitation
        if state_hash in self.q_table:
            state_actions = self.q_table[state_hash]
            # Filter by possible actions
            valid_actions = {k: v for k, v in state_actions.items() if k in possible_actions}
            if valid_actions:
                selected = max(valid_actions, key=valid_actions.get)
                self.action_history.append(selected)
                if len(self.action_history) > 20:
                    self.action_history = self.action_history[-20:]
                return selected
        
        # Fallback: 随机选择，但偏向非explore动作
        weights = [1.0 if a != 'explore' else 0.5 for a in possible_actions]
        selected = random.choices(possible_actions, weights=weights, k=1)[0]
        self.action_history.append(selected)
        if len(self.action_history) > 20:
            self.action_history = self.action_history[-20:]
        return selected

class SelfHealingExecutor:
    """
    [Phase 2] 自愈执行器
    Wraps SystemTools with LLM-based error correction.
    """
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.tools = None # Lazy load
        
    def execute(self, command: str) -> str:
        from core.system_tools import SystemTools
        if not self.tools:
            self.tools = SystemTools()
            
        return self.tools.run_command_with_retry(command, self.llm)

class WorldModel(IWorldModel):
    """
    [Pillar 4] 预测性世界模型实现 (LLM-based Simulation)
    """
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        
    async def build_causal_graph(self, observation_history: List[Dict]):
        # Placeholder for causal graph building
        pass

    async def simulate_outcome(self, action: str, current_state: Dict) -> Dict:
        """
        Use LLM to simulate the outcome of an action in the current state.
        """
        if not self.llm:
            return {"outcome": "unknown", "probability": 0.5}
            
        prompt = f"""
        You are a World Model Simulator.
        Current State: {json.dumps(current_state, default=str)[:500]}
        Action: {action}
        
        Predict the likely outcome of this action.
        Return a JSON:
        {{
            "predicted_state": "description of state",
            "success_probability": 0.8,
            "potential_risks": ["risk1", "risk2"]
        }}
        """
        try:
            response = self.llm.chat_completion(system_prompt="World Simulator", user_prompt=prompt)
            # Simple parsing
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                response = "{" + response.split("{", 1)[1].split("}", 1)[0] + "}"
            return json.loads(response)
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {"outcome": "simulation_failed", "error": str(e)}

    async def counterfactual_reasoning(self, past_event: str, alternative_action: str) -> str:
        if not self.llm:
            return "Simulation unavailable"
            
        prompt = f"""
        Counterfactual Reasoning Task:
        Past Event: {past_event}
        Alternative Action: {alternative_action}
        
        What would have likely happened if the Alternative Action was taken?
        """
        return self.llm.chat_completion(system_prompt="Counterfactual Reasoner", user_prompt=prompt)


class EvolutionController:
    """
    [The Coordinator] 进化控制器
    Orchestrates the Four Pillars of Evolution to guide the AGI's growth.
    
    Enhanced with MetaCognition layer for:
    - Extended thought chains (5→20 steps)
    - Self-observation at each tick
    - Pattern detection and meta-insights
    """
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        
        # Initialize Pillars
        # 1. Self-Modifying Runtime
        from core.research.lab import ShadowRunner
        self.shadow_runner = ShadowRunner(project_root=os.getcwd())
        self.compiler = SandboxCompiler(shadow_runner=self.shadow_runner, llm_service=self.llm)
        
        # 2. Online Neural Plasticity
        self.memory = NeuralMemory(llm_service=self.llm)
        
        # 3. Intrinsic Value Function
        self.value_network = ValueNetwork()
        
        # 4. Predictive World Model
        self.world_model = WorldModel(llm_service=self.llm)
        
        # 5. Neural Core (The Seed)
        # Initialize with standard dimensions used in step()
        # state_dim=64 (from hash resize), action_dim=4 (len(ACTIONS))
        self.seed = TheSeed(state_dim=64, action_dim=len(ACTIONS))
        self.last_state_vec = None
        self.last_action_idx = None
        
        # 6. MetaCognition Layer (NEW)
        # Extended thought chains and self-observation
        from core.metacognition import MetaCognition, create_metacognition
        self.metacognition = create_metacognition(
            seed_ref=self.seed,
            memory_ref=self.memory
            # 🔧 [2026-01-22] 移除 horizon 参数以使用默认值 99999
            # 实际深度由 simulate_trajectory 的早停机制控制（现已优化）
        )
        
        # Research Lab for experiments
        from core.research.lab import ResearchLab
        self.research_lab = ResearchLab()
        
        logger.info("🧬 Evolution Controller Online. All Pillars Active. MetaCognition Enabled.")

    async def step(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Main Interface] Handle a single evolution step.
        Called by AGI_Life_Engine.py
        """
        # Convert context dict to string for the underlying method
        context_str = json.dumps(context, default=str)
        
        # 1. Run The Seed (Neural Core)
        # Convert context to embedding for the Neural Network
        neural_action = "unknown"
        if hasattr(self.memory, 'embedder'):
            # Use 'context_str' to generate embedding
            # Note: We need a synchronous way or await the embedder if it's async
            # Here we assume we can get a quick vector. For now, use random or mock if embedder is complex.
            # In a real step, we should use the embedding from 'memory.add_short_term' if available.
            # But let's keep it simple: use random state for now to prove the connection, 
            # or better: use the hash of the context as a seed for the state vector.
            import hashlib
            h = hashlib.sha256(context_str.encode()).digest()
            # Create a 64-dim vector from hash
            state_vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
            state_vec = np.resize(state_vec, 64)
            
            # [Active Inference] Learn from the PAST transition
            # We are currently in State S_t (state_vec).
            # We performed Action A_{t-1} (last_action_idx) in State S_{t-1} (last_state_vec).
            # The result of that action led to S_t and Reward R_t.
            if self.last_state_vec is not None and self.last_action_idx is not None:
                # Calculate Reward R_t from context
                # Use score or intuition confidence as a proxy for "Goodness"
                reward = float(context.get("score", 0.0))
                
                # Construct Experience
                exp = Experience(
                    state=self.last_state_vec,
                    action=self.last_action_idx,
                    reward=reward,
                    next_state=state_vec
                )
                
                # Learn (Update Weights)
                self.seed.learn(exp)
                # logger.debug(f"🧬 Neural Core learned from experience. Reward: {reward}")

            # Act (Decide A_t for S_t)
            action_idx = self.seed.act(state_vec)
            
            # 🆕 [2026-01-09] TRAE建议: 初始化并记录动作历史
            if not hasattr(self, '_action_history'):
                from collections import deque
                self._action_history = deque(maxlen=20)  # 保留最近20步

            # 🔧 [2026-01-09] 提前计算uncertainty避免作用域错误
            _, uncertainty = self.seed.predict(state_vec, action_idx)
            neural_confidence_early = max(0.0, 1.0 - uncertainty)

            if len(self._action_history) >= 9:
                from collections import Counter
                recent_actions = list(self._action_history)[-9:]
                action_counts = Counter(recent_actions)
                most_common_action, max_count = action_counts.most_common(1)[0]
                if max_count >= 8:
                    if most_common_action == "explore":
                        desired_action = "analyze" if uncertainty >= 0.6 else "create"
                    elif most_common_action == "analyze":
                        desired_action = "rest"
                    elif most_common_action == "rest":
                        desired_action = "create"
                    else:
                        desired_action = "analyze"

                    if desired_action in ACTIONS:
                        new_action_idx = ACTIONS.index(desired_action)
                    else:
                        candidate_idxs = [i for i, name in enumerate(ACTIONS) if name != most_common_action]
                        new_action_idx = random.choice(candidate_idxs) if candidate_idxs else action_idx

                    if new_action_idx != action_idx:
                        logger.warning(
                            f"⚠️ 动作循环打断: '{most_common_action}'×{max_count} "
                            f"→ 强制切换为 '{ACTIONS[new_action_idx]}'"
                        )
                        action_idx = new_action_idx
                        _, uncertainty = self.seed.predict(state_vec, action_idx)
                        neural_confidence_early = max(0.0, 1.0 - uncertainty)

            self._action_history.append(ACTIONS[action_idx % len(ACTIONS)])
            
            # 🔧 [2026-01-09] 计算实时系统状态指标
            current_entropy = 1.0 - neural_confidence_early
            # 🆕 [2026-01-15] P1修复：从TheSeed获取真实的好奇心值，而不是用熵代替
            current_curiosity = self.seed.curiosity if hasattr(self.seed, 'curiosity') else current_entropy
            # ✅ [FIX 2026-01-09] 移除*1000放大因子，state_vec范围[0,1]，norm最大约8
            state_change_rate = (np.linalg.norm(state_vec - self.last_state_vec)
                                 if self.last_state_vec is not None else 0.0)

            # [Reasoning Internalization - 🔧 Enhanced with Adaptive Depth Selection]
            # Auto-select reasoning depth based on task complexity
            task_descriptor = context.get("user_message", context.get("intent", "default_task"))
            adaptive_horizon = self.metacognition.auto_select_horizon(
                task_descriptor=str(task_descriptor),
                context={
                    "state_vec": state_vec,
                    "action_idx": action_idx,
                    "subtask_count": context.get("subtask_count", 1),
                    # 🆕 [2026-01-09] 传递完整系统状态指标
                    "entropy": float(current_entropy),
                    "curiosity": float(current_curiosity),  # 🔧 P1修复：使用真实的好奇心值
                    "uncertainty": float(uncertainty),
                    "state_change_rate": float(state_change_rate),
                    "neural_confidence": float(neural_confidence_early),
                    "source": "evolution_controller"
                }
            )
            logger.debug(f"🧠 Adaptive Horizon Selected: {adaptive_horizon} (Task: {task_descriptor[:50]}...)")
            
            # Simulate trajectory with adaptive depth & extension
            trajectory = self.seed.simulate_trajectory(
                state_vec, 
                action_idx, 
                horizon=adaptive_horizon,
                adaptive=True,  # Enable auto-extension & early stopping
                max_horizon_extension=2000  # 🔧 [2026-01-17] 与元认知MAX_HORIZON一致
            )
            
            # [Truncation Detection - 🔧 Monitor reasoning depth quality]
            from reasoning_depth_controller import get_reasoning_monitor
            monitor = get_reasoning_monitor()
            truncation_result = monitor.detect_truncation(trajectory, task_descriptor=str(task_descriptor))
            
            # 🔧 [2026-01-09] Fix: Initialize recommended/current before conditional usage
            recommended = truncation_result.get('recommended_horizon', adaptive_horizon)
            current = adaptive_horizon
            
            if truncation_result['is_truncated']:
                logger.warning(
                    f"⚠️ Reasoning Truncation Detected! "
                    f"Current: {current}, Recommended: {recommended}, "
                    f"Confidence: {truncation_result['confidence']:.2f}, "
                    f"Analysis: {truncation_result['analysis']}"
                )
                
            # 🆕 [2026-01-09] 自动重推演机制 + 冷却期防护
            # 🔧 [2026-01-17] 降低阈值从1.2到1.1，更快响应深度差异
            current_tick = getattr(self, 'tick_count', 0)
            seed_id = id(self.seed)
            last_tick = _last_re_inference_tick.get(seed_id, -999)
            
            if recommended > current * 1.1 and (current_tick - last_tick) >= _RE_INFERENCE_COOLDOWN:
                logger.warning(f"🔄 启动自动重推演: {current} → {recommended} (Tick {current_tick})")
                _last_re_inference_tick[seed_id] = current_tick  # 记录本次重推演时间
                
                # 重新模拟更深的推理链
                # 🔧 [2026-01-17] 提升上限与元认知MAX_HORIZON=2000一致
                new_horizon = min(recommended, 2000)  # 上限2000与元认知一致
                trajectory = self.seed.simulate_trajectory(
                    state_vec,
                    action_idx,
                    horizon=new_horizon,
                    adaptive=True,
                    max_horizon_extension=2000  # 🔧 [2026-01-17] 统一推理深度限制
                )
                
                logger.info(f"✅ 重推演完成: 新深度 {new_horizon}, 轨迹长度 {len(trajectory)}")
                adaptive_horizon = new_horizon  # 更新实际使用的深度
            elif recommended > current * 1.1:
                logger.info(f"⏸️ 重推演冷却中 (上次: Tick {last_tick}, 当前: Tick {current_tick}, 需等待{_RE_INFERENCE_COOLDOWN}个Tick)")
            
            # Project thoughts for the trajectory
            thought_chain = []
            for t_state, t_unc, t_act in trajectory:
                thought = self.seed.project_thought(t_state)
                act_name = ACTIONS[t_act % len(ACTIONS)]
                thought_chain.append(f"({act_name}) -> {thought}")
            
            thought_chain_str = " => ".join(thought_chain)
            # logger.debug(f"🧠 Neural Thought Chain: {thought_chain_str}")

            # Fix: extract uncertainty from seed to help break entropy lock
            # We call predict internally to get uncertainty for the chosen action
            _, uncertainty = self.seed.predict(state_vec, action_idx)
            # Normalize uncertainty (heuristic): assuming std dev roughly 0-1, confidence is inverse
            neural_confidence = max(0.0, 1.0 - uncertainty)
            
            # 🆕 [2026-01-09] 熵注入机制：打破思维僵化
            # ✅ [FIX] 使用 uncertainty 作为熵的近似值（uncertainty越高 ≈ 熵越高）
            current_entropy = uncertainty
            # ✅ [OPTIMIZE 2026-01-09] 阈值0.35→0.40，更早干预局部最优
            if current_entropy < 0.40:  # 熵过低触发探索
                # 随机扰动动作选择
                if random.random() < (0.40 - current_entropy):  # 熵越低，扰动概率越高
                    original_action = action_idx
                    action_idx = random.randint(0, self.seed.action_dim - 1)
                    logger.debug(f"🎲 熵注入: 熵={current_entropy:.3f} < 0.40, 动作 {original_action} → {action_idx}")
            
            neural_action = ACTIONS[action_idx % len(ACTIONS)]
            
            # [MetaCognition] Observe this tick's cognitive state
            if hasattr(self, 'metacognition') and self.metacognition:
                self.metacognition.observe(
                    state_vector=state_vec,
                    action_taken=action_idx,
                    action_name=neural_action,
                    uncertainty=uncertainty,
                    thought_chain=thought_chain,
                    context=context,
                    neural_confidence=neural_confidence
                )
            
            # Update History
            self.last_state_vec = state_vec
            self.last_action_idx = action_idx

        # 2. Get Symbolic Guidance (Value Network)
        # Pass neural confidence to influence entropy
        guidance = await self.get_evolutionary_guidance(context_str, neural_confidence=neural_confidence)
        
        # 3. Merge Neural Intuition into Guidance
        guidance["neural_action"] = neural_action
        guidance["thought_chain"] = thought_chain_str if 'thought_chain_str' in locals() else ""
        
        # If the Neural Network feels strongly (e.g. if we had a confidence metric), it could override.
        # For now, we just expose it.
        
        # Return in the format expected by AGI_Life_Engine
        # The engine expects 'evolution_guidance' which usually contains 'seed_guidance'
        return {
            "seed_guidance": guidance,
            "timestamp": time.time()
        }

    async def absorb_knowledge(self, text_content: str):
        """
        [Knowledge Compression]
        Feed external textual knowledge into the Neural Core (TheSeed).
        This trains the World Model to recognize patterns in the text.
        """
        if not text_content.strip():
            return
            
        logger.info(f"🧠 Neural Core absorbing {len(text_content)} chars of knowledge...")
        
        # 1. Convert Text to Embedding (State Vector)
        # We use the same hashing trick as in step() for consistency if no embedder
        import hashlib
        # Chunk the text to avoid washing out details
        chunk_size = 500
        chunks = [text_content[i:i+chunk_size] for i in range(0, len(text_content), chunk_size)]
        
        for chunk in chunks:
            h = hashlib.sha256(chunk.encode()).digest()
            state_vec = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
            state_vec = np.resize(state_vec, 64)
            
            # 2. Train the Seed
            self.seed.internalize_knowledge(state_vec)

    async def consolidate(self):
        """
        [Main Interface] Trigger memory consolidation (Dreaming).
        Exposes the internal NeuralMemory consolidation process.
        """
        logger.info("🌌 EvolutionController: Triggering Manual Consolidation...")
        await self.memory.consolidate_memory_nocturnal()


    async def get_evolutionary_guidance(self, context_input: Union[Dict, str], neural_confidence: float = 0.0) -> Dict[str, Any]:
        """
        Generate high-level guidance based on intrinsic value and memory.
        Accepts either a dictionary context or a stringified JSON.
        """
        # 0. Normalize Input
        if isinstance(context_input, str):
            try:
                context_dict = json.loads(context_input)
            except:
                context_dict = {}
            context_str = context_input
        else:
            context_dict = context_input
            context_str = json.dumps(context_dict, default=str)

        # 1. Get Intuition from Memory FIRST (to inform Value Network)
        memory_confidence = await self.memory.retrieve_intuition(context_str)
        
        # [Entropy Fix] Hybrid Confidence:
        # Combine Memory Confidence (Long-term) with Neural Confidence (Short-term/Adaptive)
        # If memory is empty, Neural Confidence takes over. This prevents Entropy lock at 1.0.
        intuition_confidence = max(memory_confidence, neural_confidence)

        # 2. Parse Context to State with DYNAMIC values
        # Extract score/health from context if available, fallback to neutral
        # We look for 'score', 'reward', or 'last_action_reward'
        current_score = context_dict.get("score", 
                            context_dict.get("reward", 
                                context_dict.get("last_action_reward", 0.5)))
        
        # Create a state object that includes the retrieved intuition
        state = {
            "context_summary": context_str[:200],
            "score": float(current_score),
            "intuition_confidence": intuition_confidence # Pass this down to ValueNetwork
        }
        
        # 3. Calculate Metrics via Value Network
        value_assessment = self.value_network.evaluate(state)
        
        # 4. Determine Suggested Action
        # High entropy -> Explore
        # Low entropy + High value -> Create/Optimize
        # Low health -> Rest
        
        suggested_action = "explore"
        insight_trigger = f"Why is the state entropy {value_assessment.entropy:.2f}?"
        
        # 🆕 [2026-01-09] TRAE建议: 循环打断机制 - 检测动作重复并强制explore
        if hasattr(self, '_action_history'):
            # 检查最近10步中同一动作是否重复>7次
            if len(self._action_history) >= 10:
                from collections import Counter
                recent_actions = list(self._action_history)[-10:]
                action_counts = Counter(recent_actions)
                max_count = max(action_counts.values())
                most_common_action = action_counts.most_common(1)[0][0]
                
                if max_count > 7:
                    logger.warning(
                        f"⚠️ 动作循环检测: '{most_common_action}' 在最近10步中重复{max_count}次 - 强制explore打断"
                    )
                    if most_common_action == "explore":
                        suggested_action = "analyze"
                    elif most_common_action == "analyze":
                        suggested_action = "rest"
                    elif most_common_action == "rest":
                        suggested_action = "create"
                    else:
                        suggested_action = "explore"
                    insight_trigger = f"Action loop detected ({most_common_action}×{max_count}), forcing {suggested_action}"
        
        # [Fractal Awakening Logic]
        # 🆕 [2026-01-09] 分形扩张限流机制 - 防止节点爆炸
        import time
        if not hasattr(self, '_fractal_throttle'):
            self._fractal_throttle = {'count': 0, 'last_reset': time.time()}
        
        # 每分钟重置计数器
        if time.time() - self._fractal_throttle['last_reset'] > 60:
            self._fractal_throttle = {'count': 0, 'last_reset': time.time()}
            logger.info("🔄 FRACTAL限流计数器已重置")
        
        # If Entropy is CRITICAL (> 0.9), the system is overwhelmed.
        # Instead of just "analyzing", it must "grow" (Fractal Split) to accommodate the complexity.
        if value_assessment.entropy > 0.9:
            # 检查限流
            if self._fractal_throttle['count'] >= 5:
                logger.warning(f"⚠️ FRACTAL EXPANSION已限流 (本分钟已触发{self._fractal_throttle['count']}次)")
                suggested_action = "analyze"
                insight_trigger = f"Entropy={value_assessment.entropy:.2f} but fractal expansion throttled"
            else:
                logger.warning(f"💥 CRITICAL ENTROPY ({value_assessment.entropy:.2f}). Triggering FRACTAL EXPANSION...")
                self._fractal_throttle['count'] += 1
                
                # Deterministic mapping of context to a node index (for consistency)
                # In a real graph, this would be the 'closest' node.
                import hashlib
                h = int(hashlib.sha256(context_str.encode()).hexdigest(), 16)
                node_idx = h % 1000 # Map to a finite space for the demo
                
                # Trigger Expansion in Memory
                fractal_msg = self.memory.trigger_fractal_expansion(node_idx, {"reason": "high_entropy"})
                
                suggested_action = "analyze" # Force analysis of the new structure
                insight_trigger = f"System underwent {fractal_msg} due to Entropy Overload."
            
        elif value_assessment.survival_drive > 0.8:
            suggested_action = "rest"
        elif value_assessment.entropy < 0.3:
            suggested_action = "create" # or optimize
        elif value_assessment.entropy > 0.8:
            suggested_action = "analyze" # Need to understand chaos
            
        return {
            "suggested_action": suggested_action,
            "survival_drive": value_assessment.survival_drive,
            "intrinsic_curiosity": value_assessment.entropy, # Map entropy to curiosity demand
            "entropy": value_assessment.entropy,
            "intuition_confidence": intuition_confidence,
            "insight_trigger": insight_trigger
        }

    async def conduct_research(self, hypothesis: str) -> str:
        """
        Conduct an autonomous research experiment.
        """
        logger.info(f"🧪 Conducting Research: {hypothesis}")
        
        # Generate Code via Compiler (reuse generation logic or use LLM directly)
        prompt = f"""
        Generate a Python script to test this hypothesis: "{hypothesis}"
        The script should:
        1. Be self-contained.
        2. Print results to stdout.
        3. Be safe to run (no network, no system modification).
        """
        code = self.llm.chat_completion(system_prompt="Research Scientist", user_prompt=prompt)
        
        # Clean code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
            
        # Run in Lab
        result = self.research_lab.run_experiment(code, hypothesis_id="auto_research")
        return result

    async def dream(self):
        """
        Trigger memory consolidation and optimization cycles.
        """
        logger.info("💤 Dreaming (Evolution Cycle)...")
        
        # 1. Consolidate Memory
        await self.memory.forget_and_consolidate()
        await self.memory.consolidate_memory_nocturnal()
        
        # 2. Self-Optimization (Code)
        # Check for bottlenecks
        bottleneck = await self.compiler.analyze_bottleneck({"last_action_success": True}) # Mock metrics
        if bottleneck != "none":
            target = self.compiler._select_optimization_target()
            if target:
                logger.info(f"   🔨 Optimization Target Identified in Dream: {target}")
                # We could trigger optimization here, but let's keep dreaming light for now
                # or just log it as a TODO for the next waking cycle.
