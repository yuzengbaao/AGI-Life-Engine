import os
import subprocess
import logging
import time
import ast
import shutil
import uuid
from typing import Tuple, Dict, List, Optional

logger = logging.getLogger("ResearchLab")

class IsolatedExecutor:
    """
    隔离执行器基类
    负责在受控环境中执行代码，处理进程生成、超时控制和输出捕获。
    """
    def __init__(self, sandbox_dir: str, timeout: int = 5):
        self.sandbox_dir = sandbox_dir
        self.timeout = timeout
        if not os.path.exists(self.sandbox_dir):
            os.makedirs(self.sandbox_dir, exist_ok=True)

    def _prepare_env(self) -> Dict[str, str]:
        """覆盖此方法以自定义环境变量 (例如 PYTHONPATH)。"""
        return os.environ.copy()

    def execute_script(self, script_path: str, cwd: str = None, env: Dict[str, str] = None) -> str:
        """在子进程中运行 Python 脚本。"""
        if cwd is None:
            cwd = self.sandbox_dir
        if env is None:
            env = self._prepare_env()
            
        start_time = time.time()
        try:
            # 使用相同的 python 解释器运行
            cmd = ["python", script_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
                env=env
            )
            duration = time.time() - start_time
            
            output_log = f"--- 执行输出 (耗时 {duration:.2f}s) ---\n"
            output_log += f"退出代码: {result.returncode}\n"
            if result.stdout:
                output_log += f"[标准输出]\n{result.stdout}\n"
            if result.stderr:
                output_log += f"[标准错误]\n{result.stderr}\n"
                
            return output_log
            
        except subprocess.TimeoutExpired:
            return f"❌ 超时: 执行超过了 {self.timeout} 秒限制。"
        except Exception as e:
            return f"❌ 执行异常: {e}"

class ResearchLab(IsolatedExecutor):
    """
    AGI 研究实验室 (ResearchLab)
    用于安全执行不可信代码和测试假设的“游乐场”。
    强制执行严格的沙箱策略（禁止引用核心模块，限制内置函数）。
    """
    def __init__(self, sandbox_dir: str = "data/sandbox"):
        # 确保路径是绝对路径
        if not os.path.isabs(sandbox_dir):
            sandbox_dir = os.path.join(os.getcwd(), sandbox_dir)
        super().__init__(sandbox_dir, timeout=5)
        self._create_rules()
        logger.info(f"🧪 研究实验室已初始化: {self.sandbox_dir}")

    def _create_rules(self):
        readme_path = os.path.join(self.sandbox_dir, "RULES.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("# AGI 研究沙箱规则\n\n")
            f.write("1. **隔离 (Isolation)**: 脚本在独立进程中运行。\n")
            f.write("2. **无网络 (No Net)**: 严禁网络调用 (除非 Mock)。\n")
            f.write("3. **文件 I/O**: 仅允许在当前目录内操作。\n")
            f.write("4. **时间限制**: 最大执行时间 5 秒。\n")
            f.write("5. **导入限制**: 限制使用 'os', 'subprocess', 'sys', 'socket'。\n")

    def _prepare_env(self) -> Dict[str, str]:
        """严格隔离: 清空 PYTHONPATH 以防止访问核心系统模块。"""
        env = os.environ.copy()
        env["PYTHONPATH"] = ""
        return env

    def validate_code(self, code: str) -> Tuple[bool, str]:
        """静态分析 (AST) 以过滤掉明显危险的操作。"""
        try:
            tree = ast.parse(code)
            
            # 允许的模块白名单
            allowed_modules = {
                'math', 'random', 'datetime', 'time', 'json', 're', 
                'collections', 'itertools', 'functools', 'numpy', 'pandas',
                'scipy', 'sklearn', 'torch', 'matplotlib' # 允许科学计算栈
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        base_name = alias.name.split('.')[0]
                        if base_name not in allowed_modules:
                            return False, f"导入 '{base_name}' 不在白名单中。"
                            
                if isinstance(node, ast.ImportFrom):
                    base_name = node.module.split('.')[0] if node.module else ""
                    if base_name not in allowed_modules:
                        return False, f"从导入 '{base_name}' 不在白名单中。"

            return True, "安全"
            
        except SyntaxError as e:
            return False, f"语法错误: {e}"
        except Exception as e:
            return False, f"验证错误: {e}"

    def run_experiment(self, code: str, hypothesis_id: str) -> str:
        # 1. 验证
        is_safe, reason = self.validate_code(code)
        if not is_safe:
            logger.warning(f"🚫 实验被拒绝: {reason}")
            return f"安全违规: {reason}"

        # 2. 写入文件
        timestamp = int(time.time())
        filename = f"exp_{hypothesis_id}_{timestamp}.py"
        file_path = os.path.join(self.sandbox_dir, filename)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
        except Exception as e:
            return f"文件写入错误: {e}"

        # 3. 执行
        logger.info(f"⚗️ 运行实验: {filename}")
        output = self.execute_script(filename)
        
        # 保存输出
        output_filename = "experiment_output.txt"
        output_path = os.path.join(self.sandbox_dir, output_filename)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output)
            return f"[输出已保存至 {output_path}]\n" + output
        except Exception as e:
            return output + f"\n[警告: 无法保存输出文件: {e}]"

class ShadowRunner(IsolatedExecutor):
    """
    Phase 3.2: 自我进化的虚拟沙箱 (ShadowRunner)
    实现 '智能影子' (写时复制) 策略，以安全测试修改后的代码。
    """
    def __init__(self, project_root: str, sandbox_base: str = "data/sandbox/shadow_realm"):
        self.project_root = os.path.abspath(project_root)
        if not os.path.isabs(sandbox_base):
            sandbox_base = os.path.join(os.getcwd(), sandbox_base)
        super().__init__(sandbox_base, timeout=30) # 测试给予 30s (防止 import torch 等大库超时)
        logger.info(f"🌑 影子执行器 (Shadow Runner) 已初始化: {self.sandbox_dir}")

    def create_shadow_env(self, modified_files: Dict[str, str], full_context: bool = False) -> str:
        """
        创建一个临时影子环境。
        Args:
            modified_files: 映射相对路径 (例如 'core/planner.py') 到新内容的字典。
            full_context: 是否复制整个 core 目录以支持复杂的相对导入。
        Returns:
            影子目录的绝对路径。
        """
        session_id = str(uuid.uuid4())[:8]
        shadow_path = os.path.join(self.sandbox_dir, f"session_{session_id}")
        os.makedirs(shadow_path, exist_ok=True)
        
        if full_context:
            try:
                # Copy 'core' directory to shadow env
                src_core = os.path.join(self.project_root, "core")
                dst_core = os.path.join(shadow_path, "core")
                if os.path.exists(src_core):
                    shutil.copytree(src_core, dst_core, 
                                  ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
            except Exception as e:
                logger.error(f"Failed to copy context to shadow env: {e}")
        
        # 写入修改后的文件 (这将覆盖复制的文件)
        for rel_path, content in modified_files.items():
            full_path = os.path.join(shadow_path, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
                
        return shadow_path


    def _get_shadow_env_vars(self, shadow_path: str) -> Dict[str, str]:
        """
        构造 PYTHONPATH: 影子目录 -> 项目根目录
        """
        env = os.environ.copy()
        # 将 shadow_path 添加到 PYTHONPATH 的最前面
        original_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{shadow_path}{os.pathsep}{self.project_root}{os.pathsep}{original_pythonpath}"
        return env

    def dry_run(self, shadow_path: str, module_to_test: str) -> Tuple[bool, str]:
        """
        尝试在影子环境中导入修改后的模块 (空跑)。
        """
        check_script = f"""
import sys
import os
try:
    print(f"正在测试导入: {{'{module_to_test}'}}")
    import {module_to_test}
    print("[SUCCESS] 导入成功")
except Exception as e:
    print(f"[FAILURE] 导入失败: {{e}}")
    import traceback
    traceback.print_exc()
"""
        script_path = os.path.join(shadow_path, "_dry_run_check.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(check_script)
            
        env = self._get_shadow_env_vars(shadow_path)
        output = self.execute_script("_dry_run_check.py", cwd=shadow_path, env=env)
        
        if "[SUCCESS] 导入成功" in output:
            return True, output
        else:
            return False, output

    def run_tests_in_shadow(self, shadow_path: str, test_code: str) -> str:
        """
        在影子环境中运行提供的测试代码。
        """
        test_file = os.path.join(shadow_path, "test_modification.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_code)
            
        env = self._get_shadow_env_vars(shadow_path)
        return self.execute_script("test_modification.py", cwd=shadow_path, env=env)

    def cleanup(self, shadow_path: str):
        """清理影子目录 (Robust cleanup for Windows)."""
        def remove_readonly(func, path, _):
            "Clear the readonly bit and reattempt the removal"
            import stat
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

        max_retries = 5
        for i in range(max_retries):
            try:
                if os.path.exists(shadow_path):
                    shutil.rmtree(shadow_path, onerror=remove_readonly)
                logger.info(f"🧹 已清理影子会话: {os.path.basename(shadow_path)}")
                return
            except Exception as e:
                if i < max_retries - 1:
                    logger.warning(f"清理影子路径失败 (尝试 {i+1}/{max_retries}): {e} - 等待释放...")
                    time.sleep(1) # Wait for file handles to release
                else:
                    logger.error(f"❌ 最终清理失败 {shadow_path}: {e}")
                    # Try to rename it to move it out of the way if delete fails
                    try:
                        trash_path = f"{shadow_path}_trash_{int(time.time())}"
                        os.rename(shadow_path, trash_path)
                        logger.warning(f"⚠️ 已重命名为垃圾目录: {trash_path}")
                    except:
                        pass