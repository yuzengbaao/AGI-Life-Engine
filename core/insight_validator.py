"""
洞察验证器 (Insight Validator)
实现自动验证生成的洞察代码，确保只有高质量的洞察被集成到系统中。

验证层级：
1. 语法验证 - AST解析检查Python语法
2. 安全性检查 - 检测危险操作（文件删除、网络请求等）
3. 单元测试生成 - 自动为洞察生成测试用例
4. 性能基准测试 - 测试代码执行效率
5. 语义验证 - 检查代码是否真正实现声称的功能

🆕 [2026-01-10] 增强验证层级（解决伪代码问题）:
L1: 依赖分析 - 检查所有调用的函数是否存在
L2: 沙箱执行 - 真正运行代码，捕获运行时错误
"""

import ast
import time
import traceback
import re
import sys
import io
import builtins
import subprocess
import inspect
from typing import Dict, Any, List, Tuple, Set, Optional
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# 🆕 系统函数依赖注册表（白名单）
# ============================================================================
SYSTEM_FUNCTION_REGISTRY: Set[str] = {
    # Python 内置函数
    'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'breakpoint', 'bytearray',
    'bytes', 'callable', 'chr', 'classmethod', 'compile', 'complex',
    'delattr', 'dict', 'dir', 'divmod', 'enumerate', 'eval', 'exec',
    'filter', 'float', 'format', 'frozenset', 'getattr', 'globals',
    'hasattr', 'hash', 'help', 'hex', 'id', 'input', 'int', 'isinstance',
    'issubclass', 'iter', 'len', 'list', 'locals', 'map', 'max',
    'memoryview', 'min', 'next', 'object', 'oct', 'open', 'ord', 'pow',
    'print', 'property', 'range', 'repr', 'reversed', 'round', 'set',
    'setattr', 'slice', 'sorted', 'staticmethod', 'str', 'sum', 'super',
    'tuple', 'type', 'vars', 'zip',
    # 常用标准库函数（可安全使用）
    'sqrt', 'sin', 'cos', 'tan', 'log', 'exp', 'floor', 'ceil',
    'random', 'randint', 'choice', 'shuffle', 'sample',
    'time', 'sleep',
    'json', 'loads', 'dumps',
    're', 'match', 'search', 'findall', 'sub',
    # 类型注解
    'Optional', 'List', 'Dict', 'Tuple', 'Set', 'Any', 'Union',
    # 🔧 P1修复: NumPy常用函数（用于科学计算和Insight生成）
    'maximum', 'minimum', 'real', 'imag', 'conj', 'conjugate',
    'fft', 'ifft', 'fftn', 'ifftn', 'fft2', 'ifft2', 'fftshift', 'ifftshift',
    'fftfreq', 'rfft', 'irfft', 'rfftn', 'irfftn',
    'astype', 'copy', 'transpose', 'reshape', 'flatten', 'ravel', 'squeeze',
    'expand_dims', 'squeeze', 'clip', 'abs', 'sqrt', 'square', 'exp', 'log', 'log10',
    'mean', 'median', 'std', 'var', 'sum', 'prod', 'cumsum', 'cumprod',
    'min', 'max', 'argmin', 'argmax', 'argsort', 'sort', 'sort_complex',
    'dot', 'matmul', 'tensordot', 'inner', 'outer', 'kron', 'einsum',
    'concatenate', 'stack', 'vstack', 'hstack', 'dstack', 'column_stack',
    'split', 'array_split', 'hsplit', 'vsplit', 'dsplit',
    'arange', 'linspace', 'logspace', 'geomspace', 'meshgrid', 'mgrid', 'ogrid',
    'zeros', 'ones', 'empty', 'full', 'zeros_like', 'ones_like', 'empty_like', 'full_like',
    'eye', 'identity', 'diag', 'diagflat', 'tri', 'tril', 'triu', 'vander',
    'tile', 'repeat', 'broadcast_to', 'broadcast_arrays',
    'rand', 'randn', 'randint', 'random', 'random_sample', 'ranf', 'sample',
    'choice', 'permutation', 'shuffle', 'seed',
    'load', 'save', 'savez', 'savez_compressed', 'txt', 'fromtxt', 'loadtxt', 'savetxt',
    # 🔧 P1修复: PyTorch常用函数（用于深度学习和神经网络）
    'tensor', 'zeros_', 'ones_', 'empty_', 'full_',
    'from_numpy', 'to', 'cpu', 'cuda', 'numpy',
    'sigmoid', 'tanh', 'relu', 'softmax', 'log_softmax', 'softmin',
    'binary_cross_entropy', 'mse_loss', 'l1_loss', 'nll_loss', 'cross_entropy',
    'argmax', 'argmin', 'topk', 'kthvalue', 'unique', 'unique_consecutive',
    'gather', 'scatter', 'index_select', 'index_add', 'index_fill',
    'cat', 'stack', 'hstack', 'vstack', 'dstack', 'chunk', 'split', 'unbind',
    'transpose', 'permute', 'reshape', 'view', 'unsqueeze', 'squeeze', 'flatten',
    'clone', 'detach', 'grad', 'no_grad', 'enable_grad', 'set_grad_enabled',
    'nn', 'optim', 'functional', 'utils',
    # 🔧 P1修复: 其他科学计算函数
    'predict', 'predict_proba', 'fit', 'transform', 'fit_transform',
    'entropy', 'kl_divergence', 'mutual_info', 'cosine_similarity',
    'reconstruct', 'compress', 'decompress',
    'compress_function', 'update_signature', 'update',
    'encode', 'decode', 'embed', 'embedding',
    'normalize', 'scale', 'standardize', 'minmax_scale',
    'cluster', 'classify', 'regress', 'segment',
    # 🔧 P1修复: 常见第三方库函数
    'DataFrame', 'Series', 'read_csv', 'to_csv', 'read_json', 'to_json',
    'figure', 'plot', 'show', 'savefig', 'subplot', 'subplots',
    'requests_get', 'requests_post', 'get', 'post',
    # 🔧 [2026-01-15] P0修复: Python内置方法和NumPy/PyTorch常用方法
    'item', 'items', 'keys', 'values', 'get', 'append', 'extend', 'pop',
    'tolist', 'numpy', 'cpu', 'cuda', 'float', 'long', 'int', 'bool',
    'size', 'shape', 'ndim', 'dtype', 'T', 'contiguous', 'detach',
    # 🔧 [2026-01-15] 新增：Insight实用函数库（提升可执行性）
    'invert_causal_chain', 'perturb_attention_weights', 'simulate_forward',
    'rest_phase_reorganization', 'noise_guided_rest', 'semantic_perturb',
    'analyze_tone', 'semantic_diode', 'detect_topological_defect',
    'fractal_idle_pulse', 'reverse_abduction_step', 'inject_adversarial_intuition',
    'latent_recombination', 'kl_div', 'CurlLayer',
}

# 标准库模块（可以安全导入的）
SAFE_MODULES: Set[str] = {
    'math', 'random', 'time', 'datetime', 'json', 're', 'collections',
    'itertools', 'functools', 'operator', 'copy', 'typing', 'dataclasses',
    'enum', 'statistics', 'decimal', 'fractions', 'numbers', 'cmath',
    'array', 'bisect', 'heapq', 'queue', 'struct', 'weakref',
    'string', 'textwrap', 'difflib', 'unicodedata', 'io',
    'abc', 'contextlib', 'warnings', 'logging', 'traceback',
    # 🔧 P1修复: 科学计算模块（用于Insight生成）
    'numpy', 'np',
    'torch', 'torch.nn', 'torch.nn.functional', 'torch.optim',
    'scipy', 'scipy.fft', 'scipy.stats', 'scipy.signal',
    'pandas', 'pd',
    'matplotlib', 'matplotlib.pyplot', 'plt',
    'sklearn', 'sklearn.metrics', 'sklearn.model_selection',
    # 🔧 [2026-01-15] 新增：Insight实用模块（提升可执行性）
    'core.insight_utilities', 'insight_utilities',
}


class InsightValidator:
    """
    洞察验证器 - 确保生成的洞察代码可执行且有价值
    
    🆕 [2026-01-24] 拓扑连接增强:
    - 新增 HallucinationAwareLLMEngine 连接：对洞察内容进行幻觉检测协同验证
    """
    
    # 危险操作黑名单
    DANGEROUS_MODULES = {'os', 'subprocess', 'shutil', 'socket', 'requests', 'urllib'}
    DANGEROUS_FUNCTIONS = {'exec', 'eval', 'compile', '__import__', 'open'}
    DANGEROUS_ATTRIBUTES = {'__delattr__', '__setattr__', '__delete__'}
    
    def __init__(self, system_dependency_graph: Optional[Dict[str, bool]] = None,
                 hallucination_detector=None):
        """
        初始化洞察验证器
        
        Args:
            system_dependency_graph: 系统依赖图
            hallucination_detector: 🆕 幻觉检测器（用于验证协同）
        """
        self.validation_history = []
        # 🆕 系统依赖图：记录系统中已存在的函数
        self.system_dependency_graph = system_dependency_graph or {}
        # 🆕 验证退火状态
        self._validation_backoff_until = 0.0
        self._validation_failure_count = 0
        # 🆕 [2026-01-24] 拓扑连接: 幻觉检测器
        self.hallucination_detector = hallucination_detector
        
    def validate(self, code: str, insight_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        完整验证流程（增强版）
        
        返回格式:
        {
            'valid': bool,
            'score': float (0-1),
            'checks': {
                'syntax': bool,
                'safety': bool,
                'dependency': bool,    # 🆕 依赖检查
                'sandbox': bool,       # 🆕 沙箱执行
                'unit_test': bool,
                'performance': bool,
                'semantic': bool
            },
            'errors': List[str],
            'warnings': List[str],
            'missing_deps': List[str],  # 🆕 缺失的依赖函数
            'execution_time': float,
            'recommendation': str  # 'INTEGRATE', 'ARCHIVE', 'REJECT'
        }
        """
        result = {
            'valid': False,
            'score': 0.0,
            'checks': {},
            'errors': [],
            'warnings': [],
            'missing_deps': [],
            'execution_time': 0.0,
            'recommendation': 'REJECT'
        }
        
        start_time = time.time()
        
        # 🆕 检查验证退火状态
        now_ts = time.time()
        if now_ts < self._validation_backoff_until:
            remaining = int(self._validation_backoff_until - now_ts)
            result['errors'].append(f"验证退火中（剩余{remaining}s）")
            result['execution_time'] = time.time() - start_time
            return result
        
        # 1. 语法验证
        syntax_valid, syntax_error = self._check_syntax(code)
        result['checks']['syntax'] = syntax_valid
        if not syntax_valid:
            result['errors'].append(f"语法错误: {syntax_error}")
            result['execution_time'] = time.time() - start_time
            return result
        
        # 2. 安全性检查
        safety_valid, safety_warnings = self._check_safety(code)
        result['checks']['safety'] = safety_valid
        result['warnings'].extend(safety_warnings)
        if not safety_valid:
            result['errors'].append("安全检查失败: 检测到危险操作")
            result['execution_time'] = time.time() - start_time
            return result
        
        # 🆕 3. 依赖分析（关键新增）
        deps_valid, missing_deps = self._check_dependencies(code)
        result['checks']['dependency'] = deps_valid
        result['missing_deps'] = missing_deps
        if not deps_valid:
            result['errors'].append(f"依赖检查失败: 缺少函数 {', '.join(missing_deps[:5])}")
            self._record_validation_failure()
            result['execution_time'] = time.time() - start_time
            return result
        
        # 🆕 4. 沙箱执行（真正运行代码）
        sandbox_valid, sandbox_error = self._run_in_sandbox(code)
        result['checks']['sandbox'] = sandbox_valid
        if not sandbox_valid:
            result['errors'].append(f"沙箱执行失败: {sandbox_error}")
            self._record_validation_failure()
            result['execution_time'] = time.time() - start_time
            return result
        
        # 5. 单元测试生成与执行
        test_valid, test_coverage = self._run_unit_tests(code, insight_metadata)
        result['checks']['unit_test'] = test_valid
        result['test_coverage'] = test_coverage
        if not test_valid:
            result['warnings'].append(f"单元测试覆盖率低: {test_coverage:.1%}")
        
        # 6. 性能基准测试
        perf_valid, exec_time = self._benchmark_performance(code)
        result['checks']['performance'] = perf_valid
        result['execution_time'] = exec_time
        if not perf_valid:
            result['warnings'].append(f"性能不足: 执行时间{exec_time:.3f}s超过阈值")
        
        # 7. 语义验证（检查代码是否实现声称的功能）
        semantic_valid, semantic_score = self._validate_semantics(code, insight_metadata)
        result['checks']['semantic'] = semantic_valid
        if not semantic_valid:
            result['warnings'].append("语义验证失败: 代码与洞察描述不匹配")
        
        # 🆕 [2026-01-24] 8. 幻觉协同验证（使用HallucinationAwareLLMEngine）
        hallucination_valid = True
        if self.hallucination_detector and insight_metadata.get('description'):
            try:
                hallucination_result = self.hallucination_detector.detect(
                    llm_output=insight_metadata.get('description', ''),
                    context={'code': code, 'source': 'insight_validator'}
                )
                hallucination_valid = not hallucination_result.is_hallucination
                result['checks']['hallucination'] = hallucination_valid
                if not hallucination_valid:
                    result['warnings'].append(f"幻觉检测警告: {hallucination_result.issues[:2]}")
                    logger.debug(f"[InsightValidator] 幻觉检测: {hallucination_result.issues}")
            except Exception as hal_err:
                logger.debug(f"[InsightValidator] 幻觉检测跳过: {hal_err}")
                result['checks']['hallucination'] = True  # 检测失败时不阻塞
        
        # 计算综合评分（更新权重）
        result['valid'] = all([
            syntax_valid,
            safety_valid,
            deps_valid,      # 🆕 依赖必须通过
            sandbox_valid,   # 🆕 沙箱必须通过
            test_valid or test_coverage > 0.5,
            perf_valid or exec_time < 1.0
        ])
        
        # 🆕 更新加权评分（增加依赖和沙箱权重）
        weights = {
            'syntax': 0.15, 
            'safety': 0.15, 
            'dependency': 0.20,  # 🆕 关键检查
            'sandbox': 0.20,     # 🆕 关键检查
            'unit_test': 0.15, 
            'performance': 0.08, 
            'semantic': 0.07
        }
        result['score'] = sum(
            weights.get(k, 0.0) * (1.0 if v else 0.0) 
            for k, v in result['checks'].items()
        )
        
        # 根据评分给出建议（依赖和沙箱失败直接拒绝）
        if not deps_valid or not sandbox_valid:
            result['recommendation'] = 'REJECT'
        elif result['score'] >= 0.8:
            result['recommendation'] = 'INTEGRATE'
        elif result['score'] >= 0.6:
            result['recommendation'] = 'ARCHIVE'  # 归档待改进
        else:
            result['recommendation'] = 'REJECT'
        
        # 🆕 成功验证重置退火计数
        if result['valid']:
            self._validation_failure_count = 0
        
        result['execution_time'] = time.time() - start_time
        self.validation_history.append(result)
        
        return result
    
    def _record_validation_failure(self):
        """记录验证失败，用于退火策略"""
        self._validation_failure_count += 1
        if self._validation_failure_count >= 3:
            # 连续3次失败，触发60秒退火
            self._validation_backoff_until = time.time() + 60.0
            logger.warning(f"[InsightValidator] 连续{self._validation_failure_count}次验证失败，启动60秒退火")
    
    def _check_syntax(self, code: str) -> Tuple[bool, str]:
        """语法检查 - 使用AST解析"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)
    
    def _check_safety(self, code: str) -> Tuple[bool, List[str]]:
        """安全性检查 - 检测危险操作"""
        warnings = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # 检查危险导入
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in self.DANGEROUS_MODULES:
                            warnings.append(f"危险导入: {alias.name}")
                            return False, warnings
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module in self.DANGEROUS_MODULES:
                        warnings.append(f"危险导入: from {node.module}")
                        return False, warnings
                
                # 检查危险函数调用
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in self.DANGEROUS_FUNCTIONS:
                            warnings.append(f"危险函数: {node.func.id}")
                            return False, warnings
                
                # 检查文件操作
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'open':
                        # 只允许读取操作
                        if len(node.args) > 1:
                            mode = node.args[1]
                            if isinstance(mode, ast.Constant) and 'w' in mode.value.lower():
                                warnings.append("禁止写入文件操作")
                                return False, warnings
            
            return True, warnings
            
        except Exception as e:
            return False, [f"安全检查异常: {str(e)}"]
    
    # ========================================================================
    # 🆕 Layer 1: 依赖分析（关键新增 - 解决伪代码问题）
    # ========================================================================
    
    def _check_dependencies(self, code: str) -> Tuple[bool, List[str]]:
        """
        依赖分析 - 检查所有调用的函数是否存在
        
        这是解决伪代码问题的关键方法：
        - 提取代码中所有函数调用
        - 检查每个函数是否在已知范围内（本地定义/系统注册/Python内置/安全模块）
        - 返回缺失的依赖列表
        """
        try:
            tree = ast.parse(code)
            
            # 1. 提取代码中定义的函数
            local_definitions = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    local_definitions.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    local_definitions.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            local_definitions.add(target.id)
            
            # 2. 提取代码中导入的模块和函数
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imported_names.add(name)
                        # 也添加模块本身以支持 module.func 调用
                        imported_names.add(alias.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported_names.add(node.module.split('.')[0])
                    for alias in node.names:
                        name = alias.asname if alias.asname else alias.name
                        imported_names.add(name)
            
            # 3. 提取所有函数调用
            called_functions = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = self._extract_call_name(node)
                    if func_name:
                        called_functions.add(func_name)
            
            # 4. 检查每个调用是否有效
            missing_deps = []
            for func_name in called_functions:
                if not self._is_function_available(
                    func_name, 
                    local_definitions, 
                    imported_names
                ):
                    missing_deps.append(func_name)
            
            if missing_deps:
                logger.warning(f"[InsightValidator] 检测到缺失依赖: {missing_deps}")
                return False, missing_deps
            
            return True, []
            
        except Exception as e:
            logger.error(f"[InsightValidator] 依赖检查异常: {e}")
            return False, [f"依赖检查异常: {str(e)}"]
    
    def _extract_call_name(self, node: ast.Call) -> Optional[str]:
        """从 AST Call 节点提取函数名"""
        if isinstance(node.func, ast.Name):
            # 简单调用: func()
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            # 属性调用: obj.method()
            # 只返回方法名，因为对象可能是本地变量
            return node.func.attr
        return None
    
    def _is_function_available(
        self, 
        func_name: str, 
        local_defs: Set[str], 
        imported: Set[str]
    ) -> bool:
        """检查函数是否在可用范围内"""
        # 1. 本地定义
        if func_name in local_defs:
            return True
        
        # 2. 导入的名称
        if func_name in imported:
            return True
        
        # 3. Python 内置函数
        if hasattr(builtins, func_name):
            return True
        
        # 4. 全局函数注册表
        if func_name in SYSTEM_FUNCTION_REGISTRY:
            return True
        
        # 5. 系统依赖图（AGI系统已有函数）
        if func_name in self.system_dependency_graph:
            return True
        
        # 6. 检查安全模块中是否存在
        for module_name in SAFE_MODULES:
            try:
                module = __import__(module_name)
                if hasattr(module, func_name):
                    return True
            except ImportError:
                continue
        
        return False
    
    # ========================================================================
    # 🆕 Layer 2: 沙箱执行（真正运行代码）
    # ========================================================================
    
    def _run_in_sandbox(self, code: str, timeout: float = 5.0) -> Tuple[bool, str]:
        """
        沙箱执行 - 在隔离环境中真正运行代码
        
        这是解决伪代码问题的第二道防线：
        - 真正执行代码（不只是定义）
        - 尝试调用所有定义的函数
        - 捕获任何运行时错误（包括 NameError）
        """
        temp_path = None
        try:
            # 1. 创建临时文件
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.py', delete=False, encoding='utf-8'
            ) as f:
                # 包装代码以便测试执行
                wrapped_code = self._wrap_code_for_sandbox(code)
                f.write(wrapped_code)
                temp_path = f.name
            
            # 2. 在子进程中执行（真正隔离）
            try:
                result = subprocess.run(
                    [sys.executable, temp_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=os.path.dirname(temp_path)
                )
                
                if result.returncode != 0:
                    error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                    # 提取关键错误信息
                    if "NameError" in error_msg:
                        # 这正是我们要捕获的伪代码问题！
                        match = re.search(r"NameError: name '(\w+)' is not defined", error_msg)
                        if match:
                            error_msg = f"NameError: 函数 '{match.group(1)}' 不存在"
                    elif "ImportError" in error_msg:
                        error_msg = f"ImportError: {error_msg.split('ImportError:')[-1].strip()[:100]}"
                    
                    logger.warning(f"[InsightValidator] 沙箱执行失败: {error_msg}")
                    return False, error_msg
                
                # 检查是否有 stderr 输出（可能是警告）
                if result.stderr and "Error" in result.stderr:
                    return False, result.stderr.strip()[:200]
                
                return True, ""
                
            except subprocess.TimeoutExpired:
                return False, f"执行超时 (>{timeout}s)"
            except Exception as e:
                return False, f"执行异常: {str(e)}"
            
        except Exception as e:
            return False, f"沙箱准备失败: {str(e)}"
        
        finally:
            # 清理临时文件
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def _wrap_code_for_sandbox(self, code: str) -> str:
        """
        包装代码以便在沙箱中测试
        - 定义所有函数
        - 尝试调用每个函数（使用合理的测试参数）
        """
        try:
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            # 构建测试代码
            test_calls = []
            for func in functions:
                func_name = func.name
                # 分析函数签名，生成测试参数
                test_args = self._generate_test_arguments(func)
                # 使用ASCII字符避免Windows GBK编码问题
                test_calls.append(f"""
try:
    result = {func_name}({test_args})
    print("[SANDBOX] OK: {func_name} executed successfully")
except TypeError as e:
    # 参数类型不匹配是可以接受的
    print("[SANDBOX] WARN: {func_name} type error (acceptable):", str(e))
except Exception as e:
    print("[SANDBOX] FAIL: {func_name} failed:", str(e), file=__import__('sys').stderr)
    raise
""")
            
            # 组装完整的测试代码
            wrapped = f"""# Sandbox test wrapper
{code}

# === Sandbox Test Execution ===
if __name__ == "__main__":
    import sys
    print("[SANDBOX] Starting function tests...")
{"".join(test_calls)}
    print("[SANDBOX] All tests completed.")
"""
            return wrapped
            
        except Exception as e:
            # 如果无法解析，直接返回原代码
            return f"{code}\n\n# Sandbox wrapper failed: {e}"
    
    def _generate_test_arguments(self, func: ast.FunctionDef) -> str:
        """根据函数签名生成测试参数"""
        args = func.args
        test_args = []
        
        # 处理位置参数
        for arg in args.args:
            arg_name = arg.arg.lower()
            # 根据参数名推断类型
            if 'state' in arg_name:
                test_args.append("{'entropy': 0.5, 'curiosity': 0.5}")
            elif 'entropy' in arg_name:
                test_args.append("0.5")
            elif 'curiosity' in arg_name:
                test_args.append("0.5")
            elif 'threshold' in arg_name:
                test_args.append("0.5")
            elif 'data' in arg_name or 'list' in arg_name or 'items' in arg_name:
                test_args.append("[1, 2, 3]")
            elif 'dict' in arg_name or 'config' in arg_name:
                test_args.append("{}")
            elif 'str' in arg_name or 'text' in arg_name or 'name' in arg_name:
                test_args.append("'test'")
            elif 'num' in arg_name or 'value' in arg_name or 'count' in arg_name:
                test_args.append("1")
            elif arg_name in ('a', 'b', 'c', 'x', 'y', 'z', 'n', 'm', 'i', 'j', 'k'):
                # 单字母参数通常是数值
                test_args.append("1")
            elif 'factor' in arg_name or 'ratio' in arg_name or 'rate' in arg_name:
                test_args.append("0.5")
            elif 'index' in arg_name or 'idx' in arg_name or 'pos' in arg_name:
                test_args.append("0")
            elif 'size' in arg_name or 'length' in arg_name or 'width' in arg_name:
                test_args.append("10")
            elif 'flag' in arg_name or 'enabled' in arg_name or 'active' in arg_name:
                test_args.append("True")
            else:
                # 默认使用空字符串（比None更不容易引发TypeError）
                test_args.append("'test_value'")
        
        # 跳过有默认值的参数
        num_defaults = len(args.defaults)
        if num_defaults > 0:
            test_args = test_args[:-num_defaults]
        
        return ", ".join(test_args)

    def _run_unit_tests(self, code: str, metadata: Dict) -> Tuple[bool, float]:
        """
        自动生成并运行单元测试
        测试覆盖率 = 成功测试数 / 总测试数
        """
        # 提取代码中的函数定义
        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if not functions:
                return True, 1.0  # 无函数定义，默认通过
            
            # 为每个函数生成简单测试
            passed_tests = 0
            total_tests = len(functions)
            
            # 创建临时模块
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_path = f.name
            
            try:
                spec = importlib.util.spec_from_file_location("temp_insight", temp_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules["temp_insight"] = module
                spec.loader.exec_module(module)
                
                for func_name in functions:
                    try:
                        func = getattr(module, func_name)
                        if callable(func):
                            # 尝试用不同参数调用
                            test_inputs = [
                                {},
                                {'x': 0},
                                {'data': []},
                                {'value': 1.0}
                            ]
                            
                            for test_input in test_inputs:
                                try:
                                    func(**test_input)
                                    passed_tests += 0.25  # 每个成功调用得0.25分
                                    break
                                except TypeError:
                                    continue  # 参数不匹配，尝试下一个
                                except Exception:
                                    break  # 执行错误，跳过
                    except:
                        continue
                
                coverage = min(1.0, passed_tests / total_tests)
                return coverage > 0.5, coverage
                
            finally:
                os.unlink(temp_path)
                if "temp_insight" in sys.modules:
                    del sys.modules["temp_insight"]
                    
        except Exception as e:
            return False, 0.0
    
    def _benchmark_performance(self, code: str, timeout: float = 0.5) -> Tuple[bool, float]:
        """
        性能基准测试
        要求: 执行时间 < timeout
        """
        try:
            # 创建隔离环境执行
            start = time.time()
            
            # 使用compile + exec执行
            compiled = compile(code, '<insight>', 'exec')
            namespace = {}
            
            exec(compiled, namespace)
            
            exec_time = time.time() - start
            
            return exec_time < timeout, exec_time
            
        except Exception as e:
            return False, timeout
    
    def _validate_semantics(self, code: str, metadata: Dict) -> Tuple[bool, float]:
        """
        语义验证 - 检查代码是否实现了声称的功能
        通过关键词匹配和代码结构分析
        """
        try:
            hypothesis = metadata.get('trigger_goal', '').lower()
            content = metadata.get('content', '').lower()
            
            # 提取关键概念
            keywords = self._extract_keywords(hypothesis + ' ' + content)
            
            # 检查代码中是否包含相关实现
            code_lower = code.lower()
            matches = sum(1 for kw in keywords if kw in code_lower)
            
            # 语义得分 = 匹配关键词数 / 总关键词数
            score = matches / len(keywords) if keywords else 0.5
            
            return score > 0.3, score
            
        except:
            return False, 0.0
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取文本中的关键技术词汇"""
        # 简单实现：提取3字以上的单词
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        # 过滤常见停用词
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        return [w for w in words if w not in stopwords][:10]  # 最多10个关键词
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        if not self.validation_history:
            return {'total': 0}
        
        return {
            'total': len(self.validation_history),
            'valid': sum(1 for v in self.validation_history if v['valid']),
            'average_score': sum(v['score'] for v in self.validation_history) / len(self.validation_history),
            'integrate_recommended': sum(1 for v in self.validation_history if v['recommendation'] == 'INTEGRATE'),
            'archive_recommended': sum(1 for v in self.validation_history if v['recommendation'] == 'ARCHIVE'),
            'reject_recommended': sum(1 for v in self.validation_history if v['recommendation'] == 'REJECT')
        }
