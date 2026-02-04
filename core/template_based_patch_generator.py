# TemplateBasedPatchGenerator
# 2026-01-17
# 该模块用于在无LLM条件下，基于模板、程序合成、进化等方法自动生成Python代码补丁。
# 支持：
#   1. 结构化AST模板匹配与替换
#   2. 规则驱动的代码片段插拔
#   3. 简单的基因式变异/交叉生成
#   4. 可扩展的自定义补丁策略

import ast
import difflib
import copy
import random
from typing import List, Dict, Any, Optional, Tuple


def _ensure_import_present(tree: ast.Module, module: str, name: Optional[str] = None):
    """确保 AST 中存在指定的 import；如果不存在则插入顶层 import。返回是否已插入。"""
    need_from = name is not None
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name == module and not need_from:
                    return False
        if isinstance(node, ast.ImportFrom):
            if node.module == module and (name is None or any(n.name == name for n in node.names)):
                return False

    if need_from:
        imp = ast.ImportFrom(module=module, names=[ast.alias(name=name, asname=None)], level=0)
    else:
        imp = ast.Import(names=[ast.alias(name=module, asname=None)])
    tree.body.insert(0, imp)
    return True

class TemplateBasedPatchGenerator:
    """
    无LLM依赖的代码补丁生成器。支持模板匹配、结构替换、进化式变异等多种策略。
    
    关键设计:
    - 只对单个函数或少量代码块做变更，避免全文件diff过大
    - max_diff_lines 控制最大补丁行数
    """
    def __init__(self, templates: Optional[List[Dict[str, Any]]] = None, max_diff_lines: int = 80):
        # 初始化时注入一些常用模板
        builtin = [
            {
                "desc": "将pass替换为raise NotImplementedError()",
                "code": "def PLACEHOLDER():\n    raise NotImplementedError()\n"
            },
            {
                "desc": "为简单返回函数添加 lru_cache 装饰器",
                "code": "from functools import lru_cache\n\n@lru_cache(maxsize=128)\ndef PLACEHOLDER():\n    pass\n"
            },
            {
                "desc": "为函数添加docstring模板",
                "code": "def PLACEHOLDER():\n    \"\"\"TODO: Add docstring\"\"\"\n    pass\n"
            }
        ]
        self.templates = (templates or []) + builtin
        self.max_diff_lines = max_diff_lines

    def add_template(self, template: Dict[str, Any]):
        self.templates.append(template)

    def generate_patch(self, old_code: str, target_desc: str = "", strategy: str = "auto") -> str:
        """
        输入原始代码和目标描述，自动生成补丁代码。
        支持多种策略：template/ast_diff/genetic/targeted/auto
        
        targeted 策略：只修改一个随机选中的函数，生成小补丁
        """
        # 优先尝试 targeted 策略（生成小补丁）
        if strategy == "targeted" or strategy == "auto":
            patch = self._targeted_patch(old_code, target_desc)
            if patch:
                return patch
        if strategy == "template" or (strategy == "auto" and self.templates):
            patch = self._template_patch(old_code, target_desc)
            if patch:
                return patch
        if strategy == "ast_diff" or strategy == "auto":
            patch = self._ast_diff_patch(old_code, target_desc)
            if patch:
                return patch
        if strategy == "genetic":
            patch = self._genetic_patch(old_code, target_desc)
            if patch:
                return patch
        return ""  # 未能生成补丁

    def _targeted_patch(self, old_code: str, target_desc: str) -> Optional[str]:
        """
        针对性补丁：只修改一个随机选中的函数，控制补丁规模。
        支持的变更类型：
        - 添加 docstring
        - 添加日志语句
        - 添加类型提示注释
        - 添加缓存装饰器
        """
        try:
            tree = ast.parse(old_code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if not functions:
                return None
                
            # 选择一个适合变更的函数（优先选没有docstring且行数<30的）
            candidates = []
            for fn in functions:
                has_docstring = (fn.body and isinstance(fn.body[0], ast.Expr) 
                                and isinstance(fn.body[0].value, ast.Constant)
                                and isinstance(fn.body[0].value.value, str))
                fn_size = getattr(fn, 'end_lineno', fn.lineno + 10) - fn.lineno
                if fn_size < 30:
                    candidates.append((fn, has_docstring, fn_size))
            
            if not candidates:
                candidates = [(fn, False, 10) for fn in functions[:3]]
            
            # 优先选择没有docstring的小函数
            candidates.sort(key=lambda x: (x[1], x[2]))
            target_fn, has_docstring, _ = candidates[0]
            
            # 获取函数的源代码行范围
            lines = old_code.splitlines(keepends=True)
            start_line = target_fn.lineno - 1
            end_line = getattr(target_fn, 'end_lineno', start_line + len(target_fn.body) + 5)
            
            # 决定变更类型 - 智能选择可行的变更
            available_changes = ['logging']  # logging 总是可行
            if not has_docstring:
                available_changes.append('docstring')
            if len(target_fn.body) <= 5 and not target_fn.decorator_list:
                available_changes.append('decorator')
            
            change_type = random.choice(available_changes)
            
            new_lines = lines[:]
            changed = False
            
            if change_type == 'docstring' and not has_docstring:
                # 在函数定义后插入docstring
                indent = self._get_indent(lines[start_line])
                docstring = f'{indent}    """TODO: 自动生成的文档注释 - {target_fn.name}"""\n'
                insert_pos = start_line + 1
                new_lines.insert(insert_pos, docstring)
                changed = True
                
            elif change_type == 'logging':
                # 在函数开头插入日志语句
                indent = self._get_indent(lines[start_line])
                log_stmt = f'{indent}    # [AutoPatch] 函数入口日志: {target_fn.name}\n'
                insert_pos = start_line + 1
                # 如果有docstring，跳过它
                if has_docstring:
                    insert_pos += 1
                new_lines.insert(insert_pos, log_stmt)
                changed = True
                
            elif change_type == 'decorator':
                # 在函数定义前添加装饰器（仅对简单函数）
                if len(target_fn.body) <= 5 and not target_fn.decorator_list:
                    indent = self._get_indent(lines[start_line])
                    decorator = f'{indent}# @lru_cache  # [AutoPatch] 建议添加缓存\n'
                    new_lines.insert(start_line, decorator)
                    changed = True
            
            if changed:
                new_code = ''.join(new_lines)
                if new_code != old_code:
                    return new_code
                
        except Exception as e:
            pass
        return None
    
    def _get_indent(self, line: str) -> str:
        """获取行的缩进"""
        return line[:len(line) - len(line.lstrip())]

    def _template_patch(self, old_code: str, target_desc: str) -> Optional[str]:
        # 简单模板匹配与替换
        for tpl in self.templates:
            if tpl.get("desc") in target_desc:
                # 用PLACEHOLDER替换为原函数名（若能推断）
                try:
                    tree = ast.parse(old_code)
                    # 找到第一个函数名
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef):
                            name = node.name
                            return tpl.get("code", "").replace("PLACEHOLDER", name)
                except Exception:
                    return tpl.get("code", "")
        return None

    def _ast_diff_patch(self, old_code: str, target_desc: str) -> Optional[str]:
        # 基于AST的结构化diff与重组
        try:
            old_ast = ast.parse(old_code)
            # 支持多种基于目标描述的结构化变换
            class ASTTransformer(ast.NodeTransformer):
                def visit_Pass(self, node):
                    return ast.Raise(
                        exc=ast.Call(
                            func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                            args=[], keywords=[]),
                        cause=None)

                def visit_FunctionDef(self, node: ast.FunctionDef):
                    # 如果目标描述包含cache关键词，为返回简单值的函数注入 lru_cache
                    self.generic_visit(node)
                    if 'cache' in target_desc.lower():
                        # 在函数定义上插入装饰器标记（实际添加 import 在模块层面处理）
                        dec = ast.Name(id='lru_cache', ctx=ast.Load())
                        node.decorator_list.insert(0, dec)
                    # 如果函数体只有pass，插入docstring或raise
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        node.body[0] = ast.Raise(
                            exc=ast.Call(func=ast.Name(id='NotImplementedError', ctx=ast.Load()), args=[], keywords=[]),
                            cause=None)
                    return node

            new_ast = ASTTransformer().visit(old_ast)
            # 如果注入了 lru_cache 装饰器，确保导入存在
            if any(isinstance(n, ast.FunctionDef) and n.decorator_list for n in new_ast.body):
                _ensure_import_present(new_ast, 'functools', 'lru_cache')

            ast.fix_missing_locations(new_ast)
            new_code = ast.unparse(new_ast)
            if new_code != old_code:
                return new_code
        except Exception as e:
            return None
        return None

    def _genetic_patch(self, old_code: str, target_desc: str) -> Optional[str]:
        # 简单基因式变异/交叉生成（示例）
        # 简单进化策略：变量重命名、插入缓存装饰器、行交换
        try:
            tree = ast.parse(old_code)

            # 变异1: 随机重命名局部变量
            class VarRenamer(ast.NodeTransformer):
                def visit_Name(self, node: ast.Name):
                    if isinstance(node.ctx, ast.Store) and node.id.isalpha() and len(node.id) > 1:
                        if random.random() < 0.05:
                            return ast.copy_location(ast.Name(id=node.id + '_v', ctx=node.ctx), node)
                    return node

            # 变异2: 对简单函数注入 lru_cache
            class CacheInjector(ast.NodeTransformer):
                def visit_FunctionDef(self, node: ast.FunctionDef):
                    self.generic_visit(node)
                    # 如果函数仅返回常量或简单表达式，则注入缓存装饰器
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                        if random.random() < 0.5:
                            node.decorator_list.insert(0, ast.Name(id='lru_cache', ctx=ast.Load()))
                    return node

            transformers = [VarRenamer(), CacheInjector()]
            for t in transformers:
                tree = t.visit(tree)

            _ensure_import_present(tree, 'functools', 'lru_cache')
            ast.fix_missing_locations(tree)
            new_code = ast.unparse(tree)
            if new_code != old_code:
                return new_code
        except Exception:
            pass
        # 兜底：简单行交换
        lines = old_code.splitlines()
        if len(lines) > 3:
            i, j = 1, -2
            new_lines = lines[:]
            new_lines[i], new_lines[j] = new_lines[j], new_lines[i]
            return "\n".join(new_lines)
        return None

    def list_templates(self) -> List[Dict[str, Any]]:
        return self.templates

    def clear_templates(self):
        self.templates = []
