#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试targeted_patch方法"""
import sys
import ast
import random
random.seed(42)

sys.path.insert(0, 'd:/TRAE_PROJECT/AGI')
from pathlib import Path

module_path = Path('d:/TRAE_PROJECT/AGI/core/math_utils.py')
with open(module_path, 'r', encoding='utf-8') as f:
    old_code = f.read()

tree = ast.parse(old_code)
functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
print(f'Found {len(functions)} functions')

for fn in functions:
    has_docstring = (fn.body and isinstance(fn.body[0], ast.Expr) 
                    and isinstance(fn.body[0].value, ast.Constant)
                    and isinstance(fn.body[0].value.value, str))
    fn_size = getattr(fn, 'end_lineno', fn.lineno + 10) - fn.lineno
    print(f'  {fn.name}: has_docstring={has_docstring}, size={fn_size}, lineno={fn.lineno}')

lines = old_code.splitlines(keepends=True)
print(f'Total lines: {len(lines)}')

# 模拟 targeted_patch
target_fn = functions[0]
start_line = target_fn.lineno - 1
print(f'\nTarget function: {target_fn.name}, start_line={start_line}')
print(f'Line content: {repr(lines[start_line])}')

def _get_indent(line):
    return line[:len(line) - len(line.lstrip())]

indent = _get_indent(lines[start_line])
print(f'Indent: {repr(indent)}')

# Check docstring
has_docstring = (target_fn.body and isinstance(target_fn.body[0], ast.Expr) 
                and isinstance(target_fn.body[0].value, ast.Constant)
                and isinstance(target_fn.body[0].value.value, str))
print(f'has_docstring: {has_docstring}')

if not has_docstring:
    docstring = f'{indent}    """TODO: 自动生成的文档注释 - {target_fn.name}"""\n'
    insert_pos = start_line + 1
    new_lines = lines[:]
    new_lines.insert(insert_pos, docstring)
    new_code = ''.join(new_lines)
    print(f'Old len: {len(old_code)}, New len: {len(new_code)}')
    print(f'Equal: {new_code == old_code}')
    if new_code != old_code:
        print('Patch would be generated!')
        # Show diff
        from difflib import unified_diff
        diff = list(unified_diff(
            old_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True)
        ))
        print(f'Diff lines: {len(diff)}')
        for line in diff[:15]:
            print(line, end='')
else:
    print('Function already has docstring, checking body...')
    print(f'Body[0] type: {type(target_fn.body[0]).__name__}')
