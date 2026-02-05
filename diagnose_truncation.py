#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnose truncation detection issue"""

from token_budget import TokenBudget
from validators import CodeValidator

print("=" * 80)
print("TRUNCATION DETECTION DIAGNOSIS")
print("=" * 80)
print()

# Read generated code
with open('output/test_v62.py', 'r', encoding='utf-8') as f:
    final_code = f.read()

# Test with TokenBudget directly
print("1. TokenBudget.detect_truncation():")
print("-" * 80)
tb = TokenBudget()
result = tb.detect_truncation(final_code)
print(f"  is_truncated: {result.is_truncated}")
print(f"  confidence: {result.confidence}")
print(f"  suggestions: {result.suggestions}")
print()

# Test with CodeValidator
print("2. CodeValidator.validate_code():")
print("-" * 80)
validator = CodeValidator()
validation = validator.validate_code(final_code, 'test_v62.py')
print(f"  is_valid: {validation.is_valid}")
print(f"  error_type: {validation.error_type}")
print(f"  error_message: {validation.error_message}")
print(f"  suggestions: {validation.suggestions}")
print()

# Check AST parsing
print("3. AST Parsing:")
print("-" * 80)
import ast
try:
    tree = ast.parse(final_code)
    print(f"  Parsed successfully: Yes")
    print(f"  Number of classes: {len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])}")
    print(f"  Number of functions: {len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])}")
except SyntaxError as e:
    print(f"  Parsed successfully: No")
    print(f"  Error: {e}")
print()

# Code stats
print("4. Code Statistics:")
print("-" * 80)
lines = final_code.splitlines()
print(f"  Total lines: {len(lines)}")
print(f"  Non-empty lines: {len([l for l in lines if l.strip()])}")
print(f"  Characters: {len(final_code)}")
print(f"  Estimated tokens (chars/4): {len(final_code) // 4}")
print()

# Show the code
print("5. Generated Code:")
print("-" * 80)
print(final_code)
print("=" * 80)
