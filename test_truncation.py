#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test truncation detection"""

from token_budget import TokenBudget

# Read the generated code
with open('output/test_v62.py', 'r', encoding='utf-8') as f:
    code = f.read()

print("=" * 80)
print("CODE TO TEST:")
print("=" * 80)
print(code)
print("=" * 80)
print()

# Test truncation detection
tb = TokenBudget()
result = tb.detect_truncation(code, detailed=True)

print("TRUNCATION DETECTION RESULT:")
print(f"  Is Truncated: {result.is_truncated}")
print(f"  Confidence: {result.confidence}")
print()
print("Details:")
for key, value in result.details.items():
    print(f"  {key}: {value}")
print()
print("Suggestions:")
for suggestion in result.suggestions:
    print(f"  - {suggestion}")
