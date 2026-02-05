#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from token_budget import TokenBudget

# Read batch 1 raw code
with open('output/test_v62_batch1_raw.py', 'r', encoding='utf-8') as f:
    code = f.read()

print("Testing batch 1 code for truncation...")
print(f"Lines: {len(code.splitlines())}")
print(f"Characters: {len(code)}")
print()

tb = TokenBudget()
result = tb.detect_truncation(code, detailed=True)

print(f"Is Truncated: {result.is_truncated}")
print(f"Confidence: {result.confidence}")
print()
print("Details:")
for key, value in result.details.items():
    if value:
        print(f"  {key}: {value}")
print()
if result.suggestions:
    print("Suggestions:")
    for s in result.suggestions:
        print(f"  - {s}")
