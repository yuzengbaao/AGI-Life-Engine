#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from validators import CodeValidator

# Read batch 1 raw code
with open('output/test_v62_batch1_raw.py', 'r', encoding='utf-8') as f:
    code = f.read()

print("Testing CodeValidator with batch 1 code...")
print(f"Lines: {len(code.splitlines())}")
print()

validator = CodeValidator()
result = validator.validate_code(code, 'test_v62_batch1_raw.py')

print(f"Is Valid: {result.is_valid}")
print(f"Error Type: {result.error_type}")
print(f"Error Message: {result.error_message}")
print()

if result.metadata.get('truncation_skipped'):
    print(f"Truncation Skipped: {result.metadata['truncation_skipped']}")

if result.suggestions:
    print("Suggestions:")
    for s in result.suggestions:
        print(f"  - {s}")
