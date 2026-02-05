#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from validators import CodeValidator
from token_budget import TokenBudget

# Read batch 1 raw code
with open('output/test_v62_batch1_raw.py', 'r', encoding='utf-8') as f:
    code = f.read()

print("Testing truncation filter logic...")
print()

tb = TokenBudget()
truncation_info = tb.detect_truncation(code, detailed=True)

print(f"truncation_info.is_truncated: {truncation_info.is_truncated}")
print()
print("All details:")
for k, v in truncation_info.details.items():
    print(f"  {k}: {v}")
print()

# Test the filter condition
has_unterminated_string = truncation_info.details.get('unterminated_string')
other_issues = [k for k, v in truncation_info.details.items() if v and k != 'unterminated_string']

print(f"unterminated_string: {has_unterminated_string}")
print(f"Other issues: {other_issues}")
print(f"Count of other issues: {len(other_issues)}")
print()

# The condition
should_skip = (has_unterminated_string and len(other_issues) == 0)
print(f"Should skip truncation: {should_skip}")
