#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

FILES_TO_DELETE = [
    'AGI_AUTONOMOUS_CORE_V6_0.py',
    'AGI_AUTONOMOUS_CORE_V6_1.py',
    'AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py',
    'V6_0_FINAL_COMPREHENSIVE_REPORT.md',
    'V6_1_IMPLEMENTATION_REPORT.md',
    'V62_CAPABILITIES_ASSESSMENT.md',
    'V62_FINAL_SUMMARY.txt',
    'V62_FINAL_TEST_REPORT.md',
    'V62_TEST_REPORT.md',
    'V62_INTEGRATION_COMPLETE.md',
    'V62_HOTFIX_20260205.md',
    'FINAL_STATUS_CHECK.md',
    'TEST_SESSION_SUMMARY.md',
    'compare_models.py',
    'diagnose_truncation.py',
    'test_batch1.py',
    'test_debug.py',
    'test_session_monitor.py',
    'test_truncation.py',
    'test_validator_fix.py',
    'core/batch_regression_results.json',
    'output/full_test.py',
]

BACKUP_PATTERNS = ['*.bak*', '*.backup']

print("=" * 80)
print("AGI Project Cleanup - Execution")
print("=" * 80)
print()

root = Path('.')
deleted = []
errors = []

print("Phase 1: Deleting specified files...")
print("-" * 80)
for file_path in FILES_TO_DELETE:
    path = root / file_path
    if path.exists():
        try:
            if path.is_file():
                path.unlink()
                deleted.append(str(path))
                print(f"  [OK] Deleted: {file_path}")
            elif path.is_dir():
                shutil.rmtree(path)
                deleted.append(str(path))
                print(f"  [OK] Deleted (dir): {file_path}")
        except Exception as e:
            errors.append(f"{file_path}: {e}")
            print(f"  [ERROR] {file_path} - {e}")
    else:
        print(f"  [-] Not found: {file_path}")

print()
print("Phase 2: Deleting backup files...")
print("-" * 80)
for pattern in BACKUP_PATTERNS:
    matches = root.rglob(pattern)
    for match in matches:
        if '.git' in str(match):
            continue
        if 'cleanup.py' in str(match):
            continue
        if match.is_file():
            try:
                rel_path = str(match.relative_to(root))
                match.unlink()
                deleted.append(rel_path)
                print(f"  [OK] Deleted: {rel_path}")
            except Exception as e:
                errors.append(f"{match}: {e}")
                print(f"  [ERROR] {match} - {e}")

print()
print("=" * 80)
print("Cleanup Summary")
print("=" * 80)
print(f"Files deleted: {len(deleted)}")
print(f"Errors: {len(errors)}")

if errors:
    print()
    print("Errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print()
    print("[SUCCESS] All cleanup operations completed!")

print()
print("=" * 80)

# Save log
with open('cleanup_log.txt', 'w', encoding='utf-8') as f:
    f.write(f"Deleted: {len(deleted)} files\n")
    f.write(f"Errors: {len(errors)}\n\n")
    for d in deleted:
        f.write(f"  - {d}\n")
    for e in errors:
        f.write(f"  ERROR: {e}\n")

print("Cleanup log saved to: cleanup_log.txt")
