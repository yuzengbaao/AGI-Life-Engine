#!/usr/bin/env python3
"""AGI Project Cleanup Script"""

import os
import shutil
from pathlib import Path

# 要删除的文件列表
FILES_TO_DELETE = [
    # 旧版本程序
    'AGI_AUTONOMOUS_CORE_V6_0.py',
    'AGI_AUTONOMOUS_CORE_V6_1.py',
    'AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py',

    # 临时文档
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

    # 测试脚本
    'compare_models.py',
    'diagnose_truncation.py',
    'test_batch1.py',
    'test_debug.py',
    'test_session_monitor.py',
    'test_truncation.py',
    'test_validator_fix.py',

    # 临时数据
    'core/batch_regression_results.json',
    'output/full_test.py',
]

# 要删除的备份文件模式
BACKUP_PATTERNS = [
    '*.bak*',
    '*.backup',
]

def cleanup():
    """执行清理"""
    root = Path('.')
    deleted = []
    errors = []

    print("=" * 80)
    print("AGI Project Cleanup - Execution")
    print("=" * 80)
    print()

    # Phase 1: 删除指定文件
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
                    print(f"  ✓ Deleted (dir): {file_path}")
            except Exception as e:
                errors.append(f"{file_path}: {e}")
                print(f"  ✗ Error: {file_path} - {e}")
        else:
            print(f"  - Not found: {file_path}")

    # Phase 2: 删除备份文件
    print()
    print("Phase 2: Deleting backup files...")
    print("-" * 80)
    for pattern in BACKUP_PATTERNS:
        matches = root.rglob(pattern)
        for match in matches:
            # 跳过 .git 目录
            if '.git' in str(match):
                continue
            # 跳过脚本本身
            if 'cleanup_project.py' in str(match):
                continue
            if match.is_file():
                try:
                    rel_path = str(match.relative_to(root))
                    match.unlink()
                    deleted.append(rel_path)
                    print(f"  ✓ Deleted: {rel_path}")
                except Exception as e:
                    errors.append(f"{match}: {e}")
                    print(f"  ✗ Error: {match} - {e}")

    # 摘要
    print()
    print("=" * 80)
    print("Cleanup Summary")
    print("=" * 80)
    print(f"Files deleted: {len(deleted)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print()
        print("Errors encountered:")
        for error in errors:
            print(f"  - {error}")
    else:
        print()
        print("✓ All cleanup operations completed successfully!")

    print()
    print("=" * 80)
    return deleted, errors

if __name__ == '__main__':
    deleted, errors = cleanup()

    # 保存清理记录
    with open('cleanup_log.txt', 'w', encoding='utf-8') as f:
        f.write("Cleanup Log\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Files deleted: {len(deleted)}\n")
        f.write(f"Errors: {len(errors)}\n\n")
        if deleted:
            f.write("Deleted files:\n")
            for f_path in deleted:
                f.write(f"  - {f_path}\n")
        if errors:
            f.write("\nErrors:\n")
            for error in errors:
                f.write(f"  - {error}\n")

    print("Cleanup log saved to: cleanup_log.txt")
