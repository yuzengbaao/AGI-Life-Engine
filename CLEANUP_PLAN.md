# AGI 项目文件清理计划

**目标**: 清理无关的测试、过程和依赖文件，保留核心功能

---

## 📊 文件分类

### ✅ 核心文件（必须保留）

#### 主程序
- `AGI_AUTONOMOUS_CORE_V6_2.py` ⭐ 最新版本
- `AGI_Life_Engine.py` ⭐ 完整系统
- `README.md` 项目说明
- `QUICKSTART.md` 快速启动
- `STARTUP_GUIDE_V62.md` 启动指南

#### Phase 1 & 2 组件（核心功能）
- `token_budget.py`
- `validators.py`
- `fixers.py`
- `adaptive_batch_processor.py`
- `incremental_validator.py`
- `error_classifier.py`
- `fix_optimizer.py`

#### 配置文件
- `.env.multi_model`
- `.env.example`
- `requirements.txt`
- `.gitignore`

#### 输出目录
- `output/test_v62.py` ⭐ 最新生成
- `output/test_v62_batch1_raw.py`
- `output/test_v62_batch2_raw.py`

#### 最新文档（保留）
- `ACCEPTANCE_GUIDE.md` 验收指南
- `GENERATED_FILES_EXPLANATION.md` 文件解释
- `FILES_ANALYSIS_VISUAL.md` 可视化分析
- `TEST_EXECUTION_REPORT_20260205.md` 测试报告
- `V62_TRUNCATION_FIX_REPORT.md` 修复文档

---

### ❌ 可以删除的文件

#### 旧版本程序
```
AGI_AUTONOMOUS_CORE_V6_0.py              ❌ V6.0 旧版
AGI_AUTONOMOUS_CORE_V6_1.py              ❌ V6.1 旧版
AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py   ❌ V6.1 多模型旧版
```

#### 过程文档（临时报告）
```
# V6.0 相关
V6_0_FINAL_COMPREHENSIVE_REPORT.md       ❌ 旧版报告
V6_1_IMPLEMENTATION_REPORT.md            ❌ 旧版报告

# V6.1 相关
V62_CAPABILITIES_ASSESSMENT.md           ❌ 已有更新版本
V62_FINAL_SUMMARY.txt                    ❌ 临时总结
V62_FINAL_TEST_REPORT.md                 ❌ 临时报告
V62_TEST_REPORT.md                       ❌ 临时报告
V62_INTEGRATION_COMPLETE.md              ❌ 集成完成报告

# V6.2 临时文档
V62_HOTFIX_20260205.md                   ❌ 修复记录（可归档）
FINAL_STATUS_CHECK.md                    ❌ 状态检查（可归档）
TEST_SESSION_SUMMARY.md                  ❌ 会话总结（可归档）
```

#### 测试和调试文件
```
compare_models.py                        ❌ 测试脚本
diagnose_truncation.py                   ❌ 调试脚本
test_batch1.py                          ❌ 临时测试
test_debug.py                            ❌ 调试脚本
test_session_monitor.py                 ❌ 监控脚本
test_truncation.py                       ❌ 测试脚本
test_validator_fix.py                    ❌ 测试脚本
verify_output.py                         ❌ 如存在则删除
```

#### 输出目录旧文件
```
output/full_test.py                      ❌ 旧测试
```

#### 备份文件（.bak, .backup）
```
core/actions/forage.py.bak_1766772461     ❌ 备份
core/agents_legacy.py.bak_1766792593      ❌ 备份
# 其他 .bak, .backup 文件
```

#### 临时 JSON 数据
```
core/batch_regression_results.json       ❌ 临时数据
```

---

### 📁 文件归档（可选保留）

#### 创建归档目录
```
archive/
├── old_versions/           # 旧版本程序
├── reports/                # 历史报告
├── test_scripts/           # 测试脚本
└── backups/                # 备份文件
```

---

## 🎯 清理执行计划

### Phase 1: 删除旧版本程序
```bash
# 删除 V6.0, V6.1 旧版
rm AGI_AUTONOMOUS_CORE_V6_0.py
rm AGI_AUTONOMOUS_CORE_V6_1.py
rm AGI_AUTONOMOUS_CORE_V6_1_MULTI_BASE.py
```

### Phase 2: 删除临时文档
```bash
# 删除旧报告和临时文档
rm V6_0_FINAL_COMPREHENSIVE_REPORT.md
rm V6_1_IMPLEMENTATION_REPORT.md
rm V62_CAPABILITIES_ASSESSMENT.md
rm V62_FINAL_SUMMARY.txt
rm V62_FINAL_TEST_REPORT.md
rm V62_TEST_REPORT.md
rm V62_INTEGRATION_COMPLETE.md
rm V62_HOTFIX_20260205.md
rm FINAL_STATUS_CHECK.md
rm TEST_SESSION_SUMMARY.md
```

### Phase 3: 删除测试脚本
```bash
rm compare_models.py
rm diagnose_truncation.py
rm test_batch1.py
rm test_debug.py
rm test_session_monitor.py
rm test_truncation.py
rm test_validator_fix.py
```

### Phase 4: 删除备份文件
```bash
find . -name "*.bak*" -type f -delete
find . -name "*.backup" -type f -delete
find . -name "*_backup_*" -type f -delete
```

### Phase 5: 删除临时数据
```bash
rm core/batch_regression_results.json
rm output/full_test.py
```

---

## 📋 清理后的目录结构

```
AGI/
├── AGI_AUTONOMOUS_CORE_V6_2.py     ⭐ 主程序
├── AGI_Life_Engine.py              ⭐ 完整系统
├── README.md                        ⭐ 项目说明
├── QUICKSTART.md                    ⭐ 快速启动
├── STARTUP_GUIDE_V62.md             ⭐ 启动指南
├── ACCEPTANCE_GUIDE.md              ⭐ 验收指南
├── GENERATED_FILES_EXPLANATION.md   ⭐ 文件说明
├── FILES_ANALYSIS_VISUAL.md        ⭐ 可视化
├── TEST_EXECUTION_REPORT_20260205.md ⭐ 测试报告
├── V62_TRUNCATION_FIX_REPORT.md    ⭐ 修复文档
├── requirements.txt                 ⭐ 依赖
├── .env.multi_model                ⭐ 配置
│
├── core/                            ⭐ 核心模块
│   ├── __init__.py
│   ├── goal_system.py
│   ├── llm_client.py
│   ├── system_tools.py
│   └── ... (其他核心模块)
│
├── token_budget.py                  ⭐ Phase 1
├── validators.py                    ⭐ Phase 1
├── fixers.py                        ⭐ Phase 1
├── adaptive_batch_processor.py     ⭐ Phase 2
├── incremental_validator.py        ⭐ Phase 2
├── error_classifier.py              ⭐ Phase 2
├── fix_optimizer.py                 ⭐ Phase 2
│
└── output/                          ⭐ 输出
    ├── test_v62.py                  ⭐ 主要输出
    ├── test_v62_batch1_raw.py
    └── test_v62_batch2_raw.py
```

---

## 🔧 清理脚本

创建 `cleanup_project.py`:
```python
#!/usr/bin/env python3
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
    '*_backup_*',
]

def cleanup():
    """执行清理"""
    root = Path('.')
    deleted = []
    errors = []

    print("=" * 80)
    print("AGI Project Cleanup")
    print("=" * 80)
    print()

    # 删除指定文件
    print("Phase 1: Deleting specified files...")
    for file_path in FILES_TO_DELETE:
        path = root / file_path
        if path.exists():
            try:
                if path.is_file():
                    path.unlink()
                    deleted.append(str(path))
                    print(f"  ✓ Deleted: {file_path}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    deleted.append(str(path))
                    print(f"  ✓ Deleted (dir): {file_path}")
            except Exception as e:
                errors.append(f"{file_path}: {e}")
                print(f"  ✗ Error: {file_path} - {e}")
        else:
            print(f"  - Not found: {file_path}")

    # 删除备份文件
    print()
    print("Phase 2: Deleting backup files...")
    for pattern in BACKUP_PATTERNS:
        matches = root.rglob(pattern)
        for match in matches:
            if match.is_file() and '.git' not in str(match):
                try:
                    match.unlink()
                    deleted.append(str(match.relative_to(root)))
                    print(f"  ✓ Deleted: {match.relative_to(root)}")
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
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print()
        print("✓ All cleanup operations completed successfully!")

    print()

if __name__ == '__main__':
    # 确认
    print("This will delete the files listed above.")
    response = input("Continue? (yes/no): ")
    if response.lower() == 'yes':
        cleanup()
    else:
        print("Cleanup cancelled.")
```

---

## ⚠️ 注意事项

1. **备份重要文件**: 在执行清理前，建议先备份整个项目
2. **Git 清理**: 删除文件后需要 `git add` 和 `git commit`
3. **逐步执行**: 建议分阶段执行，每阶段后检查系统是否正常
4. **保留文档**: 保留用户指南、API文档等重要文档

---

## 📊 清理效果

### 清理前
- 文件数: ~500+ (包括大量过程文件)
- 目录: 混乱
- 维护: 困难

### 清理后
- 文件数: ~200 (核心文件)
- 目录: 清晰
- 维护: 容易

---

**准备执行清理？请确认是否继续。**
