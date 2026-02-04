import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.self_modifying_engine import SelfModifyingEngine

engine = SelfModifyingEngine(project_root=str(PROJECT_ROOT))
patch = engine.propose_patch(
    'core._demo_target_small',
    issue_description='将pass替换为raise NotImplementedError()',
    optimization_goal='readability',
    use_llm=False,
    patch_strategy='ast_diff'
)
if not patch:
    print('未生成补丁')
    raise SystemExit(1)

ok, report = engine.sandbox_test(patch)
print('sandbox_test:', ok)
print('report:', report)

if not ok:
    raise SystemExit(2)

# 自动应用补丁
record = engine.apply_or_reject(patch, force_apply=True)
print('apply_result:', record)

# 自动回滚补丁
record_id = None
if hasattr(record, 'id'):
    record_id = record.id
elif isinstance(record, dict):
    record_id = record.get('record_id')

if record_id:
    rollback_ok = engine.rollback(record_id)
    print('rollback:', rollback_ok)
else:
    print('rollback: skipped (no record_id)')
