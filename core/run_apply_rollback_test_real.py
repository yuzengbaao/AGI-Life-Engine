import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.self_modifying_engine import SelfModifyingEngine

# 使用真实模块进行集成回归测试（自动应用/回滚）
TARGET_MODULE = 'agent_protocol'

engine = SelfModifyingEngine(project_root=str(PROJECT_ROOT))
patch = engine.propose_patch(
    TARGET_MODULE,
    issue_description='为函数添加docstring模板',
    optimization_goal='readability',
    # 使用内部可读性优化（不依赖LLM）
    use_llm=True,
    patch_strategy='auto'
)

if not patch:
    print('未生成补丁')
    raise SystemExit(1)

ok, report = engine.sandbox_test(patch)
print('sandbox_test:', ok)
print('report:', report)

if not ok:
    raise SystemExit(2)

record = engine.apply_or_reject(patch, force_apply=True)
print('apply_result:', record)

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
