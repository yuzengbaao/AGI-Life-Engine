import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.self_modifying_engine import SelfModifyingEngine

engine = SelfModifyingEngine(project_root=str(Path.cwd()))
patch = engine.propose_patch(
    'core._demo_target_small',
    issue_description='将pass替换为raise NotImplementedError()',
    optimization_goal='readability',
    use_llm=False,
    patch_strategy='ast_diff'
)
print('patch_generated:', bool(patch))
if not patch:
    raise SystemExit(1)

ok, report = engine.sandbox_test(patch)
print('sandbox_test:', ok)
print('report:', report)
