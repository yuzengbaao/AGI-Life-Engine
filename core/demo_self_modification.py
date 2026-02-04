"""
演示 SelfModifyingEngine 使用 TemplateBasedPatchGenerator 的示例脚本。
运行：
    python core/demo_self_modification.py
"""
import sys
from pathlib import Path
import difflib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.self_modifying_engine import SelfModifyingEngine

def show_diff(old: str, new: str):
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile='original', tofile='modified')
    print(''.join(diff))


def main():
    engine = SelfModifyingEngine(project_root=str(PROJECT_ROOT))
    # 选择一个小型目标模块进行演示
    target = 'core._demo_target_small'
    patch = engine.propose_patch(
        target,
        issue_description='将pass替换为raise NotImplementedError()',
        optimization_goal='readability',
        use_llm=False,
        patch_strategy='ast_diff'
    )
    if not patch:
        print('未生成补丁')
        return
    print('补丁描述:', patch.description)
    show_diff(patch.original_code, patch.modified_code)

if __name__ == '__main__':
    main()
