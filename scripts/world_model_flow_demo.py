"""世界模型完整链路演示脚本

运行步骤:
    python scripts/world_model_flow_demo.py

流程:
    1. 调用 generate 生成简单场景 (桌子+杯子)
    2. 调用 simulate 对杯子执行移动动作
    3. 调用 observe 观察最新状态

输出: JSON 摘要, 包含每一步 success/error 以及 world_id
"""
import sys, json
from typing import Any, Dict

sys.path.insert(0, r'd:\TRAE_PROJECT\AGI')

from enhanced_tools_collection import get_tool_manager


def main() -> int:
    tm = get_tool_manager()

    # 1) generate
    gen_prompt = "在房间里放一张桌子和一个杯子，桌子在(0,0,0)，杯子在(0.5,0,0.75)"
    res_g = tm.execute_tool('world_model', operation='generate', prompt=gen_prompt)

    world_id = None
    if res_g.data:
        world_id = res_g.data.get('world_id') or (res_g.data.get('world_data') or {}).get('world_id')

    # 2) simulate (仅在生成成功且有 world_id 时执行)
    actions = [
        {
            'type': 'move',
            'object': 'cup',
            'to': {'x': 0.6, 'y': 0.0, 'z': 0.75}
        }
    ]
    res_s = tm.execute_tool('world_model', operation='simulate', world_id=world_id, actions=actions) if world_id else None

    # 3) observe
    res_o = tm.execute_tool('world_model', operation='observe', world_id=world_id) if world_id else None

    summary: Dict[str, Any] = {
        'generate': {
            'success': res_g.success,
            'error': res_g.error,
            'world_id': world_id,
        },
        'simulate': {
            'success': (res_s.success if res_s else None),
            'error': (res_s.error if res_s else None),
        },
        'observe': {
            'success': (res_o.success if res_o else None),
            'error': (res_o.error if res_o else None),
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
