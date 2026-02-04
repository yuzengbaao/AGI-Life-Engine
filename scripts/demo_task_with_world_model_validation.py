"""高层任务执行示例：规划→世界模型物理验证

演示流程:
1. 构造一个简单“移动杯子”任务规划（伪规划）
2. 使用 world_model generate 创建场景
3. 将计划中的动作通过 simulate 进行物理仿真
4. 使用 observe 获取结果与 physics_state 快照
5. 根据 physics_state / 对象位置做简单验证反馈

运行: python scripts/demo_task_with_world_model_validation.py
"""
import sys, json
sys.path.insert(0, r'd:\TRAE_PROJECT\AGI')

from enhanced_tools_collection import get_tool_manager

def main() -> int:
    tm = get_tool_manager()

    # 1) 任务规划（占位：实际系统可替换为真实规划模块）
    task_goal = "将杯子稍微向右移动保持在桌面上"
    planned_actions = [
        { 'type':'move', 'object':'cup', 'to': {'x':0.6,'y':0.0,'z':0.75} }
    ]

    # 2) 生成初始世界
    gen_prompt = "晨光中的房间里有一张桌子和一个杯子"
    res_gen = tm.execute_tool('world_model', operation='generate', prompt=gen_prompt)
    world_id = (res_gen.data or {}).get('world_id') or ((res_gen.data or {}).get('world_data') or {}).get('world_id')

    # 3) 仿真计划中的动作
    res_sim = tm.execute_tool('world_model', operation='simulate', world_id=world_id, actions=planned_actions)

    # 4) 观测最新世界状态
    res_obs = tm.execute_tool('world_model', operation='observe', world_id=world_id)
    observations = (res_obs.data or {}).get('observations', {})

    # 5) 简单验证：检查杯子位置是否接近目标 (x,y,z)
    target = planned_actions[0]['to']
    target_xyz = [target['x'], target['y'], target['z']]
    cup_pos = None
    for obj in observations.get('object_positions', []):
        if obj.get('type') == 'cup' or obj.get('id') == 'cup_1':
            cup_pos = obj.get('position')
            break

    def _xyz_match(a, b, tol=1e-6):
        return (
            isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == 3 and len(b) == 3 and
            abs(a[0]-b[0]) < tol and abs(a[1]-b[1]) < tol and abs(a[2]-b[2]) < tol
        )

    ps = observations.get('physics_state', {})
    actions_applied = ps.get('actions_applied', []) or []
    last_action = actions_applied[-1] if actions_applied else None

    validation = {
        'goal': task_goal,
        'world_id': world_id,
        'target_xyz': target_xyz,
        'cup_position': cup_pos,
        'position_match': _xyz_match(cup_pos, target_xyz) if cup_pos else False,
        'physics_state_keys': list(ps.keys()),
        'physics_summary': {
            'collisions_predicted_count': len(ps.get('collisions_predicted', []) or []),
            'last_action_type': last_action.get('type') if last_action else None,
            'last_action_object': last_action.get('object') if last_action else None,
            'last_action_to': last_action.get('to') if last_action else None,
        }
    }

    summary = {
        'generate_success': res_gen.success,
        'simulate_success': res_sim.success if res_sim else None,
        'observe_success': res_obs.success if res_obs else None,
        'validation': validation
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
