"""细粒度世界模型物理与环境字段测试

验证 generate/simulate/observe 返回的扩展字段结构存在与基本类型正确。
运行: pytest tests/test_world_model_physics_details.py -q
"""
import os
import json
import pytest

from enhanced_tools_collection import get_tool_manager

@pytest.mark.asyncio
async def test_generate_contains_environment_and_objects():
    tm = get_tool_manager()
    res = tm.execute_tool('world_model', operation='generate', prompt='桌子和一个杯子在晨光房间里')
    assert res.success, f"generate failed: {res.error}"
    data = res.data or {}
    world_data = data.get('world_data', {})
    assert 'environment' in world_data, 'missing environment'
    env = world_data['environment']
    for key in ['lighting','time_of_day','materials','interactive_events']:
        assert key in env, f'missing env key {key}'
    assert isinstance(env['lighting'], dict)
    assert isinstance(env['materials'], dict)
    objects = world_data.get('objects', [])
    assert objects, 'objects list should not be empty'
    # 取第一个对象校验结构
    obj = objects[0]
    for key in ['id','type','position','material','size']:
        assert key in obj, f'missing object key {key}'
    assert isinstance(obj['position'], list) and len(obj['position']) == 3

@pytest.mark.asyncio
async def test_simulate_returns_physics_state():
    tm = get_tool_manager()
    res_g = tm.execute_tool('world_model', operation='generate', prompt='桌子和一个杯子')
    world_id = (res_g.data or {}).get('world_id') or ((res_g.data or {}).get('world_data') or {}).get('world_id')
    actions = [{'type':'move','object':'cup','to':{'x':0.6,'y':0.0,'z':0.75}}]
    res_s = tm.execute_tool('world_model', operation='simulate', world_id=world_id, actions=actions)
    assert res_s.success, f"simulate failed: {res_s.error}"
    data = res_s.data or {}
    physics_state = data.get('physics_state') or (data.get('updated_state') or {}).get('physics_state')
    assert physics_state, 'physics_state missing'
    for key in ['energy','momentum','collisions_predicted','actions_applied']:
        assert key in physics_state, f'missing physics_state key {key}'
    assert isinstance(physics_state['actions_applied'], list)

@pytest.mark.asyncio
async def test_observe_includes_physics_state_snapshot():
    tm = get_tool_manager()
    res_g = tm.execute_tool('world_model', operation='generate', prompt='桌子和一个杯子')
    world_id = (res_g.data or {}).get('world_id') or ((res_g.data or {}).get('world_data') or {}).get('world_id')
    target = {'x':0.65,'y':0.0,'z':0.2}
    tm.execute_tool('world_model', operation='simulate', world_id=world_id, actions=[{'type':'move','object':'cup','to':target}])
    res_o = tm.execute_tool('world_model', operation='observe', world_id=world_id)
    assert res_o.success, f"observe failed: {res_o.error}"
    data = res_o.data or {}
    obs = data.get('observations', {})
    assert 'physics_state' in obs, 'physics_state not in observations'
    assert 'object_positions' in obs, 'object_positions missing'
    assert isinstance(obs['object_positions'], list)
    # 确认至少一个对象位置包含id/type/position
    if obs['object_positions']:
        entry0 = obs['object_positions'][0]
        for key in ['id','type','position']:
            assert key in entry0, f'missing object position key {key}'
        # 进一步校验：cup 的位置与最近一次 simulate 目标一致（允许极小误差）
        cup = next((e for e in obs['object_positions'] if e.get('type') == 'cup' or e.get('id') == 'cup_1'), None)
        assert cup is not None, 'cup position entry missing'
        pos = cup.get('position')
        assert isinstance(pos, list) and len(pos) == 3, 'cup position should be [x,y,z]'
        assert pos[0] == pytest.approx(target['x'], abs=1e-6)
        assert pos[1] == pytest.approx(target['y'], abs=1e-6)
        assert pos[2] == pytest.approx(target['z'], abs=1e-6)
        # 校验 physics_state 中记录的动作包含该 move（宽松匹配数值）
        ps = obs.get('physics_state', {})
        aa = ps.get('actions_applied', []) or []
        last = aa[-1] if aa else {}
        if last:
            assert last.get('type') == 'move'
            assert last.get('object') in ('cup','cup_1')
            to = (last.get('to') or {})
            assert to.get('x') == pytest.approx(target['x'], abs=1e-6)
            assert to.get('y') == pytest.approx(target['y'], abs=1e-6)
            assert to.get('z') == pytest.approx(target['z'], abs=1e-6)
            
