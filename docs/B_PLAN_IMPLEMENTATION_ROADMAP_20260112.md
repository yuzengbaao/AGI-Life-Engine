# B方案系统升级实施路线图

**创建日期**: 2026-01-12
**目标**: 将TRAE AGI从A组（组件组装）升级到B组（自指涉分形拓扑）
**预期结果**: 智能等级L3 → L3.5-L4，外部依赖80% → 10%
**安全级别**: 高（当前系统正在运行，需谨慎操作）

---

## 🎯 总体目标

### 核心升级目标

| 目标 | A组（当前） | B组（目标） | 改进幅度 |
|------|-----------|-----------|---------|
| **核心架构** | 组件组装（M1-M4） | 自指涉分形拓扑 | 质的飞跃 |
| **目标函数** | 固定 | 可质疑、可修改 | 质的飞跃 |
| **外部依赖** | ~80% | ~10% | -87.5% |
| **智能等级** | L3 | L3.5-L4 | +1级 |
| **自指涉性** | ❌ 否 | ✅ 是 | 质的飞跃 |
| **熵计算** | ❌ 负数bug | ✅ 0-1范围 | bug修复 |

---

## 📋 实施原则

### 1. 安全第一原则

- ✅ **备份优先**: 修改任何组件前必须备份
- ✅ **沙箱测试**: 所有新功能先在沙箱测试
- ✅ **灰度发布**: 逐步替换，而非全面重写
- ✅ **回滚机制**: 出问题可立即回滚

### 2. 最小化干扰原则

- ✅ **不中断服务**: 系统继续运行
- ✅ **向后兼容**: 新旧组件可共存
- ✅ **渐进式升级**: 分阶段实施

### 3. 可验证原则

- ✅ **AB测试**: 每个改动都有明确的测试指标
- ✅ **数据对比**: 详细记录A组基准数据
- ✅ **性能监控**: 实时监控升级效果

---

## 🗺️ 实施路线图（6个阶段）

### 阶段0：准备工作（1-2小时）

**目标**: 完成备份和基准测量

#### 任务清单

- [ ] 0.1 备份当前系统关键文件
  - [ ] `core/seed.py` → `core/seed.py.backup_A`
  - [ ] `core/self_modifying_engine.py` → `core/self_modifying_engine.py.backup_A`
  - [ ] `core/recursive_self_memory.py` → `core/recursive_self_memory.py.backup_A`
  - [ ] `AGI_Life_Engine.py` → `AGI_Life_Engine.py.backup_A`

- [ ] 0.2 记录A组基准指标
  - [ ] 当前运行时长
  - [ ] 当前Insight总数
  - [ ] 当前平均熵值
  - [ ] 当前外部依赖度

- [ ] 0.3 创建升级分支
  - [ ] Git分支: `feature/B-plan-upgrade`
  - [ ] 文档分支: `docs/B-plan-progress`

---

### 阶段1：创建核心模块（4-6小时）

**目标**: 实现自指涉分形拓扑的核心数学结构

#### 任务清单

- [ ] 1.1 创建 `core/fluid_intelligence.py`
  - [ ] 实现 `SelfReferentialFractalNet` 类
  - [ ] 实现 `FractalRecursiveBlock` 类
  - [ ] 实现 `GoalQuestionerActive` 类
  - [ ] 实现 `CuriosityPressureValve` 类

- [ ] 1.2 集成到现有架构
  - [ ] 创建适配器连接到 `TheSeed`
  - [ ] 创建适配器连接到 `EvolutionController`
  - [ ] 确保向后兼容

- [ ] 1.3 单元测试
  - [ ] 测试自指涉性质
  - [ ] 测试分形标度律
  - [ ] 测试目标演化

#### 关键代码结构

```python
# core/fluid_intelligence.py

class SelfReferentialFractalNet(nn.Module):
    """
    自指涉分形神经网络

    数学对应：Φ = f(Φ, x)
    """
    def __init__(self, state_dim=64):
        # 自指涉状态
        self.self_representation = nn.Parameter(...)
        # 分形递归块
        self.fractal_blocks = nn.ModuleList([...])
        # 目标质疑（Active模式）
        self.goal_questioner = GoalQuestionerActive(state_dim)
        # 好奇心压力阀
        self.curiosity_valve = CuriosityPressureValve(state_dim)

    def forward(self, x, return_meta=True):
        # 自指涉前向传播
        state = self.input_projection(x)
        self_awareness = self._compute_self_awareness(state)
        # 分形递归处理
        integrated = self._integrate_fractal_blocks(state, self_awareness)
        output = self.output_projection(integrated)
        # 元信息
        entropy = self._compute_entropy(output)
        goal_score = self.goal_questioner(integrated)
        return output, {...}


class GoalQuestionerActive(nn.Module):
    """
    目标质疑模块 - Active模式

    与A组区别：能真正修改目标函数
    """
    def __init__(self, state_dim):
        self.goal_representation = nn.Parameter(...)
        self.mode = 'active'  # 关键改动

    def forward(self, state):
        # 评估目标合理性
        question_score = self._evaluate_goal(state)
        return question_score

    def modify_goal(self, state):
        # 真正修改目标（A组只能suggest）
        with torch.enable_grad():
            goal_grad = torch.autograd.grad(...)
        with torch.no_grad():
            self.goal_representation += lr * goal_grad
```

---

### 阶段2：升级M2 GoalQuestioner（2-3小时）

**目标**: 将目标质疑从`建议模式`升级到`主动模式`

#### 任务清单

- [ ] 2.1 定位M2 GoalQuestioner
  - [ ] 查找当前M2实现位置
  - [ ] 分析当前`suggest_only`模式
  - [ ] 确认升级接口

- [ ] 2.2 实现Active模式
  - [ ] 添加`modify_goal()`方法
  - [ ] 添加目标梯度计算
  - [ ] 添加安全检查机制

- [ ] 2.3 集成到系统
  - [ ] 更新配置：`mode=suggest_only` → `mode=active`
  - [ ] 更新日志输出
  - [ ] 添加监控指标

#### 关键改动

```python
# 当前（A组）
class GoalQuestioner:
    def __init__(self, mode='suggest_only'):
        self.mode = mode  # 只能建议

# 升级后（B组）
class GoalQuestioner:
    def __init__(self, mode='active'):  # 关键改动
        self.mode = mode  # 可以真正修改

    def modify_goal(self, state):
        if self.mode == 'active':
            # 真正修改目标函数
            self._update_goal_representation(state)
        else:
            # 仅建议（保持向后兼容）
            return self._suggest_goal_change(state)
```

---

### 阶段3：修复熵计算bug（1-2小时）

**目标**: 修复熵值计算为负数的bug

#### 任务清单

- [ ] 3.1 定位熵计算代码
  - [ ] 搜索`entropy`计算位置
  - [ ] 分析负数产生原因
  - [ ] 确认影响范围

- [ ] 3.2 实现修复
  - [ ] 添加边界检查（确保≥0）
  - [ ] 添加数值稳定性处理
  - [ ] 添加异常处理

- [ ] 3.3 验证修复
  - [ ] 单元测试：熵值在[0, 1]范围
  - [ ] 集成测试：系统运行正常
  - [ ] 回归测试：不影响其他功能

#### 关键修复

```python
# 当前（有bug）
def compute_entropy(output):
    probs = F.softmax(output, dim=-1)
    entropy = -(probs * torch.log(probs)).sum(dim=-1)
    return entropy.mean()  # 可能返回负数

# 修复后
def compute_entropy(output):
    probs = F.softmax(output, dim=-1)
    # 添加小量防止log(0)
    log_probs = torch.log(probs + 1e-8)
    entropy = -(probs * log_probs).sum(dim=-1)
    # 确保非负
    entropy = torch.clamp(entropy, min=0.0, max=1.0)
    return entropy.mean()
```

---

### 阶段4：降低外部LLM依赖（6-8小时）

**目标**: 将外部依赖从80%降到10%

#### 任务清单

- [ ] 4.1 分析当前依赖
  - [ ] 统计LLM调用频率
  - [ ] 识别可替换的调用
  - [ ] 评估替换难度

- [ ] 4.2 实现本地推理
  - [ ] 用`SelfReferentialFractalNet`替换简单推理
  - [ ] 用`TheSeed`替换主动推理
  - [ ] LLM仅用于最终验证和复杂任务

- [ ] 4.3 优化调用策略
  - [ ] 添加缓存机制
  - [ ] 添加批处理
  - [ ] 添加降级策略

#### 关键改动

```python
# 当前（A组）
def decide_action(state, goal):
    # 直接调用外部LLM
    response = llm_service.chat(prompt)
    return parse_action(response)

# 升级后（B组）
def decide_action(state, goal):
    # 优先使用本地推理
    local_decision = fluid_internet_net(state)
    confidence = local_decision['confidence']

    if confidence > 0.8:
        # 高置信度：直接使用本地结果
        return local_decision['action']
    else:
        # 低置信度：调用LLM验证
        llm_response = llm_service.chat(prompt)
        return parse_action(llm_response)
```

---

### 阶段5：沙箱测试（4-6小时）

**目标**: 在沙箱环境验证B组系统

#### 任务清单

- [ ] 5.1 创建测试环境
  - [ ] 复制当前系统到沙箱
  - [ ] 隔离测试数据
  - [ ] 准备测试用例

- [ ] 5.2 功能测试
  - [ ] 测试自指涉性质
  - [ ] 测试目标修改能力
  - [ ] 测试熵计算正确性
  - [ ] 测试外部依赖降低

- [ ] 5.3 性能测试
  - [ ] 测试响应速度
  - [ ] 测试资源占用
  - [ ] 测试稳定性

- [ ] 5.4 AB对比测试
  - [ ] 相同任务对比A组和B组
  - [ ] 测量改进幅度
  - [ ] 确认达到预期目标

---

### 阶段6：生产部署（2-3小时）

**目标**: 将B组部署到生产环境

#### 任务清单

- [ ] 6.1 部署准备
  - [ ] 最终备份
  - [ ] 部署检查清单
  - [ ] 回滚预案

- [ ] 6.2 灰度发布
  - [ ] 先部署10%流量
  - [ ] 监控关键指标
  - [ ] 逐步扩大到100%

- [ ] 6.3 监控验证
  - [ ] 实时监控日志
  - [ ] 验证系统稳定性
  - [ ] 对比AB测试结果

- [ ] 6.4 文档更新
  - [ ] 更新系统文档
  - [ ] 更新AB测试报告
  - [ ] 记录经验教训

---

## 📊 AB测试指标

### 核心指标

| 指标 | A组基准 | B组目标 | 测量方法 |
|------|---------|---------|---------|
| **智能等级** | L3 | L3.5-L4 | 专家评估 |
| **外部依赖** | ~80% | ~10% | 日志分析 |
| **目标可修改性** | ❌ 否 | ✅ 是 | 功能测试 |
| **自指涉性** | ❌ 否 | ✅ 是 | 架构分析 |
| **熵计算准确性** | ❌ 负数 | ✅ 0-1 | 数值检查 |
| **Insight质量** | 4.4/5 | 4.8+/5 | 人工评分 |
| **系统稳定性** | 14h无崩溃 | >14h无崩溃 | 运行时长 |
| **响应速度** | 基准 | ≤基准 | 性能测试 |

---

## ⚠️ 风险与应对

### 风险1：系统不稳定

**概率**: 中
**影响**: 高
**应对**:
- ✅ 完整备份
- ✅ 沙箱测试
- ✅ 快速回滚机制

### 风险2：性能下降

**概率**: 中
**影响**: 中
**应对**:
- ✅ 性能基准测试
- ✅ 优化关键路径
- ✅ 降级策略

### 风险3：功能回归

**概率**: 低
**影响**: 中
**应对**:
- ✅ 向后兼容设计
- ✅ 完整回归测试
- ✅ 渐进式部署

---

## 📝 实施日志

### 2026-01-12

**21:40** - 创建实施计划文档
**21:45** - 开始阶段0：准备工作
**21:50** - 待更新...

---

## ✅ 检查清单

### 部署前

- [ ] 所有备份完成
- [ ] 沙箱测试通过
- [ ] AB测试基准确认
- [ ] 回滚预案就绪

### 部署中

- [ ] 按阶段逐步实施
- [ ] 每步完成后验证
- [ ] 记录所有改动
- [ ] 实时监控系统状态

### 部署后

- [ ] 系统稳定运行
- [ ] AB测试指标达到
- [ ] 文档更新完成
- [ ] 经验教训记录

---

## 🎯 成功标准

B方案升级成功，当且仅当：

1. ✅ 系统稳定运行超过A组基准时长（>14小时）
2. ✅ 外部依赖降低到10%以下
3. ✅ 目标函数可以真正修改（Active模式）
4. ✅ 熵值计算正确（0-1范围）
5. ✅ 自指涉性质可用（网络能观察自身）
6. ✅ 智能等级提升到L3.5或L4
7. ✅ Insight质量提升（>4.6/5.0）

---

**创建时间**: 2026-01-12 21:45
**创建者**: Claude Code (Sonnet 4.5)
**状态**: 待执行

---

*本路线图将根据实际执行情况动态更新*
