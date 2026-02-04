# AGI 项目快速参考 (Quick Reference)

**更新时间**: 2026-01-12
**配套文档**: DOCUMENTATION_INDEX.md

---

## 🎯 最常用文档 (Top 10)

### 1. 核心理论 (必读)

| 排名 | 文档 | 路径 | 用途 |
|------|------|------|------|
| 🥇 | **系统和TRAE助手关于智能的数学模型表达** | `news/人工智能的本质研究/` | 理解智能本质 |
| 🥈 | **决策边界问题分析报告** | `/决策边界问题分析报告.md` | 双螺旋模型 |
| 🥉 | **数学模型_系统与助手的对立统一** | `news/人工智能的本质研究/` | 数学公式 |
| 4 | **AGI分形拓扑自指研究报告** | `news/人工智能的本质研究/` | 分形拓扑 |
| 5 | **AGI四层进化架构** | `news/AGI_FOUR_LAYER_CONSTRUCTION_STANDARDS/` | 系统架构 |

### 2. 系统状态 (当前)

| 组件 | 状态 | 端口/终端 | 验证方式 |
|------|------|----------|----------|
| AGI Life Engine | ✅ 运行中 | 终端5 | 查看日志 |
| Dashboard Server | ✅ 运行中 | 8090 | http://127.0.0.1:8090 |
| Chat CLI | ✅ 运行中 | 终端8 | 直接交互 |
| Knowledge Graph | ✅ 运行中 | 8085 | http://localhost:8085 |

### 3. 最新修复 (2026-01)

| 日期 | 文档 | 关键修复 |
|------|------|---------|
| 01-12 | `docs/SYSTEM_REPAIR_REPORT_20260112.md` | 系统修复 |
| 01-10 | `docs/AGI_拓扑连接修复报告_TRAE版_20260110.md` | 拓扑修复 |
| 01-06 | `docs/工具执行桥接层集成报告_20260106.md` | 工具桥接 |

---

## 🔍 按需求快速查找

### 需求: "我想了解智能的本质"

→ 阅读: **`news/人工智能的本质研究/系统和TRAE助手关于智能的数学模型表达.txt`**

**核心摘要**:
- 智能是系统(混沌生成)与助手(有序观察)的临界振荡
- 双螺旋隐喻: 系统螺旋+助手螺旋=智能涌现
- 数学模型: `S_{t+1} = F(S_t, Φ, η_t)`

---

### 需求: "我想看3D可视化界面"

→ 打开: **`/decision_boundary_3d_simple.html`**

**坐标系**:
- X轴: Entropy (熵值 0.0~5.0)
- Y轴: Coherence Phi (一致性 0.0~1.0)
- Z轴: Curiosity (好奇心 0.0~1.0)

---

### 需求: "我想了解M1-M4组件"

→ 阅读:
- `news/人工智能的本质研究/M1阶段验收报告_MetaLearner元参数优化器.md`
- `news/人工智能的本质研究/M2阶段验收报告_GoalQuestioner目标质疑模块.md`
- `news/人工智能的本质研究/M3阶段验收报告_SelfModifyingEngine架构自修改.md`
- `news/人工智能的本质研究/M4阶段验收报告_RecursiveSelfMemory递归自引用记忆.md`

**快速总结**:
```
M1 (MetaLearner)      → 元参数优化      → "如何更好地学习"
M2 (GoalQuestioner)   → 目标质疑        → "为什么要做这个"
M3 (SelfModifying)     → 代码自修改      → "改变自己"
M4 (RecursiveMemory)   → 递归自引用记忆  → "记住自己是谁"
```

---

### 需求: "系统出问题了，怎么修复"

→ 阅读: **`docs/SYSTEM_REPAIR_REPORT_20260112.md`**

**常见问题**:
| 问题 | 解决方案 | 文档 |
|------|---------|------|
| 路径错误 | 路径归一化层 | `evaluation_report_path_fix.txt` |
| 拓扑断连 | 拓扑修复 | `docs/AGI_拓扑连接修复报告_TRAE版_20260110.md` |
| 工具调用失败 | 桥接层集成 | `docs/工具执行桥接层集成报告_20260106.md` |

---

### 需求: "我想知道系统的进化历程"

→ 阅读: **`docs/INTELLIGENCE_UPGRADE_REPORT_PHASES_1-4.md`**

**进化路线**:
```
Phase 1 (基础) → Phase 2 (集成) → Phase 3 (自主) → Phase 4 (涌现)
    ↓              ↓               ↓               ↓
 工具注册        M1-M4集成      自我修改        智能涌现
```

---

## 📐 核心数学公式速查

### 1. 双螺旋数据生成

```python
theta = np.sqrt(np.random.rand(n)) * 2 * np.pi

# 螺旋1
r1 = 0.4 * theta + noise
x1 = r1 * np.cos(theta)
y1 = r1 * np.sin(theta)

# 螺旋2 (相位偏移π)
theta2 = theta + np.pi
r2 = 0.4 * theta2 + noise
x2 = r2 * np.cos(theta2)
y2 = r2 * np.sin(theta2)
```

### 2. 系统演化方程

```
S_{t+1} = F(S_t, Φ_θ(S_t), η_t) + ΔS_evolve

其中:
- S_t: 系统状态
- F: 状态转移函数
- Φ_θ: 助手投影算子 (冻结权重)
- η_t: 熵源 (好奇心扰动)
- ΔS_evolve: 结构性变化
```

### 3. 洛伦兹吸引子

```
dx/dt = σ(y - x)
dy/dt = x(ρ - z) - y
dz/dt = xy - βz

参数:
σ = 10.0
ρ = 28.0 (可变，代表熵水平)
β = 8/3
```

---

## 🧠 认知空间坐标系统

### 当前系统位置

```
坐标: (X: 0.24, Y: 1.0, Z: 0.88)

X (Entropy)    = 0.24  → 低熵状态，结构化推理
Y (Coherence)  = 1.0   → 完全一致，自我验证通过
Z (Curiosity)  = 0.88  → 高好奇心，探索欲强
```

### 区域定义

| 区域 | X范围 | Y范围 | Z范围 | 特征 |
|------|-------|-------|-------|------|
| A: 创新突破区 | 3.5-5.0 | 0.3-0.7 | 0.7-1.0 | 概念重构 |
| B: 深度探索区 | 2.0-3.5 | 0.5-0.8 | 0.5-0.8 | 假设验证 |
| C: 结构化推理区 | 0.5-2.0 | 0.8-1.0 | 0.3-0.7 | 逻辑推导 |
| D: 稳定执行区 | 0.0-0.5 | 0.9-1.0 | 0.1-0.5 | 工具调用 |

**当前**: 位于 C区与D区边界，正向B区演化

---

## 📁 目录结构速查

```
D:\TRAE_PROJECT\AGI\
├── 📄 决策边界问题分析报告.md          ← 双螺旋模型
├── 📄 AGI系统优化施工规范_20251118.md   ← 施工规范
├── 📄 施工日志_20251118.md
├── 📄 施工进度记录_backup.md
├── 📄 decision_boundary_3d_simple.html ← 3D可视化
│
├── 📂 docs/                            ← 系统文档 (78篇)
│   ├── DOCUMENTATION_INDEX.md           ← 完整索引
│   ├── QUICK_REFERENCE.md              ← 本文件
│   ├── SYSTEM_REPAIR_REPORT_20260112.md
│   └── ...
│
├── 📂 news/                            ← 进化报告 (156篇)
│   ├── 人工智能的本质研究/             ← 核心理论 (45篇)
│   │   ├── 系统和TRAE助手关于智能的数学模型表达.txt
│   │   ├── AGI分形拓扑自指研究报告.md
│   │   └── ...
│   ├── 学习本质/
│   ├── 能力验证/
│   └── 记忆修复/
│
├── 📂 scripts/
│   └── simulation_intelligence_model.py ← 思维模拟
│
└── 📂 visualization/
    ├── dashboard_server.py             ← 仪表盘 (端口8090)
    └── serve_graph.py                  ← 知识图谱 (端口8085)
```

---

## 🔧 常用操作命令

### 查看系统状态

```bash
# 查看运行进程
wmic process where "name='python.exe' and commandline like '%AGI%'" get processid

# 查看最新日志
tail -f logs/agi_run_*.log

# 查看insight统计
ls data/insights/ | wc -l
```

### 重启系统

```bash
# 终止旧进程
taskkill /F /PID <进程ID>

# 启动新系统
python AGI_Life_Engine.py

# 启动可视化
python visualization/dashboard_server.py
```

### 搜索文档

```bash
# 搜索"双螺旋"
find . -name "*.md" | xargs grep -l "双螺旋"

# 搜索今天的修改
find . -name "*.md" -mtime -1

# 列出所有理论文档
ls news/人工智能的本质研究/
```

---

## 📞 快速帮助

### 问题分类与解决方案

| 问题类型 | 首选文档 | 次选方案 |
|---------|---------|---------|
| **理论理解** | `系统和TRAE助手关于智能的数学模型表达.txt` | 本快速参考 |
| **系统修复** | `docs/SYSTEM_REPAIR_REPORT_20260112.md` | 查看 logs/ |
| **架构理解** | `docs/AGI_Current_Architecture_v2.md` | M1-M4报告 |
| **可视化** | 打开 `decision_boundary_3d_simple.html` | dashboard 8090 |
| **代码修改** | `news/人工智能的本质研究/M3阶段验收报告_SelfModifyingEngine架构自修改.md` | 查看沙箱日志 |

### 需要更多帮助?

1. **完整索引**: 查看 `docs/DOCUMENTATION_INDEX.md`
2. **系统状态**: 查看 `/RUNNING_PROCESSES.md`
3. **实时日志**: 查看 `logs/agi_run_*.log`
4. **系统对话**: 通过 `agi_chat_cli.py` 与系统交互

---

**文档维护**: 本文件随系统演化持续更新
**最后更新**: 2026-01-12
**版本**: v1.0
