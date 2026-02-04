# 知识图谱加载错误修复报告

**日期**: 2026-01-15
**版本**: v1.0
**状态**: ✅ 修复完成

---

## 1. 问题描述

### 1.1 错误现象
```
Failed to load graph: 'edges'
```

系统日志显示 `WorldModel` 无法预测（no sufficient data），因为知识图谱无法正确加载。

### 1.2 根本原因分析

| 因素 | 详情 |
|------|------|
| **NetworkX 版本** | 3.6.1 |
| **问题** | NetworkX 3.x 的 `node_link_graph()` 默认期望 `edges` 键 |
| **数据格式** | 历史备份文件使用 `links` 键（NetworkX 2.x 默认格式）|
| **兼容性** | 新版本函数不向后兼容旧数据格式 |

### 1.3 数据丢失事件

在调试过程中发现 AGI 系统运行时覆盖了恢复的数据：

| 时间 | 事件 | 文件大小 |
|------|------|----------|
| 18:32:26 | 创建 pre_merge 备份 | 1.2 MB |
| 18:47:xx | 从备份恢复完整数据 | 184 MB |
| 18:53:20 | AGI 系统覆盖为新空图 | 3.5 KB |

---

## 2. 修复方案

### 2.1 代码修改

**文件**: `core/knowledge_graph.py`

#### 修复 `_load_graph()` 方法
```python
def _load_graph(self):
    if os.path.exists(self.graph_file):
        try:
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # [FIX 2026-01-15] NetworkX 3.x 默认期望 'edges' 键，但旧数据使用 'links'
                # 自动检测并使用正确的参数
                edges_key = 'edges' if 'edges' in data else 'links'
                self.graph = nx.node_link_graph(data, edges=edges_key)
                print(f"   [KnowledgeGraph] Loaded {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        except Exception as e:
            print(f"Failed to load graph: {e}")
            self.graph = nx.DiGraph()
```

#### 修复 `save_graph()` 方法
```python
def save_graph(self):
    """带文件锁的安全保存机制"""
    # [FIX 2026-01-15] 使用 edges='links' 保持与历史数据格式一致
    data = nx.node_link_data(self.graph, edges='links')
    # ... 其余代码不变
```

### 2.2 数据恢复

从 `arch_graph_backup_full.json` 恢复完整知识图谱：
```powershell
Copy-Item "arch_graph_backup_full.json" "arch_graph.json" -Force
```

---

## 3. 验证结果

```
   [KnowledgeGraph] Loaded 77017 nodes, 119340 edges
Test complete!
```

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 节点数 | 0 (加载失败) | 77,017 |
| 边数 | 0 | 119,340 |
| 文件大小 | 3.5 KB | 184 MB |
| 加载状态 | ❌ KeyError | ✅ 成功 |

---

## 4. 技术细节

### 4.1 NetworkX 版本变更

NetworkX 从 2.x 升级到 3.x 时，`node_link_data()` 和 `node_link_graph()` 的默认参数发生了变化：

| 版本 | 默认边键名 |
|------|-----------|
| NetworkX 2.x | `links` |
| NetworkX 3.x | `edges` |

### 4.2 兼容性策略

修复后的代码自动检测数据格式：
1. 如果 JSON 包含 `edges` 键 → 使用 `edges='edges'`
2. 如果 JSON 包含 `links` 键 → 使用 `edges='links'`
3. 保存时统一使用 `links` 格式以保持一致性

---

## 5. 预防措施

### 5.1 建议

1. **定期备份**: 每次启动前自动备份知识图谱
2. **版本检测**: 在日志中记录 NetworkX 版本
3. **数据验证**: 加载后验证节点/边数量是否合理

### 5.2 备份文件列表

```
data/knowledge/
├── arch_graph.json                    # 当前工作文件 (184 MB)
├── arch_graph_backup_full.json        # 完整备份 (184 MB)
└── arch_graph_pre_merge_20260115.json # 合并前备份 (1.2 MB)
```

---

## 6. 相关问题状态

| 问题 | 状态 | 备注 |
|------|------|------|
| M1 MetaLearner Enum导入 | ✅ 已修复 | 添加 `from enum import Enum` |
| BridgeAutoRepair | ✅ 已激活 | 系统运行时自动激活 |
| 知识图谱加载错误 | ✅ 已修复 | 本报告 |
| WorldModel 预测失败 | ⚠️ 待验证 | 依赖知识图谱加载 |

---

**修复人**: GitHub Copilot
**验证时间**: 2026-01-15 18:55
