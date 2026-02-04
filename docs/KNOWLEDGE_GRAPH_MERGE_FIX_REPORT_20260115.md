# 知识图谱合并与加载修复报告

**日期**: 2026-01-15  
**版本**: v2.0  
**状态**: ✅ 修复完成并验证

---

## 1. 问题概述

### 1.1 发现的问题

| 问题编号 | 问题描述 | 严重程度 |
|----------|----------|----------|
| #1 | `Failed to load graph: 'edges'` - NetworkX 版本兼容性问题 | 🔴 严重 |
| #2 | 知识图谱数据未正确合并，使用旧备份覆盖了新数据 | 🟠 中等 |
| #3 | AGI运行时保存会覆盖外部合并的数据 | 🟠 中等 |

### 1.2 根因分析

#### 问题 #1: NetworkX 版本兼容性
```
NetworkX 2.x → 默认使用 'links' 键存储边
NetworkX 3.x → 默认使用 'edges' 键存储边
```

项目使用 **NetworkX 3.6.1**，但历史数据使用 `links` 键格式，导致 `nx.node_link_graph()` 默认查找 `edges` 键时失败。

#### 问题 #2: 数据未合并
初始修复时直接用旧备份 (`arch_graph_backup_full.json`) 覆盖了当前文件，导致 1月15日新生成的 1,993 个节点丢失。

#### 问题 #3: 运行时覆盖
AGI Life Engine 在后台运行时持续调用 `save_graph()`，会用内存中的图覆盖磁盘上的合并结果。

---

## 2. 修复方案

### 2.1 修复 #1: NetworkX 兼容性

**文件**: `core/knowledge_graph.py`

**修改 `_load_graph()` 方法**:
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

**修改 `save_graph()` 方法**:
```python
def save_graph(self):
    """带文件锁的安全保存机制"""
    # [FIX 2026-01-15] 使用 edges='links' 保持与历史数据格式一致
    data = nx.node_link_data(self.graph, edges='links')
    # ... 其余保存逻辑
```

### 2.2 修复 #2 & #3: 增量合并保存

**新增 `_merge_with_disk()` 方法**:
```python
def _merge_with_disk(self):
    """[FIX 2026-01-15] 保存前合并磁盘上可能被其他进程更新的数据"""
    if os.path.exists(self.graph_file):
        try:
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                disk_data = json.load(f)
            edges_key = 'edges' if 'edges' in disk_data else 'links'
            disk_graph = nx.node_link_graph(disk_data, edges=edges_key)
            # 合并：保留两边的所有节点和边
            self.graph = nx.compose(disk_graph, self.graph)
        except Exception as e:
            pass  # 如果无法读取磁盘文件，继续使用内存中的图
```

**修改 `save_graph()` 调用合并**:
```python
def save_graph(self):
    """带文件锁的安全保存机制"""
    # [FIX 2026-01-15] 保存前先合并磁盘上的数据，防止覆盖其他进程的更新
    self._merge_with_disk()
    # [FIX 2026-01-15] 使用 edges='links' 保持与历史数据格式一致
    data = nx.node_link_data(self.graph, edges='links')
    # ... 其余保存逻辑
```

---

## 3. 数据合并执行

### 3.1 合并策略

采用三方合并策略，确保所有数据都被保留：

```
旧备份(1月13日) + pre_merge(1月15日) + 当前运行时 → 合并结果
```

### 3.2 合并命令

```python
import networkx as nx

# 加载三个数据源
sources = [
    'data/knowledge/arch_graph_backup_full.json',      # 旧备份
    'data/knowledge/arch_graph_pre_merge_20260115_183226.json',  # 今日新数据
    'data/knowledge/arch_graph.json',                  # 当前运行时
]

merged = nx.DiGraph()
for path in sources:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    key = 'links' if 'links' in data else 'edges'
    g = nx.node_link_graph(data, edges=key)
    merged = nx.compose(merged, g)

# 保存合并结果
merged_data = nx.node_link_data(merged, edges='links')
with open('data/knowledge/arch_graph.json', 'w', encoding='utf-8') as f:
    json.dump(merged_data, f, ensure_ascii=False)
```

### 3.3 合并结果

| 数据来源 | 节点数 | 说明 |
|----------|--------|------|
| 旧备份 (1月13日) | 77,017 | 历史积累数据 |
| Pre-merge (1月15日) | 1,993 | 今日18:32前生成 |
| 当前运行时 | 77,028 | AGI运行时新增 |
| **合并后总计** | **79,021** | 去重后合并 |

| 指标 | 最终值 |
|------|--------|
| 总节点数 | 79,021 |
| 总边数 | 120,926 |
| 今日(20260115)节点 | 1,587 |
| 文件大小 | 177.96 MB |

---

## 4. 验证结果

### 4.1 加载验证

```
验证AGI系统加载合并后的知识图谱...
   [KnowledgeGraph] Loaded 79021 nodes, 120926 edges
今日节点数: 1587
✅ 合并成功！新增知识已可被系统调用!
```

### 4.2 可视化验证

- **Knowledge Graph Server**: http://localhost:8085
- **Dashboard Server V2**: http://localhost:8090

可视化界面可正常显示合并后的完整知识图谱。

### 4.3 系统集成验证

AGI Life Engine 启动时正确加载了合并后的数据：
```
[KnowledgeGraph] Loaded 79021 nodes, 120926 edges
[System] 🧠 NeuroSymbolic Bridge (Semantic Drift Detection) Online.
[Bridge] Hydrating from Knowledge Graph (79021 nodes)...
```

---

## 5. 文件变更清单

### 5.1 修改的文件

| 文件路径 | 修改内容 |
|----------|----------|
| `core/knowledge_graph.py` | 添加 `_merge_with_disk()` 方法；修复 `_load_graph()` 兼容性；修复 `save_graph()` 格式 |

### 5.2 数据文件状态

| 文件 | 大小 | 时间 | 用途 |
|------|------|------|------|
| `arch_graph.json` | 177.96 MB | 19:12 | 当前工作文件（合并后） |
| `arch_graph_backup_full.json` | 184.24 MB | 1月13日 | 完整备份 |
| `arch_graph_pre_merge_20260115.json` | 1.15 MB | 18:32 | 合并前快照 |

---

## 6. 技术要点

### 6.1 NetworkX 版本兼容性

| 版本 | `node_link_data()` 默认 | `node_link_graph()` 默认 |
|------|-------------------------|--------------------------|
| 2.x | `links` | 期望 `links` |
| 3.x | `edges` | 期望 `edges` |

**解决方案**: 使用 `edges` 参数显式指定键名：
```python
nx.node_link_graph(data, edges='links')  # 加载
nx.node_link_data(graph, edges='links')  # 保存
```

### 6.2 并发保存问题

当多个进程/实例访问同一个知识图谱文件时：
1. 使用文件锁 (`.lock` 文件) 防止并发写入
2. 保存前先合并磁盘数据，防止覆盖

### 6.3 数据格式标准

统一使用以下 JSON 格式：
```json
{
  "directed": true,
  "multigraph": false,
  "graph": {},
  "nodes": [...],
  "links": [...]  // 使用 'links' 而非 'edges'
}
```

---

## 7. 相关问题修复状态

| 问题 | 状态 | 修复时间 |
|------|------|----------|
| M1 MetaLearner Enum导入错误 | ✅ 已修复 | 2026-01-15 18:35 |
| BridgeAutoRepair 未激活 | ✅ 已激活 | 系统运行时自动 |
| 知识图谱加载错误 ('edges') | ✅ 已修复 | 2026-01-15 18:55 |
| 数据合并丢失 | ✅ 已修复 | 2026-01-15 19:12 |
| 运行时覆盖问题 | ✅ 已修复 | 2026-01-15 19:10 |

---

## 8. 后续建议

### 8.1 预防措施

1. **定期备份**: 建议每日自动备份知识图谱
2. **版本记录**: 在图的 `graph` 属性中记录版本信息
3. **合并日志**: 记录每次合并操作的来源和时间

### 8.2 监控建议

```python
# 启动时打印图状态
print(f"[KnowledgeGraph] Loaded {nodes} nodes, {edges} edges")
print(f"[KnowledgeGraph] Today's nodes: {today_count}")
```

---

**修复人**: GitHub Copilot  
**验证人**: 用户  
**完成时间**: 2026-01-15 19:15
