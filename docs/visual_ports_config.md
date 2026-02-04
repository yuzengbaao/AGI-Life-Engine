# AGI 可视化服务端口配置文档

**更新日期**: 2026-01-20
**用途**: 记录可视化服务的实际端口配置

---

## 📊 服务端口列表

| 服务名称 | 进程文件 | PID | 默认端口 | 当前端口 | 访问地址 | 状态 |
|---------|---------|-----|----------|----------|----------|------|
| Dashboard V2 | visualization/dashboard_server_v2.py | 22312 | 8090 | 8090 | http://127.0.0.1:8090 | ✅ 运行中 |
| Graph Server | visualization/serve_graph.py | 22444 | 8085 | 8085 | http://localhost:8085 | ✅ 运行中 |

---

## 🔧 配置详情

### Dashboard V2 (FastAPI)
- **端口来源**: 环境变量 `AGI_DASHBOARD_V2_PORT`
- **默认值**: `8090`
- **主机来源**: 环境变量 `AGI_DASHBOARD_V2_HOST`
- **默认值**: `127.0.0.1`
- **修改方式**:
  ```bash
  # Windows PowerShell
  $env:AGI_DASHBOARD_V2_PORT="9000"
  $env:AGI_DASHBOARD_V2_HOST="0.0.0.0"

  # Windows CMD
  set AGI_DASHBOARD_V2_PORT=9000
  set AGI_DASHBOARD_V2_HOST=0.0.0.0

  # Linux/Mac
  export AGI_DASHBOARD_V2_PORT=9000
  export AGI_DASHBOARD_V2_HOST=0.0.0.0
  ```

### Graph Server (HTTP Server)
- **端口来源**: 硬编码在 `serve_graph.py`
- **当前值**: `8085`（第11行：`PORT = 8085`）
- **监听范围**: `0.0.0.0`（所有接口）
- **修改方式**: 编辑 `visualization/serve_graph.py` 第11行

---

## 🌐 访问端点

### Dashboard V2 端点
- 主页: http://127.0.0.1:8090/
- API文档: http://127.0.0.1:8090/docs
- 认知报告: http://127.0.0.1:8090/cognitive_report
- 架构图数据: http://127.0.0.1:8090/api/arch_graph
- 拓扑图数据: http://127.0.0.1:8090/api/topology_graph
- 认知报告数据: http://127.0.0.1:8090/api/cognitive_report_data

### Graph Server 端点
- 知识图谱可视化: http://localhost:8085/
- 图数据API: http://localhost:8085/api/graph

---

## 🔍 端口检查命令

### Windows
```cmd
netstat -ano | findstr ":8090"
netstat -ano | findstr ":8085"
```

### PowerShell
```powershell
Get-NetTCPConnection -State Listen -LocalPort 8090,8085
```

### Linux/Mac
```bash
lsof -i :8090
lsof -i :8085
```

---

## ⚠️ 常见问题

### Q: 为什么端口不是8000和5000？
A: 3D拓扑图中显示的是逻辑端口，实际端口配置如下：
- Dashboard: 8090（不是8000）
- Graph: 8085（不是5000）

### Q: 如何修改端口？
A: 参考上面的"配置详情"部分。Dashboard通过环境变量修改，Graph需要编辑源文件。

### Q: 端口冲突怎么办？
A: 使用以下命令检查端口占用：
```cmd
netstat -ano | findstr ":8090"
taskkill /PID <进程ID> /F
```

---

## 📝 更新历史

- **2026-01-20**: 初始文档，确认实际端口配置（8090, 8085）
