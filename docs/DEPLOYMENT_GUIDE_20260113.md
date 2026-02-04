# B方案生产部署指南

**版本**: v1.0
**日期**: 2026-01-13
**状态**: ✅ 实践任务已完成

---

## 📋 实践任务完成清单

### ✅ 已完成的优化

1. **熵值计算优化** (P1问题)
   - 文件：`core/fractal_intelligence.py`
   - 改进：添加temperature参数（默认2.0）
   - 效果：支持更好的熵值控制

2. **生产环境配置系统**
   - 文件：`config/production_config.py`
   - 功能：环境配置、监控配置、告警配置
   - 支持：Production/Staging/Development

3. **监控系统**
   - 文件：`monitoring/fractal_monitor.py`
   - 功能：指标收集、实时统计、告警、仪表板
   - 特性：后台监控线程、自动导出、装饰器支持

4. **灰度发布脚本**
   - 文件：`scripts/gradual_rollout.py`
   - 功能：自动化灰度发布、自动回滚、健康检查
   - 支持：10%->50%->100%阶段化发布

---

## 🚀 快速部署步骤

### 步骤1: 准备配置（5分钟）

```bash
# 创建生产配置
python config/production_config.py

# 检查配置文件
cat config/production_config.json
```

### 步骤2: 启动监控（1分钟）

```python
from config.production_config import get_production_config
from monitoring.fractal_monitor import get_monitor

config = get_production_config()
monitor = get_monitor(config)
monitor.start()

# 监控后台运行
```

### 步骤3: 快速测试灰度（5分钟）

```bash
# 10%流量，5分钟测试
python scripts/gradual_rollout.py --percentage 10 --duration 5

# 检查结果
cat monitoring/rollout_log.json
```

### 步骤4: 查看监控仪表板

```python
from monitoring.fractal_monitor import get_monitor

monitor = get_monitor()
monitor.print_dashboard()
```

### 步骤5: 完整灰度发布（可选）

```bash
# 执行完整流程：10%->50%->100%，每阶段60分钟
python scripts/gradual_rollout.py --full
```

---

## 📊 监控指标说明

### 关键指标

| 指标 | 阈值 | 说明 |
|------|------|------|
| **响应时间P95** | <100ms | 95%请求的响应时间 |
| **错误率** | <1% | 错误请求比例 |
| **外部依赖率** | <20% | 需要外部LLM验证的比例 |
| **平均置信度** | >0.6 | 决策置信度 |
| **熵值** | 0.3-0.7 | 认知熵（探索程度） |

### 告警规则

- ⚠️ 高延迟：P95 > 100ms
- 🚨 高错误率：错误率 > 5%
- ⚠️ 高外部依赖：外部依赖 > 30%

---

## 🔧 回滚方案

### 快速回滚（命令）

```python
from core.fractal_adapter import create_fractal_seed_adapter, IntelligenceMode

# 切换回A组
adapter = create_fractal_seed_adapter(mode="GROUP_B")
adapter.set_mode(IntelligenceMode.GROUP_A)
```

### 恢复备份

```bash
# 恢复原始文件
mv core/seed.py.backup_A_20260112_214953 core/seed.py
mv core/self_modifying_engine.py.backup_A_20260112_214959 core/self_modifying_engine.py
mv core/recursive_self_memory.py.backup_A_20260112_215008 core/recursive_self_memory.py
mv AGI_Life_Engine.py.backup_A_20260112_215008 AGI_Life_Engine.py
```

---

## 📁 新增文件清单

### 优化代码
1. `core/fractal_intelligence.py` - 已优化熵计算

### 配置文件
2. `config/production_config.py` - 生产环境配置
3. `config/production_config.json` - 配置实例

### 监控系统
4. `monitoring/fractal_monitor.py` - 监控主系统
5. `monitoring/metrics_*.json` - 指标导出（自动生成）
6. `monitoring/rollout_log.json` - 发布日志（自动生成）

### 脚本工具
7. `scripts/gradual_rollout.py` - 灰度发布脚本

### 文档
8. `docs/DEPLOYMENT_GUIDE_20260113.md` - 本文档

---

## ✅ 验证清单

部署后请验证：

- [ ] 生产配置文件已创建
- [ ] 监控系统正常启动
- [ ] 快速灰度测试通过（10%流量）
- [ ] 监控仪表板正常显示
- [ ] 响应时间 <100ms
- [ ] 外部依赖 <20%
- [ ] 错误率 <1%
- [ ] 回滚方案可用

---

## 🎯 成功标准

灰度发布成功的标志：

1. ✅ 10%阶段稳定运行1小时
2. ✅ 扩大到50%继续稳定
3. ✅ 扩大到100%完全稳定
4. ✅ 外部依赖降低到<20%
5. ✅ 性能指标达标
6. ✅ 无严重错误

---

## 📞 问题排查

### 问题1: 监控系统不工作

```bash
# 检查日志
cat logs/fractal_production.log

# 重新初始化
python monitoring/fractal_monitor.py
```

### 问题2: 灰度发布失败

```bash
# 查看失败日志
cat monitoring/rollout_log.json

# 手动回滚
python -c "from core.fractal_adapter import create_fractal_seed_adapter, IntelligenceMode; adapter = create_fractal_seed_adapter(mode='GROUP_B'); adapter.set_mode(IntelligenceMode.GROUP_A)"
```

### 问题3: 性能下降

```bash
# 检查监控仪表板
python monitoring/fractal_monitor.py

# 如果响应时间>200ms，建议回滚
```

---

## 📚 相关文档

- 验收报告：`docs/B_PLAN_ACCEPTANCE_REPORT_20260112.md`
- 执行总结：`docs/B_PLAN_EXECUTIVE_SUMMARY_20260112.md`
- 问题清单：`docs/B_PLAN_ISSUES_AND_RECOMMENDATIONS_20260112.md`

---

**部署指南完成** - 系统已准备好生产部署！
