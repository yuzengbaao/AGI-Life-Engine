# 真实场景测试 - 终端命令参考

**创建日期**: 2026-01-14
**用途**: 在终端执行真实场景测试

---

## 🚀 快速开始（推荐）

### 1️⃣ 运行真实场景测试（最全面）

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_real_world_scenarios.py
```

**输出**: 5个真实场景的完整测试报告

---

### 2️⃣ 运行集成系统基础测试

```bash
cd "D:\TRAE_PROJECT\AGI"
python integrated_agi_system.py
```

**输出**: 5次端到端处理

---

### 3️⃣ 运行修复验证测试

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_fixes.py
```

**输出**: None字段、首次决策时间、熵归一化测试

---

### 4️⃣ 运行完整烟雾测试

```bash
cd "D:\TRAE_PROJECT\AGI"
python smoke_test.py
```

**输出**: 20项测试全面验证

---

### 5️⃣ 运行MVP快速验证

```bash
cd "D:\TRAE_PROJECT\AGI"
python quick_mvp_test_v2.py
```

**输出**: 100次决策质量验证

---

## 📊 高级测试命令

### 6️⃣ 保存测试结果到文件

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_real_world_scenarios.py > real_world_test_results.txt 2>&1
```

**查看结果**:
```bash
cat real_world_test_results.txt
# 或
type real_world_test_results.txt
```

---

### 7️⃣ 实时查看并保存输出

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_real_world_scenarios.py 2>&1 | tee real_world_live.log
```

**说明**: 同时显示在终端和保存到文件

---

### 8️⃣ 后台运行测试（长时间）

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_real_world_scenarios.py > test.log 2>&1 &
```

**查看实时日志**:
```bash
tail -f test.log
```

**停止查看**: 按 `Ctrl+C`

---

### 9️⃣ 只看关键信息（过滤警告）

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_real_world_scenarios.py 2>&1 | grep -v "FutureWarning\|pynvml\|UserWarning"
```

---

### 🔟 只看错误和失败

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_real_world_scenarios.py 2>&1 | grep -i "error\|fail\|exception"
```

---

## 📈 性能测试命令

### 1️⃣1️⃣ 性能基准测试（100次处理）

```bash
cd "D:\TRAE_PROJECT\AGI"
python -c "
from integrated_agi_system import IntegratedAGISystem, SystemInput
import time, numpy as np

system = IntegratedAGISystem()
times = []

for i in range(100):
    inp = SystemInput(
        visual={'frame': np.random.rand(480,640,3), 'timestamp': time.time()},
        audio={'chunk': np.random.randn(16000), 'sample_rate': 16000}
    )
    start = time.time()
    system.process(inp)
    times.append((time.time()-start)*1000)

print(f'100次处理统计:')
print(f'  平均: {sum(times)/len(times):.1f}ms')
print(f'  最小: {min(times):.1f}ms')
print(f'  最大: {max(times):.1f}ms')
print(f'  标准差: {(sum((x-sum(times)/len(times))**2 for x in times)/len(times))**0.5:.1f}ms')
" 2>&1 | grep -v "Warning\|pynvml"
```

---

### 1️⃣2️⃣ 长时间稳定性测试（1000次处理）

```bash
cd "D:\TRAE_PROJECT\AGI"
python -c "
from integrated_agi_system import IntegratedAGISystem, SystemInput
import time, numpy as np

system = IntegratedAGISystem()
print('开始1000次处理测试...')

start_time = time.time()
for i in range(1000):
    inp = SystemInput(
        visual={'frame': np.random.rand(480,640,3), 'timestamp': time.time()},
        audio={'chunk': np.random.randn(16000), 'sample_rate': 16000}
    )
    system.process(inp)
    if (i+1) % 100 == 0:
        print(f'完成 {i+1}/1000')

total_time = time.time() - start_time
print(f'\\n总耗时: {total_time:.1f}秒')
print(f'平均每次: {total_time*1000/1000:.1f}ms')
stats = system.get_statistics()
print(f'记忆数量: {stats[\"memory\"][\"total_memories\"]}')
" 2>&1 | grep -v "Warning\|pynvml"
```

---

## 🔍 对比测试命令

### 1️⃣3️⃣ 对比MVP vs 阶段2系统

```bash
cd "D:\TRAE_PROJECT\AGI"
python -c "
from decision_adapter_v2 import DecisionAdapterV2
from integrated_agi_system import IntegratedAGISystem, SystemInput
from mvp_utils import generate_test_scenario
import time, numpy as np

print('='*60)
print(' MVP vs 阶段2 系统对比测试')
print('='*60)

# MVP测试（仅L5决策层）
print('\\n[MVP测试] DecisionAdapterV2 (仅L5决策层)...')
mvp_adapter = DecisionAdapterV2(state_dim=64, action_dim=4)
mvp_times = []
for i in range(20):
    ctx = generate_test_scenario(i)
    start = time.time()
    mvp_adapter.decide(ctx)
    mvp_times.append((time.time()-start)*1000)

print(f'  平均响应: {sum(mvp_times)/len(mvp_times):.1f}ms')
print(f'  层次: 仅L5决策层')

# 阶段2测试（完整L1-L6）
print('\\n[阶段2测试] IntegratedAGISystem (完整L1-L6)...')
integrated_system = IntegratedAGISystem()
integrated_times = []
for i in range(20):
    inp = SystemInput(
        visual={'frame': np.random.rand(480,640,3), 'timestamp': time.time()},
        audio={'chunk': np.random.randn(16000), 'sample_rate': 16000}
    )
    start = time.time()
    integrated_system.process(inp)
    integrated_times.append((time.time()-start)*1000)

print(f'  平均响应: {sum(integrated_times)/len(integrated_times):.1f}ms')
print(f'  层次: L1-L6完整流程')

# 对比
print(f'\\n[对比] 响应时间:')
print(f'  MVP: {sum(mvp_times)/len(mvp_times):.1f}ms')
print(f'  阶段2: {sum(integrated_times)/len(integrated_times):.1f}ms')
print(f'  差异: {sum(integrated_times)/len(integrated_times) - sum(mvp_times)/len(mvp_times):+.1f}ms')
" 2>&1 | grep -v "Warning\|pynvml"
```

---

## 🛠️ 调试命令

### 1️⃣4️⃣ 查看详细日志（DEBUG模式）

```bash
cd "D:\TRAE_PROJECT\AGI"
python -c "
import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

from integrated_agi_system import IntegratedAGISystem, SystemInput
import numpy as np

system = IntegratedAGISystem()
inp = SystemInput(
    visual={'frame': np.random.rand(480,640,3), 'timestamp': 0},
    audio={'chunk': np.random.randn(16000), 'sample_rate': 16000}
)
print('\\n处理单次输入（DEBUG模式）...')
system.process(inp)
" 2>&1 | grep -v "FutureWarning\|pynvml" | head -100
```

---

### 1️⃣5️⃣ 测试特定场景

```bash
cd "D:\TRAE_PROJECT\AGI"
python -c "
from integrated_agi_system import IntegratedAGISystem, SystemInput
import numpy as np

system = IntegratedAGISystem()

# 测试空输入场景
print('[测试] 空输入场景...')
inp = SystemInput()
output = system.process(inp)
print(f'结果: {output.action_taken}')
print(f'成功: {output.result[\"success\"]}')
" 2>&1 | grep -v "Warning\|pynvml"
```

---

## 📋 完整测试流程（推荐顺序）

### 标准测试流程

```bash
# 步骤1: 快速功能验证
python test_fixes.py

# 步骤2: 系统集成测试
python integrated_agi_system.py

# 步骤3: 真实场景测试
python test_real_world_scenarios.py

# 步骤4: 完整烟雾测试
python smoke_test.py

# 步骤5: 性能压力测试
python -c "from integrated_agi_system import IntegratedAGISystem, SystemInput; import time, numpy as np; system = IntegratedAGISystem(); [system.process(SystemInput(visual={'frame': np.random.rand(480,640,3), 'timestamp': time.time()}, audio={'chunk': np.random.randn(16000), 'sample_rate': 16000})) for _ in range(100)]" 2>&1 | tail -10
```

---

## 💡 常用命令组合

### 保存结果并立即查看

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_real_world_scenarios.py 2>&1 | tee test_results.txt && cat test_results.txt
```

### 只看统计结果

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_real_world_scenarios.py 2>&1 | grep -A 20 "统计分析"
```

### 对比两次运行结果

```bash
cd "D:\TRAE_PROJECT\AGI"
python test_real_world_scenarios.py > run1.txt 2>&1
python test_real_world_scenarios.py > run2.txt 2>&1
diff run1.txt run2.txt
```

---

## 🔧 故障排查命令

### 检查导入是否正常

```bash
cd "D:\TRAE_PROJECT\AGI"
python -c "from integrated_agi_system import IntegratedAGISystem; print('导入成功')" 2>&1 | grep -v "Warning"
```

### 检查系统能否创建

```bash
cd "D:\TRAE_PROJECT\AGI"
python -c "from integrated_agi_system import IntegratedAGISystem; s = IntegratedAGISystem(); print('创建成功')" 2>&1 | grep -v "Warning" | tail -20
```

### 测试单次处理

```bash
cd "D:\TRAE_PROJECT\AGI"
python -c "
from integrated_agi_system import IntegratedAGISystem, SystemInput
import numpy as np
s = IntegratedAGISystem(enable_memory=False, enable_feedback=False)
out = s.process(SystemInput(visual={'frame': np.zeros((10,10,3)), 'timestamp': 0}))
print(f'成功: {out.result[\"success\"]}')
" 2>&1 | grep -v "Warning" | tail -10
```

---

## 📊 结果文件位置

运行测试后，结果保存在：

| 文件 | 内容 |
|------|------|
| `test_fixes_output.txt` | 修复验证结果 |
| `integrated_system_test.txt` | 集成系统测试结果 |
| `real_world_test_results.txt` | 真实场景测试结果 |
| `smoke_test_output.txt` | 烟雾测试结果 |
| `mvp_v2_output.txt` | MVP测试结果 |

---

## 🎯 推荐执行顺序

### 第一次测试（新用户）

```bash
# 1. 最简单：检查系统
python test_fixes.py

# 2. 中等复杂度：集成系统
python integrated_agi_system.py

# 3. 最全面：真实场景
python test_real_world_scenarios.py
```

### 完整验证（已熟悉）

```bash
# 运行所有测试
python test_fixes.py && python integrated_agi_system.py && python test_real_world_scenarios.py && python smoke_test.py
```

---

## ⚡ 快速参考

| 想要... | 运行命令 |
|---------|---------|
| 快速验证 | `python test_fixes.py` |
| 集成测试 | `python integrated_agi_system.py` |
| 真实场景 | `python test_real_world_scenarios.py` |
| 完整测试 | `python smoke_test.py` |
| MVP验证 | `python quick_mvp_test_v2.py` |

---

**更新日期**: 2026-01-14
**版本**: 1.0.0
