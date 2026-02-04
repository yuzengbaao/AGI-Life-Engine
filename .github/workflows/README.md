# GitHub Actions CI/CD 配置

本目录包含AGI系统的自动化测试和CI/CD配置。

## 工作流文件

### `tests.yml` - 主测试工作流

包含以下测试任务：

#### 1. 单元测试 (unit-tests)
- **触发条件**: push到main/develop分支，PR
- **测试文件**:
  - `tests/test_memory_lifecycle_manager.py`
  - `tests/test_tool_call_cache.py`
  - `tests/test_dynamic_recursion_limiter.py`
- **覆盖率要求**: ≥70%
- **输出**: HTML覆盖率报告 + Codecov上传

#### 2. 集成测试 (integration-tests)
- **触发条件**: push到main/develop分支，PR
- **测试文件**:
  - `tests/test_integration_memory_and_cache.py`
  - `tests/test_integration_decision_flow.py`
  - `tests/test_integration_tool_execution.py`
- **输出**: Codecov上传

#### 3. 性能基准测试 (benchmark)
- **触发条件**: 所有push和PR
- **迭代次数**: 1000次
- **输出**: JSON基准报告 + PR评论

#### 4. 稳定性测试 (stability-test-short)
- **测试时长**: 5分钟
- **监控**: 内存泄漏、性能回归
- **输出**: JSON稳定性报告

#### 5. 代码质量检查 (code-quality)
- **工具**: pylint
- **覆盖率检查**: ≥70%

#### 6. 安全扫描 (security-scan)
- **工具**: bandit
- **输出**: JSON安全报告

## 使用方式

### 自动触发

工作流会在以下情况自动运行：

1. **Push到main或develop分支**
   - 运行所有测试任务
   - 生成测试报告
   - 上传覆盖率到Codecov

2. **创建Pull Request**
   - 运行所有测试任务
   - 在PR中评论性能基准结果
   - 显示测试状态

3. **手动触发**
   - 在GitHub Actions页面点击"Run workflow"

### 本地验证

在推送前，建议本地运行相同的测试：

```bash
# 运行单元测试
pytest tests/test_memory_lifecycle_manager.py \
      tests/test_tool_call_cache.py \
      tests/test_dynamic_recursion_limiter.py \
      -v --cov=core --cov-report=term-missing

# 运行集成测试
pytest tests/test_integration_*.py -v

# 运行性能基准
python tests/benchmark_test.py --iterations 1000

# 运行稳定性测试（1分钟）
python tests/stability_test.py --duration 60
```

## 状态徽章

可以将以下徽章添加到README.md：

```markdown
[![Unit Tests](https://github.com/yourusername/AGI/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/AGI/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/yourusername/AGI/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/AGI)
```

## 配置说明

### 环境变量

如需自定义配置，可在GitHub仓库设置中添加以下Secrets：

- `CODECOV_TOKEN`: Codecov上传令牌（可选）

### 调整测试参数

修改工作流文件中的参数：

```yaml
# 修改单元测试覆盖率要求
--cov-fail-over=70  # 改为80或其他值

# 修改基准测试迭代次数
--iterations 1000  # 改为更多或更少

# 修改稳定性测试时长
--duration 300  # 改为其他秒数
```

## 测试报告

### 查看报告

1. **GitHub Actions**: 点击Actions标签，选择具体的工作流运行
2. **Artifacts**: 下载生成的报告文件
3. **Codecov**: 访问codecov.io查看详细覆盖率
4. **PR评论**: 性能基准结果会自动评论到PR

### 报告类型

| 报告类型 | 文件名 | 说明 |
|---------|--------|------|
| 单元测试覆盖率 | htmlcov/ | HTML格式覆盖率报告 |
| 性能基准 | benchmark_report_*.json | JSON格式性能数据 |
| 稳定性测试 | stability_test_report_*.json | JSON格式稳定性数据 |
| 安全扫描 | security_report.json | JSON格式安全扫描结果 |

## 故障排查

### 测试失败

如果测试失败，检查：

1. **依赖安装**: 确保所有依赖正确安装
2. **Python版本**: 使用Python 3.12
3. **环境变量**: 检查必要的Secrets和配置
4. **测试日志**: 查看GitHub Actions日志

### 覆盖率未上传

如果Codecov未收到覆盖率：

1. 检查`CODECOV_TOKEN`是否正确配置
2. 查看Codecov上传步骤的日志
3. 确保coverage.xml文件正确生成

## 最佳实践

1. **推送前本地测试**: 在推送前运行测试套件
2. **小步提交**: 频繁推送以便及早发现问题
3. **查看PR评论**: 关注自动化评论的性能基准结果
4. **定期更新依赖**: 保持测试依赖最新版本

## 扩展工作流

### 添加新的测试任务

在`tests.yml`中添加新的job：

```yaml
new-test-job:
  name: New Test
  runs-on: ubuntu-latest
  steps:
    - name: Checkout code
      uses: actions/checkout@v4
    - name: Run test
      run: pytest tests/new_test.py -v
```

### 添加定时任务

```yaml
scheduled-nightly:
  name: Nightly Tests
  runs-on: ubuntu-latest
  steps:
    - name: Run 24h stability test
      run: python tests/stability_test.py --duration 86400
```

触发条件：
```yaml
on:
  schedule:
    - cron: '0 0 * * *'  # 每天午夜运行
```

---

**配置版本**: v1.0
**最后更新**: 2026-02-04
