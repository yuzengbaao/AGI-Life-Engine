# 数据处理工具

一个功能强大的 Python 工具，用于处理、验证和转换数据文件。

## 功能特性

- 从 CSV/Excel 文件读取数据
- 数据验证和清洗
- 数据转换和聚合
- 生成报告（Excel/PDF）
- 命令行界面
- 配置文件支持
- 完善的日志记录

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python main.py --input data.csv --output result.xlsx
```

### 使用配置文件

```bash
python main.py --config config.yaml --input data.csv
```

### 生成 PDF 报告

```bash
python main.py --input data.csv --output report.pdf --format pdf
```

## 配置说明

创建 `config.yaml` 文件：

```yaml
input:
  encoding: utf-8        # 输入文件编码
  sheet_name: 0          # Excel 工作表名称或索引

output:
  format: xlsx           # 输出格式
  include_index: false   # 是否包含索引列

processing:
  remove_duplicates: true  # 是否移除重复数据
  fill_missing: forward    # 缺失值填充方式
```

## 项目结构

```
.
├── main.py              # 程序入口
├── config.py            # 配置管理
├── utils/
│   └── helpers.py       # 辅助工具函数
├── core/
│   ├── validator.py     # 数据验证
│   ├── processor.py     # 数据处理
│   └── reporter.py      # 报告生成
├── README.md            # 英文文档
├── README_CN.md         # 中文文档（本文件）
└── requirements.txt     # 依赖包列表
```

## 命令行参数

### 必需参数

- `--input` / `-i`: 输入文件路径（CSV 或 Excel）

### 可选参数

- `--output` / `-o`: 输出文件路径（默认：output/result.xlsx）
- `--config` / `-c`: 配置文件路径（YAML 格式）
- `--format` / `-f`: 输出格式（xlsx/csv/pdf，默认：xlsx）
- `--verbose` / `-v`: 启用详细日志输出
- `--mode`: 处理模式（fast/accurate/balanced，默认：balanced）

## 使用示例

### 示例 1: 基本数据处理

```bash
# 处理 CSV 文件，输出为 Excel
python main.py --input data.csv --output result.xlsx

# 处理 Excel 文件
python main.py --input data.xlsx --output processed.xlsx
```

### 示例 2: 使用配置文件

```bash
# 使用自定义配置
python main.py --config my_config.yaml --input data.csv
```

### 示例 3: 生成 PDF 报告

```bash
# 生成 PDF 格式的处理报告
python main.py --input data.csv --output report.pdf --format pdf
```

### 示例 4: 详细日志模式

```bash
# 启用详细日志输出，便于调试
python main.py --input data.csv --verbose
```

### 示例 5: 选择处理模式

```bash
# 快速模式（速度优先）
python main.py --input data.csv --mode fast

# 精确模式（质量优先）
python main.py --input data.csv --mode accurate

# 平衡模式（默认）
python main.py --input data.csv --mode balanced
```

## 配置文件详解

### 完整配置示例

```yaml
# 输入配置
input:
  encoding: utf-8        # 文件编码：utf-8, gbk, etc.
  sheet_name: 0          # Excel 工作表：索引（0）或名称（"Sheet1"）
  skip_rows: 0           # 跳过的行数

# 输出配置
output:
  format: xlsx           # 输出格式：xlsx, csv, pdf
  include_index: false   # 是否包含行索引
  date_format: "%Y-%m-%d"  # 日期格式

# 数据处理配置
processing:
  # 数据清洗
  remove_duplicates: true      # 移除重复行
  fill_missing: forward        # 缺失值填充：forward, backward, mean, median, drop
  normalize_text: true         # 文本规范化（去除多余空格等）

  # 数据转换
  convert_types: true          # 自动转换数据类型
  date_columns: ["date"]       # 日期列名列表

  # 数据聚合
  aggregate_by: []             # 分组列名列表
  aggregations:                # 聚合操作
    - column: "amount"
      function: "sum"          # sum, mean, median, count, etc.

# 日志配置
logging:
  level: INFO                  # 日志级别：DEBUG, INFO, WARNING, ERROR
  file: app.log                # 日志文件路径
```

## 输出说明

### Excel 输出格式

- 包含原始数据和处理后的数据
- 支持多个工作表
- 保留数据类型和格式

### CSV 输出格式

- 纯文本格式
- UTF-8 编码
- 逗号分隔

### PDF 报告格式

- 包含数据统计信息
- 处理步骤说明
- 数据质量报告
- 可视化图表（如果配置）

## 日志说明

程序运行时会输出详细的日志信息：

```
2024-02-06 10:00:00 - INFO - Starting application with mode: balanced
2024-02-06 10:00:01 - INFO - Input file: data.csv
2024-02-06 10:00:01 - INFO - Output directory: output/
2024-02-06 10:00:02 - DEBUG - Validating input file path...
2024-02-06 10:00:03 - INFO - Processing data...
2024-02-06 10:00:05 - INFO - Processing completed successfully.
```

## 错误处理

程序包含完善的错误处理机制：

### 常见错误及解决方案

1. **文件不存在**
   ```
   FileNotFoundError: data.csv not found
   ```
   解决：检查文件路径是否正确

2. **编码错误**
   ```
   UnicodeDecodeError: 'gbk' codec can't decode
   ```
   解决：在配置文件中指定正确的编码（如 `encoding: utf-8`）

3. **权限错误**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   解决：检查文件是否被其他程序占用，或是否有写入权限

## 性能优化建议

1. **大文件处理**
   - 使用 `--mode fast` 加快处理速度
   - 考虑分批处理数据

2. **内存优化**
   - 对于超大文件，使用 CSV 格式逐步处理
   - 调整 `chunk_size` 配置参数

3. **并行处理**
   - 某些操作支持并行处理
   - 在配置文件中启用 `parallel: true`

## 依赖包

主要依赖包：

- `pandas`: 数据处理
- `openpyxl`: Excel 文件读写
- `reportlab`: PDF 报告生成
- `pyyaml`: 配置文件解析

完整列表见 `requirements.txt`

## 开发说明

### 添加新的数据处理功能

1. 在 `core/processor.py` 中添加处理函数
2. 在配置文件中添加相应参数
3. 更新文档说明

### 添加新的验证规则

1. 在 `core/validator.py` 中添加验证函数
2. 在 `main.py` 中调用验证
3. 处理验证结果

## 常见问题 FAQ

**Q: 支持哪些文件格式？**
A: 支持 CSV、Excel（.xlsx, .xls）格式，输出支持 Excel、CSV、PDF。

**Q: 如何处理大文件？**
A: 使用 `--mode fast` 模式，或考虑分批处理数据。

**Q: 可以自定义数据处理逻辑吗？**
A: 可以，通过修改 `core/processor.py` 中的处理函数实现。

**Q: 输出的 PDF 报告包含哪些内容？**
A: 包含数据统计、处理步骤、质量报告和可视化图表。

**Q: 如何查看详细日志？**
A: 使用 `--verbose` 参数启用详细日志输出。

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**最后更新**: 2026-02-06
**版本**: 1.0.0
