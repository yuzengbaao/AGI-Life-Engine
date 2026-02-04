"""
文档索引生成器
生成工作区文档的结构化索引
"""
import os
import json
from datetime import datetime
from pathlib import Path

def generate_document_index():
    project_root = Path('D:/TRAE_PROJECT/AGI')
    
    # 读取JSON索引
    with open(project_root / 'data/document_index.json', 'r', encoding='utf-8') as f:
        index = json.load(f)

    # 生成Markdown索引
    lines = []
    lines.append('# 📚 AGI 工作区文档索引')
    lines.append('')
    lines.append(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append(f'**总文档数**: {index["total_docs"]}')
    lines.append(f'**分类数**: {len(index["categories"])}')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 📊 分类统计')
    lines.append('')
    lines.append('| 分类 | 文档数 |')
    lines.append('|------|--------|')
    
    # 按数量排序
    sorted_cats = sorted(index['categories'].items(), key=lambda x: x[1], reverse=True)
    for cat, count in sorted_cats:
        lines.append(f'| {cat} | {count} |')
    
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 🔑 核心文档（根目录）')
    lines.append('')
    lines.append('以下为项目根目录的重要文档，按修改时间倒序排列：')
    lines.append('')
    lines.append('| 文档名 | 大小(KB) | 修改日期 |')
    lines.append('|--------|----------|----------|')
    
    # 获取根目录文档，按修改日期排序
    root_docs = index['documents'].get('根目录', [])
    root_docs_sorted = sorted(root_docs, key=lambda x: x['modified'], reverse=True)
    
    # 只显示前100个
    for doc in root_docs_sorted[:100]:
        name = doc['name']
        path = doc['path']
        size = doc['size_kb']
        date = doc['modified']
        lines.append(f'| [{name}]({path}) | {size} | {date} |')
    
    if len(root_docs) > 100:
        lines.append(f'')
        lines.append(f'*...还有 {len(root_docs) - 100} 个文档*')
    
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 📁 重要子目录文档')
    lines.append('')
    
    # docs目录
    docs_count = index['categories'].get('docs', 0)
    lines.append(f'### docs/ 目录 ({docs_count} 个)')
    lines.append('')
    lines.append('| 文档名 | 大小(KB) | 修改日期 |')
    lines.append('|--------|----------|----------|')
    
    docs_dir = index['documents'].get('docs', [])
    docs_sorted = sorted(docs_dir, key=lambda x: x['modified'], reverse=True)
    for doc in docs_sorted[:30]:
        name = doc['name']
        path = doc['path']
        size = doc['size_kb']
        date = doc['modified']
        lines.append(f'| [{name}]({path}) | {size} | {date} |')
    
    # core目录
    lines.append('')
    lines.append('### core/ 目录')
    lines.append('')
    lines.append('| 文档名 | 路径 |')
    lines.append('|--------|------|')
    
    core_docs = index['documents'].get('core', [])
    for doc in core_docs:
        name = doc['name']
        path = doc['path']
        lines.append(f'| [{name}]({path}) | {path} |')
    
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 🔍 快速查找命令')
    lines.append('')
    lines.append('### 按关键词搜索文档：')
    lines.append('```powershell')
    lines.append('# 搜索包含特定关键词的文档标题')
    lines.append('Get-ChildItem -Path D:\\TRAE_PROJECT\\AGI -Include *.md -Recurse | Where-Object { $_.Name -match "关键词" }')
    lines.append('')
    lines.append('# 在文档内容中搜索')
    lines.append('Select-String -Path "D:\\TRAE_PROJECT\\AGI\\*.md" -Pattern "搜索词"')
    lines.append('```')
    lines.append('')
    lines.append('### 读取文档索引：')
    lines.append('```python')
    lines.append('import json')
    lines.append("with open('data/document_index.json', 'r', encoding='utf-8') as f:")
    lines.append('    index = json.load(f)')
    lines.append('# 获取所有分类')
    lines.append("print(index['categories'].keys())")
    lines.append('# 获取某分类下的文档')
    lines.append("print(index['documents']['根目录'])")
    lines.append('```')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 📋 文档分类说明')
    lines.append('')
    lines.append('| 分类 | 说明 |')
    lines.append('|------|------|')
    lines.append('| 根目录 | 项目主要文档、报告、指南 |')
    lines.append('| data | 系统生成的insight、记忆、日志 |')
    lines.append('| docs | 技术文档、API参考、使用指南 |')
    lines.append('| knowledge_base | 知识库文档 |')
    lines.append('| news | 新闻、更新日志 |')
    lines.append('| archive | 历史归档文档 |')
    lines.append('| core | 核心模块文档 |')
    lines.append('| backbag | 备份文档包 |')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('*此索引由系统自动生成，JSON格式索引位于 `data/document_index.json`*')
    
    # 保存Markdown索引
    md_content = '\n'.join(lines)
    with open(project_root / 'DOCUMENT_INDEX.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print('✅ Markdown索引已生成: DOCUMENT_INDEX.md')
    print(f'✅ JSON索引位于: data/document_index.json')
    print(f'\n总计索引 {index["total_docs"]} 个文档，分布在 {len(index["categories"])} 个目录中')

if __name__ == '__main__':
    generate_document_index()
