#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGI 拓扑连接自动复核脚本
========================

从 system_topology_3d.html 提取 67 条边，逐条验证代码证据。

使用方法:
    cd D:\\TRAE_PROJECT\\AGI
    python scripts/verify_topology_links.py

输出:
    - 控制台: 逐条验证结果
    - 文件: docs/AGI_拓扑连接验证结果_自动生成.md
"""

import re
import os
import ast
import sys
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from dataclasses import dataclass
from typing import Optional, List, Tuple
from datetime import datetime

# ============================================================
# 配置区
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
TOPOLOGY_FILE = PROJECT_ROOT / "workspace" / "system_topology_3d.html"
OUTPUT_FILE = PROJECT_ROOT / "docs" / "AGI_拓扑连接验证结果_自动生成.md"

# 代码证据映射: (source, target) -> (file_pattern, search_pattern, description)
# 用于精确验证特定连接
EVIDENCE_MAP = {
    # Layer 0: 入口层
    ("AGI_Life_Engine", "LLMService"): (
        "AGI_Life_Engine.py", r"self\.llm_service\s*=", "Engine初始化LLMService"
    ),
    ("AGI_Life_Engine", "GoalManager"): (
        "AGI_Life_Engine.py", r"self\.goal_manager\s*=", "Engine初始化GoalManager"
    ),
    ("AGI_Life_Engine", "PlannerAgent"): (
        "AGI_Life_Engine.py", r"self\.planner\s*=", "Engine初始化PlannerAgent"
    ),
    ("AGI_Life_Engine", "ExecutorAgent"): (
        "AGI_Life_Engine.py", r"self\.executor\s*=", "Engine初始化ExecutorAgent"
    ),
    ("AGI_Life_Engine", "CriticAgent"): (
        "AGI_Life_Engine.py", r"self\.critic\s*=", "Engine初始化CriticAgent"
    ),
    ("AGI_Life_Engine", "EvolutionController"): (
        "AGI_Life_Engine.py", r"self\.evolution_controller\s*=", "Engine初始化EvolutionController"
    ),
    ("AGI_Life_Engine", "BiologicalMemory"): (
        "AGI_Life_Engine.py", r"self\.biological_memory\s*=", "Engine初始化BiologicalMemory"
    ),
    ("AGI_Life_Engine", "PerceptionManager"): (
        "AGI_Life_Engine.py", r"self\.perception\s*=", "Engine初始化PerceptionManager"
    ),
    ("AGI_Life_Engine", "InsightValidator"): (
        "AGI_Life_Engine.py", r"self\.insight_validator\s*=", "Engine初始化InsightValidator"
    ),
    ("AGI_Life_Engine", "IntentDialogueBridge"): (
        "AGI_Life_Engine.py", r"self\.intent_bridge\s*=|get_intent_bridge", "Engine获取IntentDialogueBridge"
    ),
    
    # V-I-E Loop
    ("InsightValidator", "InsightIntegrator"): (
        "AGI_Life_Engine.py", r"integration_result\s*=\s*self\.insight_integrator\.integrate", "Engine在验证通过后调用Integrator"
    ),
    ("InsightIntegrator", "InsightEvaluator"): (
        "AGI_Life_Engine.py", r"self\.insight_evaluator\.record_call", "Engine在集成成功后记录到Evaluator"
    ),
    ("InsightIntegrator", "BiologicalMemory"): (
        "AGI_Life_Engine.py", r"self\.biological_memory\.internalize_items", "Engine在V-I-E链路中写入BiologicalMemory"
    ),
    ("InsightEvaluator", "AGI_Life_Engine"): (
        "AGI_Life_Engine.py", r"insight_evaluator\.generate_report", "Engine轮询Evaluator报告"
    ),
    
    # 组件协调
    ("ComponentCoordinator", "AGI_Life_Engine"): (
        "AGI_Life_Engine.py", r"self\.component_coordinator\s*=\s*ComponentCoordinator", "Engine初始化Coordinator"
    ),
    ("ComponentCoordinator", "SecurityManager"): (
        "agi_component_coordinator.py", r"security|SecurityManager", "Coordinator引用SecurityManager"
    ),
    ("SecurityManager", "ExecutorAgent"): (
        "security_framework.py", r"executor|validate|check", "SecurityManager检查执行"
    ),
    
    # ImmutableCore (概念性连接)
    ("ImmutableCore", "SecurityManager"): (
        "core/layered_identity.py", r"frozen|dataclass", "ImmutableCore是frozen dataclass（概念性）"
    ),
    ("ImmutableCore", "CriticAgent"): (
        "core/layered_identity.py", r"frozen|dataclass", "ImmutableCore是frozen dataclass（概念性）"
    ),
    
    # 桥接层
    ("ToolExecutionBridge", "ComponentCoordinator"): (
        "tool_execution_bridge.py", r"component_coordinator|ComponentCoordinator", "Bridge引用Coordinator"
    ),
    ("ToolExecutionBridge", "ExecutorAgent"): (
        "AGI_Life_Engine.py", r"tool_bridge", "Engine使用ToolBridge"
    ),
    ("ToolFactory", "ComponentCoordinator"): (
        "agi_tool_factory.py", r"component_coordinator|ComponentCoordinator", "Factory引用Coordinator"
    ),
    ("BridgeAutoRepair", "ToolExecutionBridge"): (
        "bridge_auto_repair.py", r"tool_execution_bridge|ToolExecutionBridge", "AutoRepair操作Bridge"
    ),
    ("BridgeAutoRepair", "ComponentCoordinator"): (
        "bridge_auto_repair.py", r"component_coordinator|ComponentCoordinator", "AutoRepair发布事件"
    ),
}

# ============================================================
# 数据结构
# ============================================================

@dataclass
class TopologyLink:
    """拓扑连接定义"""
    source: str
    target: str
    link_type: str  # data, control, event
    
@dataclass
class VerificationResult:
    """验证结果"""
    link: TopologyLink
    status: str  # ✅已实现, ⚠️部分实现, ❌未实现, 🔵概念性
    evidence_file: Optional[str] = None
    evidence_line: Optional[int] = None
    evidence_snippet: Optional[str] = None
    note: Optional[str] = None

# ============================================================
# 提取拓扑连接
# ============================================================

def extract_links_from_html(html_path: Path) -> List[TopologyLink]:
    """从拓扑HTML中提取所有连接"""
    content = html_path.read_text(encoding='utf-8')
    
    # 匹配 { source: "X", target: "Y", type: "Z" }
    pattern = r'\{\s*source:\s*"([^"]+)",\s*target:\s*"([^"]+)",\s*type:\s*"([^"]+)"'
    matches = re.findall(pattern, content)
    
    links = []
    for source, target, link_type in matches:
        links.append(TopologyLink(source=source, target=target, link_type=link_type))
    
    return links

# ============================================================
# 代码证据搜索
# ============================================================

def search_evidence(file_pattern: str, search_pattern: str) -> Tuple[bool, Optional[str], Optional[int], Optional[str]]:
    """
    在指定文件中搜索证据
    
    返回: (found, file_path, line_number, snippet)
    """
    # 构建可能的文件路径
    possible_paths = [
        PROJECT_ROOT / file_pattern,
        PROJECT_ROOT / "core" / file_pattern,
        PROJECT_ROOT / "core" / "agents" / file_pattern,
        PROJECT_ROOT / "core" / "memory" / file_pattern,
    ]
    
    for file_path in possible_paths:
        if file_path.exists():
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for i, line in enumerate(lines, 1):
                    if re.search(search_pattern, line, re.IGNORECASE):
                        # 获取上下文片段
                        start = max(0, i - 2)
                        end = min(len(lines), i + 1)
                        snippet = '\n'.join(lines[start:end])
                        
                        rel_path = file_path.relative_to(PROJECT_ROOT)
                        return True, str(rel_path), i, snippet
                        
            except Exception as e:
                pass
    
    return False, None, None, None

def verify_link(link: TopologyLink) -> VerificationResult:
    """验证单条连接"""
    key = (link.source, link.target)
    
    # 检查是否有预定义的证据映射
    if key in EVIDENCE_MAP:
        file_pattern, search_pattern, description = EVIDENCE_MAP[key]
        found, file_path, line_num, snippet = search_evidence(file_pattern, search_pattern)
        
        if found:
            # 检查是否是概念性连接
            if "概念性" in description or link.source == "ImmutableCore":
                status = "🔵概念性"
            else:
                status = "✅已实现"
            
            return VerificationResult(
                link=link,
                status=status,
                evidence_file=file_path,
                evidence_line=line_num,
                evidence_snippet=snippet,
                note=description
            )
        else:
            return VerificationResult(
                link=link,
                status="❌未实现",
                note=f"未找到证据: {file_pattern} 中的 {search_pattern}"
            )
    
    # 通用搜索: 在常见位置搜索 source 和 target 的关联
    generic_patterns = [
        (f"{link.target.lower()}", f"AGI_Life_Engine.py"),
        (f"self.{link.target.lower()}", "AGI_Life_Engine.py"),
        (link.target, "*.py"),
    ]
    
    # 尝试在 Engine 中找到 target 的初始化
    found, file_path, line_num, snippet = search_evidence(
        "AGI_Life_Engine.py", 
        rf"self\.{link.target.lower()}|{link.target}\("
    )
    
    if found:
        return VerificationResult(
            link=link,
            status="⚠️部分实现",
            evidence_file=file_path,
            evidence_line=line_num,
            evidence_snippet=snippet,
            note="通用搜索找到，需人工确认语义"
        )
    
    return VerificationResult(
        link=link,
        status="⚠️待验证",
        note="无预定义证据映射，需人工确认"
    )

# ============================================================
# 报告生成
# ============================================================

def generate_report(results: List[VerificationResult]) -> str:
    """生成Markdown报告"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 统计
    stats = {
        "✅已实现": 0,
        "⚠️部分实现": 0,
        "⚠️待验证": 0,
        "❌未实现": 0,
        "🔵概念性": 0,
    }
    for r in results:
        stats[r.status] = stats.get(r.status, 0) + 1
    
    report = f"""# AGI 拓扑连接验证结果（自动生成）

**生成时间**: {now}  
**脚本**: `scripts/verify_topology_links.py`  
**拓扑源**: `workspace/system_topology_3d.html`

---

## 统计摘要

| 状态 | 数量 | 百分比 |
|------|------|--------|
| ✅ 已实现 | {stats.get('✅已实现', 0)} | {stats.get('✅已实现', 0) / len(results) * 100:.1f}% |
| ⚠️ 部分实现 | {stats.get('⚠️部分实现', 0)} | {stats.get('⚠️部分实现', 0) / len(results) * 100:.1f}% |
| ⚠️ 待验证 | {stats.get('⚠️待验证', 0)} | {stats.get('⚠️待验证', 0) / len(results) * 100:.1f}% |
| ❌ 未实现 | {stats.get('❌未实现', 0)} | {stats.get('❌未实现', 0) / len(results) * 100:.1f}% |
| 🔵 概念性 | {stats.get('🔵概念性', 0)} | {stats.get('🔵概念性', 0) / len(results) * 100:.1f}% |
| **总计** | {len(results)} | 100% |

---

## 逐条验证结果

| # | 连接 | 类型 | 状态 | 代码证据 | 备注 |
|---|------|------|------|----------|------|
"""
    
    for i, r in enumerate(results, 1):
        link_str = f"`{r.link.source}` → `{r.link.target}`"
        evidence = ""
        if r.evidence_file and r.evidence_line:
            evidence = f"`{r.evidence_file}#L{r.evidence_line}`"
        
        note = r.note or ""
        if len(note) > 40:
            note = note[:37] + "..."
        
        report += f"| {i} | {link_str} | {r.link.link_type} | {r.status} | {evidence} | {note} |\n"
    
    report += """
---

## 代码证据详情

"""
    
    # 只展示已实现和部分实现的证据
    for i, r in enumerate(results, 1):
        if r.evidence_snippet and r.status in ["✅已实现", "⚠️部分实现"]:
            report += f"""### #{i} {r.link.source} → {r.link.target}

**文件**: `{r.evidence_file}` (L{r.evidence_line})

```python
{r.evidence_snippet}
```

---

"""
    
    report += """
*本文件由 `scripts/verify_topology_links.py` 自动生成，请勿手动编辑*
"""
    
    return report

# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("AGI 拓扑连接自动复核")
    print("=" * 60)
    print()
    
    # 1. 提取连接
    print(f"[1/3] 从拓扑HTML提取连接...")
    links = extract_links_from_html(TOPOLOGY_FILE)
    print(f"      找到 {len(links)} 条连接")
    print()
    
    # 2. 逐条验证
    print(f"[2/3] 验证代码证据...")
    results = []
    for i, link in enumerate(links, 1):
        result = verify_link(link)
        results.append(result)
        
        status_icon = result.status.split()[0] if result.status else "?"
        print(f"      [{i:2d}/{len(links)}] {link.source} → {link.target}: {status_icon}")
    
    print()
    
    # 3. 生成报告
    print(f"[3/3] 生成验证报告...")
    report = generate_report(results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(report, encoding='utf-8')
    print(f"      已保存到: {OUTPUT_FILE}")
    print()
    
    # 4. 统计摘要
    stats = {}
    for r in results:
        stats[r.status] = stats.get(r.status, 0) + 1
    
    print("=" * 60)
    print("验证完成!")
    print("=" * 60)
    for status, count in sorted(stats.items()):
        print(f"  {status}: {count}")
    print()

if __name__ == "__main__":
    main()
