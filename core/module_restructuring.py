"""
模块精简重构系统 - P2修复
目标: 200+模块 → 50核心模块
策略: 删除legacy代码、合并相似模块、统一接口层
"""

import os
import sys
import time
import re
import json
import shutil
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class ModuleInfo:
    """模块信息"""
    path: str
    name: str
    size_bytes: int
    imports: List[str]
    exports: List[str]
    referenced_by: List[str]
    category: str
    is_legacy: bool
    merge_target: Optional[str] = None


class ModuleAnalyzer:
    """模块分析器"""
    
    # Legacy模块模式
    LEGACY_PATTERNS = [
        r"deprecated",
        r"legacy",
        r"old_",
        r"_v\d+\.",
        r"backup",
        r"temp_",
        r"test_",
        r"experiment",
    ]
    
    # 类别关键词
    CATEGORY_KEYWORDS = {
        "memory": ["memory", "cache", "storage", "persist"],
        "cognitive": ["cognitive", "thinking", "reasoning", "inference"],
        "perception": ["perception", "sense", "input", "recognize"],
        "learning": ["learning", "train", "optimize", "adapt"],
        "interface": ["interface", "bridge", "adapter", "connector"],
        "evolution": ["evolution", "grow", "expand", "mutate"],
        "utility": ["util", "helper", "common", "shared"],
    }
    
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.core_dir = os.path.join(project_root, "core")
        self.modules: Dict[str, ModuleInfo] = {}
        self._reference_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def analyze_all_modules(self) -> Dict[str, ModuleInfo]:
        """分析所有模块"""
        print("[ModuleAnalyzer] Analyzing all modules...")
        
        for root, dirs, files in os.walk(self.core_dir):
            # 排除__pycache__
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.project_root)
                    
                    info = self._analyze_single_module(rel_path)
                    self.modules[rel_path] = info
        
        # 构建引用图
        self._build_reference_graph()
        
        print(f"[ModuleAnalyzer] Analyzed {len(self.modules)} modules")
        return self.modules
    
    def _analyze_single_module(self, rel_path: str) -> ModuleInfo:
        """分析单个模块"""
        full_path = os.path.join(self.project_root, rel_path)
        
        # 读取文件内容
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            content = ""
        
        # 基本信息
        name = os.path.basename(rel_path)[:-3]  # 去掉.py
        size = os.path.getsize(full_path)
        
        # 分析导入
        imports = self._extract_imports(content)
        
        # 分析导出（类、函数定义）
        exports = self._extract_exports(content)
        
        # 检测是否legacy
        is_legacy = self._detect_legacy(rel_path, content)
        
        # 分类
        category = self._categorize_module(name, content)
        
        return ModuleInfo(
            path=rel_path,
            name=name,
            size_bytes=size,
            imports=imports,
            exports=exports,
            referenced_by=[],  # 稍后填充
            category=category,
            is_legacy=is_legacy
        )
    
    def _extract_imports(self, content: str) -> List[str]:
        """提取导入语句"""
        imports = []
        
        # 匹配import语句
        import_patterns = [
            r"^from\s+([\w.]+)\s+import",
            r"^import\s+([\w.]+)",
        ]
        
        for pattern in import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                imports.append(match.group(1))
        
        return imports
    
    def _extract_exports(self, content: str) -> List[str]:
        """提取导出定义"""
        exports = []
        
        # 匹配类定义
        class_pattern = r"^class\s+(\w+)"
        for match in re.finditer(class_pattern, content, re.MULTILINE):
            exports.append(f"class:{match.group(1)}")
        
        # 匹配函数定义
        func_pattern = r"^def\s+(\w+)"
        for match in re.finditer(func_pattern, content, re.MULTILINE):
            exports.append(f"func:{match.group(1)}")
        
        return exports
    
    def _detect_legacy(self, path: str, content: str) -> bool:
        """检测是否为legacy模块"""
        path_lower = path.lower()
        content_lower = content.lower()
        
        for pattern in self.LEGACY_PATTERNS:
            if re.search(pattern, path_lower) or re.search(pattern, content_lower):
                return True
        
        # 检查是否包含deprecated标记
        if "deprecated" in content_lower or "legacy" in content_lower:
            return True
        
        return False
    
    def _categorize_module(self, name: str, content: str) -> str:
        """分类模块"""
        text = (name + " " + content).lower()
        
        scores = {}
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[category] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "other"
    
    def _build_reference_graph(self):
        """构建模块引用图"""
        for path, info in self.modules.items():
            for imp in info.imports:
                # 尝试匹配到本地模块
                for other_path, other_info in self.modules.items():
                    if other_info.name in imp or imp in other_path:
                        self._reference_graph[other_path].add(path)
                        info.referenced_by.append(other_path)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            "total_modules": len(self.modules),
            "legacy_modules": sum(1 for m in self.modules.values() if m.is_legacy),
            "by_category": defaultdict(int),
            "total_size_mb": sum(m.size_bytes for m in self.modules.values()) / (1024 * 1024),
            "orphan_modules": [],  # 未被引用的模块
            "hot_modules": []      # 被引用最多的模块
        }
        
        for info in self.modules.values():
            stats["by_category"][info.category] += 1
            
            if not info.referenced_by:
                stats["orphan_modules"].append(info.path)
        
        # 最热模块（被引用最多）
        sorted_modules = sorted(
            self.modules.values(),
            key=lambda m: len(m.referenced_by),
            reverse=True
        )
        stats["hot_modules"] = [(m.path, len(m.referenced_by)) for m in sorted_modules[:10]]
        
        return stats


class ModuleRestructuringPlan:
    """模块重构计划"""
    
    TARGET_MODULE_COUNT = 50
    
    def __init__(self, analyzer: ModuleAnalyzer):
        self.analyzer = analyzer
        self.plan: List[Dict] = []
        self.new_structure: Dict[str, List[str]] = {}
    
    def generate_plan(self) -> List[Dict]:
        """生成重构计划"""
        print("[RestructuringPlan] Generating restructuring plan...")
        
        stats = self.analyzer.get_statistics()
        current_count = stats["total_modules"]
        legacy_count = stats["legacy_modules"]
        
        print(f"  Current: {current_count} modules")
        print(f"  Legacy: {legacy_count} modules")
        print(f"  Target: {self.TARGET_MODULE_COUNT} modules")
        
        # 1. 标记删除legacy模块
        for path, info in self.analyzer.modules.items():
            if info.is_legacy:
                self.plan.append({
                    "action": "delete",
                    "source": path,
                    "reason": "legacy_code",
                    "priority": "high"
                })
        
        # 2. 合并相似模块
        self._plan_category_merges()
        
        # 3. 提取公共接口
        self._plan_interface_extraction()
        
        # 4. 重组织目录结构
        self._plan_directory_restructure()
        
        print(f"[RestructuringPlan] Generated {len(self.plan)} actions")
        return self.plan
    
    def _plan_category_merges(self):
        """计划按类别合并"""
        # 按类别分组
        by_category = defaultdict(list)
        for path, info in self.analyzer.modules.items():
            if not info.is_legacy:  # 跳过legacy
                by_category[info.category].append(info)
        
        # 对每个类别，如果模块数>5，计划合并
        for category, modules in by_category.items():
            if len(modules) > 5:
                # 按大小排序，最大的作为主模块
                sorted_modules = sorted(modules, key=lambda m: m.size_bytes, reverse=True)
                
                main_module = sorted_modules[0]
                merge_targets = sorted_modules[1:6]  # 最多合并5个
                
                for target in merge_targets:
                    self.plan.append({
                        "action": "merge",
                        "source": target.path,
                        "target": main_module.path,
                        "reason": f"category_consolidation_{category}",
                        "priority": "medium"
                    })
                    
                    # 更新分析器中的合并目标
                    target.merge_target = main_module.path
    
    def _plan_interface_extraction(self):
        """计划提取公共接口"""
        # 查找被引用最多的导出
        export_counter = defaultdict(int)
        for info in self.analyzer.modules.values():
            for export in info.exports:
                export_counter[export] += 1
        
        # 如果被多次引用，计划提取到公共接口
        common_exports = [(e, c) for e, c in export_counter.items() if c > 2]
        
        if common_exports:
            self.plan.append({
                "action": "create",
                "target": "core/interfaces/common.py",
                "exports": [e for e, _ in common_exports],
                "reason": "extract_common_interface",
                "priority": "medium"
            })
    
    def _plan_directory_restructure(self):
        """计划目录重组织"""
        # 目标结构
        self.new_structure = {
            "core/01_cognitive_kernel/": [
                "planning", "execution", "reasoning"
            ],
            "core/02_memory_system/": [
                "unified_memory", "storage", "retrieval"
            ],
            "core/03_perception_bridge/": [
                "input", "recognition", "encoding"
            ],
            "core/04_evolution_core/": [
                "capability", "growth", "adaptation"
            ],
            "core/05_interface_layer/": [
                "intent", "tool", "output"
            ],
            "core/06_utility/": [
                "common", "helpers"
            ]
        }
        
        # 为现有模块分配新位置
        for path, info in self.analyzer.modules.items():
            if info.is_legacy or info.merge_target:
                continue
            
            # 根据类别分配
            target_dir = self._map_category_to_directory(info.category)
            if target_dir:
                self.plan.append({
                    "action": "move",
                    "source": path,
                    "target": os.path.join(target_dir, os.path.basename(path)),
                    "reason": "directory_restructure",
                    "priority": "low"
                })
    
    def _map_category_to_directory(self, category: str) -> Optional[str]:
        """映射类别到目标目录"""
        mapping = {
            "cognitive": "core/01_cognitive_kernel/",
            "memory": "core/02_memory_system/",
            "perception": "core/03_perception_bridge/",
            "learning": "core/04_evolution_core/",
            "evolution": "core/04_evolution_core/",
            "interface": "core/05_interface_layer/",
            "utility": "core/06_utility/",
        }
        return mapping.get(category, "core/06_utility/")
    
    def estimate_result(self) -> Dict:
        """预估重构结果"""
        delete_count = sum(1 for p in self.plan if p["action"] == "delete")
        merge_count = sum(1 for p in self.plan if p["action"] == "merge")
        
        current = len(self.analyzer.modules)
        estimated = current - delete_count - merge_count + 6  # +6是新目录
        
        return {
            "current_modules": current,
            "estimated_modules": estimated,
            "deletions": delete_count,
            "merges": merge_count,
            "target_reached": estimated <= self.TARGET_MODULE_COUNT
        }
    
    def export_plan(self, output_path: str):
        """导出重构计划"""
        export_data = {
            "statistics": self.analyzer.get_statistics(),
            "restructuring_plan": self.plan,
            "new_structure": self.new_structure,
            "estimate": self.estimate_result(),
            "generated_at": time.time()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"[RestructuringPlan] Plan exported to {output_path}")


class ModuleRestructuringExecutor:
    """模块重构执行器"""
    
    def __init__(self, project_root: str, plan: List[Dict]):
        self.project_root = project_root
        self.plan = plan
        self.executed: List[Dict] = []
        self.errors: List[Dict] = []
    
    def execute_plan(self, dry_run: bool = True) -> bool:
        """执行重构计划"""
        print(f"[RestructuringExecutor] Executing plan (dry_run={dry_run})...")
        
        # 排序：先删除，再合并，最后移动
        sorted_plan = sorted(self.plan, key=lambda p: {
            "delete": 0,
            "merge": 1,
            "create": 2,
            "move": 3
        }.get(p["action"], 4))
        
        for action in sorted_plan:
            try:
                if action["action"] == "delete":
                    self._execute_delete(action, dry_run)
                elif action["action"] == "merge":
                    self._execute_merge(action, dry_run)
                elif action["action"] == "create":
                    self._execute_create(action, dry_run)
                elif action["action"] == "move":
                    self._execute_move(action, dry_run)
                
                self.executed.append(action)
                
            except Exception as e:
                self.errors.append({"action": action, "error": str(e)})
                print(f"[RestructuringExecutor] Error: {e}")
        
        print(f"[RestructuringExecutor] Executed {len(self.executed)} actions, {len(self.errors)} errors")
        return len(self.errors) == 0
    
    def _execute_delete(self, action: Dict, dry_run: bool):
        """执行删除"""
        source = os.path.join(self.project_root, action["source"])
        
        if dry_run:
            print(f"  [DRY RUN] Would delete: {action['source']}")
            return
        
        if os.path.exists(source):
            os.remove(source)
            print(f"  Deleted: {action['source']}")
    
    def _execute_merge(self, action: Dict, dry_run: bool):
        """执行合并"""
        source = os.path.join(self.project_root, action["source"])
        target = os.path.join(self.project_root, action["target"])
        
        if dry_run:
            print(f"  [DRY RUN] Would merge {action['source']} into {action['target']}")
            return
        
        # 读取源文件内容
        if os.path.exists(source):
            with open(source, 'r', encoding='utf-8') as f:
                source_content = f.read()
            
            # 追加到目标文件
            with open(target, 'a', encoding='utf-8') as f:
                f.write(f"\n\n# Merged from {action['source']}\n")
                f.write(source_content)
            
            # 删除源文件
            os.remove(source)
            print(f"  Merged: {action['source']} -> {action['target']}")
    
    def _execute_create(self, action: Dict, dry_run: bool):
        """执行创建"""
        target = os.path.join(self.project_root, action["target"])
        
        if dry_run:
            print(f"  [DRY RUN] Would create: {action['target']}")
            return
        
        # 创建目录
        os.makedirs(os.path.dirname(target), exist_ok=True)
        
        # 创建文件
        content = f"\"\"\"\nCommon interface module\nGenerated by Module Restructuring\n\"\"\"\n\n"
        for export in action.get("exports", []):
            content += f"# TODO: Import and re-export {export}\n"
        
        with open(target, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Created: {action['target']}")
    
    def _execute_move(self, action: Dict, dry_run: bool):
        """执行移动"""
        source = os.path.join(self.project_root, action["source"])
        target = os.path.join(self.project_root, action["target"])
        
        if dry_run:
            print(f"  [DRY RUN] Would move {action['source']} -> {action['target']}")
            return
        
        if os.path.exists(source):
            # 创建目标目录
            os.makedirs(os.path.dirname(target), exist_ok=True)
            
            # 移动文件
            shutil.move(source, target)
            print(f"  Moved: {action['source']} -> {action['target']}")


# 便捷函数
def analyze_and_plan_restructuring(project_root: str) -> ModuleRestructuringPlan:
    """分析并生成重构计划"""
    analyzer = ModuleAnalyzer(project_root)
    analyzer.analyze_all_modules()
    
    plan = ModuleRestructuringPlan(analyzer)
    plan.generate_plan()
    
    return plan


# 测试代码
if __name__ == "__main__":
    print("模块精简重构系统")
    print("=" * 70)
    
    # 分析模块
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plan = analyze_and_plan_restructuring(project_root)
    
    # 打印统计
    print("\n当前模块统计:")
    stats = plan.analyzer.get_statistics()
    print(f"  总模块数: {stats['total_modules']}")
    print(f"  Legacy模块: {stats['legacy_modules']}")
    print(f"  总大小: {stats['total_size_mb']:.2f} MB")
    print(f"  未引用模块: {len(stats['orphan_modules'])}")
    
    print("\n类别分布:")
    for cat, count in stats["by_category"].items():
        print(f"  {cat}: {count}")
    
    print("\n最热模块 (被引用最多):")
    for path, refs in stats["hot_modules"][:5]:
        print(f"  {path}: {refs} references")
    
    # 预估结果
    print("\n重构预估:")
    estimate = plan.estimate_result()
    print(f"  当前: {estimate['current_modules']} 模块")
    print(f"  预估: {estimate['estimated_modules']} 模块")
    print(f"  删除: {estimate['deletions']} 个")
    print(f"  合并: {estimate['merges']} 个")
    print(f"  目标达成: {'是' if estimate['target_reached'] else '否'}")
    
    # 导出计划
    plan.export_plan("module_restructuring_plan.json")
    
    print("\n重构计划已导出到: module_restructuring_plan.json")
