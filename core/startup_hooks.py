"""
AGI System Startup Hooks
========================
自动在 AGI 系统启动时加载预配置的资源和执行初始化任务。

使用方式：
    在 AGI_Life_Engine.py 初始化阶段调用：
    from core.startup_hooks import StartupHooks
    hooks = StartupHooks(knowledge_graph, llm_service)
    hooks.execute_all()

配置文件：data/startup_config.json
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class StartupHooks:
    """AGI系统启动钩子管理器"""
    
    CONFIG_PATH = "data/startup_config.json"
    DEFAULT_CONFIG = {
        "version": "1.0.0",
        "enabled": True,
        "auto_index": {
            "enabled": True,
            "document_index_path": "data/document_index.json",
            "load_to_knowledge_graph": True,
            "max_entries_to_load": 1000
        },
        "startup_tasks": [
            {
                "name": "load_document_index",
                "enabled": True,
                "priority": 1
            }
        ],
        "last_executed": None
    }
    
    def __init__(self, knowledge_graph=None, llm_service=None):
        """
        初始化启动钩子管理器
        
        Args:
            knowledge_graph: ArchitectureKnowledgeGraph 实例
            llm_service: LLMService 实例（可选，用于需要LLM的任务）
        """
        self.knowledge_graph = knowledge_graph
        self.llm_service = llm_service
        self.config = self._load_config()
        self.execution_log = []
        
    def _load_config(self) -> Dict[str, Any]:
        """加载或创建启动配置"""
        config_path = Path(self.CONFIG_PATH)
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 合并默认配置（确保新字段存在）
                    merged = {**self.DEFAULT_CONFIG, **config}
                    return merged
            except Exception as e:
                logger.warning(f"加载启动配置失败: {e}，使用默认配置")
                return self.DEFAULT_CONFIG.copy()
        else:
            # 创建默认配置文件
            self._save_config(self.DEFAULT_CONFIG)
            return self.DEFAULT_CONFIG.copy()
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置"""
        config_path = Path(self.CONFIG_PATH)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存启动配置失败: {e}")
    
    def execute_all(self) -> Dict[str, Any]:
        """
        执行所有启用的启动钩子
        
        Returns:
            执行结果摘要
        """
        if not self.config.get("enabled", True):
            logger.info("启动钩子已禁用")
            return {"status": "disabled"}
        
        print("   [StartupHooks] 🚀 执行启动钩子...")
        results = {
            "start_time": datetime.now().isoformat(),
            "tasks": []
        }
        
        # 执行文档索引加载
        if self.config.get("auto_index", {}).get("enabled", True):
            result = self._load_document_index()
            results["tasks"].append({
                "name": "load_document_index",
                "result": result
            })
        
        # 执行其他配置的启动任务
        for task in self.config.get("startup_tasks", []):
            if task.get("enabled", True):
                task_name = task.get("name", "unknown")
                if task_name != "load_document_index":  # 避免重复执行
                    result = self._execute_task(task_name)
                    results["tasks"].append({
                        "name": task_name,
                        "result": result
                    })
        
        results["end_time"] = datetime.now().isoformat()
        
        # 更新配置中的执行时间
        self.config["last_executed"] = results["end_time"]
        self._save_config(self.config)
        
        # 打印摘要
        success_count = sum(1 for t in results["tasks"] if t["result"].get("success", False))
        print(f"   [StartupHooks] ✅ 完成 {success_count}/{len(results['tasks'])} 个启动任务")
        
        return results
    
    def _load_document_index(self) -> Dict[str, Any]:
        """
        加载文档索引到知识图谱
        
        Returns:
            执行结果
        """
        auto_index_config = self.config.get("auto_index", {})
        index_path = auto_index_config.get("document_index_path", "data/document_index.json")
        max_entries = auto_index_config.get("max_entries_to_load", 1000)
        
        result = {
            "success": False,
            "message": "",
            "entries_loaded": 0
        }
        
        # 检查索引文件是否存在
        if not os.path.exists(index_path):
            result["message"] = f"索引文件不存在: {index_path}"
            print(f"   [StartupHooks] ⚠️ {result['message']}")
            return result
        
        try:
            # 加载索引
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # 验证索引结构
            if not isinstance(index_data, dict):
                result["message"] = "索引格式无效：不是字典"
                return result
            
            # 提取元数据 - 兼容多种索引格式
            # 获取文档/目录信息 - 兼容多种索引格式
            # 格式1: {"metadata": {...}, "documents": {...}}
            # 格式2: {"generated": "...", "total_docs": N, "categories": {...}, "documents": {...}}
            # 格式3: {"total_documents": N, "by_extension": {...}, "summaries": [...}, "documents": {...}}
            total_files = (
                index_data.get("metadata", {}).get("total_files", 0) or
                index_data.get("total_docs", 0) or
                index_data.get("total_documents", 0) or
                0
            )
            index_timestamp = (
                index_data.get("metadata", {}).get("generated_at") or
                index_data.get("generated") or
                "unknown"
            )
            
            # 获取文档/目录信息
            documents = index_data.get("documents", {})
            categories = index_data.get("categories", {})
            
            # 确保 documents 是字典类型
            if not isinstance(documents, dict):
                logger.warning(f"文档索引格式不兼容，documents 不是字典类型: {type(documents)}")
                result["message"] = "索引格式不兼容：documents 不是字典"
                return result
            
            # 统计实际文件数（如果metadata中没有）
            if total_files == 0 and documents:
                if isinstance(documents, dict):
                    for dir_files in documents.values():
                        if isinstance(dir_files, list):
                            total_files += len(dir_files)
            
            print(f"   [StartupHooks] 📚 加载文档索引 ({total_files} 个文件, 生成于 {index_timestamp})")
            
            # 如果有知识图谱，加载索引元数据
            if self.knowledge_graph and auto_index_config.get("load_to_knowledge_graph", True):
                # 创建索引元节点
                index_node_id = f"document_index_{index_timestamp}"
                total_directories = len(documents) if isinstance(documents, dict) else 0
                self.knowledge_graph.add_node(
                    index_node_id,
                    type="document_index",
                    total_files=total_files,
                    total_directories=total_directories,
                    total_categories=len(categories) if isinstance(categories, dict) else 0,
                    index_path=index_path,
                    generated_at=index_timestamp,
                    loaded_at=datetime.now().isoformat()
                )
                
                # 加载部分目录信息作为节点（防止过载）
                entries_loaded = 0
                
                # 确保 documents 是字典类型
                if not isinstance(documents, dict):
                    logger.warning(f"无法加载目录节点：documents 不是字典类型，类型为 {type(documents)}")
                    result["entries_loaded"] = 0
                else:
                    for dir_path, files in documents.items():
                        if entries_loaded >= max_entries:
                            break
                        
                        # 创建目录节点
                        dir_node_id = f"dir:{dir_path}"
                        file_count = len(files) if isinstance(files, list) else 0
                        
                        self.knowledge_graph.graph.add_node(
                            dir_node_id,
                            type="directory",
                            path=dir_path,
                            file_count=file_count
                        )
                        
                        # 连接到索引节点
                        self.knowledge_graph.graph.add_edge(
                            index_node_id,
                            dir_node_id,
                            relation="contains_directory"
                        )
                        
                        entries_loaded += 1
                
                # 保存图谱
                self.knowledge_graph.save_graph()
                
                result["entries_loaded"] = entries_loaded
                print(f"   [StartupHooks] 📊 已加载 {entries_loaded} 个目录节点到知识图谱")
            
            result["success"] = True
            result["message"] = f"成功加载索引（{total_files} 个文件）"
            
        except Exception as e:
            result["message"] = f"加载索引失败: {str(e)}"
            logger.error(result["message"])
            print(f"   [StartupHooks] ❌ {result['message']}")
        
        return result
    
    def _execute_task(self, task_name: str) -> Dict[str, Any]:
        """
        执行指定的启动任务
        
        Args:
            task_name: 任务名称
            
        Returns:
            执行结果
        """
        result = {
            "success": False,
            "message": f"未知任务: {task_name}"
        }
        
        # 可扩展的任务注册表
        task_registry = {
            "load_document_index": self._load_document_index,
            # 可以在这里添加更多任务
        }
        
        if task_name in task_registry:
            result = task_registry[task_name]()
        
        return result
    
    def add_startup_task(self, name: str, enabled: bool = True, priority: int = 10):
        """
        添加新的启动任务
        
        Args:
            name: 任务名称
            enabled: 是否启用
            priority: 优先级（数字越小越优先）
        """
        tasks = self.config.get("startup_tasks", [])
        
        # 检查是否已存在
        existing = next((t for t in tasks if t.get("name") == name), None)
        if existing:
            existing["enabled"] = enabled
            existing["priority"] = priority
        else:
            tasks.append({
                "name": name,
                "enabled": enabled,
                "priority": priority
            })
        
        # 按优先级排序
        tasks.sort(key=lambda t: t.get("priority", 10))
        self.config["startup_tasks"] = tasks
        self._save_config(self.config)
        
        print(f"   [StartupHooks] ➕ 添加启动任务: {name} (优先级: {priority})")
    
    def set_auto_index_enabled(self, enabled: bool):
        """启用/禁用自动索引加载"""
        if "auto_index" not in self.config:
            self.config["auto_index"] = {}
        self.config["auto_index"]["enabled"] = enabled
        self._save_config(self.config)
        
        status = "启用" if enabled else "禁用"
        print(f"   [StartupHooks] 🔄 自动索引加载已{status}")


# 便捷函数：用于快速集成
def run_startup_hooks(knowledge_graph=None, llm_service=None) -> Dict[str, Any]:
    """
    运行所有启动钩子的便捷函数
    
    Args:
        knowledge_graph: 知识图谱实例
        llm_service: LLM服务实例
        
    Returns:
        执行结果
    """
    hooks = StartupHooks(knowledge_graph, llm_service)
    return hooks.execute_all()
