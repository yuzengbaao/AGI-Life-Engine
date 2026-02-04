#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统状态接地机制 (System Grounding Mechanism)
============================================

核心问题诊断 (2026-01-24):
    LLM基于预训练的"通用项目结构"推理，而非基于当前系统的真实状态推理。
    这导致LLM尝试读取不存在的文件（如ARCHITECTURE.md），被误判为"幻觉"。
    
    实际上这不是幻觉，而是**缺乏系统状态接地** (Grounding Gap)。

解决方案:
    1. 收集系统运行时真实状态
    2. 将状态注入到LLM的系统提示中
    3. 让LLM基于真实状态推理，而非基于预训练假设

设计理念:
    ┌─────────────────────────────────────────────────────────────┐
    │                    LLM (外置智能引擎)                        │
    │                         ↓                                   │
    │              系统状态接地层 (SystemGrounder)                 │
    │              - 当前工作目录                                  │
    │              - 实际存在的文件清单                            │
    │              - 可用工具及其规范                              │
    │              - 系统能力边界                                  │
    │                         ↓                                   │
    │                    AGI系统 (实际运行时)                      │
    └─────────────────────────────────────────────────────────────┘

作者: Claude (Opus 4.5)
日期: 2026-01-24
版本: 1.0.0
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class SystemState:
    """系统状态数据结构"""
    
    # 文件系统状态
    working_directory: str = ""
    existing_files: List[str] = field(default_factory=list)
    existing_directories: List[str] = field(default_factory=list)
    
    # 工具状态
    available_tools: List[Dict[str, str]] = field(default_factory=list)
    tool_usage_rules: List[str] = field(default_factory=list)
    
    # 能力边界
    capability_boundaries: List[str] = field(default_factory=list)
    
    # 模块状态
    initialized_modules: List[str] = field(default_factory=list)
    
    # 元数据
    grounding_timestamp: str = ""
    grounding_version: str = "1.0.0"


class SystemGrounder:
    """
    系统状态接地器
    
    负责收集系统运行时的真实状态，并生成可注入到LLM提示中的接地信息。
    
    核心功能:
        1. 扫描文件系统，获取真实存在的文件
        2. 收集可用工具及其调用规范
        3. 定义系统能力边界
        4. 生成结构化的接地提示
    """
    
    # 默认忽略的目录（不扫描）
    DEFAULT_IGNORE_DIRS: Set[str] = {
        '.git', '.venv', 'venv', '__pycache__', 'node_modules',
        '.pytest_cache', '.mypy_cache', 'backups', 'backbag',
        '.idea', '.vscode', 'dist', 'build', 'egg-info'
    }
    
    # 默认忽略的文件扩展名
    DEFAULT_IGNORE_EXTENSIONS: Set[str] = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe',
        '.egg', '.whl', '.tar', '.gz', '.zip'
    }
    
    # 重要文件（优先显示）
    IMPORTANT_FILES: Set[str] = {
        'README.md', 'readme.md', 'README.txt',
        'requirements.txt', 'setup.py', 'pyproject.toml',
        'config.yaml', 'config.json', '.env.example',
        'main.py', 'app.py', '__init__.py'
    }
    
    def __init__(
        self,
        workspace_root: Optional[str] = None,
        max_files: int = 100,
        max_depth: int = 3,
        ignore_dirs: Optional[Set[str]] = None,
        ignore_extensions: Optional[Set[str]] = None
    ):
        """
        初始化系统接地器
        
        Args:
            workspace_root: 工作空间根目录，默认为当前目录
            max_files: 最大返回文件数，防止提示过长
            max_depth: 最大扫描深度
            ignore_dirs: 自定义忽略目录
            ignore_extensions: 自定义忽略扩展名
        """
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self.max_files = max_files
        self.max_depth = max_depth
        self.ignore_dirs = ignore_dirs or self.DEFAULT_IGNORE_DIRS
        self.ignore_extensions = ignore_extensions or self.DEFAULT_IGNORE_EXTENSIONS
        
        # 缓存状态（可设置过期时间）
        self._cached_state: Optional[SystemState] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds: int = 60  # 缓存60秒
        
        logger.info(f"[SystemGrounder] 初始化完成，工作空间: {self.workspace_root}")
    
    def get_system_state(self, force_refresh: bool = False) -> SystemState:
        """
        获取当前系统状态
        
        Args:
            force_refresh: 是否强制刷新缓存
            
        Returns:
            SystemState: 系统状态数据
        """
        # 检查缓存
        if not force_refresh and self._is_cache_valid():
            return self._cached_state
        
        # 收集新状态
        state = SystemState(
            working_directory=str(self.workspace_root),
            grounding_timestamp=datetime.now().isoformat(timespec='seconds')
        )
        
        # 1. 扫描文件系统
        files, dirs = self._scan_filesystem()
        state.existing_files = files
        state.existing_directories = dirs
        
        # 2. 收集工具信息
        state.available_tools = self._collect_available_tools()
        state.tool_usage_rules = self._get_tool_usage_rules()
        
        # 3. 定义能力边界
        state.capability_boundaries = self._define_capability_boundaries()
        
        # 4. 收集已初始化模块
        state.initialized_modules = self._collect_initialized_modules()
        
        # 更新缓存
        self._cached_state = state
        self._cache_timestamp = datetime.now()
        
        logger.debug(f"[SystemGrounder] 状态已刷新，文件数: {len(files)}, 目录数: {len(dirs)}")
        return state
    
    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        if self._cached_state is None or self._cache_timestamp is None:
            return False
        
        elapsed = (datetime.now() - self._cache_timestamp).total_seconds()
        return elapsed < self._cache_ttl_seconds
    
    def _scan_filesystem(self) -> tuple[List[str], List[str]]:
        """
        扫描文件系统，获取真实存在的文件和目录
        
        Returns:
            (files, directories): 文件列表和目录列表
        """
        root_files: List[str] = []  # 根目录文件优先
        important_files: List[str] = []  # 重要文件次之
        other_files: List[str] = []  # 其他文件
        directories: List[str] = []
        
        try:
            # 首先单独扫描根目录的文件（最重要）
            for item in self.workspace_root.iterdir():
                if item.is_file():
                    if item.suffix in self.ignore_extensions:
                        continue
                    if item.name.startswith('.'):
                        continue
                    rel_path = item.name
                    if item.name in self.IMPORTANT_FILES:
                        root_files.insert(0, rel_path)  # 重要文件放最前
                    else:
                        root_files.append(rel_path)
            
            # 然后扫描子目录
            for item in self._walk_directory(self.workspace_root, depth=0):
                rel_path = str(item.relative_to(self.workspace_root))
                
                # 跳过已经在根目录扫描过的文件
                if item.parent == self.workspace_root:
                    continue
                
                if item.is_file():
                    # 检查是否是重要文件
                    if item.name in self.IMPORTANT_FILES:
                        important_files.append(rel_path)
                    else:
                        other_files.append(rel_path)
                elif item.is_dir():
                    directories.append(rel_path + "/")
            
            # 🆕 [2026-01-24] 优先级排序: 根目录文件 > 重要文件 > 其他文件
            all_files = root_files + important_files + other_files
            
            # 限制数量
            if len(all_files) > self.max_files:
                all_files = all_files[:self.max_files]
                logger.debug(f"[SystemGrounder] 文件数超限，截断至 {self.max_files}")
            
            return all_files, directories[:50]  # 目录也限制数量
            
        except Exception as e:
            logger.warning(f"[SystemGrounder] 文件系统扫描失败: {e}")
            return [], []
    
    def _walk_directory(self, path: Path, depth: int):
        """
        递归遍历目录
        
        Args:
            path: 当前路径
            depth: 当前深度
            
        Yields:
            Path: 文件或目录路径
        """
        if depth > self.max_depth:
            return
        
        try:
            for item in path.iterdir():
                # 跳过忽略的目录
                if item.is_dir():
                    if item.name in self.ignore_dirs:
                        continue
                    if item.name.startswith('.'):
                        continue
                    yield item
                    yield from self._walk_directory(item, depth + 1)
                
                # 跳过忽略的文件
                elif item.is_file():
                    if item.suffix in self.ignore_extensions:
                        continue
                    if item.name.startswith('.'):
                        continue
                    yield item
                    
        except PermissionError:
            pass  # 忽略权限错误
        except Exception as e:
            logger.debug(f"[SystemGrounder] 遍历 {path} 时出错: {e}")
    
    def _collect_available_tools(self) -> List[Dict[str, str]]:
        """
        收集可用工具列表
        
        Returns:
            工具列表，每个工具包含 name, description, usage
        """
        # 基于AGI系统的实际工具
        tools = [
            {
                "name": "local_document_reader.read",
                "description": "读取本地文档内容",
                "usage": "read(path='相对或绝对路径')",
                "constraints": "文件必须存在于工作空间内，使用下方的【实际存在的文件】列表确认"
            },
            {
                "name": "web_search.search",
                "description": "搜索网络获取实时信息",
                "usage": "search(query='搜索关键词')",
                "constraints": "需要网络连接"
            },
            {
                "name": "image_understanding.analyze",
                "description": "分析图像内容",
                "usage": "analyze(image_path='图像路径')",
                "constraints": "支持 jpg, png, webp 格式"
            },
            {
                "name": "code_executor.run",
                "description": "执行Python代码",
                "usage": "run(code='Python代码')",
                "constraints": "在安全沙箱中执行，有超时限制"
            }
        ]
        
        return tools
    
    def _get_tool_usage_rules(self) -> List[str]:
        """
        获取工具使用规则
        
        Returns:
            规则列表
        """
        return [
            "【关键规则】调用 local_document_reader.read() 前，必须先确认文件存在于【实际存在的文件】列表中",
            "如果不确定文件是否存在，应先询问用户或请求列出目录",
            "不要假设常见文件（如 ARCHITECTURE.md, DESIGN.md）存在，除非在文件列表中看到",
            "工具调用失败时，应报告失败原因，而非假装成功",
            "对于不存在的文件，明确告知用户'该文件不存在'，而非尝试读取"
        ]
    
    def _define_capability_boundaries(self) -> List[str]:
        """
        定义系统能力边界
        
        Returns:
            能力边界描述列表
        """
        return [
            "可以读取工作空间内的文件，但仅限于【实际存在的文件】列表中的文件",
            "可以搜索网络获取实时信息",
            "可以分析图像、音频、视频（如果文件存在）",
            "可以执行Python代码进行计算和分析",
            "不能修改系统配置或执行危险操作",
            "不能访问工作空间外的文件系统",
            "如果能力范围内无法完成任务，应明确告知用户"
        ]
    
    def _collect_initialized_modules(self) -> List[str]:
        """
        收集已初始化的模块
        
        Returns:
            模块名称列表
        """
        # 这些是系统实际初始化的模块
        return [
            "世界模型 (WorldModel)",
            "持续学习框架 (ContinualLearning)", 
            "自我优化器 (SelfOptimizer)",
            "创新方案生成器 (InnovationGenerator)",
            "图像理解 (ImageUnderstanding)",
            "音频处理 (AudioProcessor)",
            "视频处理 (VideoProcessor)",
            "跨模态对齐 (CrossModalAlignment)",
            "OCR识别 (PaddleOCR)",
            "监控系统 (MonitoringSystem)",
            "安全框架 (SafetyFramework)",
            "网络搜索 (WebSearch)",
            "本地文档读取 (LocalDocumentReader)",
            "幻觉检测器 (HallucinationDetector)",
            "系统接地器 (SystemGrounder)"  # 自己
        ]
    
    def generate_grounding_prompt(self, state: Optional[SystemState] = None) -> str:
        """
        生成系统状态接地提示
        
        Args:
            state: 系统状态，如果为None则自动获取
            
        Returns:
            可注入到系统提示中的接地信息
        """
        if state is None:
            state = self.get_system_state()
        
        # 构建接地提示
        sections = []
        
        # 1. 工作目录
        sections.append(f"【当前工作目录】\n{state.working_directory}")
        
        # 2. 实际存在的文件（关键部分）
        if state.existing_files:
            file_list = "\n".join(f"  - {f}" for f in state.existing_files[:50])
            total = len(state.existing_files)
            if total > 50:
                file_list += f"\n  ... 还有 {total - 50} 个文件"
            sections.append(f"【实际存在的文件】(共 {total} 个)\n{file_list}")
        else:
            sections.append("【实际存在的文件】\n  (扫描失败或目录为空)")
        
        # 3. 工具使用规则（关键部分）
        if state.tool_usage_rules:
            rules = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(state.tool_usage_rules))
            sections.append(f"【工具使用规则】\n{rules}")
        
        # 4. 能力边界
        if state.capability_boundaries:
            boundaries = "\n".join(f"  • {b}" for b in state.capability_boundaries)
            sections.append(f"【能力边界】\n{boundaries}")
        
        # 5. 元数据
        sections.append(f"【接地时间】{state.grounding_timestamp}")
        
        # 组合
        grounding_prompt = "\n\n".join(sections)
        
        return f"""
=== 系统状态接地信息 (System Grounding) ===
以下是当前系统的真实运行状态，请基于这些信息进行推理，而非基于预训练假设。

{grounding_prompt}

=== 接地信息结束 ===
"""
    
    def check_file_exists(self, file_path: str) -> bool:
        """
        检查文件是否存在（供幻觉检测器使用）
        
        Args:
            file_path: 文件路径（相对或绝对）
            
        Returns:
            是否存在
        """
        # 尝试解析为绝对路径
        if os.path.isabs(file_path):
            target = Path(file_path)
        else:
            target = self.workspace_root / file_path
        
        return target.exists()
    
    def get_file_existence_map(self) -> Dict[str, bool]:
        """
        获取文件存在性映射（用于快速查询）
        
        Returns:
            {文件路径: 是否存在}
        """
        state = self.get_system_state()
        return {f: True for f in state.existing_files}


# 全局单例（可选使用）
_global_grounder: Optional[SystemGrounder] = None


def get_global_grounder(workspace_root: Optional[str] = None) -> SystemGrounder:
    """
    获取全局系统接地器实例
    
    Args:
        workspace_root: 工作空间根目录
        
    Returns:
        SystemGrounder实例
    """
    global _global_grounder
    
    if _global_grounder is None:
        _global_grounder = SystemGrounder(workspace_root=workspace_root)
    
    return _global_grounder


def generate_grounded_system_prompt(base_prompt: str, workspace_root: Optional[str] = None) -> str:
    """
    生成带接地信息的系统提示
    
    这是给 llm_provider.py 调用的便捷函数。
    
    Args:
        base_prompt: 基础系统提示
        workspace_root: 工作空间根目录
        
    Returns:
        增强后的系统提示
    """
    grounder = get_global_grounder(workspace_root)
    grounding_info = grounder.generate_grounding_prompt()
    
    return f"{base_prompt}\n\n{grounding_info}"


# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    # 测试
    grounder = SystemGrounder(workspace_root=r"D:\TRAE_PROJECT\AGI")
    state = grounder.get_system_state()
    
    print("=" * 60)
    print("系统状态接地测试")
    print("=" * 60)
    print(f"工作目录: {state.working_directory}")
    print(f"文件数量: {len(state.existing_files)}")
    print(f"目录数量: {len(state.existing_directories)}")
    print(f"工具数量: {len(state.available_tools)}")
    print()
    print("接地提示预览:")
    print("-" * 60)
    print(grounder.generate_grounding_prompt(state))
