#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态技能管理器 - Static Skills (anthropics/skills pattern)
实现模块化技能加载和管理，遵循 anthropics/skills 模式

功能：
1. 动态加载技能（从 skills/ 目录）
2. 解析 SKILL.md 文件（YAML frontmatter + markdown）
3. 技能调用和执行
4. 技能依赖管理
5. 技能状态监控

与现有 SkillManager 的区别：
- 现有 SkillManager: 动态生成的 Python 技能（从 insights 提取代码）
- StaticSkillManager: 静态 markdown 技能（SKILL.md + YAML frontmatter）
"""

import os
import re
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SkillStatus(Enum):
    """技能状态"""
    LOADED = "loaded"
    ERROR = "error"
    DISABLED = "disabled"
    NOT_FOUND = "not_found"


@dataclass
class SkillMetadata:
    """技能元数据（从 YAML frontmatter 解析）"""
    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0  # 优先级，数字越小优先级越高
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SkillMetadata':
        """从字典创建元数据"""
        return cls(
            name=data.get('name', 'unknown'),
            description=data.get('description', ''),
            version=data.get('version', '1.0.0'),
            author=data.get('author', ''),
            category=data.get('category', 'general'),
            tags=data.get('tags', []),
            dependencies=data.get('dependencies', []),
            enabled=data.get('enabled', True),
            priority=data.get('priority', 0),
            parameters=data.get('parameters', {})
        )


@dataclass
class StaticSkill:
    """静态技能对象"""
    metadata: SkillMetadata
    instructions: str  # Markdown 格式的指令
    path: Path
    scripts: Dict[str, Path] = field(default_factory=dict)  # 脚本文件路径
    resources: Dict[str, Path] = field(default_factory=dict)  # 资源文件路径
    status: SkillStatus = SkillStatus.LOADED
    error_message: Optional[str] = None
    load_time: Optional[datetime] = None

    def get_full_prompt(self) -> str:
        """获取完整的技能提示（包含元数据和指令）"""
        prompt = f"""# 技能: {self.metadata.name}

**版本**: {self.metadata.version}
**作者**: {self.metadata.author}
**分类**: {self.metadata.category}
**标签**: {', '.join(self.metadata.tags)}

## 描述
{self.metadata.description}

## 指令
{self.instructions}
"""
        return prompt


class StaticSkillManager:
    """静态技能管理器

    职责：
    1. 扫描 skills/ 目录
    2. 解析 SKILL.md 文件
    3. 加载和管理技能
    4. 提供技能调用接口
    5. 管理技能依赖关系

    遵循 anthropics/skills 模式：
    https://github.com/anthropics/skills
    """

    def __init__(self, skills_dir: str = "skills"):
        """
        初始化静态技能管理器

        Args:
            skills_dir: 技能目录路径
        """
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, StaticSkill] = {}
        self.skill_categories: Dict[str, List[str]] = {}
        self._initialized = False

        logger.info(f"StaticSkillManager 初始化: skills_dir={self.skills_dir}")

    def initialize(self) -> bool:
        """初始化技能管理器，加载所有技能"""
        if self._initialized:
            logger.warning("StaticSkillManager 已经初始化")
            return True

        try:
            # 确保技能目录存在
            if not self.skills_dir.exists():
                logger.warning(f"技能目录不存在: {self.skills_dir}")
                self.skills_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建技能目录: {self.skills_dir}")
                return True

            # 扫描并加载所有技能
            self._scan_skills()
            self._resolve_dependencies()
            self._build_categories()

            self._initialized = True
            logger.info(f"✅ StaticSkillManager 初始化完成，加载了 {len(self.skills)} 个技能")
            return True

        except Exception as e:
            logger.error(f"StaticSkillManager 初始化失败: {e}", exc_info=True)
            return False

    def _scan_skills(self):
        """扫描 skills/ 目录，加载所有技能"""
        logger.info("开始扫描静态技能...")

        for skill_path in self.skills_dir.iterdir():
            # 跳过非目录和隐藏目录
            if not skill_path.is_dir() or skill_path.name.startswith('_'):
                continue

            try:
                skill = self._load_skill(skill_path)
                if skill and skill.metadata.enabled:
                    self.skills[skill.metadata.name] = skill
                    logger.info(f"  ✅ 加载技能: {skill.metadata.name} v{skill.metadata.version}")
                elif skill and not skill.metadata.enabled:
                    logger.info(f"  ⏸️  跳过禁用的技能: {skill.metadata.name}")

            except Exception as e:
                logger.error(f"  ❌ 加载技能失败 {skill_path.name}: {e}")
                # 创建错误状态的技能对象
                error_skill = StaticSkill(
                    metadata=SkillMetadata(
                        name=skill_path.name,
                        description=f"加载失败: {str(e)}",
                        enabled=False
                    ),
                    instructions="",
                    path=skill_path,
                    status=SkillStatus.ERROR,
                    error_message=str(e)
                )
                self.skills[skill_path.name] = error_skill

    def _load_skill(self, skill_path: Path) -> Optional[StaticSkill]:
        """加载单个技能

        Args:
            skill_path: 技能目录路径

        Returns:
            StaticSkill 对象，如果加载失败则返回 None
        """
        skill_md_path = skill_path / "SKILL.md"

        # 检查 SKILL.md 是否存在
        if not skill_md_path.exists():
            logger.warning(f"SKILL.md 不存在: {skill_path}")
            return None

        # 读取 SKILL.md
        with open(skill_md_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析 YAML frontmatter 和 markdown 内容
        metadata, instructions = self._parse_skill_md(content)

        # 扫描脚本和资源文件
        scripts = {}
        resources = {}

        for file_path in skill_path.iterdir():
            if file_path.is_file() and file_path.name != "SKILL.md":
                if file_path.suffix in ['.py', '.sh', '.js']:
                    scripts[file_path.stem] = file_path
                elif file_path.suffix in ['.txt', '.json', '.yaml', '.md']:
                    resources[file_path.stem] = file_path

        # 创建技能对象
        skill = StaticSkill(
            metadata=metadata,
            instructions=instructions,
            path=skill_path,
            scripts=scripts,
            resources=resources,
            status=SkillStatus.LOADED,
            load_time=datetime.now()
        )

        return skill

    def _parse_skill_md(self, content: str) -> tuple[SkillMetadata, str]:
        """解析 SKILL.md 文件

        Args:
            content: SKILL.md 文件内容

        Returns:
            (SkillMetadata, instructions) 元组
        """
        # 提取 YAML frontmatter (在 --- 之间)
        frontmatter_pattern = r'^---\n(.*?)\n---\n(.*)$'
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            yaml_str = match.group(1)
            instructions = match.group(2).strip()

            # 解析 YAML
            try:
                metadata_dict = yaml.safe_load(yaml_str)
                metadata = SkillMetadata.from_dict(metadata_dict)
            except yaml.YAMLError as e:
                logger.error(f"YAML 解析失败: {e}")
                metadata = SkillMetadata(
                    name='unknown',
                    description='YAML 解析失败'
                )
        else:
            # 没有 frontmatter，整个文件都是指令
            logger.warning("SKILL.md 缺少 YAML frontmatter")
            metadata = SkillMetadata(
                name='unknown',
                description='缺少元数据'
            )
            instructions = content.strip()

        return metadata, instructions

    def _resolve_dependencies(self):
        """解析技能依赖关系，按优先级排序"""
        logger.info("解析技能依赖关系...")

        # 简单拓扑排序（基于优先级）
        sorted_skills = sorted(
            self.skills.values(),
            key=lambda s: s.metadata.priority
        )

        # 重建 skills 字典
        self.skills = {
            skill.metadata.name: skill
            for skill in sorted_skills
        }

        # TODO: 实现完整的依赖解析
        # 检查循环依赖
        # 验证依赖是否存在

    def _build_categories(self):
        """构建技能分类索引"""
        self.skill_categories = {}

        for skill_name, skill in self.skills.items():
            category = skill.metadata.category
            if category not in self.skill_categories:
                self.skill_categories[category] = []
            self.skill_categories[category].append(skill_name)

        logger.info(f"技能分类: {list(self.skill_categories.keys())}")

    def get_skill(self, skill_name: str) -> Optional[StaticSkill]:
        """获取技能对象

        Args:
            skill_name: 技能名称

        Returns:
            StaticSkill 对象，如果不存在则返回 None
        """
        return self.skills.get(skill_name)

    def get_skills_by_category(self, category: str) -> List[StaticSkill]:
        """获取指定分类的所有技能

        Args:
            category: 技能分类

        Returns:
            技能列表
        """
        skill_names = self.skill_categories.get(category, [])
        return [self.skills[name] for name in skill_names if name in self.skills]

    def get_all_skills(self) -> Dict[str, StaticSkill]:
        """获取所有技能"""
        return self.skills.copy()

    def get_skill_names(self) -> List[str]:
        """获取所有技能名称"""
        return list(self.skills.keys())

    def search_skills(self, query: str) -> List[StaticSkill]:
        """搜索技能（按名称、描述、标签）

        Args:
            query: 搜索关键词

        Returns:
            匹配的技能列表
        """
        query_lower = query.lower()
        results = []

        for skill in self.skills.values():
            # 搜索名称
            if query_lower in skill.metadata.name.lower():
                results.append(skill)
                continue

            # 搜索描述
            if query_lower in skill.metadata.description.lower():
                results.append(skill)
                continue

            # 搜索标签
            if any(query_lower in tag.lower() for tag in skill.metadata.tags):
                results.append(skill)
                continue

        return results

    def invoke_skill(self, skill_name: str, context: Dict[str, Any] = None) -> str:
        """调用技能（返回技能提示）

        Args:
            skill_name: 技能名称
            context: 上下文信息

        Returns:
            技能执行结果
        """
        skill = self.get_skill(skill_name)

        if not skill:
            return f"❌ 技能不存在: {skill_name}"

        if skill.status != SkillStatus.LOADED:
            return f"❌ 技能未加载: {skill_name} (状态: {skill.status.value})"

        if not skill.metadata.enabled:
            return f"❌ 技能已禁用: {skill_name}"

        # 检查依赖
        missing_deps = self._check_dependencies(skill)
        if missing_deps:
            return f"❌ 缺少依赖: {', '.join(missing_deps)}"

        logger.info(f"调用技能: {skill_name}")

        # 构建技能提示
        prompt = skill.get_full_prompt()

        # TODO: 实现实际的技能执行逻辑
        # 这里可以调用 LLM 执行技能指令
        # 或者执行技能脚本

        return f"✅ 技能已激活: {skill_name}\n\n{prompt}"

    def _check_dependencies(self, skill: StaticSkill) -> List[str]:
        """检查技能依赖是否满足

        Args:
            skill: 技能对象

        Returns:
            缺失的依赖列表
        """
        missing = []

        for dep in skill.metadata.dependencies:
            if dep not in self.skills:
                missing.append(dep)

        return missing

    def reload_skill(self, skill_name: str) -> bool:
        """重新加载技能

        Args:
            skill_name: 技能名称

        Returns:
            是否成功
        """
        skill = self.get_skill(skill_name)

        if not skill:
            logger.error(f"技能不存在，无法重新加载: {skill_name}")
            return False

        try:
            new_skill = self._load_skill(skill.path)
            if new_skill and new_skill.metadata.enabled:
                self.skills[skill_name] = new_skill
                logger.info(f"✅ 重新加载技能: {skill_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"重新加载技能失败 {skill_name}: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """获取技能管理器状态"""
        return {
            "initialized": self._initialized,
            "skills_dir": str(self.skills_dir),
            "total_skills": len(self.skills),
            "enabled_skills": sum(1 for s in self.skills.values() if s.metadata.enabled),
            "categories": list(self.skill_categories.keys()),
            "skills": {
                name: {
                    "version": skill.metadata.version,
                    "category": skill.metadata.category,
                    "status": skill.status.value,
                    "enabled": skill.metadata.enabled
                }
                for name, skill in self.skills.items()
            }
        }

    def print_status(self):
        """打印技能状态"""
        print("\n" + "=" * 70)
        print("📦 AGI Static Skills System 状态")
        print("=" * 70)

        status = self.get_status()
        print(f"初始化: {'✅' if status['initialized'] else '❌'}")
        print(f"技能目录: {status['skills_dir']}")
        print(f"总技能数: {status['total_skills']}")
        print(f"已启用: {status['enabled_skills']}")
        print(f"分类: {', '.join(status['categories'])}")

        print("\n技能列表:")
        for name, skill in self.skills.items():
            status_icon = "✅" if skill.metadata.enabled else "⏸️ "
            print(f"  {status_icon} {name} v{skill.metadata.version} [{skill.metadata.category}]")

        print("=" * 70 + "\n")

    def reload(self) -> bool:
        """重新加载全部技能（从磁盘刷新）"""
        try:
            self.skills.clear()
            self.skill_categories.clear()
            self._initialized = False
            return self.initialize()
        except Exception as e:
            logger.error(f"StaticSkillManager 重新加载失败: {e}", exc_info=True)
            return False


# 全局单例
_static_skill_manager_instance: Optional[StaticSkillManager] = None


def get_static_skill_manager(skills_dir: str = "skills") -> StaticSkillManager:
    """获取静态技能管理器单例

    Args:
        skills_dir: 技能目录路径

    Returns:
        StaticSkillManager 实例
    """
    global _static_skill_manager_instance

    if _static_skill_manager_instance is None:
        _static_skill_manager_instance = StaticSkillManager(skills_dir)
        _static_skill_manager_instance.initialize()

    return _static_skill_manager_instance
