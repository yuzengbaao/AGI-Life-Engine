#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本地文档写入器
Local Document Writer for AGI

安全地写入本地项目文档，支持多种操作模式
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class LocalDocumentWriter:
    """
    本地文档写入器

    安全特性:
    1. 路径白名单 - 只允许写入项目目录
    2. 文件类型限制 - 只允许安全格式
    3. 自动备份 - 写入前创建备份
    4. 敏感文件保护 - 拒绝覆盖系统关键文件
    5. 原子写入 - 防止写入失败导致文件损坏
    """

    # 允许写入的文件类型（扩展读取器的类型）
    ALLOWED_EXTENSIONS = {
        '.md', '.txt', '.rst', '.py', '.js', '.ts', '.json',
        '.yaml', '.yml', '.xml', '.html', '.css'
    }

    # 禁止写入的敏感文件模式
    PROTECTED_PATTERNS = [
        r'\.env$',
        r'\.key$',
        r'\.pem$',
        r'password',
        r'secret',
        r'credential',
        r'token',
        r'\.git/',
        r'node_modules/',
        r'\.venv/',
        r'__pycache__/'
    ]

    # 文件大小限制 (MB)
    MAX_FILE_SIZE_MB = 100

    # 备份目录
    BACKUP_DIR = ".backups"

    def __init__(self, project_root: str = None):
        """
        初始化文档写入器

        Args:
            project_root: 项目根目录路径
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent

        self.project_root = Path(project_root).resolve()
        self.write_history = []
        self.backup_dir = self.project_root / self.BACKUP_DIR

        # 创建备份目录
        self.backup_dir.mkdir(exist_ok=True)

        logger.info(f"✍️ 本地文档写入器已初始化")
        logger.info(f"   项目根目录: {self.project_root}")
        logger.info(f"   备份目录: {self.backup_dir}")
        logger.info(f"   允许的文件类型: {len(self.ALLOWED_EXTENSIONS)} 种")

    def is_safe_path(self, file_path: Path) -> bool:
        """检查路径是否安全（在项目目录内）"""
        try:
            resolved_path = file_path.resolve()
            resolved_path.relative_to(self.project_root)
            return True
        except ValueError:
            return False

    def is_protected_file(self, file_path: Path) -> bool:
        """检查文件是否受保护（禁止覆盖）"""
        for pattern in self.PROTECTED_PATTERNS:
            if re.search(pattern, str(file_path), re.IGNORECASE):
                return True
        return False

    def is_allowed_file_type(self, file_path: Path) -> bool:
        """检查文件类型是否允许写入"""
        if not file_path.suffix:
            # 无扩展名的文件允许创建（如新文件）
            return True

        return file_path.suffix.lower() in self.ALLOWED_EXTENSIONS

    def create_backup(self, file_path: Path) -> Optional[Path]:
        """
        创建文件备份

        Args:
            file_path: 要备份的文件路径

        Returns:
            备份文件路径，或 None（如果文件不存在）
        """
        if not file_path.exists():
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name

            # 确保备份目录存在
            backup_path.parent.mkdir(parents=True, exist_ok=True)

            # 复制文件
            import shutil
            shutil.copy2(file_path, backup_path)

            logger.debug(f"📦 已创建备份: {backup_path.name}")
            return backup_path

        except Exception as e:
            logger.warning(f"创建备份失败: {e}")
            return None

    def atomic_write(self, file_path: Path, content: str, encoding: str = 'utf-8') -> Dict:
        """
        原子写入 - 先写入临时文件，然后重命名

        Args:
            file_path: 目标文件路径
            content: 文件内容
            encoding: 文件编码

        Returns:
            操作结果字典
        """
        try:
            # 创建临时文件
            temp_path = file_path.with_suffix(file_path.suffix + '.tmp')

            # 写入临时文件
            with open(temp_path, 'w', encoding=encoding) as f:
                f.write(content)

            # 原子重命名
            temp_path.replace(file_path)

            return {'success': True, 'temp_path': str(temp_path)}

        except Exception as e:
            # 清理临时文件
            if temp_path.exists():
                temp_path.unlink()

            return {'success': False, 'error': str(e)}

    def write_file(self, file_path: str, content: str, create_dirs: bool = True,
                   backup: bool = True, encoding: str = 'utf-8') -> Dict:
        """
        写入文件（覆盖或新建）

        Args:
            file_path: 文件路径（相对或绝对）
            content: 文件内容
            create_dirs: 是否自动创建目录
            backup: 是否创建备份（仅覆盖现有文件时）
            encoding: 文件编码

        Returns:
            操作结果字典
        """
        try:
            # 解析路径
            path = Path(file_path)

            # 如果是相对路径，基于项目根目录
            if not path.is_absolute():
                path = self.project_root / path

            # 安全检查
            if not self.is_safe_path(path):
                return {
                    'success': False,
                    'error': '路径超出项目目录范围',
                    'path': str(path)
                }

            # 检查文件类型
            if not self.is_allowed_file_type(path):
                return {
                    'success': False,
                    'error': f'不允许的文件类型: {path.suffix}',
                    'allowed_types': list(self.ALLOWED_EXTENSIONS)
                }

            # 检查是否为受保护文件
            if path.exists() and self.is_protected_file(path):
                return {
                    'success': False,
                    'error': '受保护的文件，禁止覆盖',
                    'path': str(path)
                }

            # 检查大小
            content_size_mb = len(content.encode(encoding)) / (1024 * 1024)
            if content_size_mb > self.MAX_FILE_SIZE_MB:
                return {
                    'success': False,
                    'error': f'内容过大: {content_size_mb:.1f}MB (限制: {self.MAX_FILE_SIZE_MB}MB)'
                }

            # 创建目录（如果需要）
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)

            # 创建备份（如果文件存在且需要备份）
            backup_path = None
            if backup and path.exists():
                backup_path = self.create_backup(path)

            # 原子写入
            write_result = self.atomic_write(path, content, encoding)

            if not write_result['success']:
                return {
                    'success': False,
                    'error': f'写入失败: {write_result.get("error")}',
                    'path': str(path)
                }

            # 记录写入历史
            self.write_history.append({
                'path': str(path),
                'relative_path': str(path.relative_to(self.project_root)),
                'timestamp': datetime.now().isoformat(),
                'size': len(content),
                'backup': str(backup_path) if backup_path else None,
                'operation': 'write' if not backup_path else 'overwrite'
            })

            return {
                'success': True,
                'path': str(path),
                'relative_path': str(path.relative_to(self.project_root)),
                'size': len(content),
                'backup': str(backup_path) if backup_path else None,
                'operation': 'write' if not backup_path else 'overwrite',
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"写入文件失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': file_path
            }

    def append_file(self, file_path: str, content: str, create_if_not_exists: bool = True,
                    separator: str = '\n\n', encoding: str = 'utf-8') -> Dict:
        """
        追加内容到文件

        Args:
            file_path: 文件路径
            content: 要追加的内容
            create_if_not_exists: 文件不存在时是否创建
            separator: 内容分隔符
            encoding: 文件编码

        Returns:
            操作结果字典
        """
        try:
            path = Path(file_path)

            if not path.is_absolute():
                path = self.project_root / path

            # 安全检查
            if not self.is_safe_path(path):
                return {
                    'success': False,
                    'error': '路径超出项目目录范围',
                    'path': str(path)
                }

            # 检查文件类型
            if not self.is_allowed_file_type(path):
                return {
                    'success': False,
                    'error': f'不允许的文件类型: {path.suffix}'
                }

            # 文件不存在时的处理
            if not path.exists():
                if create_if_not_exists:
                    return self.write_file(file_path, content, encoding=encoding)
                else:
                    return {
                        'success': False,
                        'error': '文件不存在',
                        'path': str(path)
                    }

            # 读取现有内容
            try:
                with open(path, 'r', encoding=encoding) as f:
                    existing_content = f.read()
            except Exception as e:
                return {
                    'success': False,
                    'error': f'读取现有内容失败: {e}',
                    'path': str(path)
                }

            # 追加内容
            new_content = existing_content + separator + content

            # 写入
            return self.write_file(file_path, new_content, backup=True, encoding=encoding)

        except Exception as e:
            logger.error(f"追加文件失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': file_path
            }

    def edit_file(self, file_path: str, old_content: str, new_content: str,
                  backup: bool = True, encoding: str = 'utf-8') -> Dict:
        """
        编辑文件 - 替换指定内容

        Args:
            file_path: 文件路径
            old_content: 要替换的旧内容
            new_content: 新内容
            backup: 是否创建备份
            encoding: 文件编码

        Returns:
            操作结果字典
        """
        try:
            path = Path(file_path)

            if not path.is_absolute():
                path = self.project_root / path

            # 安全检查
            if not self.is_safe_path(path):
                return {
                    'success': False,
                    'error': '路径超出项目目录范围'
                }

            if not path.exists():
                return {
                    'success': False,
                    'error': '文件不存在',
                    'path': str(path)
                }

            # 读取文件
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
            except Exception as e:
                return {
                    'success': False,
                    'error': f'读取文件失败: {e}'
                }

            # 检查旧内容是否存在
            if old_content not in content:
                return {
                    'success': False,
                    'error': '未找到指定的旧内容',
                    'note': '可能内容已被修改或包含特殊字符'
                }

            # 替换内容
            new_file_content = content.replace(old_content, new_content)

            # 写入
            result = self.write_file(file_path, new_file_content, backup=backup, encoding=encoding)

            if result['success']:
                result['operation'] = 'edit'
                result['replacements'] = content.count(old_content)

            return result

        except Exception as e:
            logger.error(f"编辑文件失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': file_path
            }

    def prepend_file(self, file_path: str, content: str,
                     separator: str = '\n\n', encoding: str = 'utf-8') -> Dict:
        """
        在文件开头插入内容

        Args:
            file_path: 文件路径
            content: 要插入的内容
            separator: 内容分隔符
            encoding: 文件编码

        Returns:
            操作结果字典
        """
        try:
            path = Path(file_path)

            if not path.is_absolute():
                path = self.project_root / path

            # 安全检查
            if not self.is_safe_path(path):
                return {
                    'success': False,
                    'error': '路径超出项目目录范围'
                }

            # 文件不存在时的处理
            if not path.exists():
                return self.write_file(file_path, content, encoding=encoding)

            # 读取现有内容
            try:
                with open(path, 'r', encoding=encoding) as f:
                    existing_content = f.read()
            except Exception as e:
                return {
                    'success': False,
                    'error': f'读取现有内容失败: {e}'
                }

            # 在开头插入
            new_content = content + separator + existing_content

            # 写入
            result = self.write_file(file_path, new_content, backup=True, encoding=encoding)

            if result['success']:
                result['operation'] = 'prepend'

            return result

        except Exception as e:
            logger.error(f"插入文件开头失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': file_path
            }

    def create_markdown_report(self, title: str, content: str,
                               output_path: str = None, encoding: str = 'utf-8') -> Dict:
        """
        创建 Markdown 格式报告

        Args:
            title: 报告标题
            content: 报告内容
            output_path: 输出路径（可选，默认生成带时间戳的文件名）
            encoding: 文件编码

        Returns:
            操作结果字典
        """
        try:
            # 生成默认文件名
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_title = re.sub(r'[^\w\s-]', '', title).strip()[:50]
                safe_title = re.sub(r'[-\s]+', '_', safe_title)
                output_path = f"reports/{safe_title}_{timestamp}.md"

            # 构建报告内容
            report_content = f"""# {title}

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

{content}
"""

            # 写入文件
            return self.write_file(output_path, report_content, encoding=encoding)

        except Exception as e:
            logger.error(f"创建报告失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def get_statistics(self) -> Dict:
        """获取写入器统计信息"""
        return {
            'project_root': str(self.project_root),
            'backup_dir': str(self.backup_dir),
            'total_writes': len(self.write_history),
            'allowed_extensions': list(self.ALLOWED_EXTENSIONS),
            'max_size_mb': self.MAX_FILE_SIZE_MB,
            'recent_writes': self.write_history[-10:]  # 最近10次写入
        }

    def list_backups(self, file_pattern: str = None) -> List[Dict]:
        """
        列出备份文件

        Args:
            file_pattern: 文件名模式（可选）

        Returns:
            备份文件列表
        """
        try:
            if not self.backup_dir.exists():
                return []

            backups = []

            for backup_file in self.backup_dir.iterdir():
                if backup_file.is_file():
                    stat = backup_file.stat()

                    # 筛选文件名模式
                    if file_pattern and file_pattern not in backup_file.name:
                        continue

                    backups.append({
                        'name': backup_file.name,
                        'path': str(backup_file),
                        'size': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
                    })

            # 按时间倒序
            backups.sort(key=lambda x: x['created'], reverse=True)

            return backups

        except Exception as e:
            logger.error(f"列出备份失败: {e}")
            return []


# ==================== 单例实例 ====================

_writer_instance = None

def get_document_writer(project_root: str = None) -> LocalDocumentWriter:
    """获取文档写入器单例"""
    global _writer_instance
    if _writer_instance is None:
        _writer_instance = LocalDocumentWriter(project_root)
    return _writer_instance


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import asyncio

    async def test_document_writer():
        """测试文档写入器"""
        print("=" * 80)
        print("本地文档写入器测试")
        print("=" * 80)

        writer = get_document_writer()

        # 1. 创建新文件
        print("\n[1] 创建新文件")
        result = writer.write_file(
            "test_output.md",
            "# 测试文档\n\n这是由 local_document_writer 创建的测试文件。"
        )
        if result['success']:
            print(f"✅ 文件已创建: {result['relative_path']}")
            print(f"   大小: {result['size']} 字符")
        else:
            print(f"❌ 创建失败: {result['error']}")

        # 2. 追加内容
        print("\n[2] 追加内容")
        result = writer.append_file(
            "test_output.md",
            "\n\n## 追加内容\n\n这是追加的内容。"
        )
        if result['success']:
            print(f"✅ 内容已追加")
        else:
            print(f"❌ 追加失败: {result['error']}")

        # 3. 创建报告
        print("\n[3] 创建报告")
        result = writer.create_markdown_report(
            "系统测试报告",
            "这是测试报告的内容。\n\n- 测试项1: 通过\n- 测试项2: 通过"
        )
        if result['success']:
            print(f"✅ 报告已创建: {result['relative_path']}")
        else:
            print(f"❌ 创建报告失败: {result['error']}")

        # 4. 统计信息
        print("\n[4] 写入器统计")
        stats = writer.get_statistics()
        print(f"✅ 项目根目录: {stats['project_root']}")
        print(f"✅ 总写入次数: {stats['total_writes']}")

        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)

    asyncio.run(test_document_writer())
