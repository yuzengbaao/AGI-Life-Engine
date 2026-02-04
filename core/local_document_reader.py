#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
本地文档读取器
Local Document Reader for AGI

安全地读取本地项目区文档，支持多种格式
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LocalDocumentReader:
    """
    本地文档读取器

    安全特性:
    1. 路径白名单 - 只允许读取项目目录
    2. 文件类型限制 - 只允许安全格式
    3. 大小限制 - 防止读取过大文件
    4. 敏感文件过滤 - 排除密钥、配置等
    """

    # 允许读取的文件类型
    ALLOWED_EXTENSIONS = {
        '.md', '.txt', '.rst', '.py', '.js', '.ts', '.json',
        '.yaml', '.yml', '.xml', '.html', '.css'
    }

    # 排除的敏感文件模式
    SENSITIVE_PATTERNS = [
        r'\.env$',
        r'\.key$',
        r'\.pem$',
        r'password',
        r'secret',
        r'credential',
        r'token',
        r'__pycache__',
        r'\.git',
        r'node_modules',
        r'\.venv'
    ]

    # 文件大小限制 (MB)
    # 🆕 [2026-01-22] 提高限制以支持大型文件（如 vector_memory.json）
    MAX_FILE_SIZE_MB = 1000  # 设置为1GB，支持超大型数据文件

    def __init__(self, project_root: str = None):
        """
        初始化文档读取器

        Args:
            project_root: 项目根目录路径
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent

        self.project_root = Path(project_root).resolve()
        self.read_history = []

        logger.info(f"📖 本地文档读取器已初始化")
        logger.info(f"   项目根目录: {self.project_root}")
        logger.info(f"   允许的文件类型: {len(self.ALLOWED_EXTENSIONS)} 种")
        logger.info(f"   安全限制: {self.MAX_FILE_SIZE_MB}MB")

    def is_safe_path(self, file_path: Path) -> bool:
        """检查路径是否安全（在项目目录内）"""
        try:
            resolved_path = file_path.resolve()
            # 检查是否在项目根目录内
            resolved_path.relative_to(self.project_root)
            return True
        except ValueError:
            return False

    def is_safe_file(self, file_path: Path) -> bool:
        """检查文件是否安全读取"""
        # 检查文件扩展名（允许无扩展名的可执行文件）
        if file_path.suffix:
            if file_path.suffix.lower() not in self.ALLOWED_EXTENSIONS:
                return False
        
        # 检查敏感文件模式
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, str(file_path), re.IGNORECASE):
                return False

        # 检查文件大小
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.MAX_FILE_SIZE_MB:
                logger.warning(f"文件过大: {file_path.name} ({size_mb:.1f}MB)")
                return False
        except Exception as e:
            logger.warning(f"无法获取文件大小: {e}")
            return False

        return True

    def read_file(self, file_path: str) -> Dict[str, any]:
        """
        读取单个文件

        Args:
            file_path: 文件路径（相对或绝对）

        Returns:
            包含文件内容和元数据的字典
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

            if not path.exists():
                return {
                    'success': False,
                    'error': '文件不存在',
                    'path': str(path)
                }

            if not self.is_safe_file(path):
                return {
                    'success': False,
                    'error': '文件类型不安全或包含敏感信息',
                    'path': str(path)
                }

            # 读取文件
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 记录读取历史
            self.read_history.append({
                'path': str(path),
                'timestamp': datetime.now().isoformat(),
                'size': len(content)
            })

            return {
                'success': True,
                'path': str(path),
                'relative_path': str(path.relative_to(self.project_root)),
                'content': content,
                'size': len(content),
                'lines': len(content.split('\n')),
                'extension': path.suffix,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': file_path
            }

    def list_documents(self, directory: str = ".", pattern: str = "*", recursive: bool = True, max_results: int = 100) -> List[Dict]:
        """
        列出目录中的文档

        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            recursive: 是否递归查找
            max_results: 🆕 [2026-01-24] 最大返回数量，防止全量扫描（默认100）

        Returns:
            文档信息列表
        """
        try:
            dir_path = Path(directory)

            # 如果是相对路径，基于项目根目录
            if not dir_path.is_absolute():
                dir_path = self.project_root / dir_path

            # 安全检查
            if not self.is_safe_path(dir_path):
                return []

            if not dir_path.exists() or not dir_path.is_dir():
                return []

            documents = []

            # 🆕 [2026-01-24] 排除大型目录，避免全量扫描
            EXCLUDED_DIRS = {
                '.git', '__pycache__', 'node_modules', '.venv', 'venv',
                'backups', 'backbag', '.backups', 'pyvista-0.46.4',
                'data', 'logs', 'memory_db', '.mypy_cache', '.pytest_cache',
                'workspace'  # 🆕 [2026-01-28] 排除 workspace 目录，避免扫描旧测试记录
            }

            # 递归或非递归遍历
            if recursive:
                files = dir_path.rglob(pattern)
            else:
                files = dir_path.glob(pattern)

            scanned_count = 0
            for file_path in files:
                try:
                    # 🆕 [2026-01-24] 早期终止：达到最大数量后停止扫描
                    if len(documents) >= max_results:
                        logger.info(f"📂 已达最大返回数量 {max_results}，停止扫描（已扫描 {scanned_count} 个文件）")
                        break
                    
                    scanned_count += 1
                    
                    # 跳过符号链接，避免权限问题
                    if file_path.is_symlink():
                        continue
                    
                    # 🆕 [2026-01-24] 跳过排除目录中的文件
                    if any(excluded in file_path.parts for excluded in EXCLUDED_DIRS):
                        continue
                    
                    if file_path.is_file() and self.is_safe_file(file_path):
                        stat = file_path.stat()
                        documents.append({
                            'path': str(file_path),
                            'relative_path': str(file_path.relative_to(self.project_root)),
                            'name': file_path.name,
                            'extension': file_path.suffix,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                except (OSError, PermissionError) as e:
                    # 跳过无法访问的文件（如符号链接）
                    logger.debug(f"跳过无法访问的文件: {file_path.name} ({e})")
                    continue

            logger.info(f"📂 返回 {len(documents)} 个文档（扫描了 {scanned_count} 个文件）")
            return documents

        except Exception as e:
            logger.error(f"列出文档失败: {e}")
            return []

    def search_in_documents(self, query: str, directory: str = ".", max_results: int = 20) -> List[Dict]:
        """
        在文档中搜索关键词

        Args:
            query: 搜索关键词
            directory: 搜索目录
            max_results: 最大结果数

        Returns:
            匹配结果列表
        """
        try:
            documents = self.list_documents(directory, recursive=True)
            results = []

            for doc in documents[:max_results * 2]:  # 多检查一些文件
                result = self.read_file(doc['relative_path'])

                if result['success']:
                    content = result['content']
                    lines = content.split('\n')

                    # 搜索匹配行
                    matches = []
                    for i, line in enumerate(lines):
                        if query.lower() in line.lower():
                            matches.append({
                                'line_number': i + 1,
                                'content': line.strip(),
                                'preview': line.strip()[:100]
                            })

                    if matches:
                        results.append({
                            'file': doc['relative_path'],
                            'matches': matches[:5],  # 最多显示5个匹配
                            'total_matches': len(matches)
                        })

                if len(results) >= max_results:
                    break

            logger.info(f"🔍 搜索 '{query}' 在 {len(results)} 个文件中找到匹配")
            return results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_document_summary(self, file_path: str) -> Dict:
        """
        获取文档摘要

        Args:
            file_path: 文件路径

        Returns:
            文档摘要信息
        """
        result = self.read_file(file_path)

        if not result['success']:
            return result

        content = result['content']
        lines = content.split('\n')

        # 提取标题（Markdown）
        titles = []
        for line in lines:
            if line.startswith('#'):
                titles.append(line.strip())

        # 统计信息
        summary = {
            'success': True,
            'path': result['path'],
            'relative_path': result['relative_path'],
            'titles': titles[:10],  # 最多10个标题
            'total_lines': result['lines'],
            'total_chars': result['size'],
            'extension': result['extension'],
            'encoding': 'utf-8',
            'preview': content[:500]  # 前500字符预览
        }

        return summary

    def index_project_docs(self, exclude_dirs: List[str] = None, force_rebuild: bool = False) -> Dict:
        """
        🆕 [2026-01-24] 带持久化缓存的项目文档索引

        工作流程：
        1. 检查是否存在已保存的索引文件
        2. 如果存在且未过期（24小时内），直接返回缓存
        3. 如果不存在或过期，执行全量索引并保存

        Args:
            exclude_dirs: 排除的目录列表
            force_rebuild: 强制重建索引（忽略缓存）

        Returns:
            索引统计信息
        """
        import json
        
        # 索引文件路径
        index_file = self.project_root / "data" / "document_index.json"
        index_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 检查缓存是否有效（24小时内）
        cache_valid = False
        cached_index = None
        
        if not force_rebuild and index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    cached_index = json.load(f)
                
                # 检查时间戳
                cached_time = datetime.fromisoformat(cached_index.get('timestamp', '2000-01-01'))
                age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                
                if age_hours < 24:
                    cache_valid = True
                    logger.info(f"📂 使用缓存索引（{age_hours:.1f}小时前创建，包含 {cached_index.get('total_documents', 0)} 个文档）")
                else:
                    logger.info(f"📂 索引缓存已过期（{age_hours:.1f}小时），重新构建...")
                    
            except Exception as e:
                logger.warning(f"读取缓存索引失败: {e}，重新构建...")
        
        # 如果缓存有效，直接返回
        if cache_valid and cached_index:
            return cached_index
        
        # 执行全量索引
        logger.info("📂 开始全量文档索引...")
        
        try:
            if exclude_dirs is None:
                exclude_dirs = ['.git', '__pycache__', 'node_modules', '.venv', '.conda', 'venv', 
                               'backups', 'backbag', '.backups', 'pyvista-0.46.4', '.mypy_cache',
                               'workspace']  # 🆕 [2026-01-28] 排除 workspace 目录，避免扫描旧测试记录

            # 全量扫描（不限制数量）
            all_docs = self._full_scan_documents(".", exclude_dirs)

            # 按类型分组
            by_extension = {}
            for doc in all_docs:
                ext = doc['extension']
                if ext not in by_extension:
                    by_extension[ext] = []
                by_extension[ext].append(doc)

            # 读取重要文档的摘要
            important_docs = [doc for doc in all_docs
                            if doc['extension'] in ['.md', '.txt', '.rst']
                            and doc['size'] < 1024 * 1024 * 50]  # 小于50MB

            summaries = []
            for doc in important_docs[:50]:  # 最多索引50个重要文档
                try:
                    summary = self.get_document_summary(doc['relative_path'])
                    if summary['success']:
                        summaries.append({
                            'path': summary['relative_path'],
                            'titles': summary.get('titles', []),
                            'lines': summary.get('total_lines', 0),
                            'chars': summary.get('total_chars', 0)
                        })
                except Exception:
                    continue

            index_data = {
                'total_documents': len(all_docs),
                'by_extension': {k: len(v) for k, v in by_extension.items()},
                'indexed_summaries': len(summaries),
                'summaries': summaries,
                'documents': all_docs,  # 完整文档列表
                'timestamp': datetime.now().isoformat(),
                'exclude_dirs': exclude_dirs
            }
            
            # 保存索引到文件
            try:
                with open(index_file, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f, ensure_ascii=False, indent=2)
                logger.info(f"✅ 索引已保存到 {index_file}（{len(all_docs)} 个文档）")
            except Exception as e:
                logger.warning(f"保存索引失败: {e}")

            return index_data

        except Exception as e:
            logger.error(f"索引失败: {e}")
            return {
                'total_documents': 0,
                'error': str(e)
            }

    def _full_scan_documents(self, directory: str, exclude_dirs: List[str]) -> List[Dict]:
        """
        🆕 [2026-01-24] 全量扫描文档（仅供索引使用）
        
        与 list_documents 不同，这个方法会扫描所有文档用于构建索引
        """
        try:
            dir_path = Path(directory)
            if not dir_path.is_absolute():
                dir_path = self.project_root / dir_path

            if not self.is_safe_path(dir_path):
                return []

            if not dir_path.exists() or not dir_path.is_dir():
                return []

            documents = []
            exclude_set = set(exclude_dirs)

            for file_path in dir_path.rglob('*'):
                try:
                    if file_path.is_symlink():
                        continue
                    
                    # 跳过排除目录
                    if any(excluded in file_path.parts for excluded in exclude_set):
                        continue
                    
                    if file_path.is_file() and self.is_safe_file(file_path):
                        stat = file_path.stat()
                        documents.append({
                            'path': str(file_path),
                            'relative_path': str(file_path.relative_to(self.project_root)),
                            'name': file_path.name,
                            'extension': file_path.suffix,
                            'size': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                except (OSError, PermissionError):
                    continue

            logger.info(f"📂 全量扫描完成: {len(documents)} 个文档")
            return documents

        except Exception as e:
            logger.error(f"全量扫描失败: {e}")
            return []

    def get_cached_index(self) -> Optional[Dict]:
        """
        🆕 [2026-01-24] 获取缓存的索引（不触发重建）
        
        用于快速查询已知文档，不执行扫描
        """
        import json
        index_file = self.project_root / "data" / "document_index.json"
        
        if not index_file.exists():
            return None
            
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None

    def search_in_index(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        🆕 [2026-01-24] 在已缓存的索引中搜索（快速）
        
        不读取文件内容，只搜索文件名和路径
        """
        cached = self.get_cached_index()
        if not cached:
            logger.warning("⚠️ 未找到文档索引，请先执行 index 操作")
            return []
        
        query_lower = query.lower()
        results = []
        
        for doc in cached.get('documents', []):
            # 搜索文件名和路径
            if query_lower in doc['name'].lower() or query_lower in doc['relative_path'].lower():
                results.append(doc)
                if len(results) >= max_results:
                    break
        
        logger.info(f"🔍 在索引中搜索 '{query}': 找到 {len(results)} 个匹配")
        return results

    def get_statistics(self) -> Dict:
        """获取读取器统计信息"""
        return {
            'project_root': str(self.project_root),
            'total_read': len(self.read_history),
            'allowed_extensions': list(self.ALLOWED_EXTENSIONS),
            'max_size_mb': self.MAX_FILE_SIZE_MB,
            'recent_reads': self.read_history[-10:]  # 最近10次读取
        }

    def write_file(self, file_path: str, content: str, **kwargs) -> Dict:
        """
        写入文件（代理到 LocalDocumentWriter）

        此方法提供向后兼容性，将写入请求转发到 LocalDocumentWriter

        Args:
            file_path: 文件路径
            content: 文件内容
            **kwargs: 其他参数（create_dirs, backup, encoding等）

        Returns:
            操作结果字典
        """
        try:
            from core.local_document_writer import get_document_writer
            writer = get_document_writer(str(self.project_root))
            return writer.write_file(file_path, content, **kwargs)
        except ImportError as e:
            logger.error(f"无法导入 LocalDocumentWriter: {e}")
            return {
                'success': False,
                'error': f'LocalDocumentWriter 不可用: {e}'
            }

    def append_file(self, file_path: str, content: str, **kwargs) -> Dict:
        """
        追加内容到文件（代理到 LocalDocumentWriter）

        Args:
            file_path: 文件路径
            content: 要追加的内容
            **kwargs: 其他参数

        Returns:
            操作结果字典
        """
        try:
            from core.local_document_writer import get_document_writer
            writer = get_document_writer(str(self.project_root))
            return writer.append_file(file_path, content, **kwargs)
        except ImportError as e:
            logger.error(f"无法导入 LocalDocumentWriter: {e}")
            return {
                'success': False,
                'error': f'LocalDocumentWriter 不可用: {e}'
            }

    def edit_file(self, file_path: str, old_content: str, new_content: str, **kwargs) -> Dict:
        """
        编辑文件（代理到 LocalDocumentWriter）

        Args:
            file_path: 文件路径
            old_content: 要替换的旧内容
            new_content: 新内容
            **kwargs: 其他参数

        Returns:
            操作结果字典
        """
        try:
            from core.local_document_writer import get_document_writer
            writer = get_document_writer(str(self.project_root))
            return writer.edit_file(file_path, old_content, new_content, **kwargs)
        except ImportError as e:
            logger.error(f"无法导入 LocalDocumentWriter: {e}")
            return {
                'success': False,
                'error': f'LocalDocumentWriter 不可用: {e}'
            }


# ==================== 单例实例 ====================

_reader_instance = None

def get_document_reader(project_root: str = None) -> LocalDocumentReader:
    """获取文档读取器单例"""
    global _reader_instance
    if _reader_instance is None:
        _reader_instance = LocalDocumentReader(project_root)
    return _reader_instance


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import asyncio

    async def test_document_reader():
        """测试文档读取器"""
        print("=" * 80)
        print("本地文档读取器测试")
        print("=" * 80)

        reader = get_document_reader()

        # 1. 索引项目文档
        print("\n[1] 索引项目文档")
        index = reader.index_project_docs()
        print(f"✅ 总文档数: {index.get('total_documents', 0)}")
        print(f"✅ 已索引摘要: {index.get('indexed_summaries', 0)}")

        if 'by_extension' in index:
            print("\n文件类型分布:")
            for ext, count in sorted(index['by_extension'].items()):
                print(f"  {ext}: {count} 个文件")

        # 2. 读取示例文件
        print("\n[2] 读取README文件")
        readme_result = reader.read_file("README.md")
        if readme_result['success']:
            print(f"✅ 文件: {readme_result['relative_path']}")
            print(f"   大小: {readme_result['size']} 字符")
            print(f"   行数: {readme_result['lines']}")
            print(f"   预览: {readme_result['content'][:200]}...")
        else:
            print(f"❌ {readme_result['error']}")

        # 3. 搜索文档
        print("\n[3] 搜索包含'LLM'的文档")
        search_results = reader.search_in_documents("LLM", max_results=5)
        print(f"✅ 找到 {len(search_results)} 个匹配文件")
        for result in search_results:
            print(f"  - {result['file']}: {result['total_matches']} 处匹配")

        # 4. 统计信息
        print("\n[4] 读取器统计")
        stats = reader.get_statistics()
        print(f"✅ 项目根目录: {stats['project_root']}")
        print(f"✅ 总读取次数: {stats['total_read']}")

        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)

    asyncio.run(test_document_reader())
