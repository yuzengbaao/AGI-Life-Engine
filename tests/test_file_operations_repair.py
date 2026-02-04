"""
测试 file_operations 工具修复效果

验证:
1. detect_file_type() 函数正确性
2. read_file_safe() 函数处理文本和二进制文件
3. 原失败用例（SQLite数据库）现在能正确处理
4. 各种文件类型检测
5. 边界条件和错误处理
"""

import sys
import os
import pytest
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_tools_collection import detect_file_type, read_file_safe, FileOperationTool


class TestDetectFileType:
    """测试 detect_file_type() 函数"""
    
    def test_text_file_by_extension(self, tmp_path):
        """测试通过扩展名识别文本文件"""
        # 创建测试文件
        test_files = {
            'test.txt': 'text',
            'readme.md': 'text',
            'script.py': 'text',
            'data.json': 'text',
            'config.yaml': 'text',
        }
        
        for filename, expected_type in test_files.items():
            file_path = tmp_path / filename
            file_path.write_text("Test content")
            result = detect_file_type(file_path)
            assert result == expected_type, f"{filename} 应该被识别为 {expected_type}"
    
    def test_database_file_by_extension(self, tmp_path):
        """测试数据库文件识别"""
        file_path = tmp_path / "test.db"
        file_path.write_bytes(b'SQLite format 3\x00' + b'\x00' * 100)
        
        result = detect_file_type(file_path)
        assert result == 'database'
    
    def test_image_file_by_extension(self, tmp_path):
        """测试图像文件识别"""
        image_files = ['test.jpg', 'test.png', 'test.gif']
        
        for filename in image_files:
            file_path = tmp_path / filename
            file_path.write_bytes(b'\xFF\xD8\xFF' + b'\x00' * 100)  # JPEG header
            result = detect_file_type(file_path)
            assert result == 'image'
    
    def test_archive_file_by_extension(self, tmp_path):
        """测试压缩文件识别"""
        file_path = tmp_path / "test.zip"
        file_path.write_bytes(b'PK\x03\x04' + b'\x00' * 100)
        
        result = detect_file_type(file_path)
        assert result == 'archive'
    
    def test_binary_detection_by_null_bytes(self, tmp_path):
        """测试通过NULL字节识别二进制文件"""
        file_path = tmp_path / "binary.dat"
        file_path.write_bytes(b'Some data\x00with\x00null\x00bytes')
        
        result = detect_file_type(file_path)
        assert result == 'binary'
    
    def test_text_detection_by_content(self, tmp_path):
        """测试通过内容识别文本文件（无明确扩展名）"""
        file_path = tmp_path / "no_extension"
        file_path.write_text("This is plain text content")
        
        result = detect_file_type(file_path)
        assert result == 'text'
    
    def test_nonexistent_file(self, tmp_path):
        """测试不存在的文件"""
        file_path = tmp_path / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            detect_file_type(file_path)


class TestReadFileSafe:
    """测试 read_file_safe() 函数"""
    
    def test_read_text_file_utf8(self, tmp_path):
        """测试读取UTF-8文本文件"""
        file_path = tmp_path / "test.txt"
        content = "Hello World\n这是中文测试\n"
        file_path.write_text(content, encoding='utf-8')
        
        result = read_file_safe(file_path)
        
        assert result['file_type'] == 'text'
        assert result['content'] == content
        assert result['encoding'] == 'utf-8'
        assert result['lines'] == 2
    
    def test_read_text_file_gbk(self, tmp_path):
        """测试读取GBK编码文本文件"""
        file_path = tmp_path / "gbk.txt"
        content = "这是GBK编码的中文"
        file_path.write_bytes(content.encode('gbk'))
        
        result = read_file_safe(file_path)
        
        assert result['file_type'] == 'text'
        assert result['content'] == content
        assert result['encoding'] == 'gbk'
    
    def test_read_binary_database_file(self, tmp_path):
        """测试读取二进制数据库文件（原失败用例场景）"""
        file_path = tmp_path / "test.db"
        # 模拟SQLite数据库文件头
        file_path.write_bytes(b'SQLite format 3\x00' + b'\x00' * 1000)
        
        result = read_file_safe(file_path)
        
        assert result['file_type'] == 'database'
        assert result['content'] is None  # 二进制文件无文本内容
        assert 'binary_info' in result
        assert 'SQLite' in result['binary_info']['suggestion'] or 'database' in result['binary_info']['message']
    
    def test_read_binary_image_file(self, tmp_path):
        """测试读取二进制图像文件"""
        file_path = tmp_path / "test.jpg"
        file_path.write_bytes(b'\xFF\xD8\xFF' + b'\x00' * 500)
        
        result = read_file_safe(file_path)
        
        assert result['file_type'] == 'image'
        assert result['content'] is None
        assert 'binary_info' in result
        assert 'hex_preview' in result['binary_info']
    
    def test_file_size_limit(self, tmp_path):
        """测试文件大小限制"""
        file_path = tmp_path / "large.txt"
        # 创建2MB文件
        large_content = "x" * (2 * 1024 * 1024)
        file_path.write_text(large_content)
        
        # 默认限制10MB - 应该成功
        result = read_file_safe(file_path, max_size_mb=10)
        assert result['content'] == large_content
        
        # 限制1MB - 应该失败
        with pytest.raises(ValueError, match="文件过大"):
            read_file_safe(file_path, max_size_mb=1)
    
    def test_nonexistent_file(self, tmp_path):
        """测试不存在的文件"""
        file_path = tmp_path / "nonexistent.txt"
        
        with pytest.raises(FileNotFoundError):
            read_file_safe(file_path)


class TestFileOperationToolIntegration:
    """测试 FileOperationTool 工具集成"""
    
    def setup_method(self):
        """每个测试前初始化工具"""
        self.tool = FileOperationTool()
    
    def test_read_text_file_success(self, tmp_path):
        """测试读取文本文件成功"""
        file_path = tmp_path / "test.txt"
        content = "Test content\n中文内容\n"
        file_path.write_text(content, encoding='utf-8')
        
        result = self.tool.execute(operation="read", file_path=str(file_path))
        
        assert result.success is True
        assert result.data['content'] == content
        assert result.data['file_type'] == 'text'
        assert result.data['encoding'] == 'utf-8'
    
    def test_original_failure_case_database(self, tmp_path):
        """测试原失败用例 - SQLite数据库文件"""
        # 模拟原失败场景: 读取 agi_text_memory.db
        db_path = tmp_path / "agi_text_memory.db"
        db_path.write_bytes(b'SQLite format 3\x00' + b'\x00' * 1000)
        
        result = self.tool.execute(operation="read", file_path=str(db_path))
        
        # 现在应该成功，但返回二进制文件信息
        assert result.success is True
        assert result.data['file_type'] == 'database'
        assert result.data['content'] is None
        assert 'binary_info' in result.data
        assert 'SQLite' in result.data['binary_info']['suggestion']
    
    def test_read_with_custom_size_limit(self, tmp_path):
        """测试自定义文件大小限制"""
        file_path = tmp_path / "large.txt"
        large_content = "x" * (2 * 1024 * 1024)  # 2MB
        file_path.write_text(large_content)
        
        # 使用自定义限制
        result = self.tool.execute(
            operation="read", 
            file_path=str(file_path),
            max_size_mb=5
        )
        assert result.success is True
        
        # 超过限制
        result = self.tool.execute(
            operation="read", 
            file_path=str(file_path),
            max_size_mb=1
        )
        assert result.success is False
        assert "文件过大" in result.error
    
    def test_read_nonexistent_file(self, tmp_path):
        """测试读取不存在的文件"""
        file_path = tmp_path / "nonexistent.txt"
        
        result = self.tool.execute(operation="read", file_path=str(file_path))
        
        assert result.success is False
        assert "不存在" in result.error
    
    def test_write_operation_unaffected(self, tmp_path):
        """测试写入操作未受影响"""
        file_path = tmp_path / "write_test.txt"
        content = "New content"
        
        result = self.tool.execute(
            operation="write",
            file_path=str(file_path),
            content=content
        )
        
        assert result.success is True
        assert result.data['written'] is True
        assert file_path.read_text() == content
    
    def test_info_operation_unaffected(self, tmp_path):
        """测试info操作未受影响"""
        file_path = tmp_path / "info_test.txt"
        file_path.write_text("Test")
        
        result = self.tool.execute(operation="info", file_path=str(file_path))
        
        assert result.success is True
        assert result.data['name'] == 'info_test.txt'
        assert result.data['is_file'] is True
    
    def test_execution_time_recorded(self, tmp_path):
        """测试执行时间被记录"""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Test")
        
        result = self.tool.execute(operation="read", file_path=str(file_path))
        
        assert result.execution_time is not None
        assert result.execution_time >= 0


class TestRealWorldScenarios:
    """真实场景测试"""
    
    def test_read_actual_database_if_exists(self):
        """测试读取实际的数据库文件（如果存在）"""
        db_path = Path("agi_text_memory.db")
        
        if not db_path.exists():
            pytest.skip("实际数据库文件不存在，跳过测试")
        
        tool = FileOperationTool()
        result = tool.execute(operation="read", file_path=str(db_path))
        
        # 应该成功，识别为database
        assert result.success is True
        assert result.data['file_type'] == 'database'
        print(f"\n✅ 成功处理真实数据库: {db_path}")
        print(f"   文件大小: {result.data['size_mb']:.2f} MB")
        print(f"   建议: {result.data['binary_info']['suggestion']}")


if __name__ == "__main__":
    print("=" * 60)
    print("File Operations 修复测试")
    print("=" * 60)
    
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-p", "no:warnings"
    ])
