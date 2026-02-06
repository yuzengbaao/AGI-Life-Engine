"""
Unit tests for utils/helpers.py module.

Tests helper utility functions for file I/O, logging, and formatting.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from utils.helpers import (
    format_bytes,
    read_file,
    setup_logger,
    write_file,
)


class TestReadFile:
    """Tests for read_file function."""

    def test_read_file_success(self, temp_dir):
        """Test successful file reading."""
        test_file = temp_dir / "test.txt"
        content = "Hello, World!"
        test_file.write_text(content, encoding="utf-8")

        result = read_file(str(test_file))

        assert result == content

    def test_read_file_with_default_encoding(self, temp_dir):
        """Test reading file with default UTF-8 encoding."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("UTF-8 content: 你好世界", encoding="utf-8")

        result = read_file(str(test_file))

        assert "你好世界" in result

    def test_read_file_with_custom_encoding(self, temp_dir):
        """Test reading file with custom encoding."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Content", encoding="latin-1")

        result = read_file(str(test_file), encoding="latin-1")

        assert result == "Content"

    def test_read_file_with_path_object(self, temp_dir):
        """Test reading file with Path object."""
        test_file = temp_dir / "test.txt"
        content = "Test content"
        test_file.write_text(content)

        result = read_file(test_file)

        assert result == content

    def test_read_file_not_found(self, temp_dir):
        """Test reading nonexistent file raises FileNotFoundError."""
        nonexistent = temp_dir / "nonexistent.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            read_file(str(nonexistent))

        assert "not found" in str(exc_info.value).lower()

    def test_read_file_permission_denied(self, temp_dir):
        """Test reading file without permissions."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Content")

        # Make file unreadable (on Unix systems)
        try:
            os.chmod(test_file, 0o000)

            with pytest.raises(IOError):
                read_file(str(test_file))
        finally:
            # Clean up - restore permissions for deletion
            try:
                os.chmod(test_file, 0o644)
            except:
                pass


class TestWriteFile:
    """Tests for write_file function."""

    def test_write_file_success(self, temp_dir):
        """Test successful file writing."""
        test_file = temp_dir / "output.txt"
        content = "Test content"

        write_file(str(test_file), content)

        assert test_file.exists()
        assert test_file.read_text(encoding="utf-8") == content

    def test_write_file_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created if they don't exist."""
        test_file = temp_dir / "subdir" / "nested" / "output.txt"
        content = "Nested content"

        write_file(str(test_file), content)

        assert test_file.exists()
        assert test_file.parent.exists()
        assert test_file.read_text() == content

    def test_write_file_with_custom_encoding(self, temp_dir):
        """Test writing file with custom encoding."""
        test_file = temp_dir / "output.txt"
        content = "Latin content: éàù"

        write_file(str(test_file), content, encoding="latin-1")

        result = test_file.read_text(encoding="latin-1")
        assert result == content

    def test_write_file_overwrites_existing(self, temp_dir):
        """Test that write_file overwrites existing file."""
        test_file = temp_dir / "output.txt"
        test_file.write_text("Old content")

        new_content = "New content"
        write_file(str(test_file), new_content)

        assert test_file.read_text() == new_content

    def test_write_file_permission_error(self, temp_dir):
        """Test writing to read-only directory."""
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()

        try:
            # Make directory read-only (on Unix systems)
            os.chmod(readonly_dir, 0o444)

            test_file = readonly_dir / "test.txt"

            with pytest.raises(IOError):
                write_file(str(test_file), "Content")
        finally:
            # Clean up - restore permissions for deletion
            try:
                os.chmod(readonly_dir, 0o755)
            except:
                pass


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_setup_logger_returns_logger(self):
        """Test that setup_logger returns a Logger instance."""
        logger = setup_logger("test_logger")

        assert isinstance(logger, logging.Logger)

    def test_setup_logger_with_default_params(self):
        """Test logger setup with default parameters."""
        logger = setup_logger("test_logger")

        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logger_with_debug_level(self):
        """Test logger setup with DEBUG level."""
        logger = setup_logger("test_logger", level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_setup_logger_with_file_handler(self, temp_dir):
        """Test logger setup with file handler."""
        log_file = temp_dir / "test.log"

        logger = setup_logger("test_logger", log_file=str(log_file))

        # Check that file handler was added
        has_file_handler = any(
            isinstance(h, logging.FileHandler) for h in logger.handlers
        )
        assert has_file_handler
        assert log_file.exists()

    def test_setup_logger_custom_format(self):
        """Test logger with custom format."""
        custom_format = "%(levelname)s - %(message)s"
        logger = setup_logger("test_logger", log_format=custom_format)

        handler = logger.handlers[0]
        formatter = handler.formatter

        assert custom_format in formatter._fmt

    def test_setup_logger_prevents_duplicate_handlers(self):
        """Test that calling setup_logger multiple times doesn't add duplicate handlers."""
        logger_name = "test_logger_duplicate"

        logger1 = setup_logger(logger_name)
        initial_handlers = len(logger1.handlers)

        logger2 = setup_logger(logger_name)
        final_handlers = len(logger2.handlers)

        # Should return same logger with same number of handlers
        assert initial_handlers == final_handlers
        assert logger1 is logger2

    def test_setup_logger_logs_to_file(self, temp_dir, caplog):
        """Test that logger actually writes to file."""
        log_file = temp_dir / "test.log"
        logger = setup_logger("test_file_logger", log_file=str(log_file))

        test_message = "Test log message"
        logger.info(test_message)

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        # Check file content
        log_content = log_file.read_text()
        assert test_message in log_content


class TestFormatBytes:
    """Tests for format_bytes function."""

    def test_format_bytes_zero(self):
        """Test formatting 0 bytes."""
        result = format_bytes(0)

        assert result == "0.00 B"

    def test_format_bytes_less_than_kb(self):
        """Test formatting bytes less than 1 KB."""
        result = format_bytes(512)

        assert result == "512.00 B"

    def test_format_bytes_kilobytes(self):
        """Test formatting kilobytes."""
        result = format_bytes(1024)

        assert result == "1.00 KB"

    def test_format_bytes_megabytes(self):
        """Test formatting megabytes."""
        result = format_bytes(1024 * 1024)

        assert result == "1.00 MB"

    def test_format_bytes_gigabytes(self):
        """Test formatting gigabytes."""
        result = format_bytes(1024 * 1024 * 1024)

        assert result == "1.00 GB"

    def test_format_bytes_terabytes(self):
        """Test formatting terabytes."""
        result = format_bytes(1024 * 1024 * 1024 * 1024)

        assert result == "1.00 TB"

    def test_format_bytes_petabytes(self):
        """Test formatting petabytes."""
        result = format_bytes(1024 * 1024 * 1024 * 1024 * 1024)

        assert result == "1.00 PB"

    def test_format_bytes_realistic_value(self):
        """Test formatting realistic file size."""
        result = format_bytes(1536000)

        assert result == "1.46 MB"

    def test_format_bytes_invalid_type(self):
        """Test format_bytes raises TypeError for non-integer input."""
        with pytest.raises(TypeError):
            format_bytes("1024")

        with pytest.raises(TypeError):
            format_bytes(1024.5)

    def test_format_bytes_negative(self):
        """Test format_bytes handles negative values."""
        result = format_bytes(-1024)

        # Should handle negative, though not typical
        assert "KB" in result or "B" in result
