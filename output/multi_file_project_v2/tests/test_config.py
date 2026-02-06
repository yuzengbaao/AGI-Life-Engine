"""
Unit tests for config.py module.

Tests configuration loading, validation, and ConfigManager class.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config import (
    ConfigError,
    ConfigManager,
    LoadError,
    ValidationError,
    load_config,
    validate_config,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_success(self, temp_dir):
        """Test successful config loading from JSON file."""
        config_data = {
            "app_name": "TestApp",
            "version": "1.0.0",
            "debug": True
        }
        config_path = temp_dir / "test_config.json"
        config_path.write_text(json.dumps(config_data))

        result = load_config(config_path)

        assert result == config_data

    def test_load_config_file_not_found(self, temp_dir):
        """Test loading nonexistent file raises LoadError."""
        nonexistent_path = temp_dir / "nonexistent.json"

        with pytest.raises(LoadError) as exc_info:
            load_config(nonexistent_path)

        assert "not found" in str(exc_info.value).lower()

    def test_load_config_invalid_json(self, temp_dir):
        """Test loading file with invalid JSON raises LoadError."""
        config_path = temp_dir / "invalid.json"
        config_path.write_text("{ invalid json }")

        with pytest.raises(LoadError) as exc_info:
            load_config(config_path)

        assert "invalid json" in str(exc_info.value).lower()

    def test_load_config_with_path_object(self, temp_dir):
        """Test loading config with Path object."""
        config_data = {"app_name": "TestApp"}
        config_path = temp_dir / "test.json"
        config_path.write_text(json.dumps(config_data))

        result = load_config(Path(config_path))

        assert result == config_data


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_validate_config_success(self):
        """Test validation of valid config."""
        config = {
            "app_name": "TestApp",
            "version": "1.0.0",
            "debug": True,
            "database": {"host": "localhost"}
        }

        result = validate_config(config)

        assert result is True

    def test_validate_config_missing_required_key(self):
        """Test validation fails when required key is missing."""
        config = {
            "app_name": "TestApp"
            # Missing "version" key
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_config(config)

        assert "missing" in str(exc_info.value).lower()

    def test_validate_config_wrong_type(self):
        """Test validation fails with wrong data type."""
        config = {
            "app_name": "TestApp",
            "version": 123,  # Should be str
            "debug": True
        }

        with pytest.raises(ValidationError) as exc_info:
            validate_config(config)

        assert "version" in str(exc_info.value).lower()
        assert "str" in str(exc_info.value)

    def test_validate_config_optional_key_missing(self):
        """Test validation succeeds when optional key is missing."""
        config = {
            "app_name": "TestApp",
            "version": "1.0.0"
            # "debug" is optional, so it's okay that it's missing
        }

        result = validate_config(config)

        assert result is True


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_init_without_path(self):
        """Test initialization without file path."""
        manager = ConfigManager()

        assert manager._config_path is None
        assert manager._config == {}
        assert manager._is_validated is False

    def test_init_with_path(self, temp_dir):
        """Test initialization with file path."""
        config_path = temp_dir / "config.json"
        manager = ConfigManager(config_path)

        assert manager._config_path == config_path
        assert manager._config == {}
        assert manager._is_validated is False

    def test_load_success(self, temp_dir):
        """Test successful config loading."""
        config_data = {"app_name": "TestApp", "version": "1.0.0"}
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps(config_data))

        manager = ConfigManager(config_path)
        manager.load()

        assert manager._config == config_data
        assert manager._is_validated is False

    def test_load_without_path_raises_error(self):
        """Test loading without path raises LoadError."""
        manager = ConfigManager()

        with pytest.raises(LoadError):
            manager.load()

    def test_load_with_override_path(self, temp_dir):
        """Test loading with override path."""
        config_data = {"app_name": "OverrideApp"}
        config_path = temp_dir / "override.json"
        config_path.write_text(json.dumps(config_data))

        manager = ConfigManager("other.json")
        manager.load(config_path)

        assert manager._config == config_data
        assert manager._config_path == config_path

    def test_validate_success(self):
        """Test successful validation."""
        config = {
            "app_name": "TestApp",
            "version": "1.0.0",
            "debug": True
        }
        manager = ConfigManager()
        manager._config = config

        manager.validate()

        assert manager._is_validated is True

    def test_validate_failure(self):
        """Test validation failure."""
        config = {"app_name": "TestApp"}  # Missing required "version"
        manager = ConfigManager()
        manager._config = config

        with pytest.raises(ValidationError):
            manager.validate()

    def test_get_existing_key(self):
        """Test getting existing configuration value."""
        config = {"app_name": "TestApp", "version": "1.0.0"}
        manager = ConfigManager()
        manager._config = config

        result = manager.get("app_name")

        assert result == "TestApp"

    def test_get_nonexistent_key_with_default(self):
        """Test getting nonexistent key with default value."""
        manager = ConfigManager()
        manager._config = {"app_name": "TestApp"}

        result = manager.get("nonexistent", "default_value")

        assert result == "default_value"

    def test_get_nonexistent_key_without_default(self):
        """Test getting nonexistent key without default returns None."""
        manager = ConfigManager()
        manager._config = {"app_name": "TestApp"}

        result = manager.get("nonexistent")

        assert result is None

    def test_config_property(self):
        """Test config property returns raw config dict."""
        config = {"app_name": "TestApp"}
        manager = ConfigManager()
        manager._config = config

        assert manager.config == config

    def test_is_validated_property(self):
        """Test is_validated property returns validation status."""
        manager = ConfigManager()

        assert manager.is_validated is False

        manager._is_validated = True

        assert manager.is_validated is True

    def test_reload_success(self, temp_dir):
        """Test successful reload."""
        config_data = {"app_name": "TestApp"}
        config_path = temp_dir / "config.json"
        config_path.write_text(json.dumps(config_data))

        manager = ConfigManager(config_path)
        manager.load()

        # Modify file
        new_config = {"app_name": "ReloadedApp"}
        config_path.write_text(json.dumps(new_config))

        manager.reload()

        assert manager._config == new_config
        assert manager._is_validated is False

    def test_reload_without_path_raises_error(self):
        """Test reload without path raises LoadError."""
        manager = ConfigManager()

        with pytest.raises(LoadError) as exc_info:
            manager.reload()

        assert "no file path" in str(exc_info.value).lower()


class TestConfigExceptions:
    """Tests for custom exception classes."""

    def test_config_error(self):
        """Test ConfigError can be raised and caught."""
        with pytest.raises(ConfigError):
            raise ConfigError("Test error")

    def test_load_error_is_config_error(self):
        """Test LoadError inherits from ConfigError."""
        with pytest.raises(ConfigError):
            raise LoadError("Load error")

    def test_validation_error_is_config_error(self):
        """Test ValidationError inherits from ConfigError."""
        with pytest.raises(ConfigError):
            raise ValidationError("Validation error")
