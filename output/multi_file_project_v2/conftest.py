"""
Pytest configuration and shared fixtures.

This file contains common fixtures and configuration for all tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import pytest


# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files.

    Yields:
        Path: Path to temporary directory

    Example:
        >>> def test_something(temp_dir):
        ...     file = temp_dir / "test.txt"
        ...     file.write_text("content")
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_path(temp_dir: Path) -> Path:
    """Create a sample configuration file for testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Path to sample config file

    Example:
        >>> def test_config_loading(sample_config_path):
        ...     config = load_config(sample_config_path)
    """
    import json

    config_data = {
        "app_name": "TestApp",
        "version": "1.0.0",
        "debug": True,
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db"
        }
    }

    config_path = temp_dir / "test_config.json"
    config_path.write_text(json.dumps(config_data, indent=2))

    return config_path


@pytest.fixture
def sample_csv_path(temp_dir: Path) -> Path:
    """Create a sample CSV file for testing.

    Args:
        temp_dir: Temporary directory fixture

    Returns:
        Path: Path to sample CSV file
    """
    import pandas as pd

    data = {
        "Name": ["Alice", "Bob", "Charlie", "David"],
        "Age": [25, 30, 35, 28],
        "Score": [85.5, 90.0, 78.5, 92.0],
        "City": ["New York", "London", "Paris", "Tokyo"]
    }

    df = pd.DataFrame(data)
    csv_path = temp_dir / "sample_data.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame for testing.

    Returns:
        pd.DataFrame: Sample DataFrame

    Example:
        >>> def test_dataframe_processing(sample_dataframe):
        ...     result = clean_data(sample_dataframe)
    """
    import pandas as pd

    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "Age": [25, 30, 35, 28, 32],
        "Score": [85.5, 90.0, 78.5, 92.0, 88.0],
        "City": ["New York", "London", "Paris", "Tokyo", "Berlin"],
        "Active": [True, True, False, True, True]
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_with_issues():
    """Create a DataFrame with common data quality issues for testing.

    Returns:
        pd.DataFrame: DataFrame with missing values, duplicates, etc.

    Example:
        >>> def test_data_cleaning(sample_dataframe_with_issues):
        ...     cleaned = clean_data(sample_dataframe_with_issues)
    """
    import pandas as pd
    import numpy as np

    data = {
        "Name": ["Alice", "Bob", None, "Bob", "Eve"],  # None + duplicate
        "Age": [25, 30, 35, 30, None],  # None + duplicate
        "Score": [85.5, 90.0, np.nan, 90.0, 88.0],  # NaN + duplicate
        "City": ["New York", " London ", "Paris", " London ", "Berlin"]  # Extra spaces
    }

    return pd.DataFrame(data)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing.

    Returns:
        Mock: Mock logger object

    Example:
        >>> def test_logging(mock_logger):
        ...     logger.info("Test message")
    """
    from unittest.mock import Mock
    import logging

    logger = Mock(spec=logging.Logger)
    return logger


# Pytest hooks

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower, may require external deps)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark test files in test_integration.py as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
