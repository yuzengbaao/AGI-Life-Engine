"""
Unit tests for core/processor.py module.

Tests data cleaning, transformation, and aggregation functionality.
"""

import pandas as pd
import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from core.processor import (
    DataProcessor,
    aggregate_data,
    clean_data,
    transform_data,
)


class TestCleanData:
    """Tests for clean_data function."""

    def test_clean_data_basic(self, sample_dataframe):
        """Test basic data cleaning."""
        result = clean_data(sample_dataframe)

        assert isinstance(result, DataFrame)
        assert len(result) == len(sample_dataframe)

    def test_clean_data_drop_na(self, sample_dataframe_with_issues):
        """Test cleaning with drop_na=True."""
        result = clean_data(sample_dataframe_with_issues, drop_na=True)

        # Should drop rows with NA values
        assert result.isnull().sum().sum() == 0

    def test_clean_data_fill_na(self, sample_dataframe_with_issues):
        """Test cleaning with fill_na strategy."""
        fill_strategy = {
            "Name": "Unknown",
            "Age": 0,
            "Score": 0.0,
            "City": "Unknown"
        }
        result = clean_data(
            sample_dataframe_with_issues,
            drop_na=False,
            fill_na=fill_strategy
        )

        # Should not have any NA values
        assert result.isnull().sum().sum() == 0

    def test_clean_data_drop_duplicates(self, sample_dataframe_with_issues):
        """Test cleaning drops duplicate rows."""
        initial_len = len(sample_dataframe_with_issues)
        result = clean_data(
            sample_dataframe_with_issues,
            drop_duplicates=True
        )

        # Should have fewer rows after dropping duplicates
        assert len(result) < initial_len

    def test_clean_data_strip_whitespace(self, sample_dataframe_with_issues):
        """Test cleaning strips whitespace from string columns."""
        result = clean_data(
            sample_dataframe_with_issues,
            strip_whitespace=True
        )

        # Check that " London " becomes "London"
        assert " London " not in result["City"].values
        assert "London" in result["City"].values

    def test_clean_data_empty_dataframe(self):
        """Test cleaning empty DataFrame."""
        empty_df = pd.DataFrame()
        result = clean_data(empty_df)

        assert_frame_equal(result, empty_df)

    def test_clean_data_invalid_input(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError):
            clean_data("not a dataframe")

    def test_clean_data_preserves_dtypes(self, sample_dataframe):
        """Test that cleaning preserves data types."""
        result = clean_data(sample_dataframe)

        assert result["Age"].dtype == sample_dataframe["Age"].dtype
        assert result["Score"].dtype == sample_dataframe["Score"].dtype


class TestTransformData:
    """Tests for transform_data function."""

    def test_transform_data_add_column(self, sample_dataframe):
        """Test adding a new column."""
        result = transform_data(
            sample_dataframe,
            operations=[
                {"operation": "add_column", "column": "Age_Double", "value": lambda df: df["Age"] * 2}
            ]
        )

        assert "Age_Double" in result.columns
        assert (result["Age_Double"] == sample_dataframe["Age"] * 2).all()

    def test_transform_data_rename_columns(self, sample_dataframe):
        """Test renaming columns."""
        result = transform_data(
            sample_dataframe,
            operations=[
                {"operation": "rename_columns", "mapping": {"Name": "Full_Name"}}
            ]
        )

        assert "Full_Name" in result.columns
        assert "Name" not in result.columns

    def test_transform_data_drop_columns(self, sample_dataframe):
        """Test dropping columns."""
        result = transform_data(
            sample_dataframe,
            operations=[
                {"operation": "drop_columns", "columns": ["Active"]}
            ]
        )

        assert "Active" not in result.columns
        assert "Name" in result.columns

    def test_transform_data_filter_rows(self, sample_dataframe):
        """Test filtering rows."""
        result = transform_data(
            sample_dataframe,
            operations=[
                {"operation": "filter", "condition": lambda df: df["Age"] > 30}
            ]
        )

        assert all(result["Age"] > 30)

    def test_transform_data_sort(self, sample_dataframe):
        """Test sorting data."""
        result = transform_data(
            sample_dataframe,
            operations=[
                {"operation": "sort", "by": "Age", "ascending": False}
            ]
        )

        assert result["Age"].is_monotonic_decreasing


class TestAggregateData:
    """Tests for aggregate_data function."""

    def test_aggregate_data_sum(self, sample_dataframe):
        """Test sum aggregation."""
        result = aggregate_data(
            sample_dataframe,
            group_by=["City"],
            aggregations={
                "Age": "sum",
                "Score": "sum"
            }
        )

        assert "City" in result.columns
        assert "Age" in result.columns
        assert "Score" in result.columns

    def test_aggregate_data_mean(self, sample_dataframe):
        """Test mean aggregation."""
        result = aggregate_data(
            sample_dataframe,
            group_by=["City"],
            aggregations={
                "Age": "mean",
                "Score": "mean"
            }
        )

        assert "City" in result.columns
        assert len(result) == sample_dataframe["City"].nunique()

    def test_aggregate_data_multiple_operations(self, sample_dataframe):
        """Test multiple aggregation operations."""
        result = aggregate_data(
            sample_dataframe,
            group_by=["City"],
            aggregations={
                "Age": ["mean", "min", "max"],
                "Score": ["mean", "std"]
            }
        )

        # Should have multi-level columns
        assert isinstance(result.columns, pd.MultiIndex)

    def test_aggregate_data_count(self, sample_dataframe):
        """Test count aggregation."""
        result = aggregate_data(
            sample_dataframe,
            group_by=["City"],
            aggregations={
                "Name": "count"
            }
        )

        assert "City" in result.columns
        assert "Name" in result.columns


class TestDataProcessor:
    """Tests for DataProcessor class."""

    def test_processor_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()

        assert processor is not None

    def test_processor_clean_and_validate(self, sample_dataframe_with_issues):
        """Test processor clean and validate method."""
        processor = DataProcessor()
        result = processor.clean_and_validate(sample_dataframe_with_issues)

        assert isinstance(result, DataFrame)
        # Should handle duplicates and NA values

    def test_processor_full_pipeline(self, sample_dataframe):
        """Test processor full pipeline."""
        processor = DataProcessor()
        result = processor.process_pipeline(
            sample_dataframe,
            clean_config={"drop_duplicates": True},
            transform_config=[
                {"operation": "add_column", "column": "Age_Times_Two", "value": lambda df: df["Age"] * 2}
            ]
        )

        assert isinstance(result, DataFrame)
        assert "Age_Times_Two" in result.columns

    def test_processor_with_custom_config(self, sample_dataframe):
        """Test processor with custom configuration."""
        processor = DataProcessor(config={"strict_mode": True})
        result = processor.clean(sample_dataframe)

        assert isinstance(result, DataFrame)


class TestErrorHandling:
    """Tests for error handling in data processing."""

    def test_clean_data_with_invalid_fill_na(self, sample_dataframe_with_issues):
        """Test that invalid fill_na dictionary is handled gracefully."""
        # Provide fill_na for non-existent columns
        fill_strategy = {
            "NonExistent": "value"
        }

        result = clean_data(
            sample_dataframe_with_issues,
            drop_na=False,
            fill_na=fill_strategy
        )

        # Should still work, just won't fill those columns
        assert isinstance(result, DataFrame)

    def test_transform_data_with_invalid_operation(self, sample_dataframe):
        """Test that invalid operation is handled."""
        with pytest.raises(Exception):
            transform_data(
                sample_dataframe,
                operations=[
                    {"operation": "invalid_operation"}
                ]
            )

    def test_aggregate_with_nonexistent_column(self, sample_dataframe):
        """Test aggregation with non-existent column."""
        with pytest.raises(KeyError):
            aggregate_data(
                sample_dataframe,
                group_by=["NonExistent"],
                aggregations={"Age": "mean"}
            )


class TestDataIntegrity:
    """Tests for data integrity during processing."""

    def test_clean_data_preserves_index(self, sample_dataframe):
        """Test that cleaning preserves index when intended."""
        custom_index_df = sample_dataframe.reset_index(drop=True)
        result = clean_data(custom_index_df, drop_na=False)

        assert isinstance(result.index, pd.RangeIndex)

    def test_transform_data_idempotent(self, sample_dataframe):
        """Test that some transformations are idempotent."""
        operation = {"operation": "sort", "by": "Age", "ascending": True}

        result1 = transform_data(sample_dataframe, operations=[operation])
        result2 = transform_data(result1, operations=[operation])

        assert_frame_equal(result1, result2)

    def test_aggregate_data_deterministic(self, sample_dataframe):
        """Test that aggregation is deterministic."""
        result1 = aggregate_data(
            sample_dataframe,
            group_by=["City"],
            aggregations={"Age": "mean"}
        )
        result2 = aggregate_data(
            sample_dataframe,
            group_by=["City"],
            aggregations={"Age": "mean"}
        )

        assert_frame_equal(result1, result2)


class TestPerformanceConsiderations:
    """Tests for performance-related aspects."""

    def test_clean_data_large_dataframe(self):
        """Test cleaning on larger DataFrame."""
        large_df = pd.DataFrame({
            "col1": range(10000),
            "col2": ["value"] * 10000,
            "col3": [1, 2, 3, 4, 5] * 2000
        })

        result = clean_data(large_df)

        assert len(result) == len(large_df)

    def test_processor_handles_sparse_data(self):
        """Test processor handles data with many missing values."""
        sparse_df = pd.DataFrame({
            "col1": [1, None, None, 4, 5],
            "col2": [None, None, 3, None, 5],
            "col3": [1, 2, 3, 4, 5]
        })

        processor = DataProcessor()
        result = processor.clean(sparse_df, drop_na=True)

        # Should only keep rows without NA
        assert len(result) > 0
        assert result.isnull().sum().sum() == 0
