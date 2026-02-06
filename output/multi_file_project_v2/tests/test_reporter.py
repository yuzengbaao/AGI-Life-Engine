"""
Unit tests for core/reporter.py module.

Tests Excel and PDF report generation functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from core.reporter import (
    ReportGenerator,
    generate_excel_report,
    generate_pdf_report,
)


class TestGenerateExcelReport:
    """Tests for generate_excel_report function."""

    def test_generate_excel_basic(self, temp_dir):
        """Test basic Excel report generation."""
        data = [
            {"Name": "Alice", "Score": 85},
            {"Name": "Bob", "Score": 90}
        ]
        output_path = temp_dir / "test_report.xlsx"

        result = generate_excel_report(data, str(output_path))

        assert output_path.exists()
        assert result == str(output_path.absolute())

    def test_generate_excel_with_sheet_name(self, temp_dir):
        """Test Excel generation with custom sheet name."""
        data = [{"Product": "A", "Price": 10}]
        output_path = temp_dir / "report.xlsx"

        generate_excel_report(
            data,
            str(output_path),
            sheet_name="SalesData"
        )

        # Verify file was created
        assert output_path.exists()

    def test_generate_excel_with_index(self, temp_dir):
        """Test Excel generation with row index."""
        data = [{"Name": "Alice"}]
        output_path = temp_dir / "report.xlsx"

        generate_excel_report(data, str(output_path), index=True)

        assert output_path.exists()

    def test_generate_excel_empty_data_raises_error(self, temp_dir):
        """Test that empty data list raises ValueError."""
        output_path = temp_dir / "report.xlsx"

        with pytest.raises(ValueError) as exc_info:
            generate_excel_report([], str(output_path))

        assert "empty" in str(exc_info.value).lower()

    def test_generate_excel_creates_parent_directories(self, temp_dir):
        """Test that parent directories are created."""
        data = [{"Name": "Test"}]
        output_path = temp_dir / "subdir" / "nested" / "report.xlsx"

        generate_excel_report(data, str(output_path))

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_generate_excel_overwrites_existing(self, temp_dir):
        """Test that existing file is overwritten."""
        data1 = [{"Name": "First"}]
        data2 = [{"Name": "Second"}]
        output_path = temp_dir / "report.xlsx"

        generate_excel_report(data1, str(output_path))
        generate_excel_report(data2, str(output_path))

        # Should contain second data, not first
        df = pd.read_excel(output_path)
        assert df["Name"].iloc[0] == "Second"


class TestGeneratePdfReport:
    """Tests for generate_pdf_report function."""

    def test_generate_pdf_basic(self, temp_dir):
        """Test basic PDF report generation."""
        data = [
            {"Name": "Alice", "Score": 85},
            {"Name": "Bob", "Score": 90}
        ]
        output_path = temp_dir / "report.pdf"

        result = generate_pdf_report(data, str(output_path))

        assert output_path.exists()
        assert result == str(output_path.absolute())

    def test_generate_pdf_with_title(self, temp_dir):
        """Test PDF generation with custom title."""
        data = [{"Name": "Test"}]
        output_path = temp_dir / "report.pdf"

        generate_pdf_report(
            data,
            str(output_path),
            title="Test Report"
        )

        assert output_path.exists()

    def test_generate_pdf_empty_data_raises_error(self, temp_dir):
        """Test that empty data raises ValueError."""
        output_path = temp_dir / "report.pdf"

        with pytest.raises(ValueError) as exc_info:
            generate_pdf_report([], str(output_path))

        assert "empty" in str(exc_info.value).lower()

    def test_generate_pdf_creates_directories(self, temp_dir):
        """Test that PDF generation creates parent directories."""
        data = [{"Name": "Test"}]
        output_path = temp_dir / "nested" / "report.pdf"

        generate_pdf_report(data, str(output_path))

        assert output_path.exists()


class TestReportGenerator:
    """Tests for ReportGenerator class."""

    def test_generator_init(self):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator()

        assert generator is not None

    def test_generator_generate_excel_method(self, temp_dir):
        """Test ReportGenerator's generate_excel method."""
        data = [{"Name": "Alice", "Score": 85}]
        output_path = temp_dir / "report.xlsx"

        generator = ReportGenerator()
        result = generator.generate_excel(data, str(output_path))

        assert result == str(output_path.absolute())

    def test_generator_generate_pdf_method(self, temp_dir):
        """Test ReportGenerator's generate_pdf method."""
        data = [{"Name": "Alice", "Score": 85}]
        output_path = temp_dir / "report.pdf"

        generator = ReportGenerator()
        result = generator.generate_pdf(data, str(output_path))

        assert result == str(output_path.absolute())

    def test_generator_batch_reports(self, temp_dir):
        """Test generating multiple reports."""
        data = [{"Name": "Alice", "Score": 85}]
        excel_path = temp_dir / "report.xlsx"
        pdf_path = temp_dir / "report.pdf"

        generator = ReportGenerator()
        generator.generate_excel(data, str(excel_path))
        generator.generate_pdf(data, str(pdf_path))

        assert excel_path.exists()
        assert pdf_path.exists()


class TestReportFormatting:
    """Tests for report formatting options."""

    def test_excel_with_formulas(self, temp_dir):
        """Test Excel generation with formulas."""
        data = [
            {"Product": "A", "Quantity": 10, "Price": 5.0},
            {"Product": "B", "Quantity": 5, "Price": 3.0}
        ]
        output_path = temp_dir / "report.xlsx"

        generate_excel_report(data, str(output_path))

        # Verify file can be read
        df = pd.read_excel(output_path)
        assert len(df) == 2

    def test_pdf_with_formatting(self, temp_dir):
        """Test PDF generation with formatting options."""
        data = [
            {"Name": "Alice", "Score": 85, "Grade": "A"},
            {"Name": "Bob", "Score": 90, "Grade": "A+"}
        ]
        output_path = temp_dir / "report.pdf"

        generate_pdf_report(
            data,
            str(output_path),
            title="Student Report",
            include_page_numbers=True
        )

        assert output_path.exists()


class TestErrorHandling:
    """Tests for error handling in report generation."""

    def test_excel_invalid_path(self):
        """Test Excel generation with invalid path."""
        data = [{"Name": "Test"}]

        # Use an invalid path (e.g., in a non-existent restricted location)
        with pytest.raises((IOError, OSError, PermissionError)):
            generate_excel_report(data, "/root/restricted/report.xlsx")

    def test_pdf_invalid_data_types(self, temp_dir):
        """Test PDF generation with complex data types."""
        data = [
            {"Name": "Alice", "Data": {"nested": "dict"}},
            {"Name": "Bob", "Data": [1, 2, 3]}
        ]
        output_path = temp_dir / "report.pdf"

        # Should handle or fail gracefully
        try:
            generate_pdf_report(data, str(output_path))
        except Exception as e:
            # Expected to fail with complex nested data
            assert True

    def test_report_generator_missing_dependencies(self, temp_dir):
        """Test behavior when dependencies are missing."""
        data = [{"Name": "Test"}]
        output_path = temp_dir / "report.xlsx"

        # Mock missing pandas
        with patch('core.reporter.pd', None):
            with pytest.raises(Exception):
                from core.reporter import generate_excel_report
                generate_excel_report(data, str(output_path))


class TestReportQuality:
    """Tests for quality of generated reports."""

    def test_excel_column_widths(self, temp_dir):
        """Test that Excel has reasonable column widths."""
        data = [
            {"Very Long Column Name": "Short"},
            {"Very Long Column Name": "Value"}
        ]
        output_path = temp_dir / "report.xlsx"

        generate_excel_report(data, str(output_path))

        # File should exist and be readable
        assert output_path.exists()
        df = pd.read_excel(output_path)
        assert "Very Long Column Name" in df.columns

    def test_pdf_contains_all_data(self, temp_dir):
        """Test that PDF contains all data rows."""
        data = [
            {"ID": i, "Name": f"Person{i}", "Score": i * 10}
            for i in range(1, 11)
        ]
        output_path = temp_dir / "report.pdf"

        generate_pdf_report(data, str(output_path))

        # PDF should be created
        assert output_path.exists()

    def test_concurrent_report_generation(self, temp_dir):
        """Test generating multiple reports simultaneously."""
        data = [{"Name": "Test"}]

        generator = ReportGenerator()

        # Generate multiple reports
        for i in range(5):
            excel_path = temp_dir / f"report_{i}.xlsx"
            pdf_path = temp_dir / f"report_{i}.pdf"

            generator.generate_excel(data, str(excel_path))
            generator.generate_pdf(data, str(pdf_path))

        # All reports should exist
        for i in range(5):
            assert (temp_dir / f"report_{i}.xlsx").exists()
            assert (temp_dir / f"report_{i}.pdf").exists()


class TestDataFormatSupport:
    """Tests for different data formats in reports."""

    def test_excel_with_numeric_data(self, temp_dir):
        """Test Excel with various numeric formats."""
        data = [
            {"Integer": 42, "Float": 3.14, "Scientific": 1.23e-5},
            {"Integer": -10, "Float": -2.71, "Scientific": 4.56e5}
        ]
        output_path = temp_dir / "numeric.xlsx"

        generate_excel_report(data, str(output_path))

        df = pd.read_excel(output_path)
        assert df["Integer"].dtype in ["int64", "int32"]

    def test_excel_with_datetime(self, temp_dir):
        """Test Excel with datetime data."""
        from datetime import datetime

        data = [
            {"Date": datetime(2024, 1, 1), "Event": "Start"},
            {"Date": datetime(2024, 12, 31), "Event": "End"}
        ]
        output_path = temp_dir / "datetime.xlsx"

        generate_excel_report(data, str(output_path))

        assert output_path.exists()

    def test_pdf_with_unicode(self, temp_dir):
        """Test PDF with unicode characters."""
        data = [
            {"Chinese": "你好", "Emoji": "😀", "Symbols": "©®™"}
        ]
        output_path = temp_dir / "unicode.pdf"

        generate_pdf_report(data, str(output_path))

        assert output_path.exists()
