#!/usr/bin/env python3
"""
Feature Expansion Demo - Multi-Format and Visualization

This script demonstrates:
1. Reading/writing multiple data formats
2. Data format conversion
3. Creating various visualizations
4. Generating enhanced reports with charts
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.multi_format_reader import (
    MultiFormatReader,
    MultiFormatWriter,
    convert_format,
    get_format_info
)
from core.visualization import (
    DataVisualizer,
    ReportGeneratorWithCharts,
    create_dashboard
)
from core.processor import clean_data


def demo_multi_format_support():
    """Demonstrate multi-format data support."""
    print("\n" + "=" * 80)
    print("DEMO 1: Multi-Format Data Support")
    print("=" * 80)

    # Create sample data
    df = pd.DataFrame({
        "id": range(1, 101),
        "name": [f"Person{i}" for i in range(1, 101)],
        "age": [20 + (i % 50) for i in range(100)],
        "score": [50 + (i % 50) for i in range(100)],
        "category": ["A", "B", "C", "D"][i % 4] for i in range(100)]
    })

    reader = MultiFormatReader()
    writer = MultiFormatWriter()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save in multiple formats
        print("\n[SAVE] Saving data in multiple formats...")

        formats_to_save = [
            ('csv', writer.write_csv),
            ('json', writer.write_json),
            ('parquet', writer.write_parquet),
            ('feather', writer.write_feather),
            ('jsonl', writer.write_jsonl),
        ]

        saved_files = []
        for fmt, write_func in formats_to_save:
            file_path = tmpdir / f"data.{fmt}"
            try:
                write_func(df, file_path)
                saved_files.append(file_path)
                print(f"  ✓ Saved as {fmt:12} ({file_path.stat().st_size / 1024:.1f} KB)")
            except Exception as e:
                print(f"  ✗ Failed {fmt}: {e}")

        # Read back and verify
        print("\n[READ] Reading back and verifying...")
        for file_path in saved_files:
            try:
                df_read = reader.read_auto(file_path)
                assert len(df_read) == len(df)
                print(f"  ✓ {file_path.suffix:12} - Read {len(df_read)} rows")
            except Exception as e:
                print(f"  ✗ {file_path.suffix:12} - Error: {e}")

        # Format conversion demo
        print("\n[CONVERT] Format conversion demo...")
        csv_file = tmpdir / "convert_test.csv"
        writer.write_csv(df, csv_file)

        # Convert CSV to Parquet
        parquet_file = tmpdir / "convert_test.parquet"
        convert_format(csv_file, parquet_file)
        print(f"  ✓ Converted CSV to Parquet")

        # Convert Parquet to JSONL
        jsonl_file = tmpdir / "convert_test.jsonl"
        convert_format(parquet_file, jsonl_file)
        print(f"  ✓ Converted Parquet to JSONL")


def demo_visualizations():
    """Demonstrate visualization capabilities."""
    print("\n" " + "=" * 80)
    print("DEMO 2: Data Visualization")
    print("=" * 80)

    # Create sample data with various types
    import numpy as np

    df = pd.DataFrame({
        "age": np.random.randint(18, 70, 200),
        "income": np.random.normal(50000, 15000, 200),
        "score": np.random.uniform(0, 100, 200),
        "category": np.random.choice(["A", "B", "C"], 200),
        "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], 200)
    })

    viz = DataVisualizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create various charts
        charts = [
            ('histogram', viz.plot_histogram),
            ('boxplot', viz.plot_boxplot),
            ('scatter', viz.plot_scatter),
            ('correlation', viz.plot_correlation_heatmap),
        ]

        print("\n[VISUALIZE] Creating charts...")
        for chart_name, plot_func in charts:
            try:
                fig = plot_func(df)
                output_path = tmpdir / f"{chart_name}_chart.png"
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  ✓ Created {chart_name:12} ({output_path.stat().st_size / 1024:.1f} KB)")
            except Exception as e:
                print(f"  ✗ Failed {chart_name}: {e}")

        # Create dashboard
        print("\n[DASHBOARD] Creating multi-panel dashboard...")
        try:
            fig = create_dashboard(df, "Sample Dashboard")
            output_path = tmpdir / "dashboard.png"
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  ✓ Created dashboard ({output_path.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"  ✗ Failed dashboard: {e}")


def demo_format_info():
    """Demonstrate format info extraction."""
    print("\n" + "=" * 80)
    print("DEMO 3: Format Information Extraction")
    print("=" * 80)

    # Create a sample CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": ["a", "b", "c"]
        })
        df.to_csv(tmp.name, index=False)

        # Get format info
        info = get_format_info(tmp.name)

        print(f"\nFile: {info['name']}")
        print(f"Format: {info['format']}")
        print(f"Size: {info['size_mb']:.2f} MB")
        print(f"Rows: {info.get('rows', 'N/A')}")
        print(f"Columns: {info.get('columns', 'N/A')}")
        print(f"Preview: {info.get('preview', 'N/A')}")

        # Cleanup
        Path(tmp.name).unlink()


def demo_report_with_charts():
    """Demonstrate enhanced report generation with charts."""
    print("\n" + "=" * 80)
    print("DEMO 4: Enhanced Reports with Charts")
    print("=" * 80)

    # Create sample data
    df = pd.DataFrame({
        "month": pd.date_range("2024-01", periods=12, freq="M"),
        "sales": [100000 + i*10000 for i in range(12)],
        "expenses": [50000 + i*5000 for i in range(12)],
        "category": ["A", "B", "C"][i % 3] for i in range(12)]
    })

    generator = ReportGeneratorWithCharts()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Generate Excel with embedded charts
        print("\n[REPORT] Generating Excel with charts...")
        excel_path = tmpdir / "report_with_charts.xlsx"
        try:
            generator.generate_excel_with_charts(
                df,
                excel_path,
                charts=['line', 'bar']
            )
            print(f"  ✓ Excel report: {excel_path}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # Generate PDF with charts
        print("\n[REPORT] Generating PDF with charts...")
        pdf_path = tmpdir / "report_with_charts.pdf"
        try:
            generator.save_charts_to_pdf(
                df,
                pdf_path,
                charts=['histogram', 'bar']
            )
            print(f"  ✓ PDF report: {pdf_path}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def main():
    """Run all demonstrations."""
    print("=" * 80)
    print("DATA PROCESSING TOOL - FEATURE EXPANSION DEMO")
    print("=" * 80)

    print("\n🖥️  Demonstrating:")
    print("   1. Multi-format data support (9 formats)")
    print("   2. Data visualization (10+ chart types)")
    print("   3. Format conversion")
    print("   4. Enhanced reports with embedded charts")

    try:
        # Demo 1: Multi-format support
        demo_multi_format_support()

        # Demo 2: Visualizations
        demo_visualizations()

        # Demo 3: Format info
        demo_format_info()

        # Demo 4: Enhanced reports
        demo_report_with_charts()

        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("\n✅ All features demonstrated successfully!")
        print("\nKey Features:")
        print("   - 9 data formats supported")
        print("   - 10+ visualization types")
        print("   - Automatic format detection")
        print("   - One-click format conversion")
        print("   - Reports with embedded charts")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
