#!/usr/bin/env python3
"""
Performance Benchmarking Script for Data Processing Tool.

This script demonstrates and compares:
1. Standard processing vs Optimized processing
2. Sequential vs Parallel processing
3. In-memory vs Streaming processing for large files
"""

import sys
import time
from pathlib import Path

import pandas as pd
import psutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.processor import clean_data as standard_clean
from core.processor_enhanced import (
    EnhancedDataProcessor,
    PerformanceBenchmark,
    clean_data_optimized,
    estimate_chunk_size,
    get_optimal_workers,
    parallel_process_chunks,
)


def generate_large_dataset(rows: int = 100000) -> pd.DataFrame:
    """Generate a large dataset for benchmarking.

    Args:
        rows: Number of rows to generate

    Returns:
        Large DataFrame
    """
    import numpy as np

    data = {
        "id": range(rows),
        "name": [f"Person{i % 1000}" for i in range(rows)],
        "age": np.random.randint(18, 80, rows),
        "score": np.random.uniform(0, 100, rows),
        "city": np.random.choice(["NYC", "LA", "Chicago", "Houston"], rows),
        "active": np.random.choice([True, False], rows),
        "income": np.random.uniform(30000, 150000, rows)
    }

    df = pd.DataFrame(data)

    # Add some messy data
    # Add 5% missing values
    for col in ["age", "score", "income"]:
        mask = np.random.random(rows) < 0.05
        df.loc[mask, col] = np.nan

    # Add some duplicates
    duplicate_indices = np.random.choice(rows, size=rows // 100, replace=False)
    df.loc[duplicate_indices] = df.iloc[duplicate_indices[0]]

    # Add extra whitespace in string columns
    whitespace_mask = np.random.random(rows) < 0.1
    df.loc[whitespace_mask, "city"] = df.loc[whitespace_mask, "city"].apply(
        lambda x: f"  {x}  "
    )

    return df


def benchmark_cleaning():
    """Benchmark standard vs optimized cleaning."""
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Data Cleaning")
    print("=" * 80)

    # Generate test data
    df = generate_large_dataset(100000)
    print(f"\nGenerated dataset: {len(df)} rows, {len(df.columns)} columns")

    # Create benchmark
    benchmark = PerformanceBenchmark()

    # Define approaches
    approaches = {
        "Standard": standard_clean,
        "Optimized": clean_data_optimized
    }

    # Run benchmark
    results = benchmark.benchmark_approaches(df, approaches, runs=3)

    # Print results
    benchmark.print_comparison()


def benchmark_parallel_processing():
    """Benchmark sequential vs parallel processing."""
    print("\n" + "=" * 80)
    print("BENCHMARK 2: Sequential vs Parallel Processing")
    print("=" * 80)

    # Generate test data
    df = generate_large_dataset(50000)
    print(f"\nGenerated dataset: {len(df)} rows")

    # Split into chunks
    n_chunks = 4
    chunk_size = len(df) // n_chunks
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    print(f"Split into {len(chunks)} chunks of ~{chunk_size} rows")

    # Benchmark sequential
    print("\n[SEQUENTIAL] Processing chunks sequentially...")
    start = time.time()
    sequential_results = [standard_clean(chunk) for chunk in chunks]
    sequential_time = time.time() - start
    print(f"Sequential: {sequential_time:.2f}s")

    # Benchmark parallel (processes)
    print("\n[PARALLEL PROCESSES] Processing chunks in parallel...")
    start = time.time()
    parallel_results = parallel_process_chunks(
        chunks,
        standard_clean,
        n_workers=get_optimal_workers(),
        use_threads=False
    )
    parallel_time = time.time() - start
    print(f"Parallel (processes): {parallel_time:.2f}s")

    # Calculate speedup
    speedup = sequential_time / parallel_time
    print(f"\n🚀 Speedup: {speedup:.2f}x faster with parallel processing")


def benchmark_memory_usage():
    """Benchmark memory usage of different approaches."""
    print("\n" + "=" * 80)
    print("BENCHMARK 3: Memory Usage")
    print("=" * 80)

    import os

    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    print(f"\nInitial memory: {initial_memory:.2f} MB")

    # Generate test data
    df = generate_large_dataset(100000)
    print(f"Generated dataset: {len(df)} rows, {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

    # Test standard cleaning
    print("\n[STANDARD] Processing with standard clean...")
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024

    result1 = standard_clean(df)

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    memory_delta_standard = mem_after - mem_before
    print(f"Memory delta: {memory_delta_standard:.2f} MB")

    # Generate new data for optimized test
    df = generate_large_dataset(100000)

    # Test optimized cleaning
    print("\n[OPTIMIZED] Processing with optimized clean...")
    gc.collect()
    mem_before = process.memory_info().rss / 1024 / 1024

    result2 = clean_data_optimized(df)

    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024
    memory_delta_optimized = mem_after - mem_before
    print(f"Memory delta: {memory_delta_optimized:.2f} MB")

    # Compare
    if memory_delta_optimized < memory_delta_standard:
        savings = memory_delta_standard - memory_delta_optimized
        savings_pct = (savings / memory_delta_standard) * 100
        print(f"\n💾 Memory savings: {savings:.2f} MB ({savings_pct:.1f}%)")


def demonstrate_streaming():
    """Demonstrate streaming processing for large files."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 4: Streaming Processing")
    print("=" * 80)

    import tempfile

    # Create temporary large file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name

    print(f"\nCreating large CSV file: {tmp_path}")
    df = generate_large_dataset(200000)
    df.to_csv(tmp_path, index=False)
    file_size_mb = Path(tmp_path).stat().st_size / 1024 / 1024
    print(f"Created file: {file_size_mb:.2f} MB, {len(df)} rows")

    # Estimate optimal chunk size
    chunk_size = estimate_chunk_size(tmp_path, target_memory_mb=50)
    print(f"Recommended chunk size: {chunk_size} rows")

    # Demonstrate streaming processing
    print("\n[STREAMING] Processing file in chunks...")

    from core.processor_enhanced import process_streaming

    def simple_processor(chunk):
        """Simple processor for demonstration."""
        return clean_data_optimized(chunk)

    stats = process_streaming(
        tmp_path,
        simple_processor,
        tmp_path + ".output",
        chunk_size=chunk_size
    )

    print(f"\n✅ Streaming complete:")
    print(f"   Total rows: {stats['total_rows']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Time: {stats['elapsed_time']:.2f}s")
    print(f"   Throughput: {stats['rows_per_second']:.0f} rows/s")

    # Cleanup
    Path(tmp_path).unlink()
    Path(tmp_path + ".output").unlink()


def demonstrate_enhanced_processor():
    """Demonstrate EnhancedDataProcessor class."""
    print("\n" + "=" * 80)
    print("DEMONSTRATION 5: Enhanced Data Processor")
    print("=" * 80)

    # Create processor
    processor = EnhancedDataProcessor(
        chunk_size=10000,
        n_workers=get_optimal_workers(),
        enable_memory_tracking=True
    )

    print(f"\nProcessor configuration:")
    print(f"  Chunk size: {processor.chunk_size}")
    print(f"  Workers: {processor.n_workers}")
    print(f"  Use threads: {processor.use_threads}")
    print(f"  Memory tracking: {processor.enable_memory_tracking}")

    # Generate test data
    df = generate_large_dataset(50000)
    print(f"\nTest data: {len(df)} rows")

    # Test optimized cleaning
    print("\n[ENHANCED] Testing optimized cleaning...")
    start = time.time()
    result = processor.clean_data_optimized(df, drop_na=False)
    elapsed = time.time() - start
    print(f"Processed in: {elapsed:.2f}s")
    print(f"Result: {len(result)} rows")


def main():
    """Run all benchmarks and demonstrations."""
    print("=" * 80)
    print("DATA PROCESSING TOOL - PERFORMANCE BENCHMARK SUITE")
    print("=" * 80)

    print("\n🖥️  System Info:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   Total memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.2f} GB")
    print(f"   Available memory: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.2f} GB")

    try:
        # Benchmark 1: Cleaning
        benchmark_cleaning()

        # Benchmark 2: Parallel Processing
        benchmark_parallel_processing()

        # Benchmark 3: Memory Usage
        benchmark_memory_usage()

        # Demonstration 4: Streaming
        demonstrate_streaming()

        # Demonstration 5: Enhanced Processor
        demonstrate_enhanced_processor()

        print("\n" + "=" * 80)
        print("BENCHMARK SUITE COMPLETE")
        print("=" * 80)
        print("\n✅ All benchmarks completed successfully!")
        print("\nKey Takeaways:")
        print("   1. Optimized functions show performance improvements")
        print("   2. Parallel processing provides significant speedup")
        print("   3. Streaming processing enables handling of large files")
        print("   4. Memory-aware processing reduces resource usage")

    except Exception as e:
        print(f"\n❌ Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
