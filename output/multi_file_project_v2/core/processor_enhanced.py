"""Performance-Enhanced Data Processing Engine.

This module provides optimized data processing with:
1. Streaming/Chunked processing - handle large files without loading entire file into memory
2. Parallel processing - leverage multiple CPU cores for faster processing
3. Memory monitoring - track and optimize memory usage
4. Performance benchmarking - measure and report performance improvements

Dependencies:
    - pandas: Data manipulation
    - concurrent.futures: Parallel processing
    - multiprocessing: CPU-bound parallelism
"""

import gc
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple, Union
from pathlib import Path

import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Configure module logger
logger = logging.getLogger(__name__)


# ============================================================================
# Performance Monitoring Decorators
# ============================================================================

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time.

    Args:
        func: Function to decorate

    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time

        logger.info(
            f"[PERF] {func.__name__} completed in {elapsed_time:.2f}s"
        )
        return result

    return wrapper


def track_memory_usage(func: Callable) -> Callable:
    """Decorator to track peak memory usage during function execution.

    Args:
        func: Function to decorate

    Returns:
        Wrapped function with memory tracking
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        import tracemalloc
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 / 1024
        logger.info(
            f"[MEMORY] {func.__name__} peak memory: {peak_mb:.2f} MB"
        )
        return result

    return wrapper


# ============================================================================
# Streaming / Chunked Processing
# ============================================================================

def read_csv_chunks(
    file_path: Union[str, Path],
    chunk_size: int = 10000,
    **csv_kwargs
) -> Iterator[pd.DataFrame]:
    """Read large CSV file in chunks to avoid memory issues.

    Args:
        file_path: Path to CSV file
        chunk_size: Number of rows per chunk
        **csv_kwargs: Additional arguments passed to pd.read_csv

    Yields:
        DataFrame chunks

    Example:
        >>> for chunk in read_csv_chunks("large_file.csv", chunk_size=5000):
        ...     processed = clean_data(chunk)
        ...     # Save chunk to disk
    """
    file_path = Path(file_path)

    logger.info(
        f"[STREAMING] Reading {file_path} in chunks of {chunk_size} rows"
    )

    chunk_iterator = pd.read_csv(
        file_path,
        chunksize=chunk_size,
        **csv_kwargs
    )

    total_rows = 0
    for i, chunk in enumerate(chunk_iterator, 1):
        total_rows += len(chunk)
        logger.debug(
            f"[STREAMING] Processing chunk {i}, "
            f"rows so far: {total_rows}"
        )
        yield chunk

    logger.info(f"[STREAMING] Completed reading {total_rows} total rows")


@log_execution_time
def process_streaming(
    file_path: Union[str, Path],
    processor_func: Callable[[pd.DataFrame], pd.DataFrame],
    output_path: Union[str, Path],
    chunk_size: int = 10000,
    **csv_kwargs
) -> Dict[str, Any]:
    """Process large file in streaming fashion.

    Args:
        file_path: Input file path
        processor_func: Function to process each chunk
        output_path: Output file path
        chunk_size: Rows per chunk
        **csv_kwargs: Additional CSV reading arguments

    Returns:
        Dictionary with processing statistics

    Example:
        >>> def my_processor(chunk):
        ...     return clean_data(chunk)
        >>> stats = process_streaming(
        ...     "large_input.csv",
        ...     my_processor,
        ...     "output.csv",
        ...     chunk_size=5000
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_rows": 0,
        "total_chunks": 0,
        "start_time": time.time()
    }

    # Process first chunk to initialize output file
    first_chunk = True

    for chunk in read_csv_chunks(file_path, chunk_size, **csv_kwargs):
        # Process chunk
        processed_chunk = processor_func(chunk)

        # Write to file (append mode after first chunk)
        processed_chunk.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )

        first_chunk = False
        stats["total_rows"] += len(processed_chunk)
        stats["total_chunks"] += 1

        # Force garbage collection to free memory
        del chunk, processed_chunk
        gc.collect()

    stats["elapsed_time"] = time.time() - stats["start_time"]
    stats["rows_per_second"] = stats["total_rows"] / stats["elapsed_time"]

    logger.info(
        f"[STREAMING] Processed {stats['total_rows']} rows in "
        f"{stats['total_chunks']} chunks ({stats['elapsed_time']:.2f}s, "
        f"{stats['rows_per_second']:.0f} rows/s)"
    )

    return stats


# ============================================================================
# Parallel Processing
# ============================================================================

@log_execution_time
def parallel_process_chunks(
    chunks: List[pd.DataFrame],
    processor_func: Callable[[pd.DataFrame], pd.DataFrame],
    n_workers: Optional[int] = None,
    use_threads: bool = False
) -> List[pd.DataFrame]:
    """Process multiple DataFrame chunks in parallel.

    Args:
        chunks: List of DataFrame chunks to process
        processor_func: Function to apply to each chunk
        n_workers: Number of workers (default: CPU count)
        use_threads: Use threads instead of processes (for I/O-bound tasks)

    Returns:
        List of processed chunks

    Example:
        >>> chunks = [df1, df2, df3]
        >>> results = parallel_process_chunks(chunks, clean_data, n_workers=3)
    """
    import os

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    logger.info(
        f"[PARALLEL] Processing {len(chunks)} chunks with "
        f"{n_workers} workers ({'threads' if use_threads else 'processes'})"
    )

    results = []

    with executor_class(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_chunk = {
            executor.submit(processor_func, chunk): i
            for i, chunk in enumerate(chunks)
        }

        # Collect results as they complete
        for future in as_completed(future_to_chunk):
            chunk_idx = future_to_chunk[future]
            try:
                result = future.result()
                results.append((chunk_idx, result))
                logger.debug(
                    f"[PARALLEL] Completed chunk {chunk_idx + 1}/{len(chunks)}"
                )
            except Exception as e:
                logger.error(
                    f"[PARALLEL] Error processing chunk {chunk_idx}: {e}"
                )
                raise

    # Sort results by original chunk order
    results.sort(key=lambda x: x[0])
    results = [r[1] for r in results]

    logger.info(f"[PARALLEL] Completed processing {len(results)} chunks")

    return results


@log_execution_time
def parallel_apply(
    df: pd.DataFrame,
    func: Callable,
    n_workers: Optional[int] = None,
    use_threads: bool = False
) -> pd.Series:
    """Apply function to DataFrame in parallel.

    Args:
        df: Input DataFrame
        func: Function to apply (must be pickleable for processes)
        n_workers: Number of workers
        use_threads: Use threads instead of processes

    Returns:
        Series with results

    Example:
        >>> def process_row(row):
        ...     return row['A'] + row['B']
        >>> result = parallel_apply(df, process_row, n_workers=4)
    """
    import os

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    logger.info(
        f"[PARALLEL] Applying function to {len(df)} rows "
        f"with {n_workers} workers"
    )

    # Split DataFrame into chunks
    chunk_size = len(df) // n_workers
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Process chunks in parallel
    processed_chunks = parallel_process_chunks(
        chunks,
        lambda chunk: chunk.apply(func),
        n_workers=n_workers,
        use_threads=use_threads
    )

    # Combine results
    result = pd.concat(processed_chunks)

    logger.info(f"[PARALLEL] Applied function to {len(result)} rows")

    return result


# ============================================================================
# Enhanced Data Processor Class
# ============================================================================

class EnhancedDataProcessor:
    """Enhanced data processor with streaming and parallel support.

    Attributes:
        chunk_size: Default chunk size for streaming
        n_workers: Default number of parallel workers
        use_threads: Default to use threads for parallel tasks
        enable_memory_tracking: Track memory usage during processing

    Example:
        >>> processor = EnhancedDataProcessor(chunk_size=5000, n_workers=4)
        >>> stats = processor.process_streaming_large_file(
        ...     "large.csv",
        ...     "output.csv",
        ...     clean_data
        ... )
    """

    def __init__(
        self,
        chunk_size: int = 10000,
        n_workers: Optional[int] = None,
        use_threads: bool = False,
        enable_memory_tracking: bool = False
    ):
        """Initialize EnhancedDataProcessor.

        Args:
            chunk_size: Rows per chunk for streaming processing
            n_workers: Number of parallel workers (default: CPU count)
            use_threads: Use threads instead of processes
            enable_memory_tracking: Enable memory usage tracking
        """
        import os
        self.chunk_size = chunk_size
        self.n_workers = n_workers or os.cpu_count() or 1
        self.use_threads = use_threads
        self.enable_memory_tracking = enable_memory_tracking

        self.stats = {}

    @log_execution_time
    def clean_data_optimized(
        self,
        df: pd.DataFrame,
        drop_na: bool = False,
        fill_na: Optional[Dict[str, Any]] = None,
        drop_duplicates: bool = True,
        strip_whitespace: bool = True,
    ) -> pd.DataFrame:
        """Optimized data cleaning with reduced memory footprint.

        This version minimizes memory copies and uses in-place operations
        where possible.

        Args:
            df: Input DataFrame
            drop_na: Drop rows with NA values
            fill_na: Fill NA values with specified strategy
            drop_duplicates: Remove duplicate rows
            strip_whitespace: Strip whitespace from string columns

        Returns:
            Cleaned DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, got {type(df)}")

        if df.empty:
            logger.warning("Attempted to clean an empty DataFrame.")
            return df.copy()

        # Work on copy to avoid modifying original
        df_cleaned = df.copy()

        # Optimize string operations
        if strip_whitespace:
            str_cols = df_cleaned.select_dtypes(include=['object', 'string']).columns
            for col in str_cols:
                # Use in-place operation to save memory
                df_cleaned[col] = df_cleaned[col].str.strip()

        # Handle missing values
        if drop_na:
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            dropped_rows = initial_rows - len(df_cleaned)
            logger.info(f"[OPTIMIZED] Dropped {dropped_rows} rows with NA values")
        elif fill_na:
            df_cleaned = df_cleaned.fillna(fill_na)
            logger.info(f"[OPTIMIZED] Filled NA values")

        # Handle duplicates
        if drop_duplicates:
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            dropped_rows = initial_rows - len(df_cleaned)
            logger.info(f"[OPTIMIZED] Dropped {dropped_rows} duplicates")

        return df_cleaned

    @track_memory_usage
    def process_large_file_streaming(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        processor_func: Callable,
        **csv_kwargs
    ) -> Dict[str, Any]:
        """Process large file using streaming approach.

        Args:
            input_path: Input file path
            output_path: Output file path
            processor_func: Function to process each chunk
            **csv_kwargs: Additional CSV reading arguments

        Returns:
            Processing statistics dictionary
        """
        if self.enable_memory_tracking:
            return process_streaming(
                input_path,
                processor_func,
                output_path,
                chunk_size=self.chunk_size,
                **csv_kwargs
            )
        else:
            # Disable tracking for better performance
            import functools
            # Temporarily disable decorator
            original_func = process_streaming
            undecorated = original_func.__wrapped__ if hasattr(original_func, '__wrapped__') else None

            if undecorated:
                return undecorated(
                    input_path,
                    processor_func,
                    output_path,
                    chunk_size=self.chunk_size,
                    **csv_kwargs
                )
            else:
                return process_streaming(
                    input_path,
                    processor_func,
                    output_path,
                    chunk_size=self.chunk_size,
                    **csv_kwargs
                )

    def process_chunks_parallel(
        self,
        chunks: List[pd.DataFrame],
        processor_func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> List[pd.DataFrame]:
        """Process multiple chunks in parallel.

        Args:
            chunks: List of DataFrame chunks
            processor_func: Processing function

        Returns:
            List of processed chunks
        """
        return parallel_process_chunks(
            chunks,
            processor_func,
            n_workers=self.n_workers,
            use_threads=self.use_threads
        )

    @log_execution_time
    def aggregate_parallel(
        self,
        df: pd.DataFrame,
        group_by: Union[str, List[str]],
        aggregations: Dict[str, Union[str, List[str]]],
    ) -> pd.DataFrame:
        """Perform aggregation in parallel for large datasets.

        Args:
            df: Input DataFrame
            group_by: Column(s) to group by
            aggregations: Aggregation operations

        Returns:
            Aggregated DataFrame
        """
        logger.info(
            f"[PARALLEL AGG] Aggregating {len(df)} rows by {group_by}"
        )

        # Split data into chunks for parallel aggregation
        chunk_size = max(len(df) // self.n_workers, 1)
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        # Aggregate each chunk in parallel
        def agg_chunk(chunk):
            return chunk.groupby(group_by).agg(aggregations)

        chunk_results = parallel_process_chunks(
            chunks,
            agg_chunk,
            n_workers=self.n_workers,
            use_threads=False  # Use processes for CPU-bound aggregation
        )

        # Combine chunk results (may have overlapping groups)
        # Need to aggregate again to merge overlapping groups
        final_result = pd.concat(chunk_results)
        final_result = final_result.groupby(group_by).agg(aggregations)

        logger.info(f"[PARALLEL AGG] Final aggregation complete")

        return final_result


# ============================================================================
# Performance Benchmarking
# ============================================================================

class PerformanceBenchmark:
    """Benchmark and compare performance of different processing approaches.

    Example:
        >>> benchmark = PerformanceBenchmark()
        >>> results = benchmark.benchmark_approaches(
        ...     df,
        ...     {
        ...         'standard': lambda df: clean_data(df),
        ...         'optimized': lambda df: clean_data_optimized(df)
        ...     }
        ... )
    """

    def __init__(self):
        """Initialize performance benchmark."""
        self.results = {}

    @staticmethod
    def get_memory_usage() -> float:
        """Get current process memory usage in MB.

        Returns:
            Memory usage in MB
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def benchmark_function(
        self,
        func: Callable,
        *args,
        warmup_runs: int = 1,
        benchmark_runs: int = 3,
        **kwargs
    ) -> Dict[str, Any]:
        """Benchmark a function's performance.

        Args:
            func: Function to benchmark
            *args: Function arguments
            warmup_runs: Number of warmup runs (not timed)
            benchmark_runs: Number of benchmark runs
            **kwargs: Function keyword arguments

        Returns:
            Dictionary with timing and memory results
        """
        # Warmup
        for _ in range(warmup_runs):
            func(*args, **kwargs)

        # Benchmark
        times = []
        memory_before = []
        memory_after = []

        for _ in range(benchmark_runs):
            mem_before = self.get_memory_usage()
            memory_before.append(mem_before)

            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start

            mem_after = self.get_memory_usage()
            memory_after.append(mem_after)

            times.append(elapsed)

        return {
            "mean_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "mean_memory_delta": (
                sum(a - b for a, b in zip(memory_after, memory_before)) /
                len(memory_before)
            ),
            "runs": benchmark_runs
        }

    def benchmark_approaches(
        self,
        test_data: pd.DataFrame,
        approaches: Dict[str, Callable],
        runs: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark multiple approaches against same data.

        Args:
            test_data: Test DataFrame
            approaches: Dictionary of approach name to function
            runs: Number of benchmark runs per approach

        Returns:
            Dictionary of results per approach

        Example:
            >>> results = benchmark.benchmark_approaches(
            ...     df,
            ...     {
            ...         'standard': clean_data,
            ...         'optimized': enhanced_clean
            ...     }
            ... )
        """
        results = {}

        for name, func in approaches.items():
            logger.info(f"[BENCHMARK] Benchmarking {name}...")
            results[name] = self.benchmark_function(
                func,
                test_data,
                warmup_runs=1,
                benchmark_runs=runs
            )

        self.results = results
        return results

    def print_comparison(self):
        """Print comparison table of benchmark results."""
        if not self.results:
            print("[BENCHMARK] No results to display")
            return

        print("\n" + "=" * 80)
        print("Performance Comparison Results")
        print("=" * 80)

        # Header
        print(f"{'Approach':<20} {'Mean Time (s)':<15} {'Min (s)':<10} {'Max (s)':<10} {'Memory Δ (MB)':<15}")
        print("-" * 80)

        # Results
        for name, stats in self.results.items():
            print(
                f"{name:<20} "
                f"{stats['mean_time']:<15.4f} "
                f"{stats['min_time']:<10.4f} "
                f"{stats['max_time']:<10.4f} "
                f"{stats['mean_memory_delta']:<15.2f}"
            )

        print("=" * 80)

        # Find fastest
        fastest = min(self.results.items(), key=lambda x: x[1]['mean_time'])
        print(f"\n🏆 Fastest: {fastest[0]} ({fastest[1]['mean_time']:.4f}s)")

        # Calculate speedup
        baseline_time = list(self.results.values())[0]['mean_time']
        for name, stats in self.results.items():
            speedup = baseline_time / stats['mean_time']
            if speedup > 1:
                print(f"   {name}: {speedup:.2f}x faster")


# ============================================================================
# Utility Functions
# ============================================================================

def estimate_chunk_size(
    file_path: Union[str, Path],
    target_memory_mb: int = 100,
    avg_row_size_bytes: int = 100
) -> int:
    """Estimate optimal chunk size for streaming processing.

    Args:
        file_path: Path to data file
        target_memory_mb: Target memory usage per chunk (MB)
        avg_row_size_bytes: Estimated average row size in bytes

    Returns:
        Recommended chunk size (number of rows)

    Example:
        >>> chunk_size = estimate_chunk_size("large.csv", target_memory_mb=50)
        >>> for chunk in read_csv_chunks("large.csv", chunk_size=chunk_size):
        ...     process(chunk)
    """
    file_path = Path(file_path)

    # Get file size
    file_size_bytes = file_path.stat().st_size

    # Calculate rows per chunk to stay within memory target
    target_memory_bytes = target_memory_mb * 1024 * 1024
    chunk_size = target_memory_bytes // avg_row_size_bytes

    # Ensure minimum chunk size
    chunk_size = max(chunk_size, 1000)

    logger.info(
        f"[CHUNK] File: {file_path.name} "
        f"({file_size_bytes / 1024 / 1024:.1f} MB), "
        f"Recommended chunk size: {chunk_size} rows"
    )

    return chunk_size


def get_optimal_workers() -> int:
    """Get optimal number of workers for parallel processing.

    Returns:
        Optimal worker count based on CPU count
    """
    import os
    cpu_count = os.cpu_count() or 1

    # Use slightly fewer than CPU count to leave headroom
    optimal = max(1, cpu_count - 1)

    logger.info(f"[PARALLEL] Optimal workers: {optimal} (CPU count: {cpu_count})")

    return optimal
