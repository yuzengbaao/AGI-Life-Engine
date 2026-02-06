"""Multi-Format Data Reader/Writer.

Supports multiple data formats for flexible data I/O:
- CSV
- Excel
- JSON
- JSONL
- Parquet
- Feather
- Pickle
- HDF5
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Multi-Format Reader
# ============================================================================

class MultiFormatReader:
    """Read data files in multiple formats.

    Supported formats:
    - csv: Comma-separated values
    - excel: Excel files (.xlsx, .xls)
    - json: JSON files
    - jsonl: JSON Lines (one JSON object per line)
    - parquet: Apache Parquet columnar storage
    - feather: Feather format (fast binary)
    - pickle: Python pickle format
    - hdf5: Hierarchical Data Format v5
    """

    @staticmethod
    def read_csv(
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Read CSV file.

        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            DataFrame
        """
        logger.info(f"[READ CSV] {file_path}")
        return pd.read_csv(file_path, **kwargs)

    @staticmethod
    def read_excel(
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Read Excel file.

        Args:
            file_path: Path to Excel file
            **kwargs: Additional arguments for pd.read_excel

        Returns:
            DataFrame
        """
        logger.info(f"[READ EXCEL] {file_path}")
        return pd.read_excel(file_path, **kwargs)

    @staticmethod
    def read_json(
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Read JSON file.

        Args:
            file_path: Path to JSON file
            **kwargs: Additional arguments for pd.read_json

        Returns:
            DataFrame
        """
        logger.info(f"[READ JSON] {file_path}")
        return pd.read_json(file_path, **kwargs)

    @staticmethod
    def read_jsonl(
        file_path: Union[str, Path],
        lines: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Read JSON Lines file (JSONL).

        Each line is a separate JSON object.

        Args:
            file_path: Path to JSONL file
            lines: Read as JSONL (one JSON per line)
            **kwargs: Additional arguments for pd.read_json

        Returns:
            DataFrame
        """
        logger.info(f"[READ JSONL] {file_path}")
        return pd.read_json(file_path, lines=lines, **kwargs)

    @staticmethod
    def read_parquet(
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Read Parquet file.

        Parquet is a columnar storage format optimized for performance.

        Args:
            file_path: Path to Parquet file
            **kwargs: Additional arguments for pd.read_parquet

        Returns:
            DataFrame
        """
        logger.info(f"[READ PARQUET] {file_path}")
        return pd.read_parquet(file_path, **kwargs)

    @staticmethod
    def read_feather(
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Read Feather file.

        Feather is a fast binary format for DataFrames.

        Args:
            file_path: Path to Feather file
            **kwargs: Additional arguments for pd.read_feather

        Returns:
            DataFrame
        """
        logger.info(f"[READ FEATHER] {file_path}")
        return pd.read_feather(file_path, **kwargs)

    @staticmethod
    def read_pickle(
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Read Pickle file.

        Args:
            file_path: Path to Pickle file
            **kwargs: Additional arguments for pd.read_pickle

        Returns:
            DataFrame
        """
        logger.info(f"[READ PICKLE] {file_path}")
        return pd.read_pickle(file_path, **kwargs)

    @staticmethod
    def read_hdf5(
        file_path: Union[str, Path],
        key: str = "data",
        **kwargs
    ) -> pd.DataFrame:
        """Read HDF5 file.

        Args:
            file_path: Path to HDF5 file
            key: Dataset key within HDF5 file
            **kwargs: Additional arguments for pd.read_hdf

        Returns:
            DataFrame
        """
        logger.info(f"[READ HDF5] {file_path} (key: {key})")
        return pd.read_hdf(file_path, key=key, **kwargs)

    @classmethod
    def read_auto(
        cls,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """Auto-detect format and read file.

        Args:
            file_path: Path to data file
            **kwargs: Additional arguments for reader

        Returns:
            DataFrame

        Raises:
            ValueError: If format is not supported
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        # Format mapping
        format_handlers = {
            '.csv': cls.read_csv,
            '.xlsx': cls.read_excel,
            '.xls': cls.read_excel,
            '.json': cls.read_json,
            '.jsonl': cls.read_jsonl,
            '.parquet': cls.read_parquet,
            '.feather': cls.read_feather,
            '.pkl': cls.read_pickle,
            '.pickle': cls.read_pickle,
            '.h5': cls.read_hdf5,
            '.hdf5': cls.read_hdf5,
        }

        if suffix not in format_handlers:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported: {list(format_handlers.keys())}"
            )

        handler = format_handlers[suffix]
        return handler(file_path, **kwargs)


# ============================================================================
# Multi-Format Writer
# ============================================================================

class MultiFormatWriter:
    """Write data frames to multiple formats.

    Supported formats:
    - csv: Comma-separated values
    - excel: Excel files
    - json: JSON files
    - jsonl: JSON Lines
    - parquet: Apache Parquet
    - feather: Feather format
    - pickle: Python pickle
    - hdf5: Hierarchical Data Format v5
    """

    @staticmethod
    def write_csv(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Write to CSV file.

        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional arguments for df.to_csv
        """
        logger.info(f"[WRITE CSV] {file_path}")
        df.to_csv(file_path, index=False, **kwargs)

    @staticmethod
    def write_excel(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Write to Excel file.

        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional arguments for df.to_excel
        """
        logger.info(f"[WRITE EXCEL] {file_path}")
        df.to_excel(file_path, index=False, **kwargs)

    @staticmethod
    def write_json(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        orient: str = "records",
        **kwargs
    ) -> None:
        """Write to JSON file.

        Args:
            df: DataFrame to write
            file_path: Output file path
            orient: JSON orientation (records, index, columns, values)
            **kwargs: Additional arguments for df.to_json
        """
        logger.info(f"[WRITE JSON] {file_path}")
        df.to_json(file_path, orient=orient, **kwargs)

    @staticmethod
    def write_jsonl(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        orient: str = "records",
        **kwargs
    ) -> None:
        """Write to JSON Lines file.

        Args:
            df: DataFrame to write
            file_path: Output file path
            orient: JSON orientation
            **kwargs: Additional arguments
        """
        logger.info(f"[WRITE JSONL] {file_path}")

        # Write line by line
        with open(file_path, 'w', encoding='utf-8') as f:
            for record in df.to_dict(orient=orient):
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')

    @staticmethod
    def write_parquet(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Write to Parquet file.

        Parquet is optimized for columnar data and compression.

        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional arguments for df.to_parquet
        """
        logger.info(f"[WRITE PARQUET] {file_path}")
        df.to_parquet(file_path, **kwargs)

    @staticmethod
    def write_feather(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Write to Feather file.

        Feather is a fast binary format for DataFrames.

        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional arguments for df.to_feather
        """
        logger.info(f"[WRITE FEATHER] {file_path}")
        df.to_feather(file_path, **kwargs)

    @staticmethod
    def write_pickle(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Write to Pickle file.

        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional arguments for df.to_pickle
        """
        logger.info(f"[WRITE PICKLE] {file_path}")
        df.to_pickle(file_path, **kwargs)

    @staticmethod
    def write_hdf5(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        key: str = "data",
        **kwargs
    ) -> None:
        """Write to HDF5 file.

        Args:
            df: DataFrame to write
            file_path: Output file path
            key: Dataset key within HDF5 file
            **kwargs: Additional arguments for df.to_hdf
        """
        logger.info(f"[WRITE HDF5] {file_path} (key: {key})")
        df.to_hdf(file_path, key=key, **kwargs)

    @classmethod
    def write_auto(
        cls,
        df: pd.DataFrame,
        file_path: Union[str, Path],
        **kwargs
    ) -> None:
        """Auto-detect format from file extension and write.

        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional arguments for writer

        Raises:
            ValueError: If format is not supported
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()

        # Format mapping
        format_handlers = {
            '.csv': cls.write_csv,
            '.xlsx': cls.write_excel,
            '.xls': cls.write_excel,
            '.json': cls.write_json,
            '.jsonl': cls.write_jsonl,
            '.parquet': cls.write_parquet,
            '.feather': cls.write_feather,
            '.pkl': cls.write_pickle,
            '.pickle': cls.write_pickle,
            '.h5': cls.write_hdf5,
            '.hdf5': cls.write_hdf5,
        }

        if suffix not in format_handlers:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported: {list(format_handlers.keys())}"
            )

        handler = format_handlers[suffix]
        handler(df, file_path, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================

def convert_format(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs
) -> None:
    """Convert data file from one format to another.

    Args:
        input_path: Input file path
        output_path: Output file path
        **kwargs: Additional reader/writer arguments

    Example:
        >>> convert_format("data.csv", "data.parquet")
        >>> convert_format("data.xlsx", "data.jsonl")
    """
    logger.info(f"[CONVERT] {input_path} -> {output_path}")

    # Auto-detect formats and convert
    df = MultiFormatReader.read_auto(input_path, **kwargs)
    MultiFormatWriter.write_auto(df, output_path, **kwargs)

    logger.info("[CONVERT] Complete")


def get_format_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get information about a data file.

    Args:
        file_path: Path to data file

    Returns:
        Dictionary with file information

    Example:
        >>> info = get_format_info("data.csv")
        >>> print(f"Format: {info['format']}, Size: {info['size_mb']} MB")
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get basic info
    info = {
        "path": str(file_path),
        "name": file_path.name,
        "format": file_path.suffix[1:].upper(),
        "size_bytes": file_path.stat().st_size,
        "size_mb": file_path.stat().st_size / 1024 / 1024,
    }

    # Try to read and get data info
    try:
        df = MultiFormatReader.read_auto(file_path, nrows=1000)
        info.update({
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "preview": df.head(5).to_dict()
        })
    except Exception as e:
        logger.warning(f"Could not read file for preview: {e}")

    return info
