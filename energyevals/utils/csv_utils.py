import io
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from energyevals.utils.constants import CSV_SAMPLE_SIZE, CSV_THRESHOLD_LARGE


def generate_timestamp() -> str:
    """Generate a timestamp string for file naming.

    Returns:
        Timestamp in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_to_csv(
    df: pd.DataFrame,
    prefix: str,
    output_dir: Path | str = ".",
) -> Path:
    """Save a DataFrame to CSV with timestamped filename.

    Args:
        df: DataFrame to save
        prefix: Prefix for the filename (e.g., "gridstatus", "battery")
        output_dir: Directory to save the file (default: current directory)

    Returns:
        Path to the saved CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = generate_timestamp()
    filename = f"{prefix}_{timestamp}.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    logger.info(f"Saved {len(df)} rows to {filepath}")

    return filepath


def save_dataframe_to_csv(
    df: pd.DataFrame,
    prefix: str,
    output_dir: Path | str,
) -> Path:
    """Save a DataFrame to CSV with timestamped filename.

    Args:
        df: DataFrame to save
        prefix: Prefix for the filename (e.g., "gridstatus", "battery")
        output_dir: Directory to save the file

    Returns:
        Path to the saved CSV file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = generate_timestamp()
    filename = f"{prefix}_{timestamp}.csv"
    filepath = output_dir / filename

    df.to_csv(filepath, index=False)
    return filepath


def process_large_dataframe_result(
    df: pd.DataFrame,
    prefix: str,
    output_dir: Path | str,
    csv_threshold: int = CSV_THRESHOLD_LARGE,
) -> dict[str, Any]:
    """Process a DataFrame result, saving to CSV if it exceeds threshold.

    For small results, returns the data as JSON. For large results,
    saves to CSV and returns metadata with file path.

    Args:
        df: DataFrame to process
        prefix: Prefix for filename if saving to CSV
        output_dir: Directory for CSV output
        csv_threshold: Number of rows above which to save as CSV

    Returns:
        Dictionary with result data or CSV metadata
    """
    if len(df) <= csv_threshold:
        return {
            "row_count": len(df),
            "data": df.to_dict(orient="records"),
        }
    else:
        csv_path = save_dataframe_to_csv(df, prefix, output_dir)
        return {
            "row_count": len(df),
            "csv_saved": True,
            "csv_path": str(csv_path),
            "message": f"Result saved to {csv_path.name} ({len(df)} rows)",
            "sample": df.head(CSV_SAMPLE_SIZE).to_dict(orient="records"),
        }


def csv_string_to_dataframe(csv_string: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(csv_string))


def dataframe_to_csv_string(df: pd.DataFrame) -> str:
    result: str = df.to_csv(index=False)
    return result
