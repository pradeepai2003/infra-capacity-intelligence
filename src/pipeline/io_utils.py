"""
Small shared helper so every pipeline stage saves its output to disk AND
prints a clear, consistent confirmation to the terminal -- rather than each
module inventing its own logging format.
"""

from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def save_and_log(df: pd.DataFrame, path: str, label: str) -> str:
    """
    Save a DataFrame to CSV and print/log a confirmation line to the terminal.

    Args:
        df: the DataFrame to persist
        path: full output file path, e.g. "data/raw/compute_metrics.csv"
        label: human-readable description, e.g. "Synthetic compute metrics"

    Returns:
        the path the file was saved to (for convenience/chaining)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("[SAVED] %s: %s rows -> %s", label, f"{len(df):,}", path)
    return path
