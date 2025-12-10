"""
data.py — Utilities for loading raw data and connecting to the SQLite database.
"""

from pathlib import Path
import sqlite3
import pandas as pd
import kagglehub


def get_kaggle_db_path() -> Path:
    """Download (or reuse cached) Kaggle European Soccer DB and return the path."""
    path = kagglehub.dataset_download("hugomathien/soccer")
    db_path = Path(path) / "database.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")
    return db_path


def connect_to_db(db_path: Path):
    """Open a SQLite connection to the Kaggle DB."""
    return sqlite3.connect(str(db_path))


def query_db(conn, sql: str, params=None) -> pd.DataFrame:
    """Run SQL and return results as a DataFrame."""
    return pd.read_sql_query(sql, conn, params or {})

