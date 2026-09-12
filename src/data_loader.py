"""Data loading utilities for tennis and football datasets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests

BASE_DATA_DIR = Path(os.getenv("SPORTS_DATA_DIR", Path(__file__).parent.parent / "data"))
RAW_DIR = BASE_DATA_DIR / "raw"
PROCESSED_DIR = BASE_DATA_DIR / "processed"


def load_tennis_atp(file_name: str = "atp_matches.csv") -> pd.DataFrame:
    return _read_table(RAW_DIR / file_name, "ATP tennis")


def load_tennis_wta(file_name: str = "wta_matches.csv") -> pd.DataFrame:
    return _read_table(RAW_DIR / file_name, "WTA tennis")


def load_football_matches(file_name: str = "matches.csv", source: str = "local", timeout: int = 20) -> pd.DataFrame:
    if source == "local":
        return _read_table(RAW_DIR / file_name, "football")
    if source == "openfoot":
        api_key = os.getenv("OPENFOOT_API_KEY")
        if not api_key:
            raise RuntimeError("OPENFOOT_API_KEY is not configured")
        response = requests.get(
            "https://api.openfootapi.com/fixtures",
            params={"key": api_key},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return pd.DataFrame(payload)
    raise ValueError(f"Unknown source: {source}")


def save_processed_data(df: pd.DataFrame, file_name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / file_name
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Processed data must be CSV or Parquet")
    return path


def _read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} dataset not found at {path}. Add the dataset to data/raw/."
        )
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")
