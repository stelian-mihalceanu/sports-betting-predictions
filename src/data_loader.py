"""
Data loading utilities for tennis and football datasets.

Supports loading from:
- Local CSV/Parquet files
- Jeff Sackmann tennis datasets (ATP/WTA)
- Football data APIs (OpenFoot, Football-Data.org, etc.)
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union
import requests

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_tennis_atp(file_name: str = "atp_matches.csv") -> pd.DataFrame:
    """
    Load ATP matches dataset.
    
    Parameters
    ----------
    file_name : str, default "atp_matches.csv"
        Name of the CSV file in data/raw/.
    
    Returns
    -------
    pd.DataFrame
        ATP matches DataFrame.
    """
    path = RAW_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"File {path} not found. Download tennis data from "
            "https://github.com/JeffSackmann/tennis_atp and place in data/raw/"
        )
    return pd.read_csv(path)


def load_tennis_wta(file_name: str = "wta_matches.csv") -> pd.DataFrame:
    """
    Load WTA matches dataset.
    
    Parameters
    ----------
    file_name : str, default "wta_matches.csv"
        Name of the CSV file in data/raw/.
    
    Returns
    -------
    pd.DataFrame
        WTA matches DataFrame.
    """
    path = RAW_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(
            f"File {path} not found. Download tennis data from "
            "https://github.com/JeffSackmann/tennis_wta and place in data/raw/"
        )
    return pd.read_csv(path)


def load_football_matches(
    file_name: str = "matches.csv",
    source: str = "local"
) -> pd.DataFrame:
    """
    Load football matches dataset.
    
    Parameters
    ----------
    file_name : str, default "matches.csv"
        Name of the CSV/Parquet file in data/raw/.
    source : str, default "local"
        Data source: "local", "openfoot", "football-data".
    
    Returns
    -------
    pd.DataFrame
        Football matches DataFrame.
    """
    if source == "local":
        path = RAW_DIR / file_name
        if not path.exists():
            raise FileNotFoundError(
                f"File {path} not found. Add football data to data/raw/"
            )
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    
    elif source == "openfoot":
        # Example: OpenFoot API integration
        api_key = "YOUR_API_KEY"  # Replace with actual key
        url = f"https://api.openfootapi.com/fixtures?key={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    
    else:
        raise ValueError(f"Unknown source: {source}")


def save_processed_data(df: pd.DataFrame, file_name: str) -> None:
    """
    Save processed DataFrame to data/processed/.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    file_name : str
        Output file name (CSV or Parquet).
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / file_name
    
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    
    print(f"Saved processed data to {path}")
