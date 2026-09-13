from __future__ import annotations

from io import BytesIO

import pandas as pd
import requests

FOOTBALL_DATA_URL = "https://www.football-data.co.uk/new/"
OPENFOOTBALL_EUROPE_URL = "https://raw.githubusercontent.com/openfootball/europe/master"
OPENFOOTBALL_CL_URL = "https://raw.githubusercontent.com/openfootball/champions-league/master"


def load_football_data_csv(code: str, timeout: int = 8) -> pd.DataFrame:
    """Load a public Football-Data.co.uk CSV for an extra league/competition."""
    try:
        response = requests.get(f"{FOOTBALL_DATA_URL}{code}.csv", timeout=timeout)
        response.raise_for_status()
        return pd.read_csv(BytesIO(response.content), encoding="latin1")
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def load_openfootball_season(path: str, timeout: int = 8) -> pd.DataFrame:
    """Load an openfootball JSON season file when available."""
    try:
        response = requests.get(path, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return pd.DataFrame()
    matches = payload.get("matches") if isinstance(payload, dict) else None
    if not isinstance(matches, list):
        return pd.DataFrame()
    return pd.json_normalize(matches)
