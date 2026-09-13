from __future__ import annotations

from typing import Any

import pandas as pd
import requests

BASE_URL = "https://betbetter.world"


def load_open_model(competition_slug: str, timeout: int = 8) -> pd.DataFrame:
    """Load Bet Better's public JSON model feed for a competition page."""
    url = f"{BASE_URL}/{competition_slug.strip('/')}/picks"
    try:
        response = requests.get(url, params={"format": "json"}, timeout=timeout)
        response.raise_for_status()
        payload: Any = response.json()
    except (requests.RequestException, ValueError):
        return pd.DataFrame()

    items = payload.get("picks", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return pd.DataFrame()
    return pd.json_normalize(items)


def normalize_open_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize public model fields into a stable comparison schema."""
    if frame.empty:
        return frame.copy()
    aliases = {
        "game": "game",
        "gameTimeUtc": "game_time_utc",
        "market": "market",
        "selection": "selection",
        "line": "line",
        "modelProbabilityPct": "model_probability_pct",
        "fairOdds": "fair_odds",
    }
    result = frame.rename(columns={k: v for k, v in aliases.items() if k in frame.columns}).copy()
    for col in ["model_probability_pct", "fair_odds", "line"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")
    return result
