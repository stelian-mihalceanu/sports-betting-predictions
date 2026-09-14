"""Fast, resilient football/tennis data loaders with Streamlit caching."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BASE_DATA_DIR = Path(os.getenv("SPORTS_DATA_DIR", Path(__file__).parent.parent / "data"))
RAW_DIR = BASE_DATA_DIR / "raw"
PROCESSED_DIR = BASE_DATA_DIR / "processed"
CACHE_DIR = BASE_DATA_DIR / "cache"
FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_LEAGUES = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga", "I1": "Serie A", "RO1": "Liga 1", "PL": "Ekstraklasa"}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["league_code", "date", "home_team", "away_team", "competition", "home_goals", "away_goals", "season"])


def _fetch(url: str, code: str, season: str, timeout: int = 6) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "BetLens/2.0"})
        response.raise_for_status()
        frame = pd.read_csv(BytesIO(response.content), encoding="latin1")
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError, UnicodeError):
        return _empty()
    frame = frame.rename(columns={"Div": "league_code", "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team", "FTHG": "home_goals", "FTAG": "away_goals"}).copy()
    if not {"date", "home_team", "away_team"}.issubset(frame.columns):
        return _empty()
    return pd.DataFrame({
        "league_code": code,
        "date": pd.to_datetime(frame["date"], dayfirst=True, errors="coerce"),
        "home_team": frame["home_team"].astype(str).str.strip(),
        "away_team": frame["away_team"].astype(str).str.strip(),
        "competition": FOOTBALL_LEAGUES.get(code, code),
        "home_goals": pd.to_numeric(frame.get("home_goals"), errors="coerce"),
        "away_goals": pd.to_numeric(frame.get("away_goals"), errors="coerce"),
        "season": season,
        "HTHG": pd.to_numeric(frame.get("HTHG"), errors="coerce"),
        "HTAG": pd.to_numeric(frame.get("HTAG"), errors="coerce"),
        "HC": pd.to_numeric(frame.get("HC"), errors="coerce"),
        "AC": pd.to_numeric(frame.get("AC"), errors="coerce"),
        "HY": pd.to_numeric(frame.get("HY"), errors="coerce"),
        "AY": pd.to_numeric(frame.get("AY"), errors="coerce"),
        "HR": pd.to_numeric(frame.get("HR"), errors="coerce"),
        "AR": pd.to_numeric(frame.get("AR"), errors="coerce"),
    }).dropna(subset=["date", "home_team", "away_team"])


def _season_code(season: int | str) -> str:
    start = int(str(season)[:4])
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


@st.cache_data(ttl=86400, show_spinner=False)
def load_football_history(seasons: tuple[int, ...] = (2024, 2025), timeout: int = 6) -> pd.DataFrame:
    """Load historical leagues once per day; prefer persisted processed data."""
    processed = PROCESSED_DIR / "football_history.parquet"
    if processed.exists():
        try:
            return pd.read_parquet(processed)
        except (OSError, ValueError):
            pass
    jobs = [(f"{FOOTBALL_DATA}/{_season_code(season)}/{league}.csv", league, _season_code(season)) for season in seasons for league in FOOTBALL_LEAGUES]
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_fetch, url, league, season, timeout) for url, league, season in jobs]
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False).sort_values("date").reset_index(drop=True) if frames else _empty()
    return result


@st.cache_data(ttl=900, show_spinner=False)
def load_upcoming_football_fixtures(timeout: int = 5) -> pd.DataFrame:
    """Load current-season fixtures. Cache for 15 minutes and never fail the UI."""
    season = _season_code(2026)
    jobs = [(f"{FOOTBALL_DATA}/{season}/{league}.csv", league, season) for league in FOOTBALL_LEAGUES]
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(_fetch, url, league, season, timeout) for url, league, season in jobs]
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return _empty()
    result = pd.concat(frames, ignore_index=True, sort=False)
    today = pd.Timestamp.now().normalize()
    return result[(result["date"] >= today) & result["home_goals"].isna() & result["away_goals"].isna()].drop_duplicates(["date", "home_team", "away_team", "competition"]).sort_values("date").reset_index(drop=True)


def load_football_matches(file_name: str = "matches.csv", source: str = "local", timeout: int = 6) -> pd.DataFrame:
    local = RAW_DIR / file_name
    if source == "local" and local.exists():
        return pd.read_parquet(local) if local.suffix.lower() == ".parquet" else pd.read_csv(local)
    if source == "football-data":
        return load_football_history(timeout=timeout)
    raise ValueError("Only local and football-data sources are supported")


def load_tennis_atp(file_name: str = "atp_matches.csv") -> pd.DataFrame:
    local = RAW_DIR / file_name
    return pd.read_parquet(local) if local.exists() and local.suffix.lower() == ".parquet" else pd.read_csv(local) if local.exists() else pd.DataFrame()


def load_tennis_wta(file_name: str = "wta_matches.csv") -> pd.DataFrame:
    local = RAW_DIR / file_name
    return pd.read_parquet(local) if local.exists() and local.suffix.lower() == ".parquet" else pd.read_csv(local) if local.exists() else pd.DataFrame()


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
