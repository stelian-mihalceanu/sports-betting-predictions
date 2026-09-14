"""Small, resilient loaders for the BetLens Streamlit app."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

BASE_DATA_DIR = Path(os.getenv("SPORTS_DATA_DIR", Path(__file__).parent.parent / "data"))
RAW_DIR = BASE_DATA_DIR / "raw"
PROCESSED_DIR = BASE_DATA_DIR / "processed"

TENNIS_ARCHIVE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_LEAGUES = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga", "I1": "Serie A", "RO1": "Liga 1"}


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["league_code", "date", "home_team", "away_team", "competition", "home_goals", "away_goals", "season"])


def _fetch(url: str, code: str, season: str, timeout: int = 6) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        frame = pd.read_csv(BytesIO(response.content), encoding="latin1")
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError, UnicodeError):
        return _empty()
    frame = frame.rename(columns={"Div": "league_code", "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team", "FTHG": "home_goals", "FTAG": "away_goals"}).copy()
    frame["league_code"] = code
    required = {"date", "home_team", "away_team"}
    if not required.issubset(frame.columns):
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
    }).dropna(subset=["date", "home_team", "away_team"])


def _load_seasons(seasons, timeout: int = 6) -> pd.DataFrame:
    jobs = []
    for season in seasons:
        code = str(season)[-2:] + str(int(str(season)[-2:]) + 1).zfill(2)
        for league in FOOTBALL_LEAGUES:
            jobs.append((f"{FOOTBALL_DATA}/{code}/{league}.csv", league, code))
    frames = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_fetch, url, league, season, timeout) for url, league, season in jobs]
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return _empty()
    return pd.concat(frames, ignore_index=True, sort=False).sort_values("date").reset_index(drop=True)


def load_football_matches(file_name: str = "matches.csv", source: str = "local", timeout: int = 6) -> pd.DataFrame:
    local = RAW_DIR / file_name
    if source == "local" and local.exists():
        return pd.read_parquet(local) if local.suffix.lower() == ".parquet" else pd.read_csv(local)
    if source != "football-data":
        raise ValueError("Only local and football-data sources are supported")
    # Two seasons are enough for a lightweight form/Elo baseline.
    return _load_seasons((2025, 2026), timeout=timeout)


def load_upcoming_football_fixtures(timeout: int = 5) -> pd.DataFrame:
    """Load only the current season's five supported leagues.

    No secondary fixture feeds are queried here. A failed league request simply
    disappears from the result instead of breaking the app.
    """
    season = "2627"
    jobs = [(f"{FOOTBALL_DATA}/{season}/{league}.csv", league, season) for league in FOOTBALL_LEAGUES]
    frames = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_fetch, url, league, code, timeout) for url, league, code in jobs]
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return _empty()
    result = pd.concat(frames, ignore_index=True, sort=False)
    today = pd.Timestamp.now().normalize()
    return result[(result["date"] >= today) & result["home_goals"].isna() & result["away_goals"].isna()].drop_duplicates(["date", "home_team", "away_team", "competition"]).sort_values("date").reset_index(drop=True)


def load_tennis_atp(file_name: str = "atp_matches.csv") -> pd.DataFrame:
    local = RAW_DIR / file_name
    if local.exists():
        return pd.read_parquet(local) if local.suffix.lower() == ".parquet" else pd.read_csv(local)
    return pd.DataFrame()


def load_tennis_wta(file_name: str = "wta_matches.csv") -> pd.DataFrame:
    local = RAW_DIR / file_name
    if local.exists():
        return pd.read_parquet(local) if local.suffix.lower() == ".parquet" else pd.read_csv(local)
    return pd.DataFrame()


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
