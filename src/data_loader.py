"""Data loading utilities for tennis and football datasets."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests

BASE_DATA_DIR = Path(os.getenv("SPORTS_DATA_DIR", Path(__file__).parent.parent / "data"))
RAW_DIR = BASE_DATA_DIR / "raw"
PROCESSED_DIR = BASE_DATA_DIR / "processed"

TENNIS_ARCHIVE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_FIXTURES = "https://www.football-data.co.uk/matches/resources/fixtures.csv"
FOOTBALL_LEAGUES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "RO1": "Liga 1",
}


def load_tennis_atp(file_name: str = "atp_matches.csv") -> pd.DataFrame:
    local = RAW_DIR / file_name
    if local.exists():
        return _read_table(local, "ATP tennis")
    years = range(2022, 2027)
    return _load_tennis_years("atp", years)


def load_tennis_wta(file_name: str = "wta_matches.csv") -> pd.DataFrame:
    local = RAW_DIR / file_name
    if local.exists():
        return _read_table(local, "WTA tennis")
    years = range(2022, 2027)
    return _load_tennis_years("wta", years)


def load_football_matches(
    file_name: str = "matches.csv",
    source: str = "local",
    timeout: int = 20,
) -> pd.DataFrame:
    local = RAW_DIR / file_name
    if source == "local" and local.exists():
        return _read_table(local, "football")
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
        return pd.DataFrame(response.json())
    if source not in {"local", "football-data"}:
        raise ValueError(f"Unknown source: {source}")
    return _load_football_data(range(2022, 2027), timeout=timeout)


def load_upcoming_football_fixtures(timeout: int = 20) -> pd.DataFrame:
    """Load current/future fixtures from both the weekly feed and season CSVs."""
    frames: list[pd.DataFrame] = []

    # The weekly feed covers the main leagues, but Romania is published as an
    # extra league. Keep it in scope by also reading the current-season CSVs.
    try:
        response = requests.get(FOOTBALL_FIXTURES, timeout=timeout)
        response.raise_for_status()
        frames.append(pd.read_csv(pd.io.common.BytesIO(response.content), encoding="latin1"))
    except (requests.RequestException, OSError, ValueError):
        pass

    current_season = "2627"
    for code in FOOTBALL_LEAGUES:
        url = f"{FOOTBALL_DATA}/{current_season}/{code}.csv"
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            frames.append(pd.read_csv(pd.io.common.BytesIO(response.content), encoding="latin1"))
        except (requests.RequestException, OSError, ValueError):
            continue

    if not frames:
        raise RuntimeError("Unable to load current football fixtures")

    normalized_frames = []
    for frame in frames:
        required = {"Div", "Date", "HomeTeam", "AwayTeam"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.rename(columns={
            "Div": "league_code",
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
        })
        frame = frame[frame["league_code"].isin(FOOTBALL_LEAGUES)].copy()
        if frame.empty:
            continue
        frame["competition"] = frame["league_code"].map(FOOTBALL_LEAGUES)
        frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
        frame["home_goals"] = pd.to_numeric(frame.get("home_goals"), errors="coerce")
        frame["away_goals"] = pd.to_numeric(frame.get("away_goals"), errors="coerce")
        frame["season"] = "current"
        normalized_frames.append(frame[["league_code", "date", "home_team", "away_team", "competition", "home_goals", "away_goals", "season"]])

    if not normalized_frames:
        raise RuntimeError("Football fixture feeds are missing expected columns")

    result = pd.concat(normalized_frames, ignore_index=True)
    result = result.dropna(subset=["date", "home_team", "away_team"])
    result = result[result["home_goals"].isna() & result["away_goals"].isna()]
    return result.drop_duplicates(subset=["date", "home_team", "away_team", "competition"]).sort_values("date").reset_index(drop=True)


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


def _load_tennis_years(tour: str, years) -> pd.DataFrame:
    frames = []
    for year in years:
        url = f"{TENNIS_ARCHIVE}/{tour}/{tour}_matches_{year}.csv"
        try:
            frame = pd.read_csv(url)
            frame["tour"] = tour.upper()
            frames.append(frame)
        except (requests.RequestException, OSError, ValueError):
            continue
    if not frames:
        raise RuntimeError(f"Unable to load remote {tour.upper()} tennis data")
    return pd.concat(frames, ignore_index=True)


def _load_football_data(years, timeout: int = 20) -> pd.DataFrame:
    frames = []
    for start_year in years:
        season = f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"
        for code, competition in FOOTBALL_LEAGUES.items():
            url = f"{FOOTBALL_DATA}/{season}/{code}.csv"
            try:
                frame = pd.read_csv(url, encoding="latin1")
            except (requests.RequestException, OSError, ValueError):
                continue
            frame = frame.rename(
                columns={
                    "Date": "date",
                    "HomeTeam": "home_team",
                    "AwayTeam": "away_team",
                    "FTHG": "home_goals",
                    "FTAG": "away_goals",
                    "FTR": "result",
                }
            )
            required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
            if not required.issubset(frame.columns):
                continue
            frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
            frame["competition"] = competition
            frame["season"] = season
            frames.append(frame)
    if not frames:
        raise RuntimeError("Unable to load public football data")
    result = pd.concat(frames, ignore_index=True)
    for column in ["home_goals", "away_goals"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values("date").reset_index(drop=True)


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
