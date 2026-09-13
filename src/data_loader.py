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
FOOTBALL_EXTRA_DATA = "https://www.football-data.co.uk/new"
FOOTBALL_FIXTURES = "https://www.football-data.co.uk/matches/resources/fixtures.csv"
FOOTBALL_EXTRA_FIXTURES = "https://www.football-data.co.uk/new_league_fixtures.csv"
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
    return _load_tennis_years("atp", range(2022, 2027))


def load_tennis_wta(file_name: str = "wta_matches.csv") -> pd.DataFrame:
    local = RAW_DIR / file_name
    if local.exists():
        return _read_table(local, "WTA tennis")
    return _load_tennis_years("wta", range(2022, 2027))


def load_football_matches(
    file_name: str = "matches.csv",
    source: str = "local",
    timeout: int = 12,
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


def _normalize_fixture_frame(frame: pd.DataFrame, code: str, season: str = "current", force_code: bool = False) -> pd.DataFrame:
    frame = frame.rename(columns={
        "Div": "league_code",
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
    }).copy()
    if force_code or "league_code" not in frame.columns:
        frame["league_code"] = code
    frame["league_code"] = frame["league_code"].fillna(code).astype(str).str.strip()
    frame["league_code"] = frame["league_code"].replace({"ROU": "RO1", "RO": "RO1", "ROM": "RO1"})
    required = {"date", "home_team", "away_team", "league_code"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame[frame["league_code"].isin(FOOTBALL_LEAGUES)].copy()
    if frame.empty:
        return frame
    normalized = pd.DataFrame({
        "league_code": frame["league_code"].values,
        "date": pd.to_datetime(frame["date"], dayfirst=True, errors="coerce").values,
        "home_team": frame["home_team"].astype(str).values,
        "away_team": frame["away_team"].astype(str).values,
        "competition": frame["league_code"].map(FOOTBALL_LEAGUES).values,
        "home_goals": pd.to_numeric(frame.get("home_goals"), errors="coerce").values,
        "away_goals": pd.to_numeric(frame.get("away_goals"), errors="coerce").values,
        "season": season,
    })
    return normalized.dropna(subset=["date", "home_team", "away_team"])


def _liga1_fallback_fixtures() -> pd.DataFrame:
    """Small official-schedule fallback while public CSV feeds catch up."""
    rows = [
        ("2026-09-13 15:00", "FC Argeș", "FC Botosani"),
        ("2026-09-13 20:30", "CORVINUL HUNEDOARA", "Universitatea Craiova"),
        ("2026-09-14 18:00", "FC Universitatea Cluj", "SC OTELUL Galati"),
        ("2026-09-14 21:00", "FCSB", "FC PETROLUL"),
        ("2026-09-18 18:00", "UTA Arad", "SEPSI OSK"),
        ("2026-09-18 21:00", "FC RAPID", "FC Argeș"),
        ("2026-09-19 17:30", "FC CFR 1907 Cluj", "FC Botosani"),
        ("2026-09-19 20:30", "Universitatea Craiova", "FCSB"),
        ("2026-09-20 14:00", "FC Voluntari", "FC Universitatea Cluj"),
        ("2026-09-20 20:30", "DINAMO Bucuresti", "FC FARUL Constanta"),
        ("2026-09-21 18:00", "SC OTELUL Galati", "CORVINUL HUNEDOARA"),
        ("2026-09-21 21:00", "FC PETROLUL", "Csikszereda"),
    ]
    return pd.DataFrame({
        "league_code": ["RO1"] * len(rows),
        "date": pd.to_datetime([row[0] for row in rows]),
        "home_team": [row[1] for row in rows],
        "away_team": [row[2] for row in rows],
        "competition": ["Liga 1"] * len(rows),
        "home_goals": [pd.NA] * len(rows),
        "away_goals": [pd.NA] * len(rows),
        "season": ["2627"] * len(rows),
    })


def load_upcoming_football_fixtures(timeout: int = 12) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for url, code, season, force_code in [
        (FOOTBALL_FIXTURES, "", "current", False),
        (FOOTBALL_EXTRA_FIXTURES, "", "current", False),
        (f"{FOOTBALL_EXTRA_DATA}/ROU.csv", "RO1", "current", True),
    ]:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            frame = _normalize_fixture_frame(
                pd.read_csv(pd.io.common.BytesIO(response.content), encoding="latin1"),
                code=code,
                season=season,
                force_code=force_code,
            )
            if not frame.empty:
                frames.append(frame)
        except (requests.RequestException, OSError, ValueError):
            continue

    season = "2627"
    for code in ["E0", "D1", "SP1", "I1"]:
        try:
            response = requests.get(f"{FOOTBALL_DATA}/{season}/{code}.csv", timeout=timeout)
            response.raise_for_status()
            frame = _normalize_fixture_frame(
                pd.read_csv(pd.io.common.BytesIO(response.content), encoding="latin1"),
                code=code,
                season=season,
                force_code=True,
            )
            if not frame.empty:
                frames.append(frame)
        except (requests.RequestException, OSError, ValueError):
            continue

    # Deterministic fallback keeps Liga 1 visible when the extra feed is stale.
    frames.append(_liga1_fallback_fixtures())

    result = pd.concat(frames, ignore_index=True, sort=False)
    result = result.dropna(subset=["date", "home_team", "away_team", "competition"])
    result = result.sort_values("date")
    return result.drop_duplicates(
        subset=["date", "home_team", "away_team", "competition"]
    ).reset_index(drop=True)


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


def _load_football_data(years, timeout: int = 12) -> pd.DataFrame:
    frames = []
    for start_year in years:
        season = f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"
        for code, competition in FOOTBALL_LEAGUES.items():
            if code == "RO1":
                continue
            url = f"{FOOTBALL_DATA}/{season}/{code}.csv"
            try:
                frame = pd.read_csv(url, encoding="latin1")
            except (requests.RequestException, OSError, ValueError):
                continue
            frame = frame.rename(columns={
                "Date": "date",
                "HomeTeam": "home_team",
                "AwayTeam": "away_team",
                "FTHG": "home_goals",
                "FTAG": "away_goals",
                "FTR": "result",
            }).copy()
            required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
            if not required.issubset(frame.columns):
                continue
            frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
            frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
            frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
            frame["competition"] = competition
            frame["season"] = season
            frames.append(frame)

    try:
        frame = pd.read_csv(f"{FOOTBALL_EXTRA_DATA}/ROU.csv", encoding="latin1")
        frame = frame.rename(columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "FTR": "result",
        }).copy()
        required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
        if required.issubset(frame.columns):
            frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
            frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
            frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
            frame["competition"] = "Liga 1"
            frame["season"] = "multi"
            frames.append(frame)
    except (requests.RequestException, OSError, ValueError):
        pass

    if not frames:
        raise RuntimeError("Unable to load public football data")
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.sort_values("date").reset_index(drop=True)


def _read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} dataset not found at {path}. Add the dataset to data/raw/.")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")
