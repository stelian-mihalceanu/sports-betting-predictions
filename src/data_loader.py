"""Data loading utilities for tennis and football datasets."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def load_football_matches(file_name: str = "matches.csv", source: str = "local", timeout: int = 8) -> pd.DataFrame:
    local = RAW_DIR / file_name
    if source == "local" and local.exists():
        return _read_table(local, "football")
    if source == "openfoot":
        api_key = os.getenv("OPENFOOT_API_KEY")
        if not api_key:
            raise RuntimeError("OPENFOOT_API_KEY is not configured")
        response = requests.get("https://api.openfootapi.com/fixtures", params={"key": api_key}, timeout=timeout)
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
    return pd.DataFrame({
        "league_code": frame["league_code"].to_numpy(),
        "date": pd.to_datetime(frame["date"], dayfirst=True, errors="coerce").to_numpy(),
        "home_team": frame["home_team"].astype(str).to_numpy(),
        "away_team": frame["away_team"].astype(str).to_numpy(),
        "competition": frame["league_code"].map(FOOTBALL_LEAGUES).to_numpy(),
        "home_goals": pd.to_numeric(frame.get("home_goals"), errors="coerce").to_numpy(),
        "away_goals": pd.to_numeric(frame.get("away_goals"), errors="coerce").to_numpy(),
        "season": [season] * len(frame),
    }).dropna(subset=["date", "home_team", "away_team"])


def _liga1_fallback_fixtures() -> pd.DataFrame:
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


def _fetch_fixture_source(url: str, code: str = "", season: str = "current", force_code: bool = False, timeout: int = 8) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return _normalize_fixture_frame(
            pd.read_csv(pd.io.common.BytesIO(response.content), encoding="latin1"),
            code=code,
            season=season,
            force_code=force_code,
        )
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()


def load_upcoming_football_fixtures(timeout: int = 8) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for url in [FOOTBALL_FIXTURES, FOOTBALL_EXTRA_FIXTURES]:
        frame = _fetch_fixture_source(url, timeout=timeout)
        if not frame.empty:
            frames.append(frame)

    present_codes = set()
    if frames:
        present_codes = set(pd.concat(frames, ignore_index=True)["league_code"].dropna().astype(str))

    if "RO1" not in present_codes:
        frame = _fetch_fixture_source(
            f"{FOOTBALL_EXTRA_DATA}/ROU.csv",
            code="RO1",
            season="current",
            force_code=True,
            timeout=timeout,
        )
        if not frame.empty:
            frames.append(frame)
            present_codes.add("RO1")

    season = "2627"
    for code in ["E0", "D1", "SP1", "I1"]:
        if code in present_codes:
            continue
        frame = _fetch_fixture_source(
            f"{FOOTBALL_DATA}/{season}/{code}.csv",
            code=code,
            season=season,
            force_code=True,
            timeout=timeout,
        )
        if not frame.empty:
            frames.append(frame)
            present_codes.add(code)

    if "RO1" not in present_codes:
        frames.append(_liga1_fallback_fixtures())

    if not frames:
        return pd.DataFrame(columns=[
            "league_code", "date", "home_team", "away_team",
            "competition", "home_goals", "away_goals", "season",
        ])

    result = pd.concat(frames, ignore_index=True, sort=False)
    result = result.dropna(subset=["date", "home_team", "away_team", "competition"])
    return result.sort_values("date").drop_duplicates(
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


def _load_one_football_file(start_year: int, code: str, competition: str, timeout: int) -> pd.DataFrame:
    season = f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"
    url = f"{FOOTBALL_DATA}/{season}/{code}.csv"
    try:
        frame = pd.read_csv(url, encoding="latin1")
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()

    frame = frame.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",
    })
    required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    if not all(column in frame.columns for column in required):
        return pd.DataFrame()

    columns = required + (["result"] if "result" in frame.columns else [])
    normalized = frame[columns].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], dayfirst=True, errors="coerce")
    normalized["home_goals"] = pd.to_numeric(normalized["home_goals"], errors="coerce")
    normalized["away_goals"] = pd.to_numeric(normalized["away_goals"], errors="coerce")
    normalized = normalized.dropna(subset=["date", "home_team", "away_team"])
    normalized["competition"] = competition
    normalized["season"] = f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"
    return normalized


def _load_football_data(years, timeout: int = 8) -> pd.DataFrame:
    jobs = [
        (start_year, code, competition)
        for start_year in years
        for code, competition in FOOTBALL_LEAGUES.items()
        if code != "RO1"
    ]
    frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_load_one_football_file, start_year, code, competition, timeout): (start_year, code)
            for start_year, code, competition in jobs
        }
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
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
        })
        required = ["date", "home_team", "away_team", "home_goals", "away_goals"]
        if all(column in frame.columns for column in required):
            columns = required + (["result"] if "result" in frame.columns else [])
            normalized = frame[columns].copy()
            normalized["date"] = pd.to_datetime(normalized["date"], dayfirst=True, errors="coerce")
            normalized["home_goals"] = pd.to_numeric(normalized["home_goals"], errors="coerce")
            normalized["away_goals"] = pd.to_numeric(normalized["away_goals"], errors="coerce")
            normalized = normalized.dropna(subset=["date", "home_team", "away_team"])
            normalized["competition"] = "Liga 1"
            normalized["season"] = "multi"
            frames.append(normalized)
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError):
        pass

    if not frames:
        raise RuntimeError("Unable to load public football data")
    return pd.concat(frames, ignore_index=True, sort=False).sort_values("date").reset_index(drop=True)


def _read_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} dataset not found at {path}. Add the dataset to data/raw/.")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {suffix}")