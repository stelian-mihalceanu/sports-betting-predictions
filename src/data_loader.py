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
FIXTURE_DOWNLOAD = "https://fixturedownload.com/download"
EUROPEAN_FIXTURES = {
    "champions-league": "UEFA Champions League",
    "europa-league": "UEFA Europa League",
    "conference-league": "UEFA Conference League",
}
FOOTBALL_LEAGUES = {"E0": "Premier League", "D1": "Bundesliga", "SP1": "La Liga", "I1": "Serie A", "RO1": "Liga 1"}


def load_tennis_atp(file_name: str = "atp_matches.csv") -> pd.DataFrame:
    local = RAW_DIR / file_name
    if local.exists():
        return _read_table(local, "ATP tennis")
    return _load_tennis_years("atp", range(2024, 2027))


def load_tennis_wta(file_name: str = "wta_matches.csv") -> pd.DataFrame:
    local = RAW_DIR / file_name
    if local.exists():
        return _read_table(local, "WTA tennis")
    return _load_tennis_years("wta", range(2024, 2027))


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
    frame = frame.rename(columns={"Div": "league_code", "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team", "FTHG": "home_goals", "FTAG": "away_goals"}).copy()
    if force_code or "league_code" not in frame.columns:
        frame["league_code"] = code
    frame["league_code"] = frame["league_code"].fillna(code).astype(str).str.strip().replace({"ROU": "RO1", "RO": "RO1", "ROM": "RO1"})
    required = {"date", "home_team", "away_team", "league_code"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame[frame["league_code"].isin(FOOTBALL_LEAGUES)].copy()
    if frame.empty:
        return frame
    home_goals = frame["home_goals"] if "home_goals" in frame.columns else pd.Series(pd.NA, index=frame.index)
    away_goals = frame["away_goals"] if "away_goals" in frame.columns else pd.Series(pd.NA, index=frame.index)
    return pd.DataFrame({
        "league_code": frame["league_code"].to_numpy(),
        "date": pd.to_datetime(frame["date"], dayfirst=True, errors="coerce").to_numpy(),
        "home_team": frame["home_team"].astype(str).to_numpy(),
        "away_team": frame["away_team"].astype(str).to_numpy(),
        "competition": frame["league_code"].map(FOOTBALL_LEAGUES).to_numpy(),
        "home_goals": pd.to_numeric(home_goals, errors="coerce").to_numpy(),
        "away_goals": pd.to_numeric(away_goals, errors="coerce").to_numpy(),
        "season": [season] * len(frame),
    }).dropna(subset=["date", "home_team", "away_team"])


def _liga1_fallback_fixtures() -> pd.DataFrame:
    rows = [("2026-09-14 18:00", "FC Universitatea Cluj", "SC OTELUL Galati"), ("2026-09-14 21:00", "FCSB", "FC PETROLUL"), ("2026-09-18 18:00", "UTA Arad", "SEPSI OSK"), ("2026-09-18 21:00", "FC RAPID", "FC Argeș"), ("2026-09-19 17:30", "FC CFR 1907 Cluj", "FC Botosani"), ("2026-09-19 20:30", "Universitatea Craiova", "FCSB"), ("2026-09-20 14:00", "FC Voluntari", "FC Universitatea Cluj"), ("2026-09-20 20:30", "DINAMO Bucuresti", "FC FARUL Constanta"), ("2026-09-21 18:00", "SC OTELUL Galati", "CORVINUL HUNEDOARA"), ("2026-09-21 21:00", "FC PETROLUL", "Csikszereda")]
    return pd.DataFrame({"league_code": ["RO1"] * len(rows), "date": pd.to_datetime([r[0] for r in rows]), "home_team": [r[1] for r in rows], "away_team": [r[2] for r in rows], "competition": ["Liga 1"] * len(rows), "home_goals": [pd.NA] * len(rows), "away_goals": [pd.NA] * len(rows), "season": ["2627"] * len(rows)})


def _fetch_fixture_source(url: str, code: str = "", season: str = "current", force_code: bool = False, timeout: int = 8) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return _normalize_fixture_frame(pd.read_csv(pd.io.common.BytesIO(response.content), encoding="latin1"), code=code, season=season, force_code=force_code)
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()


def _fetch_fixture_download(url: str, competition: str, timeout: int = 8) -> pd.DataFrame:
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        frame = pd.read_csv(pd.io.common.BytesIO(response.content), encoding="utf-8")
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    date_col, home_col, away_col = columns.get("date"), columns.get("home team"), columns.get("away team")
    result_col = columns.get("result")
    if not all([date_col, home_col, away_col]):
        return pd.DataFrame()
    result = pd.DataFrame({"league_code": competition, "date": pd.to_datetime(frame[date_col], dayfirst=True, errors="coerce"), "home_team": frame[home_col].astype(str).str.strip(), "away_team": frame[away_col].astype(str).str.strip(), "competition": competition, "home_goals": pd.NA, "away_goals": pd.NA, "season": "2627"})
    if result_col:
        parsed = frame[result_col].astype(str).str.extract(r"(\d+)\s*[-–]\s*(\d+)")
        result["home_goals"], result["away_goals"] = pd.to_numeric(parsed[0], errors="coerce"), pd.to_numeric(parsed[1], errors="coerce")
    return result.dropna(subset=["date", "home_team", "away_team"])


def load_upcoming_football_fixtures(timeout: int = 6) -> pd.DataFrame:
    frames = []
    for url in (FOOTBALL_FIXTURES, FOOTBALL_EXTRA_FIXTURES):
        frame = _fetch_fixture_source(url, timeout=timeout)
        if not frame.empty:
            frames.append(frame)
    now = pd.Timestamp.now().normalize()
    future_codes: set[str] = set()
    if frames:
        initial = pd.concat(frames, ignore_index=True, sort=False)
        future_codes = set(initial.loc[(initial["date"] >= now) & initial["home_goals"].isna() & initial["away_goals"].isna(), "league_code"].dropna().astype(str))

    for code in ("E0", "D1", "SP1", "I1"):
        if code in future_codes:
            continue
        frame = _fetch_fixture_source(f"{FOOTBALL_DATA}/2627/{code}.csv", code=code, season="2627", force_code=True, timeout=timeout)
        if not frame.empty:
            frames.append(frame)
            future_codes.update(frame.loc[(frame["date"] >= now) & frame["home_goals"].isna() & frame["away_goals"].isna(), "league_code"].dropna().astype(str))

    if "RO1" not in future_codes:
        frame = _fetch_fixture_source(f"{FOOTBALL_EXTRA_DATA}/ROU.csv", code="RO1", season="current", force_code=True, timeout=timeout)
        if not frame.empty:
            frames.append(frame)
            future_codes.update(frame.loc[(frame["date"] >= now) & frame["home_goals"].isna() & frame["away_goals"].isna(), "league_code"].dropna().astype(str))
        if "RO1" not in future_codes:
            frames.append(_liga1_fallback_fixtures())

    def load_european(item):
        slug, competition = item
        return _fetch_fixture_download(f"{FIXTURE_DOWNLOAD}/{slug}-2026-UTC.csv", competition, timeout=timeout)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(load_european, item) for item in EUROPEAN_FIXTURES.items()]
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["league_code", "date", "home_team", "away_team", "competition", "home_goals", "away_goals", "season"])
    result = pd.concat(frames, ignore_index=True, sort=False).dropna(subset=["date", "home_team", "away_team", "competition"])
    return result.sort_values("date").drop_duplicates(["date", "home_team", "away_team", "competition"]).reset_index(drop=True)


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
    def fetch(year: int) -> pd.DataFrame:
        try:
            response = requests.get(f"{TENNIS_ARCHIVE}/{tour}/{tour}_matches_{year}.csv", timeout=12)
            response.raise_for_status()
            return pd.read_csv(pd.io.common.BytesIO(response.content))
        except (requests.RequestException, OSError, ValueError, pd.errors.ParserError):
            return pd.DataFrame()
    year_list = list(years)
    with ThreadPoolExecutor(max_workers=min(3, len(year_list))) as executor:
        frames = [future.result() for future in [executor.submit(fetch, year) for year in year_list]]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_football_data(seasons, timeout: int = 8) -> pd.DataFrame:
    frames, jobs = [], []
    for season in seasons:
        code = str(season)[-2:] + str(int(str(season)[-2:]) + 1).zfill(2)
        for league_code in FOOTBALL_LEAGUES:
            jobs.append((f"{FOOTBALL_DATA}/{code}/{league_code}.csv", league_code, code))
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(_fetch_fixture_source, url, league_code, code, True, timeout): (league_code, code) for url, league_code, code in jobs}
        for future in as_completed(future_map):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["league_code", "date", "home_team", "away_team", "competition", "home_goals", "away_goals", "season"])
    return pd.concat(frames, ignore_index=True, sort=False).sort_values("date").reset_index(drop=True)


def _read_table(path: Path, label: str) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
