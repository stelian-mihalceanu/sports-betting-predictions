from __future__ import annotations

from io import BytesIO

import pandas as pd
import requests

POLAND_RESULTS_URL = "https://www.football-data.co.uk/new/POL.csv"
EXTRA_FIXTURES_URL = "https://www.football-data.co.uk/new_league_fixtures.csv"


def _normalize(frame: pd.DataFrame, upcoming: bool = False) -> pd.DataFrame:
    frame = frame.rename(
        columns={
            "Div": "league_code",
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTHG": "home_goals",
            "FTAG": "away_goals",
            "HTHG": "HTHG",
            "HTAG": "HTAG",
            "HC": "HC",
            "AC": "AC",
            "HY": "HY",
            "AY": "AY",
            "HR": "HR",
            "AR": "AR",
        }
    ).copy()
    required = {"date", "home_team", "away_team"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
    frame["home_team"] = frame["home_team"].astype(str).str.strip()
    frame["away_team"] = frame["away_team"].astype(str).str.strip()
    frame["competition"] = "Ekstraklasa"
    frame["league_code"] = "POL"
    if "home_goals" not in frame.columns:
        frame["home_goals"] = pd.NA
    if "away_goals" not in frame.columns:
        frame["away_goals"] = pd.NA
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    for col in ["HTHG", "HTAG", "HC", "AC", "HY", "AY", "HR", "AR"]:
        if col not in frame.columns:
            frame[col] = pd.NA
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["season"] = "2627"
    cols = [
        "league_code", "date", "home_team", "away_team", "competition",
        "home_goals", "away_goals", "HTHG", "HTAG", "HC", "AC", "HY", "AY", "HR", "AR", "season"
    ]
    return frame[cols].dropna(subset=["date", "home_team", "away_team"])


def _read_csv(url: str, timeout: int = 10) -> pd.DataFrame:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content), encoding="latin1")


def load_poland_history(timeout: int = 10) -> pd.DataFrame:
    """Load current Ekstraklasa results plus match-level stats when supplied."""
    try:
        return _normalize(_read_csv(POLAND_RESULTS_URL, timeout))
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()


def load_poland_upcoming(timeout: int = 10) -> pd.DataFrame:
    """Load future Polish fixtures from Football-Data's extra-league fixture feed."""
    try:
        frame = _read_csv(EXTRA_FIXTURES_URL, timeout)
    except (requests.RequestException, OSError, ValueError, pd.errors.ParserError, UnicodeError):
        return pd.DataFrame()

    if "Div" in frame.columns:
        code = frame["Div"].astype(str).str.strip().str.upper()
        mask = code.isin({"POL", "PL", "POL1", "EKSTRAKLASA"})
        frame = frame.loc[mask].copy()
    elif "League" in frame.columns:
        name = frame["League"].astype(str).str.casefold()
        frame = frame.loc[name.str.contains("poland|ekstraklasa", regex=True)].copy()
    else:
        return pd.DataFrame()

    if frame.empty:
        return frame
    result = _normalize(frame, upcoming=True)
    today = pd.Timestamp.now().normalize()
    return result[result["date"] >= today].reset_index(drop=True)
