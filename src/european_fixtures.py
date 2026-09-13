from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

FEEDS = {
    "champions-league-2026": "UEFA Champions League",
    "europa-league-2026": "UEFA Europa League",
    "conference-league-2026": "UEFA Conference League",
}
BASE_URL = "https://fixturedownload.com/feed/json"


def _load_one(slug: str, competition: str, timeout: int = 6) -> pd.DataFrame:
    try:
        response = requests.get(f"{BASE_URL}/{slug}", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError, OSError):
        return pd.DataFrame()

    if not isinstance(payload, list):
        return pd.DataFrame()

    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        date = pd.to_datetime(item.get("DateUtc"), utc=True, errors="coerce")
        home = str(item.get("HomeTeam", "")).strip()
        away = str(item.get("AwayTeam", "")).strip()
        if pd.isna(date) or not home or not away:
            continue
        rows.append({
            "date": date.tz_convert(None),
            "home_team": home,
            "away_team": away,
            "competition": competition,
            "home_goals": pd.to_numeric(item.get("HomeTeamScore"), errors="coerce"),
            "away_goals": pd.to_numeric(item.get("AwayTeamScore"), errors="coerce"),
            "season": "2627",
            "source": "Fixture Download",
        })
    return pd.DataFrame(rows)


def load_european_fixtures(timeout: int = 6) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_load_one, slug, competition, timeout) for slug, competition in FEEDS.items()]
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=[
            "date", "home_team", "away_team", "competition",
            "home_goals", "away_goals", "season", "source",
        ])

    result = pd.concat(frames, ignore_index=True, sort=False)
    result = result[result["date"] >= pd.Timestamp.now().normalize()]
    return result.drop_duplicates(
        subset=["date", "home_team", "away_team", "competition"]
    ).sort_values("date").reset_index(drop=True)
