"""Feature engineering utilities for tennis and football prediction models."""

from __future__ import annotations

from typing import List
import pandas as pd
import numpy as np


def add_tennis_form_features(df: pd.DataFrame, window: int = 10, min_periods: int = 3) -> pd.DataFrame:
    _require_columns(df, ["tourney_date", "tourney_name", "winner_name", "loser_name"])
    result = df.sort_values(["tourney_date", "tourney_name"]).reset_index(drop=True).copy()
    for side, player_col, games_col in [
        ("winner", "winner_name", "winner_games_won"),
        ("loser", "loser_name", "loser_games_won"),
    ]:
        if games_col in result.columns:
            stats = _player_rolling_stats(result, player_col, games_col, window, min_periods)
            result = result.join(stats.add_prefix(f"{side}_"))
    return result


def _player_rolling_stats(df: pd.DataFrame, player_col: str, stats_col: str, window: int, min_periods: int) -> pd.DataFrame:
    grouped = df.groupby(player_col, sort=False)[stats_col]
    return pd.DataFrame({
        "form_mean": grouped.transform(lambda x: x.rolling(window, min_periods=min_periods).mean().shift(1)),
        "form_std": grouped.transform(lambda x: x.rolling(window, min_periods=min_periods).std().shift(1)),
    }).replace([np.inf, -np.inf], np.nan).fillna(0)


def add_elo_features(df: pd.DataFrame, surface_weighted: bool = True, k_factor: float = 32.0, initial_rating: float = 1500.0) -> pd.DataFrame:
    """Calculate pre-match tennis ELO, optionally separated by surface."""
    _require_columns(df, ["winner_name", "loser_name"])
    result = df.copy()
    ratings: dict[tuple[str, str], float] = {}
    global_ratings: dict[str, float] = {}
    winner_elos, loser_elos = [], []
    for _, row in result.iterrows():
        surface = str(row.get("surface", "ALL")) if surface_weighted else "ALL"
        winner = str(row["winner_name"])
        loser = str(row["loser_name"])
        key_w, key_l = (surface, winner), (surface, loser)
        rw = ratings.get(key_w, global_ratings.get(winner, initial_rating))
        rl = ratings.get(key_l, global_ratings.get(loser, initial_rating))
        winner_elos.append(rw)
        loser_elos.append(rl)
        expected_w = 1.0 / (1.0 + 10.0 ** ((rl - rw) / 400.0))
        ratings[key_w] = rw + k_factor * (1.0 - expected_w)
        ratings[key_l] = rl + k_factor * (0.0 - (1.0 - expected_w))
        global_ratings[winner] = ratings[key_w]
        global_ratings[loser] = ratings[key_l]
    result["winner_elo"] = winner_elos
    result["loser_elo"] = loser_elos
    result["elo_diff"] = result["winner_elo"] - result["loser_elo"]
    return result


def add_football_form_features(df: pd.DataFrame, window: int = 5, min_periods: int = 2) -> pd.DataFrame:
    _require_columns(df, ["home_team", "away_team", "home_goals", "away_goals", "date"])
    result = df.sort_values(["date"]).reset_index(drop=True).copy()
    history: dict[str, list[tuple[float, float, float]]] = {}
    rows = []
    for _, row in result.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        hf = _team_summary(history.get(home, []), window, min_periods)
        af = _team_summary(history.get(away, []), window, min_periods)
        rows.append({"home_form_points": hf[0], "home_form_goals_scored": hf[1], "home_form_goals_conceded": hf[2],
                     "away_form_points": af[0], "away_form_goals_scored": af[1], "away_form_goals_conceded": af[2]})
        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        hp, ap = (3.0, 0.0) if hg > ag else ((0.0, 3.0) if hg < ag else (1.0, 1.0))
        history.setdefault(home, []).append((hp, hg, ag))
        history.setdefault(away, []).append((ap, ag, hg))
    return result.join(pd.DataFrame(rows))


def _team_summary(history: list[tuple[float, float, float]], window: int, min_periods: int) -> tuple[float, float, float]:
    recent = history[-window:]
    if len(recent) < min_periods:
        return 0.0, 0.0, 0.0
    arr = np.asarray(recent, dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean())


def add_elo_football_features(df: pd.DataFrame, k_factor: float = 32.0, initial_rating: float = 1500.0, home_advantage: float = 60.0) -> pd.DataFrame:
    """Calculate pre-match football ELO ratings without leaking the result."""
    _require_columns(df, ["home_team", "away_team", "home_goals", "away_goals"])
    result = df.sort_values("date").reset_index(drop=True).copy() if "date" in df.columns else df.copy()
    ratings: dict[str, float] = {}
    home_elos, away_elos = [], []
    for _, row in result.iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        rh, ra = ratings.get(home, initial_rating), ratings.get(away, initial_rating)
        home_elos.append(rh)
        away_elos.append(ra)
        expected_home = 1.0 / (1.0 + 10.0 ** (((ra) - (rh + home_advantage)) / 400.0))
        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        actual_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        change = k_factor * (actual_home - expected_home)
        ratings[home] = rh + change
        ratings[away] = ra - change
    result["home_elo"] = home_elos
    result["away_elo"] = away_elos
    result["elo_diff"] = result["home_elo"] - result["away_elo"]
    return result


def _require_columns(df: pd.DataFrame, columns: List[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
