from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def predict_tennis_match(frame: pd.DataFrame, player_a: str, player_b: str, surface: str | None = None) -> dict[str, Any]:
    """Estimate a tennis matchup from Elo, recent form, serve, H2H and fatigue."""
    data = _prepare(frame, surface or "All")
    elo_a, elo_b = _elo(data, player_a, surface or "All"), _elo(data, player_b, surface or "All")
    form_a, form_b = _recent_win_rate(data, player_a, surface or "All"), _recent_win_rate(data, player_b, surface or "All")
    serve_a, serve_b = _serve_profile(data, player_a, surface or "All"), _serve_profile(data, player_b, surface or "All")
    h2h_a, h2h_b = _h2h(data, player_a, player_b)
    fatigue_a, fatigue_b = _fatigue(data, player_a), _fatigue(data, player_b)

    logit = (
        1.05 * ((elo_a - elo_b) / 400.0)
        + 1.10 * (form_a - form_b)
        + 0.85 * (serve_a["serve_points_won"] - serve_b["serve_points_won"])
        + 0.75 * (serve_a["return_points_won"] - serve_b["return_points_won"])
        - 0.20 * (fatigue_a - fatigue_b)
    )
    p_a = min(0.95, max(0.05, 1.0 / (1.0 + math.exp(-logit))))
    h2h_total = h2h_a + h2h_b
    if h2h_total >= 3:
        p_a = 0.92 * p_a + 0.08 * (h2h_a / h2h_total)

    return {
        "player_a": player_a, "player_b": player_b, "surface": surface or "All",
        "a_win": p_a, "b_win": 1.0 - p_a, "pick": player_a if p_a >= 0.5 else player_b,
        "confidence": max(p_a, 1.0 - p_a), "elo_a": elo_a, "elo_b": elo_b,
        "form_a": form_a, "form_b": form_b, "fatigue_a": fatigue_a, "fatigue_b": fatigue_b,
        "serve_a": serve_a["serve_points_won"], "serve_b": serve_b["serve_points_won"],
        "return_a": serve_a["return_points_won"], "return_b": serve_b["return_points_won"],
        "aces_a": serve_a["aces"], "aces_b": serve_b["aces"],
        "double_faults_a": serve_a["double_faults"], "double_faults_b": serve_b["double_faults"],
        "h2h_a": h2h_a, "h2h_b": h2h_b, "h2h_total": h2h_total,
    }


def _prepare(frame: pd.DataFrame, surface: str) -> pd.DataFrame:
    data = frame.copy()
    data["date"] = pd.to_datetime(data["tourney_date"], format="%Y%m%d", errors="coerce")
    data = data.dropna(subset=["date", "winner_name", "loser_name"]).sort_values("date")
    if surface != "All" and "surface" in data.columns:
        data = data[data["surface"].astype(str).eq(surface)]
    return data


def _elo(frame: pd.DataFrame, player: str, surface: str) -> float:
    ratings: dict[str, float] = {}
    surface_ratings: dict[str, float] = {}
    for m in frame.itertuples(index=False):
        winner, loser = str(m.winner_name), str(m.loser_name)
        rw, rl = ratings.get(winner, 1500.0), ratings.get(loser, 1500.0)
        expected = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
        ratings[winner], ratings[loser] = rw + 24.0 * (1.0 - expected), rl - 24.0 * (1.0 - expected)
        surf = str(getattr(m, "surface", "Unknown"))
        if surf != "Unknown":
            sw, sl = surface_ratings.get(winner, 1500.0), surface_ratings.get(loser, 1500.0)
            sexp = 1.0 / (1.0 + 10 ** ((sl - sw) / 400.0))
            surface_ratings[winner], surface_ratings[loser] = sw + 24.0 * (1.0 - sexp), sl - 24.0 * (1.0 - sexp)
    return surface_ratings.get(player, ratings.get(player, 1500.0)) if surface != "All" else ratings.get(player, 1500.0)


def _recent_win_rate(frame: pd.DataFrame, player: str, surface: str) -> float:
    mask = frame["winner_name"].eq(player) | frame["loser_name"].eq(player)
    if surface != "All":
        mask &= frame["surface"].astype(str).eq(surface)
    subset = frame.loc[mask].tail(12)
    return float(np.mean(subset["winner_name"].eq(player))) if not subset.empty else 0.5


def _fatigue(frame: pd.DataFrame, player: str) -> float:
    """Higher score means more recent workload/density."""
    mask = frame["winner_name"].eq(player) | frame["loser_name"].eq(player)
    subset = frame.loc[mask].tail(8)
    if subset.empty:
        return 0.0
    dates = subset["date"].dropna().sort_values().tolist()
    if len(dates) < 2:
        return 0.0
    gaps = np.diff(np.array(dates, dtype="datetime64[D]")).astype(int)
    density = max(0.0, 3.0 - float(np.mean(np.minimum(gaps, 7))) / 2.0)
    return float(density + 0.25 * max(0, 4 - len(set(dates))))


def _serve_profile(frame: pd.DataFrame, player: str, surface: str) -> dict[str, float]:
    mask = frame["winner_name"].eq(player) | frame["loser_name"].eq(player)
    if surface != "All": mask &= frame["surface"].astype(str).eq(surface)
    subset = frame.loc[mask].tail(40)
    rows = []
    for m in subset.itertuples(index=False):
        won = str(m.winner_name) == player
        prefix, opp = ("w_", "l_") if won else ("l_", "w_")
        svpt = _num(getattr(m, prefix + "svpt", np.nan))
        first = _num(getattr(m, prefix + "1stWon", np.nan)) or 0.0
        second = _num(getattr(m, prefix + "2ndWon", np.nan)) or 0.0
        ace = _num(getattr(m, prefix + "ace", np.nan))
        df = _num(getattr(m, prefix + "df", np.nan))
        opp_svpt = _num(getattr(m, opp + "svpt", np.nan))
        opp_first = _num(getattr(m, opp + "1stWon", np.nan)) or 0.0
        opp_second = _num(getattr(m, opp + "2ndWon", np.nan)) or 0.0
        rows.append({"serve_points_won": (first + second) / svpt if svpt and svpt > 0 else np.nan, "return_points_won": 1.0 - ((opp_first + opp_second) / opp_svpt) if opp_svpt and opp_svpt > 0 else np.nan, "aces": ace, "double_faults": df})
    result = pd.DataFrame(rows)
    if result.empty:
        return {"serve_points_won": 0.62, "return_points_won": 0.38, "aces": 4.0, "double_faults": 2.0}
    defaults = {"serve_points_won": 0.62, "return_points_won": 0.38, "aces": 4.0, "double_faults": 2.0}
    return {k: _safe_mean(result[k], v) for k, v in defaults.items()}


def _h2h(frame: pd.DataFrame, player_a: str, player_b: str) -> tuple[int, int]:
    pair = frame[((frame["winner_name"].eq(player_a)) & (frame["loser_name"].eq(player_b))) | ((frame["winner_name"].eq(player_b)) & (frame["loser_name"].eq(player_a)))]
    if pair.empty:
        return 0, 0
    weights = np.exp(np.linspace(-1.5, 0.0, len(pair)))
    a_wins = float(np.sum(weights * pair["winner_name"].eq(player_a).to_numpy()))
    b_wins = float(np.sum(weights * pair["winner_name"].eq(player_b).to_numpy()))
    return int(round(a_wins)), int(round(b_wins))


def _num(value) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _safe_mean(series: pd.Series, default: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else default
