from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def predict_tennis_match(player_a: str, player_b: str, matches: pd.DataFrame, surface: str = "All") -> dict[str, Any]:
    """Estimate a tennis matchup from Elo, recent form, surface form, serve and H2H."""
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["tourney_date"], format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["date", "winner_name", "loser_name"]).sort_values("date")

    elo_a, elo_b = _elo(frame, player_a, surface), _elo(frame, player_b, surface)
    form_a, form_b = _recent_win_rate(frame, player_a, surface), _recent_win_rate(frame, player_b, surface)
    serve_a, serve_b = _serve_profile(frame, player_a, surface), _serve_profile(frame, player_b, surface)
    h2h_a, h2h_b = _h2h(frame, player_a, player_b)
    sample_a, sample_b = _match_count(frame, player_a), _match_count(frame, player_b)

    rating_edge = (elo_a - elo_b) / 400.0
    form_edge = form_a - form_b
    serve_edge = serve_a["serve_points_won"] - serve_b["serve_points_won"]
    break_edge = serve_a["return_points_won"] - serve_b["return_points_won"]
    logit = 1.05 * rating_edge + 1.10 * form_edge + 0.85 * serve_edge + 0.75 * break_edge
    p_a = 1.0 / (1.0 + math.exp(-logit))
    p_a = min(0.95, max(0.05, p_a))

    if h2h_a + h2h_b >= 3:
        p_a = 0.90 * p_a + 0.10 * (h2h_a / (h2h_a + h2h_b))

    weakest_sample = min(sample_a, sample_b)
    full_confidence_threshold = 15
    shrink = min(1.0, weakest_sample / full_confidence_threshold)
    p_a = 0.5 + shrink * (p_a - 0.5)

    return {
        "player_a": player_a, "player_b": player_b, "surface": surface,
        "a_win": p_a, "b_win": 1.0 - p_a, "pick": player_a if p_a >= 0.5 else player_b,
        "confidence": max(p_a, 1.0 - p_a), "elo_a": elo_a, "elo_b": elo_b,
        "form_a": form_a, "form_b": form_b,
        "serve_a": serve_a["serve_points_won"], "serve_b": serve_b["serve_points_won"],
        "return_a": serve_a["return_points_won"], "return_b": serve_b["return_points_won"],
        "aces_a": serve_a["aces"], "aces_b": serve_b["aces"],
        "double_faults_a": serve_a["double_faults"], "double_faults_b": serve_b["double_faults"],
        "h2h_a": h2h_a, "h2h_b": h2h_b, "h2h_total": h2h_a + h2h_b,
        "sample_a": sample_a, "sample_b": sample_b, "low_sample": weakest_sample < full_confidence_threshold,
    }


def _match_count(frame: pd.DataFrame, player: str) -> int:
    return int((frame["winner_name"].eq(player) | frame["loser_name"].eq(player)).sum())


def _elo(frame: pd.DataFrame, player: str, surface: str) -> float:
    ratings: dict[str, float] = {}
    surface_ratings: dict[str, float] = {}
    for _, m in frame.iterrows():
        winner, loser = str(m["winner_name"]), str(m["loser_name"])
        rw, rl = ratings.get(winner, 1500.0), ratings.get(loser, 1500.0)
        expected = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
        ratings[winner] = rw + 24.0 * (1.0 - expected)
        ratings[loser] = rl - 24.0 * (1.0 - expected)
        surf = str(m.get("surface", "Unknown"))
        if surf != "Unknown":
            sw, sl = surface_ratings.get(winner, 1500.0), surface_ratings.get(loser, 1500.0)
            sexp = 1.0 / (1.0 + 10 ** ((sl - sw) / 400.0))
            surface_ratings[winner] = sw + 24.0 * (1.0 - sexp)
            surface_ratings[loser] = sl - 24.0 * (1.0 - sexp)
    if surface != "All":
        return surface_ratings.get(player, ratings.get(player, 1500.0))
    return ratings.get(player, 1500.0)


def _recent_win_rate(frame: pd.DataFrame, player: str, surface: str) -> float:
    mask = frame["winner_name"].eq(player) | frame["loser_name"].eq(player)
    if surface != "All": mask &= frame["surface"].astype(str).eq(surface)
    subset = frame.loc[mask].tail(12)
    if subset.empty: return 0.5
    return float(np.mean(subset["winner_name"].eq(player)))


def _serve_profile(frame: pd.DataFrame, player: str, surface: str) -> dict[str, float]:
    mask = frame["winner_name"].eq(player) | frame["loser_name"].eq(player)
    if surface != "All": mask &= frame["surface"].astype(str).eq(surface)
    subset = frame.loc[mask].tail(40)
    rows = []
    for _, m in subset.iterrows():
        won = str(m["winner_name"]) == player
        prefix = "w_" if won else "l_"
        svpt = _num(m.get(prefix + "svpt"))
        first = _num(m.get(prefix + "1stWon")) or 0.0
        second = _num(m.get(prefix + "2ndWon")) or 0.0
        ace = _num(m.get(prefix + "ace"))
        df = _num(m.get(prefix + "df"))
        opp = "l_" if won else "w_"
        opp_svpt = _num(m.get(opp + "svpt"))
        opp_first = _num(m.get(opp + "1stWon")) or 0.0
        opp_second = _num(m.get(opp + "2ndWon")) or 0.0
        return_points = 1.0 - ((opp_first + opp_second) / opp_svpt) if opp_svpt and opp_svpt > 0 else np.nan
        rows.append({
            "serve_points_won": (first + second) / svpt if svpt and svpt > 0 else np.nan,
            "return_points_won": return_points,
            "aces": ace,
            "double_faults": df,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return {"serve_points_won": 0.62, "return_points_won": 0.38, "aces": 4.0, "double_faults": 2.0}
    return {c: _safe_mean(result[c], default) for c, default in {"serve_points_won": 0.62, "return_points_won": 0.38, "aces": 4.0, "double_faults": 2.0}.items()}


def _h2h(frame: pd.DataFrame, player_a: str, player_b: str) -> tuple[int, int]:
    pair = frame[((frame["winner_name"].eq(player_a)) & (frame["loser_name"].eq(player_b))) | ((frame["winner_name"].eq(player_b)) & (frame["loser_name"].eq(player_a)))]
    a = int(pair["winner_name"].eq(player_a).sum())
    b = int(pair["winner_name"].eq(player_b).sum())
    return a, b


def _num(value) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _safe_mean(series: pd.Series, default: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else default
