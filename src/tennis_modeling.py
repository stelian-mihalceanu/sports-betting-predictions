from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def player_snapshot(matches: pd.DataFrame, player: str, surface: str = "All", window: int = 20) -> dict[str, Any]:
    frame = _prepare(matches, surface)
    mask = frame["winner_name"].eq(player) | frame["loser_name"].eq(player)
    recent = frame.loc[mask].tail(window)
    wins = int(recent["winner_name"].eq(player).sum())
    total = len(recent)
    return {
        "matches": total,
        "wins": wins,
        "win_rate": wins / total if total else 0.5,
        "serve_points_won": _player_stat(recent, player, "serve_points_won", 0.62),
        "return_points_won": _player_stat(recent, player, "return_points_won", 0.38),
        "aces": _player_stat(recent, player, "aces", 4.0),
        "double_faults": _player_stat(recent, player, "double_faults", 2.0),
        "recent_form": _form_score(recent, player),
    }


def build_tennis_match_features(matches: pd.DataFrame, player_a: str, player_b: str, surface: str = "All") -> dict[str, float]:
    a, b = player_snapshot(matches, player_a, surface), player_snapshot(matches, player_b, surface)
    h2h_a, h2h_b = _h2h(matches, player_a, player_b)
    return {
        "elo_diff": _elo(matches, player_a) - _elo(matches, player_b),
        "surface_elo_diff": _surface_elo(matches, player_a, surface) - _surface_elo(matches, player_b, surface),
        "form_diff": a["win_rate"] - b["win_rate"],
        "serve_diff": a["serve_points_won"] - b["serve_points_won"],
        "return_diff": a["return_points_won"] - b["return_points_won"],
        "ace_diff": a["aces"] - b["aces"],
        "df_diff": b["double_faults"] - a["double_faults"],
        "h2h_diff": float(h2h_a - h2h_b),
        "h2h_total": float(h2h_a + h2h_b),
        "experience_diff": float(a["matches"] - b["matches"]),
    }


def _prepare(matches: pd.DataFrame, surface: str) -> pd.DataFrame:
    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["tourney_date"], format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["date", "winner_name", "loser_name"]).sort_values("date")
    if surface != "All" and "surface" in frame.columns:
        frame = frame[frame["surface"].astype(str).eq(surface)]
    return frame


def _elo(matches: pd.DataFrame, player: str) -> float:
    frame = _prepare(matches, "All")
    ratings: dict[str, float] = {}
    for _, m in frame.iterrows():
        w, l = str(m["winner_name"]), str(m["loser_name"])
        rw, rl = ratings.get(w, 1500.0), ratings.get(l, 1500.0)
        expected = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
        ratings[w] = rw + 24.0 * (1.0 - expected)
        ratings[l] = rl - 24.0 * (1.0 - expected)
    return ratings.get(player, 1500.0)


def _surface_elo(matches: pd.DataFrame, player: str, surface: str) -> float:
    if surface == "All":
        return _elo(matches, player)
    frame = _prepare(matches, surface)
    ratings: dict[str, float] = {}
    for _, m in frame.iterrows():
        w, l = str(m["winner_name"]), str(m["loser_name"])
        rw, rl = ratings.get(w, 1500.0), ratings.get(l, 1500.0)
        expected = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
        ratings[w] = rw + 24.0 * (1.0 - expected)
        ratings[l] = rl - 24.0 * (1.0 - expected)
    return ratings.get(player, 1500.0)


def _player_stat(frame: pd.DataFrame, player: str, stat: str, default: float) -> float:
    values = []
    for _, m in frame.iterrows():
        won = str(m["winner_name"]) == player
        prefix = "w_" if won else "l_"
        if stat == "serve_points_won":
            svpt, first, second = _num(m.get(prefix + "svpt")), _num(m.get(prefix + "1stWon")), _num(m.get(prefix + "2ndWon"))
            value = (first + second) / svpt if svpt and svpt > 0 else np.nan
        elif stat == "return_points_won":
            opp = "l_" if won else "w_"
            svpt, first, second = _num(m.get(opp + "svpt")), _num(m.get(opp + "1stWon")), _num(m.get(opp + "2ndWon"))
            value = 1.0 - ((first + second) / svpt) if svpt and svpt > 0 else np.nan
        else:
            value = _num(m.get(prefix + ("ace" if stat == "aces" else "df")))
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else default


def _form_score(frame: pd.DataFrame, player: str) -> float:
    recent = frame.tail(5)
    if recent.empty:
        return 0.5
    points = [1.0 if str(m["winner_name"]) == player else 0.0 for _, m in recent.iterrows()]
    return float(np.average(points, weights=np.arange(1, len(points) + 1)))


def _h2h(matches: pd.DataFrame, a: str, b: str) -> tuple[int, int]:
    frame = _prepare(matches, "All")
    pair = frame[((frame["winner_name"].eq(a)) & (frame["loser_name"].eq(b))) | ((frame["winner_name"].eq(b)) & (frame["loser_name"].eq(a)))]
    return int(pair["winner_name"].eq(a).sum()), int(pair["winner_name"].eq(b).sum())


def _num(value: Any) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan
