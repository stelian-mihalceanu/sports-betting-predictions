from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def build_team_stats(history: pd.DataFrame, window: int = 12) -> dict[str, dict[str, float]]:
    """Build recency-weighted team rates with home/away context and fallbacks."""
    if history.empty:
        return {}
    rows = []
    for _, match in history.sort_values("date").iterrows():
        home, away = str(match.get("home_team", "")).strip(), str(match.get("away_team", "")).strip()
        hg, ag = _number(match.get("home_goals")), _number(match.get("away_goals"))
        if not home or not away or hg is None or ag is None:
            continue
        rows.append(_team_row(match, home, hg, ag, "home"))
        rows.append(_team_row(match, away, ag, hg, "away"))
    if not rows:
        return {}

    frame = pd.DataFrame(rows).sort_values("date")
    numeric = [c for c in frame.columns if c not in {"date", "team", "venue"}]
    global_means = frame[numeric].mean(numeric_only=True).to_dict()
    result: dict[str, dict[str, float]] = {}
    for team, group in frame.groupby("team", sort=False):
        recent = group.tail(window)
        stats = {k: _weighted_mean(recent, k, global_means.get(k, 0.0)) for k in numeric}
        home = group[group["venue"] == "home"].tail(max(6, window // 2))
        away = group[group["venue"] == "away"].tail(max(6, window // 2))
        stats.update({
            "home_goals_for": _weighted_mean(home, "goals_for", stats["goals_for"]),
            "home_goals_against": _weighted_mean(home, "goals_against", stats["goals_against"]),
            "away_goals_for": _weighted_mean(away, "goals_for", stats["goals_for"]),
            "away_goals_against": _weighted_mean(away, "goals_against", stats["goals_against"]),
            "matches": float(len(group)),
            "recent_form": _recent_points(group, 5),
        })
        result[team] = stats
    result["__global__"] = {k: _weighted_mean(frame.tail(max(window * 4, 40)), k, global_means.get(k, 0.0)) for k in numeric}
    result["__global__"]["matches"] = float(len(frame) / 2)
    return result


def _team_row(match, team: str, goals_for: float, goals_against: float, side: str) -> dict[str, object]:
    return {
        "date": match.get("date"), "team": team, "venue": side,
        "goals_for": goals_for, "goals_against": goals_against,
        "ht_goals_for": _number(match.get("HTHG" if side == "home" else "HTAG")),
        "ht_goals_against": _number(match.get("HTAG" if side == "home" else "HTHG")),
        "corners_for": _number(match.get("HC" if side == "home" else "AC")),
        "corners_against": _number(match.get("AC" if side == "home" else "HC")),
        "yellow_for": _number(match.get("HY" if side == "home" else "AY")),
        "yellow_against": _number(match.get("AY" if side == "home" else "HY")),
        "red_for": _number(match.get("HR" if side == "home" else "AR")),
        "red_against": _number(match.get("AR" if side == "home" else "HR")),
    }


def predict_match(home: str, away: str, elo_state: Mapping[str, Mapping[str, float]], team_stats: Mapping[str, Mapping[str, float]]) -> dict[str, float | str]:
    """Return a richer probabilistic profile for football markets."""
    global_stats = team_stats.get("__global__", {})
    hs, aws = team_stats.get(home, global_stats), team_stats.get(away, global_stats)
    home_elo = _finite_or_default(elo_state.get(home, {}).get("elo"), 1500.0)
    away_elo = _finite_or_default(elo_state.get(away, {}).get("elo"), 1500.0)
    elo_edge = math.tanh((home_elo + 55.0 - away_elo) / 350.0)

    home_attack = _venue_rate(hs, "home_goals_for", "goals_for", 1.35)
    home_def = _venue_rate(hs, "home_goals_against", "goals_against", 1.15)
    away_attack = _venue_rate(aws, "away_goals_for", "goals_for", 1.10)
    away_def = _venue_rate(aws, "away_goals_against", "goals_against", 1.35)
    home_xg = (0.48 * home_attack + 0.52 * away_def) * (1.0 + 0.10 * elo_edge)
    away_xg = (0.48 * away_attack + 0.52 * home_def) * (1.0 - 0.08 * elo_edge)
    home_xg, away_xg = max(0.15, min(3.9, home_xg)), max(0.15, min(3.6, away_xg))

    ht_home_xg = max(0.05, min(2.0, 0.62 * _finite_or_default(hs.get("ht_goals_for"), home_xg * 0.44) + 0.38 * home_xg * 0.44))
    ht_away_xg = max(0.05, min(2.0, 0.62 * _finite_or_default(aws.get("ht_goals_for"), away_xg * 0.44) + 0.38 * away_xg * 0.44))
    ft_home, ft_draw, ft_away = poisson_1x2(home_xg, away_xg)
    ht_home, ht_draw, ht_away = poisson_1x2(ht_home_xg, ht_away_xg)

    total_goals = home_xg + away_xg
    second_half_goals = max(0.05, total_goals - ht_home_xg - ht_away_xg)
    corners_home = _venue_rate(hs, "corners_for", "corners_for", 5.0)
    corners_away = _venue_rate(aws, "corners_for", "corners_for", 4.2)
    expected_corners = max(2.0, min(16.0, 0.55 * (corners_home + corners_away) + 0.45 * _global(global_stats, "corners_for", 4.6) * 2))
    expected_cards = max(1.0, min(10.0, _card_rate(hs) + _card_rate(aws)))
    score_home, score_away = most_likely_score(home_xg, away_xg)
    btts = btts_probability(home_xg, away_xg)
    ht_total = ht_home_xg + ht_away_xg
    second_btts = btts_probability(max(0.03, home_xg - ht_home_xg), max(0.03, away_xg - ht_away_xg))

    return {
        "home_xg": home_xg, "away_xg": away_xg,
        "likely_home_goals": score_home, "likely_away_goals": score_away,
        "ft_home": ft_home, "ft_draw": ft_draw, "ft_away": ft_away,
        "ht_home": ht_home, "ht_draw": ht_draw, "ht_away": ht_away,
        "expected_corners": expected_corners, "expected_cards": expected_cards,
        "over_1_5": poisson_over(total_goals, 1.5), "under_1_5": 1 - poisson_over(total_goals, 1.5),
        "over_2_5": poisson_over(total_goals, 2.5), "under_2_5": 1 - poisson_over(total_goals, 2.5),
        "over_3_5": poisson_over(total_goals, 3.5), "under_3_5": 1 - poisson_over(total_goals, 3.5),
        "btts": btts, "btts_no": 1 - btts,
        "ht_over_0_5": poisson_over(ht_total, 0.5), "ht_over_1_5": poisson_over(ht_total, 1.5),
        "second_half_over_0_5": poisson_over(second_half_goals, 0.5),
        "second_half_over_1_5": poisson_over(second_half_goals, 1.5),
        "ht_btts": btts_probability(ht_home_xg, ht_away_xg), "second_half_btts": second_btts,
        "over_7_5_corners": poisson_over(expected_corners, 7.5),
        "over_8_5_corners": poisson_over(expected_corners, 8.5),
        "over_9_5_corners": poisson_over(expected_corners, 9.5),
        "over_10_5_corners": poisson_over(expected_corners, 10.5),
        "over_2_5_cards": poisson_over(expected_cards, 2.5),
        "over_3_5_cards": poisson_over(expected_cards, 3.5),
        "over_4_5_cards": poisson_over(expected_cards, 4.5),
        "ft_pick": _pick((ft_home, "1"), (ft_draw, "X"), (ft_away, "2")),
        "ht_pick": _pick((ht_home, "1"), (ht_draw, "X"), (ht_away, "2")),
        "double_1x": ft_home + ft_draw, "double_x2": ft_draw + ft_away, "double_12": ft_home + ft_away,
        "confidence": max(ft_home, ft_draw, ft_away),
    }


def most_likely_score(home_lambda: float, away_lambda: float, max_goals: int = 7) -> tuple[int, int]:
    best = (0, 0, -1.0)
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = _poisson(home_lambda, h) * _poisson(away_lambda, a)
            if p > best[2]: best = (h, a, p)
    return best[0], best[1]


def poisson_1x2(home_lambda: float, away_lambda: float, max_goals: int = 10) -> tuple[float, float, float]:
    hp = [_poisson(home_lambda, k) for k in range(max_goals + 1)]
    ap = [_poisson(away_lambda, k) for k in range(max_goals + 1)]
    home = sum(hp[i] * ap[j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i > j)
    draw = sum(hp[i] * ap[i] for i in range(max_goals + 1))
    away = sum(hp[i] * ap[j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i < j)
    total = home + draw + away
    return home / total, draw / total, away / total


def poisson_over(lam: float, line: float) -> float:
    threshold = int(math.floor(line))
    return 1.0 - sum(_poisson(lam, k) for k in range(threshold + 1))


def btts_probability(home_lambda: float, away_lambda: float) -> float:
    return 1.0 - math.exp(-home_lambda) - math.exp(-away_lambda) + math.exp(-(home_lambda + away_lambda))


def _venue_rate(stats: Mapping[str, float], venue_key: str, general_key: str, fallback: float) -> float:
    return 0.65 * _finite_or_default(stats.get(venue_key), fallback) + 0.35 * _finite_or_default(stats.get(general_key), fallback)


def _weighted_mean(group: pd.DataFrame, key: str, fallback: float) -> float:
    if group.empty or key not in group.columns: return float(fallback)
    values = pd.to_numeric(group[key], errors="coerce")
    mask = values.notna()
    if not mask.any(): return float(fallback)
    vals = values[mask].to_numpy(dtype=float)
    weights = np.exp(np.linspace(-1.4, 0.0, len(vals)))
    return float(np.average(vals, weights=weights))


def _recent_points(group: pd.DataFrame, n: int) -> float:
    recent = group.tail(n)
    if recent.empty: return 0.0
    return float(np.mean([3.0 if float(r.goals_for) > float(r.goals_against) else 1.0 if float(r.goals_for) == float(r.goals_against) else 0.0 for _, r in recent.iterrows()]))


def _card_rate(stats: Mapping[str, float]) -> float:
    return _finite_or_default(stats.get("yellow_for"), 1.8) + 2.0 * _finite_or_default(stats.get("red_for"), 0.0)


def _poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _pick(*options: tuple[float, str]) -> str:
    return max(options, key=lambda item: item[0])[1]


def _number(value) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _finite_or_default(value, default: float) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _global(stats: Mapping[str, float], key: str, default: float) -> float:
    return _finite_or_default(stats.get(key), default)
