from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def build_team_stats(history: pd.DataFrame, window: int = 12) -> dict[str, dict[str, float]]:
    """Build recent team rates used for pre-match market estimates."""
    if history.empty:
        return {}
    rows = []
    for _, match in history.sort_values("date").iterrows():
        home = str(match.get("home_team", "")).strip()
        away = str(match.get("away_team", "")).strip()
        if not home or not away:
            continue
        hg = _number(match.get("home_goals"))
        ag = _number(match.get("away_goals"))
        if hg is None or ag is None:
            continue
        rows.append(_team_row(match, home, hg, ag, "home"))
        rows.append(_team_row(match, away, ag, hg, "away"))

    if not rows:
        return {}
    frame = pd.DataFrame(rows).sort_values("date")
    numeric = [c for c in frame.columns if c not in {"date", "team"}]
    global_means = frame[numeric].mean(numeric_only=True).to_dict()
    result: dict[str, dict[str, float]] = {}
    for team, group in frame.groupby("team", sort=False):
        recent = group.tail(window)
        result[team] = {
            key: _finite_or_default(recent[key].mean(), global_means.get(key, 0.0))
            for key in numeric
        }
    result["__global__"] = {key: _finite_or_default(value, 0.0) for key, value in global_means.items()}
    return result


def _team_row(match, team: str, goals_for: float, goals_against: float, side: str) -> dict[str, object]:
    return {
        "date": match.get("date"),
        "team": team,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "ht_goals_for": _number(match.get("HTHG" if side == "home" else "HTAG")),
        "ht_goals_against": _number(match.get("HTAG" if side == "home" else "HTHG")),
        "corners_for": _number(match.get("HC" if side == "home" else "AC")),
        "corners_against": _number(match.get("AC" if side == "home" else "HC")),
        "yellow_for": _number(match.get("HY" if side == "home" else "AY")),
        "yellow_against": _number(match.get("AY" if side == "home" else "HY")),
        "red_for": _number(match.get("HR" if side == "home" else "AR")),
        "red_against": _number(match.get("AR" if side == "home" else "HR")),
    }


def predict_match(
    home: str,
    away: str,
    elo_state: Mapping[str, Mapping[str, float]],
    team_stats: Mapping[str, Mapping[str, float]],
) -> dict[str, float | str]:
    """Return probabilistic pre-match estimates for common football markets."""
    global_stats = team_stats.get("__global__", {})
    hs = team_stats.get(home, global_stats)
    aws = team_stats.get(away, global_stats)
    home_elo = _finite_or_default(elo_state.get(home, {}).get("elo"), 1500.0)
    away_elo = _finite_or_default(elo_state.get(away, {}).get("elo"), 1500.0)

    home_xg = _blend(hs, "goals_for", aws, "goals_against", _global(global_stats, "goals_for", 1.35))
    away_xg = _blend(aws, "goals_for", hs, "goals_against", _global(global_stats, "goals_for", 1.15))
    elo_factor = max(-0.15, min(0.15, (home_elo - away_elo) / 2000.0))
    home_xg = max(0.15, min(3.8, home_xg * (1.05 + elo_factor)))
    away_xg = max(0.15, min(3.5, away_xg * (0.98 - elo_factor * 0.65)))

    ht_home_xg = max(0.05, home_xg * 0.44)
    ht_away_xg = max(0.05, away_xg * 0.44)
    ft_home, ft_draw, ft_away = poisson_1x2(home_xg, away_xg)
    ht_home, ht_draw, ht_away = poisson_1x2(ht_home_xg, ht_away_xg)

    corners_home = _blend(hs, "corners_for", aws, "corners_against", _global(global_stats, "corners_for", 5.0))
    corners_away = _blend(aws, "corners_for", hs, "corners_against", _global(global_stats, "corners_for", 4.2))
    expected_corners = max(2.0, min(15.0, corners_home + corners_away))

    expected_cards = max(1.0, min(10.0, _card_rate(hs) + _card_rate(aws)))
    total_goals = home_xg + away_xg
    score_home, score_away = most_likely_score(home_xg, away_xg)

    return {
        "home_xg": home_xg,
        "away_xg": away_xg,
        "likely_home_goals": score_home,
        "likely_away_goals": score_away,
        "ft_home": ft_home,
        "ft_draw": ft_draw,
        "ft_away": ft_away,
        "ht_home": ht_home,
        "ht_draw": ht_draw,
        "ht_away": ht_away,
        "expected_corners": expected_corners,
        "expected_cards": expected_cards,
        "over_1_5": poisson_over(total_goals, 1.5),
        "over_2_5": poisson_over(total_goals, 2.5),
        "btts": btts_probability(home_xg, away_xg),
        "over_8_5_corners": poisson_over(expected_corners, 8.5),
        "over_3_5_cards": poisson_over(expected_cards, 3.5),
        "ft_pick": _pick((ft_home, "1"), (ft_draw, "X"), (ft_away, "2")),
        "ht_pick": _pick((ht_home, "1"), (ht_draw, "X"), (ht_away, "2")),
    }


def most_likely_score(home_lambda: float, away_lambda: float, max_goals: int = 6) -> tuple[int, int]:
    best = (0, 0, -1.0)
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            probability = _poisson(home_lambda, home_goals) * _poisson(away_lambda, away_goals)
            if probability > best[2]:
                best = (home_goals, away_goals, probability)
    return best[0], best[1]


def _blend(first: Mapping[str, float], first_key: str, second: Mapping[str, float], second_key: str, fallback: float) -> float:
    values = [first.get(first_key), second.get(second_key)]
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else fallback


def _card_rate(stats: Mapping[str, float]) -> float:
    yellow = stats.get("yellow_for")
    red = stats.get("red_for")
    if yellow is None or not math.isfinite(float(yellow)):
        return 1.8
    return float(yellow) + 2.0 * (float(red) if red is not None and math.isfinite(float(red)) else 0.0)


def poisson_1x2(home_lambda: float, away_lambda: float, max_goals: int = 8) -> tuple[float, float, float]:
    home_probs = [_poisson(home_lambda, k) for k in range(max_goals + 1)]
    away_probs = [_poisson(away_lambda, k) for k in range(max_goals + 1)]
    home = sum(home_probs[i] * away_probs[j] for i in range(max_goals + 1) for j in range(max_goals + 1) if i > j)
    draw = sum(home_probs[i] * away_probs[i] for i in range(max_goals + 1))
    total = home + draw
    return home / total, draw / total, max(0.0, 1.0 - home / total - draw / total)


def poisson_over(lam: float, line: float) -> float:
    threshold = int(math.floor(line))
    return 1.0 - sum(_poisson(lam, k) for k in range(threshold + 1))


def btts_probability(home_lambda: float, away_lambda: float) -> float:
    return 1.0 - math.exp(-home_lambda) - math.exp(-away_lambda) + math.exp(-(home_lambda + away_lambda))


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
