from __future__ import annotations

import math
from typing import Mapping


def predict_basketball(home: Mapping[str, float], away: Mapping[str, float]) -> dict[str, float | str]:
    """Transparent baseline for basketball moneyline/spread/total markets.

    Inputs may contain net_rating, pace, home_rating, rest_days and elo.
    Missing values fall back to league-neutral baselines.
    """
    h_elo = float(home.get("elo", 1500.0))
    a_elo = float(away.get("elo", 1500.0))
    h_net = float(home.get("net_rating", 0.0))
    a_net = float(away.get("net_rating", 0.0))
    h_pace = float(home.get("pace", 99.0))
    a_pace = float(away.get("pace", 99.0))
    rest_edge = max(-2.0, min(2.0, float(home.get("rest_days", 1.0)) - float(away.get("rest_days", 1.0))))
    rating_edge = 0.055 * (h_elo - a_elo) + 2.2 * (h_net - a_net) + 1.5 * rest_edge + 55.0
    p_home = 1.0 / (1.0 + math.exp(-rating_edge / 100.0))
    expected_total = max(180.0, min(260.0, 0.5 * (h_pace + a_pace) * 1.02 + 0.35 * (h_net + a_net)))
    expected_margin = max(-25.0, min(25.0, 14.0 * (p_home - 0.5)))
    return {
        "home_win": p_home,
        "away_win": 1.0 - p_home,
        "expected_margin": expected_margin,
        "expected_total": expected_total,
        "pick": "Home" if p_home >= 0.5 else "Away",
        "confidence": max(p_home, 1.0 - p_home),
    }
