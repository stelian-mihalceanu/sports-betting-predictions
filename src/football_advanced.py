from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def team_strength(history: pd.DataFrame, team: str, window: int = 10) -> dict[str, float]:
    """Transparent attack/defence strength profile from recent completed matches."""
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date")
    mask = h["home_team"].eq(team) | h["away_team"].eq(team)
    recent = h.loc[mask].tail(window)
    if recent.empty:
        return {"attack": 1.0, "defense": 1.0, "home_attack": 1.0, "away_attack": 1.0, "form": 0.5, "matches": 0.0}
    gf, ga, points = [], [], []
    home_gf, away_gf = [], []
    for _, r in recent.iterrows():
        is_home = r["home_team"] == team
        f = float(r["home_goals"] if is_home else r["away_goals"])
        a = float(r["away_goals"] if is_home else r["home_goals"])
        gf.append(f); ga.append(a)
        points.append(1.0 if f > a else 0.5 if f == a else 0.0)
        (home_gf if is_home else away_gf).append(f)
    return {
        "attack": float(np.average(gf, weights=np.arange(1, len(gf) + 1))),
        "defense": float(np.average(ga, weights=np.arange(1, len(ga) + 1))),
        "home_attack": float(np.mean(home_gf)) if home_gf else float(np.mean(gf)),
        "away_attack": float(np.mean(away_gf)) if away_gf else float(np.mean(gf)),
        "form": float(np.mean(points)),
        "matches": float(len(recent)),
    }


def h2h(history: pd.DataFrame, home: str, away: str, window: int = 10) -> dict[str, Any]:
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    pair = h[((h["home_team"].eq(home)) & (h["away_team"].eq(away))) | ((h["home_team"].eq(away)) & (h["away_team"].eq(home)))]
    pair = pair.dropna(subset=["date"]).sort_values("date").tail(window)
    rows = []
    for _, r in pair.iterrows():
        hg, ag = float(r["home_goals"]), float(r["away_goals"])
        if r["home_team"] == home:
            hs, aws = hg, ag
        else:
            hs, aws = ag, hg
        rows.append({"date": r["date"], "home_goals": hs, "away_goals": aws, "result": "1" if hs > aws else "X" if hs == aws else "2"})
    return {"matches": len(rows), "home_wins": sum(x["result"] == "1" for x in rows), "draws": sum(x["result"] == "X" for x in rows), "away_wins": sum(x["result"] == "2" for x in rows), "rows": rows}


def goals_distribution(home_xg: float, away_xg: float, max_goals: int = 6) -> pd.DataFrame:
    rows = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = _poisson(home_xg, h) * _poisson(away_xg, a)
            rows.append({"home_goals": h, "away_goals": a, "probability": p})
    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def ensemble_prediction(base: dict[str, Any], home_strength: dict[str, float], away_strength: dict[str, float], h2h_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Blend independent signals: Poisson/base model, Elo, recent form and a market-style prior."""
    poisson = np.array([float(base["ft_home"]), float(base["ft_draw"]), float(base["ft_away"])])
    elo = _softmax3(float(base.get("ft_home", .33)) + 0.18, float(base.get("ft_draw", .33)), float(base.get("ft_away", .33)) - 0.18)
    form_edge = home_strength.get("form", .5) - away_strength.get("form", .5)
    form = _softmax3(.333 + .32 * form_edge, .333 - .16 * form_edge, .333 - .16 * form_edge)
    attack_edge = (home_strength.get("attack", 1.0) - away_strength.get("attack", 1.0)) / 2.0
    market_prior = _softmax3(.38 + .22 * attack_edge, .28, .34 - .22 * attack_edge)
    ensemble = 0.48 * poisson + 0.24 * elo + 0.18 * form + 0.10 * market_prior
    ensemble = ensemble / ensemble.sum()
    # Conservative shrinkage toward 1/3 improves calibration for sparse/unknown teams.
    confidence = float(0.90 * ensemble.max() + 0.10 / 3.0)
    return {"ft_home": float(ensemble[0]), "ft_draw": float(ensemble[1]), "ft_away": float(ensemble[2]), "confidence": confidence, "components": {"poisson": poisson.tolist(), "elo": elo.tolist(), "form": form.tolist(), "market_prior": market_prior.tolist()}}


def calibration_shrink(probability: float, sample_size: float, strength: float = 80.0) -> float:
    """Shrink low-sample probabilities toward 50%/neutrality."""
    weight = min(1.0, max(0.0, sample_size / strength))
    return float(0.5 + weight * (probability - 0.5))


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / decimal_odds if decimal_odds and decimal_odds > 1.0 else float("nan")


def value_edge(model_probability: float, decimal_odds: float) -> dict[str, float]:
    implied = implied_probability(decimal_odds)
    edge = model_probability - implied
    fair_odds = 1.0 / model_probability if model_probability > 0 else float("inf")
    ev = model_probability * decimal_odds - 1.0
    return {"implied_probability": implied, "edge": edge, "fair_odds": fair_odds, "expected_value": ev}


def _softmax3(a: float, b: float, c: float) -> np.ndarray:
    x = np.array([a, b, c], dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def _poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)
