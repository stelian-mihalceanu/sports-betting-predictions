from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def team_strength(history: pd.DataFrame, team: str, window: int = 10) -> dict[str, float]:
    """Transparent attack/defence strength profile from recent completed matches."""
    h = history.copy(); h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date")
    recent = h.loc[h["home_team"].eq(team) | h["away_team"].eq(team)].tail(window)
    if recent.empty: return {"attack": 1.0, "defense": 1.0, "home_attack": 1.0, "away_attack": 1.0, "form": 0.5, "matches": 0.0}
    gf, ga, points, home_gf, away_gf = [], [], [], [], []
    for _, r in recent.iterrows():
        is_home = r["home_team"] == team
        f = float(r["home_goals"] if is_home else r["away_goals"]); a = float(r["away_goals"] if is_home else r["home_goals"])
        gf.append(f); ga.append(a); points.append(1.0 if f > a else 0.5 if f == a else 0.0)
        (home_gf if is_home else away_gf).append(f)
    weights = np.arange(1, len(gf) + 1)
    return {"attack": float(np.average(gf, weights=weights)), "defense": float(np.average(ga, weights=weights)), "home_attack": float(np.mean(home_gf)) if home_gf else float(np.mean(gf)), "away_attack": float(np.mean(away_gf)) if away_gf else float(np.mean(gf)), "form": float(np.mean(points)), "matches": float(len(recent))}


def h2h(history: pd.DataFrame, home: str, away: str, window: int = 10) -> dict[str, Any]:
    h = history.copy(); h["date"] = pd.to_datetime(h["date"], errors="coerce")
    pair = h[((h["home_team"].eq(home)) & (h["away_team"].eq(away))) | ((h["home_team"].eq(away)) & (h["away_team"].eq(home)))].dropna(subset=["date"]).sort_values("date").tail(window)
    rows = []
    for _, r in pair.iterrows():
        hg, ag = float(r["home_goals"]), float(r["away_goals"]); hs, aws = (hg, ag) if r["home_team"] == home else (ag, hg)
        rows.append({"date": r["date"], "home_goals": hs, "away_goals": aws, "result": "1" if hs > aws else "X" if hs == aws else "2"})
    return {"matches": len(rows), "home_wins": sum(x["result"] == "1" for x in rows), "draws": sum(x["result"] == "X" for x in rows), "away_wins": sum(x["result"] == "2" for x in rows), "rows": rows}


def goals_distribution(home_xg: float, away_xg: float, max_goals: int = 6) -> pd.DataFrame:
    rows = [{"home_goals": h, "away_goals": a, "probability": _poisson(home_xg, h) * _poisson(away_xg, a)} for h in range(max_goals + 1) for a in range(max_goals + 1)]
    return pd.DataFrame(rows).sort_values("probability", ascending=False).reset_index(drop=True)


def ensemble_prediction(base: dict[str, Any], home_strength: dict[str, float], away_strength: dict[str, float], h2h_data: dict[str, Any] | None = None, elo_home: float = 1500.0, elo_away: float = 1500.0) -> dict[str, Any]:
    """Blend Poisson, explicit Elo, recent form and a market-style structural prior."""
    poisson = np.array([float(base["ft_home"]), float(base["ft_draw"]), float(base["ft_away"])])
    elo_edge = math.tanh(((elo_home + 55.0) - elo_away) / 350.0)
    elo = _softmax3(0.333 + 0.30 * elo_edge, 0.333 - 0.08 * abs(elo_edge), 0.333 - 0.22 * elo_edge)
    form_edge = home_strength.get("form", .5) - away_strength.get("form", .5)
    form = _softmax3(0.333 + 0.36 * form_edge, 0.333 - 0.18 * form_edge, 0.333 - 0.18 * form_edge)
    attack_edge = (home_strength.get("attack", 1.0) - away_strength.get("attack", 1.0)) / 2.0
    market_prior = _softmax3(0.38 + 0.22 * attack_edge, 0.28, 0.34 - 0.22 * attack_edge)
    if h2h_data and h2h_data.get("matches", 0) >= 3:
        total = h2h_data["matches"]
        hh = np.array([h2h_data["home_wins"] / total, h2h_data["draws"] / total, h2h_data["away_wins"] / total])
        ensemble = 0.44 * poisson + 0.22 * elo + 0.16 * form + 0.08 * market_prior + 0.10 * hh
    else:
        ensemble = 0.48 * poisson + 0.24 * elo + 0.18 * form + 0.10 * market_prior
    ensemble = ensemble / ensemble.sum()
    confidence = float(0.90 * ensemble.max() + 0.10 / 3.0)
    return {"ft_home": float(ensemble[0]), "ft_draw": float(ensemble[1]), "ft_away": float(ensemble[2]), "confidence": confidence, "components": {"poisson": poisson.tolist(), "elo": elo.tolist(), "form": form.tolist(), "market_prior": market_prior.tolist()}}


def calibration_shrink(probability: float, sample_size: float, strength: float = 80.0) -> float:
    weight = min(1.0, max(0.0, sample_size / strength))
    return float(0.5 + weight * (probability - 0.5))


def implied_probability(decimal_odds: float) -> float:
    return 1.0 / decimal_odds if decimal_odds and decimal_odds > 1.0 else float("nan")


def value_edge(model_probability: float, decimal_odds: float, kelly_fraction: float = 0.25) -> dict[str, float]:
    """Compare the model probability with the market-implied probability.

    ``kelly_fraction`` applies a fractional Kelly (default quarter-Kelly) on
    top of the full Kelly stake, since full Kelly is too aggressive for model
    estimates that carry real uncertainty. The suggested stake is clipped to
    0 when the model has no edge, so it never recommends betting into a
    negative-EV price.
    """
    implied = implied_probability(decimal_odds)
    edge = round(model_probability - implied, 6)
    fair_odds = 1.0 / model_probability if model_probability > 0 else float("inf")
    ev = round(model_probability * decimal_odds - 1.0, 6)
    b = decimal_odds - 1.0
    full_kelly = ((b * model_probability) - (1.0 - model_probability)) / b if b > 0 else 0.0
    suggested_stake = round(max(0.0, full_kelly) * kelly_fraction, 6)
    return {
        "implied_probability": round(implied, 6),
        "edge": edge,
        "fair_odds": round(fair_odds, 6) if math.isfinite(fair_odds) else fair_odds,
        "expected_value": ev,
        "kelly_stake_pct": suggested_stake,
    }


def _softmax3(a: float, b: float, c: float) -> np.ndarray:
    x = np.array([a, b, c], dtype=float); x -= np.max(x); e = np.exp(x); return e / e.sum()


def _poisson(lam: float, k: int) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)
