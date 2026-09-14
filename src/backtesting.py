from __future__ import annotations

import math
import pandas as pd

from src.predictions import build_team_stats, predict_match


def football_backtest(history: pd.DataFrame, max_matches: int = 500) -> dict[str, float | int]:
    data = history.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    if len(data) < 30:
        return {"matches": int(len(data)), "accuracy": float("nan"), "brier": float("nan"), "log_loss": float("nan")}
    data = data.tail(max_matches).reset_index(drop=True)
    ratings: dict[str, dict[str, float]] = {}
    outcomes: list[int] = []
    probs: list[tuple[float, float, float]] = []
    prior = data.iloc[:1]
    history_rows: list[pd.Series] = []
    for _, row in data.iterrows():
        hist = pd.DataFrame(history_rows)
        stats = build_team_stats(hist) if not hist.empty else {}
        prediction = predict_match(row.home_team, row.away_team, ratings, stats)
        probs.append((float(prediction["ft_home"]), float(prediction["ft_draw"]), float(prediction["ft_away"])))
        hg, ag = float(row.home_goals), float(row.away_goals)
        outcomes.append(0 if hg > ag else 1 if hg == ag else 2)
        home = str(row.home_team); away = str(row.away_team)
        rh = ratings.get(home, {}).get("elo", 1500.0); ra = ratings.get(away, {}).get("elo", 1500.0)
        expected = 1.0 / (1.0 + 10.0 ** ((ra - (rh + 60.0)) / 400.0))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        change = 24.0 * (actual - expected)
        ratings[home] = {"elo": rh + change}; ratings[away] = {"elo": ra - change}
        history_rows.append(row)
    correct = sum(max(range(3), key=lambda j: probs[i][j]) == outcomes[i] for i in range(len(outcomes)))
    brier = sum((probs[i][j] - (1.0 if outcomes[i] == j else 0.0)) ** 2 for i in range(len(outcomes)) for j in range(3)) / len(outcomes)
    ll = -sum(math.log(max(1e-9, probs[i][outcomes[i]])) for i in range(len(outcomes))) / len(outcomes)
    return {"matches": len(outcomes), "accuracy": correct / len(outcomes), "brier": brier, "log_loss": ll}
