from __future__ import annotations

from typing import Any

import math
import numpy as np


def normalize_three_way(home: float, draw: float, away: float) -> dict[str, float]:
    values = np.array([float(home), float(draw), float(away)], dtype=float)
    if values.max() > 1.0:
        values /= 100.0
    values = np.clip(values, 0.0, None)
    total = values.sum()
    if total <= 0:
        return {"1": 1 / 3, "X": 1 / 3, "2": 1 / 3}
    values /= total
    return {"1": float(values[0]), "X": float(values[1]), "2": float(values[2])}


def source_consensus(
    internal: dict[str, float],
    external_sources: dict[str, dict[str, float] | None] | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Weighted probability consensus with source coverage and disagreement metrics."""
    sources: dict[str, dict[str, float]] = {"Internal": normalize_three_way(internal["1"], internal["X"], internal["2"])}
    for name, probs in (external_sources or {}).items():
        if probs:
            sources[name] = normalize_three_way(probs["1"], probs["X"], probs["2"])

    default_weights = {"Internal": 0.55, "Bet Better": 0.20, "Sportmonks": 0.25}
    weights = weights or default_weights
    active = {name: weight for name, weight in weights.items() if name in sources and weight > 0}
    if not active:
        active = {name: 1.0 for name in sources}
    total_weight = sum(active.values())
    active = {name: weight / total_weight for name, weight in active.items()}

    consensus = {k: 0.0 for k in ("1", "X", "2")}
    for name, weight in active.items():
        for outcome in consensus:
            consensus[outcome] += weight * sources[name][outcome]

    entropy = -sum(p * math.log(max(p, 1e-12), 2) for p in consensus.values())
    max_disagreement = 0.0
    for name, probs in sources.items():
        distance = sum(abs(probs[k] - consensus[k]) for k in consensus) / 2.0
        max_disagreement = max(max_disagreement, distance)

    return {
        "probabilities": consensus,
        "sources": sources,
        "weights": active,
        "source_count": len(sources),
        "max_disagreement": float(max_disagreement),
        "entropy": float(entropy),
        "agreement": float(max(0.0, 1.0 - max_disagreement)),
    }


def multiclass_log_loss(probabilities: list[dict[str, float]], outcomes: list[str]) -> float:
    if not probabilities or len(probabilities) != len(outcomes):
        return float("nan")
    losses = []
    for probs, outcome in zip(probabilities, outcomes):
        p = max(min(float(probs.get(outcome, 0.0)), 1.0), 1e-15)
        losses.append(-math.log(p))
    return float(np.mean(losses))


def brier_score(probabilities: list[dict[str, float]], outcomes: list[str]) -> float:
    if not probabilities or len(probabilities) != len(outcomes):
        return float("nan")
    losses = []
    for probs, outcome in zip(probabilities, outcomes):
        losses.append(sum((float(probs.get(k, 0.0)) - (1.0 if outcome == k else 0.0)) ** 2 for k in ("1", "X", "2")))
    return float(np.mean(losses))


def calibration_bins(probabilities: list[dict[str, float]], outcomes: list[str], bins: int = 10) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for outcome in ("1", "X", "2"):
        for idx in range(bins):
            lo, hi = idx / bins, (idx + 1) / bins
            selected = []
            for probs, actual in zip(probabilities, outcomes):
                p = float(probs.get(outcome, 0.0))
                if (lo <= p < hi) or (idx == bins - 1 and p <= hi):
                    selected.append((p, 1.0 if actual == outcome else 0.0))
            if selected:
                rows.append({"outcome": outcome, "bin_low": lo, "bin_high": hi, "predicted": float(np.mean([x[0] for x in selected])), "observed": float(np.mean([x[1] for x in selected])), "samples": float(len(selected))})
    return rows
