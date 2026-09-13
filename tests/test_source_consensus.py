import math

from src.source_consensus import brier_score, calibration_bins, multiclass_log_loss, source_consensus


def test_consensus_normalizes_and_weights_sources():
    result = source_consensus(
        {"1": 0.60, "X": 0.20, "2": 0.20},
        {"Bet Better": {"1": 55, "X": 25, "2": 20}},
        weights={"Internal": 0.5, "Bet Better": 0.5},
    )
    assert math.isclose(sum(result["probabilities"].values()), 1.0)
    assert result["probabilities"]["1"] > result["probabilities"]["2"]
    assert result["source_count"] == 2


def test_metrics_and_calibration_bins():
    probs = [
        {"1": 0.7, "X": 0.2, "2": 0.1},
        {"1": 0.4, "X": 0.3, "2": 0.3},
    ]
    outcomes = ["1", "2"]
    assert multiclass_log_loss(probs, outcomes) > 0
    assert brier_score(probs, outcomes) >= 0
    bins = calibration_bins(probs, outcomes, bins=5)
    assert bins
    assert all("predicted" in row and "observed" in row for row in bins)
