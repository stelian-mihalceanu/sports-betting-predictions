import numpy as np

from src.basketball_predictions import predict_basketball
from src.calibration import binary_calibration, calibration_label


def test_calibration_metrics_are_finite():
    result = binary_calibration([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert np.isfinite(result["brier"])
    assert np.isfinite(result["log_loss"])
    assert calibration_label(result["brier"]) == "Strong"


def test_basketball_baseline_returns_valid_probability():
    result = predict_basketball(
        {"elo": 1540, "net_rating": 4.0, "pace": 100, "rest_days": 2},
        {"elo": 1490, "net_rating": 1.0, "pace": 98, "rest_days": 1},
    )
    assert 0 < result["home_win"] < 1
    assert abs(result["home_win"] + result["away_win"] - 1) < 1e-9
    assert result["pick"] == "Home"
