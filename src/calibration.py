from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


def binary_calibration(y_true, probabilities, bins: int = 10) -> dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) == 0:
        return {"brier": float("nan"), "log_loss": float("nan"), "fraction_positive": [], "mean_predicted": []}
    frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]))),
        "fraction_positive": frac.tolist(),
        "mean_predicted": mean.tolist(),
    }


def calibration_label(brier: float) -> str:
    if not np.isfinite(brier):
        return "Insufficient sample"
    if brier < 0.18:
        return "Strong"
    if brier < 0.22:
        return "Good"
    if brier < 0.26:
        return "Fair"
    return "Needs work"
