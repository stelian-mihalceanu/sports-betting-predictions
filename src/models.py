"""Machine learning model utilities for tennis and football predictions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def _validate_training_frame(df: pd.DataFrame, target_col: str) -> None:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    if len(df) < 10:
        raise ValueError("At least 10 rows are required for model training")
    if df[target_col].nunique(dropna=True) < 2:
        raise ValueError("Target must contain at least two classes")


def _split_and_clean(df: pd.DataFrame, target_col: str, feature_cols: Optional[List[str]], test_size: float):
    _validate_training_frame(df, target_col)
    cols = feature_cols or _get_numeric_feature_cols(df, exclude=[target_col])
    if not cols:
        raise ValueError("No numeric feature columns available")
    X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    return cols, X_train, X_test, y_train, y_test


def _evaluate(model, X_test, y_test) -> Dict[str, float]:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    if y_proba.shape[1] == 2:
        probabilities = y_proba[:, 1]
    else:
        probabilities = y_proba
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, probabilities)),
    }


def train_tennis_model(df: pd.DataFrame, target_col: str = "winner", feature_cols: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[XGBClassifier, List[str], Dict[str, float]]:
    """Train an XGBoost tennis outcome classifier using a temporal holdout."""
    cols, X_train, X_test, y_train, y_test = _split_and_clean(df, target_col, feature_cols, test_size)
    model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=random_state)
    model.fit(X_train, y_train)
    return model, cols, _evaluate(model, X_test, y_test)


def train_football_1x2_model(df: pd.DataFrame, target_col: str = "result", feature_cols: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[XGBClassifier, List[str], Dict[str, float]]:
    """Train an XGBoost 1X2 football outcome classifier."""
    cols, X_train, X_test, y_train, y_test = _split_and_clean(df, target_col, feature_cols, test_size)
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=random_state)
    model.fit(X_train, y_train)
    return model, cols, _evaluate(model, X_test, y_test)


def train_football_goals_model(df: pd.DataFrame, target_col: str = "total_goals", feature_cols: Optional[List[str]] = None, test_size: float = 0.2, random_state: int = 42) -> Tuple[XGBClassifier, List[str], Dict[str, float]]:
    """Train an XGBoost classifier for football goal-market outcomes."""
    cols, X_train, X_test, y_train, y_test = _split_and_clean(df, target_col, feature_cols, test_size)
    model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=random_state)
    model.fit(X_train, y_train)
    return model, cols, _evaluate(model, X_test, y_test)


def _get_numeric_feature_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Return numeric columns while excluding targets and obvious identifiers."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    ignored = set(exclude) | {"id", "match_id", "player_id", "team_id"}
    return [c for c in numeric_cols if c not in ignored]


def predict_match_outcome(model: XGBClassifier, feature_cols: List[str], match_features: Dict[str, Any]) -> Tuple[int, float]:
    """Predict the class and return its model probability."""
    missing = [c for c in feature_cols if c not in match_features]
    if missing:
        raise ValueError(f"Missing prediction features: {', '.join(missing)}")
    X_match = pd.DataFrame([match_features])[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    prediction = int(model.predict(X_match)[0])
    probability = float(model.predict_proba(X_match)[0].max())
    return prediction, probability
