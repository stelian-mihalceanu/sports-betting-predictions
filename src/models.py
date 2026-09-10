"""
Machine learning model utilities for tennis and football predictions.

Includes:
- XGBoost classifiers for match outcomes
- Poisson regression for goal totals (football)
- Training and evaluation functions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, classification_report
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# TENNIS MODELS
# =============================================================================

def train_tennis_model(
    df: pd.DataFrame,
    target_col: str = "winner",
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[XGBClassifier, List[str], Dict[str, float]]:
    """
    Train XGBoost classifier for tennis match outcomes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tennis matches DataFrame with features and target.
    target_col : str, default "winner"
        Target column name (binary: 1/0).
    feature_cols : List[str], optional
        List of feature column names. If None, auto-detect numeric columns.
    test_size : float, default 0.2
        Test set proportion.
    random_state : int, default 42
        Random seed for reproducibility.
    
    Returns
    -------
    Tuple[XGBClassifier, List[str], Dict[str, float]]
        Trained model, feature columns, and evaluation metrics.
    """
    if feature_cols is None:
        feature_cols = _get_numeric_feature_cols(df, exclude=[target_col])
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Temporal split (no shuffling for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba)
    }
    
    print("Tennis Model Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")
    
    return model, feature_cols, metrics


# =============================================================================
# FOOTBALL MODELS
# =============================================================================

def train_football_1x2_model(
    df: pd.DataFrame,
    target_col: str = "result",
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[XGBClassifier, List[str], Dict[str, float]]:
    """
    Train XGBoost classifier for football 1X2 outcomes.
    
    Parameters
    ----------
    df : pd.DataFrame
        Football matches DataFrame with features and target.
    target_col : str, default "result"
        Target column name (multi-class: 1/X/2 or 0/1/2).
    feature_cols : List[str], optional
        List of feature column names. If None, auto-detect numeric columns.
    test_size : float, default 0.2
        Test set proportion.
    random_state : int, default 42
        Random seed for reproducibility.
    
    Returns
    -------
    Tuple[XGBClassifier, List[str], Dict[str, float]]
        Trained model, feature columns, and evaluation metrics.
    """
    if feature_cols is None:
        feature_cols = _get_numeric_feature_cols(df, exclude=[target_col])
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Temporal split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba)
    }
    
    print("Football 1X2 Model Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, feature_cols, metrics


def train_football_goals_model(
    df: pd.DataFrame,
    target_col: str = "total_goals",
    feature_cols: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[XGBClassifier, List[str], Dict[str, float]]:
    """
    Train XGBoost classifier for football goal totals (e.g., Over/Under 2.5).
    
    Parameters
    ----------
    df : pd.DataFrame
        Football matches DataFrame with features and target.
    target_col : str, default "total_goals"
        Target column name (binary: Over/Under).
    feature_cols : List[str], optional
        List of feature column names. If None, auto-detect numeric columns.
    test_size : float, default 0.2
        Test set proportion.
    random_state : int, default 42
        Random seed for reproducibility.
    
    Returns
    -------
    Tuple[XGBClassifier, List[str], Dict[str, float]]
        Trained model, feature columns, and evaluation metrics.
    """
    if feature_cols is None:
        feature_cols = _get_numeric_feature_cols(df, exclude=[target_col])
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": log_loss(y_test, y_proba)
    }
    
    print("Football Goals Model Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")
    
    return model, feature_cols, metrics


# =============================================================================
# UTILITIES
# =============================================================================

def _get_numeric_feature_cols(
    df: pd.DataFrame,
    exclude: List[str]
) -> List[str]:
    """
    Auto-detect numeric feature columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    exclude : List[str]
        Columns to exclude.
    
    Returns
    -------
    List[str]
        List of numeric column names.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def predict_match_outcome(
    model: XGBClassifier,
    feature_cols: List[str],
    match_features: Dict[str, Any]
) -> Tuple[int, float]:
    """
    Predict match outcome using trained model.
    
    Parameters
    ----------
    model : XGBClassifier
        Trained XGBoost model.
    feature_cols : List[str]
        Feature column names used during training.
    match_features : Dict[str, Any]
        Dictionary of feature values for the match.
    
    Returns
    -------
    Tuple[int, float]
        Predicted class and probability.
    """
    X_match = pd.DataFrame([match_features])[feature_cols].fillna(0)
    
    prediction = model.predict(X_match)[0]
    probability = model.predict_proba(X_match)[0].max()
    
    return int(prediction), float(probability)
