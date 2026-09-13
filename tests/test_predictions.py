import pandas as pd

from src.predictions import build_team_stats, most_likely_score, predict_match


def test_prediction_engine_returns_markets():
    history = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4),
            "home_team": ["A", "B", "A", "B"],
            "away_team": ["B", "A", "B", "A"],
            "home_goals": [2, 0, 1, 2],
            "away_goals": [0, 1, 1, 1],
            "HTHG": [1, 0, 1, 1],
            "HTAG": [0, 0, 0, 0],
            "HC": [6, 3, 5, 4],
            "AC": [2, 4, 3, 2],
            "HY": [2, 1, 3, 2],
            "AY": [1, 2, 1, 3],
            "HR": [0, 0, 0, 0],
            "AR": [0, 0, 0, 0],
        }
    )
    stats = build_team_stats(history)
    prediction = predict_match("A", "B", {"A": {"elo": 1550}, "B": {"elo": 1480}}, stats)

    assert prediction["ft_pick"] in {"1", "X", "2"}
    assert prediction["ht_pick"] in {"1", "X", "2"}
    assert prediction["home_xg"] > 0
    assert prediction["away_xg"] > 0
    assert prediction["expected_corners"] > 0
    assert prediction["expected_cards"] > 0
    assert 0 <= prediction["over_2_5"] <= 1
    assert 0 <= prediction["btts"] <= 1


def test_most_likely_score_is_integer_pair():
    home, away = most_likely_score(1.8, 0.8)
    assert isinstance(home, int)
    assert isinstance(away, int)
    assert home >= 0
    assert away >= 0
