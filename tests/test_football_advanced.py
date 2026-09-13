import pandas as pd

from src.football_advanced import ensemble_prediction, goals_distribution, h2h, team_strength, value_edge


def _history():
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=6),
        "home_team": ["A", "B", "A", "C", "B", "A"],
        "away_team": ["B", "C", "C", "A", "A", "B"],
        "home_goals": [2, 1, 3, 0, 1, 2],
        "away_goals": [0, 1, 1, 2, 2, 1],
    })


def test_team_strength_and_h2h():
    history = _history()
    strength = team_strength(history, "A")
    assert strength["matches"] > 0
    assert strength["attack"] >= 0
    hh = h2h(history, "A", "B")
    assert hh["matches"] == 3


def test_ensemble_is_probability_vector():
    base = {"ft_home": 0.55, "ft_draw": 0.25, "ft_away": 0.20}
    result = ensemble_prediction(base, {"form": 0.7, "attack": 1.8}, {"form": 0.4, "attack": 1.1}, elo_home=1580, elo_away=1490)
    total = result["ft_home"] + result["ft_draw"] + result["ft_away"]
    assert abs(total - 1.0) < 1e-9
    assert result["confidence"] >= 1 / 3


def test_value_scanner_and_score_distribution():
    value = value_edge(0.60, 2.0)
    assert value["implied_probability"] == 0.5
    assert value["edge"] == 0.1
    dist = goals_distribution(1.5, 1.0)
    assert not dist.empty
    assert dist.iloc[0]["probability"] > 0
