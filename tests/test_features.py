import pandas as pd

from src.data_filters import filter_tennis_target
from src.features import add_elo_football_features, add_elo_features, add_football_form_features


def test_tennis_elo_uses_pre_match_rating():
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "winner_name": ["A", "B"],
        "loser_name": ["B", "A"],
        "surface": ["Hard", "Hard"],
    })
    out = add_elo_features(df)
    assert out.loc[0, "winner_elo"] == 1500
    assert out.loc[0, "loser_elo"] == 1500
    assert out.loc[1, "winner_elo"] != 1500


def test_football_elo_updates_after_result():
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "home_team": ["A", "A"],
        "away_team": ["B", "B"],
        "home_goals": [2, 0],
        "away_goals": [0, 1],
    })
    out = add_elo_football_features(df)
    assert out.loc[0, "home_elo"] == 1500
    assert out.loc[1, "home_elo"] > out.loc[1, "away_elo"]


def test_football_form_excludes_current_match():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=3),
        "home_team": ["A", "A", "B"],
        "away_team": ["B", "C", "C"],
        "home_goals": [3, 0, 2],
        "away_goals": [0, 0, 1],
    })
    out = add_football_form_features(df, window=2, min_periods=1)
    assert out.loc[1, "home_form_goals_scored"] == 3
    assert out.loc[1, "home_form_goals_conceded"] == 0


def test_wta_level_a_is_not_classified_as_atp_500():
    atp = pd.DataFrame({
        "tourney_name": ["Test ATP"],
        "tourney_level": ["A"],
    })
    wta = pd.DataFrame({
        "tourney_name": ["Test WTA"],
        "tourney_level": ["A"],
    })
    atp_out, wta_out = filter_tennis_target(atp, wta)
    assert atp_out.loc[atp_out.index[0], "category"] == "ATP 500"
    assert wta_out.loc[wta_out.index[0], "category"] == "WTA 500"
