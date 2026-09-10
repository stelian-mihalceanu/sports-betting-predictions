"""
Feature engineering utilities for tennis and football prediction models.

Includes:
- Form features (recent performance)
- ELO ratings (tennis: by surface; football: home/away)
- Head-to-head statistics
- Temporal features (days since last match, etc.)
"""

import pandas as pd
import numpy as np
from typing import List, Optional


# =============================================================================
# TENNIS FEATURES
# =============================================================================

def add_tennis_form_features(
    df: pd.DataFrame,
    window: int = 10,
    min_periods: int = 3
) -> pd.DataFrame:
    """
    Add form features for tennis players (ATP/WTA).
    
    Features added:
    - winner_form_wins: rolling mean of wins for winner
    - loser_form_wins: rolling mean of wins for loser
    - winner_form_games_won, loser_form_games_won
    
    Parameters
    ----------
    df : pd.DataFrame
        Tennis matches DataFrame with columns:
        'winner_name', 'loser_name', 'winner_games_won', 'loser_games_won'.
    window : int, default 10
        Rolling window size for form calculation.
    min_periods : int, default 3
        Minimum periods required for rolling calculation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added form features.
    """
    df = df.sort_values(["tourney_date", "tourney_name"]).reset_index(drop=True)
    
    # Winner form
    winner_stats = _player_rolling_stats(
        df, "winner_name", "winner_games_won", window, min_periods
    )
    df = df.join(winner_stats.add_prefix("winner_"))
    
    # Loser form
    loser_stats = _player_rolling_stats(
        df, "loser_name", "loser_games_won", window, min_periods
    )
    df = df.join(loser_stats.add_prefix("loser_"))
    
    return df


def _player_rolling_stats(
    df: pd.DataFrame,
    player_col: str,
    stats_col: str,
    window: int,
    min_periods: int
) -> pd.DataFrame:
    """
    Calculate rolling statistics per player.
    
    Parameters
    ----------
    df : pd.DataFrame
        Matches DataFrame.
    player_col : str
        Column name for player identifier.
    stats_col : str
        Column name for statistic to aggregate.
    window : int
        Rolling window size.
    min_periods : int
        Minimum periods for rolling calculation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with rolling mean and std columns.
    """
    grp = df.groupby(player_col)[stats_col]
    rolling_mean = grp.transform(
        lambda x: x.rolling(window, min_periods=min_periods).mean().shift(1)
    )
    rolling_std = grp.transform(
        lambda x: x.rolling(window, min_periods=min_periods).std().shift(1)
    )
    
    result = pd.DataFrame({
        "form_mean": rolling_mean,
        "form_std": rolling_std
    })
    return result.fillna(0)


def add_elo_features(
    df: pd.DataFrame,
    surface_weighted: bool = True
) -> pd.DataFrame:
    """
    Add ELO rating features for tennis players.
    
    Parameters
    ----------
    df : pd.DataFrame
        Tennis matches DataFrame.
    surface_weighted : bool, default True
        If True, calculate separate ELO per surface.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with ELO features.
    """
    # Simplified ELO calculation (to be extended)
    df["winner_elo"] = 1500  # Placeholder
    df["loser_elo"] = 1500   # Placeholder
    df["elo_diff"] = df["winner_elo"] - df["loser_elo"]
    
    return df


# =============================================================================
# FOOTBALL FEATURES
# =============================================================================

def add_football_form_features(
    df: pd.DataFrame,
    window: int = 5,
    min_periods: int = 2
) -> pd.DataFrame:
    """
    Add form features for football teams.
    
    Features added:
    - home_form_points, away_form_points
    - home_form_goals_scored, away_form_goals_scored
    - home_form_goals_conceded, away_form_goals_conceded
    
    Parameters
    ----------
    df : pd.DataFrame
        Football matches DataFrame with columns:
        'home_team', 'away_team', 'home_goals', 'away_goals',
        'date', 'competition'.
    window : int, default 5
        Rolling window size for form calculation.
    min_periods : int, default 2
        Minimum periods for rolling calculation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added form features.
    """
    df = df.sort_values(["competition", "date"]).reset_index(drop=True)
    
    # Home team form
    home_form = _team_form_stats(
        df, "home_team", "home_goals", "away_goals", window, min_periods
    )
    df = df.join(home_form.add_prefix("home_"))
    
    # Away team form
    away_form = _team_form_stats(
        df, "away_team", "away_goals", "home_goals", window, min_periods
    )
    df = df.join(away_form.add_prefix("away_"))
    
    return df


def _team_form_stats(
    df: pd.DataFrame,
    team_col: str,
    goals_for_col: str,
    goals_against_col: str,
    window: int,
    min_periods: int
) -> pd.DataFrame:
    """
    Calculate rolling form statistics per team.
    
    Parameters
    ----------
    df : pd.DataFrame
        Matches DataFrame.
    team_col : str
        Column name for team identifier.
    goals_for_col : str
        Column name for goals scored.
    goals_against_col : str
        Column name for goals conceded.
    window : int
        Rolling window size.
    min_periods : int
        Minimum periods for rolling calculation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with form features.
    """
    grp = df.groupby(team_col)
    
    form_goals_for = grp[goals_for_col].transform(
        lambda x: x.rolling(window, min_periods=min_periods).mean().shift(1)
    )
    form_goals_against = grp[goals_against_col].transform(
        lambda x: x.rolling(window, min_periods=min_periods).mean().shift(1)
    )
    
    result = pd.DataFrame({
        "form_goals_scored": form_goals_for,
        "form_goals_conceded": form_goals_against
    })
    return result.fillna(0)


def add_elo_football_features(
    df: pd.DataFrame,
    k_factor: float = 32.0
) -> pd.DataFrame:
    """
    Add ELO rating features for football teams.
    
    Parameters
    ----------
    df : pd.DataFrame
        Football matches DataFrame.
    k_factor : float, default 32.0
        ELO K-factor for rating updates.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with ELO features.
    """
    # Simplified ELO (to be extended with full implementation)
    df["home_elo"] = 1500  # Placeholder
    df["away_elo"] = 1500  # Placeholder
    df["elo_diff"] = df["home_elo"] - df["away_elo"]
    
    return df
