"""
Data filtering utilities for tennis and football prediction models.

Filters datasets to include only target competitions:
- Tennis: Grand Slams, ATP/WTA 1000, ATP/WTA 500
- Football: Champions League, Europa League, Conference League,
  La Liga, Bundesliga, Premier League, Liga 1 (Romania)
"""

import pandas as pd
from typing import Tuple

# =============================================================================
# TENNIS FILTERS
# =============================================================================

GRAND_SLAM_NAMES = [
    "Australian Open",
    "Roland Garros",
    "Wimbledon",
    "US Open"
]

def filter_tennis_target(
    df_atp: pd.DataFrame,
    df_wta: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter ATP and WTA DataFrames to include only target tournaments.
    
    ATP tourney_level codes:
    - G: Grand Slam
    - M: Masters 1000
    - A: ATP 500
    
    WTA tourney_level codes:
    - G: Grand Slam
    - P: WTA 1000 (Premier Mandatory/Premier 5)
    - A: WTA 500 (Premier)
    
    Parameters
    ----------
    df_atp : pd.DataFrame
        ATP matches DataFrame with 'tourney_level' and 'tourney_name' columns.
    df_wta : pd.DataFrame
        WTA matches DataFrame with 'tourney_level' and 'tourney_name' columns.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Filtered (atp_target, wta_target) DataFrames.
    """
    # ATP: Grand Slam + Masters 1000 + ATP 500
    atp_target = df_atp[
        df_atp["tourney_level"].isin(["G", "M", "A"])
    ].copy()
    
    # WTA: Grand Slam + WTA 1000 + WTA 500
    wta_target = df_wta[
        df_wta["tourney_level"].isin(["G", "P", "A"])
    ].copy()
    
    # Add category column
    atp_target["category"] = atp_target.apply(_categorize_tennis_tourney, axis=1)
    wta_target["category"] = wta_target.apply(_categorize_tennis_tourney, axis=1)
    
    return atp_target, wta_target


def _categorize_tennis_tourney(row: pd.Series) -> str:
    """
    Categorize tennis tournament based on name and level.
    
    Parameters
    ----------
    row : pd.Series
        Row with 'tourney_name' and 'tourney_level' columns.
    
    Returns
    -------
    str
        Category string (e.g., 'Grand Slam', 'ATP 1000', 'WTA 500').
    """
    name = row.get("tourney_name", "")
    level = row.get("tourney_level", "")
    tour = row.get("tour", "ATP")  # Assume ATP if not specified
    
    if name in GRAND_SLAM_NAMES:
        return "Grand Slam"
    
    if tour == "ATP" or level in ["M", "A"]:
        if level == "M":
            return "ATP 1000"
        if level == "A":
            return "ATP 500"
    
    if tour == "WTA" or level in ["P", "A"]:
        if level == "P":
            return "WTA 1000"
        if level == "A":
            return "WTA 500"
    
    return "Other"


# =============================================================================
# FOOTBALL FILTERS
# =============================================================================

TARGET_FOOTBALL_COMPETITIONS = [
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
    "LaLiga",
    "La Liga",
    "Bundesliga",
    "Premier League",
    "SuperLiga",
    "Liga 1",
    "Romania Liga 1"
]

TARGET_COMPETITION_IDS = [
    "UCL",      # Champions League
    "UEL",      # Europa League
    "UECL",     # Conference League
    "ES1",      # La Liga
    "DE1",      # Bundesliga
    "GB1",      # Premier League
    "RO1",      # Liga 1 Romania
]


def filter_football_target(
    df: pd.DataFrame,
    use_names: bool = True
) -> pd.DataFrame:
    """
    Filter football DataFrame to include only target competitions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Football matches DataFrame with 'competition' or 'competition_id' column.
    use_names : bool, default True
        If True, filter by competition names.
        If False, filter by competition IDs.
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with only target competitions.
    """
    if use_names:
        col = "competition" if "competition" in df.columns else "league"
        df_target = df[df[col].isin(TARGET_FOOTBALL_COMPETITIONS)].copy()
    else:
        col = "competition_id"
        df_target = df[df[col].isin(TARGET_COMPETITION_IDS)].copy()
    
    # Add normalized category
    df_target["category"] = df_target[col].apply(_categorize_football_competition)
    
    return df_target


def _categorize_football_competition(comp_name: str) -> str:
    """
    Categorize football competition.
    
    Parameters
    ----------
    comp_name : str
        Competition name.
    
    Returns
    -------
    str
        Category string (e.g., 'European', 'La Liga', 'Premier League').
    """
    if "Champions League" in comp_name:
        return "European - Champions League"
    if "Europa League" in comp_name:
        return "European - Europa League"
    if "Conference League" in comp_name:
        return "European - Conference League"
    if "LaLiga" in comp_name or "La Liga" in comp_name:
        return "La Liga"
    if "Bundesliga" in comp_name:
        return "Bundesliga"
    if "Premier League" in comp_name:
        return "Premier League"
    if "SuperLiga" in comp_name or "Liga 1" in comp_name:
        return "Liga 1 (Romania)"
    return "Other"
