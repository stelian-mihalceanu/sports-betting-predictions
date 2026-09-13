"""
Data filtering utilities for tennis and football prediction models.

Filters datasets to include only target competitions:
- Tennis: Grand Slams, ATP/WTA 1000, ATP/WTA 500
- Football: Champions League, Europa League, Conference League,
  Serie A, La Liga, Bundesliga, Premier League, Liga 1 (Romania)
"""

import pandas as pd
from typing import Tuple

GRAND_SLAM_NAMES = [
    "Australian Open",
    "Roland Garros",
    "Wimbledon",
    "US Open",
]


def filter_tennis_target(
    df_atp: pd.DataFrame,
    df_wta: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter ATP and WTA frames to the supported tournament levels."""
    atp_target = df_atp[df_atp["tourney_level"].isin(["G", "M", "A"])].copy()
    wta_target = df_wta[df_wta["tourney_level"].isin(["G", "P", "A"])].copy()
    atp_target["category"] = atp_target.apply(_categorize_tennis_tourney, axis=1)
    wta_target["category"] = wta_target.apply(_categorize_tennis_tourney, axis=1)
    return atp_target, wta_target


def _categorize_tennis_tourney(row: pd.Series) -> str:
    name = str(row.get("tourney_name", ""))
    level = str(row.get("tourney_level", ""))
    tour = str(row.get("tour", "ATP")).upper()

    if name in GRAND_SLAM_NAMES:
        return "Grand Slam"
    if tour == "WTA":
        if level == "P":
            return "WTA 1000"
        if level == "A":
            return "WTA 500"
    else:
        if level == "M":
            return "ATP 1000"
        if level == "A":
            return "ATP 500"
    return "Other"


TARGET_FOOTBALL_COMPETITIONS = [
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
    "LaLiga",
    "La Liga",
    "Bundesliga",
    "Premier League",
    "Serie A",
    "SuperLiga",
    "Liga 1",
    "Romania Liga 1",
]

TARGET_COMPETITION_IDS = [
    "UCL",
    "UEL",
    "UECL",
    "ES1",
    "DE1",
    "GB1",
    "IT1",
    "RO1",
]


def filter_football_target(df: pd.DataFrame, use_names: bool = True) -> pd.DataFrame:
    """Filter football data to the supported target competitions."""
    if use_names:
        col = "competition" if "competition" in df.columns else "league"
        normalized = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        aliases = {
            "superliga": "Liga 1",
            "super liga": "Liga 1",
            "liga 1": "Liga 1",
            "romania liga 1": "Liga 1",
            "serie a": "Serie A",
            "italy serie a": "Serie A",
            "laliga": "La Liga",
        }
        canonical = normalized.str.lower().map(aliases).fillna(normalized)
        allowed = {
            "uefa champions league",
            "uefa europa league",
            "uefa conference league",
            "laliga",
            "la liga",
            "bundesliga",
            "premier league",
            "serie a",
            "superliga",
            "liga 1",
            "romania liga 1",
        }
        df_target = df[canonical.str.lower().isin(allowed)].copy()
        df_target["competition"] = canonical[canonical.index.isin(df_target.index)].values
    else:
        col = "competition_id"
        df_target = df[df[col].isin(TARGET_COMPETITION_IDS)].copy()
    df_target["category"] = df_target[col].apply(_categorize_football_competition)
    return df_target


def _categorize_football_competition(comp_name: str) -> str:
    value = str(comp_name).strip().lower()
    if "champions league" in value:
        return "European - Champions League"
    if "europa league" in value:
        return "European - Europa League"
    if "conference league" in value:
        return "European - Conference League"
    if "laliga" in value or "la liga" in value:
        return "La Liga"
    if "bundesliga" in value:
        return "Bundesliga"
    if "premier league" in value:
        return "Premier League"
    if "serie a" in value:
        return "Serie A"
    if "superliga" in value or "liga 1" in value or "romania liga 1" in value:
        return "Liga 1 (Romania)"
    return "Other"
