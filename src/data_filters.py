"""
Data filtering utilities for tennis and football prediction models.

Filters datasets to include only target competitions:
- Tennis: Grand Slams, ATP/WTA 1000, ATP/WTA 500
- Football: Champions League, Europa League, Conference League,
  Serie A, La Liga, Bundesliga, Premier League, Liga 1 (Romania)
"""

import pandas as pd
from typing import Tuple

GRAND_SLAM_NAMES = ["Australian Open", "Roland Garros", "Wimbledon", "US Open"]


def filter_tennis_target(df_atp: pd.DataFrame, df_wta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

TARGET_COMPETITION_IDS = ["UCL", "UEL", "UECL", "ES1", "DE1", "GB1", "IT1", "RO1"]


def _normalize_football_name(value: object) -> str:
    value = str(value).strip().casefold()
    value = value.replace("_", " ")
    value = " ".join(value.split())
    aliases = {
        "superliga": "liga 1",
        "super liga": "liga 1",
        "liga 1": "liga 1",
        "romania liga 1": "liga 1",
        "românia liga 1": "liga 1",
        "serie a": "serie a",
        "italy serie a": "serie a",
        "laliga": "la liga",
        "la liga": "la liga",
        "premier league": "premier league",
        "bundesliga": "bundesliga",
        "uefa champions league": "uefa champions league",
        "champions league": "uefa champions league",
        "uefa europa league": "uefa europa league",
        "europa league": "uefa europa league",
        "uefa conference league": "uefa conference league",
        "conference league": "uefa conference league",
    }
    return aliases.get(value, value)


def filter_football_target(df: pd.DataFrame, use_names: bool = True) -> pd.DataFrame:
    """Filter football data to the supported target competitions."""
    if df.empty:
        return df.copy()
    if use_names:
        col = "competition" if "competition" in df.columns else "league"
        if col not in df.columns:
            raise ValueError("Football data must contain a competition or league column")
        normalized = df[col].map(_normalize_football_name)
        allowed = {
            "uefa champions league",
            "uefa europa league",
            "uefa conference league",
            "la liga",
            "bundesliga",
            "premier league",
            "serie a",
            "liga 1",
        }
        mask = normalized.isin(allowed)
        df_target = df.loc[mask].copy()
        df_target["competition"] = normalized.loc[mask].map({
            "uefa champions league": "UEFA Champions League",
            "uefa europa league": "UEFA Europa League",
            "uefa conference league": "UEFA Conference League",
            "la liga": "La Liga",
            "bundesliga": "Bundesliga",
            "premier league": "Premier League",
            "serie a": "Serie A",
            "liga 1": "Liga 1",
        }).to_numpy()
    else:
        col = "competition_id"
        if col not in df.columns:
            raise ValueError("Football data must contain competition_id")
        df_target = df.loc[df[col].isin(TARGET_COMPETITION_IDS)].copy()
    df_target["category"] = df_target["competition"].map(_categorize_football_competition)
    return df_target


def _categorize_football_competition(comp_name: str) -> str:
    value = str(comp_name).strip().casefold()
    if "champions league" in value:
        return "European - Champions League"
    if "europa league" in value:
        return "European - Europa League"
    if "conference league" in value:
        return "European - Conference League"
    if value in {"laliga", "la liga"}:
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
