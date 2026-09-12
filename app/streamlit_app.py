from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_filters import filter_football_target, filter_tennis_target
from src.data_loader import load_football_matches, load_tennis_atp, load_tennis_wta
from src.features import add_elo_football_features, add_elo_features, add_football_form_features

st.set_page_config(page_title="Sports Betting Predictions", page_icon="🏆", layout="wide")
st.title("🏆 Sports Betting Predictions")
st.caption("Football & tennis analytics dashboard")


def safe_load(loader):
    try:
        return loader(), None
    except Exception as exc:
        return None, str(exc)


with st.sidebar:
    sport = st.radio("Sport", ["Football", "Tennis"])
    st.divider()
    st.subheader("Dashboard")
    st.write("Explore target competitions, recent form and pre-match ELO ratings.")

if sport == "Football":
    data, error = safe_load(load_football_matches)
    if data is None:
        st.warning("Football data is not configured yet.")
        st.code(error)
        st.stop()

    filtered = filter_football_target(data)
    if filtered.empty:
        st.warning("No target football matches were found in the loaded dataset.")
        st.stop()

    filtered = add_football_form_features(filtered)
    filtered = add_elo_football_features(filtered)
    categories = sorted(filtered["category"].dropna().unique())
    category = st.selectbox("Competition", ["All"] + categories)
    view = filtered if category == "All" else filtered[filtered["category"] == category]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(view):,}")
    c2.metric("Competitions", view["category"].nunique())
    c3.metric("Avg home ELO", f"{view['home_elo'].mean():.0f}")
    c4.metric("Avg away ELO", f"{view['away_elo'].mean():.0f}")

    st.subheader("Match analytics")
    columns = [c for c in ["date", "competition", "home_team", "away_team", "home_elo", "away_elo", "elo_diff", "home_form_points", "away_form_points"] if c in view.columns]
    st.dataframe(view[columns].tail(250), use_container_width=True, hide_index=True)

else:
    atp, atp_error = safe_load(load_tennis_atp)
    wta, wta_error = safe_load(load_tennis_wta)
    if atp is None and wta is None:
        st.warning("ATP/WTA data is not configured yet.")
        st.code(atp_error or wta_error)
        st.stop()

    atp = atp if atp is not None else pd.DataFrame()
    wta = wta if wta is not None else pd.DataFrame()
    atp_target, wta_target = filter_tennis_target(atp, wta)
    view = pd.concat([atp_target.assign(tour="ATP"), wta_target.assign(tour="WTA")], ignore_index=True)
    if view.empty:
        st.warning("No target tennis matches were found in the loaded datasets.")
        st.stop()

    view = add_elo_features(view)
    tour = st.selectbox("Tour", ["All", "ATP", "WTA"])
    if tour != "All":
        view = view[view["tour"] == tour]

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", f"{len(view):,}")
    c2.metric("ATP", f"{len(view[view['tour'] == 'ATP']):,}")
    c3.metric("WTA", f"{len(view[view['tour'] == 'WTA']):,}")

    st.subheader("Match analytics")
    columns = [c for c in ["tourney_date", "tourney_name", "tour", "winner_name", "loser_name", "winner_elo", "loser_elo", "elo_diff"] if c in view.columns]
    st.dataframe(view[columns].tail(250), use_container_width=True, hide_index=True)

st.divider()
st.caption("Probabilities and ratings are model estimates, not guarantees of outcomes or profit.")
