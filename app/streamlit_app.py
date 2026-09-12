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

st.set_page_config(page_title="Sports Betting Predictions", page_icon="🏆", layout="wide")

st.title("🏆 Sports Betting Predictions")
st.caption("Decision-support dashboard for football and tennis model outputs.")


def _safe_load(loader, *args, **kwargs):
    try:
        return loader(*args, **kwargs), None
    except FileNotFoundError as exc:
        return None, str(exc)
    except Exception as exc:  # surface data/config problems without crashing the UI
        return None, f"Could not load data: {exc}"


with st.sidebar:
    st.header("Settings")
    sport = st.selectbox("Sport", ["Football", "Tennis"])
    st.divider()
    st.write("Data directory")
    st.code(os.getenv("DATEWISE_DATA_DIR", "data/"), language="text")

if sport == "Football":
    st.subheader("Football")
    df, error = _safe_load(load_football_matches)
    if df is None:
        st.warning("No local football dataset is configured yet.")
        st.info(error)
        st.stop()

    filtered = filter_football_target(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", f"{len(filtered):,}")
    c2.metric("Competitions", filtered["category"].nunique())
    c3.metric("Teams", pd.unique(pd.concat([filtered["home_team"], filtered["away_team"]])).size)

    categories = ["All"] + sorted(filtered["category"].dropna().unique().tolist())
    category = st.selectbox("Competition", categories)
    view = filtered if category == "All" else filtered[filtered["category"] == category]

    st.subheader("Matches")
    st.dataframe(view.head(250), use_container_width=True, hide_index=True)

else:
    st.subheader("Tennis")
    atp, atp_error = _safe_load(load_tennis_atp)
    wta, wta_error = _safe_load(load_tennis_wta)
    if atp is None and wta is None:
        st.warning("No local ATP/WTA datasets are configured yet.")
        st.info(atp_error or wta_error)
        st.stop()

    atp = atp if atp is not None else pd.DataFrame()
    wta = wta if wta is not None else pd.DataFrame()
    atp_target, wta_target = filter_tennis_target(atp, wta)
    combined = pd.concat([
        atp_target.assign(tour="ATP"),
        wta_target.assign(tour="WTA"),
    ], ignore_index=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Matches", f"{len(combined):,}")
    c2.metric("ATP", f"{len(atp_target):,}")
    c3.metric("WTA", f"{len(wta_target):,}")

    tour = st.selectbox("Tour", ["All", "ATP", "WTA"])
    view = combined if tour == "All" else combined[combined["tour"] == tour]
    st.dataframe(view.head(250), use_container_width=True, hide_index=True)

st.divider()
st.caption("Predictions are probabilistic estimates, not guarantees. Use responsibly.")
