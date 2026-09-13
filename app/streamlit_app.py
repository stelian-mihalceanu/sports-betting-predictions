from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_filters import filter_football_target, filter_tennis_target
from src.data_loader import (
    load_football_matches,
    load_tennis_atp,
    load_tennis_wta,
    load_upcoming_football_fixtures,
)
from src.features import add_elo_features, add_football_form_features
from src.predictions import build_team_stats, predict_match

st.set_page_config(page_title="Sports Betting Predictions", page_icon="🏆", layout="wide")

st.title("🏆 Sports Betting Predictions")
st.caption("Football & tennis analytics dashboard — public data + pre-match ELO/model estimates")


@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_all_football() -> pd.DataFrame:
    data = load_football_matches(source="football-data")
    try:
        fixtures = load_upcoming_football_fixtures()
        if not fixtures.empty:
            data = pd.concat([data, fixtures], ignore_index=True, sort=False)
    except Exception as exc:
        st.warning(f"Upcoming football feed temporarily unavailable: {exc}")
    return data.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_all_tennis() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_tennis_atp(), load_tennis_wta()


def safe_load(loader):
    try:
        return loader(), None
    except Exception as exc:
        return None, str(exc)


def football_history_and_upcoming(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    completed = data.dropna(subset=["home_goals", "away_goals", "date"]).copy()
    upcoming = data[data["date"].notna() & data["home_goals"].isna() & data["away_goals"].isna()].copy()
    upcoming = upcoming[upcoming["date"] >= pd.Timestamp.now().normalize()]
    upcoming = upcoming.drop_duplicates(subset=["date", "home_team", "away_team", "competition"])
    return completed, upcoming


def build_football_features(completed: pd.DataFrame) -> pd.DataFrame:
    if completed.empty:
        return completed
    featured = add_football_form_features(completed)
    from src.features import add_elo_football_features
    return add_elo_football_features(featured)


def team_state(history: pd.DataFrame) -> dict[str, dict[str, float]]:
    ratings: dict[str, float] = {}
    form: dict[str, list[tuple[float, float, float]]] = {}
    for _, row in history.sort_values("date").iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        expected_home = 1.0 / (1.0 + 10.0 ** (((ra) - (rh + 60.0)) / 400.0))
        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        actual_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        change = 32.0 * (actual_home - expected_home)
        ratings[home], ratings[away] = rh + change, ra - change
        hp, ap = (3.0, 0.0) if hg > ag else ((0.0, 3.0) if hg < ag else (1.0, 1.0))
        form.setdefault(home, []).append((hp, hg, ag))
        form.setdefault(away, []).append((ap, ag, hg))
    state = {}
    for team in set(ratings) | set(form):
        recent = form.get(team, [])[-5:]
        state[team] = {
            "elo": ratings.get(team, 1500.0),
            "form_points": float(np.mean([x[0] for x in recent])) if recent else 0.0,
        }
    return state


@st.cache_data(ttl=30 * 60, show_spinner=False)
def build_prediction_state(completed: pd.DataFrame):
    return team_state(completed), build_team_stats(completed)


with st.sidebar:
    sport = st.radio("Sport", ["Football", "Tennis"], key="sport")
    st.divider()
    st.subheader("Data")
    st.caption("Football: Football-Data.co.uk + fixture feeds · Tennis: Sackmann archive")
    st.divider()
    st.caption("Data is refreshed automatically from the public sources.")

if sport == "Football":
    with st.spinner("Loading football data…"):
        data, error = safe_load(load_all_football)
    if data is None:
        st.error("Football data could not be loaded.")
        st.code(error)
        st.stop()

    filtered = filter_football_target(data)
    completed, upcoming = football_history_and_upcoming(filtered)
    featured = build_football_features(completed)

    completed_categories = set(featured["category"].dropna().unique()) if "category" in featured.columns else set()
    upcoming_categories = set(upcoming["category"].dropna().unique()) if "category" in upcoming.columns else set()
    categories = sorted(completed_categories | upcoming_categories)
    if not categories:
        st.warning("No target football competitions are currently available from the public feeds.")
        st.stop()

    category = st.selectbox("Competition", ["All"] + categories, key="football_category")
    view = featured if category == "All" else featured[featured["category"] == category]
    upcoming_view = upcoming if category == "All" else upcoming[upcoming["category"] == category]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed matches", f"{len(view):,}")
    c2.metric("Competitions", view["category"].nunique() if not view.empty else 0)
    c3.metric("Avg home ELO", f"{view['home_elo'].mean():.0f}" if not view.empty else "—")
    c4.metric("Avg away ELO", f"{view['away_elo'].mean():.0f}" if not view.empty else "—")

    tab1, tab2 = st.tabs(["📊 Analytics", "🔮 Upcoming predictions"])
    with tab1:
        st.subheader("Recent match analytics")
        if view.empty:
            st.info("No completed matches are currently available for this competition.")
        else:
            columns = [c for c in ["date", "competition", "home_team", "away_team", "home_goals", "away_goals", "home_elo", "away_elo", "elo_diff", "home_form_points", "away_form_points"] if c in view.columns]
            st.dataframe(view.sort_values("date", ascending=False)[columns].head(250), width="stretch", hide_index=True)

    with tab2:
        if upcoming_view.empty:
            st.info("No future fixtures are currently available in the public feed.")
        else:
            elo_state, market_state = build_prediction_state(completed)
            rows = []
            for _, match in upcoming_view.sort_values("date").head(150).iterrows():
                home, away = str(match["home_team"]), str(match["away_team"])
                prediction = predict_match(home, away, elo_state, market_state)
                rows.append({
                    "Date": match["date"],
                    "Competition": match["competition"],
                    "Match": f"{home} - {away}",
                    "FT": prediction["ft_pick"],
                    "FT 1": prediction["ft_home"],
                    "FT X": prediction["ft_draw"],
                    "FT 2": prediction["ft_away"],
                    "HT": prediction["ht_pick"],
                    "HT 1": prediction["ht_home"],
                    "HT X": prediction["ht_draw"],
                    "HT 2": prediction["ht_away"],
                    "Score": f"{prediction['likely_home_goals']}-{prediction['likely_away_goals']}",
                    "xG": f"{prediction['home_xg']:.1f}-{prediction['away_xg']:.1f}",
                    "Over 1.5": prediction["over_1_5"],
                    "Over 2.5": prediction["over_2_5"],
                    "BTTS": prediction["btts"],
                    "Corners": prediction["expected_corners"],
                    "Over 8.5 C": prediction["over_8_5_corners"],
                    "Cards": prediction["expected_cards"],
                    "Over 3.5 Cards": prediction["over_3_5_cards"],
                })

            predictions = pd.DataFrame(rows)
            percent_columns = [
                "FT 1", "FT X", "FT 2", "HT 1", "HT X", "HT 2",
                "Over 1.5", "Over 2.5", "BTTS", "Over 8.5 C", "Over 3.5 Cards",
            ]
            for column in percent_columns:
                predictions[column] = predictions[column].map(lambda value: f"{value:.0%}")
            predictions["Corners"] = predictions["Corners"].map(lambda value: f"{value:.1f}")
            predictions["Cards"] = predictions["Cards"].map(lambda value: f"{value:.1f}")

            st.caption(
                "Predictions are statistical estimates from recent goals, ELO, corners and card history. "
                "They are not bookmaker odds or guarantees."
            )
            st.dataframe(predictions, width="stretch", hide_index=True)

            st.subheader("📌 Market guide")
            st.markdown(
                "- **FT / HT:** model pick for final score or half-time result — **1 = home, X = draw, 2 = away**.\n"
                "- **Score / xG:** most likely exact score and expected goals.\n"
                "- **Over 1.5 / 2.5:** probability that total goals exceed the line.\n"
                "- **BTTS:** probability that both teams score.\n"
                "- **Corners:** expected total corners and probability of Over 8.5.\n"
                "- **Cards:** expected total cards and probability of Over 3.5."
            )

else:
    with st.spinner("Loading tennis data…"):
        tennis_data, error = safe_load(load_all_tennis)
    if tennis_data is None:
        st.error("Tennis data could not be loaded.")
        st.code(error)
        st.stop()
    atp, wta = tennis_data

    atp_target, wta_target = filter_tennis_target(atp, wta)
    atp_target["tour"] = "ATP"
    wta_target["tour"] = "WTA"
    view = pd.concat([atp_target, wta_target], ignore_index=True)
    view["tourney_date"] = pd.to_datetime(view["tourney_date"], format="%Y%m%d", errors="coerce")
    view = view.sort_values("tourney_date").reset_index(drop=True)
    view = add_elo_features(view)

    tour = st.selectbox("Tour", ["All", "ATP", "WTA"], key="tennis_tour")
    category = st.selectbox("Tournament category", ["All"] + sorted(view["category"].dropna().unique()), key="tennis_category")
    filtered_view = view
    if tour != "All":
        filtered_view = filtered_view[filtered_view["tour"] == tour]
    if category != "All":
        filtered_view = filtered_view[filtered_view["category"] == category]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(filtered_view):,}")
    c2.metric("ATP", f"{len(filtered_view[filtered_view['tour'] == 'ATP']):,}")
    c3.metric("WTA", f"{len(filtered_view[filtered_view['tour'] == 'WTA']):,}")
    c4.metric("Players", f"{pd.unique(pd.concat([filtered_view['winner_name'], filtered_view['loser_name']])).size:,}")

    tab1, tab2 = st.tabs(["📊 Match analytics", "🎾 Player ELO"])
    with tab1:
        columns = [c for c in ["tourney_date", "tourney_name", "tour", "winner_name", "loser_name", "winner_elo", "loser_elo", "elo_diff", "surface"] if c in filtered_view.columns]
        st.dataframe(filtered_view.sort_values("tourney_date", ascending=False)[columns].head(300), width="stretch", hide_index=True)
    with tab2:
        latest = {}
        for _, row in view.sort_values("tourney_date").iterrows():
            latest[row["winner_name"]] = row["winner_elo"]
            latest[row["loser_name"]] = row["loser_elo"]
        ratings = pd.DataFrame(sorted(latest.items(), key=lambda item: item[1], reverse=True), columns=["Player", "Pre-match ELO"])
        st.dataframe(ratings.head(100), width="stretch", hide_index=True)

st.divider()
st.caption("This dashboard is for research and decision support. Probabilities are estimates, not guarantees of outcomes or profit.")
