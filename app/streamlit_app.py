from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_filters import filter_football_target, filter_tennis_target
from src.data_loader import load_football_matches, load_tennis_atp, load_tennis_wta
from src.features import add_elo_features, add_football_form_features

st.set_page_config(page_title="Sports Betting Predictions", page_icon="🏆", layout="wide")

st.title("🏆 Sports Betting Predictions")
st.caption("Football & tennis analytics dashboard — public data + pre-match ELO/model estimates")

FOOTBALL_DATA = "https://www.football-data.co.uk/mmz4281"
FOOTBALL_LEAGUES = {
    "E0": "Premier League",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "RO1": "Liga 1",
}
TENNIS_ARCHIVE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_all_football() -> pd.DataFrame:
    try:
        return load_football_matches(source="football-data")
    except Exception as exc:
        if "Unknown source" not in str(exc):
            raise
        frames = []
        for start_year in range(2022, 2027):
            season = f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"
            for code, competition in FOOTBALL_LEAGUES.items():
                url = f"{FOOTBALL_DATA}/{season}/{code}.csv"
                response = requests.get(url, timeout=20)
                if response.status_code != 200:
                    continue
                frame = pd.read_csv(pd.io.common.BytesIO(response.content), encoding="latin1")
                frame = frame.rename(columns={
                    "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
                    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
                })
                required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
                if not required.issubset(frame.columns):
                    continue
                frame["date"] = pd.to_datetime(frame["date"], dayfirst=True, errors="coerce")
                frame["competition"] = competition
                frame["season"] = season
                frames.append(frame)
        if not frames:
            raise RuntimeError("Unable to load public football data") from exc
        result = pd.concat(frames, ignore_index=True)
        for column in ["home_goals", "away_goals"]:
            result[column] = pd.to_numeric(result[column], errors="coerce")
        return result.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_all_tennis() -> tuple[pd.DataFrame, pd.DataFrame]:
    def load_tour(tour: str) -> pd.DataFrame:
        try:
            return load_tennis_atp() if tour == "atp" else load_tennis_wta()
        except Exception:
            frames = []
            for year in range(2022, 2027):
                url = f"{TENNIS_ARCHIVE}/{tour}/{tour}_matches_{year}.csv"
                response = requests.get(url, timeout=20)
                if response.status_code == 200:
                    frame = pd.read_csv(pd.io.common.BytesIO(response.content))
                    frame["tour"] = tour.upper()
                    frames.append(frame)
            if not frames:
                raise RuntimeError(f"Unable to load remote {tour.upper()} tennis data")
            return pd.concat(frames, ignore_index=True)

    return load_tour("atp"), load_tour("wta")


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


def heuristic_1x2(home_elo: float, away_elo: float) -> tuple[float, float, float]:
    strength = 1.0 / (1.0 + 10.0 ** (((away_elo) - (home_elo + 60.0)) / 400.0))
    draw = 0.24
    home = strength * (1.0 - draw)
    away = (1.0 - strength) * (1.0 - draw)
    return home, draw, away


with st.sidebar:
    sport = st.radio("Sport", ["Football", "Tennis"])
    st.divider()
    st.subheader("Data")
    st.caption("Football: Football-Data.co.uk · Tennis: Sackmann archive")
    st.divider()
    st.caption("Data is refreshed automatically from the public sources.")

if sport == "Football":
    data, error = safe_load(load_all_football)
    if data is None:
        st.error("Football data could not be loaded.")
        st.code(error)
        st.stop()

    filtered = filter_football_target(data)
    completed, upcoming = football_history_and_upcoming(filtered)
    if completed.empty:
        st.warning("No completed football matches are available for the target competitions.")
        st.stop()

    featured = build_football_features(completed)
    categories = sorted(featured["category"].dropna().unique())
    category = st.selectbox("Competition", ["All"] + categories)
    view = featured if category == "All" else featured[featured["category"] == category]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed matches", f"{len(view):,}")
    c2.metric("Competitions", view["category"].nunique())
    c3.metric("Avg home ELO", f"{view['home_elo'].mean():.0f}")
    c4.metric("Avg away ELO", f"{view['away_elo'].mean():.0f}")

    tab1, tab2 = st.tabs(["📊 Analytics", "🔮 Upcoming predictions"])
    with tab1:
        st.subheader("Recent match analytics")
        columns = [c for c in ["date", "competition", "home_team", "away_team", "home_goals", "away_goals", "home_elo", "away_elo", "elo_diff", "home_form_points", "away_form_points"] if c in view.columns]
        st.dataframe(view.sort_values("date", ascending=False)[columns].head(250), use_container_width=True, hide_index=True)

    with tab2:
        if upcoming.empty:
            st.info("No future fixtures are currently available in the public feed.")
        else:
            state = team_state(completed)
            rows = []
            for _, match in upcoming.sort_values("date").head(100).iterrows():
                home, away = str(match["home_team"]), str(match["away_team"])
                hs = state.get(home, {"elo": 1500.0})
                aws = state.get(away, {"elo": 1500.0})
                ph, pd, pa = heuristic_1x2(hs["elo"], aws["elo"])
                rows.append({
                    "Date": match["date"], "Competition": match["competition"],
                    "Home": home, "Away": away, "Home win": ph, "Draw": pd,
                    "Away win": pa, "Model pick": max([(ph, "1"), (pd, "X"), (pa, "2")])[1],
                    "ELO diff": hs["elo"] - aws["elo"],
                })
            predictions = pd.DataFrame(rows)
            for column in ["Home win", "Draw", "Away win"]:
                predictions[column] = predictions[column].map(lambda value: f"{value:.0%}")
            st.caption("ELO probabilities are model estimates, not bookmaker odds or guarantees.")
            st.dataframe(predictions, use_container_width=True, hide_index=True)

else:
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

    tour = st.selectbox("Tour", ["All", "ATP", "WTA"])
    category = st.selectbox("Tournament category", ["All"] + sorted(view["category"].dropna().unique()))
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
        st.dataframe(filtered_view.sort_values("tourney_date", ascending=False)[columns].head(300), use_container_width=True, hide_index=True)
    with tab2:
        latest = {}
        for _, row in view.sort_values("tourney_date").iterrows():
            latest[row["winner_name"]] = row["winner_elo"]
            latest[row["loser_name"]] = row["loser_elo"]
        ratings = pd.DataFrame(sorted(latest.items(), key=lambda item: item[1], reverse=True), columns=["Player", "Pre-match ELO"])
        st.dataframe(ratings.head(100), use_container_width=True, hide_index=True)

st.divider()
st.caption("This dashboard is for research and decision support. Probabilities are estimates, not guarantees of outcomes or profit.")
