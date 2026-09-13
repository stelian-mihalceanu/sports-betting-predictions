from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_filters import filter_football_target, filter_tennis_target
from src.data_loader import load_football_matches, load_tennis_atp, load_tennis_wta, load_upcoming_football_fixtures
from src.features import add_elo_features, add_football_form_features
from src.predictions import build_team_stats, predict_match

st.set_page_config(page_title="Sports Betting Predictions", page_icon="🏆", layout="wide")
st.title("🏆 Sports Betting Predictions")
st.caption("Fast football & tennis analytics with probability-based pre-match estimates")

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_all_football() -> pd.DataFrame:
    data = load_football_matches(source="football-data")
    try:
        fixtures = load_upcoming_football_fixtures()
        if not fixtures.empty:
            data = pd.concat([data, fixtures], ignore_index=True, sort=False)
    except Exception:
        pass
    return data.sort_values("date").reset_index(drop=True)

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_all_tennis() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_tennis_atp(), load_tennis_wta()

def safe_load(loader):
    try:
        return loader(), None
    except Exception as exc:
        return None, str(exc)

def split_football(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    completed = data.dropna(subset=["home_goals", "away_goals", "date"]).copy()
    upcoming = data[data["date"].notna() & data["home_goals"].isna() & data["away_goals"].isna()].copy()
    upcoming = upcoming[upcoming["date"] >= pd.Timestamp.now().normalize()]
    return completed, upcoming.drop_duplicates(subset=["date", "home_team", "away_team", "competition"])

def team_state(history: pd.DataFrame) -> dict[str, dict[str, float]]:
    ratings: dict[str, float] = {}
    for _, row in history.sort_values("date").iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        expected = 1 / (1 + 10 ** ((ra - (rh + 60)) / 400))
        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        actual = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        change = 32 * (actual - expected)
        ratings[home], ratings[away] = rh + change, ra - change
    return {team: {"elo": rating} for team, rating in ratings.items()}

@st.cache_data(ttl=30 * 60, show_spinner=False)
def prediction_state(completed: pd.DataFrame):
    return team_state(completed), build_team_stats(completed)

def build_predictions(upcoming: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    elo_state, stats = prediction_state(completed)
    rows = []
    for _, match in upcoming.sort_values("date").head(150).iterrows():
        home, away = str(match["home_team"]), str(match["away_team"])
        p = predict_match(home, away, elo_state, stats)
        confidence = max(p["ft_home"], p["ft_draw"], p["ft_away"])
        rows.append({
            "date": match["date"], "competition": match["competition"], "home": home, "away": away,
            "prediction": p["ft_pick"], "confidence": confidence,
            "ft_1": p["ft_home"], "ft_x": p["ft_draw"], "ft_2": p["ft_away"],
            "ht_pick": p["ht_pick"], "ht_1": p["ht_home"], "ht_x": p["ht_draw"], "ht_2": p["ht_away"],
            "score": f"{p['likely_home_goals']}-{p['likely_away_goals']}", "xg": f"{p['home_xg']:.1f}-{p['away_xg']:.1f}",
            "over15": p["over_1_5"], "over25": p["over_2_5"], "btts": p["btts"],
            "corners": p["expected_corners"], "over85c": p["over_8_5_corners"],
            "cards": p["expected_cards"], "over35cards": p["over_3_5_cards"],
        })
    return pd.DataFrame(rows)

with st.sidebar:
    sport = st.radio("Sport", ["Football", "Tennis"], key="sport")
    st.divider()
    st.caption("Football: Football-Data.co.uk + fixture feeds")
    st.caption("Tennis: Sackmann archive")
    st.divider()
    st.caption("Probabilities are model estimates, not bookmaker odds or guarantees.")

if sport == "Football":
    with st.spinner("Loading football data…"):
        data, error = safe_load(load_all_football)
    if data is None:
        st.error("Football data could not be loaded.")
        st.code(error)
        st.stop()

    filtered = filter_football_target(data)
    completed, upcoming = split_football(filtered)
    featured = add_football_form_features(completed) if not completed.empty else completed
    if not featured.empty:
        from src.features import add_elo_football_features
        featured = add_elo_football_features(featured)

    categories = sorted(set(featured.get("category", pd.Series(dtype=str)).dropna()) | set(upcoming.get("category", pd.Series(dtype=str)).dropna()))
    if not categories:
        st.warning("No supported football competitions are currently available.")
        st.stop()

    st.subheader("Upcoming matches")
    f1, f2, f3 = st.columns([1.3, 1.3, 1])
    category = f1.selectbox("Competition", ["All"] + categories, key="football_category")
    period = f2.selectbox("When", ["Today", "Tomorrow", "Next 3 days", "Next 7 days", "All upcoming"], key="football_period")
    min_conf = f3.slider("Minimum FT confidence", 0.34, 0.80, 0.45, 0.01, key="min_conf")

    upcoming_view = upcoming if category == "All" else upcoming[upcoming["category"] == category]
    today = pd.Timestamp.now().normalize()
    if period == "Today": upcoming_view = upcoming_view[upcoming_view["date"].dt.normalize() == today]
    elif period == "Tomorrow": upcoming_view = upcoming_view[upcoming_view["date"].dt.normalize() == today + pd.Timedelta(days=1)]
    elif period == "Next 3 days": upcoming_view = upcoming_view[upcoming_view["date"] < today + pd.Timedelta(days=3)]
    elif period == "Next 7 days": upcoming_view = upcoming_view[upcoming_view["date"] < today + pd.Timedelta(days=7)]

    predictions = build_predictions(upcoming_view, completed) if not upcoming_view.empty else pd.DataFrame()
    if not predictions.empty: predictions = predictions[predictions["confidence"] >= min_conf].sort_values(["date", "confidence"], ascending=[True, False])
    if predictions.empty:
        st.info("No upcoming matches match the selected filters.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matches", len(predictions))
    m2.metric("High-confidence", int((predictions["confidence"] >= 0.60).sum()))
    m3.metric("Best Over 2.5", f"{predictions['over25'].max():.0%}")
    m4.metric("Best BTTS", f"{predictions['btts'].max():.0%}")

    tab_top, tab_matches, tab_markets, tab_analytics = st.tabs(["🔥 Top tips", "📅 Matches", "🎯 Markets", "📊 Analytics"])
    with tab_top:
        top = predictions.sort_values("confidence", ascending=False).head(12).copy()
        top["Match"] = top["home"] + " — " + top["away"]
        top["FT"] = top["prediction"] + "  " + top["confidence"].map(lambda x: f"{x:.0%}")
        top["HT"] = top["ht_pick"]
        top["Score"] = top["score"]
        top["xG"] = top["xg"]
        top["O2.5"] = top["over25"].map(lambda x: f"{x:.0%}")
        top["BTTS"] = top["btts"].map(lambda x: f"{x:.0%}")
        top["Corners"] = top["corners"].map(lambda x: f"{x:.1f}")
        top["Cards"] = top["cards"].map(lambda x: f"{x:.1f}")
        st.dataframe(top[["date", "competition", "Match", "FT", "HT", "Score", "xG", "O2.5", "BTTS", "Corners", "Cards"]], width="stretch", hide_index=True)
        st.caption("UX direction inspired by Forebet and BetMines: surface the decision signals first, without presenting model probabilities as bookmaker odds.")

    with tab_matches:
        for _, row in predictions.head(30).iterrows():
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([2.4, 1.1, 1.1, 1.1])
                c1.markdown(f"**{row['home']}**  vs  **{row['away']}**")
                c1.caption(f"{row['date']:%d %b, %H:%M} · {row['competition']}")
                c2.metric("FT", f"{row['prediction']} · {row['confidence']:.0%}")
                c3.metric("Score", row["score"])
                c4.metric("xG", row["xg"])
                with st.expander("View markets"):
                    a, b, c, d = st.columns(4)
                    a.metric("FT 1 / X / 2", f"{row['ft_1']:.0%} / {row['ft_x']:.0%} / {row['ft_2']:.0%}")
                    b.metric("HT 1 / X / 2", f"{row['ht_1']:.0%} / {row['ht_x']:.0%} / {row['ht_2']:.0%}")
                    c.metric("O1.5 / O2.5", f"{row['over15']:.0%} / {row['over25']:.0%}")
                    d.metric("BTTS", f"{row['btts']:.0%}")
                    e, f, g, h = st.columns(4)
                    e.metric("Corners", f"{row['corners']:.1f}")
                    f.metric("O8.5 corners", f"{row['over85c']:.0%}")
                    g.metric("Cards", f"{row['cards']:.1f}")
                    h.metric("O3.5 cards", f"{row['over35cards']:.0%}")

    with tab_markets:
        market_labels = {"FT 1": "ft_1", "FT X": "ft_x", "FT 2": "ft_2", "HT 1": "ht_1", "HT X": "ht_x", "HT 2": "ht_2", "Over 1.5 goals": "over15", "Over 2.5 goals": "over25", "BTTS": "btts", "Over 8.5 corners": "over85c", "Over 3.5 cards": "over35cards"}
        selected = st.selectbox("Rank matches by market", list(market_labels), key="market_selector")
        col = market_labels[selected]
        market = predictions.sort_values(col, ascending=False).head(20).copy()
        market["Match"] = market["home"] + " — " + market["away"]
        market["Probability"] = market[col].map(lambda x: f"{x:.0%}")
        st.dataframe(market[["date", "competition", "Match", "Probability", "prediction", "score"]], width="stretch", hide_index=True)

    with tab_analytics:
        if featured.empty:
            st.info("No completed match history is available.")
        else:
            columns = [c for c in ["date", "competition", "home_team", "away_team", "home_goals", "away_goals", "home_elo", "away_elo", "elo_diff", "home_form_points", "away_form_points"] if c in featured.columns]
            st.dataframe(featured.sort_values("date", ascending=False)[columns].head(250), width="stretch", hide_index=True)

else:
    with st.spinner("Loading tennis data…"):
        tennis_data, error = safe_load(load_all_tennis)
    if tennis_data is None:
        st.error("Tennis data could not be loaded.")
        st.code(error)
        st.stop()
    atp, wta = tennis_data
    atp_target, wta_target = filter_tennis_target(atp, wta)
    atp_target["tour"], wta_target["tour"] = "ATP", "WTA"
    view = pd.concat([atp_target, wta_target], ignore_index=True)
    view["tourney_date"] = pd.to_datetime(view["tourney_date"], format="%Y%m%d", errors="coerce")
    view = add_elo_features(view).sort_values("tourney_date")
    tour = st.selectbox("Tour", ["All", "ATP", "WTA"], key="tennis_tour")
    category = st.selectbox("Tournament category", ["All"] + sorted(view["category"].dropna().unique()), key="tennis_category")
    filtered_view = view.copy()
    if tour != "All": filtered_view = filtered_view[filtered_view["tour"] == tour]
    if category != "All": filtered_view = filtered_view[filtered_view["category"] == category]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(filtered_view):,}")
    c2.metric("ATP", f"{len(filtered_view[filtered_view['tour'] == 'ATP']):,}")
    c3.metric("WTA", f"{len(filtered_view[filtered_view['tour'] == 'WTA']):,}")
    c4.metric("Players", f"{pd.unique(pd.concat([filtered_view['winner_name'], filtered_view['loser_name']])).size:,}")
    st.dataframe(filtered_view.sort_values("tourney_date", ascending=False).head(300), width="stretch", hide_index=True)

st.divider()
st.caption("Research and decision-support dashboard. Model probabilities are estimates, not guarantees of outcomes or profit.")