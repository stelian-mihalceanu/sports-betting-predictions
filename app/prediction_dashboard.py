from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtesting import football_backtest
from src.data_loader import load_football_history, load_upcoming_football_fixtures, load_tennis_atp, load_tennis_wta
from src.predictions import build_team_stats, predict_match
from src.tennis_predictions import predict_tennis_match

st.set_page_config(page_title="BetLens", page_icon="🎯", layout="wide", initial_sidebar_state="expanded")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _build_elo(history: pd.DataFrame) -> dict[str, dict[str, float]]:
    ratings: dict[str, float] = {}
    for row in history.sort_values("date").itertuples(index=False):
        home, away = str(row.home_team), str(row.away_team)
        rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        expected = 1.0 / (1.0 + 10 ** ((ra - (rh + 60.0)) / 400.0))
        hg, ag = float(row.home_goals), float(row.away_goals)
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        change = 24.0 * (actual - expected)
        ratings[home], ratings[away] = rh + change, ra - change
    return {team: {"elo": rating} for team, rating in ratings.items()}


@st.cache_data(ttl=86400, show_spinner=False)
def prepared_football() -> tuple[pd.DataFrame, dict, dict]:
    history = load_football_history()
    if history.empty:
        return history, {}, {}
    history = history.dropna(subset=["home_goals", "away_goals"]).copy()
    stats = build_team_stats(history)
    elo = _build_elo(history)
    return history, stats, elo


@st.cache_data(ttl=900, show_spinner=False)
def upcoming_predictions() -> pd.DataFrame:
    history, stats, elo = prepared_football()
    fixtures = load_upcoming_football_fixtures()
    if fixtures.empty or history.empty:
        return fixtures
    rows = []
    for r in fixtures.itertuples(index=False):
        p = predict_match(str(r.home_team), str(r.away_team), elo, stats)
        rows.append({
            "date": r.date, "competition": r.competition, "home_team": r.home_team, "away_team": r.away_team,
            "pick": p["ft_pick"], "confidence": p["confidence"], "home_xg": p["home_xg"], "away_xg": p["away_xg"],
            "over_2_5": p["over_2_5"], "btts": p["btts"], "corners": p["expected_corners"], "cards": p["expected_cards"],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_backtest() -> dict:
    history, _, _ = prepared_football()
    return football_backtest(history, max_matches=350) if not history.empty else {"matches": 0, "accuracy": math.nan, "brier": math.nan, "log_loss": math.nan}


def football_page() -> None:
    st.title("🎯 Football Signals")
    st.caption("Pre-match probabilities from recency-weighted scoring, home/away context, Elo and Poisson markets.")
    with st.spinner("Loading football data…"):
        predictions = upcoming_predictions()
    if predictions.empty:
        st.warning("No upcoming football fixtures are available from the configured public feeds.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Fixtures", len(predictions))
    c2.metric("High confidence", int((predictions.confidence >= 0.65).sum()))
    c3.metric("Average confidence", _pct(float(predictions.confidence.mean())))
    competitions = ["All"] + sorted(predictions.competition.dropna().unique().tolist())
    selected = st.selectbox("Competition", competitions)
    min_conf = st.slider("Minimum confidence", 0.50, 0.85, 0.55, 0.01)
    view = predictions[predictions.confidence >= min_conf].copy()
    if selected != "All":
        view = view[view.competition == selected]
    view = view.sort_values(["confidence", "date"], ascending=[False, True])
    st.subheader("Top signals")
    for row in view.head(20).itertuples(index=False):
        with st.container(border=True):
            a, b, c, d = st.columns([2.8, 1.2, 1.3, 1.4])
            a.markdown(f"**{row.home_team} — {row.away_team}**")
            a.caption(f"{row.competition} · {pd.Timestamp(row.date):%d %b %H:%M}")
            b.metric("Pick", row.pick)
            c.metric("Confidence", _pct(row.confidence))
            d.metric("xG", f"{row.home_xg:.2f} – {row.away_xg:.2f}")
            st.caption(f"O2.5 {_pct(row.over_2_5)} · BTTS {_pct(row.btts)} · corners {row.corners:.1f} · cards {row.cards:.1f}")


def match_center_page() -> None:
    st.title("⚽ Match Center")
    history, stats, elo = prepared_football()
    if history.empty:
        st.warning("Historical football data is unavailable.")
        return
    teams = sorted(set(history.home_team) | set(history.away_team))
    left, right = st.columns(2)
    home = left.selectbox("Home team", teams, index=0)
    away = right.selectbox("Away team", teams, index=min(1, len(teams) - 1))
    if home == away:
        st.info("Select two different teams.")
        return
    p = predict_match(home, away, elo, stats)
    st.subheader(f"{home} vs {away}")
    cols = st.columns(5)
    cols[0].metric("FT pick", p["ft_pick"])
    cols[1].metric("Confidence", _pct(p["confidence"]))
    cols[2].metric("xG", f'{p["home_xg"]:.2f} – {p["away_xg"]:.2f}')
    cols[3].metric("O2.5", _pct(p["over_2_5"]))
    cols[4].metric("BTTS", _pct(p["btts"]))
    st.write(pd.DataFrame({"Market": ["Home", "Draw", "Away", "1X", "X2", "12"], "Probability": [p["ft_home"], p["ft_draw"], p["ft_away"], p["double_1x"], p["double_x2"], p["double_12"]]}).assign(Probability=lambda x: x.Probability.map(_pct)))
    st.caption(f'Likely score: {p["likely_home_goals"]} – {p["likely_away_goals"]}')


def tennis_page() -> None:
    st.title("🎾 Tennis Match Lab")
    atp, wta = load_tennis_atp(), load_tennis_wta()
    tour = st.radio("Tour", ["ATP", "WTA"], horizontal=True)
    frame = atp if tour == "ATP" else wta
    if frame.empty:
        st.info("Add the public ATP/WTA archive to data/raw/ to enable the Tennis Match Lab.")
        return
    players = sorted(set(frame.winner_name.dropna()) | set(frame.loser_name.dropna()))
    a, b = st.columns(2)
    player_a = a.selectbox("Player A", players)
    player_b = b.selectbox("Player B", players, index=min(1, len(players) - 1))
    surface = st.selectbox("Surface", ["All", "Hard", "Clay", "Grass", "Carpet"])
    if player_a == player_b:
        st.info("Select two different players.")
        return
    with st.spinner("Calculating matchup…"):
        p = predict_tennis_match(frame, player_a, player_b, surface)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pick", p["pick"])
    c2.metric("Win probability", _pct(p["a_win"] if p["pick"] == player_a else p["b_win"]))
    c3.metric("Confidence", _pct(p["confidence"]))
    st.write(pd.DataFrame({"Metric": ["Elo", "Recent win rate", "Serve points won", "Return points won", "H2H"], player_a: [round(p["elo_a"]), _pct(p["form_a"]), _pct(p["serve_a"]), _pct(p["return_a"]), p["h2h_a"]], player_b: [round(p["elo_b"]), _pct(p["form_b"]), _pct(p["serve_b"]), _pct(p["return_b"]), p["h2h_b"]]}))


def basketball_page() -> None:
    st.title("🏀 Basketball Baseline")
    st.caption("Transparent pace + net-rating + Elo baseline. Use as a research signal until a full league dataset is connected.")
    st.info("Basketball prediction inputs are intentionally explicit here; the next data pipeline can populate them automatically from NBA/EuroLeague feeds.")
    home_net = st.number_input("Home net rating", value=3.0, step=0.1)
    away_net = st.number_input("Away net rating", value=0.0, step=0.1)
    home_elo = st.number_input("Home Elo", value=1500.0, step=10.0)
    away_elo = st.number_input("Away Elo", value=1500.0, step=10.0)
    pace = st.number_input("Expected pace", value=99.0, step=0.5)
    rest = st.slider("Home rest-day edge", -2, 2, 0)
    edge = 0.055 * (home_elo - away_elo) + 2.2 * (home_net - away_net) + 1.5 * rest + 55.0
    p_home = 1.0 / (1.0 + math.exp(-edge / 100.0))
    c1, c2, c3 = st.columns(3)
    c1.metric("Moneyline", "Home" if p_home >= 0.5 else "Away")
    c2.metric("Home win", _pct(p_home))
    c3.metric("Expected total", f"{pace * 1.02:.0f}")


def analytics_page() -> None:
    st.title("📈 Model validation")
    with st.spinner("Running rolling backtest…"):
        result = cached_backtest()
    if result["matches"] == 0:
        st.warning("Not enough historical data for validation.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", result["matches"])
    c2.metric("1X2 accuracy", _pct(result["accuracy"]))
    c3.metric("Brier score", f'{result["brier"]:.3f}')
    c4.metric("Log loss", f'{result["log_loss"]:.3f}')
    st.caption("Backtest is chronological and evaluates each match using information available before that match; it is not a guarantee of future betting returns.")


def run_app() -> None:
    with st.sidebar:
        st.title("BetLens")
        st.caption("Research-first sports analytics")
        page = st.radio("Navigate", ["Football Signals", "Match Center", "Tennis", "Basketball", "Model Validation"])
        st.divider()
        st.caption("Probabilities are estimates, not guarantees or bookmaker odds.")
    if page == "Football Signals": football_page()
    elif page == "Match Center": match_center_page()
    elif page == "Tennis": tennis_page()
    elif page == "Basketball": basketball_page()
    else: analytics_page()
