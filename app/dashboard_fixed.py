from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_filters import filter_football_target, filter_tennis_target
from src.data_loader import load_football_matches, load_tennis_atp, load_tennis_wta, load_upcoming_football_fixtures
from src.extra_football import load_poland_history, load_poland_upcoming
from src.predictions import build_team_stats, predict_match
from src.tennis_predictions import predict_tennis_match

st.set_page_config(page_title="BetLens", page_icon="⚡", layout="wide")


def fair_odds(prob: float) -> float:
    return round(1.0 / float(prob), 2) if prob and prob > 0 else 0.0


def pct(prob: float) -> str:
    return f"{float(prob):.0%}"


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def football_history() -> pd.DataFrame:
    data = load_football_matches(source="football-data")
    if data.empty:
        return data
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return filter_football_target(data).drop_duplicates(["date", "home_team", "away_team", "competition"])


@st.cache_data(ttl=15 * 60, show_spinner=False)
def football_upcoming() -> pd.DataFrame:
    data = load_upcoming_football_fixtures()
    if data.empty:
        return data
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return filter_football_target(data).drop_duplicates(["date", "home_team", "away_team", "competition"])


@st.cache_data(ttl=30 * 60, show_spinner=False)
def poland_bundle():
    history, upcoming = load_poland_history(), load_poland_upcoming()
    for frame in (history, upcoming):
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return history, upcoming


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def tennis_raw():
    return load_tennis_atp(), load_tennis_wta()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def football_state(completed: pd.DataFrame):
    if completed.empty:
        return {}, {}
    ratings: dict[str, float] = {}
    ordered = completed.sort_values("date")
    for r in ordered.itertuples(index=False):
        try:
            hg, ag = float(r.home_goals), float(r.away_goals)
        except (TypeError, ValueError):
            continue
        home, away = str(r.home_team), str(r.away_team)
        rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        expected = 1 / (1 + 10 ** ((ra - (rh + 55)) / 400))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        change = 30 * (actual - expected)
        ratings[home], ratings[away] = rh + change, ra - change
    return {k: {"elo": v} for k, v in ratings.items()}, build_team_stats(ordered)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def football_predictions(upcoming: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    if upcoming.empty:
        return pd.DataFrame()
    elo, stats = football_state(completed)
    rows = []
    for r in upcoming.sort_values("date").head(80).itertuples(index=False):
        p = predict_match(str(r.home_team), str(r.away_team), elo, stats)
        ft = {"1": p["ft_home"], "X": p["ft_draw"], "2": p["ft_away"]}
        pick = p["ft_pick"]
        markets = {
            f"FT {pick}": ft[pick],
            "O2.5": p["over_2_5"],
            "BTTS": p["btts"],
            "O8.5 corners": p["over_8_5_corners"],
            "O3.5 cards": p["over_3_5_cards"],
        }
        signal, signal_prob = max(markets.items(), key=lambda item: item[1])
        rows.append({
            "date": r.date, "competition": str(r.competition),
            "home": str(r.home_team), "away": str(r.away_team), "pick": pick,
            "confidence": p["confidence"], "score": f"{p['likely_home_goals']}-{p['likely_away_goals']}",
            "xg": f"{p['home_xg']:.1f}-{p['away_xg']:.1f}", "o25": p["over_2_5"],
            "btts": p["btts"], "signal": signal, "signal_prob": signal_prob,
            "fair_odds": fair_odds(signal_prob), "ft_home": p["ft_home"],
            "ft_draw": p["ft_draw"], "ft_away": p["ft_away"],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def tennis_target(tour: str) -> pd.DataFrame:
    atp, wta = tennis_raw()
    a, w = filter_tennis_target(atp, wta)
    return a if tour == "ATP" else w


@st.cache_data(ttl=30 * 60, show_spinner=False)
def tennis_players(tour: str) -> list[str]:
    data = tennis_target(tour)
    if data.empty:
        return []
    return sorted(set(data["winner_name"].dropna().astype(str)) | set(data["loser_name"].dropna().astype(str)))


st.markdown("# ⚡ BetLens")
st.caption("Strong model signals first. Probabilities and fair odds are model estimates, not bookmaker odds or guarantees.")
sport = st.sidebar.radio("Sport", ["Football", "Tennis"], horizontal=True)

if sport == "Football":
    with st.spinner("Loading fixtures…"):
        upcoming = football_upcoming()
        history = football_history()

    if upcoming.empty:
        st.warning("No upcoming football fixtures are available right now.")
        st.stop()

    if upcoming["competition"].astype(str).isin(["Liga 1", "Ekstraklasa"]).any():
        pol_history, pol_upcoming = poland_bundle()
        if not pol_upcoming.empty:
            upcoming = pd.concat([upcoming, pol_upcoming], ignore_index=True, sort=False)
        if not pol_history.empty:
            history = pd.concat([history, pol_history], ignore_index=True, sort=False)

    completed = history.dropna(subset=["home_goals", "away_goals"])
    now = pd.Timestamp.now().normalize()
    upcoming = upcoming[upcoming["home_goals"].isna() & upcoming["away_goals"].isna() & (upcoming["date"] >= now)].copy()

    c1, c2, c3 = st.columns(3)
    categories = sorted(upcoming.get("category", pd.Series(dtype=str)).dropna().astype(str).unique())
    category = c1.selectbox("Competition", ["All"] + categories)
    window = c2.selectbox("Time", ["Today", "Tomorrow", "Next 3 days", "Next 7 days", "All upcoming"], index=2)
    focus = c3.selectbox("Rank by", ["Best signal", "FT result", "Goals", "BTTS"])

    view = upcoming if category == "All" else upcoming[upcoming["category"].astype(str) == category]
    if window == "Today":
        view = view[view.date.dt.normalize() == now]
    elif window == "Tomorrow":
        view = view[view.date.dt.normalize() == now + pd.Timedelta(days=1)]
    elif window == "Next 3 days":
        view = view[view.date < now + pd.Timedelta(days=3)]
    elif window == "Next 7 days":
        view = view[view.date < now + pd.Timedelta(days=7)]

    pred = football_predictions(view, completed)
    if pred.empty:
        st.info("No matches fit the selected filters.")
        st.stop()

    rank = {"Best signal": "signal_prob", "FT result": "confidence", "Goals": "o25", "BTTS": "btts"}[focus]
    pred = pred.sort_values([rank, "date"], ascending=[False, True]).reset_index(drop=True)
    a, b, c, d = st.columns(4)
    a.metric("Matches", len(pred))
    b.metric("Signals ≥ 65%", int((pred.signal_prob >= 0.65).sum()))
    c.metric("Best signal", pct(pred.signal_prob.max()))
    d.metric("Best O2.5", pct(pred.o25.max()))

    st.subheader("🔥 Top signals")
    for idx, r in enumerate(pred.head(10).itertuples(index=False)):
        title = f"{'🔥 ' if idx < 3 else ''}{r.home} vs {r.away} · {r.signal} {pct(r.signal_prob)}"
        with st.expander(title, expanded=idx == 0):
            left, mid, right = st.columns([1.6, 1, 1])
            left.markdown(f"**{r.home}** vs **{r.away}**")
            left.caption(f"{r.date:%a, %d %b · %H:%M} · {r.competition}")
            left.write(f"Signal: **{r.signal} {pct(r.signal_prob)}** · Fair odds **{r.fair_odds:.2f}**")
            mid.metric("FT confidence", pct(r.confidence))
            mid.metric("Likely score", r.score)
            right.metric("O2.5", pct(r.o25))
            right.metric("BTTS", pct(r.btts))

    st.subheader("📋 All matches")
    st.dataframe(pred[["date", "competition", "home", "away", "signal", "signal_prob", "fair_odds", "pick", "confidence", "score", "xg", "o25", "btts"]], hide_index=True, width="stretch")

else:
    tour = st.radio("Tour", ["ATP", "WTA"], horizontal=True)
    data = tennis_target(tour)
    if data.empty:
        st.warning("Tennis data is currently unavailable.")
        st.stop()
    surface = st.selectbox("Surface", ["All", "Hard", "Clay", "Grass", "Carpet"])
    players = tennis_players(tour)
    if len(players) < 2:
        st.warning("Not enough players are available in the current dataset.")
        st.stop()
    p1, p2 = st.columns(2)
    player1 = p1.selectbox("Player 1", players)
    player2 = p2.selectbox("Player 2", players, index=1)
    if player1 == player2:
        st.warning("Choose two different players.")
        st.stop()
    with st.spinner("Calculating matchup…"):
        result = predict_tennis_match(data, player1, player2, surface=None if surface == "All" else surface)
    st.subheader(f"{player1} vs {player2}")
    a, b, c = st.columns(3)
    a.metric(player1, pct(result["a_win"]))
    b.metric(player2, pct(result["b_win"]))
    c.metric("Confidence", pct(result["confidence"]))
    st.success(f"Model pick: {result['pick']} · Fair odds {fair_odds(max(result['a_win'], result['b_win'])):.2f}")
    evidence = pd.DataFrame({"Metric": ["ELO", "Recent win rate", "Serve points won", "Return points won", "Aces", "Double faults"], player1: [result["elo_a"], result["form_a"], result["serve_a"], result["return_a"], result["aces_a"], result["double_faults_a"]], player2: [result["elo_b"], result["form_b"], result["serve_b"], result["return_b"], result["aces_b"], result["double_faults_b"]]})
    st.dataframe(evidence.round(3), hide_index=True, width="stretch")

st.divider()
st.caption("BetLens · public sports data · probabilistic estimates · no guarantee of outcome")
