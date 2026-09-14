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

st.set_page_config(page_title="BetLens", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
.stApp{background:#07111f;color:#ecf3fb}
.block-container{max-width:1450px;padding:1.15rem 2rem 2rem}
[data-testid="stSidebar"]{background:#091727;border-right:1px solid #1d334d}
.hero{padding:.15rem 0 .8rem}.eyebrow{color:#42e6ab;font-weight:800;letter-spacing:.16em;font-size:.72rem}
.hero h1{font-size:2.45rem;letter-spacing:-.045em;margin:.08rem 0}.hero p,.muted{color:#93a9c2}
.card{background:linear-gradient(180deg,#10233b,#0d1b2e);border:1px solid #1e3652;border-radius:16px;padding:16px;margin-bottom:10px}
.badge{display:inline-block;border:1px solid rgba(66,230,171,.22);background:rgba(66,230,171,.08);color:#42e6ab;border-radius:999px;padding:4px 8px;font-size:.72rem;font-weight:800;margin:3px 4px 0 0}
[data-testid="stMetric"]{background:#0d1b2e;border:1px solid #1e3652;border-radius:14px;padding:10px 12px}
div[data-testid="stDataFrame"]{border:1px solid #1e3652;border-radius:12px;overflow:hidden}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><div class="eyebrow">SPORTS INTELLIGENCE</div><h1>⚡ BetLens</h1><p>Fast football & tennis matchup research — strongest signals first, evidence behind every estimate.</p></div>',
    unsafe_allow_html=True,
)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def football_history() -> pd.DataFrame:
    data = load_football_matches(source="football-data")
    if data.empty:
        return data
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return filter_football_target(data).drop_duplicates(["date", "home_team", "away_team", "competition"])


@st.cache_data(ttl=15 * 60, show_spinner=False)
def football_upcoming() -> pd.DataFrame:
    data = load_upcoming_football_fixtures()
    if data.empty:
        return data
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return filter_football_target(data).drop_duplicates(["date", "home_team", "away_team", "competition"])


@st.cache_data(ttl=30 * 60, show_spinner=False)
def poland_bundle() -> tuple[pd.DataFrame, pd.DataFrame]:
    history = load_poland_history()
    upcoming = load_poland_upcoming()
    for frame in (history, upcoming):
        if not frame.empty:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return history, upcoming


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def tennis_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_tennis_atp(), load_tennis_wta()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def football_state(completed: pd.DataFrame):
    ratings: dict[str, float] = {}
    if completed.empty:
        return {}, {}
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
    for r in upcoming.sort_values("date").head(120).itertuples(index=False):
        p = predict_match(str(r.home_team), str(r.away_team), elo, stats)
        rows.append({
            "date": r.date,
            "competition": r.competition,
            "home": str(r.home_team),
            "away": str(r.away_team),
            "pick": p["ft_pick"],
            "confidence": p["confidence"],
            "score": f"{p['likely_home_goals']}-{p['likely_away_goals']}",
            "xg": f"{p['home_xg']:.1f}-{p['away_xg']:.1f}",
            "o25": p["over_2_5"],
            "btts": p["btts"],
            "corners": p["expected_corners"],
            "cards": p["expected_cards"],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def tennis_target(tour: str) -> pd.DataFrame:
    atp, wta = tennis_raw()
    if atp.empty and wta.empty:
        return pd.DataFrame()
    atp_target, wta_target = filter_tennis_target(atp, wta)
    return atp_target if tour == "ATP" else wta_target


@st.cache_data(ttl=30 * 60, show_spinner=False)
def tennis_players(tour: str) -> list[str]:
    data = tennis_target(tour)
    if data.empty:
        return []
    return sorted(set(data["winner_name"].dropna().astype(str)) | set(data["loser_name"].dropna().astype(str)))

with st.sidebar:
    sport = st.radio("Sport", ["Football", "Tennis"], horizontal=True)
    st.divider()
    st.caption("Fixtures: 15 min · history: 24 h · tennis: 24 h")
    st.caption("Model outputs are estimates, not bookmaker odds or guarantees.")

if sport == "Football":
    with st.spinner("Loading football…"):
        upcoming = football_upcoming()
        history = football_history()

    if upcoming.empty:
        st.warning("No upcoming football fixtures are available right now.")
        st.stop()

    # Extra Poland data is only requested when the current fixture set needs it.
    pol_history = pd.DataFrame()
    has_poland = upcoming["competition"].astype(str).eq("Liga 1").any() | upcoming["competition"].astype(str).eq("Ekstraklasa").any()
    if has_poland:
        pol_history, pol_upcoming = poland_bundle()
        if not pol_upcoming.empty:
            upcoming = pd.concat([upcoming, pol_upcoming], ignore_index=True, sort=False)

    completed = history.dropna(subset=["home_goals", "away_goals"])
    if not pol_history.empty:
        completed = pd.concat([completed, pol_history.dropna(subset=["home_goals", "away_goals"])], ignore_index=True, sort=False)

    now = pd.Timestamp.now().normalize()
    upcoming = upcoming[upcoming["home_goals"].isna() & upcoming["away_goals"].isna() & (upcoming["date"] >= now)].copy()
    cats = sorted(upcoming.get("category", pd.Series(dtype=str)).dropna().unique().tolist())
    if not cats:
        st.info("No matches fit the current public data feeds.")
        st.stop()

    c1, c2, c3 = st.columns([1.4, 1.15, 1.15])
    category = c1.selectbox("Competition", ["All"] + cats)
    window = c2.selectbox("Window", ["Today", "Tomorrow", "Next 3 days", "Next 7 days", "All upcoming"])
    focus = c3.selectbox("Focus", ["FT result", "Goals", "BTTS", "Corners", "Cards"])

    view = upcoming if category == "All" else upcoming[upcoming["category"] == category]
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

    rank_col = {"Goals": "o25", "BTTS": "btts", "Corners": "corners", "Cards": "cards", "FT result": "confidence"}[focus]
    pred["rank"] = pred[rank_col]
    pred = pred.sort_values(["rank", "date"], ascending=[False, True]).reset_index(drop=True)

    xg_split = pred["xg"].str.split("-", n=1, expand=True).astype(float)
    avg_xg = xg_split.sum(axis=1).mean()
    a, b, c, d, e = st.columns(5)
    a.metric("Matches", len(pred))
    b.metric("High confidence", int((pred.confidence >= 0.60).sum()))
    c.metric("Best O2.5", f"{pred.o25.max():.0%}")
    d.metric("Best BTTS", f"{pred.btts.max():.0%}")
    e.metric("Avg xG", f"{avg_xg:.1f}")

    t1, t2 = st.tabs(["🔥 Top signals", "📋 All matches"])
    with t1:
        for r in pred.head(12).itertuples(index=False):
            st.markdown(
                f'<div class="card"><b>{r.home}</b> <span class="muted">vs</span> <b>{r.away}</b>'
                f'<div class="muted">{r.date:%a, %d %b %H:%M} · {r.competition}</div>'
                f'<div style="margin-top:8px"><span class="badge">FT {r.pick}</span>'
                f'<span class="badge">{r.confidence:.0%} confidence</span><span class="badge">Score {r.score}</span>'
                f'<span class="badge">xG {r.xg}</span><span class="badge">O2.5 {r.o25:.0%}</span><span class="badge">BTTS {r.btts:.0%}</span></div></div>',
                unsafe_allow_html=True,
            )
    with t2:
        table = pred[["date", "competition", "home", "away", "pick", "confidence", "score", "xg", "o25", "btts", "corners", "cards"]].rename(
            columns={"home": "Home", "away": "Away", "pick": "FT", "confidence": "Confidence", "score": "Score", "xg": "xG", "o25": "O2.5", "btts": "BTTS", "corners": "Corners", "cards": "Cards"}
        )
        st.dataframe(table, hide_index=True, width="stretch")

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
        st.warning("Choose different players.")
        st.stop()

    with st.spinner("Calculating matchup…"):
        r = predict_tennis_match(data, player1, player2, surface=None if surface == "All" else surface)

    a, b, c, d = st.columns(4)
    a.metric(player1, f"{r['a_win']:.1%}")
    b.metric(player2, f"{r['b_win']:.1%}")
    c.metric("Confidence", f"{r['confidence']:.1%}")
    d.metric("H2H", f"{r['h2h_a']}–{r['h2h_b']}")

    evidence = pd.DataFrame({
        "Metric": ["ELO", "Recent form", "Serve points won", "Return points won", "Aces", "Double faults"],
        player1: [r["elo_a"], r["form_a"], r["serve_a"], r["return_a"], r["aces_a"], r["double_faults_a"]],
        player2: [r["elo_b"], r["form_b"], r["serve_b"], r["return_b"], r["aces_b"], r["double_faults_b"]],
    })
    st.dataframe(evidence.round(3), hide_index=True, width="stretch")

st.divider()
st.caption("BetLens · public sports data · transparent probabilistic estimates")
