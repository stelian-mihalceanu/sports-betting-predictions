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

st.markdown("""
<style>
.stApp{background:#07111f;color:#edf4fb}
.block-container{max-width:1500px;padding:1.1rem 2rem 2.5rem}
[data-testid="stSidebar"]{background:#091727;border-right:1px solid #1d334d}
.hero{padding:.1rem 0 .9rem}.eyebrow{color:#48e0aa;font-weight:800;letter-spacing:.16em;font-size:.68rem}
.hero h1{font-size:2.55rem;letter-spacing:-.05em;margin:.05rem 0}.hero p,.muted{color:#91a8c1}
.section-title{font-size:1.05rem;font-weight:800;margin:.7rem 0 .45rem}.subtle{color:#7f96af;font-size:.82rem}
.card{background:linear-gradient(180deg,#10243b,#0c1a2c);border:1px solid #1e3854;border-radius:16px;padding:15px 16px;margin:0 0 10px}
.card-hot{border-color:rgba(72,224,170,.42);box-shadow:0 8px 28px rgba(0,0,0,.16)}
.badge{display:inline-block;border:1px solid #27445f;background:#102941;color:#b9cce0;border-radius:999px;padding:4px 8px;font-size:.7rem;font-weight:750;margin:3px 4px 0 0}
.badge-hot{color:#48e0aa;border-color:rgba(72,224,170,.3);background:rgba(72,224,170,.07)}
[data-testid="stMetric"]{background:#0d1b2e;border:1px solid #1e3854;border-radius:14px;padding:10px 12px}
div[data-testid="stDataFrame"]{border:1px solid #1e3854;border-radius:12px;overflow:hidden}
div[data-testid="stExpander"]{border-color:#1e3854;background:#0b192a}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="eyebrow">SPORTS INTELLIGENCE</div><h1>⚡ BetLens</h1><p>Strongest model signals first. Probabilities and fair odds are estimates — not bookmaker odds or guarantees.</p></div>', unsafe_allow_html=True)


def _fair_odds(prob: float) -> float:
    return round(1.0 / prob, 2) if prob and prob > 0 else 0.0


def _pct(value: float) -> str:
    return f"{float(value):.0%}"


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
def poland_bundle() -> tuple[pd.DataFrame, pd.DataFrame]:
    history, upcoming = load_poland_history(), load_poland_upcoming()
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
    for r in upcoming.sort_values("date").head(80).itertuples(index=False):
        p = predict_match(str(r.home_team), str(r.away_team), elo, stats)
        ft_probs = {"1": p["ft_home"], "X": p["ft_draw"], "2": p["ft_away"]}
        ft_pick = p["ft_pick"]
        market_probs = {
            f"FT {ft_pick}": ft_probs[ft_pick],
            "O2.5": p["over_2_5"],
            "BTTS": p["btts"],
            "O8.5 corners": p["over_8_5_corners"],
            "O3.5 cards": p["over_3_5_cards"],
        }
        market, probability = max(market_probs.items(), key=lambda x: x[1])
        rows.append({
            "date": r.date, "competition": str(r.competition), "category": str(getattr(r, "category", "")),
            "home": str(r.home_team), "away": str(r.away_team), "pick": ft_pick,
            "confidence": p["confidence"], "score": f"{p['likely_home_goals']}-{p['likely_away_goals']}",
            "xg": f"{p['home_xg']:.1f}-{p['away_xg']:.1f}", "o25": p["over_2_5"], "btts": p["btts"],
            "corners": p["expected_corners"], "cards": p["expected_cards"], "signal": market,
            "signal_prob": probability, "fair_odds": _fair_odds(probability),
            "ft_home": p["ft_home"], "ft_draw": p["ft_draw"], "ft_away": p["ft_away"],
            "over_15": p["over_1_5"], "under_25": p["under_2_5"], "btts_no": p["btts_no"],
            "o85_corners": p["over_8_5_corners"], "o35_cards": p["over_3_5_cards"],
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


with st.sidebar:
    sport = st.radio("Sport", ["Football", "Tennis"], horizontal=True)
    st.divider()
    if sport == "Football":
        st.caption("Fixtures refresh every 15 min · history every 24 h")
    else:
        st.caption("Tennis data refreshes every 24 h")
    st.caption("Use model probability for research. Fair odds = 1 / probability.")

if sport == "Football":
    with st.spinner("Loading the latest fixtures…"):
        upcoming = football_upcoming()
        history = football_history()

    if upcoming.empty:
        st.warning("No upcoming football fixtures are available from the public feeds right now.")
        st.stop()

    pol_history = pd.DataFrame()
    has_poland = upcoming["competition"].astype(str).isin(["Liga 1", "Ekstraklasa"]).any()
    if has_poland:
        pol_history, pol_upcoming = poland_bundle()
        if not pol_upcoming.empty:
            upcoming = pd.concat([upcoming, pol_upcoming], ignore_index=True, sort=False)

    completed = history.dropna(subset=["home_goals", "away_goals"])
    if not pol_history.empty:
        completed = pd.concat([completed, pol_history.dropna(subset=["home_goals", "away_goals"])], ignore_index=True, sort=False)
    now = pd.Timestamp.now().normalize()
    upcoming = upcoming[upcoming["home_goals"].isna() & upcoming["away_goals"].isna() & (upcoming["date"] >= now)].copy()

    competitions = sorted(upcoming.get("category", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    c1, c2, c3 = st.columns([1.25, 1.05, 1.15])
    category = c1.selectbox("Competition", ["All"] + competitions)
    window = c2.selectbox("Time", ["Today", "Tomorrow", "Next 3 days", "Next 7 days", "All upcoming"], index=2)
    focus = c3.selectbox("Rank by", ["Best signal", "FT result", "Goals", "BTTS", "Corners", "Cards"])

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

    rank_col = {"Best signal": "signal_prob", "FT result": "confidence", "Goals": "o25", "BTTS": "btts", "Corners": "corners", "Cards": "cards"}[focus]
    pred = pred.sort_values([rank_col, "date"], ascending=[False, True]).reset_index(drop=True)

    st.markdown('<div class="section-title">Today at a glance</div>', unsafe_allow_html=True)
    a, b, c, d, e = st.columns(5)
    a.metric("Matches", len(pred))
    b.metric("Signals ≥ 65%", int((pred.signal_prob >= .65).sum()))
    c.metric("Best signal", _pct(pred.signal_prob.max()))
    d.metric("Best O2.5", _pct(pred.o25.max()))
    e.metric("Best BTTS", _pct(pred.btts.max()))

    st.markdown('<div class="section-title">🔥 Top signals</div><div class="subtle">Ranked by the selected focus. Open a match to see the model evidence and fair odds.</div>', unsafe_allow_html=True)
    for idx, r in enumerate(pred.head(10).itertuples(index=False)):
        with st.expander(f"{'🔥 ' if idx < 3 else ''}{r.home}  vs  {r.away}   ·   {r.signal} {_pct(r.signal_prob)}", expanded=idx == 0):
            left, mid, right = st.columns([1.5, 1.2, 1.0])
            left.markdown(f"**{r.home}** vs **{r.away}**  
{r.date:%a, %d %b · %H:%M} · {r.competition}")
            left.markdown(f'<span class="badge badge-hot">Signal: {r.signal} {_pct(r.signal_prob)}</span><span class="badge">Fair odds {r.fair_odds:.2f}</span><span class="badge">FT {r.pick}</span><span class="badge">xG {r.xg}</span>', unsafe_allow_html=True)
            mid.metric("FT confidence", _pct(r.confidence))
            mid.metric("Likely score", r.score)
            right.metric("O2.5", _pct(r.o25))
            right.metric("BTTS", _pct(r.btts))
            st.markdown("**Why this signal?**")
            evidence = pd.DataFrame([
                ["FT 1", r.ft_home, _fair_odds(r.ft_home)], ["FT X", r.ft_draw, _fair_odds(r.ft_draw)], ["FT 2", r.ft_away, _fair_odds(r.ft_away)],
                ["Over 1.5", r.over_15, _fair_odds(r.over_15)], ["Over 2.5", r.o25, _fair_odds(r.o25)], ["BTTS", r.btts, _fair_odds(r.btts)],
                ["Over 8.5 corners", r.o85_corners, _fair_odds(r.o85_corners)], ["Over 3.5 cards", r.o35_cards, _fair_odds(r.o35_cards)],
            ], columns=["Market", "Model probability", "Fair odds"])
            st.dataframe(evidence.style.format({"Model probability": "{:.1%}", "Fair odds": "{:.2f}"}), hide_index=True, width="stretch")

    st.markdown('<div class="section-title">📋 All matches</div>', unsafe_allow_html=True)
    table = pred[["date", "competition", "home", "away", "signal", "signal_prob", "fair_odds", "pick", "confidence", "score", "xg", "o25", "btts"]].rename(columns={
        "date":"Date", "competition":"Competition", "home":"Home", "away":"Away", "signal":"Top signal", "signal_prob":"Probability", "fair_odds":"Fair odds", "pick":"FT", "confidence":"FT conf.", "score":"Score", "xg":"xG", "o25":"O2.5", "btts":"BTTS"
    })
    st.dataframe(table.style.format({"Probability":"{:.1%}", "Fair odds":"{:.2f}", "FT conf.":"{:.1%}", "O2.5":"{:.1%}", "BTTS":"{:.1%}"}), hide_index=True, width="stretch")

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
        r = predict_tennis_match(data, player1, player2, surface=None if surface == "All" else surface)

    st.markdown(f'<div class="card card-hot"><b>{r["pick"]}</b> has the stronger model probability · {_pct(r["confidence"])} confidence · fair odds <b>{_fair_odds(max(r["a_win"], r["b_win"])):.2f}</b></div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric(player1, _pct(r["a_win"]))
    b.metric(player2, _pct(r["b_win"]))
    c.metric("Confidence", _pct(r["confidence"]))
    d.metric("H2H", f'{r["h2h_a"]}–{r["h2h_b"]}')

    st.markdown('<div class="section-title">Why the model leans this way</div>', unsafe_allow_html=True)
    evidence = pd.DataFrame({"Metric":["ELO","Recent win rate","Serve points won","Return points won","Aces","Double faults"], player1:[r["elo_a"],r["form_a"],r["serve_a"],r["return_a"],r["aces_a"],r["double_faults_a"]], player2:[r["elo_b"],r["form_b"],r["serve_b"],r["return_b"],r["aces_b"],r["double_faults_b"]]})
    st.dataframe(evidence.round(3), hide_index=True, width="stretch")

st.divider()
st.caption("BetLens · public sports data · transparent probabilistic estimates · no guarantee of outcome")
