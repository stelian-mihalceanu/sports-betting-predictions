from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_filters import filter_football_target
from src.data_loader import load_football_matches, load_upcoming_football_fixtures
from src.extra_football import load_poland_history, load_poland_upcoming
from src.football_advanced import goals_distribution, h2h, team_strength, ensemble_prediction, value_edge
from src.predictions import build_team_stats, predict_match

st.set_page_config(page_title="Football Match Center", page_icon="⚽", layout="wide")
st.title("⚽ Football Match Center")
st.caption("Deep-dive one match without reloading the entire model on every click.")

@st.cache_data(ttl=15 * 60, show_spinner=False)
def load_upcoming() -> pd.DataFrame:
    frames = [load_upcoming_football_fixtures()]
    try:
        p = load_poland_upcoming()
        if not p.empty:
            frames.append(p)
    except Exception:
        pass
    data = pd.concat([x for x in frames if not x.empty], ignore_index=True, sort=False) if any(not x.empty for x in frames) else pd.DataFrame()
    if data.empty:
        return data
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return filter_football_target(data).drop_duplicates(["date", "home_team", "away_team", "competition"])

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_history() -> pd.DataFrame:
    data = load_football_matches(source="football-data")
    if data.empty:
        return data
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return filter_football_target(data).drop_duplicates(["date", "home_team", "away_team", "competition"])

@st.cache_data(ttl=30 * 60, show_spinner=False)
def poland_history() -> pd.DataFrame:
    try:
        data = load_poland_history()
        if not data.empty:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60 * 60, show_spinner=False)
def model_state(completed: pd.DataFrame):
    ratings: dict[str, float] = {}
    ordered = completed.sort_values("date")
    for r in ordered.itertuples(index=False):
        try:
            hg, ag = float(r.home_goals), float(r.away_goals)
        except (TypeError, ValueError):
            continue
        h, a = str(r.home_team), str(r.away_team)
        rh, ra = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
        expected = 1 / (1 + 10 ** ((ra - (rh + 55)) / 400))
        actual = 1.0 if hg > ag else .5 if hg == ag else 0.0
        change = 30 * (actual - expected)
        ratings[h], ratings[a] = rh + change, ra - change
    return {k: {"elo": v} for k, v in ratings.items()}, build_team_stats(ordered)

with st.spinner("Loading match list…"):
    upcoming = load_upcoming()
if upcoming.empty:
    st.warning("No upcoming matches are available from the public feeds.")
    st.stop()

now = pd.Timestamp.now().normalize()
upcoming = upcoming[upcoming["home_goals"].isna() & upcoming["away_goals"].isna() & (upcoming["date"] >= now)].sort_values("date").head(150).reset_index(drop=True)
labels = [f"{r.home_team} — {r.away_team} · {r.date:%d %b %H:%M} · {r.competition}" for r in upcoming.itertuples(index=False)]
selected = upcoming.iloc[st.selectbox("Choose a match", range(len(labels)), format_func=lambda i: labels[i])]
home, away = str(selected.home_team), str(selected.away_team)

with st.spinner("Loading model context…"):
    history = load_history()
    completed = history.dropna(subset=["home_goals", "away_goals"]).copy()
    if str(selected.competition) in {"Liga 1", "Ekstraklasa"}:
        ph = poland_history()
        if not ph.empty:
            completed = pd.concat([completed, ph.dropna(subset=["home_goals", "away_goals"])], ignore_index=True, sort=False)
    elo, stats = model_state(completed)

base = predict_match(home, away, elo, stats)
home_strength = team_strength(completed, home)
away_strength = team_strength(completed, away)
hh = h2h(completed, home, away)
ensemble = ensemble_prediction(base, home_strength, away_strength, hh, elo.get(home, {"elo": 1500})["elo"], elo.get(away, {"elo": 1500})["elo"])
ft1, ftx, ft2 = ensemble["ft_home"], ensemble["ft_draw"], ensemble["ft_away"]
pick = "1" if ft1 >= max(ftx, ft2) else "X" if ftx >= ft2 else "2"

st.markdown(f"### {home}  vs  {away}")
st.caption(f"{selected.date:%A, %d %B %Y · %H:%M} · {selected.competition}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ensemble pick", pick)
c2.metric("Confidence", f"{ensemble['confidence']:.0%}")
c3.metric("Model xG", f"{base['home_xg']:.1f} — {base['away_xg']:.1f}")
c4.metric("Likely score", f"{base['likely_home_goals']} — {base['likely_away_goals']}")
p1, p2, p3 = st.columns(3)
p1.metric("FT 1", f"{ft1:.1%}"); p2.metric("FT X", f"{ftx:.1%}"); p3.metric("FT 2", f"{ft2:.1%}")

st.divider()
tab_detail, tab_strength, tab_distribution, tab_value = st.tabs(["🔎 Match detail", "💪 Team strength", "🎯 Goals distribution", "💰 Value scanner"])

with tab_detail:
    d1, d2 = st.columns(2)
    for col, team, title in [(d1, home, "Last 5 — home team"), (d2, away, "Last 5 — away team")]:
        with col:
            st.subheader(title)
            recent = completed[(completed.home_team == team) | (completed.away_team == team)].sort_values("date", ascending=False).head(5).copy()
            recent["Result"] = recent.apply(lambda r: ("W" if (r.home_team == team and r.home_goals > r.away_goals) or (r.away_team == team and r.away_goals > r.home_goals) else "D" if r.home_goals == r.away_goals else "L"), axis=1)
            recent["Match"] = recent["home_team"] + " — " + recent["away_team"]
            recent["Score"] = recent["home_goals"].astype(int).astype(str) + "-" + recent["away_goals"].astype(int).astype(str)
            st.dataframe(recent[["date", "Match", "Score", "Result"]], hide_index=True, width="stretch")
    st.subheader("H2H")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Matches", hh["matches"]); h2.metric(f"{home} wins", hh["home_wins"]); h3.metric("Draws", hh["draws"]); h4.metric(f"{away} wins", hh["away_wins"])
    if hh["rows"]:
        st.dataframe(pd.DataFrame(hh["rows"]), hide_index=True, width="stretch")
    st.subheader("Key model markets")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("O1.5", f"{base['over_1_5']:.0%}"); k2.metric("O2.5", f"{base['over_2_5']:.0%}"); k3.metric("BTTS", f"{base['btts']:.0%}"); k4.metric("Corners", f"{base['expected_corners']:.1f}"); k5.metric("Cards", f"{base['expected_cards']:.1f}")

with tab_strength:
    strength = pd.DataFrame([
        {"Metric":"Attack", home:home_strength["attack"], away:away_strength["attack"]},
        {"Metric":"Defense conceded", home:home_strength["defense"], away:away_strength["defense"]},
        {"Metric":"Recent form", home:home_strength["form"], away:away_strength["form"]},
        {"Metric":"Matches", home:home_strength["matches"], away:away_strength["matches"]},
        {"Metric":"ELO", home:elo.get(home,{"elo":1500})["elo"], away:elo.get(away,{"elo":1500})["elo"]},
    ])
    st.dataframe(strength.round(2), hide_index=True, width="stretch")
    comp = ensemble["components"]
    st.subheader("Ensemble components")
    st.dataframe(pd.DataFrame({"Component":["Poisson","ELO","Form","Market prior"],"1":[x[0] for x in comp.values()],"X":[x[1] for x in comp.values()],"2":[x[2] for x in comp.values()]}).round(3), hide_index=True, width="stretch")

with tab_distribution:
    dist = goals_distribution(float(base["home_xg"]), float(base["away_xg"])).head(20).copy()
    dist["Score"] = dist["home_goals"].astype(str) + "-" + dist["away_goals"].astype(str)
    dist["Probability"] = dist["probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(dist[["Score","Probability"]], hide_index=True, width="stretch")

with tab_value:
    st.subheader("Model probability vs bookmaker price")
    market = st.selectbox("Market", ["FT 1","FT X","FT 2","Over 2.5","BTTS"])
    model_prob = {"FT 1":ft1,"FT X":ftx,"FT 2":ft2,"Over 2.5":base["over_2_5"],"BTTS":base["btts"]}[market]
    bookmaker = st.selectbox("Bookmaker", ["Superbet","Betano","Unibet","MrBit","Other"])
    odds = st.number_input("Decimal odds", min_value=1.01, value=2.00, step=0.01)
    val = value_edge(model_prob, odds)
    a, b, c, d = st.columns(4)
    a.metric("Model", f"{model_prob:.1%}"); b.metric("Implied", f"{val['implied_probability']:.1%}"); c.metric("Edge", f"{val['edge']:+.1%}"); d.metric("Fair odds", f"{val['fair_odds']:.2f}")
    if val["edge"] >= .03:
        st.success(f"Model edge above 3% for {market} at {odds:.2f} ({bookmaker}).")
    else:
        st.info("No model edge above 3% at the entered price.")
    st.caption(f"Expected value: {val['expected_value']:+.1%}. This is a model-vs-price comparison, not a guarantee of outcome.")

st.divider()
st.caption("BetLens · public sports data · transparent probabilistic estimates")