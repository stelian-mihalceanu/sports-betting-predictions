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
st.caption("Match detail · team strength · ensemble prediction · goals distribution · value scanner")

@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_data() -> pd.DataFrame:
    frames = [load_football_matches(source="football-data")]
    for loader in (load_upcoming_football_fixtures, load_poland_upcoming):
        try:
            f = loader()
            if not f.empty:
                frames.append(f)
        except Exception:
            pass
    try:
        p = load_poland_history()
        if not p.empty:
            frames.append(p)
    except Exception:
        pass
    data = pd.concat(frames, ignore_index=True, sort=False)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return filter_football_target(data).drop_duplicates(subset=["date", "home_team", "away_team", "competition"])

def elo_state(history: pd.DataFrame) -> dict[str, dict[str, float]]:
    ratings: dict[str, float] = {}
    for _, r in history.sort_values("date").iterrows():
        h, a = str(r["home_team"]), str(r["away_team"])
        rh, ra = ratings.get(h, 1500.0), ratings.get(a, 1500.0)
        expected = 1 / (1 + 10 ** ((ra - rh - 55) / 400))
        hg, ag = float(r["home_goals"]), float(r["away_goals"])
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        change = 30 * (actual - expected)
        ratings[h], ratings[a] = rh + change, ra - change
    return {k: {"elo": v} for k, v in ratings.items()}

with st.spinner("Loading football data…"):
    data = load_data()

completed = data.dropna(subset=["home_goals", "away_goals"]).copy()
upcoming = data[data["home_goals"].isna() & data["away_goals"].isna() & (data["date"] >= pd.Timestamp.now().normalize())].copy()
if upcoming.empty:
    st.warning("No upcoming matches are available in the current public feeds.")
    st.stop()

ordered = upcoming.sort_values("date").head(200).reset_index(drop=True)
labels = [f"{r.home_team} — {r.away_team} · {r.date:%d %b %H:%M} · {r.competition}" for _, r in ordered.iterrows()]
selected = ordered.iloc[st.selectbox("Choose a match", range(len(labels)), format_func=lambda i: labels[i])]
home, away = str(selected.home_team), str(selected.away_team)

home_strength = team_strength(completed, home)
away_strength = team_strength(completed, away)
elo = elo_state(completed)
stats = build_team_stats(completed)
base = predict_match(home, away, elo, stats)
ensemble = ensemble_prediction(base, home_strength, away_strength, h2h(completed, home, away))
ft1, ftx, ft2 = ensemble["ft_home"], ensemble["ft_draw"], ensemble["ft_away"]

st.markdown(f"### {home}  vs  {away}")
st.caption(f"{selected.date:%A, %d %B %Y · %H:%M} · {selected.competition}")

c1, c2, c3, c4 = st.columns(4)
pick = "1" if ft1 == max(ft1, ftx, ft2) else "X" if ftx == max(ft1, ftx, ft2) else "2"
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
    hh = h2h(completed, home, away)
    st.subheader("H2H")
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Matches", hh["matches"]); h2.metric(f"{home} wins", hh["home_wins"]); h3.metric("Draws", hh["draws"]); h4.metric(f"{away} wins", hh["away_wins"])
    if hh["rows"]: st.dataframe(pd.DataFrame(hh["rows"]), hide_index=True, width="stretch")
    st.subheader("Key model markets")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("O1.5", f"{base['over_1_5']:.0%}"); k2.metric("O2.5", f"{base['over_2_5']:.0%}"); k3.metric("BTTS", f"{base['btts']:.0%}"); k4.metric("Corners", f"{base['expected_corners']:.1f}"); k5.metric("Cards", f"{base['expected_cards']:.1f}")

with tab_strength:
    strength = pd.DataFrame([
        {"Metric": "Attack", home: home_strength["attack"], away: away_strength["attack"]},
        {"Metric": "Defense conceded", home: home_strength["defense"], away: away_strength["defense"]},
        {"Metric": "Recent form", home: home_strength["form"], away: away_strength["form"]},
        {"Metric": "Matches", home: home_strength["matches"], away: away_strength["matches"]},
        {"Metric": "ELO", home: elo.get(home, {"elo": 1500})["elo"], away: elo.get(away, {"elo": 1500})["elo"]},
    ])
    st.dataframe(strength.round(2), hide_index=True, width="stretch")
    st.caption("Attack = recent goals scored; defense = recent goals conceded. Transparent indicators, not league-normalized ratings.")
    comp = ensemble["components"]
    st.subheader("Ensemble components")
    st.dataframe(pd.DataFrame({"Component": ["Poisson", "ELO", "Form", "Market-style prior"], "1": [x[0] for x in comp.values()], "X": [x[1] for x in comp.values()], "2": [x[2] for x in comp.values()]}).round(3), hide_index=True, width="stretch")

with tab_distribution:
    dist = goals_distribution(float(base["home_xg"]), float(base["away_xg"])).head(20).copy()
    dist["Score"] = dist["home_goals"].astype(str) + "-" + dist["away_goals"].astype(str)
    dist["Probability"] = dist["probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(dist[["Score", "Probability"]], hide_index=True, width="stretch")
    st.caption("Poisson score distribution based on model xG.")

with tab_value:
    st.subheader("Model probability vs bookmaker price")
    st.caption("Enter a decimal price from a bookmaker. The scanner calculates implied probability, edge, fair odds and simple expected value.")
    market = st.selectbox("Market", ["FT 1", "FT X", "FT 2", "Over 2.5", "BTTS"])
    model_prob = {"FT 1": ft1, "FT X": ftx, "FT 2": ft2, "Over 2.5": base["over_2_5"], "BTTS": base["btts"]}[market]
    b1, b2, b3 = st.columns(3)
    bookmaker = b1.selectbox("Bookmaker", ["Superbet", "Betano", "Unibet", "MrBit", "Other"])
    odds = b2.number_input("Decimal odds", min_value=1.01, value=2.00, step=0.01)
    min_edge = b3.slider("Minimum edge", 0.0, 0.20, 0.03, 0.01)
    val = value_edge(model_prob, odds)
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.metric("Model", f"{model_prob:.1%}"); v2.metric("Implied", f"{val['implied_probability']:.1%}"); v3.metric("Edge", f"{val['edge']:+.1%}"); v4.metric("Fair odds", f"{val['fair_odds']:.2f}"); v5.metric("Suggested stake", f"{val['kelly_stake_pct']:.1%}", help="Quarter-Kelly stake as a % of bankroll, clipped to 0 when there is no model edge. This is a mathematical sizing reference, not a recommendation.")
    if val["edge"] >= min_edge: st.success(f"Potential model value: {market} at {odds:.2f} ({bookmaker}). Edge {val['edge']:+.1%}. Suggested stake (quarter-Kelly): {val['kelly_stake_pct']:.1%} of bankroll.")
    else: st.info("No model edge above the selected threshold.")
    st.warning("Value and stake sizing are model estimates only — not a guarantee, not financial advice, and not a recommendation to gamble. Bet responsibly.")

st.divider()
st.subheader("Bookmaker sources")
st.markdown("- [Superbet football odds](https://superbet.ro/pariuri-sportive/fotbal)\n- [Betano Romania](https://ro.betano.com/)\n- [Unibet Romania](https://www.unibet.ro/betting/odds)\n- [MrBit Romania](https://mrbit.ro/ro/betting)")
st.caption("Bookmaker odds are dynamic. For production-grade automated odds ingestion, use an authorized odds API/feed rather than fragile HTML scraping.")