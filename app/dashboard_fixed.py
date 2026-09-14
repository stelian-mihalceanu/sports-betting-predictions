from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_filters import filter_football_target
from src.data_loader import load_football_matches, load_upcoming_football_fixtures
from src.predictions import build_team_stats, predict_match

st.set_page_config(page_title="BetLens", page_icon="⚽", layout="wide")


def pct(x: float) -> str:
    return f"{float(x):.0%}"


def fair_odds(x: float) -> float:
    return round(1 / float(x), 2) if x and x > 0 else 0.0


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def history() -> pd.DataFrame:
    data = load_football_matches(source="football-data", timeout=5)
    if data.empty:
        return data
    data = filter_football_target(data).copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data.dropna(subset=["date"]).drop_duplicates(["date", "home_team", "away_team", "competition"])


@st.cache_data(ttl=15 * 60, show_spinner=False)
def upcoming() -> pd.DataFrame:
    data = load_upcoming_football_fixtures(timeout=5)
    if data.empty:
        return data
    data = filter_football_target(data).copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data.dropna(subset=["date"]).drop_duplicates(["date", "home_team", "away_team", "competition"])


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def model_state(data: pd.DataFrame):
    if data.empty:
        return {}, {}
    data = data.dropna(subset=["home_goals", "away_goals"]).sort_values("date")
    ratings: dict[str, float] = {}
    for row in data.itertuples(index=False):
        home, away = str(row.home_team), str(row.away_team)
        try:
            hg, ag = float(row.home_goals), float(row.away_goals)
        except (TypeError, ValueError):
            continue
        rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        expected = 1 / (1 + 10 ** ((ra - rh - 55) / 400))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        change = 30 * (actual - expected)
        ratings[home], ratings[away] = rh + change, ra - change
    return {k: {"elo": v} for k, v in ratings.items()}, build_team_stats(data)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def predictions(fixtures: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame()
    elo, stats = model_state(hist)
    rows = []
    for r in fixtures.sort_values("date").head(40).itertuples(index=False):
        try:
            p = predict_match(str(r.home_team), str(r.away_team), elo, stats)
        except Exception:
            continue
        probs = {"1": p["ft_home"], "X": p["ft_draw"], "2": p["ft_away"]}
        pick = p["ft_pick"]
        markets = {f"FT {pick}": probs[pick], "O2.5": p["over_2_5"], "BTTS": p["btts"]}
        signal, probability = max(markets.items(), key=lambda item: item[1])
        rows.append({
            "date": r.date, "competition": r.competition, "home": r.home_team, "away": r.away_team,
            "signal": signal, "probability": probability, "fair_odds": fair_odds(probability),
            "ft_pick": pick, "ft_confidence": p["confidence"],
            "score": f"{p['likely_home_goals']}-{p['likely_away_goals']}",
            "o25": p["over_2_5"], "btts": p["btts"],
        })
    return pd.DataFrame(rows)


st.title("⚽ BetLens")
st.caption("Predicții simple de fotbal. Probabilitățile sunt estimări ale modelului.")

with st.spinner("Se încarcă meciurile…"):
    fixtures = upcoming()
    hist = history()

if fixtures.empty:
    st.warning("Nu sunt disponibile meciuri viitoare acum.")
    st.stop()

now = pd.Timestamp.now().normalize()
period = st.radio("Perioadă", ["Azi", "Mâine", "Următoarele 3 zile", "Toate"], horizontal=True)
if period == "Azi":
    view = fixtures[fixtures.date.dt.normalize() == now]
elif period == "Mâine":
    view = fixtures[fixtures.date.dt.normalize() == now + pd.Timedelta(days=1)]
elif period == "Următoarele 3 zile":
    view = fixtures[fixtures.date < now + pd.Timedelta(days=3)]
else:
    view = fixtures

if view.empty:
    st.info("Nu există meciuri pentru perioada aleasă.")
    st.stop()

pred = predictions(view, hist)
if pred.empty:
    st.error("Nu am putut calcula predicțiile.")
    st.stop()

pred = pred.sort_values(["probability", "date"], ascending=[False, True]).reset_index(drop=True)

c1, c2, c3 = st.columns(3)
c1.metric("Meciuri", len(pred))
c2.metric("Cel mai puternic", pct(pred.probability.max()))
c3.metric("Semnale ≥65%", int((pred.probability >= 0.65).sum()))

st.subheader("Top predicții")
for row in pred.head(10).itertuples(index=False):
    with st.container(border=True):
        a, b, c = st.columns([2.4, 1, 1])
        a.markdown(f"**{row.home} — {row.away}**")
        a.caption(f"{row.date:%a, %d %b %H:%M} · {row.competition}")
        a.write(f"**{row.signal}** · {pct(row.probability)} · fair odds {row.fair_odds:.2f}")
        b.metric("FT", f"{row.ft_pick} · {pct(row.ft_confidence)}")
        b.metric("Scor", row.score)
        c.metric("O2.5", pct(row.o25))
        c.metric("BTTS", pct(row.btts))

st.subheader("Toate meciurile")
st.dataframe(
    pred[["date", "competition", "home", "away", "signal", "probability", "fair_odds", "ft_pick", "ft_confidence", "score"]],
    hide_index=True,
    use_container_width=True,
)

st.caption("BetLens · date publice · estimări probabilistice")
