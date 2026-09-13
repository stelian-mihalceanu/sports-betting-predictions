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
from src.extra_football import load_poland_history, load_poland_upcoming
from src.features import add_elo_features, add_football_form_features
from src.predictions import build_team_stats, predict_match
from src.tennis_predictions import predict_tennis_match

st.set_page_config(page_title="Sports Betting Predictions", page_icon="🏆", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
.small-muted {color: #777; font-size: 0.86rem;}
.hero {padding: 0.3rem 0 0.7rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero">', unsafe_allow_html=True)
st.title("🏆 Sports Betting Predictions")
st.caption("Football & tennis research dashboard · probabilities, form, market signals and matchup analysis")
st.markdown('</div>', unsafe_allow_html=True)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def load_all_football() -> pd.DataFrame:
    frames = [load_football_matches(source="football-data")]
    try:
        upcoming = load_upcoming_football_fixtures()
        if not upcoming.empty:
            frames.append(upcoming)
    except Exception:
        pass
    try:
        pol_history = load_poland_history()
        if not pol_history.empty:
            frames.append(pol_history)
        pol_upcoming = load_poland_upcoming()
        if not pol_upcoming.empty:
            frames.append(pol_upcoming)
    except Exception:
        pass
    return pd.concat(frames, ignore_index=True, sort=False).sort_values("date").reset_index(drop=True)


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


def football_elo_state(history: pd.DataFrame) -> dict[str, dict[str, float]]:
    ratings: dict[str, float] = {}
    for _, row in history.sort_values("date").iterrows():
        home, away = str(row["home_team"]), str(row["away_team"])
        rh, ra = ratings.get(home, 1500.0), ratings.get(away, 1500.0)
        expected = 1.0 / (1.0 + 10 ** ((ra - (rh + 55.0)) / 400.0))
        hg, ag = float(row["home_goals"]), float(row["away_goals"])
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        change = 30.0 * (actual - expected)
        ratings[home], ratings[away] = rh + change, ra - change
    return {team: {"elo": rating} for team, rating in ratings.items()}


@st.cache_data(ttl=30 * 60, show_spinner=False)
def prediction_state(completed: pd.DataFrame):
    return football_elo_state(completed), build_team_stats(completed)


def build_predictions(upcoming: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    elo_state, stats = prediction_state(completed)
    rows = []
    for _, match in upcoming.sort_values("date").head(200).iterrows():
        home, away = str(match["home_team"]), str(match["away_team"])
        p = predict_match(home, away, elo_state, stats)
        rows.append({
            "date": match["date"], "competition": match["competition"], "home": home, "away": away,
            "prediction": p["ft_pick"], "confidence": p["confidence"],
            "ft_1": p["ft_home"], "ft_x": p["ft_draw"], "ft_2": p["ft_away"],
            "ht_pick": p["ht_pick"], "ht_1": p["ht_home"], "ht_x": p["ht_draw"], "ht_2": p["ht_away"],
            "score": f"{p['likely_home_goals']}-{p['likely_away_goals']}",
            "xg": f"{p['home_xg']:.1f}-{p['away_xg']:.1f}",
            "over15": p["over_1_5"], "over25": p["over_2_5"], "under25": p["under_2_5"],
            "over35": p["over_3_5"], "btts": p["btts"], "double_1x": p["double_1x"], "double_x2": p["double_x2"],
            "ht_o05": p["ht_over_0_5"], "ht_o15": p["ht_over_1_5"], "ht_btts": p["ht_btts"],
            "corners": p["expected_corners"], "o75c": p["over_7_5_corners"], "o85c": p["over_8_5_corners"], "o95c": p["over_9_5_corners"], "o105c": p["over_10_5_corners"],
            "cards": p["expected_cards"], "o25cards": p["over_2_5_cards"], "o35cards": p["over_3_5_cards"], "o45cards": p["over_4_5_cards"],
        })
    return pd.DataFrame(rows)


with st.sidebar:
    sport = st.radio("Sport", ["Football", "Tennis"], key="sport")
    st.divider()
    if sport == "Football":
        st.subheader("What matters?")
        st.caption("FT / HT · goals · BTTS · corners · cards · double chance")
        st.caption("The model combines recent form, venue splits, ELO and available match statistics.")
    else:
        st.subheader("Tennis model")
        st.caption("ELO · surface form · recent form · serve/return profile · H2H")
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

    st.subheader("⚽ Upcoming matches")
    f1, f2, f3, f4 = st.columns([1.4, 1.2, 1.2, 1])
    category = f1.selectbox("Competition", ["All"] + categories, key="football_category")
    period = f2.selectbox("When", ["Today", "Tomorrow", "Next 3 days", "Next 7 days", "All upcoming"], key="football_period")
    market_filter = f3.selectbox("Focus", ["FT result", "Goals", "BTTS", "Corners", "Cards", "HT"], key="football_focus")
    min_conf = f4.slider("Min confidence", 0.34, 0.80, 0.45, 0.01, key="min_conf")

    upcoming_view = upcoming if category == "All" else upcoming[upcoming["category"] == category]
    today = pd.Timestamp.now().normalize()
    if period == "Today": upcoming_view = upcoming_view[upcoming_view["date"].dt.normalize() == today]
    elif period == "Tomorrow": upcoming_view = upcoming_view[upcoming_view["date"].dt.normalize() == today + pd.Timedelta(days=1)]
    elif period == "Next 3 days": upcoming_view = upcoming_view[upcoming_view["date"] < today + pd.Timedelta(days=3)]
    elif period == "Next 7 days": upcoming_view = upcoming_view[upcoming_view["date"] < today + pd.Timedelta(days=7)]

    predictions = build_predictions(upcoming_view, completed) if not upcoming_view.empty else pd.DataFrame()
    if not predictions.empty:
        predictions = predictions[predictions["confidence"] >= min_conf].copy()
        if market_filter == "Goals": predictions["rank_signal"] = predictions["over25"]
        elif market_filter == "BTTS": predictions["rank_signal"] = predictions["btts"]
        elif market_filter == "Corners": predictions["rank_signal"] = predictions["o85c"]
        elif market_filter == "Cards": predictions["rank_signal"] = predictions["o35cards"]
        elif market_filter == "HT": predictions["rank_signal"] = predictions["ht_o05"]
        else: predictions["rank_signal"] = predictions["confidence"]
        predictions = predictions.sort_values(["date", "rank_signal"], ascending=[True, False])

    if predictions.empty:
        st.info("No upcoming matches match the selected filters. Try All upcoming or lower the confidence threshold.")
        st.stop()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Matches", len(predictions))
    m2.metric("Model confidence ≥60%", int((predictions["confidence"] >= 0.60).sum()))
    m3.metric("Best O2.5", f"{predictions['over25'].max():.0%}")
    m4.metric("Best BTTS", f"{predictions['btts'].max():.0%}")
    m5.metric("Avg xG", f"{predictions['xg'].str.split('-').map(lambda x: sum(map(float, x))).mean():.1f}")

    tab_top, tab_matches, tab_markets, tab_league, tab_history = st.tabs(["🔥 Top signals", "📅 Matches", "🎯 Markets", "🏆 League pulse", "📊 History"])

    with tab_top:
        top = predictions.sort_values("rank_signal", ascending=False).head(15).copy()
        top["Match"] = top["home"] + " — " + top["away"]
        top["FT"] = top["prediction"] + " · " + top["confidence"].map(lambda x: f"{x:.0%}")
        top["HT"] = top["ht_pick"]
        top["Score"] = top["score"]
        top["xG"] = top["xg"]
        top["O2.5"] = top["over25"].map(lambda x: f"{x:.0%}")
        top["BTTS"] = top["btts"].map(lambda x: f"{x:.0%}")
        top["Corners"] = top["corners"].map(lambda x: f"{x:.1f}")
        top["Cards"] = top["cards"].map(lambda x: f"{x:.1f}")
        st.dataframe(top[["date", "competition", "Match", "FT", "HT", "Score", "xG", "O2.5", "BTTS", "Corners", "Cards"]], width="stretch", hide_index=True)
        st.caption("Top signals are ranked by the selected focus market. A high model probability is not a guarantee of an outcome.")

    with tab_matches:
        for _, row in predictions.head(35).iterrows():
            with st.container(border=True):
                a, b, c, d = st.columns([2.5, 1.1, 1.1, 1.1])
                a.markdown(f"**{row['home']}**  vs  **{row['away']}**")
                a.caption(f"{row['date']:%d %b %Y, %H:%M} · {row['competition']}")
                b.metric("FT", f"{row['prediction']} · {row['confidence']:.0%}")
                c.metric("Score", row["score"])
                d.metric("xG", row["xg"])
                with st.expander("All markets"):
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("FT 1 / X / 2", f"{row['ft_1']:.0%} / {row['ft_x']:.0%} / {row['ft_2']:.0%}")
                    p2.metric("HT 1 / X / 2", f"{row['ht_1']:.0%} / {row['ht_x']:.0%} / {row['ht_2']:.0%}")
                    p3.metric("Double chance 1X / X2", f"{row['double_1x']:.0%} / {row['double_x2']:.0%}")
                    p4.metric("BTTS", f"{row['btts']:.0%}")
                    p5, p6, p7, p8 = st.columns(4)
                    p5.metric("O1.5 / O2.5 / O3.5", f"{row['over15']:.0%} / {row['over25']:.0%} / {row['over35']:.0%}")
                    p6.metric("HT O0.5 / O1.5", f"{row['ht_o05']:.0%} / {row['ht_o15']:.0%}")
                    p7.metric("HT BTTS", f"{row['ht_btts']:.0%}")
                    p8.metric("Corners 7.5 / 8.5 / 9.5", f"{row['o75c']:.0%} / {row['o85c']:.0%} / {row['o95c']:.0%}")
                    p9, p10, p11, p12 = st.columns(4)
                    p9.metric("Corners", f"{row['corners']:.1f}")
                    p10.metric("Cards", f"{row['cards']:.1f}")
                    p11.metric("Cards O2.5 / O3.5", f"{row['o25cards']:.0%} / {row['o35cards']:.0%}")
                    p12.metric("Cards O4.5", f"{row['o45cards']:.0%}")

    with tab_markets:
        market_labels = {
            "FT 1": "ft_1", "FT X": "ft_x", "FT 2": "ft_2", "HT 1": "ht_1", "HT X": "ht_x", "HT 2": "ht_2",
            "Double chance 1X": "double_1x", "Double chance X2": "double_x2", "Over 1.5 goals": "over15", "Over 2.5 goals": "over25", "Over 3.5 goals": "over35",
            "BTTS": "btts", "HT Over 0.5": "ht_o05", "HT Over 1.5": "ht_o15", "HT BTTS": "ht_btts",
            "Over 7.5 corners": "o75c", "Over 8.5 corners": "o85c", "Over 9.5 corners": "o95c", "Over 10.5 corners": "o105c",
            "Over 2.5 cards": "o25cards", "Over 3.5 cards": "o35cards", "Over 4.5 cards": "o45cards",
        }
        selected = st.selectbox("Rank matches by market", list(market_labels), key="market_selector")
        col = market_labels[selected]
        market = predictions.sort_values(col, ascending=False).head(25).copy()
        market["Match"] = market["home"] + " — " + market["away"]
        market["Probability"] = market[col].map(lambda x: f"{x:.0%}")
        st.dataframe(market[["date", "competition", "Match", "Probability", "prediction", "score", "xg"]], width="stretch", hide_index=True)

    with tab_league:
        if featured.empty:
            st.info("No completed history available.")
        else:
            league = featured.copy()
            league["total_goals"] = league["home_goals"] + league["away_goals"]
            summary = league.groupby("category").agg(
                Matches=("total_goals", "size"),
                Goals=("total_goals", "mean"),
                HomeGoals=("home_goals", "mean"),
                AwayGoals=("away_goals", "mean"),
            ).reset_index()
            summary["O2.5"] = league.groupby("category")["total_goals"].apply(lambda s: float((s > 2).mean())).values
            if "HY" in league.columns and "AY" in league.columns:
                summary["Cards"] = league.assign(cards=pd.to_numeric(league["HY"], errors="coerce").fillna(0) + pd.to_numeric(league["AY"], errors="coerce").fillna(0)).groupby("category")["cards"].mean().values
            st.dataframe(summary.round(2), width="stretch", hide_index=True)
            st.caption("League pulse uses the completed public match sample currently loaded by the app.")

    with tab_history:
        columns = [c for c in ["date", "competition", "home_team", "away_team", "home_goals", "away_goals", "home_elo", "away_elo", "elo_diff", "home_form_points", "away_form_points"] if c in featured.columns]
        st.dataframe(featured.sort_values("date", ascending=False)[columns].head(300), width="stretch", hide_index=True)

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
    view = view.sort_values("tourney_date").reset_index(drop=True)
    view = add_elo_features(view)

    tour = st.selectbox("Tour", ["All", "ATP", "WTA"], key="tennis_tour")
    category = st.selectbox("Tournament level", ["All"] + sorted(view["category"].dropna().unique()), key="tennis_category")
    filtered_view = view.copy()
    if tour != "All": filtered_view = filtered_view[filtered_view["tour"] == tour]
    if category != "All": filtered_view = filtered_view[filtered_view["category"] == category]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matches", f"{len(filtered_view):,}")
    m2.metric("ATP", f"{len(filtered_view[filtered_view['tour'] == 'ATP']):,}")
    m3.metric("WTA", f"{len(filtered_view[filtered_view['tour'] == 'WTA']):,}")
    m4.metric("Players", f"{pd.unique(pd.concat([filtered_view['winner_name'], filtered_view['loser_name']])).size:,}")

    tab_matchup, tab_players, tab_history = st.tabs(["🎾 Match Lab", "🏆 Player board", "📊 Match history"])
    with tab_matchup:
        players = sorted(pd.unique(pd.concat([filtered_view["winner_name"], filtered_view["loser_name"]])).tolist())
        c1, c2, c3 = st.columns([2, 2, 1])
        player_a = c1.selectbox("Player A", players, index=0 if players else None, key="tennis_a")
        default_b = 1 if len(players) > 1 else 0
        player_b = c2.selectbox("Player B", players, index=default_b, key="tennis_b")
        surface = c3.selectbox("Surface", ["All", "Hard", "Clay", "Grass", "Carpet"], key="tennis_surface")
        if player_a == player_b:
            st.warning("Choose two different players.")
        else:
            p = predict_tennis_match(player_a, player_b, view, surface)
            st.caption("Match Lab is a research matchup model; it is not a live bookmaker feed and does not claim an official fixture exists.")
            a, b, c, d = st.columns(4)
            a.metric("Model pick", p["pick"])
            b.metric(f"{player_a} win", f"{p['a_win']:.0%}")
            c.metric(f"{player_b} win", f"{p['b_win']:.0%}")
            d.metric("Confidence", f"{p['confidence']:.0%}")
            st.progress(float(p["a_win"]), text=f"{player_a} {p['a_win']:.0%} · {player_b} {p['b_win']:.0%}")
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("ELO", f"{p['elo_a']:.0f} vs {p['elo_b']:.0f}")
            x2.metric("Recent form", f"{p['form_a']:.0%} vs {p['form_b']:.0%}")
            x3.metric("Serve pts won", f"{p['serve_a']:.0%} vs {p['serve_b']:.0%}")
            x4.metric("Return pts won", f"{p['return_a']:.0%} vs {p['return_b']:.0%}")
            y1, y2, y3 = st.columns(3)
            y1.metric("Aces / match", f"{p['aces_a']:.1f} vs {p['aces_b']:.1f}")
            y2.metric("Double faults", f"{p['double_faults_a']:.1f} vs {p['double_faults_b']:.1f}")
            y3.metric("H2H", f"{p['h2h_a']} - {p['h2h_b']}")

    with tab_players:
        player_rows = []
        for player in pd.unique(pd.concat([filtered_view["winner_name"], filtered_view["loser_name"]], ignore_index=True)):
            matches_player = filtered_view[(filtered_view["winner_name"] == player) | (filtered_view["loser_name"] == player)].tail(20)
            wins = int((matches_player["winner_name"] == player).sum())
            player_rows.append({"Player": player, "Recent W%": wins / len(matches_player) if len(matches_player) else 0.5, "Matches": len(matches_player)})
        board = pd.DataFrame(player_rows).sort_values(["Recent W%", "Matches"], ascending=[False, False]).head(100)
        board["Recent W%"] = board["Recent W%"].map(lambda x: f"{x:.0%}")
        st.dataframe(board, width="stretch", hide_index=True)

    with tab_history:
        columns = [c for c in ["tourney_date", "tourney_name", "tour", "winner_name", "loser_name", "winner_elo", "loser_elo", "elo_diff", "surface"] if c in filtered_view.columns]
        st.dataframe(filtered_view.sort_values("tourney_date", ascending=False)[columns].head(400), width="stretch", hide_index=True)

st.divider()
st.caption("Research and decision-support dashboard. Model probabilities are estimates, not guarantees of outcomes or profit.")
