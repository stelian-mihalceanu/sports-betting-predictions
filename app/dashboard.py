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
.stApp{background:#07111f;color:#ecf3fb}.block-container{max-width:1450px;padding-top:1.2rem}
[data-testid="stSidebar"]{background:#091727;border-right:1px solid #1d334d}
.hero{padding:.3rem 0 1rem}.eyebrow{color:#42e6ab;font-weight:800;letter-spacing:.16em;font-size:.72rem}
.hero h1{font-size:2.5rem;letter-spacing:-.04em;margin:.1rem 0}.hero p,.muted{color:#93a9c2}
.card{background:linear-gradient(180deg,#10233b,#0d1b2e);border:1px solid #1e3652;border-radius:16px;padding:16px;margin-bottom:12px}
.badge{display:inline-block;border:1px solid rgba(66,230,171,.25);background:rgba(66,230,171,.1);color:#42e6ab;border-radius:999px;padding:4px 8px;font-size:.72rem;font-weight:800;margin:3px 4px 0 0}
[data-testid="stMetric"]{background:#0d1b2e;border:1px solid #1e3652;border-radius:14px;padding:10px 12px}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="eyebrow">SPORTS INTELLIGENCE</div><h1>⚡ BetLens</h1><p>Fast football & tennis matchup research — strongest signals first, evidence behind every estimate.</p></div>', unsafe_allow_html=True)

# Keep expensive sources independent: historical results change slowly, while
# fixtures change often. This avoids invalidating the whole football cache when
# only the upcoming schedule changes.
@st.cache_data(ttl=24*60*60, show_spinner=False)
def football_history():
    data = load_football_matches(source="football-data")
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return filter_football_target(data).drop_duplicates(["date", "home_team", "away_team", "competition"])

@st.cache_data(ttl=20*60, show_spinner=False)
def football_upcoming():
    data = load_upcoming_football_fixtures()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return filter_football_target(data).drop_duplicates(["date", "home_team", "away_team", "competition"])

@st.cache_data(ttl=30*60, show_spinner=False)
def poland_bundle():
    history = load_poland_history()
    upcoming = load_poland_upcoming()
    for frame in (history, upcoming):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return history, upcoming

@st.cache_data(ttl=24*60*60, show_spinner=False)
def tennis_data():
    return load_tennis_atp(),load_tennis_wta()

@st.cache_data(ttl=45*60, show_spinner=False)
def football_state(completed):
    ratings={}
    for r in completed.sort_values("date").itertuples(index=False):
        h,a=str(r.home_team),str(r.away_team); rh,ra=ratings.get(h,1500.),ratings.get(a,1500.)
        ex=1/(1+10**((ra-(rh+55))/400)); actual=1 if r.home_goals>r.away_goals else .5 if r.home_goals==r.away_goals else 0
        ch=30*(actual-ex); ratings[h]=rh+ch; ratings[a]=ra-ch
    return {k:{"elo":v} for k,v in ratings.items()},build_team_stats(completed)

def predict_rows(upcoming,completed):
    elo,stats=football_state(completed); out=[]
    for r in upcoming.sort_values("date").head(120).itertuples(index=False):
        p=predict_match(str(r.home_team),str(r.away_team),elo,stats)
        out.append({"date":r.date,"competition":r.competition,"home":str(r.home_team),"away":str(r.away_team),"pick":p["ft_pick"],"confidence":p["confidence"],"score":f"{p['likely_home_goals']}-{p['likely_away_goals']}","xg":f"{p['home_xg']:.1f}-{p['away_xg']:.1f}","o25":p["over_2_5"],"btts":p["btts"],"corners":p["expected_corners"],"cards":p["expected_cards"]})
    return pd.DataFrame(out)

with st.sidebar:
    sport=st.radio("Sport",["Football","Tennis"],horizontal=True)
    st.divider()
    st.caption("Data cache: fixtures 20 min · history 24 h · tennis 24 h")
    st.caption("Model outputs are estimates, not bookmaker odds or guarantees.")

if sport=="Football":
    with st.spinner("Loading fixtures…"):
        upcoming = football_upcoming()
    # The default path uses the fast core dataset. Poland is loaded only when
    # its fixtures are actually present/selected, rather than on every cold start.
    if "Ekstraklasa" in upcoming.get("competition", pd.Series(dtype=str)).astype(str).unique():
        pol_history, pol_upcoming = poland_bundle()
        upcoming = pd.concat([upcoming, pol_upcoming], ignore_index=True, sort=False)
    with st.spinner("Loading model history…"):
        history = football_history()
    completed=history.dropna(subset=["home_goals","away_goals"])
    upcoming=upcoming[upcoming["home_goals"].isna() & upcoming["away_goals"].isna() & (upcoming["date"]>=pd.Timestamp.now().normalize())]
    # Add Polish history only when Polish fixtures are visible. This keeps the
    # initial page fast for the much larger domestic/European audience.
    if "Ekstraklasa" in upcoming.get("competition", pd.Series(dtype=str)).astype(str).unique():
        completed = pd.concat([completed, pol_history], ignore_index=True, sort=False)
    cats=sorted(upcoming.get("category",pd.Series(dtype=str)).dropna().unique().tolist())
    if not cats: st.warning("No upcoming football fixtures are available."); st.stop()
    c1,c2,c3=st.columns([1.4,1.2,1.1]); category=c1.selectbox("Competition",["All"]+cats); window=c2.selectbox("Window",["Today","Tomorrow","Next 3 days","Next 7 days","All upcoming"]); focus=c3.selectbox("Focus",["FT result","Goals","BTTS","Corners","Cards"])
    now=pd.Timestamp.now().normalize(); view=upcoming if category=="All" else upcoming[upcoming["category"]==category]
    if window=="Today": view=view[view.date.dt.normalize()==now]
    elif window=="Tomorrow": view=view[view.date.dt.normalize()==now+pd.Timedelta(days=1)]
    elif window=="Next 3 days": view=view[view.date<now+pd.Timedelta(days=3)]
    elif window=="Next 7 days": view=view[view.date<now+pd.Timedelta(days=7)]
    pred=predict_rows(view,completed)
    if pred.empty: st.info("No matches fit the selected filters."); st.stop()
    if focus=="Goals": pred["rank"]=pred.o25
    elif focus=="BTTS": pred["rank"]=pred.btts
    elif focus=="Corners": pred["rank"]=pred.corners
    elif focus=="Cards": pred["rank"]=pred.cards
    else: pred["rank"]=pred.confidence
    pred=pred.sort_values(["rank","date"],ascending=[False,True]).reset_index(drop=True)
    a,b,c,d,e=st.columns(5); a.metric("Matches",len(pred)); b.metric("High confidence",int((pred.confidence>=.60).sum())); c.metric("Best O2.5",f"{pred.o25.max():.0%}"); d.metric("Best BTTS",f"{pred.btts.max():.0%}"); e.metric("Avg xG",f"{pred.xg.str.split('-').map(lambda x:sum(map(float,x))).mean():.1f}")
    t1,t2=st.tabs(["🔥 Top signals","📋 All matches"])
    with t1:
        for r in pred.head(12).itertuples(index=False):
            st.markdown(f'<div class="card"><b>{r.home}</b> <span class="muted">vs</span> <b>{r.away}</b><div class="muted">{r.date:%a, %d %b %H:%M} · {r.competition}</div><div style="margin-top:8px"><span class="badge">FT {r.pick}</span><span class="badge">{r.confidence:.0%} confidence</span><span class="badge">Score {r.score}</span><span class="badge">xG {r.xg}</span><span class="badge">O2.5 {r.o25:.0%}</span><span class="badge">BTTS {r.btts:.0%}</span></div></div>',unsafe_allow_html=True)
    with t2:
        st.dataframe(pred[["date","competition","home","away","pick","confidence","score","xg","o25","btts","corners","cards"]].rename(columns={"home":"Home","away":"Away","pick":"FT","confidence":"Confidence","score":"Score","xg":"xG","o25":"O2.5","btts":"BTTS","corners":"Corners","cards":"Cards"}),hide_index=True,width="stretch")
else:
    with st.spinner("Loading tennis…"): atp,wta=tennis_data()
    tour=st.radio("Tour",["ATP","WTA"],horizontal=True); surface=st.selectbox("Surface",["All","Hard","Clay","Grass","Carpet"])
    data=filter_tennis_target(atp if tour=="ATP" else wta)
    players=sorted(set(data.winner_name.dropna().astype(str))|set(data.loser_name.dropna().astype(str)))
    p1,p2=st.columns(2); player1=p1.selectbox("Player 1",players); player2=p2.selectbox("Player 2",players,index=min(1,len(players)-1))
    if player1==player2: st.warning("Choose different players."); st.stop()
    r=predict_tennis_match(data,player1,player2,surface=None if surface=="All" else surface)
    a,b,c,d=st.columns(4); a.metric(player1,f"{r['player1_probability']:.1%}"); b.metric(player2,f"{r['player2_probability']:.1%}"); c.metric("Confidence",f"{r['confidence']:.1%}"); d.metric("H2H",f"{r['h2h_player1']}–{r['h2h_player2']}")
    evidence=pd.DataFrame({"Metric":["ELO","Surface ELO","Recent form","Serve points won","Return points won","Aces","Double faults"],player1:[r['player1_elo'],r['player1_surface_elo'],r['player1_form'],r['player1_serve_points_won'],r['player1_return_points_won'],r['player1_aces'],r['player1_double_faults']],player2:[r['player2_elo'],r['player2_surface_elo'],r['player2_form'],r['player2_serve_points_won'],r['player2_return_points_won'],r['player2_aces'],r['player2_double_faults']]})
    st.dataframe(evidence.round(3),hide_index=True,width="stretch")

st.divider(); st.caption("BetLens · public sports data · transparent probabilistic estimates")
