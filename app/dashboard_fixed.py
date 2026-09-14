from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import streamlit as st

st.set_page_config(page_title="Sports Events", page_icon="🏟️", layout="wide")

TZ = ZoneInfo("Europe/Bucharest")
API = "https://site.api.espn.com/apis/site/v2/sports"

SPORTS = {
    "Fotbal": {
        "icon": "⚽",
        "leagues": {
            "Premier League": ("soccer", "eng.1"),
            "La Liga": ("soccer", "esp.1"),
            "Serie A": ("soccer", "ita.1"),
            "Bundesliga": ("soccer", "ger.1"),
            "Ligue 1": ("soccer", "fra.1"),
        },
    },
    "Baschet": {
        "icon": "🏀",
        "leagues": {
            "NBA": ("basketball", "nba"),
            "EuroLeague": ("basketball", "euroleague"),
            "WNBA": ("basketball", "wnba"),
        },
    },
    "Tenis": {
        "icon": "🎾",
        "leagues": {
            "ATP": ("tennis", "atp"),
            "WTA": ("tennis", "wta"),
        },
    },
}


def local_now() -> datetime:
    return datetime.now(TZ)


def event_date(event: dict) -> datetime | None:
    value = event.get("date") or event.get("startDate")
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(TZ)
    except ValueError:
        return None


def team_names(event: dict) -> tuple[str, str, list[dict]]:
    competitors = event.get("competitions", [{}])[0].get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home and competitors:
        home, away = competitors[0], competitors[1] if len(competitors) > 1 else None
    home_name = ((home or {}).get("team") or {}).get("displayName") or (home or {}).get("athlete", {}).get("displayName") or "TBD"
    away_name = ((away or {}).get("team") or {}).get("displayName") or (away or {}).get("athlete", {}).get("displayName") or "TBD"
    return home_name, away_name, competitors


def event_status(event: dict) -> str:
    status = event.get("status", {}).get("type", {})
    if status.get("completed") or status.get("state") == "post":
        return "Final"
    if status.get("state") == "in":
        return "LIVE"
    return "Programat"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_events(sport: str, league: str, start: str, end: str) -> list[dict]:
    url = f"{API}/{sport}/{league}/scoreboard"
    try:
        response = requests.get(url, params={"dates": f"{start}-{end}"}, timeout=7)
        response.raise_for_status()
        return response.json().get("events", [])
    except (requests.RequestException, ValueError):
        return []


def render_event(event: dict) -> None:
    when = event_date(event)
    home, away, competitors = team_names(event)
    status = event_status(event)
    competition = event.get("season", {}).get("displayName", "")
    if not competition:
        competition = event.get("name", "")
    scores = []
    for competitor in competitors[:2]:
        scores.append(competitor.get("score", ""))
    score_text = " - ".join(scores) if status in {"Final", "LIVE"} and any(scores) else "vs"
    label = f"🔴 {status}" if status == "LIVE" else status
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.1, 2.7, 1.2])
        c1.markdown(f"**{when:%d %b}**\n\n{when:%H:%M}" if when else "Time TBD")
        c2.markdown(f"**{home}**  {score_text}  **{away}**")
        c2.caption(competition)
        c3.write(label)
        if event.get("venue", {}).get("fullName"):
            c3.caption(event["venue"]["fullName"])


st.title("🏟️ Sports Events")
st.caption("Fotbal · tenis · baschet — evenimente programate și rezultate recente")

sport_name = st.radio("Sport", list(SPORTS), horizontal=True)
config = SPORTS[sport_name]
league_name = st.selectbox("Competiție", list(config["leagues"]))

period = st.radio("Perioadă", ["Recente + următoare", "Următoarele 7 zile", "Ultimele 7 zile"], horizontal=True)
now = local_now()
if period == "Următoarele 7 zile":
    start_dt, end_dt = now, now + timedelta(days=7)
elif period == "Ultimele 7 zile":
    start_dt, end_dt = now - timedelta(days=7), now
else:
    start_dt, end_dt = now - timedelta(days=2), now + timedelta(days=5)

sport_code, league_code = config["leagues"][league_name]
start_key = start_dt.strftime("%Y%m%d")
end_key = end_dt.strftime("%Y%m%d")

with st.spinner("Se încarcă evenimentele…"):
    events = fetch_events(sport_code, league_code, start_key, end_key)

parsed = [(event_date(e), e) for e in events]
parsed = [(d, e) for d, e in parsed if d is not None and start_dt <= d <= end_dt]
parsed.sort(key=lambda item: item[0])

if not parsed:
    st.info("Nu există evenimente disponibile pentru selecția curentă.")
    st.stop()

past = [(d, e) for d, e in parsed if event_status(e) == "Final"]
upcoming = [(d, e) for d, e in parsed if event_status(e) != "Final"]

m1, m2, m3 = st.columns(3)
m1.metric("Evenimente", len(parsed))
m2.metric("Urmează", len(upcoming))
m3.metric("Finalizate", len(past))

if upcoming:
    st.subheader("Urmează")
    for _, event in upcoming:
        render_event(event)

if past:
    st.subheader("Rezultate recente")
    for _, event in reversed(past):
        render_event(event)

st.caption("Date furnizate de surse publice ESPN. Aplicația este un browser de evenimente, nu o garanție de rezultate sportive.")
