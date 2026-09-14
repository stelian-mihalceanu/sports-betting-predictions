from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import requests
import streamlit as st

st.set_page_config(page_title="Sports Events", page_icon="🏟️", layout="wide")

TZ = ZoneInfo("Europe/Bucharest")
API = "https://site.api.espn.com/apis/site/v2/sports"

SPORTS = {
    "Football": {
        "icon": "⚽",
        "leagues": {
            "Premier League": ("soccer", "eng.1"),
            "La Liga": ("soccer", "esp.1"),
            "Serie A": ("soccer", "ita.1"),
            "Bundesliga": ("soccer", "ger.1"),
            "Ligue 1": ("soccer", "fra.1"),
        },
    },
    "Basketball": {
        "icon": "🏀",
        "leagues": {
            "NBA": ("basketball", "nba"),
            "EuroLeague": ("basketball", "euroleague"),
            "WNBA": ("basketball", "wnba"),
        },
    },
    "Tennis": {
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


def participant_names(event: dict) -> tuple[str, str, list[dict]]:
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
    return "Scheduled"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_events(sport: str, league: str, start: str, end: str) -> list[dict]:
    url = f"{API}/{sport}/{league}/scoreboard"
    try:
        response = requests.get(url, params={"dates": f"{start}-{end}"}, timeout=4)
        response.raise_for_status()
        return response.json().get("events", [])
    except (requests.RequestException, ValueError):
        return []


@st.cache_data(ttl=300, show_spinner=False)
def fetch_sport_events(
    league_configs: tuple[tuple[str, str, str], ...], start: str, end: str
) -> dict[str, list[dict]]:
    """Fetch every competition in the selected sport concurrently.

    The result is cached as one object, so switching competitions inside a sport
    becomes a local filter instead of another network round-trip.
    """
    results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(5, len(league_configs))) as pool:
        futures = {
            pool.submit(fetch_events, sport, league, start, end): name
            for name, sport, league in league_configs
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception:
                results[name] = []
    return results


def render_event(event: dict) -> None:
    when = event_date(event)
    home, away, competitors = participant_names(event)
    status = event_status(event)
    competition = event.get("season", {}).get("displayName", "") or event.get("name", "")
    scores = [c.get("score", "") for c in competitors[:2]]
    score_text = " - ".join(scores) if status in {"Final", "LIVE"} and any(scores) else "vs"
    label = f"🔴 {status}" if status == "LIVE" else status
    with st.container(border=True):
        c1, c2, c3 = st.columns([1.1, 2.7, 1.2])
        if when:
            c1.markdown(f"**{when:%d %b}**\n\n{when:%H:%M}")
        else:
            c1.markdown("Time TBD")
        c2.markdown(f"**{home}**  {score_text}  **{away}**")
        c2.caption(competition)
        c3.write(label)
        venue = event.get("venue", {}).get("fullName")
        if venue:
            c3.caption(venue)


st.title("🏟️ Sports Events")
st.caption("Football · Tennis · Basketball — upcoming events and recent results")

# The three sport choices behave like lightweight tabs. Streamlit reruns on
# interaction, so the expensive part is deliberately moved behind one cached,
# parallel fetch per sport and a broad date window.
sport_name = st.radio("Sport", list(SPORTS), horizontal=True, key="sport")
config = SPORTS[sport_name]

period = st.radio("Period", ["Recent + Upcoming", "Next 7 Days", "Last 7 Days"], horizontal=True, key="period")
now = local_now()
if period == "Next 7 Days":
    start_dt, end_dt = now, now + timedelta(days=7)
elif period == "Last 7 Days":
    start_dt, end_dt = now - timedelta(days=7), now
else:
    start_dt, end_dt = now - timedelta(days=2), now + timedelta(days=5)

# Fetch one broad window for the whole selected sport. Competition and period
# changes then happen entirely from cached data.
fetch_start = (now - timedelta(days=7)).strftime("%Y%m%d")
fetch_end = (now + timedelta(days=7)).strftime("%Y%m%d")
league_configs = tuple(
    (name, sport, league) for name, (sport, league) in config["leagues"].items()
)

with st.spinner("Loading events…"):
    all_events = fetch_sport_events(league_configs, fetch_start, fetch_end)

league_name = st.selectbox("Competition", list(config["leagues"]), key=f"league_{sport_name}")
events = all_events.get(league_name, [])

parsed = [(event_date(e), e) for e in events]
parsed = [(d, e) for d, e in parsed if d is not None and start_dt <= d <= end_dt]
parsed.sort(key=lambda item: item[0])

if not parsed:
    st.info("No events are available for the current selection.")
    st.stop()

past = [(d, e) for d, e in parsed if event_status(e) == "Final"]
upcoming = [(d, e) for d, e in parsed if event_status(e) != "Final"]

m1, m2, m3 = st.columns(3)
m1.metric("Events", len(parsed))
m2.metric("Upcoming", len(upcoming))
m3.metric("Completed", len(past))

if upcoming:
    st.subheader("Upcoming")
    for _, event in upcoming:
        render_event(event)

if past:
    st.subheader("Recent Results")
    for _, event in reversed(past):
        render_event(event)

st.caption("Public sports data. This app is an event browser and does not guarantee sporting outcomes.")
