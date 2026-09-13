# Sports Betting Predictions

Research-oriented Streamlit application for football and tennis analytics, with a user-friendly prediction dashboard.

## Live dashboard

When local datasets are absent, the application loads public data at runtime and caches it in Streamlit:

- **Football:** Football-Data.co.uk historical/current feeds, including Premier League, Bundesliga, La Liga, Serie A, Liga 1 (Romania) and **Ekstraklasa (Poland)**
- **European football fixtures:** UEFA Champions League, Europa League and Conference League schedules via Fixture Download
- **Tennis:** ATP/WTA historical match archive based on Jeff Sackmann data, used for surface-aware matchup analysis

The app is designed around a simple question: **which matches and markets deserve a closer look, and what evidence supports the model?**

## UX highlights

### Football
- **Top signals**: ranked upcoming matches with FT pick, model confidence, likely score, xG, O2.5, BTTS, corners and cards
- **Matches**: compact match cards with expandable market details
- **Markets**: rank upcoming fixtures by a selected market
- **League pulse**: recent goal and match-level context by competition
- **Filters**: competition, time window and minimum model confidence

### Tennis
- **Match Lab**: compare any two supported ATP/WTA players
- Surface selector: All / Hard / Clay / Grass / Carpet
- Player win probability and confidence
- Overall and surface-aware Elo
- Recent form
- Serve points won, return points won, aces and double faults
- Head-to-head record
- Player board and historical match views

## Football prediction model

The football engine is deliberately transparent rather than pretending to be a black-box oracle. It combines:

- recency-weighted team scoring rates
- home/away splits
- pre-match Elo strength
- first-half scoring rates when available
- corners and cards when the source contains them
- Poisson score distributions

The dashboard exposes FT and HT 1X2, double chance, exact-score mode, xG, goal totals, BTTS, first-half goals, corners and cards.

Additional market lines include:

- Goals: Over 1.5 / 2.5 / 3.5
- First half: Over 0.5 / 1.5 and HT BTTS
- Corners: Over 7.5 / 8.5 / 9.5 / 10.5
- Cards: Over 2.5 / 3.5 / 4.5
- Double chance: 1X / X2 / 12

Probabilities are model estimates, not bookmaker odds or guarantees.

## Tennis model

The tennis Match Lab is designed for matchup research when a live fixture feed is not available. It uses completed matches to estimate:

- Elo strength
- surface-specific Elo
- recent win rate
- serve points won
- return points won
- aces and double faults
- head-to-head history

This is intentionally presented as a **matchup model**, not as a claim that the selected pair has a fixture scheduled.

## Supported competitions

### Football
- Premier League
- Bundesliga
- La Liga
- Serie A
- Liga 1 (Romania)
- **Ekstraklasa (Poland)**
- UEFA Champions League
- UEFA Europa League
- UEFA Conference League

### Tennis
- Grand Slams: Australian Open, Roland Garros, Wimbledon, US Open
- ATP Masters 1000
- ATP 500
- WTA 1000
- WTA 500

## Data provenance

Football historical/current results come from Football-Data.co.uk public CSV feeds. Football-Data explicitly lists Poland as an extra league with Ekstraklasa coverage and provides current/historical results and match statistics. urlFootball-Data Poland datahttps://www.football-data.co.uk/poland.php

Current UEFA competition schedules are loaded from Fixture Download. UEFA's own competition pages remain useful authoritative references for official fixtures and competition statistics. urlUEFA Champions League fixtures & resultshttps://www.uefa.com/uefachampionsleague/news/02a8-2174c9e9019d-f909a77bd77a-1000--2026-27-champions-league-all-the-league-phase-fixtures/

Sofascore, Forebet, BetMines, PredictZ, Sports Mole, Football Whispers, Tiki Taka, Goal Signal and ScoreGPT were used as **UX/model-design references** for ideas such as top signals, market filters, form context, league statistics, confidence, previews and broader market coverage. Their published accuracy claims are not treated as ground truth.

For tennis, the app uses a public ATP/WTA historical archive. Official ATP and WTA pages are used as reference points for ranking/statistics concepts; the WTA Stats Zone, for example, exposes serving/returning statistics such as aces, double faults, first-serve points, service games won and return points won. urlWTA Stats Zonehttps://www.wtatennis.com/stats/2026

## Project structure

```text
sports-betting-predictions/
├─ app/
│  └─ streamlit_app.py
├─ data/
│  ├─ raw/
│  └─ processed/
├─ src/
│  ├─ data_filters.py
│  ├─ data_loader.py
│  ├─ extra_football.py
│  ├─ features.py
│  ├─ models.py
│  ├─ predictions.py
│  └─ tennis_predictions.py
└─ tests/
   ├─ test_features.py
   ├─ test_predictions.py
   └─ test_tennis_predictions.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

## Run tests

```bash
pytest -q
```

## Run the web application

```bash
streamlit run app/streamlit_app.py
```

For Streamlit Community Cloud, use `app/streamlit_app.py` as the entrypoint and Python 3.12+.

## Responsible use

Model outputs are probabilistic estimates for research and decision support. They are not guarantees of results, betting returns or profit. Do not interpret model confidence as a promise of accuracy.

## License

MIT
