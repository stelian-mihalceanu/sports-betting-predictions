# Sports Betting Predictions

Machine-learning decision-support application for football and tennis match analytics, with a browser-based Streamlit interface.

## Live dashboard

The Streamlit app can run without committing large raw datasets to this repository. When local files are absent, the application loads public data at runtime:

- **Football historical/current domestic data:** Football-Data.co.uk feeds for Premier League, Bundesliga, La Liga, Serie A and Liga 1 (Romania)
- **Football upcoming European fixtures:** current UEFA Champions League, Europa League and Conference League schedules from Fixture Download
- **Tennis:** an archival mirror of Jeff Sackmann's ATP/WTA match datasets, covering recent seasons through 2026

The app caches remote data in Streamlit for a limited period to reduce repeated downloads.

## What is included

- Football and tennis data loaders with local CSV/Parquet support and public-data fallback
- Target competition filters
- Leakage-aware rolling form features
- Pre-match tennis and football ELO ratings
- Football upcoming-fixture 1X2 estimates
- Football upcoming market estimates for goals, exact-score mode, BTTS, corners and cards
- First-half and full-time 1X2 probabilities
- Over/under-style goal, corner and card probabilities
- XGBoost model training helpers with temporal holdout evaluation
- Streamlit web dashboard for analytics and upcoming football estimates
- Pytest regression coverage for feature engineering and prediction markets

## Supported competitions

### Tennis (ATP & WTA)
- **Grand Slams**: Australian Open, Roland Garros, Wimbledon, US Open
- **ATP Masters 1000**
- **ATP 500**
- **WTA 1000**
- **WTA 500**

### Football
Domestic historical/current feeds:
- **Premier League**
- **Bundesliga**
- **La Liga**
- **Serie A**
- **Liga 1 (Romania)**

Upcoming fixture feeds:
- **UEFA Champions League**
- **UEFA Europa League**
- **UEFA Conference League**

European upcoming matches use the same prediction engine, but when a team has insufficient local historical statistics the engine falls back to broader historical team/competition averages. Market estimates are probabilistic and should not be interpreted as bookmaker odds.

## Football upcoming markets

For each upcoming match the dashboard can show:
- **FT 1 / X / 2** and the model's full-time pick
- **HT 1 / X / 2** and the model's first-half pick
- **Most likely score** and expected goals (xG)
- **Over 1.5** and **Over 2.5 goals**
- **BTTS**
- **Expected corners** and **Over 8.5 corners**
- **Expected cards** and **Over 3.5 cards**

Corner and card estimates use available historical corner/card fields from the Football-Data datasets; where a team lacks those fields, the model uses a broader fallback average.

## Project Structure

```text
sports-betting-predictions/
├─ app/
│  └─ streamlit_app.py
├─ data/
│  ├─ raw/
│  └─ processed/
├─ notebooks/
│  ├─ football_exploration.ipynb
│  └─ tennis_exploration.ipynb
├─ src/
│  ├─ data_filters.py
│  ├─ data_loader.py
│  ├─ features.py
│  ├─ models.py
│  └─ predictions.py
├─ tests/
│  ├─ test_features.py
│  └─ test_predictions.py
└─ requirements.txt
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

For Streamlit Community Cloud, use `app/streamlit_app.py` as the entrypoint and Python 3.12+. The root `requirements.txt` is detected automatically by Community Cloud.

## Local data override

By default the application reads local files under `data/raw/` when they exist. Set `SPORTS_DATA_DIR` to use another data root.

For the optional OpenFoot loader, set `OPENFOOT_API_KEY` in the hosting environment. Do not commit API keys to the repository.

## Data provenance and licensing

Domestic football data is supplied by Football-Data.co.uk and is used through its public CSV feeds. Current European fixture schedules are loaded from Fixture Download. Tennis data is based on Jeff Sackmann's public ATP/WTA datasets; the archival mirror documents the upstream provenance and CC BY-NC-SA 4.0 terms. If the tennis data is used in a distributed/non-commercial context, retain the required attribution and license terms.

## Responsible use

Model outputs and ELO probabilities are probabilistic estimates, not guarantees of future results, betting returns or profit. This project is intended for research and decision support, not financial advice.

## License

MIT
