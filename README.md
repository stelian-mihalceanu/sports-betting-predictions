# Sports Betting Predictions

Machine-learning decision-support application for football and tennis match analytics, with a browser-based Streamlit interface.

## Live dashboard

The Streamlit app can run without committing large raw datasets to this repository. When local files are absent, the application loads public data at runtime:

- **Football:** Football-Data.co.uk historical/current CSV feeds for Premier League, Bundesliga, La Liga and Liga 1 (Romania)
- **Tennis:** an archival mirror of Jeff Sackmann's ATP/WTA match datasets, covering recent seasons through 2026

The app caches remote data in Streamlit for a limited period to reduce repeated downloads.

## What is included

- Football and tennis data loaders with local CSV/Parquet support and public-data fallback
- Target competition filters
- Leakage-aware rolling form features
- Pre-match tennis and football ELO ratings
- Football upcoming-fixture 1X2 estimates based on pre-match ELO
- XGBoost model training helpers with temporal holdout evaluation
- Streamlit web dashboard for analytics and upcoming football estimates
- Pytest regression coverage for feature engineering and tournament categorization

## Supported competitions

### Tennis (ATP & WTA)
- **Grand Slams**: Australian Open, Roland Garros, Wimbledon, US Open
- **ATP Masters 1000**
- **ATP 500**
- **WTA 1000**
- **WTA 500**

### Football
The automatic public feed currently covers:
- **Premier League**
- **Bundesliga**
- **La Liga**
- **Liga 1 (Romania)**

The filtering layer also contains names/IDs for UEFA Champions League, Europa League and Conference League, but those European competitions are not included in the current no-key public loader.

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
│  └─ models.py
├─ tests/
│  └─ test_features.py
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

For Streamlit Community Cloud, use `app/streamlit_app.py` as the entrypoint and Python 3.12. The root `requirements.txt` is detected automatically by Community Cloud.

## Local data override

By default the application reads local files under `data/raw/` when they exist. Set `SPORTS_DATA_DIR` to use another data root.

For the optional OpenFoot loader, set `OPENFOOT_API_KEY` in the hosting environment. Do not commit API keys to the repository.

## Data provenance and licensing

Football data is supplied by Football-Data.co.uk and is used through its public CSV feeds. Tennis data is based on Jeff Sackmann's public ATP/WTA datasets; the archival mirror documents the upstream provenance and CC BY-NC-SA 4.0 terms. If the tennis data is used in a distributed/non-commercial context, retain the required attribution and license terms.

## Responsible use

Model outputs and ELO probabilities are probabilistic estimates, not guarantees of future results, betting returns or profit. This project is intended for research and decision support, not financial advice.

## License

MIT
