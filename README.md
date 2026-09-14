# BetLens — Sports Betting Predictions

Research-first Streamlit dashboard for football, tennis and basketball analytics.

## What changed in v2

The app is now structured around a fast cached dashboard instead of recalculating a large data pipeline on every UI interaction.

- Football: upcoming signals, Match Center, Poisson goal markets, Elo, recent form, xG, BTTS, corners and cards.
- Tennis: surface-aware Elo, recent form, serve/return profile, recency-weighted H2H and workload/fatigue signal.
- Basketball: transparent pace/net-rating/Elo baseline for moneyline, spread and total research.
- Model Validation: chronological football backtest with accuracy, Brier score and log-loss.
- Data loading: Streamlit cache (24h for historical data, 15m for current fixtures), concurrent feed requests and Parquet-ready processed storage.

## Performance architecture

The historical football dataset is loaded through a cached loader and a persisted `data/processed/football_history.parquet` is preferred when available. Current fixtures use a shorter 15-minute cache because schedules can change during the day. Network requests are concurrent and failures degrade to an empty result instead of breaking the dashboard.

For production, a scheduled GitHub Actions job can materialize the processed Parquet dataset and keep Streamlit focused on serving predictions rather than rebuilding history.

## Validation

The validation page deliberately uses a chronological rolling evaluation rather than a random split. Metrics are:

- 1X2 accuracy
- multiclass Brier score
- log-loss

Confidence is presented as model probability, not as a claim of guaranteed accuracy. Calibration utilities are included for future reliability diagrams and threshold analysis.

## Data

Football historical/current results use Football-Data.co.uk public CSV feeds. Tennis support expects ATP/WTA historical match archives in `data/raw/`. The application does not scrape bookmaker pages; value analysis should use user-entered or authorized odds feeds.

## Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Tests:

```bash
pytest -q
```

## Project structure

```text
app/
  streamlit_app.py
  prediction_dashboard.py
src/
  backtesting.py
  basketball_predictions.py
  calibration.py
  data_loader.py
  features.py
  predictions.py
  tennis_predictions.py
  models.py
tests/
```

## Responsible use

All model outputs are probabilistic estimates for research and decision support. They are not guarantees of results, betting returns or profit. Historical backtests are not promises of future performance.

## License

MIT
