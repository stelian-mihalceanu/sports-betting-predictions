# Sports Betting Predictions

Machine-learning decision-support application for football and tennis match predictions, with a browser-based Streamlit interface.

## What is included

- Football and tennis data loaders with local CSV/Parquet support
- Target competition filters for the competitions described below
- Leakage-aware rolling form features
- Pre-match tennis and football ELO ratings
- XGBoost model training helpers with temporal holdout evaluation
- Streamlit web dashboard for browsing loaded datasets
- Pytest regression coverage for the feature-engineering layer

## Target Competitions

### Tennis (ATP & WTA)
- **Grand Slams**: Australian Open, Roland Garros, Wimbledon, US Open
- **ATP Masters 1000**
- **ATP 500**
- **WTA 1000**
- **WTA 500**

### Football
- **European Competitions**: UEFA Champions League, UEFA Europa League, UEFA Conference League
- **Domestic Leagues**: La Liga, Bundesliga, Premier League, Liga 1 (Romania)

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
source venv/bin/activate  # Windows: venv\Scripts\activate
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

The app is a normal web application: after hosting it, open the Streamlit URL in a browser and use the sidebar to switch between Football and Tennis, filter the displayed competition/tour, and inspect the loaded data.

The dashboard intentionally does not invent predictions when no dataset is configured. Add the expected files under `data/raw/` before using the corresponding section.

## Data configuration

By default the application reads from `data/raw/`. Set `SPORTS_DATA_DIR` to use another data root.

For the optional OpenFoot loader, set `OPENFOOT_API_KEY` in the hosting environment. Do not commit API keys to the repository.

## Responsible use

Model outputs are probabilistic estimates and are not guarantees of future results or profit. This project is intended for research and decision support, not financial advice.

## License

MIT
