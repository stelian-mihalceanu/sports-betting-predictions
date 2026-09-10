# Sports Betting Predictions

Machine learning application for football and tennis match predictions.

## Target Competitions

### Tennis (ATP & WTA)
- **Grand Slams**: Australian Open, Roland Garros, Wimbledon, US Open
- **ATP Masters 1000**: Indian Wells, Miami, Monte-Carlo, Madrid, Roma, Canada, Cincinnati, Shanghai, Paris
- **ATP 500**: Rotterdam, Dubai, Barcelona, Queen's, Hamburg, Washington, Beijing, Tokyo, Vienna, Basel, etc.
- **WTA 1000**: Doha/Dubai, Indian Wells, Miami, Madrid, Roma, Canada, Cincinnati, Wuhan, Beijing
- **WTA 500**: Abu Dhabi, Stuttgart, Berlin, Eastbourne, San Diego, Tokyo, Guadalajara, etc.

### Football
- **European Competitions**: UEFA Champions League, UEFA Europa League, UEFA Conference League
- **Domestic Leagues**: La Liga, Bundesliga, Premier League, Liga 1 (Romania)

## Project Structure

```
sports-betting-predictions/
├─ data/
│  ├─ raw/
│  ├─ processed/
├─ notebooks/
│  ├─ tennis_exploration.ipynb
│  ├─ football_exploration.ipynb
├─ src/
│  ├─ data_loader.py
│  ├─ data_filters.py
│  ├─ features.py
│  ├─ models.py
│  ├─ backtest.py
│  ├─ predictions.py
├─ app/
│  ├─ streamlit_app.py
├─ requirements.txt
├─ README.md
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Sources

### Tennis
- Jeff Sackmann datasets: `jeffsackmann/tennis_atp`, `jeffsackmann/tennis_wta` (GitHub)
- Optional APIs: OddsPapi, SportMonks

### Football
- OpenFoot API (includes SuperLiga Româ¬¢niei + major leagues + European cups)
- Football-Data.org (CSV for major leagues)
- API-FOOTBALL / Live-Football-API

## Features

- Data loading and filtering for target competitions
- Feature engineering (form, ELO ratings, home/away stats)
- ML models: XGBoost, Logistic Regression, Poisson/Dixon-Coles
- Backtesting framework
- Streamlit dashboard for predictions

## Usage

```bash
# Run Streamlit app
streamlit run app/streamlit_app.py
```

## License

MIT
