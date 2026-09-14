"""Production Streamlit entrypoint for the sports prediction dashboard."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.prediction_dashboard import run_app

run_app()
