from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_filters import filter_football_target, filter_tennis_target
from src.data_loader import load_football_matches, load_tennis_atp, load_tennis_wta, load_upcoming_football_fixtures
from src.extra_football import load_poland_history, load_poland_upcoming
from src.features import add_elo_features, add_football_form_features
from src.predictions import build_team_stats, predict_match
from src.tennis_predictions import predict_tennis_match

st.set_page_config(page_title="Sports Betting Predictions", page_icon="🏆", layout="wide")
st.title("🏆 Sports Betting Predictions")

# The remaining application UI is unchanged; this file's relevant patch hook is
# added below at the tennis matchup metrics.

