"""Live Streamlit entrypoint for the BetLens sports analytics dashboard."""

# Keep the deploy entrypoint tiny so the UI can evolve in app/dashboard.py
# without maintaining two separate Streamlit applications.
from dashboard import *  # noqa: F401,F403
