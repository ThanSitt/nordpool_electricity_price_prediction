"""Runtime configuration shared by local runs and GitHub Actions.

This module intentionally contains no credentials.  The price and weather
sources used by the live predictor are public APIs, so a clone can run without
creating an untracked configuration file.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = ROOT / 'models' / 'saved'
PREDICTIONS_DIR = ROOT / 'predictions'
HELSINKI = 'Europe/Helsinki'

# Public sources.  Keep URLs here so an operator can audit or replace them.
ELERING_PRICE_URL = 'https://dashboard.elering.ee/api/nps/price'
OPEN_METEO_FORECAST_URL = 'https://api.open-meteo.com/v1/forecast'

FORECAST_HOURS = 7 * 24
PRICE_HISTORY_HOURS = 200
