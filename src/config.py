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

# Fingrid API — cross-border flows and nuclear generation.
# API key must be supplied via the FINGRID_API_KEY environment variable.
FINGRID_API_URL = 'https://data.fingrid.fi/api/datasets/{dataset_id}/data'
# Dataset IDs verified against catalog (2026-08-25):
#   55  Finland ↔ Estonia, power measured every 15 min
#   57  Finland ↔ Norway, measured every 15 min
#   60  Finland ↔ Northern Sweden (SE1), power measured every 15 min
#   61  Finland ↔ Central Sweden (SE3), power measured every 15 min
#  188  Nuclear power production, real-time (~3 min resolution)
FINGRID_GRID_DATASETS = {
    'fi_ee':         55,
    'fi_no':         57,
    'fi_se_north':   60,
    'fi_se_central': 61,
}
FINGRID_NUCLEAR_DATASET = 188
GRID_HISTORY_HOURS = 200  # same window as prices; covers the 7-day lag
