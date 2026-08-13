"""Plot the forecast CSVs produced by src/predict_system.py.

For every ``<model>_forecasts.csv`` in ``predictions/`` this script:

1. draws that model's own line chart (predicted price over the 7 days),
2. saves it as ``charts/<model>.png``.

It also draws ONE comparison chart with all models on the same axes and saves
it as ``charts/comparison.png``.

Run it after a forecast run:

    python src/plot_predictions.py

Notes for beginners
-------------------
- A CSV stores raw numbers; a chart is just a visualisation of those numbers.
  This script only reads the CSVs and draws pictures -- it does not predict
  anything itself.
- ``matplotlib`` is the library that draws the lines.  It is already installed
  (see requirements.txt).
"""

from pathlib import Path

import matplotlib

# Save images to files without trying to open a pop-up window.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd

import config

PREDICTIONS_DIR = config.PREDICTIONS_DIR
CHART_DIR = Path(__file__).resolve().parent.parent / 'charts'

# ── style (cosmetic, safe to tweak) ───────────────────────────────────────────
FIG_WIDTH = 14
FIG_HEIGHT = 6
DPI = 150
# One colour per model so the comparison chart is readable.
MODEL_COLORS = {
    'xgboost_v1':   '#9e9e9e',
    'xgboost_v1_5': '#6d6d6d',
    'xgboost_v2':   '#1f77b4',
    'xgboost_v2_5': '#2ca02c',
    'xgboost_v2_5_2': '#9467bd',
    'lightgbm_v2':  '#ff7f0e',
    'lightgbm_v2_5': '#d62728',
}


def load_forecasts() -> list[tuple[str, pd.DataFrame]]:
    """Return ``(model_name, dataframe)`` for every forecast CSV found."""
    frames = []
    for path in sorted(PREDICTIONS_DIR.glob('*_forecasts.csv')):
        frame = pd.read_csv(path, parse_dates=['run_date', 'target_datetime'])
        # Make sure the time axis is timezone-aware, in Helsinki time.
        if frame['target_datetime'].dt.tz is None:
            frame['target_datetime'] = (
                frame['target_datetime'].dt.tz_localize('UTC')
                .dt.tz_convert(config.HELSINKI)
            )
        else:
            frame['target_datetime'] = (
                frame['target_datetime'].dt.tz_convert(config.HELSINKI)
            )
        name = path.stem.replace('_forecasts', '')
        frames.append((name, frame))
    return frames


def _finish_and_save(fig, filename: str) -> None:
    """Tidy up a figure and save it as a PNG."""
    fig.autofmt_xdate()  # rotate the date labels so they don't overlap
    fig.tight_layout()
    out = CHART_DIR / filename
    fig.savefig(out, dpi=DPI)
    plt.close(fig)  # free the memory used by this figure
    print(f'  saved {out}')


def plot_single(name: str, frame: pd.DataFrame) -> None:
    """Draw one chart for one model."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # The main line: predicted price over the 7 future days.
    ax.plot(
        frame['target_datetime'],
        frame['predicted_price'],
        label='Predicted price (EUR/MWh)',
        color=MODEL_COLORS.get(name, '#1f77b4'),
        linewidth=1.5,
    )

    # Overlay real prices where they already exist (dashed black line).
    actual = frame.dropna(subset=['actual_price'])
    if not actual.empty:
        ax.plot(
            actual['target_datetime'],
            actual['actual_price'],
            label='Actual price (EUR/MWh)',
            color='black',
            linestyle='--',
            linewidth=1.5,
        )

    ax.set_title(f'{name} - 7-day electricity price forecast')
    ax.set_xlabel('Time (Europe/Helsinki)')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    _finish_and_save(fig, f'{name}.png')


def plot_comparison(frames: list[tuple[str, pd.DataFrame]]) -> None:
    """Draw one chart with every model on the same axes."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for name, frame in frames:
        ax.plot(
            frame['target_datetime'],
            frame['predicted_price'],
            label=name,
            color=MODEL_COLORS.get(name),
            linewidth=1.2,
        )

    ax.set_title('All models - 7-day electricity price forecast (comparison)')
    ax.set_xlabel('Time (Europe/Helsinki)')
    ax.set_ylabel('Price (EUR/MWh)')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    _finish_and_save(fig, 'comparison.png')


def main() -> None:
    CHART_DIR.mkdir(exist_ok=True)

    frames = load_forecasts()
    if not frames:
        raise SystemExit(
            f'No forecast CSVs found in {PREDICTIONS_DIR}. '
            'Run "python src/predict_system.py" first.'
        )

    print(f'Found {len(frames)} forecast files, plotting ...')
    for name, frame in frames:
        plot_single(name, frame)
    plot_comparison(frames)
    print(f'All charts saved to {CHART_DIR}')


if __name__ == '__main__':
    main()
