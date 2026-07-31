import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import features


class WeatherBufferTests(unittest.TestCase):
    def setUp(self):
        index = pd.date_range('2026-01-01 00:00', periods=30, freq='h', tz='Europe/Helsinki')
        self.buffer = features.WeatherBuffer(pd.DataFrame({
            'temp': np.arange(30, dtype=float),
            'wind_speed': np.full(30, 2.0),
            'wind_direction_deg': np.full(30, 180.0),
        }, index=index))
        self.timestamp = index[25]

    def test_rolling_hours_excludes_current_timestamp(self):
        self.assertEqual(self.buffer.rolling_hours(self.timestamp, 24, np.mean), 12.5)

    def test_rolling_minutes_uses_preceding_quarter_hours(self):
        # The buffer is hourly, so all preceding quarter-hour lookups within
        # hour 24 resolve to the same known hourly weather value.
        self.assertEqual(self.buffer.rolling_minutes(self.timestamp, 60, np.mean), 24.0)


if __name__ == '__main__':
    unittest.main()
