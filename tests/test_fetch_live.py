import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import fetch_live


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FetchPriceTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp('2026-07-10 00:00', tz='Europe/Helsinki')
        self.end = self.start + pd.Timedelta(minutes=45)

    @patch('fetch_live.requests.get')
    def test_fetches_fi_prices_at_quarter_hour_resolution(self, get):
        timestamps = pd.date_range(self.start, periods=4, freq='15min').tz_convert('UTC')
        get.return_value = _Response({'success': True, 'data': {'fi': [
            {'timestamp': int(timestamp.timestamp()), 'price': price}
            for timestamp, price in zip(timestamps, [10.0, 11.0, 12.0, 13.0])
        ]}})

        result = fetch_live.fetch_prices(self.start, self.end)

        self.assertEqual(result.tolist(), [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(str(result.index.tz), 'Europe/Helsinki')

    @patch('fetch_live.requests.get')
    def test_rejects_suspicious_all_zero_market_response(self, get):
        timestamps = pd.date_range(self.start, periods=24, freq='15min').tz_convert('UTC')
        get.return_value = _Response({'success': True, 'data': {'fi': [
            {'timestamp': int(timestamp.timestamp()), 'price': 0.0} for timestamp in timestamps
        ]}})

        with self.assertRaisesRegex(RuntimeError, 'All returned FI prices are zero'):
            fetch_live.fetch_prices(self.start, self.start + pd.Timedelta(hours=6))


if __name__ == '__main__':
    unittest.main()
