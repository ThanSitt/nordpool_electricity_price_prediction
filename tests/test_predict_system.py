import sys
import unittest
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import predict_system


class EvaluationResolutionTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range('2026-07-10 00:00', periods=4, freq='15min', tz='Europe/Helsinki')
        self.actuals = pd.Series([10.0, 20.0, 30.0, 40.0], index=self.index)

    def test_quarter_hour_model_uses_matching_actual_price(self):
        frame = pd.DataFrame({
            'run_date': [self.index[0]],
            'target_datetime': [self.index[2]],
            'predicted_price': [25.0],
            'actual_price': [float('nan')],
            'abs_error': [float('nan')],
        })
        result = predict_system.fill_actuals(frame, self.actuals, 15)
        self.assertEqual(result.at[0, 'actual_price'], 30.0)
        self.assertEqual(result.at[0, 'abs_error'], 5.0)


class NativeResolutionForecastTests(unittest.TestCase):
    class ConstantModel:
        def predict(self, frame):
            return [1.0] * len(frame)

    def test_quarter_hour_model_keeps_all_four_native_steps(self):
        price_index = pd.date_range('2026-07-01 00:00', periods=800, freq='15min', tz='Europe/Helsinki')
        weather_index = pd.date_range('2026-07-01 00:00', periods=300, freq='h', tz='Europe/Helsinki')
        prices = pd.Series(10.0, index=price_index)
        weather = pd.DataFrame({
            'temp': 10.0,
            'wind_speed': 2.0,
            'wind_direction_deg': 180.0,
        }, index=weather_index)

        original_horizon = predict_system.FORECAST_HOURS
        predict_system.FORECAST_HOURS = 1
        try:
            forecast = predict_system.run_forecast(
                self.ConstantModel(), ['temp'], 15, weather_index[200],
                predict_system.feat_lib.PriceBuffer(prices),
                predict_system.feat_lib.WeatherBuffer(weather),
            )
        finally:
            predict_system.FORECAST_HOURS = original_horizon

        self.assertEqual(len(forecast), 4)
        self.assertEqual([timestamp.minute for timestamp, _ in forecast], [0, 15, 30, 45])

    def test_hourly_model_uses_hourly_mean_not_the_last_quarter(self):
        index = pd.date_range('2026-07-10 00:00', periods=4, freq='15min', tz='Europe/Helsinki')
        actuals = pd.Series([10.0, 20.0, 30.0, 40.0], index=index)
        frame = pd.DataFrame({
            'run_date': [index[0]],
            'target_datetime': [index[0]],
            'predicted_price': [20.0],
            'actual_price': [float('nan')],
            'abs_error': [float('nan')],
        })
        result = predict_system.fill_actuals(frame, actuals, 60)
        self.assertEqual(result.at[0, 'actual_price'], 25.0)
        self.assertEqual(result.at[0, 'abs_error'], 5.0)


if __name__ == '__main__':
    unittest.main()
