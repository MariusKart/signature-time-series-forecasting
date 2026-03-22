from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA



def compute_relative_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float).ravel()
    forecast = np.asarray(forecast, dtype=float).ravel()
    eps = 1e-8
    rel_error = np.abs(forecast - actual) / np.maximum(np.abs(actual), eps)
    return float(np.mean(rel_error) * 100)



def forecast_naive(y: np.ndarray, start_t: int, end_t: int, delta_t: int) -> tuple[np.ndarray, np.ndarray]:
    forecasts = []
    actuals = []
    for t in range(start_t, end_t + 1):
        forecasts.append(y[t])
        actuals.append(y[t + delta_t])
    return np.asarray(actuals), np.asarray(forecasts)



def forecast_arima(
    y: np.ndarray,
    start_t: int,
    end_t: int,
    delta_t: int,
    order: tuple[int, int, int] = (1, 0, 1),
) -> tuple[np.ndarray, np.ndarray]:
    forecasts = []
    actuals = []
    for t in range(start_t, end_t + 1):
        try:
            model = ARIMA(np.asarray(y[: t + 1], dtype=float), order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=delta_t)
            value = forecast.iloc[-1] if hasattr(forecast, "iloc") else forecast[-1]
            forecasts.append(float(value))
            actuals.append(float(y[t + delta_t]))
        except Exception:
            continue
    return np.asarray(actuals), np.asarray(forecasts)



def load_signature_forecast(data_dir: str | Path, delta_t: int) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    path = data_dir / f"forecast_results_delta{delta_t}.csv"
    results = pd.read_csv(path)
    return results["actual_orig"].values, results["forecast_orig"].values
