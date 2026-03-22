from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler

from .adaptive_weights import calibrate_gamma_at_t, compute_weights_at_t
from .modeling import build_design_vector, construct_features_for_forecast


@dataclass
class WeightedQuantileModel:
    quantile: float
    regressor: QuantileRegressor
    scaler: StandardScaler



def fit_weighted_quantile_regression(
    X_features: np.ndarray,
    y_target: np.ndarray,
    weights: np.ndarray,
    quantile: float,
    alpha: float = 1e-2,
    solver: str = "highs",
) -> WeightedQuantileModel:
    X_features = np.asarray(X_features, dtype=float)
    y_target = np.asarray(y_target, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()

    if X_features.ndim != 2:
        raise ValueError("X_features must be a 2D array")
    if len(y_target) != len(X_features) or len(weights) != len(X_features):
        raise ValueError("X, y, and weights must have the same number of rows")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0 and 1")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    regressor = QuantileRegressor(quantile=quantile, alpha=alpha, fit_intercept=True, solver=solver)
    regressor.fit(X_scaled, y_target, sample_weight=weights)
    return WeightedQuantileModel(quantile=quantile, regressor=regressor, scaler=scaler)



def fit_weighted_quantile_models(
    X_features: np.ndarray,
    y_target: np.ndarray,
    weights: np.ndarray,
    quantiles: list[float],
    alpha: float = 1e-2,
    solver: str = "highs",
) -> dict[float, WeightedQuantileModel]:
    models = {}
    for quantile in sorted(quantiles):
        models[quantile] = fit_weighted_quantile_regression(
            X_features=X_features,
            y_target=y_target,
            weights=weights,
            quantile=float(quantile),
            alpha=alpha,
            solver=solver,
        )
    return models



def predict_weighted_quantiles(
    models: dict[float, WeightedQuantileModel],
    X_forecast: np.ndarray,
) -> dict[float, float]:
    X_forecast = np.asarray(X_forecast, dtype=float).reshape(1, -1)
    quantiles = sorted(models)
    predictions = []
    for quantile in quantiles:
        model = models[quantile]
        pred = model.regressor.predict(model.scaler.transform(X_forecast))[0]
        predictions.append(float(pred))

    monotone = np.maximum.accumulate(np.asarray(predictions, dtype=float))
    return {quantile: float(value) for quantile, value in zip(quantiles, monotone)}



def rolling_quantile_forecast(
    X: np.ndarray,
    y: np.ndarray,
    start_t: int,
    end_t: int,
    delta_t: int,
    window_size: int,
    depth: int,
    quantiles: list[float],
    alpha: float = 1e-2,
    gamma: float | None = None,
    n_target: float = 40,
    use_sig_y_only: bool = True,
    add_time: bool = True,
    solver: str = "highs",
) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    ordered_quantiles = sorted(float(q) for q in quantiles)
    results = {
        "t": [],
        "actual": [],
        "weights": [],
        "gamma": [],
        "n_eff": [],
        "quantile_forecasts": {q: [] for q in ordered_quantiles},
    }

    for t in range(start_t, end_t + 1):
        try:
            if gamma is None:
                calib = calibrate_gamma_at_t(X, y, t, delta_t, window_size, depth, n_target, add_time=add_time)
                gamma_t = 1.0 if calib is None else calib["gamma"]
            else:
                gamma_t = gamma

            weights, _, idx = compute_weights_at_t(X, y, t, delta_t, window_size, depth, gamma_t, add_time)
            n_eff = float(1.0 / np.sum(weights ** 2))

            X_train, y_train, _ = construct_features_for_forecast(
                X=X,
                y=y,
                t=t,
                delta_t=delta_t,
                window_size=window_size,
                depth=depth,
                train_idx=idx,
                use_sig_y_only=use_sig_y_only,
                add_time=add_time,
            )
            models = fit_weighted_quantile_models(
                X_features=X_train,
                y_target=y_train,
                weights=weights,
                quantiles=ordered_quantiles,
                alpha=alpha,
                solver=solver,
            )
            X_forecast = build_design_vector(
                X=X,
                y=y,
                t=t,
                window_size=window_size,
                depth=depth,
                use_sig_y_only=use_sig_y_only,
                add_time=add_time,
            )
            predicted_diff = predict_weighted_quantiles(models, X_forecast)
            level_forecasts = {q: float(y[t] + predicted_diff[q]) for q in ordered_quantiles}
            y_actual = float(y[t + delta_t])

            results["t"].append(t)
            results["actual"].append(y_actual)
            results["weights"].append(weights)
            results["gamma"].append(gamma_t)
            results["n_eff"].append(n_eff)
            for quantile in ordered_quantiles:
                results["quantile_forecasts"][quantile].append(level_forecasts[quantile])
        except (ValueError, np.linalg.LinAlgError):
            continue

    return results
