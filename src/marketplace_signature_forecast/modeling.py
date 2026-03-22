from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.preprocessing import StandardScaler

from .adaptive_weights import calibrate_gamma_at_t, compute_weights_at_t
from .signature import compute_signature, flatten_signature, signature_dimension



def two_step_lasso(
    X_features: np.ndarray,
    y_target: np.ndarray,
    weights: np.ndarray,
    lambda_lasso: float | None = None,
    alpha_cv: int = 10,
) -> dict:
    X_features = np.asarray(X_features, dtype=float)
    y_target = np.asarray(y_target, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    n_samples, n_features = X_features.shape

    if len(y_target) != n_samples or len(weights) != n_samples:
        raise ValueError("X, y, and weights must have same length")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    if lambda_lasso is None:
        alphas = np.logspace(-4, 1, alpha_cv)
        lasso = LassoCV(alphas=alphas, cv=min(5, n_samples), max_iter=10000, random_state=42)
    else:
        lasso = Lasso(alpha=lambda_lasso, max_iter=10000, random_state=42)

    lasso.fit(X_scaled, y_target, sample_weight=weights)
    selected_mask = np.abs(lasso.coef_) > 1e-10
    selected_features = np.where(selected_mask)[0]

    if len(selected_features) == 0:
        selected_mask = np.ones(n_features, dtype=bool)
        selected_features = np.arange(n_features)

    X_selected = X_scaled[:, selected_mask]
    W_sqrt = np.sqrt(weights)
    X_weighted = X_selected * W_sqrt[:, None]
    y_weighted = y_target * W_sqrt

    try:
        coef_selected = np.linalg.lstsq(X_weighted, y_weighted, rcond=None)[0]
    except np.linalg.LinAlgError:
        ridge = Ridge(alpha=1e-6)
        ridge.fit(X_selected, y_target, sample_weight=weights)
        coef_selected = ridge.coef_

    coefficients_scaled = np.zeros(n_features)
    coefficients_scaled[selected_mask] = coef_selected
    coefficients_original = coefficients_scaled / scaler.scale_
    intercept = float(np.mean(y_target) - np.dot(scaler.mean_, coefficients_original))

    return {
        "coefficients": coefficients_original,
        "intercept": intercept,
        "selected_features": selected_features,
        "lasso_model": lasso,
        "scaler": scaler,
        "n_selected": int(len(selected_features)),
        "lambda": float(lasso.alpha_) if hasattr(lasso, "alpha_") else lambda_lasso,
    }



def construct_features_for_forecast(
    X: np.ndarray,
    y: np.ndarray,
    t: int,
    delta_t: int,
    window_size: int,
    depth: int,
    train_idx: np.ndarray | None = None,
    use_sig_y_only: bool = True,
    add_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    min_tau = window_size - 1
    max_tau = t - delta_t
    if max_tau < min_tau:
        raise ValueError(f"Not enough data: max_tau={max_tau} < min_tau={min_tau}")

    valid_indices = np.arange(min_tau, max_tau + 1) if train_idx is None else np.asarray(train_idx, dtype=int)
    n_samples = len(valid_indices)

    d_aug = 2 if add_time else 1
    sig_dim = signature_dimension(d_aug, depth) if use_sig_y_only else signature_dimension(X.shape[1] + 1 + int(add_time), depth)
    d_x = X.shape[1]
    feature_dim = d_x + sig_dim if use_sig_y_only else sig_dim

    features = np.zeros((n_samples, feature_dim))
    targets = np.zeros(n_samples)

    for i, tau in enumerate(valid_indices):
        if use_sig_y_only:
            x_tau = X[tau]
            y_window = y[tau - window_size + 1 : tau + 1]
            sig_flat = flatten_signature(compute_signature(y_window, depth=depth, add_time=add_time))
            features[i] = np.concatenate([x_tau, sig_flat])
        else:
            xy_window = np.column_stack([X[tau - window_size + 1 : tau + 1], y[tau - window_size + 1 : tau + 1]])
            features[i] = flatten_signature(compute_signature(xy_window, depth=depth, add_time=add_time))
        targets[i] = y[tau + delta_t] - y[tau]

    return features, targets, valid_indices



def rolling_forecast(
    X: np.ndarray,
    y: np.ndarray,
    start_t: int,
    end_t: int,
    delta_t: int,
    window_size: int,
    depth: int,
    lambda_lasso: float | None = None,
    gamma: float | None = None,
    n_target: float = 40,
    use_sig_y_only: bool = True,
    add_time: bool = True,
) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    results = {
        "t": [],
        "forecast": [],
        "actual": [],
        "error": [],
        "coefficients": [],
        "intercept": [],
        "n_selected": [],
        "weights": [],
        "gamma": [],
        "n_eff": [],
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
                X,
                y,
                t,
                delta_t,
                window_size,
                depth,
                train_idx=idx,
                use_sig_y_only=use_sig_y_only,
                add_time=add_time,
            )
            model = two_step_lasso(X_train, y_train, weights, lambda_lasso=lambda_lasso)

            if use_sig_y_only:
                x_t = X[t]
                y_window = y[t - window_size + 1 : t + 1]
                sig_flat = flatten_signature(compute_signature(y_window, depth=depth, add_time=add_time))
                X_forecast = np.concatenate([x_t, sig_flat])
            else:
                xy_window = np.column_stack([X[t - window_size + 1 : t + 1], y[t - window_size + 1 : t + 1]])
                X_forecast = flatten_signature(compute_signature(xy_window, depth=depth, add_time=add_time))

            predicted_diff = float(np.dot(X_forecast, model["coefficients"]) + model["intercept"])
            y_forecast = float(y[t] + predicted_diff)
            y_actual = float(y[t + delta_t])

            results["t"].append(t)
            results["forecast"].append(y_forecast)
            results["actual"].append(y_actual)
            results["error"].append(y_forecast - y_actual)
            results["coefficients"].append(model["coefficients"])
            results["intercept"].append(model["intercept"])
            results["n_selected"].append(model["n_selected"])
            results["weights"].append(weights)
            results["gamma"].append(gamma_t)
            results["n_eff"].append(n_eff)
        except (ValueError, np.linalg.LinAlgError):
            continue

    return results
