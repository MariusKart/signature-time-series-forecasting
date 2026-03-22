from __future__ import annotations

import numpy as np

from .signature import compute_signature, signature_kernel_distance



def ada_weight_sig(
    X: np.ndarray,
    y: np.ndarray,
    delta_t: int,
    window_size: int,
    depth: int,
    gamma: float,
    add_time: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(y) != X.shape[0]:
        raise ValueError(f"X and y must have same length: {X.shape[0]} vs {len(y)}")

    n_valid = len(y) - delta_t
    z = np.column_stack([X[:n_valid], y[delta_t : delta_t + n_valid]])
    n_samples = n_valid - window_size + 1
    if n_samples < 1:
        raise ValueError("Not enough data for the chosen horizon and window size.")

    ref_window = z[-window_size:]
    ref_sig = compute_signature(ref_window, depth=depth, add_time=add_time)

    distances = np.zeros(n_samples - 1)
    for tau in range(n_samples - 1):
        hist_window = z[tau : tau + window_size]
        hist_sig = compute_signature(hist_window, depth=depth, add_time=add_time)
        distances[tau] = signature_kernel_distance(hist_sig, ref_sig)

    scale = np.median(distances) + 1e-12
    logits = -(gamma / scale) * distances
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights, distances



def pick_gamma_by_neff(
    gammas: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    delta_t: int,
    window_size: int,
    depth: int,
    add_time: bool,
    n_target: float,
) -> dict:
    best = None
    for gamma in gammas:
        weights, _ = ada_weight_sig(X, y, delta_t, window_size, depth, float(gamma), add_time=add_time)
        n_eff = 1.0 / np.sum(weights ** 2)
        score = abs(n_eff - n_target)
        candidate = {"gamma": float(gamma), "n_eff": float(n_eff), "score": float(score)}
        if best is None or score < best["score"]:
            best = candidate
    return best



def compute_weights_at_t(
    X: np.ndarray,
    y: np.ndarray,
    t: int,
    delta_t: int,
    window_size: int,
    depth: int,
    gamma: float,
    add_time: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = len(y)
    ref_end = t - delta_t
    ref_start = ref_end - window_size + 1
    if ref_start < 0:
        raise ValueError(f"Not enough data for reference window at t={t}")

    first_tau = window_size - 1
    last_tau = ref_end - 1
    if last_tau < first_tau:
        raise ValueError(f"Not enough training samples at t={t}")

    n_z = ref_end + 1
    if n_z + delta_t > T:
        raise ValueError(f"y index out of bounds at t={t}")

    z = np.column_stack([X[:n_z], y[delta_t : delta_t + n_z]])
    ref_window = z[ref_start : ref_end + 1]
    ref_sig = compute_signature(ref_window, depth=depth, add_time=add_time)

    valid_indices = np.arange(first_tau, last_tau + 1)
    distances = np.zeros(len(valid_indices))

    for i, tau in enumerate(valid_indices):
        window = z[tau - window_size + 1 : tau + 1]
        hist_sig = compute_signature(window, depth=depth, add_time=add_time)
        distances[i] = signature_kernel_distance(hist_sig, ref_sig)

    scale = np.median(distances) + 1e-12
    logits = -(gamma / scale) * distances
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights, distances, valid_indices



def calibrate_gamma_at_t(
    X: np.ndarray,
    y: np.ndarray,
    t: int,
    delta_t: int,
    window_size: int,
    depth: int,
    n_target: float,
    gammas: np.ndarray | None = None,
    add_time: bool = True,
) -> dict | None:
    gammas = np.logspace(-3, 1, 25) if gammas is None else gammas
    best = None
    for gamma in gammas:
        try:
            weights, _, _ = compute_weights_at_t(
                X,
                y,
                t,
                delta_t,
                window_size,
                depth,
                float(gamma),
                add_time=add_time,
            )
        except ValueError:
            continue
        n_eff = 1.0 / np.sum(weights ** 2)
        score = abs(n_eff - n_target)
        candidate = {"gamma": float(gamma), "n_eff": float(n_eff), "score": float(score)}
        if best is None or score < best["score"]:
            best = candidate
    return best



def rolling_adaptive_weights(
    X: np.ndarray,
    y: np.ndarray,
    start_t: int,
    end_t: int,
    delta_t: int,
    window_size: int,
    depth: int,
    gamma: float | None = None,
    n_target: float = 40,
    add_time: bool = True,
) -> dict:
    results = {"t": [], "weights": [], "distances": [], "valid_indices": [], "gamma": [], "n_eff": []}
    for t in range(start_t, end_t + 1):
        try:
            if gamma is None:
                calib = calibrate_gamma_at_t(X, y, t, delta_t, window_size, depth, n_target, add_time=add_time)
                gamma_t = 1.0 if calib is None else calib["gamma"]
            else:
                gamma_t = gamma
            weights, distances, valid_indices = compute_weights_at_t(
                X,
                y,
                t,
                delta_t,
                window_size,
                depth,
                gamma_t,
                add_time=add_time,
            )
        except ValueError:
            continue
        results["t"].append(t)
        results["weights"].append(weights)
        results["distances"].append(distances)
        results["valid_indices"].append(valid_indices)
        results["gamma"].append(gamma_t)
        results["n_eff"].append(float(1.0 / np.sum(weights ** 2)))
    return results
