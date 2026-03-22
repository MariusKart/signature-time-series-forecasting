from __future__ import annotations

import numpy as np



def tensor_exp(v: np.ndarray, depth: int) -> list[np.ndarray]:
    v = np.asarray(v, dtype=float).ravel()
    levels = [None] * (depth + 1)
    levels[0] = np.array([1.0])
    if depth == 0:
        return levels
    levels[1] = v.copy()
    for k in range(2, depth + 1):
        levels[k] = np.kron(levels[k - 1], v) / k
    return levels



def chen_product(a: list[np.ndarray], b: list[np.ndarray], depth: int) -> list[np.ndarray]:
    out = [None] * (depth + 1)
    for k in range(depth + 1):
        acc = None
        for j in range(k + 1):
            term = np.kron(a[j], b[k - j])
            acc = term if acc is None else acc + term
        out[k] = acc
    return out



def compute_signature(path: np.ndarray, depth: int, add_time: bool = False, times: np.ndarray | None = None) -> list[np.ndarray]:
    x = np.asarray(path, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"path must have shape (L, d), got {x.shape}")
    length, dimension = x.shape
    if length < 2:
        raise ValueError("path must have at least 2 points")

    if add_time:
        if times is None:
            t = np.arange(length, dtype=float)
        else:
            t = np.asarray(times, dtype=float)
            if t.shape != (length,):
                raise ValueError(f"times must have shape ({length},)")
        x = np.column_stack([t, x])
        dimension += 1

    sig = [None] * (depth + 1)
    sig[0] = np.array([1.0])
    for k in range(1, depth + 1):
        sig[k] = np.zeros(dimension ** k, dtype=float)

    for i in range(length - 1):
        increment = x[i + 1] - x[i]
        sig = chen_product(sig, tensor_exp(increment, depth), depth)
    return sig



def flatten_signature(levels: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([level.ravel() for level in levels], axis=0)



def signature_dimension(dimension: int, depth: int) -> int:
    if dimension == 1:
        return depth + 1
    return (dimension ** (depth + 1) - 1) // (dimension - 1)



def signature_kernel(sig_a: list[np.ndarray], sig_b: list[np.ndarray]) -> float:
    return float(np.dot(flatten_signature(sig_a), flatten_signature(sig_b)))



def signature_kernel_distance(sig_a: list[np.ndarray], sig_b: list[np.ndarray]) -> float:
    k_aa = signature_kernel(sig_a, sig_a)
    k_bb = signature_kernel(sig_b, sig_b)
    k_ab = signature_kernel(sig_a, sig_b)
    return max(0.0, float(k_aa - 2.0 * k_ab + k_bb))



def compute_signature_features(data: np.ndarray, window_size: int, depth: int, add_time: bool = True) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_rows, n_channels = data.shape
    if window_size < 2:
        raise ValueError("window_size must be at least 2")
    if window_size > n_rows:
        raise ValueError(f"window_size ({window_size}) > data length ({n_rows})")

    dimension = n_channels + 1 if add_time else n_channels
    sig_dim = signature_dimension(dimension, depth)
    n_samples = n_rows - window_size + 1
    features = np.zeros((n_samples, sig_dim))

    for i in range(n_samples):
        window = data[i : i + window_size]
        times = np.arange(window_size, dtype=float) if add_time else None
        features[i] = flatten_signature(compute_signature(window, depth=depth, add_time=add_time, times=times))

    return features
