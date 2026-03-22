from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .adaptive_weights import compute_weights_at_t


plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")



def save_or_show(fig, path: str | Path | None = None):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()



def plot_target_split(train: pd.DataFrame, validation: pd.DataFrame, target_col: str, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(train.index, train[target_col], label="Training", linewidth=1.5)
    ax.plot(validation.index, validation[target_col], label="Validation", linewidth=2, color="red")
    ax.axvline(x=train.index[-1], color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.set_title("Target series and train/validation split")
    ax.legend()
    save_or_show(fig, path)



def plot_input_grid(data: pd.DataFrame, target_col: str, path: str | Path | None = None):
    input_cols = [col for col in data.columns if col != target_col]
    n_cols = 3
    n_rows = int(np.ceil(len(input_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, col in enumerate(input_cols):
        axes[i].plot(data.index, data[col], linewidth=1)
        axes[i].set_title(col, fontsize=10)
        axes[i].tick_params(axis="x", rotation=45)

    for j in range(len(input_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle("External factors", fontsize=14, y=1.02)
    fig.tight_layout()
    save_or_show(fig, path)



def plot_correlation_matrix(data: pd.DataFrame, path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Correlation matrix")
    save_or_show(fig, path)



def plot_adaptive_weight_diagnostics(results: dict, dates_y, y, train_end_t: int, n_target: int, target_col: str):
    t_dates = [dates_y[t] for t in results["t"]]
    max_w = [weights.max() for weights in results["weights"]]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(dates_y, y, color="black", linewidth=1.2, label="Target (processed)")
    axes[0].axvline(dates_y[train_end_t], color="gray", linestyle="--", linewidth=1.5, label="Train / validation split")
    axes[0].set_title(f"Target variable (processed): {target_col}")
    axes[0].legend(frameon=False)

    axes[1].plot(t_dates, results["n_eff"], color="tab:blue", linewidth=1.2)
    axes[1].axhline(n_target, color="gray", linestyle="--", linewidth=1.2, label=f"Target n_eff = {n_target}")
    axes[1].set_ylabel("n_eff")
    axes[1].set_title("Effective sample size across forecast origins")
    axes[1].legend(frameon=False)

    axes[2].semilogy(t_dates, results["gamma"], color="tab:green", linewidth=1.2)
    axes[2].set_ylabel("γ (log scale)")
    axes[2].set_title("Calibrated temperature parameter")

    axes[3].plot(t_dates, max_w, color="tab:orange", linewidth=1.2)
    axes[3].set_ylabel("max weight")
    axes[3].set_xlabel("Date")
    axes[3].set_title("Weight concentration")
    fig.tight_layout()
    plt.show()



def plot_weights_for_origin(results: dict, k: int, X: np.ndarray, y: np.ndarray, dates_y, delta_t: int, window_size: int, depth: int):
    t = results["t"][k]
    gamma = results["gamma"][k]
    weights, distances, idx = compute_weights_at_t(X, y, t, delta_t, window_size, depth, gamma, add_time=True)
    n_eff = 1.0 / np.sum(weights ** 2)
    idx_dates = [dates_y[i] for i in idx]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
    axes[0].plot(dates_y[: t + delta_t + 1], y[: t + delta_t + 1], color="black", linewidth=1.2, label="Target (processed)")
    axes[0].axvline(dates_y[t], color="gray", linestyle="--", linewidth=1.5, label="Forecast origin t")
    axes[0].scatter([dates_y[t + delta_t]], [y[t + delta_t]], color="gray", s=60, zorder=5, label=r"Target $y_{t+\Delta t}$")
    ref_end = t - delta_t
    ref_start = ref_end - window_size + 1
    axes[0].axvspan(dates_y[ref_start], dates_y[ref_end], color="gray", alpha=0.15, label="Reference window")
    axes[0].set_title(f"Forecast origin t = {t} ({dates_y[t].date()})")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].bar(idx_dates, distances, width=5, alpha=0.8, color="tab:blue")
    axes[1].set_ylabel("distance")
    axes[1].set_title("Signature-kernel distance to the reference window")

    axes[2].bar(idx_dates, weights, width=5, alpha=0.8, color="tab:orange")
    axes[2].set_ylabel("weight")
    axes[2].set_xlabel("date")
    axes[2].set_title(f"Adaptive weights (γ = {gamma:.4f}, n_eff = {n_eff:.1f})")
    fig.tight_layout()
    plt.show()



def plot_forecast_vs_actual(forecast_df: pd.DataFrame, delta_t: int):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(forecast_df["target_date"], forecast_df["actual_orig"], "o-", label="Actual", linewidth=2, markersize=6)
    axes[0].plot(forecast_df["target_date"], forecast_df["forecast_orig"], "s-", label="Forecast", linewidth=2, markersize=6, alpha=0.7)
    axes[0].set_title(f"{delta_t}-week ahead forecast", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Consumer Loans (original scale)")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)

    errors = forecast_df["forecast_orig"] - forecast_df["actual_orig"]
    axes[1].bar(forecast_df["target_date"], errors, alpha=0.7, color="coral", edgecolor="black")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Forecast errors", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Error (Forecast - Actual)")
    axes[1].grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    plt.show()
