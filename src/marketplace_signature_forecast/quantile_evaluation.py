from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from .quantile_modeling import rolling_quantile_forecast



def inverse_transform_quantile_results(results: dict, y_scaler) -> dict:
    actual_std = np.asarray(results["actual"], dtype=float)
    actual_orig = y_scaler.inverse_transform(actual_std.reshape(-1, 1)).ravel()

    quantile_forecasts_orig: dict[float, np.ndarray] = {}
    for quantile, values in results["quantile_forecasts"].items():
        forecast_std = np.asarray(values, dtype=float)
        quantile_forecasts_orig[quantile] = y_scaler.inverse_transform(forecast_std.reshape(-1, 1)).ravel()

    return {
        "actual_orig": actual_orig,
        "quantile_forecasts_orig": quantile_forecasts_orig,
    }



def build_quantile_forecast_dataframe(
    results: dict,
    dates,
    delta_t: int,
    inverse_results: dict,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "forecast_origin": [dates[t] for t in results["t"]],
            "target_date": [dates[t + delta_t] for t in results["t"]],
            "actual_std": results["actual"],
            "actual_orig": inverse_results["actual_orig"],
            "gamma": results["gamma"],
            "n_eff": results["n_eff"],
        }
    )

    for quantile, values in results["quantile_forecasts"].items():
        q_label = str(quantile).replace(".", "_")
        df[f"q{q_label}_std"] = values
        df[f"q{q_label}_orig"] = inverse_results["quantile_forecasts_orig"][quantile]

    if 0.5 in results["quantile_forecasts"]:
        df["median_error_orig"] = df["q0_5_orig"] - df["actual_orig"]
        df["median_rel_error_orig"] = np.abs(df["median_error_orig"]) / np.maximum(np.abs(df["actual_orig"]), 1e-8)

    return df



def summarize_quantile_forecasts(
    actual: np.ndarray,
    quantile_forecasts: dict[float, np.ndarray],
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> dict:
    actual = np.asarray(actual, dtype=float).ravel()
    summary: dict[str, float] = {}

    for quantile, forecast in quantile_forecasts.items():
        forecast = np.asarray(forecast, dtype=float).ravel()
        summary[f"pinball_q{str(quantile).replace('.', '_')}"] = float(
            mean_pinball_loss(actual, forecast, alpha=quantile)
        )

    if 0.5 in quantile_forecasts:
        median_forecast = np.asarray(quantile_forecasts[0.5], dtype=float).ravel()
        errors = median_forecast - actual
        rel_errors = np.abs(errors) / np.maximum(np.abs(actual), 1e-8)
        summary["median_mae"] = float(np.mean(np.abs(errors)))
        summary["median_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
        summary["median_mre"] = float(100 * np.mean(rel_errors))

    if lower_quantile in quantile_forecasts and upper_quantile in quantile_forecasts:
        lower = np.asarray(quantile_forecasts[lower_quantile], dtype=float).ravel()
        upper = np.asarray(quantile_forecasts[upper_quantile], dtype=float).ravel()
        summary["interval_coverage"] = float(np.mean((actual >= lower) & (actual <= upper)))
        summary["avg_interval_width"] = float(np.mean(upper - lower))

    return summary



def run_multi_horizon_quantile_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dates,
    y_scaler,
    horizons: list[int],
    n_validation: int,
    window_size: int,
    depth: int,
    quantiles: list[float],
    output_dir: str | Path,
    alpha: float = 1e-2,
    gamma: float | None = None,
    n_target: float = 40,
    use_sig_y_only: bool = True,
    add_time: bool = True,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
    solver: str = "highs",
) -> tuple[dict, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_results: dict[int, dict] = {}
    summary_rows: list[dict] = []

    for delta_t in horizons:
        val_start_t = len(y) - n_validation - delta_t
        val_end_t = len(y) - delta_t - 1
        min_required = window_size + delta_t + 20
        if val_start_t < min_required:
            val_start_t = min_required
        if val_start_t > val_end_t:
            continue

        results = rolling_quantile_forecast(
            X=X,
            y=y,
            start_t=val_start_t,
            end_t=val_end_t,
            delta_t=delta_t,
            window_size=window_size,
            depth=depth,
            quantiles=quantiles,
            alpha=alpha,
            gamma=gamma,
            n_target=n_target,
            use_sig_y_only=use_sig_y_only,
            add_time=add_time,
            solver=solver,
        )
        if len(results["t"]) == 0:
            continue

        inverse_results = inverse_transform_quantile_results(results, y_scaler)
        df = build_quantile_forecast_dataframe(results, dates, delta_t, inverse_results)
        df.to_csv(output_dir / f"quantile_forecast_results_delta{delta_t}.csv", index=False)

        summary = summarize_quantile_forecasts(
            actual=inverse_results["actual_orig"],
            quantile_forecasts=inverse_results["quantile_forecasts_orig"],
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )
        summary["horizon_weeks"] = delta_t
        summary["n_forecasts"] = len(results["t"])
        summary_rows.append(summary)
        experiment_results[delta_t] = {**results, **inverse_results, **summary}

    summary_df = pd.DataFrame(summary_rows).sort_values("horizon_weeks") if summary_rows else pd.DataFrame()
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "quantile_summary_by_horizon.csv", index=False)
    return experiment_results, summary_df
