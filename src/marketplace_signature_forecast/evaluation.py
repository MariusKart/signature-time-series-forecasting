from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import compute_relative_error, forecast_arima, forecast_naive
from .modeling import rolling_forecast



def inverse_transform_forecasts(results: dict, y_scaler) -> dict:
    y_true_std = np.asarray(results["actual"])
    y_pred_std = np.asarray(results["forecast"])
    y_true = y_scaler.inverse_transform(y_true_std.reshape(-1, 1)).ravel()
    y_pred = y_scaler.inverse_transform(y_pred_std.reshape(-1, 1)).ravel()
    errors = y_pred - y_true
    rel_errors = np.abs(errors) / np.maximum(np.abs(y_true), 1e-8)
    return {
        "forecast_orig": y_pred,
        "actual_orig": y_true,
        "error_orig": errors,
        "rel_error_orig": rel_errors,
        "mre": float(100 * np.mean(rel_errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "median_re": float(100 * np.median(rel_errors)),
    }



def build_forecast_dataframe(results: dict, dates, delta_t: int, inverse_results: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_origin": [dates[t] for t in results["t"]],
            "target_date": [dates[t + delta_t] for t in results["t"]],
            "forecast_std": results["forecast"],
            "actual_std": results["actual"],
            "forecast_orig": inverse_results["forecast_orig"],
            "actual_orig": inverse_results["actual_orig"],
            "error_orig": inverse_results["error_orig"],
            "rel_error_orig": inverse_results["rel_error_orig"],
            "n_selected_features": results["n_selected"],
            "n_eff": results["n_eff"],
            "gamma": results["gamma"],
        }
    )



def run_multi_horizon_experiment(
    X: np.ndarray,
    y: np.ndarray,
    dates,
    y_scaler,
    horizons: list[int],
    n_validation: int,
    window_size: int,
    depth: int,
    n_target: int,
    output_dir: str | Path,
    lambda_lasso: float | None = None,
    gamma: float | None = None,
    use_sig_y_only: bool = True,
    add_time: bool = True,
) -> tuple[dict, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_results: dict[int, dict] = {}
    summary_rows = []

    for delta_t in horizons:
        val_start_t = len(y) - n_validation - delta_t
        val_end_t = len(y) - delta_t - 1
        min_required = window_size + delta_t + 20
        if val_start_t < min_required:
            val_start_t = min_required
        if val_start_t > val_end_t:
            continue

        results = rolling_forecast(
            X=X,
            y=y,
            start_t=val_start_t,
            end_t=val_end_t,
            delta_t=delta_t,
            window_size=window_size,
            depth=depth,
            lambda_lasso=lambda_lasso,
            gamma=gamma,
            n_target=n_target,
            use_sig_y_only=use_sig_y_only,
            add_time=add_time,
        )
        if len(results["t"]) == 0:
            continue

        inverse_results = inverse_transform_forecasts(results, y_scaler)
        df = build_forecast_dataframe(results, dates, delta_t, inverse_results)
        df.to_csv(output_dir / f"forecast_results_delta{delta_t}.csv", index=False)

        experiment_results[delta_t] = {**results, **inverse_results}
        summary_rows.append(
            {
                "horizon_weeks": delta_t,
                "signature_mre": inverse_results["mre"],
                "n_forecasts": len(results["t"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "signature_mre_by_horizon.csv", index=False)
    return experiment_results, summary_df



def compare_with_baselines(
    y_raw: np.ndarray,
    horizons: list[int],
    n_validation: int,
    signature_summary: pd.DataFrame,
) -> pd.DataFrame:
    results_table = []
    val_end = len(y_raw) - 1

    for delta_t in horizons:
        val_start = len(y_raw) - n_validation - delta_t
        actual_naive, forecast_naive_values = forecast_naive(y_raw, val_start, val_end - delta_t, delta_t)
        naive_mre = compute_relative_error(actual_naive, forecast_naive_values)

        actual_arima, forecast_arima_values = forecast_arima(y_raw, val_start, val_end - delta_t, delta_t)
        arima_mre = compute_relative_error(actual_arima, forecast_arima_values)

        if delta_t in signature_summary["horizon_weeks"].values:
            sig_mre = float(signature_summary.loc[signature_summary["horizon_weeks"] == delta_t, "signature_mre"].iloc[0])
        else:
            sig_mre = np.nan

        results_table.append(
            {
                "Horizon (weeks)": delta_t,
                "Naive (%)": naive_mre,
                "ARIMA (%)": arima_mre,
                "Signature (%)": sig_mre,
            }
        )

    comparison = pd.DataFrame(results_table)
    comparison["Improvement vs ARIMA (%)"] = (
        (comparison["ARIMA (%)"] - comparison["Signature (%)"]) / comparison["ARIMA (%)"] * 100
    )
    return comparison
