from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



def resample_to_weekly(df: pd.DataFrame, freq: str = "W-WED") -> pd.DataFrame:
    return df.resample(freq).last()



def resample_collection(data: Dict[str, pd.DataFrame], freq: str = "W-WED") -> Dict[str, pd.DataFrame]:
    return {name: resample_to_weekly(df, freq=freq) for name, df in data.items()}



def build_model_dataset(y_weekly: pd.DataFrame, x_weekly: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    data = y_weekly.copy()
    for _, df in x_weekly.items():
        data = data.join(df, how="inner")
    return data



def train_validation_split(data: pd.DataFrame, n_validation: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return data.iloc[:-n_validation].copy(), data.iloc[-n_validation:].copy()



def save_processed_bundle(
    data: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    data_dictionary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "full_dataset.csv")
    train.to_csv(output_dir / "train_data.csv")
    validation.to_csv(output_dir / "validation_data.csv")
    data_dictionary.to_csv(output_dir / "data_dictionary.csv", index=False)



def load_processed_bundle(output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir = Path(output_dir)
    full_data = pd.read_csv(output_dir / "full_dataset.csv", index_col=0, parse_dates=True)
    train = pd.read_csv(output_dir / "train_data.csv", index_col=0, parse_dates=True)
    validation = pd.read_csv(output_dir / "validation_data.csv", index_col=0, parse_dates=True)
    return full_data, train, validation



def prepare_standardized_arrays(
    full_data: pd.DataFrame,
    target_col: str,
    horizons: list[int],
    n_validation: int,
) -> dict:
    feature_cols = [col for col in full_data.columns if col != target_col]
    X_raw = full_data[feature_cols].values
    y_raw = full_data[target_col].values
    dates = full_data.index

    max_h = max(horizons)
    train_end = len(y_raw) - n_validation - max_h

    X_scaler = StandardScaler().fit(X_raw[: train_end + 1])
    X = X_scaler.transform(X_raw)

    y_scaler = StandardScaler().fit(y_raw[: train_end + 1].reshape(-1, 1))
    y = y_scaler.transform(y_raw.reshape(-1, 1)).ravel()

    return {
        "feature_cols": feature_cols,
        "X_raw": X_raw,
        "y_raw": y_raw,
        "X": X,
        "y": y,
        "dates": dates,
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "train_end": train_end,
    }
