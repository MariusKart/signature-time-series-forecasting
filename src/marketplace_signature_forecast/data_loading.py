from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Tuple

import pandas as pd
import yfinance as yf
from fredapi import Fred

from .config import FRED_INPUTS, TARGET, YAHOO_INPUTS


DataFrameDict = Dict[str, pd.DataFrame]
SeriesSpec = Mapping[str, str]
SeriesSpecDict = Mapping[str, Mapping[str, str]]


def download_fred_series(fred: Fred, series_id: str, start: str, end: str, name: str) -> pd.DataFrame | None:
    try:
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        df = pd.DataFrame(series, columns=[name])
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index()
        df[name] = df[name].ffill()
        return df
    except Exception:
        return None


def download_yahoo_series(ticker: str, start: str, end: str, name: str) -> pd.DataFrame | None:
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)["Close"]
        df = pd.DataFrame(data)
        df.columns = [name]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df.sort_index()
        df[name] = df[name].ffill()
        return df
    except Exception:
        return None


def fetch_series_bundle(
    api_key: str,
    target: SeriesSpec,
    start: str,
    end: str,
    fred_inputs: SeriesSpecDict | None = None,
    yahoo_inputs: SeriesSpecDict | None = None,
) -> Tuple[pd.DataFrame, DataFrameDict]:
    fred = Fred(api_key=api_key)
    y_raw = download_fred_series(fred, target["code"], start, end, target["name"])
    if y_raw is None:
        raise RuntimeError(f"Target series {target['code']} could not be downloaded from FRED.")

    x_raw: DataFrameDict = {}

    fred_inputs = {} if fred_inputs is None else dict(fred_inputs)
    yahoo_inputs = {} if yahoo_inputs is None else dict(yahoo_inputs)

    for code, info in fred_inputs.items():
        df = download_fred_series(fred, code, start, end, info["name"])
        if df is not None:
            x_raw[info["name"]] = df

    for ticker, info in yahoo_inputs.items():
        df = download_yahoo_series(ticker, start, end, info["name"])
        if df is not None:
            x_raw[info["name"]] = df

    return y_raw, x_raw


def fetch_marketplace_series(api_key: str, start: str, end: str) -> Tuple[pd.DataFrame, DataFrameDict]:
    return fetch_series_bundle(
        api_key=api_key,
        target=TARGET,
        start=start,
        end=end,
        fred_inputs=FRED_INPUTS,
        yahoo_inputs=YAHOO_INPUTS,
    )


def build_data_dictionary_from_specs(
    target: SeriesSpec,
    fred_inputs: SeriesSpecDict | None = None,
    yahoo_inputs: SeriesSpecDict | None = None,
) -> pd.DataFrame:
    rows = [
        {
            "variable": target["name"],
            "type": "target (y)",
            "source": target.get("source", "FRED"),
            "code": target["code"],
            "description": target.get("description", ""),
            "frequency": target.get("frequency", ""),
        }
    ]

    fred_inputs = {} if fred_inputs is None else dict(fred_inputs)
    yahoo_inputs = {} if yahoo_inputs is None else dict(yahoo_inputs)

    for code, info in fred_inputs.items():
        rows.append(
            {
                "variable": info["name"],
                "type": "input (x)",
                "source": info.get("source", "FRED"),
                "code": code,
                "description": info.get("description", ""),
                "frequency": info.get("frequency", ""),
            }
        )

    for ticker, info in yahoo_inputs.items():
        rows.append(
            {
                "variable": info["name"],
                "type": "input (x)",
                "source": info.get("source", "Yahoo Finance"),
                "code": ticker,
                "description": info.get("description", ""),
                "frequency": info.get("frequency", ""),
            }
        )

    return pd.DataFrame(rows)


def build_data_dictionary() -> pd.DataFrame:
    return build_data_dictionary_from_specs(TARGET, FRED_INPUTS, YAHOO_INPUTS)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
