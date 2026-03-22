from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yfinance as yf
from fredapi import Fred

from .config import FRED_INPUTS, TARGET, YAHOO_INPUTS


DataFrameDict = Dict[str, pd.DataFrame]


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



def fetch_marketplace_series(api_key: str, start: str, end: str) -> Tuple[pd.DataFrame, DataFrameDict]:
    fred = Fred(api_key=api_key)
    y_raw = download_fred_series(fred, TARGET["code"], start, end, TARGET["name"])
    if y_raw is None:
        raise RuntimeError("Target series could not be downloaded from FRED.")

    x_raw: DataFrameDict = {}
    for code, info in FRED_INPUTS.items():
        df = download_fred_series(fred, code, start, end, info["name"])
        if df is not None:
            x_raw[info["name"]] = df

    for ticker, info in YAHOO_INPUTS.items():
        df = download_yahoo_series(ticker, start, end, info["name"])
        if df is not None:
            x_raw[info["name"]] = df

    return y_raw, x_raw



def build_data_dictionary() -> pd.DataFrame:
    rows = [
        {
            "variable": TARGET["name"],
            "type": "target (y)",
            "source": TARGET["source"],
            "code": TARGET["code"],
            "description": TARGET["description"],
        }
    ]

    for code, info in FRED_INPUTS.items():
        rows.append(
            {
                "variable": info["name"],
                "type": "input (x)",
                "source": "FRED",
                "code": code,
                "description": info["description"],
            }
        )

    for ticker, info in YAHOO_INPUTS.items():
        rows.append(
            {
                "variable": info["name"],
                "type": "input (x)",
                "source": "Yahoo Finance",
                "code": ticker,
                "description": info["description"],
            }
        )

    return pd.DataFrame(rows)



def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
