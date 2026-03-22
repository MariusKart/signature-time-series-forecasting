from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"

FRED_API_KEY = "50358120ca362c7d3785033837493b13"
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
N_VALIDATION = 12

DEFAULT_WINDOW_SIZE = 12
DEFAULT_DEPTH = 2
DEFAULT_N_TARGET = 40
ALL_HORIZONS = [1, 2, 3, 4, 6, 8, 12]

TARGET = {
    "code": "CCLACBW027SBOG",
    "name": "consumer_loans",
    "description": "Consumer Loans: Credit Cards and Other Revolving Plans, All Commercial Banks",
    "source": "FRED",
    "frequency": "Weekly, Ending Wednesday",
}

FRED_INPUTS = {
    "SP500": {
        "name": "sp500",
        "description": "S&P 500 Stock Price",
        "frequency": "Daily",
    },
    "TOTBKCR": {
        "name": "bank_credit",
        "description": "Bank Credit, All Commercial Banks",
        "frequency": "Weekly (Wed)",
    },
    "GASREGW": {
        "name": "gas_price",
        "description": "U.S. Regular All Formulations Gas Price",
        "frequency": "Weekly (Mon)",
    },
    "TOTLL": {
        "name": "loans_leases",
        "description": "Loans and Leases in Bank Credit, All Commercial Banks",
        "frequency": "Weekly (Wed)",
    },
    "SBCACBW027SBOG": {
        "name": "securities",
        "description": "Securities in Bank Credit, All Commercial Banks (Weekly, SA, Wed)",
        "frequency": "Weekly (Wed)",
    },
    "OVXCLS": {
        "name": "ovx",
        "description": "CBOE Crude Oil ETF Volatility Index",
        "frequency": "Daily",
    },
    "DCOILWTICO": {
        "name": "wti_crude",
        "description": "Crude Oil Prices: West Texas Intermediate (WTI)",
        "frequency": "Daily",
    },
    "DCOILBRENTEU": {
        "name": "brent_crude",
        "description": "Crude Oil Prices: Brent - Europe",
        "frequency": "Daily",
    },
}

YAHOO_INPUTS = {
    "AMZN": {
        "name": "amazon_stock",
        "description": "Amazon Inc. Stock Price (Close)",
        "frequency": "Daily",
    }
}


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
