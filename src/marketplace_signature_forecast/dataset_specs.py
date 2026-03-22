INITIAL_CLAIMS_START_DATE = "2000-01-01"
INITIAL_CLAIMS_END_DATE = "2024-12-31"
INITIAL_CLAIMS_RESAMPLE_FREQ = "W-SAT"
INITIAL_CLAIMS_N_VALIDATION = 26
INITIAL_CLAIMS_PLOT_HORIZON = 4
INITIAL_CLAIMS_EXPERIMENT_NAME = "initial_claims_public"

INITIAL_CLAIMS_TARGET = {
    "code": "ICSA",
    "name": "initial_claims",
    "description": "Initial Claims, seasonally adjusted",
    "source": "FRED",
    "frequency": "Weekly, Ending Saturday",
}

INITIAL_CLAIMS_FRED_INPUTS = {
    "CCSA": {
        "name": "continued_claims",
        "description": "Continued Claims (Insured Unemployment)",
        "frequency": "Weekly, Ending Saturday",
    },
    "NFCI": {
        "name": "nfci",
        "description": "Chicago Fed National Financial Conditions Index",
        "frequency": "Weekly",
    },
    "VIXCLS": {
        "name": "vix",
        "description": "CBOE Volatility Index",
        "frequency": "Daily",
    },
    "T10Y2Y": {
        "name": "term_spread",
        "description": "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity",
        "frequency": "Daily",
    },
    "SP500": {
        "name": "sp500",
        "description": "S&P 500 Index",
        "frequency": "Daily",
    },
    "DCOILWTICO": {
        "name": "wti_crude",
        "description": "WTI Crude Oil Spot Price",
        "frequency": "Daily",
    },
    "FEDFUNDS": {
        "name": "effr",
        "description": "Effective Federal Funds Rate",
        "frequency": "Monthly",
    },
}

INITIAL_CLAIMS_YAHOO_INPUTS = {}
