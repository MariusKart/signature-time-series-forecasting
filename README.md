# Transportation Marketplace Rate Forecast Using Signature Transform

This repository implements a signature-based forecasting pipeline for weekly time series regression, inspired by *Transportation Marketplace Rate Forecast Using Signature Transform by Gu et al.* It was developed for the first-term Statistical Methods coursework in the Imperial College London MSc in Mathematics and Finance (2026–27).

The core idea is to replace hand-crafted lag features with **path signatures**: compact summaries of sequential data built from iterated integrals. In practice, this lets us transform a nonlinear forecasting problem on historical paths into a tractable regression problem on signature features, while still preserving temporal order and higher-order interactions.

## Why signature features?

Given a path $X : [a,b] \to \mathbb{R}^d$, its signature is the collection

$$
\mathrm{Sig}(X) = \left(1, \int dX, \int dX \otimes dX, \int dX \otimes dX \otimes dX, \dots \right).
$$

The truncated signature of depth $N$ keeps only terms up to order $N$, yielding a finite-dimensional feature map.

This construction is useful for regression because:

- **it captures ordered interactions** between observations, not just pointwise levels or lags;
- **it is expressive**: linear functionals of signatures can approximate a broad class of nonlinear functionals of paths;
- **it is stable and structured** for sequential data, especially after **time augmentation**, which appends time as an additional channel;
- **it supports kernel-based similarity measures**, allowing the model to upweight historical windows that resemble the current regime.

In this repository, signature features are combined with an **adaptive two-step LASSO**:
1. a signature kernel is used to compute similarity-based sample weights;
2. a weighted LASSO selects predictive features;
3. an OLS refit is performed on the selected support.

This gives an interpretable regression pipeline that can adapt to changing regimes while remaining much lighter than a deep sequence model.

## What this repository does

The code builds a complete forecasting workflow for a weekly target series using public macro and market covariates:

- downloads and aligns the raw series;
- resamples everything to a common **weekly Wednesday** frequency;
- constructs rolling path features;
- computes truncated signatures and signature-kernel adaptive weights;
- fits the two-step LASSO across multiple forecast horizons;
- compares the signature-based model with baseline methods.

The main target used here is:

- **Consumer Loans: Credit Cards and Other Revolving Plans, All Commercial Banks** (`CCLACBW027SBOG`)

The explanatory variables used in the notebooks include market and macroeconomic proxies such as:

- S&P 500,
- Amazon stock,
- WTI oil price,
- unemployment rate,
- effective federal funds rate,
- industrial production,
- personal consumption expenditures,
- consumer sentiment,
- nonfarm payrolls.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   ├── full_dataset.csv
│   ├── train_data.csv
│   ├── validation_data.csv
│   ├── data_dictionary.csv
│   ├── forecast_results_delta*.csv
│   ├── signature_mre_by_horizon.csv
│   └── table_a3_full_comparison.csv
├── figures/
├── notebooks/
│   └── 01_reproduce_gu_signature_forecast.ipynb
└── src/
    └── marketplace_signature_forecast/
        ├── __init__.py
        ├── adaptive_weights.py
        ├── baselines.py
        ├── config.py
        ├── data_loading.py
        ├── evaluation.py
        ├── modeling.py
        ├── plotting.py
        ├── preprocessing.py
        └── signature.py
```

## Module overview

### `data_loading.py`
Utilities to download the raw series from FRED and Yahoo Finance.

### `preprocessing.py`
Weekly resampling, alignment, merging, train/validation split, and dataset export.

### `signature.py`
Core signature-transform logic:
- truncated signature computation,
- signature dimension utilities,
- signature kernel and signature-kernel distance.

### `adaptive_weights.py`
Implements the similarity-based weighting scheme used to emphasize historical windows close to the current forecasting regime.

### `modeling.py`
Feature construction and weighted **two-step LASSO** for forecasting.

### `baselines.py`
Baseline forecasting models used for comparison, including naive and ARIMA-style forecasts.

### `evaluation.py`
Backtesting utilities and horizon-by-horizon error summaries.

### `plotting.py`
Exploratory plots, diagnostics for adaptive weights, and forecast-vs-actual charts.

## Forecasting pipeline

The notebook runs the following sequence:

### 1. Data preparation
All series are downloaded, cleaned, and resampled to weekly Wednesday frequency.
A merged dataset is then split into:

- **training set**,
- **12-week validation set**.

### 2. Path construction and signature features
For each forecast origin, rolling windows are extracted and converted into path objects.
The implementation uses **time augmentation** and a **truncated signature** to turn each path into a finite-dimensional feature vector.

### 3. Adaptive weighting via signature kernel
For a given forecast origin, the current market regime is compared with past windows using the **signature kernel**.
This produces observation weights that give more influence to historically similar periods.

### 4. Two-step LASSO regression
The forecasting model is fitted in two stages:

- **weighted LASSO** for variable selection,
- **OLS refit** on the selected features.

This balances flexibility, sparsity, and interpretability.

### 5. Multi-horizon evaluation
The full procedure is repeated for several forecast horizons:

$$
\Delta t \in \{1,2,3,4,6,8,12\}.
$$

Forecast accuracy is then compared against baseline models.

## Outputs

Running the notebook produces:

- `data/full_dataset.csv`: merged weekly dataset,
- `data/train_data.csv` and `data/validation_data.csv`: train/validation split,
- `data/forecast_results_delta*.csv`: forecast results for each horizon,
- `data/signature_mre_by_horizon.csv`: summary error table for the signature model,
- `data/table_a3_full_comparison.csv`: comparison with baselines,
- figures saved under `figures/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Launch the notebook and run it from top to bottom:

```bash
jupyter notebook notebooks/01_reproduce_gu_signature_forecast.ipynb
```


## Reference

Haotian Gu, Xin Guo, Timothy L. Jacobs, Philip Kaminsky, and Xinyu Li, *Transportation Marketplace Rate Forecast Using Signature Transform*.
