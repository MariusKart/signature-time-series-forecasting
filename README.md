# Transportation Marketplace Rate Forecast Using Signature Transform

This repository studies **signature-based forecasting for sequential data**. The core idea is to turn a recent path into a finite-dimensional feature vector through the **signature transform**, then fit a sparse forecasting model on top of those path features. The project also includes an **uncertainty-forecasting extension** based on weighted quantile regression, so the model can produce prediction intervals in addition to point forecasts.

## Why signatures?

For a path \(X_t\), the signature is the collection of its iterated integrals. In practice, a truncated signature acts as a compact nonlinear summary of:
- the order in which moves happened,
- the interaction between channels,
- the geometry of the recent path.

This is useful for regression because many forecasting problems are not driven only by current levels. They depend on **how the path evolved**. A linear model on signature features can therefore capture nonlinear path dependence without requiring a very large black-box architecture.

## Adaptive weighting with the signature kernel

The project also uses a **signature-kernel distance** to compare the current path with historical windows. Those distances are converted into adaptive sample weights, so the regression focuses more on past periods that look similar to the current regime.

At a high level, the workflow is:

1. download and align the time series,
2. build weekly forecasting datasets,
3. compute signature-based features over rolling windows,
4. adapt sample weights using signature similarity,
5. fit the forecasting model,
6. evaluate across multiple horizons.

## Uncertainty forecasting

The probabilistic extension keeps the same signature features and adaptive weighting, but replaces the point-forecast head with **weighted quantile regression**. This produces conditional quantiles such as 10%, 50%, and 90%, which can be turned into prediction intervals and evaluated through:
- **pinball loss**,
- **interval coverage**,
- **average interval width**.

In other words, the repository now supports both:
- **point forecasting**, and
- **uncertainty forecasting via conditional quantiles**.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── data/
│   └── processed/
├── figures/
├── notebooks/
│   ├── 01_reproduce_gu_signature_forecast.ipynb
│   ├── 02_initial_claims_public_data_experiment.ipynb
│   ├── 03_uncertainty_forecasting_consumer_loans.ipynb
│   └── 04_uncertainty_forecasting_initial_claims.ipynb
└── src/
    └── marketplace_signature_forecast/
        ├── __init__.py
        ├── config.py
        ├── dataset_specs.py
        ├── data_loading.py
        ├── preprocessing.py
        ├── signature.py
        ├── adaptive_weights.py
        ├── modeling.py
        ├── baselines.py
        ├── evaluation.py
        ├── quantile_modeling.py
        ├── quantile_evaluation.py
        └── plotting.py
```

## Notebooks

### 1. `01_reproduce_gu_signature_forecast.ipynb`
Point-forecasting experiment on the original **consumer-loans** dataset.

### 2. `02_initial_claims_public_data_experiment.ipynb`
Point-forecasting experiment on a second fully public dataset built around **U.S. initial jobless claims**.

### 3. `03_uncertainty_forecasting_consumer_loans.ipynb`
Probabilistic extension of the original consumer-loans experiment. It estimates conditional quantiles and evaluates prediction intervals.

### 4. `04_uncertainty_forecasting_initial_claims.ipynb`
Probabilistic extension of the public initial-claims experiment.

## Main modules

### `signature.py`
Implements the signature transform and the signature-kernel utilities used to compare rolling windows.

### `adaptive_weights.py`
Builds adaptive weights from path similarity and calibrates the temperature parameter through an effective-sample-size target.

### `modeling.py`
Contains the point-forecasting pipeline based on signature features and two-step LASSO.

### `quantile_modeling.py`
Contains the probabilistic extension based on weighted quantile regression.

### `evaluation.py`
Runs rolling point forecasts and exports horizon-by-horizon evaluation outputs.

### `quantile_evaluation.py`
Runs rolling probabilistic forecasts and exports quantile, interval, and calibration metrics.

### `dataset_specs.py`
Stores reusable specifications for the public initial-claims experiment.

## Datasets used in the notebooks

### Consumer-loans experiment
- target: `CCLACBW027SBOG`
- weekly alignment ending on Wednesday
- macro and market covariates from FRED and Yahoo Finance

### Initial-claims experiment
- target: `ICSA`
- weekly alignment ending on Saturday
- macro and market covariates from FRED

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the notebooks

```bash
jupyter notebook notebooks/
```

Run the notebooks from top to bottom. Processed outputs are written under `data/processed/`, and figures are written under `figures/`.

## Outputs

Depending on the notebook, the repository writes:
- processed datasets,
- forecast tables by horizon,
- summary comparison tables,
- prediction-interval plots,
- baseline comparison outputs.

## Notes

This repository uses public time series and is designed as a reproducible research project around signature-based regression and probabilistic forecasting.
