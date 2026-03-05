# Predicting Flight Arrival Delays Using Machine Learning
### A Temporal and Explainable Modelling Approach

Built as part of module MSc Group Project report (COS7048-B), applying machine learning to predict US domestic flight arrival delays using data from 2015 and 2022.

---

> ## Project Overview

This project develops and evaluates a supervised machine learning regression framework for predicting **flight arrival delay in minutes** (`ArrDelayMinutes`) for US domestic flights. The problem is framed as a regression task rather than a classification problem, as predicting the exact delay in minutes preserves temporal precision and enables more detailed operational insight than categorical delay groupings.

The prediction task is defined at the moment of departure — only features available at that point in time are used as model inputs, ensuring no data leakage from post-flight information. Cancelled and diverted flights are excluded, as they represent fundamentally different operational outcomes.

To assess how well the model generalises across different operational periods, the framework is trained on 2015 flight data and evaluated on 2022 data. This strict temporal split avoids data leakage and provides a realistic test of model robustness under different real-world conditions.

Three regression models are benchmarked — Linear Regression as a baseline, Random Forest, and XGBoost — with explainability provided via **SHAP (SHapley Additive exPlanations)** to interpret feature contributions and inform feature refinement.

> **Best Model:** Random Forest (10 features) — MAE: 5.14 mins | RMSE: 9.60 mins | R²: **0.903**

---

## Datasets

Two publicly available datasets from the **US Bureau of Transportation Statistics (BTS)**, accessed via Kaggle:

| Dataset | Year | Kaggle |
|---------|------|--------|
| DS1 — Flight Delay Prediction | 2022 | `whenamancodes/flight-delay-prediction` |
| DS2 — 2015 Flight Delays and Cancellations | 2015 | `usdot/flight-delays` |

> Data is not included in this repository due to file size. Download directly from Kaggle using `kagglehub`, which runs automatically in the notebook on first use.


---

## Pipeline
```
1.  Load DS1 (2022) and DS2 (2015)
2.  Drop diversion columns (99.8% null, post-event leakage)
3.  Drop arrival/post-flight leakage columns
4.  Remove cancelled and diverted flights
5.  Merge datasets into unified DataFrame
6.  Resolve null states via cross-dataset airport lookup
7.  Data cleaning — clip negatives, cap outliers at 99th percentile
8.  Admin/ID column analysis — Pearson correlation, then drop
9.  VIF redundancy analysis — drop CRSElapsedTime and CRSArrTime
10. Feature engineering — extract dep_hour from CRSDepTime
11. Label encoding and StandardScaler normalisation
12. Stratified sampling — 500k from 2015, all of 2022 (~1M rows total)
13. Temporal train/test split — train on 2015, test on 2022
14. Train Linear Regression, Random Forest, XGBoost
15. SHAP feature importance analysis
16. Drop OriginState and DestState based on SHAP findings, retrain
```

---

## Final Feature Set

| Category | Features |
|----------|----------|
| Temporal | Month, DayofMonth, DayOfWeek |
| Carrier | Marketing_Airline_Network |
| Aircraft | Tail_Number |
| Route | Origin, Dest |
| Schedule | dep_hour |
| Operations | DepDelayMinutes, Distance |
| Target | ArrDelayMinutes |

---

## Results

| Model | MAE (mins) | RMSE (mins) | R² |
|-------|-----------|------------|-----|
| Linear Regression | ~5.2 | ~9.8 | ~0.899 |
| **Random Forest** | **5.14** | **9.60** | **0.903** |
| XGBoost | ~5.2 | ~9.7 | ~0.902 |

SHAP analysis confirmed `DepDelayMinutes` as the dominant predictor with a mean SHAP value of 16.0, consistent with established aviation research — the model functions as a sophisticated departure delay propagator. `OriginState` and `DestState` were removed after SHAP showed negligible contributions, with no meaningful drop in accuracy.

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/flight-delay-prediction.git
cd flight-delay-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook msc_grp_project.ipynb
```

Datasets are downloaded automatically via `kagglehub` on first run. Ensure your [Kaggle API credentials](https://www.kaggle.com/docs/api) are configured at `~/.kaggle/kaggle.json`.

> A machine with 16GB+ RAM is recommended. The full dataset is ~6.24M rows, sampled to ~1M for training.

---

## Dependencies

`pandas` `numpy` `scikit-learn` `xgboost` `shap` `statsmodels` `matplotlib` `seaborn` `kagglehub`

See `requirements.txt` for version details.

---

## Licence

MIT
