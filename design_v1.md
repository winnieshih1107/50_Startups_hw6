# 50 Startups — CRISP-DM Expert Analysis Design Document

**Version:** 1.0  
**Date:** 2026-06-11  
**Dataset:** Kaggle — 50 Startups (`50_Startups.csv`)  
**Script:** `50_startups_expert.py`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Phase 1 · Business Understanding](#2-phase-1--business-understanding)
3. [Phase 2 · Data Understanding](#3-phase-2--data-understanding)
4. [Phase 3 · Data Preparation](#4-phase-3--data-preparation)
5. [Phase 4 · Modeling](#5-phase-4--modeling)
6. [Phase 5 · Evaluation](#6-phase-5--evaluation)
7. [Phase 6 · Deployment](#7-phase-6--deployment)
8. [Expert Design Decisions](#8-expert-design-decisions)
9. [File Structure](#9-file-structure)
10. [Key Findings Summary](#10-key-findings-summary)

---

## 1. Project Overview

### Problem Statement

A venture capital firm wants to predict the **annual profit** of a startup given its budget allocation across R&D, Administration, and Marketing, and its operating U.S. state. The goal is to help investors identify which spending categories have the highest return, and to build a model that can score new investment candidates.

### Dataset

| Property       | Value                                        |
|----------------|----------------------------------------------|
| Source         | Kaggle — 50 Startups                         |
| File           | `50_Startups.csv`                            |
| Rows           | 50                                           |
| Columns        | 5 (4 input features + 1 target)              |
| Target         | `Profit` (continuous USD)                    |
| Problem type   | Supervised regression                        |

### Raw Features

| Column            | Type    | Description                           |
|-------------------|---------|---------------------------------------|
| `R&D Spend`       | float64 | Annual research & development budget  |
| `Administration`  | float64 | Annual overhead / operational costs   |
| `Marketing Spend` | float64 | Annual sales and advertising budget   |
| `State`           | string  | Operating state: NY / CA / FL         |
| `Profit`          | float64 | Annual net profit — **target**        |

---

## 2. Phase 1 · Business Understanding

### Objective

> **Predict a startup's annual Profit from its spending allocation to guide investor capital decisions.**

### Business Questions

1. Which spending category (R&D / Admin / Marketing) drives profit most?
2. Does geographic location (State) meaningfully affect profit?
3. Given a new startup's budget breakdown, what profit can we forecast?

### Success Criteria

| Metric | Target |
|--------|--------|
| Test R² | ≥ 0.90 |
| Test RMSE | < $15,000 |
| CV stability | std(R²) < 0.10 across folds |

### Constraints

- **Sample size is very small (n = 50).** All modeling and evaluation decisions must account for this — overfitting risk is high, and single-split CV estimates are unreliable.
- Model must remain **interpretable** for stakeholder presentation.

---

## 3. Phase 2 · Data Understanding

### Data Quality Checks

| Check              | Result                                                              |
|--------------------|---------------------------------------------------------------------|
| Missing values     | **None** — dataset is complete                                      |
| Zero values        | R&D Spend: 2 zeros; Marketing Spend: 3 zeros — valid business data |
| Duplicate rows     | None                                                                |
| Skewness           | All features within ±0.5 — no log transform needed on raw features  |
| Outliers (IQR×1.5) | 1 borderline Profit value ($14,681) — legitimate, retained          |

### Descriptive Statistics

| Statistic | R&D Spend | Administration | Marketing Spend | Profit    |
|-----------|-----------|----------------|-----------------|-----------|
| mean      | $73,722   | $121,345       | $211,025        | $112,013  |
| std       | $45,902   | $28,018        | $122,290        | $40,306   |
| min       | $0        | $51,283        | $0              | $14,681   |
| median    | $73,051   | $122,700       | $212,716        | $107,978  |
| max       | $165,349  | $182,646       | $471,784        | $192,262  |

### Correlation with Profit

| Feature           | Pearson r | Strength           |
|-------------------|-----------|--------------------|
| R&D Spend         | **+0.973**| Very strong        |
| Marketing Spend   | +0.748    | Strong             |
| Administration    | +0.201    | Weak               |

### Categorical Feature: State

| State      | Count | Mean Profit |
|------------|-------|-------------|
| New York   | 17    | $113,756    |
| California | 17    | $103,905    |
| Florida    | 16    | $118,774    |

State differences (~$15k spread) are smaller than one standard deviation of Profit ($40k) — **not a reliable discriminator**.

### Full Correlation Matrix

|                 | R&D Spend | Administration | Marketing Spend | Profit |
|-----------------|-----------|----------------|-----------------|--------|
| R&D Spend       | 1.000     | 0.242          | 0.724           | **0.973** |
| Administration  | 0.242     | 1.000          | −0.032          | 0.201  |
| Marketing Spend | 0.724     | −0.032         | 1.000           | 0.748  |
| Profit          | 0.973     | 0.201          | 0.748           | 1.000  |

**Notable:** R&D Spend and Marketing Spend are correlated (r = 0.72) — both rise with company size, causing mild shared variance.

### Output

- Figure saved: `phase2_understanding.png`
  - Row 0: Histograms for R&D, Admin, Marketing, Profit
  - Row 1: Correlation heatmap | R&D vs Profit scatter (coloured by State) | State bar chart

---

## 4. Phase 3 · Data Preparation

### 3.1 Feature Engineering

Four derived features are created before the train/test split. The scaler is still fit **only on training data** inside the Pipeline, so there is no data leakage.

| Feature        | Formula                                           | Rationale                                                  |
|----------------|---------------------------------------------------|------------------------------------------------------------|
| `Total_Spend`  | R&D + Administration + Marketing                  | Company size proxy                                         |
| `RD_Ratio`     | R&D Spend / Total_Spend                           | Research intensity — strategic positioning signal          |
| `Log_RD`       | log(1 + R&D Spend)                                | Captures diminishing returns; compresses large-value range |
| `Admin_Ratio`  | Administration / Total_Spend                      | Overhead burden — efficiency signal                        |

### 3.2 Feature Selection Decision

Five independent methods were applied in the prior analysis to select features. Results:

| Feature          | Correlation | SelectKBest | RFE | LassoCV | Random Forest | Votes | Decision |
|------------------|:-----------:|:-----------:|:---:|:-------:|:-------------:|:-----:|:--------:|
| R&D Spend        | ✔           | ✔           | ✔   | ✔       | ✔             | 5/5   | **KEEP** |
| Marketing Spend  | ✔           | ✔           | ✔   | ✔       | ✗             | 4/5   | **KEEP** |
| Administration   | ✗           | ✔           | ✔   | ✔       | ✗             | 3/5   | Keep (raw) → **replace with ratio** |
| State_Florida    | ✗           | ✗           | ✗   | ✗       | ✗             | 0/5   | **DROP** |
| State_New York   | ✗           | ✗           | ✗   | ✗       | ✗             | 0/5   | **DROP** |

**State** is dropped entirely — not one-hot encoded — as it has zero predictive power at n = 50. This is a **sample size limitation**, not a claim that geography is universally irrelevant.

### 3.3 Feature Sets Compared

Two feature sets are evaluated side-by-side throughout Phases 4 and 5:

```python
FEATURE_SETS = {
    "baseline":   ["R&D Spend", "Administration", "Marketing Spend"],
    "engineered": ["R&D Spend", "Marketing Spend", "RD_Ratio", "Log_RD"],
}
```

| Set          | Logic                                                       |
|--------------|-------------------------------------------------------------|
| `baseline`   | Replicates the original script — 3 raw numeric features     |
| `engineered` | Expert version — replaces Administration with ratio features|

### 3.4 Train / Test Split

- **Ratio:** 80% train (40 rows) / 20% test (10 rows)
- **`random_state = 42`** for reproducibility
- Identical indices used for both feature sets (fair comparison)

> With n = 50, the test set is only 10 rows. Individual test-set metrics have wide confidence intervals. CV estimates (Phase 4) are more reliable than a single test evaluation.

### 3.5 Preprocessing Pipeline

`StandardScaler` is applied inside a `sklearn.pipeline.Pipeline`, fitted on training data only:

```
Pipeline
  └── StandardScaler    (fit on X_train only — no leakage)
  └── Model
```

Scaling is necessary because:
- Feature standard deviations differ by up to 4× (R&D: $45k vs Admin: $28k vs Marketing: $122k)
- Ridge, Lasso, ElasticNet penalise large raw coefficients — unscaled inputs bias regularisation

### 3.6 Cross-Validation Strategy — RepeatedKFold

**Problem with plain KFold(5) on n = 50:**  
The existing script produced CV R² scores of [0.89, −0.81, −0.42, −0.70, +0.43] — a range of 1.71. This is caused by which 10 rows land in each validation fold being highly influential at small n.

**Fix:** `RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)`  
- 50 validation scores per model instead of 5
- Mean and std become substantially more stable
- Still uses the same training data — no additional data required

```python
CV = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
```

### Output

- Figure saved: `phase3_preparation.png`
  - RD_Ratio distribution histogram
  - Log_RD distribution histogram
  - Before-scaling boxplots (raw scale comparison)

---

## 5. Phase 4 · Modeling

### Models

Six models are trained, each inside a `Pipeline(StandardScaler → model)`:

| Model                      | Type           | Hyperparameter strategy                          |
|----------------------------|----------------|--------------------------------------------------|
| `LinearRegression`         | Linear         | No hyperparameters                               |
| `Ridge`                    | Linear (L2)    | `alpha` auto-selected via `RidgeCV`              |
| `Lasso`                    | Linear (L1)    | `alpha` auto-selected via `LassoCV`              |
| `ElasticNet`               | Linear (L1+L2) | `alpha` and `l1_ratio` via `ElasticNetCV`        |
| `RandomForestRegressor`    | Tree ensemble  | `n_estimators=300`, `random_state=42`            |
| `GradientBoostingRegressor`| Tree ensemble  | `n_estimators=200`, `random_state=42`            |

**Alpha search grid:** `np.logspace(-3, 4, 100)` — 100 log-spaced values from 0.001 to 10,000.  
**ElasticNet l1_ratio grid:** `[0.1, 0.3, 0.5, 0.7, 0.9]`

### Training Protocol

```
For each feature_set in {baseline, engineered}:
    For each model in {LR, Ridge, Lasso, ElasticNet, RF, GBM}:
        1. cross_val_score(model, X_train, y_train,
                           cv=RepeatedKFold(5×10), scoring="r2")
        2. cross_val_score(... scoring="neg_mean_squared_error")
        3. model.fit(X_train, y_train)   # store fitted pipeline
```

Total: **12 model × feature-set combinations**, each scored with **50 CV iterations**.

### Output

- Figure saved: `phase4_cv_results.png`
  - Grouped bar chart: CV R² — baseline vs engineered per model
  - Grouped bar chart: CV RMSE — baseline vs engineered per model
  - Red dashed line marks the R² = 0.90 target

---

## 6. Phase 5 · Evaluation

### Best Model Selection

The combination with the highest **CV R² mean** across all 12 candidates is selected as the best model.

### Metrics Reported

For every model × feature-set combination, the following are computed on the held-out test set:

| Metric       | Formula                                          | Why it matters                              |
|--------------|--------------------------------------------------|---------------------------------------------|
| R²           | 1 − SS_res / SS_tot                              | Proportion of variance explained            |
| Adjusted R²  | 1 − (1−R²)(n−1)/(n−p−1)                         | Penalises unnecessary features              |
| RMSE         | √(mean((y − ŷ)²))                               | Error in same unit as Profit (USD)          |
| MAE          | mean(|y − ŷ|)                                   | Robust to outliers; easy to explain         |
| MAPE         | mean(|y − ŷ| / y) × 100                         | Percentage error — scale-independent        |

### Residual Diagnostics

- **Residuals vs Predicted:** checks for heteroscedasticity (variance should be constant)
- **Residual histogram:** checks for approximate normality (assumption of linear regression)
- **Residual mean ≈ 0:** verifies no systematic bias

### Output

- Figure saved: `phase5_evaluation.png` (2×3 grid)

| Position | Content |
|----------|---------|
| (0,0) | Actual vs Predicted scatter with perfect-fit line and residual drops |
| (0,1) | Residuals vs Predicted |
| (0,2) | Residual distribution histogram |
| (1,0) | Feature coefficients (linear) or feature importances (tree) |
| (1,1) | All 12 models ranked by Test R² — horizontal bar chart |
| (1,2) | Performance summary table (CV vs Test) |

---

## 7. Phase 6 · Deployment

### Serialisation

The best pipeline is refit on the **full dataset** (train + test combined) before saving, to maximise use of all 50 observations:

```python
best_pipe_full.fit(X_full, y)
joblib.dump(best_pipe_full, "50_startups_best_model.pkl")
```

The serialised object contains the complete pipeline (scaler + model) — a single `.predict()` call handles preprocessing automatically.

### Prediction API

```python
def predict_profit(rd_spend: float,
                   administration: float,
                   marketing_spend: float) -> float:
    """
    Return predicted annual profit (USD) for a startup.
    Computes engineered features internally.
    """
```

The function:
1. Computes `Total_Spend`, `RD_Ratio`, `Log_RD` from the three raw inputs
2. Selects only the features used by the best model
3. Calls `pipeline.predict()` and returns a float

### Demo Predictions

| Startup     | R&D Spend | Administration | Marketing | Predicted Profit |
|-------------|-----------|----------------|-----------|-----------------|
| High R&D    | $150,000  | $120,000       | $300,000  | ~$175,000        |
| Mid spender | $50,000   | $100,000       | $100,000  | ~$90,000         |
| No R&D      | $0        | $80,000        | $50,000   | ~$50,000         |

### Monitoring Recommendations

| Trigger | Action |
|---------|--------|
| Monthly RMSE on new data > $15,000 | Retrain pipeline |
| R&D Spend input > $165,349 (training max) | Warn: extrapolation risk |
| Marketing Spend input > $471,784 (training max) | Warn: extrapolation risk |
| n_new ≥ 50 additional observations | Collect and retrain |

---

## 8. Expert Design Decisions

This section documents the rationale behind decisions that differ from a standard textbook CRISP-DM approach.

### Decision 1 · RepeatedKFold instead of KFold

| | Standard approach | Expert approach |
|---|---|---|
| **CV method** | `KFold(n_splits=5)` | `RepeatedKFold(n_splits=5, n_repeats=10)` |
| **CV scores** | 5 scores | 50 scores |
| **R² range observed** | −0.81 to +0.89 (unstable) | Substantially narrower |
| **Why** | Default in most tutorials | n=50 makes single splits too sensitive to which 10 rows land in validation |

### Decision 2 · Drop State (not one-hot encode)

Most tutorials would one-hot encode State and include it. This design drops it because:

- OLS p-values: Florida = 0.953, New York = 0.990
- 0 out of 5 feature selection methods retain either dummy
- 16–17 samples per state is insufficient statistical power to detect geographic effects
- Adding near-zero predictors to a small dataset increases overfitting risk

> **Important caveat:** State may become significant with n ≥ 500. The drop decision is data-size-driven, not domain-driven.

### Decision 3 · Replace Administration with ratio features

| | Raw Administration | Expert replacement |
|---|---|---|
| **Correlation with Profit** | r = +0.20 (weak) | — |
| **OLS p-value** | 0.61 (not significant) | — |
| **Standardised coefficient** | −1,841 (negative after controlling others) | — |
| **Problem** | Absolute overhead doesn't indicate efficiency | — |
| **Replacement** | — | `Admin_Ratio` = Admin / Total_Spend |
| **Business meaning** | — | Fraction of budget consumed by overhead |

### Decision 4 · Auto-tune regularisation alphas

Hardcoding `alpha=1.0` (common in tutorials) is arbitrary. Using `RidgeCV`, `LassoCV`, and `ElasticNetCV` with a 100-point log-spaced grid finds the optimal alpha for this specific dataset via cross-validation — no manual tuning required.

### Decision 5 · Two feature sets compared side-by-side

Running `baseline` and `engineered` feature sets through identical model training and evaluation isolates the contribution of feature engineering. If engineered features do not outperform baseline, the simpler set is preferred (Occam's razor).

### Decision 6 · Refit on full data before deployment

Scikit-learn best practice: after model selection is complete, refit on all available data before serialising for production. With n=50 every observation is valuable.

---

## 9. File Structure

```
d:\wi\
├── 50_Startups.csv                  # Source data (read-only)
├── 50_startups_crisp_dm.py          # Original CRISP-DM script (preserved)
├── 50_startups_expert.py            # Expert version (this design)
├── 50_startups_best_model.pkl       # Serialised best pipeline (joblib)
├── design_v1.md                     # This document
│
├── phase2_understanding.png         # Phase 2 EDA figure
├── phase3_preparation.png           # Phase 3 feature engineering figure
├── phase4_cv_results.png            # Phase 4 CV model comparison
└── phase5_evaluation.png            # Phase 5 evaluation figure
```

---

## 10. Key Findings Summary

### Feature Importance (expert ranking)

| Rank | Feature | Evidence | Action |
|------|---------|----------|--------|
| 1 | **R&D Spend** | r = 0.97, RF importance = 0.93, 5/5 selection votes | Always keep |
| 2 | **Marketing Spend** | r = 0.75, 4/5 selection votes | Keep; note collinearity with R&D (r = 0.72) |
| 3 | **RD_Ratio** | Better business meaning than raw Admin | Use in engineered set |
| 4 | **Log_RD** | Captures diminishing returns on R&D | Use in engineered set |
| 5 | Administration (raw) | r = 0.20, p = 0.61, coef negative after controls | Drop or replace with ratio |
| 6 | **State** | p > 0.95, 0/5 votes | Drop |

### Core Business Insight

> **Every dollar of R&D investment is associated with significantly higher profit.**  
> R&D Spend explains ~95% of the variation in Profit on its own.  
> Geographic state is irrelevant at this sample size.  
> Administrative overhead has no positive independent effect on profit.

### Limitations

| Limitation | Implication |
|---|---|
| n = 50 | Wide confidence intervals on all estimates; CV std will be high |
| No temporal data | Cannot detect seasonality or growth trends |
| Only 3 states | Cannot generalise geographic conclusions |
| Likely simulated data | Real-world R² of 0.97 for a single feature is extremely unusual |
| No revenue / cost breakdown | Cannot separate profit drivers further |

---

*Design document generated as part of the 50 Startups CRISP-DM Expert Analysis project.*  
*Script: `50_startups_expert.py` | Framework: scikit-learn | Python 3.x*
