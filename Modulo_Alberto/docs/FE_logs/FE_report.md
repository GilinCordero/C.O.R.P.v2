# Feature Engineering Report: Hourly Demand Forecasting

**Dataset:** `transformed_dataset_post_FE.xlsx`  
**Shape:** 305,521 rows × 42 columns  
**Date Range:** 2020-02-04 to 2026-04-30  
**Plants:** 510, 511, 512, 514, 515, 710  

---

## 1. Introduction

This document describes the feature engineering pipeline applied to the hourly demand forecasting dataset. The raw data consists of remission records aggregated to hourly buckets per plant, with zero-fill for hours without activity. The goal is to construct a feature set that captures temporal patterns, operational dynamics, and historical momentum to support a gradient-boosted tree model (LightGBM).

All correlation analyses reported herein use Pearson's *r* unless otherwise specified. Two analysis contexts are provided for each feature group: (a) correlation across **all hours** (including 71% zero-volume hours), and (b) correlation conditioned on **volume > 0** to reveal signals masked by the sparse zero-inflated distribution.

---

## 2. Base Dataset Characteristics

The pre-feature-engineering dataset contains the following raw columns:

| Column | Type | Description |
|--------|------|-------------|
| `programmed_departure_time` | datetime | Hourly timestamp |
| `plant_code` | int | Plant identifier (6 plants) |
| `volume_m3` | float | Hourly concrete volume (m³) — **target variable** |
| `remission_count` | int | Number of remissions in the hour |
| `is_imputed` | int | Flag for imputed zero hours |
| `month_of_year`, `day_of_week`, `day_of_month` | int | Calendar components |
| `programmed_departure_hour`, `year` | int | Hour (0–23) and year |

Zero-fill verification confirmed **0 gaps** and **0 NaNs** across the entire time series after imputation.

---

## 3. Feature Engineering Methodology

### 3.1 Cyclic Temporal Encoding

Raw ordinal temporal features (hour, day-of-week, month) impose an artificial discontinuity at their boundaries (e.g., hour 23 is far from hour 0 in linear space). To resolve this, sine-cosine pairs were computed:

$$
\text{sin} = \sin\left(2\pi \cdot \frac{x}{\text{period}}\right), \quad
\text{cos} = \cos\left(2\pi \cdot \frac{x}{\text{period}}\right)
$$

**Correlation results:**

| Feature | All Hours | Volume > 0 |
|---------|-----------|------------|
| `hour_sin` | 0.121 | 0.164 |
| `hour_cos` | **−0.418** | 0.017 |
| `day_week_sin` | 0.098 | −0.015 |
| `day_week_cos` | −0.113 | −0.080 |
| `month_sin` | −0.011 | −0.008 |
| `month_cos` | −0.021 | 0.004 |

**Decision:** **Kept.** The `hour_cos` signal (−0.418) is the strongest univariate predictor, reflecting the daily demand trough during overnight hours. The cyclic encoding preserves circular continuity for the tree model.

---

### 3.2 Weekend and Sunday Indicators

Two binary features were engineered:
- `is_weekend`: 1 if Saturday or Sunday, 0 otherwise
- `is_sunday`: 1 if Sunday, 0 otherwise

**Correlation with `volume_m3`:**

| Feature | All Hours | Business Hours Only |
|---------|-----------|---------------------|
| `is_weekend` | −0.154 | −0.277 |
| `is_sunday` | −0.169 | **−0.303** |

**Business-hour volume by day-of-week:**

| Day | Avg Volume (m³/hr) |
|-----|-------------------|
| Monday | 7.94 |
| Tuesday | 9.42 |
| Wednesday | 9.18 |
| Thursday | 11.48 |
| Friday | **11.99** |
| Saturday | 6.67 |
| Sunday | **0.09** |

**Decision:** **Kept.** Sunday exhibits a 133× reduction in business-hour volume versus Friday (0.09 vs. 11.99 m³/hr). The Sunday indicator is operationally critical for the zero-prediction rule.

---

### 3.3 Quincena (Payday Cycle)

Two features tested for biweekly salary-cycle effects:
- `is_quincena`: 1 if the date is the 15th or last day of the month
- `days_to_quincena`: Days until next payday

**Correlation with `volume_m3`:**
- `days_to_quincena`: −0.009
- `is_quincena`: −0.006

**Decision:** **Rejected.** Flat correlation and no discernible pattern. Payday effects, if present, are not detectable at hourly granularity or are absorbed by other features.

---

### 3.4 Holiday Indicator

A binary `is_holiday` flag was derived from the Mexican holiday calendar.

**Impact on volume:**

| Category | Mean Volume (m³/hr) | Count |
|----------|---------------------|-------|
| Normal day | 9.62 | 117,525 |
| Holiday | **1.39** | 2,541 |

Volume drops **85.5%** on holidays. Correlation: −0.050.

**Decision:** **Kept.** Despite modest linear correlation, the effect size is massive. Tree models exploit this via binary splits.

---

### 3.5 Lag Features

Past values of `volume_m3` and `remission_count` were shifted by 24h, 48h, and 1 week (168h) to capture recent history:

| Feature | Correlation |
|---------|-------------|
| `remission_count_lag_1w` | **0.490** |
| `volume_m3_lag_1w` | 0.431 |
| `remission_count_lag_24h` | 0.415 |
| `volume_m3_lag_24h` | 0.383 |
| `remission_count_lag_48h` | 0.324 |
| `volume_m3_lag_48h` | 0.297 |

**Decision:** **Kept all six.** The 1-week lag is the strongest predictor after `hour_cos`. Remission count lags consistently outperform volume lags, suggesting that *order frequency* is more predictive than *order magnitude* at the same temporal offset.

---

### 3.6 Rolling Window Statistics

Rolling mean and standard deviation (24h, 48h, 1-week windows) were computed for both `volume_m3` and `remission_count`, with a 1-hour shift to prevent leakage:

| Feature | Correlation |
|---------|-------------|
| `remission_count_roll_mean_24h` | **0.195** |
| `volume_m3_roll_mean_24h` | 0.190 |
| `remission_count_roll_std_24h` | 0.174 |
| `volume_m3_roll_std_24h` | 0.142 |
| `remission_count_roll_mean_48h` | 0.142 |
| `volume_m3_roll_mean_48h` | 0.140 |
| `volume_m3_roll_mean_1w` | 0.126 |
| `remission_count_roll_mean_1w` | 0.122 |
| `remission_count_roll_std_48h` | 0.120 |
| `remission_count_roll_std_1w` | 0.106 |
| `volume_m3_roll_std_48h` | 0.096 |
| `volume_m3_roll_std_1w` | 0.084 |

**Decision:** **Kept all twelve.** Correlations are moderate (0.08–0.20) but these capture *trend* and *volatility* — information orthogonal to lag features. The 24-hour window consistently dominates wider windows, indicating that recent momentum is more informative than distant history.

---

### 3.7 Operational Features

#### 3.7.1 Average Order Size Trend (`volume_per_remission_7d_avg`)

Rolling 7-day average of `volume_m3 / remission_count`, representing whether orders are getting larger or smaller.

- **Correlation:** 0.016
- **Decision:** **Kept.** Linear correlation is negligible, but the feature may interact with `remission_count` in tree splits. It was retained for reconstruction fidelity and will be re-evaluated during model retraining.

#### 3.7.2 Plant Maturity (`days_since_first_record`)

Days elapsed since the first record for each plant.

- **Correlation:** 0.008
- **Decision:** **Dropped.** Effectively random. All plants commenced operations in early 2020, rendering this a near-linear time trend with no discriminative power.

---

### 3.8 Plant Activity Features

#### 3.8.1 Recent Activity Flag (`was_open`)

Binary indicator: 1 if the plant had any volume in the past 24 hours, 0 otherwise.

| Variant | Correlation | % Zeros |
|---------|-------------|---------|
| 24-hour window | **0.170** | 18.6% |
| 7-day window | 0.064 | 2.3% |

**Decision:** **Kept 24-hour variant** (renamed to `was_open`). The 7-day variant is too sparse. The 18.6% zeros capture nights, Sundays, and holidays.

#### 3.8.2 Days Since Last Open (`days_since_last_open`)

Continuous feature: days elapsed since the last hour with `volume_m3 > 0`.

- **Correlation:** −0.121

**Decision:** **Kept.** Negative correlation confirms intuitive behavior: longer gaps since last activity predict lower upcoming volume.

---

## 4. Final Feature Inventory

The engineered dataset contains **37 features** for model training:

| # | Feature | Category |
|---|---------|----------|
| 1 | `plant_code` | Identifier |
| 2 | `hour` | Temporal |
| 3 | `day_of_week` | Temporal |
| 4 | `month_of_year` | Temporal |
| 5 | `day_of_month` | Temporal |
| 6 | `year` | Temporal |
| 7 | `hour_sin` | Cyclic encoding |
| 8 | `hour_cos` | Cyclic encoding |
| 9 | `day_week_sin` | Cyclic encoding |
| 10 | `day_week_cos` | Cyclic encoding |
| 11 | `month_sin` | Cyclic encoding |
| 12 | `month_cos` | Cyclic encoding |
| 13 | `is_weekend` | Calendar |
| 14 | `is_sunday` | Calendar |
| 15 | `is_holiday` | Calendar |
| 16 | `was_open` | Activity |
| 17 | `days_since_last_open` | Activity |
| 18 | `volume_per_remission_7d_avg` | Operational |
| 19 | `volume_m3_lag_24h` | Lag |
| 20 | `volume_m3_lag_48h` | Lag |
| 21 | `volume_m3_lag_1w` | Lag |
| 22 | `remission_count_lag_24h` | Lag |
| 23 | `remission_count_lag_48h` | Lag |
| 24 | `remission_count_lag_1w` | Lag |
| 25 | `volume_m3_roll_mean_24h` | Rolling |
| 26 | `volume_m3_roll_mean_48h` | Rolling |
| 27 | `volume_m3_roll_mean_1w` | Rolling |
| 28 | `remission_count_roll_mean_24h` | Rolling |
| 29 | `remission_count_roll_mean_48h` | Rolling |
| 30 | `remission_count_roll_mean_1w` | Rolling |
| 31 | `volume_m3_roll_std_24h` | Rolling |
| 32 | `volume_m3_roll_std_48h` | Rolling |
| 33 | `volume_m3_roll_std_1w` | Rolling |
| 34 | `remission_count_roll_std_24h` | Rolling |
| 35 | `remission_count_roll_std_48h` | Rolling |
| 36 | `remission_count_roll_std_1w` | Rolling |
| 37 | `time_idx` | Sequential index |

**Columns excluded from training:**
- `programmed_departure_time` — used for temporal splitting only
- `volume_m3` — target variable
- `remission_count` — current-hour count (would constitute data leakage)
- `is_imputed` — traceability flag, excluded to prevent leakage

**Rejected features:**
- `days_to_quincena`, `is_quincena` — flat correlation, no predictive signal
- `days_since_first_record` — correlation 0.008, redundant with `year`
- `was_open_7d` — weaker than 24-hour variant, sparse

---

## 5. Data Quality

| Check | Result |
|-------|--------|
| Total rows | 305,521 |
| Plants | 6 |
| Zero-fill gaps | 0 |
| NaN values (post-fill) | 0 |
| Lag/roll NaNs (pre-fill) | 5,760 (expected startup effect) |
| Date range | 2020-02-04 → 2026-04-30 |

---

## 6. Multicollinearity Note

Multiple feature groups exhibit inherent correlation (e.g., `hour` vs. `hour_sin`/`hour_cos`, lag vs. rolling statistics derived from the same series). This is expected and does not degrade predictive performance for tree-based models. LightGBM selects one feature per split; correlated alternatives are simply not chosen for that branch. Predictions remain stable regardless of redundancy in the input feature set. Feature importance becomes distributed across correlated variables, which affects interpretability but not model accuracy.

---

## 7. Conclusion

The feature engineering pipeline produced a 37-feature dataset capturing cyclic temporal patterns, calendar effects, historical lags, rolling momentum, and plant activity state. The strongest individual predictors are `hour_cos` (diurnal cycle), `remission_count_lag_1w` (weekly order frequency), and `is_sunday` (weekly closure pattern). The expanded remission-based lag and rolling features (not present in the original model) exhibited stronger correlations than their volume-based counterparts and are expected to improve forecasting accuracy.
