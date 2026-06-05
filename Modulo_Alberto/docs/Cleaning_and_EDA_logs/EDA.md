# Exploratory Data Analysis (EDA) — Outlier Findings

## Executive Summary

This document summarizes the outlier analysis performed during the cleaning phase (`1.cleaning_phase.ipynb`). **No outliers were removed** from the dataset. All extreme values were investigated and confirmed as legitimate business events (emergency orders, large projects, operational spikes). The rationale is documented below per plant.

---

## Methodology

For each plant, an hourly aggregation of remission counts was created (`start_time` floored to the hour). **Boxplots** and **IQR upper-fence analysis** (`Q3 + 1.5 × IQR`) were used to flag potential outliers. Each flagged hour was manually inspected to determine if it corresponded to data errors or real operational events.

---

## Findings by Plant

### Plant 710

| Detail | Value |
|--------|-------|
| **Flagged outlier** | 16 remissions on **June 14th, 2021 at 10:00 AM** |
| **Detection method** | Boxplot + IQR upper fence |
| **Investigation** | All 16 remissions inspected individually. No duplicate `tkt_code`, no repeated truck codes, no systematic patterns suggesting data entry error. |
| **Conclusion** | ✅ **Legitimate spike** — likely a large construction project or emergency pour. |
| **Action** | **Kept as-is.** |

> *"We'll keep plant 710's outliers due to legitimate data and no clues of repetition or errors."*

---

### Plant 510

| Detail | Value |
|--------|-------|
| **Flagged outlier** | 21 remissions on **July 13th, 2020 at 12:00 PM** |
| **Additional note** | October 2021 also showed elevated activity |
| **Detection method** | Boxplot + IQR upper fence |
| **Investigation** | Manual inspection of all 21 remissions within the hour. No data anomalies detected. |
| **Conclusion** | ✅ **Legitimate operational spike.** |
| **Action** | **Kept as-is.** |

---

### Plant 511

| Detail | Value |
|--------|-------|
| **Flagged outlier** | 22 remissions on **December 21st, 2024 at 6:00 AM** |
| **Additional note** | April 2026 also flagged for review |
| **Detection method** | Boxplot + IQR upper fence |
| **Investigation** | Early-morning spike (6 AM) is outside typical business hours but was confirmed as a real emergency batch. |
| **Conclusion** | ✅ **Legitimate emergency order.** |
| **Action** | **Kept as-is.** |

---

### Plant 515

| Detail | Value |
|--------|-------|
| **Flagged outlier** | 21 remissions on **June 14th, 2025 at 1:00 PM** |
| **Additional note** | May 2024 also showed elevated activity |
| **Detection method** | Boxplot + IQR upper fence |
| **Investigation** | All 21 remissions within the hour inspected. No duplication, no data corruption. |
| **Conclusion** | ✅ **Legitimate large-project spike.** |
| **Action** | **Kept as-is.** |

---

### Plant 512

| Detail | Value |
|--------|-------|
| **Flagged outlier** | September 5th, 2024 at 10:00 AM — unusually high order and volume |
| **Additional context** | Heavy downturn in remissions and volume between **April 2020 and June 15, 2020** (pandemic period) |
| **Detection method** | Boxplot + visual trend analysis |
| **Conclusion (outlier)** | ✅ **Legitimate spike** — confirmed as real demand. |
| **Conclusion (downturn)** | ✅ **Real external event** (COVID-19 lockdown). |
| **Action** | **Outlier kept as-is.** Pandemic period flagged for potential `plant-specific regime feature` during modeling. |

---

### Plant 514

| Detail | Value |
|--------|-------|
| **Observation** | Begins registering data since **July 20, 2023** (late-starting plant) |
| **Outliers flagged** | None significant |
| **Conclusion** | No outlier concerns. Missing early history is handled as a "late data" scenario, not an anomaly. |
| **Action** | No action required. |

---

## Volume Distribution (`u_Volumen`)

A frequency histogram of `u_Volumen` was generated to detect impossible volumes (e.g., negative values, volumes exceeding truck capacity).

| Check | Result |
|-------|--------|
| Negative volumes | ❌ None found |
| Impossibly high volumes | ❌ None found |
| Distribution shape | Right-skewed with discrete peaks (consistent with standard mixer truck capacities) |

**Conclusion**: Volume values are physically plausible across the entire dataset. No volume-based outliers were removed.

---

## Business Hours vs. Outside-Hours Distribution

| Category | Observation |
|----------|-------------|
| **Business hours** | 7:00 AM – 6:00 PM shows expected bell-shaped concentration |
| **Outside hours** | ~10.69% of remissions occur before 7 AM or after 6 PM |
| **Interpretation** | Confirmed by the business partner as **legitimate emergency orders**. These are not data errors. |

**Conclusion**: Outside-hours remissions are a real operational pattern. No filtering applied.

---

## General Missing Data Observations

| Gap Period | Affected Plants | Likely Cause | Action |
|------------|-----------------|--------------|--------|
| Jan 1 – Feb 2, 2022 | All | Winter vacations | Documented; imputation TBD |
| Dec 31, 2023 – Feb 1, 2024 | All | Winter vacations | Documented; imputation TBD |
| Mar 31 – May 2, 2025 | All | Operational gap | To be imputed in `2.data_imputation.ipynb` |

No missing data was confused with outliers. Gaps are clearly separated from extreme-value events.

---

## Global Outlier Policy

> **"Why I will not cut this outliers? Because this phenomenons are real, I will not introduce false data, so I would rather use something like TransformedTargetRegressor to reduce the weight of spikes like this during the training process."**

### Decision Rationale

1. **All extreme values were manually verified** — no data entry errors, duplicates, or impossible values were found.
2. **Spikes correspond to real demand** — large projects, emergency pours, and operational surges are normal in construction logistics.
3. **Removing outliers would bias the model** — the model needs to learn that demand *can* spike, even if rarely.
4. **Mitigation strategy** — Instead of deletion, future modeling will use **log-transform** or `TransformedTargetRegressor` to reduce the influence of extreme spikes without discarding them.

---

## Files Referenced

- [Cleaning Notebook](../../notebooks/1.cleaning_phase.ipynb) — Source of all outlier analyses
- [Data Imputation Notebook](../../notebooks/2.data_imputation.ipynb) — Handling of missing gaps
- [Cleaning Report](./cleaning_report.md) — General data quality report
