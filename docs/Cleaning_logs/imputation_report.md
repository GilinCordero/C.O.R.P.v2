# Imputation Report

This document summarizes the two imputation cells used in [2.data_imputation.ipynb](../../notebooks/2.data_imputation.ipynb) and explains how each method builds the imputed datasets.

## First Issue: Donor + RandomForest Imputation for Plant 710

**Goal**: Fill the long missing period for plant `710` using historical behavior from similar plants, while preserving realistic hourly activity gaps.

**Method**:
- Uses a donor-based strategy centered on plant `515` and additional fallback plants selected by profile similarity.
- Trains a `RandomForestRegressor` inside a `MultiOutputRegressor` to estimate hourly remission count and hourly volume behavior from time features.
- Applies an active-hour filter learned from plant `710` history before the gap, so the synthetic output does not force remissions into hours that were historically inactive.
- Samples actual donor rows to keep the synthetic records structurally close to real observations.

**Process**:
1. Load the cleaned remission dataset and convert the date/time columns to datetimes.
2. Build a pre-gap reference window for plant `710` and plant `515`.
3. Compute hourly profiles for donor selection and choose the most similar fallback plants.
4. Estimate the probability that a given weekday/hour combination is active in the historic `710` data.
5. Train an hourly count model using donor history and predict how many rows should be generated for each missing hour.
6. Suppress hours that are historically inactive, then sample donor rows for the remaining hours.
7. Rebuild timestamps, assign `ship_plant_code = 710`, mark rows as imputed, and export the result.

**Output**:
- `../data/processed/remissions_db_imputed_710.xlsx`

**Key result**:
- The generated 710 imputation follows the historic shape of the plant schedule instead of creating remissions in every hour of the day.

## Second Issue: Multi-Plant History Imputation

**Goal**: Extend the imputation to every plant in the dataset using each plant's own pre-gap history as the primary signal.

**Method**:
- Builds a per-plant hourly history model from the available data before the target gap.
- Uses `RandomForestRegressor` with `MultiOutputRegressor` to estimate both remission count and total volume per missing hour.
- Enforces an active-hour mask derived from each plant's own historic weekday/hour schedule.
- Samples real historical rows from the same plant to preserve row-level patterns and time offsets.

**Process**:
1. Load the 710-imputed workbook and normalize the datetime columns.
2. Loop over all plant codes present in the dataset.
3. For each plant, keep only the pre-gap history and aggregate it by hour.
4. Train the multioutput model on hourly features such as hour, day of week, month, day of year, and weekend flag.
5. Predict missing hours within the gap period.
6. Apply the active-hour probability mask so hours with no historic activity remain suppressed.
7. Sample donor rows from the same plant history, rebuild the timestamps, and rescale volume when needed.
8. Concatenate original and synthetic rows, sort by time, and export the final workbook.

**Output**:
- `../data/processed/remissions_db_imputed_all_plants_2025.xlsx`

## Notes

- Both cells rely on the same core idea: keep the synthetic data anchored to observed time patterns rather than filling every hour uniformly.
- The first cell is donor-driven and plant-specific for `710`.
- The second cell is plant-history-driven and generalized to all plants.
- The active-hour masking is the main safeguard against unrealistic continuous hourly remission generation.

## Added Columns Summary

The following columns are introduced by the imputation notebook. Some are **traceability metadata** (useful), while others are **helper/modeling columns** that have been removed during the EDA and are not documented here.

| Column | Added In | Category | Purpose | Keep for Analysis? |
|---|---|---|---|---|
| `is_imputed` | Cell 1 and Cell 2 | Metadata | Flags synthetic rows created by imputation. | Yes (to ignore synthetic data for accurate analysis when needed) |
| `imputed_method` | Cell 1 and Cell 2 | Metadata | Stores the method used (`donor+rf` or `plant_history_multioutput`). | Yes (keep for audit/documentation). |

### EDA Analysis Policy

- Keep: `is_imputed` (and optionally `imputed_method`, `imputed_range`, source columns for audits).
- Drop: helper columns (`hour`, `day_of_week`, `month`, `hour_bucket`, `date`) from the final analysis table if they were exported.

### Final amount of real vs synthetic rows

- Synthetic remissions: `16699 |  4.48% `
- Real remissions: `355802 | 95.52%`

- Range imputated for plant 710 only: `May 2022` - `February 2023`
- Range imputated for all plants alike: `January 2025` - `May 2025`


