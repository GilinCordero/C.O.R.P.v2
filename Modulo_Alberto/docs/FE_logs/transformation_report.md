# Dataset Transformation to Hourly Long Format

The raw remissions dataset, comprising 356,332 individual transaction records, was transformed into a dense hourly time series to enable temporal feature engineering and predictive modeling. Each original row represented a single concrete delivery with an associated timestamp (`start_time`), volume (`u_Volumen`), and originating plant (`ship_plant_code`). This transactional structure was unsuitable for time-series forecasting due to irregular temporal spacing and the absence of explicit zero-demand observations.

The transformation pipeline consisted of five sequential steps. First, fields were standardized for clarity: `start_time` was renamed to `programmed_departure_time`, `ship_plant_code` to `plant_code`, and `order_date` to `remission_date`. Temporal features were then extracted at the remission level, including `month_of_year`, `day_of_week`, and `day_of_month`, derived directly from `programmed_departure_time`.

Next, timestamps were truncated to the hour using `.dt.floor('h')`, mapping all deliveries within the same clock hour (e.g., 07:00:00–07:59:59) to a single bucket (07:00:00). This established the target prediction granularity. Remissions were subsequently grouped by `(programmed_departure_time, plant_code)` and aggregated: `volume_m3` was summed, `remission_count` was counted, and `is_imputed` was assigned the maximum value within each bucket to flag any synthetic records originating from the data imputation phase.

A critical limitation of the aggregated result was its sparsity—hours without deliveries were entirely absent. To enable lag-based and rolling-window features, the time series was completed by generating a continuous hourly sequence for each plant from its first to last observed timestamp. Missing hours were inserted with `volume_m3 = 0`, `remission_count = 0`, and `is_imputed = 0`, yielding a dense regular sequence.

Finally, temporal fields were recalculated for all rows (original and inserted): `programmed_departure_hour` (0–23), `year`, `month_of_year`, `day_of_week`, and `day_of_month`. The dataset was sorted by `(plant_code, programmed_departure_time)` and exported to `transformed_dataset_pre_FE.xlsx`.

The resulting dataset contains 305,521 rows and 10 columns, with zero gaps across all six plants. Temporal coverage spans from February 2020 to April 2026, with plant 514 exhibiting a delayed start in July 2022 consistent with its later operational commencement (Table 1).

| Plant | Hour Range | Hours Present | Missing |
|-------|-----------|---------------|---------|
| 510 | 2020-02-04 → 2026-04-24 | 54,507 | 0 |
| 511 | 2020-02-04 → 2026-04-24 | 54,508 | 0 |
| 512 | 2020-02-04 → 2026-04-24 | 54,511 | 0 |
| 514 | 2022-07-20 → 2026-04-24 | 32,980 | 0 |
| 515 | 2020-02-04 → 2026-04-24 | 54,507 | 0 |
| 710 | 2020-02-04 → 2026-04-24 | 54,508 | 0 |

Approximately 71.3% of hourly observations (217,711 rows) record zero concrete volume. This reflects operational reality: plants operate roughly 10–11 hours per day, six days per week, with Sundays and nighttime hours exhibiting no scheduled activity. The distribution of zero-demand hours by clock hour (Table 2) confirms this pattern, with elevated frequencies during 00:00–06:00 and 18:00–23:00, and reduced frequencies during core operating hours.

| Hour | Zero-Demand Hours | Hour | Zero-Demand Hours |
|------|-------------------|------|-------------------|
| 00 | 12,726 | 12 | 3,794 |
| 01 | 12,698 | 13 | 4,464 |
| 02 | 12,723 | 14 | 5,223 |
| 03 | 12,722 | 15 | 5,366 |
| 04 | 12,700 | 16 | 6,351 |
| 05 | 12,567 | 17 | 9,432 |
| 06 | 12,067 | 18 | 12,314 |
| 07 | 4,648 | 19 | 12,659 |
| 08 | 4,074 | 20 | 12,703 |
| 09 | 3,634 | 21 | 12,725 |
| 10 | 3,523 | 22 | 12,304 |
| 11 | 3,569 | 23 | 12,726 |

The high sparsity of the target variable is a defining characteristic of this dataset and influenced the selection of LightGBM over deep-learning alternatives such as TFT, as tree-based models naturally handle zero-inflated distributions without requiring specialized parameterization. Notably, the imputation pipeline was corrected to enforce zero activity on Sundays (`day_of_week = 6`), as an earlier version probabilistically generated synthetic Sunday entries based on sparse historical emergency orders. Post-correction, no synthetic remissions are generated for Sunday hour-buckets.

Data quality was verified across all columns: zero NaN values, zero negative volumes, zero duplicate `(plant, hour)` pairs, and confirmed chronological ordering per plant. The transformed dataset is now ready for feature engineering, including cyclic temporal encodings, autoregressive lags, rolling statistics, and calendar-based features.
