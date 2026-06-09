5 Model Selection
To transition from daily aggregate planning to high-resolution hourly demand forecasting
across multiple heterogeneous facilities, a gradient-boosted decision tree framework utilizing
LightGBM was selected over the previously deployed linear statistical models and alterna-
tive deep learning architectures. While the initial SARIMAX implementation (different
project, same company) effectively tracked macro-level daily seasonality, its rigid assump-
tions of statistical stationarity and limited capacity for non-linear exogenous interactions
rendered it inadequate for mapping the highly irregular, zero-inflated (71.3%) distribu-
tions characteristic of hourly ready-mix concrete logistics. LightGBM natively resolves these
constraints by optimizing non-linear leaf-wise tree growth, seamlessly incorporating high-
dimensional exogenous predictors—such as autoregressive lags, rolling statistical windows,
and cyclic temporal encodings—and executing recursive multi-horizon projections across a
45-to-60-day planning timeline. Furthermore, compared to complex deep learning alterna-
tives like the Temporal Fusion Transformer (TFT) or Long Short-Term Memory (LSTM)
networks, which require extensive GPU infrastructure and specialized loss-function parame-
terization to prevent gradient degradation over severe zero-inflation, the tree-based ensemble
of LightGBM naturally partitions zero-heavy domains while facilitating rapid, low-latency
retraining on standard commodity CPU hardware directly within the production environ-
ment.
5.1 Light Gradient Boosting Machine
5.1.1 Justification and Architectural Advantages of LightGBM
The selection of LightGBM (LightGBM) as the core predictive engine for high-resolution
hourly demand forecasting is justified by several distinct architectural advantages over clas-
sical linear models and deep learning alternatives:
1. Inherent Non-linear Mapping: Unlike SARIMAX, which relies on strict linear as-
sumptions and requires stationary data inputs through complex differencing, LightGBM
utilizes a gradient-boosted tree-based structure. This leaf-wise optimization approach
natively isolates deep non-linear interactions and localized step-functions, which are
characteristic of highly volatile, hourly transactional intervals.
2. High-Dimensional Exogenous Integration: High-frequency demand is driven
by a convergence of interacting temporal and operational factors. The tree-based
architecture seamlessly incorporates an expansive feature matrix—including autore-
gressive lags, moving statistics, cyclic time encodings, and facility-specific operational
calendars—without encountering the dimensional bottlenecks, multicollinearity con-
straints, or convergence failures common in parametric statistical frameworks.
3. Computational Efficiency and Production Scalability: The framework oper-
ates with significantly reduced computational overhead, executing training and infer-
ence cycles orders of magnitude faster than deep learning models like the Temporal
Fusion Transformer (TFT). This efficiency permits rapid, frequent model retraining
on standard commodity CPU hardware as new delivery logs arrive, eliminating de-
pendencies on specialized GPU infrastructure.
18
Modeling a Temporal Fusion Transformer to Predict Concrete Demand
4. Recursive Multi-Horizon Forecasting: By leveraging sequentially updated lag
vectors and rolling statistical windows, the model executes recursive hour-by-hour
projections over an extended 45-to-60-day horizon. This architecture fulfills long-
range planning mandates without requiring the complex multi-quantile structures or
extreme parameterization of deep recurrent networks.
5. Feature Interpretability and Compliance: While deep neural networks act as
uninterpretable black boxes, LightGBM maintains strict operational transparency. It
provides native split- and gain-based feature importance metrics and retains native
compatibility with Shapley Additive Explanations (SHAP), ensuring that the underly-
ing drivers of concrete demand remain fully auditable for institutional stakeholders.
5.2 Modeling Process
The modeling pipeline initiates from the post-feature-engineering dataset (`transformed_dataset_post_FE.xlsx`),
comprising 305,521 hourly aggregated records across 42 columns. From this superset, a curated
feature matrix of 28 predictors was assembled for the final training configuration. Three
features were explicitly rejected during model development: `days_to_quincena`, `is_quincena`,
and `days_since_first_record`, as they introduced calendar artifacts without improving
validation loss. Conversely, `is_saturday` was engineered as a dedicated binary indicator
(`day_of_week == 5`) to disentangle the distinct Saturday operational window (07:00--13:00)
from the broader weekend and weekday structures. A duplicate temporal column (`programmed_departure_hour`)
was dropped in favor of the canonical `hour` variable.

Prior to estimator fitting, all 18 lag and rolling-window features were inspected for missingness
introduced by boundary effects at the chronological onset of each facility's record sequence.
These NaN values were imputed with `0.0`, reflecting the physical interpretation that no prior
volume exists before the first recorded observation. Post-imputation, zero missing values
remained across the entire matrix. The fully processed dataset was materialized as
`hourly_features_test.parquet` for both archival storage and production ingestion by the
forecasting application.

A strict temporal split---rather than a randomized partition---was enforced to preserve
chronological causality and emulate real-world inference conditions. The cutoff date was
fixed at `TEST_CUTOFF_DATE = "2025-01-01"`, yielding a training set of 236,623 rows
(77.4%, spanning 2020-02-04 to 2024-12-31) and a held-out test set of 68,898 rows
(22.6%, spanning 2025-01-01 to 2026-04-24). This 22.6% temporal holdout corresponds
to approximately 16 months of recent operational history, sufficient to assess model
robustness across multiple seasonal cycles and holiday calendars.
5.3 Training Configuration
Model estimation was performed with the LightGBM gradient-boosting library under a
regression objective minimizing Mean Absolute Error (MAE). The hyperparameter
configuration was deliberately conservative to prevent overfitting on the highly zero-inflated
hourly distribution, while retaining sufficient model capacity to capture multi-scale
temporal dependencies. The specification was as follows:

- `objective`: regression
- `metric`: mae
- `boosting_type`: gbdt
- `num_leaves`: 31
- `learning_rate`: 0.05
- `feature_fraction`: 0.9
- `bagging_fraction`: 0.8
- `bagging_freq`: 5
- `random_state`: 42

Training was executed for up to `num_boost_round = 500` iterations, with early stopping
patience set to `early_stopping_rounds = 50`. Both the training set and the temporal validation
set were supplied as LightGBM Dataset objects, enabling the framework to monitor validation
MAE at each boosting round. The algorithm terminated at iteration 230---the best iteration---
after 50 consecutive rounds failed to improve the validation MAE beyond 2.4267 m³/hr.

Post-prediction, all raw model outputs were clipped to non-negative values
(`np.clip(y_pred, 0, None)`), enforcing the physical constraint that concrete discharge
volume cannot be negative. This clipping step is particularly critical during recursive
multi-horizon forecasting, where lag feature contamination from negative predictions could
propagate systematic bias through the projection window.
5.4 Evaluation and Metrics
Global test-set performance was quantified across three standard regression metrics:
MAE, RMSE, and symmetric MAPE (sMAPE). The new model achieved a global MAE of
2.4190 m³/hr, an RMSE of 6.2823 m³/hr, and an sMAPE of 152.14%. While the MAE and
RMSE figures remain operationally interpretable, the elevated sMAPE is an artifact of the
severe zero-inflation inherent to hourly concrete demand: 70.2% of test-set observations
record zero volume (facility closed or no dispatches), and each zero-hour mispredicted as
positive contributes the maximum 200% per-observation sMAPE bound. When evaluated
exclusively on positive-volume hours, the sMAPE collapses to 56.99%, confirming that
the metric distortion is driven by boundary-condition misclassification rather than by
systematic magnitude error during active operational windows.

Per-facility disaggregation reveals consistent error profiles across the six plants, with
MAE ranging from 2.1023 m³/hr (Plant 514, the lowest) to 2.6875 m³/hr (Plant 510, the
highest). Plant 710, despite exhibiting the lowest original-baseline MAE, showed the largest
degradation in the new model (+1.05 m³/hr), suggesting that its previously favorable
signal-to-noise ratio was sensitive to the feature-engineering modifications. The full
per-plant breakdown is presented in Table 1.

| Plant | MAE (m³/hr) | RMSE (m³/hr) | Test Rows |
|-------|-------------|--------------|-----------|
| 510   | 2.6875      | 6.1508       | 11,482    |
| 511   | 2.6345      | 7.3836       | 11,483    |
| 512   | 2.3095      | 5.1214       | 11,484    |
| 514   | 2.1023      | 5.0343       | 11,483    |
| 515   | 2.4979      | 6.7981       | 11,483    |
| 710   | 2.2824      | 6.8315       | 11,483    |

Feature importance analysis (gain-based) identifies `volume_m3_lag_1w` as the dominant
predictor, contributing a gain score of 38,525,670---nearly three times the second-ranked
`days_since_last_open` (13,011,610). This dominance confirms that week-ahead autoregressive
volume is the strongest informational anchor for hourly demand. The remaining top predictors
are tightly clustered around short-term lags (`volume_m3_lag_24h`, `volume_m3_lag_48h`),
temporal position (`hour`, `hour_cos`, `hour_sin`), and day-of-week structure. The least
informative features are binary calendar indicators (`is_holiday`, `is_saturday`, `is_weekend`,
`is_sunday`) and low-variance cyclic encodings (`month_sin`, `month_cos`), suggesting that
calendar state alone contributes little discriminative power beyond what is already captured
by the autoregressive and operational-window features.

Residual analysis by hour of day exposes systematic bias during peak operational windows.
Hour 7 (the first active hour at most facilities) registers the largest positive mean error
(+0.561 m³/hr) and by far the highest standard deviation (13.225 m³/hr), indicating that
the model systematically under-predicts the morning dispatch surge. Conversely, night hours
(00:00--05:00 and 20:00--23:00) exhibit near-zero mean bias and low variance, reflecting
the model's confidence in correctly predicting dormant periods. These patterns suggest that
the current feature set lacks sufficient granularity to capture intra-morning operational
ramp-up dynamics, a limitation that could be addressed in future iterations through
fine-grained shift-schedule metadata or real-time queue-depth signals.