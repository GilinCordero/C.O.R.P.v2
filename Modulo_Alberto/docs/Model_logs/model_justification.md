# LightGBM Native Hourly Model

During the last iteration of this project, we focused our predictions in time series data preparation, which involved recording our data sequentially with regular intervals of days, weeks and even months.

The business now requires a more profound approach, with an hourly prediction of demand throughout all of their plants.

### What we did previously

SARIMAX gave us a series of facilities to make a decent prediction of the daily demand as well as the remissions that would most likely appear during each day.
However, it is a statistical and linear approach with strict assumptions: stationarity, limited exogenous variables, and no native support for complex non-linear interactions.
We chose it initially due to its interpretability and seasonality handling, but it proved insufficient for hourly granularity where demand patterns are highly irregular and influenced by dozens of interacting factors.

### Why we now choose LightGBM

The answer is straightforward: *we need accurate hourly predictions across multiple plants, with the ability to incorporate complex temporal and operational features*.

Here is a documented list of the advantages of using LightGBM instead of SARIMAX:

1. **Non-linear by nature**: LightGBM is a gradient-boosted tree model that captures complex, non-linear relationships without requiring data stationarity or manual differencing. SARIMAX is bound to linear assumptions.
2. **Native handling of exogenous features**: It naturally integrates dozens of engineered features — lags, rolling statistics, cyclic time encodings, holidays, and plant-specific calendars — without dimensional constraints or convergence issues.
3. **Speed and scalability**: Training and inference are orders of magnitude faster than deep learning alternatives (e.g., TFT) and significantly more robust than SARIMAX for large, high-frequency datasets.
4. **Multi-horizon forecasting via recursion**: By using lag and rolling features, the model recursively predicts hour-by-hour for horizons of 45–60 days, matching the business requirement without architectural complexity.
5. **Feature interpretability**: Unlike black-box deep learning models, LightGBM provides native feature importance scores (split/gain) and is compatible with SHAP analysis, enabling clear understanding of what drives demand.
6. **Fast retraining in production**: The model trains in minutes on commodity hardware (CPU), making it feasible to retrain frequently as new remission data arrives. This contrasts with deep learning alternatives (e.g., TFT) that require GPU resources and lengthy training cycles, and with SARIMAX which needs manual refitting and stationarity checks.

### Multicollinearity resilience

A natural concern with extensive feature engineering is multicollinearity: engineered features such as `hour` and `hour_sin`/`hour_cos`, or lag and rolling statistics derived from the same underlying series, are inherently correlated. Unlike linear models where multicollinearity destabilizes coefficient estimates and inflates standard errors, tree-based models like LightGBM are robust to it. During training, each tree selects the single feature that provides the best split at a given node; correlated alternatives are simply not chosen for that branch. Predictions remain stable and accurate regardless of redundancy among the input feature set. The only practical consequence is that feature importance becomes distributed across correlated variables, which affects interpretability but not model performance.

