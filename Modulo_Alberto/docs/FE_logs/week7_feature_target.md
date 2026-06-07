# Feature & Target Variable Document
---
### Target variable definition
- Name: `volume_m3` (by hour)
- Type: `real (double)` 
- Distribution: As for our findings documented in the `EDA.ipynb`, we can confirm a slightly right-skewed distribution for most weekdays with the x-axis representing the hours. 
- Encoding: For the training dataset, our `curr_hour_m3` has no encoding, but it is normalized.
- Justification: 

### Feature set definition (first_iteration)

| Feature Name | Original Column | Type | Encoding | Relevance (no corr analysis) | Description |
|--------------|-----------------|------|----------|-----------|----------|
| `time_idx`   | N/A            | `integer`, auto-incremental | No encoding   | Crucial  | Sequential index representing the time order of the data, it will restart for each new plant_id |
| `month_sin`, `month_cos` | `start_time` | `real (double)` | Cyclical Encoding | Critical | The month in which the orders were made for the corresponding hour. |
| `day_month_sin`, `day_month_cos`| `start_time` | `real (double)` | Cyclical Encoding | Critical | The day of the month in which the orders were made for the corresponding hour. |
| `day_week_sin`,`day_week_cos`| `start_time` | `integer` | Cyclical Encoding | Critical | The day of the week in which the orders were made for the corresponding hour. | 
| `hour_sin`, `hour_cos`| `start_time` | `real (double)` | Cyclical Encoding | Critical | The hour of the day in which a remission was programmed. |
| `24_hour_m3_lag`, `48_hour_m3_lag`, `1_week_m3_lag`  | Normalization | Essential for Time series | Sliding windows of past values for concrete in `m3` counts for each hour. |
| `24_hour_remission_lag`, `48_hour_remission_lag`, `1_week_remission_lag` | Essential for Time series | Very High | Sliding windows of past values for remission counts for each hour. |
| `24_hour_mean`, `24_hour_std_dev` (and for 48 hours and 1 week) | Essential for Time series | real (double) | Pending, probably normalization/scaling | Unknown | Just the remission and volume statistic measurements to make the model stronger | 
| `is_quincena` | `start_time` | binary | One-hot encoding | Medium | Client's purchase behaviour when quincena arrives. | 
| `days_to_quincena` | `start_time` | binary | N/A | Medium | Client's purchase behaviour when quincena arrives. | 


### Feature set definition (second_iteration)

| Feature Name | Original Column | Type | Encoding | Relevance (corr analysis) | Description |
|--------------|-----------------|------|----------|-----------|----------|
| `time_idx`   | N/A  | `integer`, auto-incremental | No encoding   | Crucial  | Sequential index representing the time order of the data, it will restart for each new plant_id |
| `month_sin`, `month_cos` | `start_time` | `real (double)` | Cyclical Encoding | Medium | The month in which the orders were made for the corresponding hour. |
| `day_month_sin`, `day_month_cos`| `start_time` | `real (double)` | Cyclical Encoding | Critical | The day of the month in which the orders were made for the corresponding hour. |
| `day_week_sin`,`day_week_cos`| `start_time` | `integer` | Cyclical Encoding | Medium | The day of the week in which the orders were made for the corresponding hour. | 
| `hour_sin`, `hour_cos`| `start_time` | `real (double)` | Cyclical Encoding | High | The hour of the day. |
| `24_hour_m3_lag`, `48_hour_m3_lag`, `1_week_m3_lag`  | Normalization | Very High | Sliding windows of past values for concrete in `m3` counts for each hour. |
| `24_hour_remission_lag`, `48_hour_remission_lag`, `1_week_remission_lag` | N/A | Very High | Sliding windows of past values for remission counts for each hour. |
| `24_hour_volume_mean`, `24_hour_volume_std_dev`, `24_hour_remission_mean`, `24_hour_remission_std_dev` (and for 48 hours and 1 week) | N/A | real (double) | Pending, probably normalization/scaling | Unknown | Just the remission and volume statistic measurements to make the model stronger | 
| `is_saturday`, `is_sunday` | `start_time` | binary | One-hot encoding | High | The model knows this days, they are neccesary to note fue to half-time work at saturdays and no work at sundays. |
| `is_holiday` | `start_time` | binary | One-hot encoding | Unknown | Some holidays are considered and maybe not by us | 
| `was_open` | `start_time` | binary | N/A | Unknown | Standarized open plant schedule is from 8am - 6pm, anything outside those ours could potentially be outlier. However we will use 7am - 5pm | 

### Feature selection rationale

The main focus behind this early feature selection came from the strength fo the selected model with time series. As it can be seen, they were chose based on their rolling/sliding window patterns with various statistic values and historic data.

Many "features" that came in the original dataset were dropped because they lacked temporal relevance or introduced sever multu-collinearity. By choosing only this mentioned variables, we ensure a first iteration with reliable data that can result in a very efficient starting-point model.




