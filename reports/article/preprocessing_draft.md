# Data Preprocessing Draft (Week 4)

## 1. Missing Value Handling
- Checked all columns for missing values.
- Imputation strategy:
  - Numerical: Imputed with median if missing (none found in sample).
  - Categorical: Imputed with mode if missing (none found in sample).

## 2. Outlier Treatment
| Column         | Detection Method | Strategy      | Details                       |
|---------------|------------------|--------------|-------------------------------|
| All numerics  | IQR rule         | Capped (1-99)| Outliers capped at 1st/99th % |

## 3. Categorical Encoding
| Column           | Unique Values | Encoding Method      | Notes                       |
|------------------|--------------|---------------------|-----------------------------|
| ship_plant_code  | Low          | One-hot             |                             |
| map_page         | Low          | One-hot             |                             |
| hour, dayofweek, month | Low    | One-hot             | Extracted from start_time   |
| truck_code       | High         | Ordinal             | Consider target encoding    |

## 4. Numerical Scaling
- All features (except target and dates) scaled with StandardScaler (mean=0, std=1).
- Fitted on training data only (if split).

## 5. Feature Engineering
| Feature Name      | Source Columns                | Description / Motivation                       |
|------------------|------------------------------|------------------------------------------------|
| hour, dayofweek, month | start_time             | Capture time-of-day, day-of-week, seasonality  |
| cycle_minutes    | typed_time, at_plant_time     | Delivery cycle duration                        |
| vol_plant_1h     | u_Volumen, ship_plant_code    | Rolling mean volume per plant (1h window)      |
| vol_truck_1h     | u_Volumen, truck_code         | Rolling mean volume per truck (1h window)      |
| vol_addr_1h      | u_Volumen, ship_addr_line     | Rolling mean volume per address (1h window)    |
| vol_truck_prev   | u_Volumen, truck_code         | Previous delivery volume for truck             |
| vol_plant_prev   | u_Volumen, ship_plant_code    | Previous delivery volume for plant             |
| truck_avg_vol    | u_Volumen, truck_code         | Truck's average delivered volume               |
| plant_avg_vol    | u_Volumen, ship_plant_code    | Plant's average delivered volume               |

## 6. Train/Validation/Test Split
- Not performed in this notebook (add if required for modeling).

## 7. Target Variable
- `u_Volumen` (volume per delivery/hour) is the prediction target.

## 8. Notes
- All preprocessing steps are fully automated in the notebook.
- Feature selection and windowing are designed for neural network input.
