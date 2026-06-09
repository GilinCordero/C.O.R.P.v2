"""
Evaluate model degradation over a 45-day forecast horizon.
Uses recursive forecasting (each prediction feeds into the next).
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import holidays

# Paths
ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = ROOT / "app" / "data" / "hourly_features.parquet"
MODEL_PATH = ROOT / "app" / "models" / "hourly_lgbm.pkl"

# Load
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

df = pd.read_parquet(PARQUET_PATH)
df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])

# Feature columns (must match model_runner.py)
FEATURE_COLS = [
    "ship_plant_code", "hour", "day_of_week", "month", "day_of_month", "year",
    "is_weekend", "is_sunday", "is_saturday", "was_open",
    "hour_sin", "hour_cos", "day_week_sin", "day_week_cos",
    "month_sin", "month_cos",
    "days_since_last_open",
    "volume_per_remission_7d_avg", "is_holiday",
    "volume_m3_lag_24h", "volume_m3_lag_48h", "volume_m3_lag_1w",
    "volume_m3_roll_mean_24h", "volume_m3_roll_mean_48h", "volume_m3_roll_mean_1w",
    "volume_m3_roll_std_24h", "volume_m3_roll_std_48h", "volume_m3_roll_std_1w",
]

plants = sorted(df["ship_plant_code"].unique())
HORIZON_DAYS = 45
HOURS = HORIZON_DAYS * 24

# Results container
results_by_horizon = {d: [] for d in range(1, HORIZON_DAYS + 1)}

for plant_id in plants:
    plant_df = df[df["ship_plant_code"] == plant_id].sort_values("hour_bucket").copy()
    
    # Split train/test
    train_df = plant_df[plant_df["hour_bucket"] < "2025-01-01"].copy()
    test_df = plant_df[plant_df["hour_bucket"] >= "2025-01-01"].copy()
    
    if len(test_df) == 0:
        continue
    
    hist_values = train_df["volume_m3"].values.copy()
    n = len(hist_values)
    predicted_values = []
    
    # Forecast start = last training hour
    forecast_start = pd.Timestamp(train_df["hour_bucket"].max())
    
    first_record_time = pd.Timestamp(plant_df["hour_bucket"].min())
    mean_vpr = plant_df["volume_per_remission_7d_avg"].mean() if "volume_per_remission_7d_avg" in plant_df.columns else 2.5
    
    # Track last open day
    open_days = train_df[train_df["was_open"] == 1]["hour_bucket"].dt.floor("D").unique()
    last_open_day = pd.Timestamp(open_days.max()) if len(open_days) > 0 else first_record_time
    
    for offset in range(1, HOURS + 1):
        target_time = forecast_start + pd.Timedelta(hours=offset)
        hr = target_time.hour
        dow = target_time.dayofweek
        month = target_time.month
        dom = target_time.day
        year = target_time.year
        day_floor = target_time.floor("D")
        
        is_open = 1 if (6 <= hr <= 20 and dow != 6) else 0
        if is_open == 1:
            last_open_day = day_floor
        days_since_last_open = (day_floor - last_open_day).days
        
        combined = list(hist_values) + list(predicted_values)
        k = len(predicted_values)
        
        def get_lag(offset_hrs):
            if offset_hrs <= k:
                return float(predicted_values[-offset_hrs])
            else:
                idx = n - (offset_hrs - k)
                return float(hist_values[idx]) if 0 <= idx < n else 0.0
        
        def rolling(w):
            vals = combined[-w:] if len(combined) >= w else combined
            return float(np.mean(vals)) if len(vals) > 0 else 0.0
        
        def rolling_std(w):
            vals = combined[-w:] if len(combined) >= w else combined
            return float(np.std(vals)) if len(vals) > 1 else 0.0
        
        row = {
            "ship_plant_code": plant_id,
            "hour": hr,
            "day_of_week": dow,
            "month": month,
            "day_of_month": dom,
            "year": year,
            "is_weekend": int(dow >= 5),
            "is_sunday": int(dow == 6),
            "is_saturday": int(dow == 5),
            "was_open": is_open,
            "hour_sin": np.sin(2 * np.pi * hr / 24),
            "hour_cos": np.cos(2 * np.pi * hr / 24),
            "day_week_sin": np.sin(2 * np.pi * dow / 7),
            "day_week_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "days_since_last_open": days_since_last_open,
            "volume_per_remission_7d_avg": mean_vpr,
            "is_holiday": int(target_time.date() in holidays.MX(years=[year])),
            "volume_m3_lag_24h": get_lag(24),
            "volume_m3_lag_48h": get_lag(48),
            "volume_m3_lag_1w": get_lag(168),
            "volume_m3_roll_mean_24h": rolling(24),
            "volume_m3_roll_mean_48h": rolling(48),
            "volume_m3_roll_mean_1w": rolling(168),
            "volume_m3_roll_std_24h": rolling_std(24),
            "volume_m3_roll_std_48h": rolling_std(48),
            "volume_m3_roll_std_1w": rolling_std(168),
        }
        
        X = pd.DataFrame([row])[FEATURE_COLS]
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)
        
        # Sunday or night hours
        if dow == 6 or hr < 6 or hr > 20:
            pred = 0.0
        
        predicted_values.append(pred)
        
        # Compare with actual if available
        actual = test_df[test_df["hour_bucket"] == target_time]
        if len(actual) > 0:
            actual_val = actual["volume_m3"].values[0]
            day_num = (target_time.date() - pd.Timestamp("2025-01-01").date()).days + 1
            if 1 <= day_num <= HORIZON_DAYS:
                results_by_horizon[day_num].append({
                    "plant": plant_id,
                    "hour": target_time,
                    "actual": actual_val,
                    "predicted": pred,
                    "error": actual_val - pred,
                    "abs_error": abs(actual_val - pred),
                })

# Aggregate by horizon day
rows = []
for day in range(1, HORIZON_DAYS + 1):
    data = results_by_horizon[day]
    if len(data) == 0:
        continue
    errors = [d["error"] for d in data]
    abs_errors = [d["abs_error"] for d in data]
    actuals = [d["actual"] for d in data]
    preds = [d["predicted"] for d in data]
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean([e**2 for e in errors]))
    smape = np.mean([200 * abs(a - p) / (abs(a) + abs(p)) if (abs(a) + abs(p)) > 0 else 0 for a, p in zip(actuals, preds)])
    
    rows.append({
        "Day": day,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "sMAPE": round(smape, 1),
        "Samples": len(data),
    })

result_df = pd.DataFrame(rows)
print(result_df.to_string(index=False))

# Show degradation vs Day 1
print("\n=== DEGRADATION vs Day 1 ===")
if len(result_df) > 0:
    day1_mae = result_df.iloc[0]["MAE"]
    result_df["MAE_delta"] = result_df["MAE"] - day1_mae
    result_df["MAE_pct_increase"] = (result_df["MAE"] / day1_mae - 1) * 100
    
    for _, row in result_df.iterrows():
        print(f"Day {int(row['Day']):2d}: MAE={row['MAE']:.3f} (+{row['MAE_delta']:+.3f}, +{row['MAE_pct_increase']:.1f}%)")
