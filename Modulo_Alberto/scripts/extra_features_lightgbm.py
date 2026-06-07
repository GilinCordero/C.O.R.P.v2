# Modulo_Alberto/scripts/extra_features.py

import numpy as np
import pandas as pd
from pathlib import Path

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "data" / "processed" / "transformed_dataset_post_FE.xlsx"

# Load dataset
df = pd.read_excel(DATA_PATH)
print(f"Loaded: {df.shape} from {DATA_PATH}")

# Ensure datetime
df["programmed_departure_time"] = pd.to_datetime(df["programmed_departure_time"])
df = df.sort_values(["plant_code", "programmed_departure_time"]).reset_index(drop=True)

# =============================================================================
# FIX: days_since_last_open (remove leakage)
# Original used current hour's volume. Fixed version uses ONLY past hours.
# =============================================================================
def days_since_last_open_fixed(group):
    # shift(1) = look at PREVIOUS hours only, not current
    is_open = group["volume_m3"].shift(1) > 0
    last_open_time = group["programmed_departure_time"].where(is_open).ffill()
    days_since = (group["programmed_departure_time"] - last_open_time).dt.total_seconds() / 86400
    # For first row of each plant, there's no past — let LightGBM handle NaN
    return days_since

df["days_since_last_open"] = df.groupby("plant_code", group_keys=False).apply(days_since_last_open_fixed)
print("Fixed: days_since_last_open (no leakage)")

# =============================================================================
# NEW FEATURE 1: Consecutive zero hours
# How many consecutive zero-volume hours just happened before now?
# =============================================================================
def consecutive_zeros(group):
    is_zero = (group["volume_m3"].shift(1) == 0).astype(int)
    # Count consecutive zeros by resetting counter on non-zero
    consec = is_zero * (is_zero.groupby((is_zero != is_zero.shift()).cumsum()).cumcount() + 1)
    return consec

df["consecutive_zero_hours"] = df.groupby("plant_code", group_keys=False).apply(consecutive_zeros)
print("Added: consecutive_zero_hours")

# =============================================================================
# NEW FEATURE 2: Same-hour historical average (past 4 weeks)
# Average volume at this exact hour (e.g., Monday 7AM) over past 4 weeks
# =============================================================================
df["volume_same_hour_4w_avg"] = df.groupby("plant_code").apply(
    lambda g: g.groupby(g["programmed_departure_time"].dt.dayofweek * 24 + g["programmed_departure_time"].dt.hour)["volume_m3"]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
).reset_index(level=0, drop=True)
print("Added: volume_same_hour_4w_avg")

# =============================================================================
# NEW FEATURE 3: Plant activity intensity (past 7 days)
# How many hours was this plant active in the past week?
# =============================================================================
df["plant_active_hours_7d"] = df.groupby("plant_code")["volume_m3"].transform(
    lambda x: (x.shift(1) > 0).rolling(168, min_periods=1).sum()
)
print("Added: plant_active_hours_7d")

# =============================================================================
# NEW FEATURE 4: Holiday proximity
# =============================================================================
# Build holiday set from existing is_holiday flag
holiday_dates = set(df[df["is_holiday"] == 1]["programmed_departure_time"].dt.date.unique())

def days_to_next_holiday(dt):
    d = dt.date()
    for i in range(1, 15):
        if (d + pd.Timedelta(days=i)) in holiday_dates:
            return i
    return 15  # cap at 15

def days_since_last_holiday(dt):
    d = dt.date()
    for i in range(1, 15):
        if (d - pd.Timedelta(days=i)) in holiday_dates:
            return i
    return 15

df["days_to_next_holiday"] = df["programmed_departure_time"].apply(days_to_next_holiday)
df["days_since_last_holiday"] = df["programmed_departure_time"].apply(days_since_last_holiday)
print("Added: days_to_next_holiday, days_since_last_holiday")

# =============================================================================
# NEW FEATURE 6: Month position
# =============================================================================
df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
df["is_month_end"] = (df["day_of_month"] >= 28).astype(int)
print("Added: is_month_start, is_month_end")

# =============================================================================
# NEW FEATURE 7: Extended lag (12h)
# =============================================================================
df["volume_m3_lag_12h"] = df.groupby("plant_code")["volume_m3"].shift(12)
df["remission_count_lag_12h"] = df.groupby("plant_code")["remission_count"].shift(12)
print("Added: volume_m3_lag_12h, remission_count_lag_12h")

# =============================================================================
# Fill any new NaNs
# =============================================================================
new_cols = [
    "consecutive_zero_hours", "volume_same_hour_4w_avg",
    "plant_active_hours_7d",
    "days_to_next_holiday", "days_since_last_holiday",
    "is_month_start", "is_month_end",
    "volume_m3_lag_12h", "remission_count_lag_12h"
]
for col in new_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(0)
        print(f"  Filled NaNs in {col}")

# Fill any remaining NaNs
# 1. days_since_last_open first row per plant
if df["days_since_last_open"].isna().any():
    df["days_since_last_open"] = df["days_since_last_open"].fillna(0)
    print("  Filled NaNs in days_since_last_open")

# 2. Lag/roll features (startup effect at beginning of each plant's time series)
lag_roll_cols = [c for c in df.columns if "lag_" in c or "roll_" in c]
for col in lag_roll_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(0.0)
        print(f"  Filled NaNs in {col}")

# Verify
assert df.isna().sum().sum() == 0, f"Unexpected NaNs in: {df.columns[df.isna().any()].tolist()}"
print(f"\nFinal shape: {df.shape}")

# Save
SAVE_PATH = SCRIPT_DIR.parent / "data" / "processed" / "transformed_dataset_post_FE.xlsx"
df.to_excel(SAVE_PATH, index=False)
print(f"Saved: {SAVE_PATH}")