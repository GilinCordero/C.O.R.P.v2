"""Inference runner for the hourly LightGBM model."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import holidays

from utils.data_loader import load_model, load_hourly_features

ROOT = Path(__file__).resolve().parents[2]

FEATURE_COLS = [
    "ship_plant_code", "hour", "day_of_week", "month", "day_of_month", "year",
    "is_weekend", "is_sunday", "was_open",
    "hour_sin", "hour_cos", "day_week_sin", "day_week_cos",
    "month_sin", "month_cos",
    "days_since_last_open", "days_to_quincena", "is_quincena",
    "volume_per_remission_7d_avg", "is_holiday", "days_since_first_record",
    "volume_m3_lag_24h", "volume_m3_lag_48h", "volume_m3_lag_1w",
    "volume_m3_roll_mean_24h", "volume_m3_roll_mean_48h", "volume_m3_roll_mean_1w",
    "volume_m3_roll_std_24h", "volume_m3_roll_std_48h", "volume_m3_roll_std_1w",
]


def _days_to_quincena(d):
    if d.day <= 15:
        return 15 - d.day
    else:
        last_day = pd.Timestamp(d.year, d.month, 1) + pd.offsets.MonthEnd(0)
        return (last_day - pd.Timestamp(d)).days


def run_hourly_forecast(plant_id: int, start_time: pd.Timestamp, hours: int = 1080) -> pd.DataFrame:
    """Run recursive hourly forecast for a plant."""
    model = load_model()
    df = load_hourly_features()
    plant_df = df[df["ship_plant_code"] == plant_id].sort_values("hour_bucket").copy()

    if plant_df.empty:
        raise ValueError(f"No historical data for plant {plant_id}")

    hist_values = plant_df["volume_m3"].values.copy()
    n = len(hist_values)
    predicted_values = []
    result_rows = []

    first_record_time = pd.Timestamp(plant_df["hour_bucket"].min())
    mean_vpr = plant_df["volume_per_remission_7d_avg"].mean() if "volume_per_remission_7d_avg" in plant_df.columns else 2.5

    if "was_open" in plant_df.columns:
        open_days = plant_df[plant_df["was_open"] == 1]["hour_bucket"].dt.floor("D").unique()
        last_open_day = pd.Timestamp(open_days.max()) if len(open_days) > 0 else first_record_time
    else:
        last_open_day = first_record_time

    for offset in range(1, hours + 1):
        target_time = start_time + timedelta(hours=offset)
        hr = target_time.hour
        dow = target_time.dayofweek
        month = target_time.month
        dom = target_time.day
        year = target_time.year
        day_floor = target_time.floor("D")

        is_open = 1 if (6 <= hr <= 16 and dow != 6) else 0

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
            "hour_bucket": target_time,
            "ship_plant_code": plant_id,
            "hour": hr,
            "day_of_week": dow,
            "month": month,
            "day_of_month": dom,
            "year": year,
            "is_weekend": int(dow >= 5),
            "is_sunday": int(dow == 6),
            "was_open": is_open,
            "hour_sin": np.sin(2 * np.pi * hr / 24),
            "hour_cos": np.cos(2 * np.pi * hr / 24),
            "day_week_sin": np.sin(2 * np.pi * dow / 7),
            "day_week_cos": np.cos(2 * np.pi * dow / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "days_since_last_open": days_since_last_open,
            "days_to_quincena": _days_to_quincena(target_time),
            "is_quincena": int(dom == 15 or (target_time + pd.offsets.Day(1)).day == 1),
            "volume_per_remission_7d_avg": mean_vpr,
            "is_holiday": int(target_time.date() in holidays.MX(years=[year])),
            "days_since_first_record": max(0, (day_floor - first_record_time).days),
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

        if dow == 6 or not (6 <= hr <= 16):
            pred = 0.0

        predicted_values.append(pred)
        row["predicted_hourly_m3"] = pred
        result_rows.append(row)

    return pd.DataFrame(result_rows)[["hour_bucket", "predicted_hourly_m3"]]


def save_forecast_csv(forecast_df: pd.DataFrame, filename: str = None):
    """Save forecast to outputs/forecasts directory."""
    out_dir = ROOT / "app" / "outputs" / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    path = out_dir / filename
    forecast_df.to_csv(path, index=False)
    return path
