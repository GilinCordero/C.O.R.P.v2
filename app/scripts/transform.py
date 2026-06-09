"""Feature Engineering: raw remissions → hourly features ready for training."""
import numpy as np
import pandas as pd
import holidays

from config import FEATURE_COLS, TARGET_COL, TIME_COL


def create_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw remissions DataFrame into hourly aggregated features.

    Input: Concatenated historic + new data (already validated).
           Must have: start_time, ship_plant_code, u_Volumen

    Output: DataFrame with FEATURE_COLS + TARGET_COL + TIME_COL
    """
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])

    # ========================================================================
    # 1. Hourly aggregation
    # ========================================================================
    df["hour_bucket"] = df["start_time"].dt.floor("h")

    hourly = (
        df.groupby(["ship_plant_code", "hour_bucket"])
        .agg(volume_m3=("u_Volumen", "sum"), remission_count=("u_Volumen", "count"))
        .reset_index()
    )

    hourly = hourly.sort_values(["ship_plant_code", "hour_bucket"]).reset_index(drop=True)

    # ========================================================================
    # 2. Time components from hour_bucket
    # ========================================================================
    hourly["hour"] = hourly["hour_bucket"].dt.hour
    hourly["day_of_week"] = hourly["hour_bucket"].dt.dayofweek
    hourly["month"] = hourly["hour_bucket"].dt.month
    hourly["day_of_month"] = hourly["hour_bucket"].dt.day
    hourly["year"] = hourly["hour_bucket"].dt.year

    # ========================================================================
    # 3. Calendar features
    # ========================================================================
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)
    hourly["is_sunday"] = (hourly["day_of_week"] == 6).astype(int)
    hourly["is_saturday"] = (hourly["day_of_week"] == 5).astype(int)

    hourly["is_holiday"] = hourly["hour_bucket"].apply(
        lambda x: 1 if x.date() in holidays.MX(years=[x.year]) else 0
    )

    # ========================================================================
    # 4. was_open
    # Business hours:
    #   Mon-Fri: 7am-5pm  → hour 7 to 16
    #   Sat:     7am-1pm  → hour 7 to 12
    #   Sun:     closed
    # ========================================================================
    is_weekday = hourly["day_of_week"] <= 4
    is_saturday = hourly["day_of_week"] == 5

    hourly["was_open"] = (
        (is_weekday & hourly["hour"].between(7, 16)) |
        (is_saturday & hourly["hour"].between(7, 12))
    ).astype(int)

    # ========================================================================
    # 5. Cyclic encoding
    # ========================================================================
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24)
    hourly["day_week_sin"] = np.sin(2 * np.pi * hourly["day_of_week"] / 7)
    hourly["day_week_cos"] = np.cos(2 * np.pi * hourly["day_of_week"] / 7)
    hourly["month_sin"] = np.sin(2 * np.pi * hourly["month"] / 12)
    hourly["month_cos"] = np.cos(2 * np.pi * hourly["month"] / 12)

    # ========================================================================
    # 6. days_since_last_open
    # Days since this plant had any daily volume > 0.
    # ========================================================================
    hourly["days_since_last_open"] = 0
    for plant in hourly["ship_plant_code"].unique():
        mask = hourly["ship_plant_code"] == plant
        sub = hourly.loc[mask].copy()

        day_floor = sub["hour_bucket"].dt.floor("D")
        daily_vol = sub.groupby(day_floor)["volume_m3"].sum()
        active_days = daily_vol[daily_vol > 0].index.sort_values()

        dsl = []
        for dt in sub["hour_bucket"]:
            d = dt.floor("D")
            past = active_days[active_days < d]
            dsl.append((d - past.max()).days if len(past) > 0 else 0)

        hourly.loc[mask, "days_since_last_open"] = dsl

    # ========================================================================
    # 7. volume_per_remission_7d_avg
    # ========================================================================
    hourly["volume_per_remission_7d_avg"] = 2.5  # default fallback
    for plant in hourly["ship_plant_code"].unique():
        mask = hourly["ship_plant_code"] == plant
        sub = hourly.loc[mask].copy()

        day_floor = sub["hour_bucket"].dt.floor("D")
        daily = (
            sub.groupby(day_floor)
            .agg(vol=("volume_m3", "sum"), cnt=("remission_count", "sum"))
            .reset_index()
        )
        daily["vpr"] = daily["vol"] / daily["cnt"].replace(0, np.nan)
        daily["vpr_7d"] = daily["vpr"].rolling(7, min_periods=1).mean()

        vpr_map = dict(zip(daily["hour_bucket"], daily["vpr_7d"]))
        for idx in sub.index:
            d = sub.loc[idx, "hour_bucket"].floor("D")
            hourly.loc[idx, "volume_per_remission_7d_avg"] = vpr_map.get(d, 2.5)

    # ========================================================================
    # 8. Lag features (24h, 48h, 1w)
    # ========================================================================
    for plant in hourly["ship_plant_code"].unique():
        mask = hourly["ship_plant_code"] == plant
        hourly.loc[mask, "volume_m3_lag_24h"] = hourly.loc[mask, "volume_m3"].shift(24)
        hourly.loc[mask, "volume_m3_lag_48h"] = hourly.loc[mask, "volume_m3"].shift(48)
        hourly.loc[mask, "volume_m3_lag_1w"] = hourly.loc[mask, "volume_m3"].shift(168)

    # ========================================================================
    # 9. Rolling mean / std (24h, 48h, 1w)
    # shift(1) avoids leakage: rolling excludes current hour
    # ========================================================================
    for plant in hourly["ship_plant_code"].unique():
        mask = hourly["ship_plant_code"] == plant
        vol = hourly.loc[mask, "volume_m3"]

        hourly.loc[mask, "volume_m3_roll_mean_24h"] = vol.shift(1).rolling(24, min_periods=1).mean()
        hourly.loc[mask, "volume_m3_roll_mean_48h"] = vol.shift(1).rolling(48, min_periods=1).mean()
        hourly.loc[mask, "volume_m3_roll_mean_1w"] = vol.shift(1).rolling(168, min_periods=1).mean()

        hourly.loc[mask, "volume_m3_roll_std_24h"] = vol.shift(1).rolling(24, min_periods=1).std()
        hourly.loc[mask, "volume_m3_roll_std_48h"] = vol.shift(1).rolling(48, min_periods=1).std()
        hourly.loc[mask, "volume_m3_roll_std_1w"] = vol.shift(1).rolling(168, min_periods=1).std()

    # ========================================================================
    # 10. Final selection + cleanup
    # ========================================================================
    # Avoid duplicates: TIME_COL and ship_plant_code may already be in FEATURE_COLS
    output_cols = list(dict.fromkeys([TIME_COL, "ship_plant_code"] + FEATURE_COLS + [TARGET_COL]))
    hourly = hourly[output_cols].copy()

    # Drop rows where lag/rolling features are NaN (first rows of each plant)
    hourly = hourly.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    hourly = hourly.fillna(0)

    return hourly


if __name__ == "__main__":
    import sys
    from validate import validate_new_data

    if len(sys.argv) > 1:
        path = sys.argv[1]
        result = validate_new_data(path)
        print(f"Validation: {result}")

        df = pd.read_excel(path)
        out = create_hourly_features(df)
        print(f"Output shape: {out.shape}")
        print(f"Columns: {len(out.columns)} -> {out.columns.tolist()}")
    else:
        print("Uso: python transform.py <archivo.xlsx>")
