"""
Train hourly LightGBM model from hourly_features.parquet.

Outputs:
    - app/models/hourly_lgbm.pkl (trained model)
    - app/models/metrics_hourly.json (metrics on test set)
    - app/outputs/forecasts/validation_hourly.csv (predictions on test set for validation/visualization)
"""
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "app" / "data" / "hourly_features.parquet"
MODEL_DIR = ROOT / "app" / "models"
OUTPUT_DIR = ROOT / "app" / "outputs" / "forecasts"

MODEL_PATH = MODEL_DIR / "hourly_lgbm.pkl"
METRICS_PATH = MODEL_DIR / "metrics_hourly.json"
VALIDATION_PATH = OUTPUT_DIR / "validation_hourly.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Final feature set — aligned with model_runner.py inference logic
FEATURE_COLS = [
    "ship_plant_code",
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
    "year",
    "is_weekend",
    "is_sunday",
    "was_open",
    "hour_sin",
    "hour_cos",
    "day_week_sin",
    "day_week_cos",
    "month_sin",
    "month_cos",
    "days_since_last_open",
    "volume_per_remission_7d_avg",
    "is_holiday",
    "volume_m3_lag_24h",
    "volume_m3_lag_48h",
    "volume_m3_lag_1w",
    "volume_m3_roll_mean_24h",
    "volume_m3_roll_mean_48h",
    "volume_m3_roll_mean_1w",
    "volume_m3_roll_std_24h",
    "volume_m3_roll_std_48h",
    "volume_m3_roll_std_1w",
]

TARGET_COL = "volume_m3"


TEST_CUTOFF_DATE = "2025-01-01"
RANDOM_STATE = 42

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": RANDOM_STATE,
}

NUM_BOOST_ROUND = 500
EARLY_STOPPING_ROUNDS = 50


def smape(y_true, y_pred):
    """Symmetric MAPE."""
    return (200 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))).mean()


def train():

    print(f"Loading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} cols")

    # Ensure datetime
    df["hour_bucket"] = pd.to_datetime(df["hour_bucket"])

    train_mask = df["hour_bucket"] < TEST_CUTOFF_DATE
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    print(f"Train: {len(train_df):,} rows ({train_df['hour_bucket'].min()} to {train_df['hour_bucket'].max()})")
    print(f"Test:  {len(test_df):,} rows ({test_df['hour_bucket'].min()} to {test_df['hour_bucket'].max()})")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    print("\nTraining LightGBM...")
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True)],
    )

    print(f"Best iteration: {model.best_iteration}")

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.clip(y_pred, 0, None)

    errors = y_test - y_pred
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    smape_val = smape(y_test, y_pred)

    print(f"\n--- Test Metrics ---")
    print(f"MAE:   {mae:.4f} m3/hr")
    print(f"RMSE:  {rmse:.4f} m3/hr")
    print(f"sMAPE: {smape_val:.2f}%")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {MODEL_PATH}")

    metrics = {
        "mae_hourly_m3": round(float(mae), 4),
        "rmse_hourly_m3": round(float(rmse), 4),
        "smape_pct": round(float(smape_val), 2),
        "validation_rows": int(len(test_df)),
        "date_calculated": datetime.now().isoformat(),
        "test_cutoff_date": TEST_CUTOFF_DATE,
        "best_iteration": int(model.best_iteration),
    }

    # Per-plant metrics
    test_df = test_df.copy()
    test_df["predicted"] = y_pred
    test_df["error"] = errors

    by_plant = {}
    for plant in sorted(test_df["ship_plant_code"].unique()):
        sub = test_df[test_df["ship_plant_code"] == plant]
        by_plant[str(int(plant))] = {
            "mae": round(float(sub["error"].abs().mean()), 4),
            "rmse": round(float(np.sqrt((sub["error"] ** 2).mean())), 4),
            "count": int(len(sub)),
        }
    metrics["by_plant"] = by_plant

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {METRICS_PATH}")

    val_out = test_df[["hour_bucket", "ship_plant_code", "volume_m3", "predicted", "error"]].copy()
    val_out.to_csv(VALIDATION_PATH, index=False)
    print(f"Validation saved: {VALIDATION_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    train()
