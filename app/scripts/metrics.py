"""Metrics calculation and reporting."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import TARGET_COL, TIME_COL


def calculate_metrics(test_df: pd.DataFrame, y_pred: np.ndarray) -> tuple:
    """
    Calculate metrics from test set predictions.

    Input:
        test_df: DataFrame with TIME_COL, ship_plant_code, TARGET_COL
        y_pred: predicted values (same length as test_df)

    Returns:
        (metrics_dict, validation_df)
    """
    df = test_df.copy()
    df["predicted"] = y_pred
    df["error"] = df[TARGET_COL] - df["predicted"]

    # Global metrics
    mae = df["error"].abs().mean()
    rmse = np.sqrt((df["error"] ** 2).mean())
    metrics = {
        "mae_hourly_m3": round(float(mae), 4),
        "rmse_hourly_m3": round(float(rmse), 4),
        "validation_rows": int(len(df)),
        "date_calculated": datetime.now().isoformat(),
    }

    # Per-plant metrics
    by_plant = {}
    for plant in sorted(df["ship_plant_code"].unique()):
        sub = df[df["ship_plant_code"] == plant]
        by_plant[str(int(plant))] = {
            "mae": round(float(sub["error"].abs().mean()), 4),
            "rmse": round(float(np.sqrt((sub["error"] ** 2).mean())), 4),
            "count": int(len(sub)),
        }
    metrics["by_plant"] = by_plant

    return metrics, df


def save_metrics(metrics: dict, path: str | Path):
    """Save metrics dict to JSON."""
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {path}")


def save_validation_csv(df: pd.DataFrame, path: str | Path):
    """Save validation DataFrame with app-compatible column names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    out = df[[TIME_COL, "ship_plant_code", TARGET_COL, "predicted", "error"]].copy()
    out.columns = ["hour_bucket", "ship_plant_code", "volume_m3", "predicted", "error"]
    out.to_csv(path, index=False)
    print(f"Validation CSV saved: {path}")


if __name__ == "__main__":
    import sys
    from train import train_model
    from transform import create_hourly_features
    from validate import validate_new_data

    if len(sys.argv) > 1:
        path = sys.argv[1]
        validate_new_data(path)

        df_raw = pd.read_excel(path)
        df_features = create_hourly_features(df_raw)

        train_result = train_model(df_features)
        metrics, val_df = calculate_metrics(train_result["test_df"], train_result["y_pred"])

        print(f"\n--- Metrics ---")
        print(f"MAE:   {metrics['mae_hourly_m3']}")
        print(f"RMSE:  {metrics['rmse_hourly_m3']}")
    else:
        print("Uso: python metrics.py <archivo.xlsx>")
