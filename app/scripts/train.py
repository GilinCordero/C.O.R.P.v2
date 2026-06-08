"""LightGBM training module."""
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
    TEST_CUTOFF_DATE,
    LGB_PARAMS,
    NUM_BOOST_ROUND,
    EARLY_STOPPING_ROUNDS,
)

warnings.filterwarnings("ignore")


def train_model(df: pd.DataFrame) -> dict:
    """
    Train LightGBM on hourly features.

    Input: DataFrame from transform.py with FEATURE_COLS + TARGET_COL + TIME_COL

    Returns:
        {
            "model": trained LightGBM model,
            "X_train": training features,
            "y_train": training target,
            "X_test": test features,
            "y_test": test target,
            "y_pred": predictions on test set,
            "test_df": test dataframe with metadata,
        }
    """
    # Temporal split
    train_mask = df[TIME_COL] < TEST_CUTOFF_DATE
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    print(f"Train: {len(train_df):,} rows")
    print(f"Test:  {len(test_df):,} rows")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    # Train
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        LGB_PARAMS,
        train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS)],
    )

    print(f"Best iteration: {model.best_iteration}")

    # Predict
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = np.clip(y_pred, 0, None)  # volume can't be negative

    return {
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "test_df": test_df,
    }


def save_model(model, path: str | Path):
    """Save trained model as pickle."""
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved: {path}")


if __name__ == "__main__":
    import sys
    from transform import create_hourly_features
    from validate import validate_new_data

    if len(sys.argv) > 1:
        path = sys.argv[1]
        validate_new_data(path)

        df_raw = pd.read_excel(path)
        df_features = create_hourly_features(df_raw)

        result = train_model(df_features)
        print(f"Training complete. Test predictions: {len(result['y_pred'])}")

        # Quick check
        mae = np.mean(np.abs(result["y_test"] - result["y_pred"]))
        print(f"MAE (quick): {mae:.4f}")
    else:
        print("Uso: python train.py <archivo.xlsx>")
