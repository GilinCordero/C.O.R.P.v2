"""
training.py
===========
Standalone TFT training script for C.O.R.P concrete demand forecasting.

Run this on your training machine (GPU recommended) after copying:
  - data/processed/train_tft.csv
  - data/processed/test_tft.csv

Usage:
  python training.py

Outputs:
  - checkpoints/tft_best.ckpt   (best model checkpoint)
  - checkpoints/tft_latest.ckpt (last epoch checkpoint)
  - results/test_predictions.csv (predictions on the test set)
  - results/metrics.txt          (MAE, RMSE, MAPE per plant)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, RMSE, MAPE
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# =============================================================================
# CONFIGURATION — adjust these to your needs
# =============================================================================

DATA_DIR = Path("data/processed")          # folder containing train_tft.csv & test_tft.csv
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

TRAIN_CSV = DATA_DIR / "train_tft.csv"
TEST_CSV = DATA_DIR / "test_tft.csv"

# TFT / training hyperparameters
MAX_ENCODER_LENGTH = 168        # history: 1 week of hourly data
MAX_PREDICTION_LENGTH = 24      # forecast horizon: next 24 hours
BATCH_SIZE = 128
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 32                # hidden size of TFT (start small, increase if underfitting)
ATTENTION_HEAD_SIZE = 2
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 16

# Early stopping patience
EARLY_STOPPING_PATIENCE = 5

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# 1. LOAD DATA
# =============================================================================

print("=" * 60)
print("C.O.R.P — TFT Training Script")
print("=" * 60)

if not TRAIN_CSV.exists() or not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Missing CSV files. Expected:\n  {TRAIN_CSV}\n  {TEST_CSV}\n"
        "Run the modeling notebook first to export them."
    )

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Ensure hour_bucket is datetime
train_df["hour_bucket"] = pd.to_datetime(train_df["hour_bucket"])
test_df["hour_bucket"] = pd.to_datetime(test_df["hour_bucket"])

# TFT requires group_ids to be categorical/string, not numeric
train_df["ship_plant_code"] = train_df["ship_plant_code"].astype(str)
test_df["ship_plant_code"] = test_df["ship_plant_code"].astype(str)

print(f"\nTrain: {len(train_df):,} rows | {train_df['hour_bucket'].min()} → {train_df['hour_bucket'].max()}")
print(f"Test:  {len(test_df):,} rows  | {test_df['hour_bucket'].min()} → {test_df['hour_bucket'].max()}")

# =============================================================================
# 2. DEFINE FEATURE GROUPS
# =============================================================================

# These are known for both encoder (history) and decoder (future)
TIME_VARYING_KNOWN_REALS = [
    "month_sin",
    "month_cos",
    "day_month_sin",
    "day_month_cos",
    "day_week_sin",
    "day_week_cos",
    "hour_sin",
    "hour_cos",
]

# These are ONLY known for the encoder (historical values)
TIME_VARYING_UNKNOWN_REALS = [
    # lags
    "volume_m3_lag_24h",
    "volume_m3_lag_48h",
    "volume_m3_lag_1w",
    "remission_count_lag_24h",
    "remission_count_lag_48h",
    "remission_count_lag_1w",
    # rolling means
    "volume_m3_roll_mean_24h",
    "volume_m3_roll_mean_48h",
    "volume_m3_roll_mean_1w",
    "remission_count_roll_mean_24h",
    "remission_count_roll_mean_48h",
    "remission_count_roll_mean_1w",
    # rolling stds
    "volume_m3_roll_std_24h",
    "volume_m3_roll_std_48h",
    "volume_m3_roll_std_1w",
    "remission_count_roll_std_24h",
    "remission_count_roll_std_48h",
    "remission_count_roll_std_1w",
]

# =============================================================================
# 3. BUILD TIME-SERIES DATASETS
# =============================================================================

print("\nBuilding TimeSeriesDataSet...")

# Training dataset
training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="volume_m3",
    group_ids=["ship_plant_code"],
    max_encoder_length=MAX_ENCODER_LENGTH,
    max_prediction_length=MAX_PREDICTION_LENGTH,
    static_categoricals=["ship_plant_code"],
    time_varying_known_reals=TIME_VARYING_KNOWN_REALS,
    time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
    target_normalizer=GroupNormalizer(groups=["ship_plant_code"]),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=False,
)

# Validation dataset = last 6 months of training data (or ~20% of time steps)
# We create it from the training dataset so it shares the same normalization.
cutoff_time_idx = train_df["time_idx"].max() - int(MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH + 5000)
validation = TimeSeriesDataSet.from_dataset(
    training,
    train_df[train_df["time_idx"] > cutoff_time_idx],
    predict=True,
    stop_randomization=True,
)

# Test dataset (predict mode)
testing = TimeSeriesDataSet.from_dataset(
    training,
    test_df,
    predict=True,
    stop_randomization=True,
)

# DataLoaders
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)
test_dataloader = testing.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)

print(f"  Training samples:   {len(training):,}")
print(f"  Validation samples: {len(validation):,}")
print(f"  Test samples:       {len(testing):,}")

# =============================================================================
# 4. DEFINE MODEL
# =============================================================================

print("\nInitializing TemporalFusionTransformer...")

# Determine number of quantiles for output (default is 7 quantiles)
# For point forecasting we can keep the default QuantileLoss.
# If you want a single point estimate, override loss=RMSE() or loss=MAE().

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LEARNING_RATE,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTENTION_HEAD_SIZE,
    dropout=DROPOUT,
    hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
    output_size=7,  # number of quantiles by default (QuantileLoss)
    loss=MAE(),     # use MAE for point forecasting; change to QuantileLoss() if you want intervals
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=3,
)

print(f"  Number of parameters: {tft.size()/1e6:.2f}M")

# =============================================================================
# 5. TRAIN
# =============================================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_DIR,
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="tft_best",
)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=EARLY_STOPPING_PATIENCE,
    verbose=True,
    mode="min",
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

logger = TensorBoardLogger(RESULTS_DIR, name="tft_logs")

trainer = Trainer(
    max_epochs=MAX_EPOCHS,
    accelerator="auto",          # uses GPU if available, otherwise CPU
    devices=1,
    gradient_clip_val=0.1,
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    logger=logger,
    enable_progress_bar=True,
)

print("\nStarting training...")
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# Save the best model path
best_model_path = checkpoint_callback.best_model_path
print(f"\nBest model saved at: {best_model_path}")

# Also save the last epoch explicitly
last_model_path = CHECKPOINT_DIR / "tft_latest.ckpt"
trainer.save_checkpoint(last_model_path)
print(f"Last model saved at: {last_model_path}")

# =============================================================================
# 6. EVALUATE ON TEST SET
# =============================================================================

print("\nEvaluating on test set...")

# Load best model for evaluation
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
best_tft.eval()

# Raw predictions (returns x, y, and raw output dict)
predictions = best_tft.predict(test_dataloader, return_x=True, return_index=True)

# predictions.output is shape (num_samples, max_prediction_length, num_quantiles)
# For MAE loss the median (index 3 of 7 quantiles) is the point prediction.
# If you used QuantileLoss, adjust accordingly.
y_pred = predictions.output.cpu().numpy()
if y_pred.ndim == 3 and y_pred.shape[-1] > 1:
    y_pred_point = y_pred[:, :, 3]  # median quantile
else:
    y_pred_point = y_pred.squeeze(-1)

# Ground truth
y_true = predictions.y[0].cpu().numpy()  # shape (num_samples, max_prediction_length)

# Index (plant + time info)
index_df = predictions.index

# Flatten for metric calculation
y_true_flat = y_true.reshape(-1)
y_pred_flat = y_pred_point.reshape(-1)
mask = ~np.isnan(y_true_flat)

mae = np.mean(np.abs(y_true_flat[mask] - y_pred_flat[mask]))
rmse = np.sqrt(np.mean((y_true_flat[mask] - y_pred_flat[mask]) ** 2))
mape = np.mean(np.abs((y_true_flat[mask] - y_pred_flat[mask]) / (y_true_flat[mask] + 1e-8))) * 100

print(f"\nOverall Test Metrics:")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAPE: {mape:.2f}%")

# Per-plant metrics
print("\nPer-Plant Metrics:")
results_records = []
for plant in sorted(index_df["ship_plant_code"].unique()):
    mask_plant = index_df["ship_plant_code"] == plant
    idxs = mask_plant.values
    yt = y_true[idxs].reshape(-1)
    yp = y_pred_point[idxs].reshape(-1)
    m = ~np.isnan(yt)
    if m.sum() == 0:
        continue
    mae_p = np.mean(np.abs(yt[m] - yp[m]))
    rmse_p = np.sqrt(np.mean((yt[m] - yp[m]) ** 2))
    mape_p = np.mean(np.abs((yt[m] - yp[m]) / (yt[m] + 1e-8))) * 100
    print(f"  Plant {plant}: MAE={mae_p:.4f} RMSE={rmse_p:.4f} MAPE={mape_p:.2f}%")
    results_records.append({"plant": plant, "MAE": mae_p, "RMSE": rmse_p, "MAPE": mape_p})

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================

# Save metrics
metrics_path = RESULTS_DIR / "metrics.txt"
with open(metrics_path, "w") as f:
    f.write("C.O.R.P TFT Test Metrics\n")
    f.write("=" * 40 + "\n")
    f.write(f"Overall MAE:  {mae:.4f}\n")
    f.write(f"Overall RMSE: {rmse:.4f}\n")
    f.write(f"Overall MAPE: {mape:.2f}%\n\n")
    f.write("Per-Plant Metrics:\n")
    for r in results_records:
        f.write(f"  Plant {r['plant']}: MAE={r['MAE']:.4f} RMSE={r['RMSE']:.4f} MAPE={r['MAPE']:.2f}%\n")
print(f"\nMetrics saved to: {metrics_path}")

# Save a sample of predictions for inspection
# Build a tidy dataframe: one row per (plant, time_idx, horizon_step)
pred_records = []
for i, row in index_df.iterrows():
    plant = row["ship_plant_code"]
    time_idx = row["time_idx"]
    for h in range(MAX_PREDICTION_LENGTH):
        pred_records.append({
            "ship_plant_code": plant,
            "prediction_time_idx": int(time_idx),
            "horizon_step": h + 1,
            "actual": float(y_true[i, h]) if not np.isnan(y_true[i, h]) else None,
            "predicted": float(y_pred_point[i, h]),
        })

pred_df = pd.DataFrame(pred_records)
pred_path = RESULTS_DIR / "test_predictions.csv"
pred_df.to_csv(pred_path, index=False)
print(f"Predictions saved to: {pred_path}")

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
