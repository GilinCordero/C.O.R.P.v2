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


DATA_DIR = Path("data/processed")      
RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")

TRAIN_CSV = DATA_DIR / "train_tft.csv"
TEST_CSV = DATA_DIR / "test_tft.csv"

MAX_ENCODER_LENGTH = 168       
MAX_PREDICTION_LENGTH = 24     
BATCH_SIZE = 128
MAX_EPOCHS = 50
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 32             
ATTENTION_HEAD_SIZE = 2
DROPOUT = 0.1
HIDDEN_CONTINUOUS_SIZE = 16

EARLY_STOPPING_PATIENCE = 5

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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

train_df["hour_bucket"] = pd.to_datetime(train_df["hour_bucket"])
test_df["hour_bucket"] = pd.to_datetime(test_df["hour_bucket"])

train_df["ship_plant_code"] = train_df["ship_plant_code"].astype(str)
test_df["ship_plant_code"] = test_df["ship_plant_code"].astype(str)

print(f"\nTrain: {len(train_df):,} rows | {train_df['hour_bucket'].min()} → {train_df['hour_bucket'].max()}")
print(f"Test:  {len(test_df):,} rows  | {test_df['hour_bucket'].min()} → {test_df['hour_bucket'].max()}")



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


print("\nBuilding TimeSeriesDataSet...")

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


cutoff_time_idx = train_df["time_idx"].max() - int(MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH + 5000)
validation = TimeSeriesDataSet.from_dataset(
    training,
    train_df[train_df["time_idx"] > cutoff_time_idx],
    predict=True,
    stop_randomization=True,
)

testing = TimeSeriesDataSet.from_dataset(
    training,
    test_df,
    predict=True,
    stop_randomization=True,
)

train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)
test_dataloader = testing.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=0)

print(f"  Training samples:   {len(training):,}")
print(f"  Validation samples: {len(validation):,}")
print(f"  Test samples:       {len(testing):,}")


print("\nInitializing TemporalFusionTransformer...")


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=LEARNING_RATE,
    hidden_size=HIDDEN_SIZE,
    attention_head_size=ATTENTION_HEAD_SIZE,
    dropout=DROPOUT,
    hidden_continuous_size=HIDDEN_CONTINUOUS_SIZE,
    output_size=7,  
    loss=MAE(),     
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=3,
)

print(f"  Number of parameters: {tft.size()/1e6:.2f}M")


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
    accelerator="auto",          
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

best_model_path = checkpoint_callback.best_model_path
print(f"\nBest model saved at: {best_model_path}")

last_model_path = CHECKPOINT_DIR / "tft_latest.ckpt"
trainer.save_checkpoint(last_model_path)
print(f"Last model saved at: {last_model_path}")


print("\nEvaluating on test set...")

best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
best_tft.eval()

predictions = best_tft.predict(test_dataloader, return_x=True, return_index=True)


y_pred = predictions.output.cpu().numpy()
if y_pred.ndim == 3 and y_pred.shape[-1] > 1:
    y_pred_point = y_pred[:, :, 3]  
else:
    y_pred_point = y_pred.squeeze(-1)

y_true = predictions.y[0].cpu().numpy() 

index_df = predictions.index

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
