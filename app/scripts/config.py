"""Pipeline configuration: paths, features, model hyperparameters."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (relative to repository root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # repo root

# Local paths (used during development / local runs)
LOCAL_PATHS = {
    "historic_raw": ROOT / "Modulo_Alberto" / "data" / "processed" / "remissions_db_imputed_all_plants_2025.xlsx",
    "new_raw_dir": ROOT / "data" / "uploads",
    "output_dir": ROOT / "app" / "models",
    "forecasts_dir": ROOT / "app" / "outputs" / "forecasts",
    "processed_dir": ROOT / "app" / "data",
}

# Google Drive paths (used in production / GitHub Actions)
DRIVE_PATHS = {
    "folder_name": "GCC_Corp",
    "historic_raw": "raw/original/remissions_db_imputed_all_plants_2025.xlsx",
    "pending_dir": "raw/pending",
    "output_parquet": "processed/hourly_features.parquet",
    "output_model": "models/current/hourly_lgbm.pkl",
    "output_metrics": "metrics/metrics_current.json",
    "output_validation": "processed/validation_hourly.csv",
    "model_archive_dir": "models/archive",
    "version_file": "config/version.json",
}

# ---------------------------------------------------------------------------
# Data schema (original column names, NOT renamed)
# ---------------------------------------------------------------------------
# Columns that MUST exist in the uploaded Excel.
# Any other columns are ignored (kept or dropped during transform).
REQUIRED_COLUMNS = [
    "start_time",           # datetime: when the order was placed
    "ship_plant_code",      # int: plant ID
    "u_Volumen",            # float: volume in m3
]

# Columns that MUST NOT have nulls.
CRITICAL_COLUMNS = REQUIRED_COLUMNS.copy()

# Valid plant codes
VALID_PLANTS = [510, 511, 512, 514, 515, 710]

# If True, the new dataset must contain at least one row per plant
REQUIRE_ALL_PLANTS = True

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# Final feature set used by the model (27 features)
FEATURE_COLS = [
    "ship_plant_code",
    "hour",
    "day_of_week",
    "month",
    "day_of_month",
    "year",
    "is_weekend",
    "is_sunday",
    "is_saturday",
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
TIME_COL = "hour_bucket"

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------
# TODO: Consider dynamic cutoff for retraining (e.g., last 20% of data)
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

# ---------------------------------------------------------------------------
# Validation thresholds
# ---------------------------------------------------------------------------
MAX_NULL_RATIO = 0.10  # reject if >10% of critical columns are null
