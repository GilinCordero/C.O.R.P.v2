"""Pipeline orchestrator: downloads data, runs full pipeline, uploads results."""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Local modules
from config import (
    LOCAL_PATHS,
    DRIVE_PATHS,
    FEATURE_COLS,
    TARGET_COL,
    TIME_COL,
)
from validate import validate_new_data, ValidationError
from transform import create_hourly_features
from train import train_model, save_model
from metrics import calculate_metrics, save_metrics, save_validation_csv

# Drive integration (only used in production / GitHub Actions)
# Ensure repo root is in path so app.api imports work in CI
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    from app.api.drive_client import (
        get_drive_service,
        get_or_create_folder,
        find_file,
        download_file,
        upload_file,
    )
    DRIVE_AVAILABLE = True
except ImportError:
    DRIVE_AVAILABLE = False


def run_pipeline(
    new_data_path: str | Path = None,
    use_drive: bool = False,
    drive_folder_id: str = None,
):
    """
    Run the full retraining pipeline.

    Args:
        new_data_path: Path to new Excel file (local mode). Ignored if use_drive=True.
        use_drive: If True, download from and upload to Google Drive.
        drive_folder_id: Google Drive folder ID for GCC_Corp root.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"=== PIPELINE START: {timestamp} ===")

    # ======================================================================
    # 1. LOAD DATA
    # ======================================================================
    if use_drive and DRIVE_AVAILABLE:
        print("\n--- Drive Mode ---")
        service = get_drive_service()

        # Find or create folder structure
        if not drive_folder_id:
            drive_folder_id = get_or_create_folder(service, DRIVE_PATHS["folder_name"])

        # Download historic from raw/original/
        historic_local = Path("/tmp/historic.xlsx")
        raw_folder_id = get_or_create_folder(service, "raw", parent_id=drive_folder_id)
        original_folder_id = get_or_create_folder(service, "original", parent_id=raw_folder_id)
        historic_id = find_file(service, "remissions_db_imputed_all_plants_2025.xlsx", parent_id=original_folder_id)
        if not historic_id:
            raise FileNotFoundError("Historic file not found in Drive at GCC_Corp/raw/original/")
        download_file(service, historic_id, historic_local)

        # Download new pending file from raw/pending/
        pending_folder_id = get_or_create_folder(service, "pending", parent_id=raw_folder_id)
        # Find the newest file in pending folder
        pending_files = service.files().list(
            q=f"'{pending_folder_id}' in parents and trashed=false",
            orderBy="createdTime desc",
            fields="files(id, name, createdTime)"
        ).execute().get("files", [])
        if not pending_files:
            raise FileNotFoundError("No pending files found in Drive at GCC_Corp/raw/pending/")
        newest = pending_files[0]
        new_local = Path(f"/tmp/{newest['name']}")
        download_file(service, newest["id"], new_local)
        print(f"Using pending file: {newest['name']}")

    else:
        print("\n--- Local Mode ---")
        if not new_data_path:
            raise ValueError("Must provide new_data_path in local mode")

        historic_local = LOCAL_PATHS["historic_raw"]
        new_local = Path(new_data_path)

        if not historic_local.exists():
            raise FileNotFoundError(f"Historic file not found: {historic_local}")
        if not new_local.exists():
            raise FileNotFoundError(f"New file not found: {new_local}")

    # ======================================================================
    # 2. VALIDATE NEW DATA
    # ======================================================================
    print("\n--- Validation ---")
    try:
        validation_result = validate_new_data(new_local)
        print(f"Validation passed: {validation_result}")
    except ValidationError as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)

    # ======================================================================
    # 3. CONCATENATE HISTORIC + NEW
    # ======================================================================
    print("\n--- Loading & Concatenating ---")
    df_historic = pd.read_excel(historic_local)
    df_new = pd.read_excel(new_local)

    # Drop helper columns that may exist in historic but not needed
    cols_to_drop = ["hour", "day_of_week", "hour_bucket", "month"]
    for col in cols_to_drop:
        if col in df_historic.columns:
            df_historic = df_historic.drop(columns=[col])

    df_combined = pd.concat([df_historic, df_new], ignore_index=True)
    print(f"Combined rows: {len(df_combined):,}")

    # ======================================================================
    # 4. FEATURE ENGINEERING
    # ======================================================================
    print("\n--- Feature Engineering ---")
    df_features = create_hourly_features(df_combined)
    print(f"Features shape: {df_features.shape}")

    # ======================================================================
    # 5. TRAIN MODEL
    # ======================================================================
    print("\n--- Training ---")
    train_result = train_model(df_features)
    model = train_result["model"]

    # ======================================================================
    # 6. METRICS
    # ======================================================================
    print("\n--- Metrics ---")
    metrics, val_df = calculate_metrics(train_result["test_df"], train_result["y_pred"])
    print(f"MAE:   {metrics['mae_hourly_m3']}")
    print(f"RMSE:  {metrics['rmse_hourly_m3']}")

    # ======================================================================
    # 7. SAVE ARTIFACTS (local)
    # ======================================================================
    print("\n--- Saving Artifacts ---")
    LOCAL_PATHS["output_dir"].mkdir(parents=True, exist_ok=True)
    LOCAL_PATHS["forecasts_dir"].mkdir(parents=True, exist_ok=True)
    LOCAL_PATHS["processed_dir"].mkdir(parents=True, exist_ok=True)

    # Model
    model_path = LOCAL_PATHS["output_dir"] / f"hourly_lgbm_{timestamp}.pkl"
    save_model(model, model_path)

    # Metrics
    metrics_path = LOCAL_PATHS["output_dir"] / f"metrics_{timestamp}.json"
    save_metrics(metrics, metrics_path)

    # Validation CSV
    val_path = LOCAL_PATHS["forecasts_dir"] / f"validation_{timestamp}.csv"
    save_validation_csv(val_df, val_path)

    # Parquet
    parquet_path = LOCAL_PATHS["processed_dir"] / f"hourly_features_{timestamp}.parquet"
    df_features.to_parquet(parquet_path, index=False)
    print(f"Parquet: {parquet_path}")

    # ======================================================================
    # 8. UPLOAD TO DRIVE (if in drive mode)
    # ======================================================================
    if use_drive and DRIVE_AVAILABLE:
        print("\n--- Uploading to Drive ---")

        # models/archive/ (timestamped backup)
        models_folder_id = get_or_create_folder(service, "models", parent_id=drive_folder_id)
        archive_folder_id = get_or_create_folder(service, "archive", parent_id=models_folder_id)
        upload_file(service, model_path, parent_id=archive_folder_id)

        # models/pending/ (latest pending model, fixed name)
        pending_models_folder_id = get_or_create_folder(service, "pending", parent_id=models_folder_id)
        # Remove previous pending model if exists
        old_pending_model = find_file(service, "hourly_lgbm.pkl", parent_id=pending_models_folder_id)
        if old_pending_model:
            from app.api.drive_client import delete_file
            delete_file(service, old_pending_model)
        upload_file(service, model_path, parent_id=pending_models_folder_id)

        # metrics/pending/
        metrics_folder_id = get_or_create_folder(service, "metrics", parent_id=drive_folder_id)
        pending_metrics_folder_id = get_or_create_folder(service, "pending", parent_id=metrics_folder_id)
        old_pending_metrics = find_file(service, "metrics_pending.json", parent_id=pending_metrics_folder_id)
        if old_pending_metrics:
            from app.api.drive_client import delete_file
            delete_file(service, old_pending_metrics)
        upload_file(service, metrics_path, parent_id=pending_metrics_folder_id)

        # validation/pending/
        validation_folder_id = get_or_create_folder(service, "validation", parent_id=drive_folder_id)
        pending_validation_folder_id = get_or_create_folder(service, "pending", parent_id=validation_folder_id)
        old_pending_val = find_file(service, "validation_pending.csv", parent_id=pending_validation_folder_id)
        if old_pending_val:
            from app.api.drive_client import delete_file
            delete_file(service, old_pending_val)
        upload_file(service, val_path, parent_id=pending_validation_folder_id)

        # config/version.json
        config_folder_id = get_or_create_folder(service, "config", parent_id=drive_folder_id)
        version_data = {
            "active": None,
            "pending": timestamp,
            "status": "pending"
        }
        version_local = Path("/tmp/version.json")
        with open(version_local, "w") as f:
            json.dump(version_data, f)
        old_version = find_file(service, "version.json", parent_id=config_folder_id)
        if old_version:
            from app.api.drive_client import delete_file
            delete_file(service, old_version)
        upload_file(service, version_local, parent_id=config_folder_id)

        print("\n--- Drive upload complete ---")

    print(f"\n=== PIPELINE COMPLETE: {timestamp} ===")
    return {
        "model_path": model_path,
        "metrics_path": metrics_path,
        "validation_path": val_path,
        "parquet_path": parquet_path,
        "metrics": metrics,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CORP Pipeline Orchestrator")
    parser.add_argument("--new-data", type=str, help="Path to new Excel file (local mode)")
    parser.add_argument("--use-drive", action="store_true", help="Use Google Drive mode")
    parser.add_argument("--drive-folder-id", type=str, help="Google Drive folder ID")

    args = parser.parse_args()

    if args.use_drive:
        run_pipeline(use_drive=True, drive_folder_id=args.drive_folder_id)
    else:
        if not args.new_data:
            print("Error: --new-data required in local mode")
            sys.exit(1)
        run_pipeline(new_data_path=args.new_data)
