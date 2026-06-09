"""Drive synchronization layer for model approval workflow.

This module reads/writes the GCC_Corp Drive structure:
    models/pending/      → new model awaiting approval
    models/current/      → approved active model
    models/archive/      → historical backups
    metrics/pending/     → new metrics awaiting approval
    metrics/current/     → approved active metrics
    processed/pending/   → new validation awaiting approval
    config/version.json  → state tracker
"""
import json
import sys
from pathlib import Path

# Ensure repo root is in path for app.api imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from app.api.drive_client import (
    get_drive_service,
    find_folder,
    get_or_create_folder,
    get_file_in_folder,
    download_file,
    upload_file,
    copy_file,
    delete_file,
)

DRIVE_FOLDER_NAME = "GCC_Corp"


def _get_gcc_folder_id(service):
    """Find the GCC_Corp root folder ID."""
    folder_id = find_folder(service, DRIVE_FOLDER_NAME)
    if not folder_id:
        raise FileNotFoundError(f"{DRIVE_FOLDER_NAME} folder not found in Drive")
    return folder_id


def _get_subfolder_id(service, gcc_id, *path_parts):
    """Navigate GCC_Corp/sub1/sub2/... and return the deepest folder ID."""
    current_id = gcc_id
    for part in path_parts:
        current_id = get_or_create_folder(service, part, parent_id=current_id)
    return current_id


def get_version(service=None):
    """Read config/version.json from Drive. Returns dict or None."""
    if service is None:
        service = get_drive_service()
    gcc_id = _get_gcc_folder_id(service)
    config_id = _get_subfolder_id(service, gcc_id, "config")
    version_file_id = get_file_in_folder(service, "version.json", config_id)
    if not version_file_id:
        return None
    local_path = Path("/tmp/drive_version.json")
    download_file(service, version_file_id, local_path)
    with open(local_path, "r") as f:
        return json.load(f)


def get_drive_pending_metrics(service=None):
    """Download and return metrics/pending/metrics_pending.json as dict.

    Returns None if no pending model exists.
    """
    if service is None:
        service = get_drive_service()

    version = get_version(service)
    if not version or version.get("status") != "pending":
        return None

    gcc_id = _get_gcc_folder_id(service)
    metrics_id = _get_subfolder_id(service, gcc_id, "metrics", "pending")
    file_id = get_file_in_folder(service, "metrics_pending.json", metrics_id)
    if not file_id:
        return None

    local_path = Path("/tmp/metrics_pending.json")
    download_file(service, file_id, local_path)
    with open(local_path, "r") as f:
        return json.load(f)


def approve_drive_model(service=None):
    """Promote pending model to current in Drive.

    1. Copy models/pending/hourly_lgbm.pkl → models/current/
    2. Copy metrics/pending/metrics_pending.json → metrics/current/metrics_current.json
    3. Update config/version.json status → "active"
    Returns True on success.
    """
    if service is None:
        service = get_drive_service()

    gcc_id = _get_gcc_folder_id(service)

    # models/pending/ → models/current/
    models_id = _get_subfolder_id(service, gcc_id, "models")
    pending_models_id = _get_subfolder_id(service, gcc_id, "models", "pending")
    current_models_id = _get_subfolder_id(service, gcc_id, "models", "current")

    pending_model_id = get_file_in_folder(service, "hourly_lgbm.pkl", pending_models_id)
    if not pending_model_id:
        raise FileNotFoundError("No pending model found in Drive")

    # Remove old current model if exists
    old_current = get_file_in_folder(service, "hourly_lgbm.pkl", current_models_id)
    if old_current:
        delete_file(service, old_current)
    copy_file(service, pending_model_id, "hourly_lgbm.pkl", parent_id=current_models_id)

    # metrics/pending/ → metrics/current/
    metrics_id = _get_subfolder_id(service, gcc_id, "metrics")
    pending_metrics_id = _get_subfolder_id(service, gcc_id, "metrics", "pending")
    current_metrics_id = _get_subfolder_id(service, gcc_id, "metrics", "current")

    pending_metrics_id_file = get_file_in_folder(service, "metrics_pending.json", pending_metrics_id)
    if pending_metrics_id_file:
        old_current_metrics = get_file_in_folder(service, "metrics_current.json", current_metrics_id)
        if old_current_metrics:
            delete_file(service, old_current_metrics)
        copy_file(service, pending_metrics_id_file, "metrics_current.json", parent_id=current_metrics_id)

    # Update version.json
    config_id = _get_subfolder_id(service, gcc_id, "config")
    old_version = get_file_in_folder(service, "version.json", config_id)
    version_data = {
        "active": version.get("pending") if version else None,
        "pending": None,
        "status": "active"
    }
    version_local = Path("/tmp/version.json")
    with open(version_local, "w") as f:
        json.dump(version_data, f)
    if old_version:
        delete_file(service, old_version)
    upload_file(service, version_local, parent_id=config_id)

    print("[Drive] Model approved: pending → current")
    return True


def reject_drive_model(service=None):
    """Reject pending model: delete pending files and update version.

    1. Delete models/pending/hourly_lgbm.pkl
    2. Delete metrics/pending/metrics_pending.json
    3. Delete processed/pending/validation_pending.csv
    4. Update config/version.json status → "rejected"
    Returns True on success.
    """
    if service is None:
        service = get_drive_service()

    gcc_id = _get_gcc_folder_id(service)

    # Delete pending model
    pending_models_id = _get_subfolder_id(service, gcc_id, "models", "pending")
    pending_model = get_file_in_folder(service, "hourly_lgbm.pkl", pending_models_id)
    if pending_model:
        delete_file(service, pending_model)

    # Delete pending metrics
    pending_metrics_id = _get_subfolder_id(service, gcc_id, "metrics", "pending")
    pending_metrics = get_file_in_folder(service, "metrics_pending.json", pending_metrics_id)
    if pending_metrics:
        delete_file(service, pending_metrics)

    # Delete pending validation
    pending_processed_id = _get_subfolder_id(service, gcc_id, "processed", "pending")
    pending_val = get_file_in_folder(service, "validation_pending.csv", pending_processed_id)
    if pending_val:
        delete_file(service, pending_val)

    # Update version.json
    config_id = _get_subfolder_id(service, gcc_id, "config")
    old_version = get_file_in_folder(service, "version.json", config_id)
    version_data = {
        "active": None,
        "pending": None,
        "status": "rejected"
    }
    version_local = Path("/tmp/version.json")
    with open(version_local, "w") as f:
        json.dump(version_data, f)
    if old_version:
        delete_file(service, old_version)
    upload_file(service, version_local, parent_id=config_id)

    print("[Drive] Model rejected: pending files deleted")
    return True


def download_drive_current_to_local(service=None):
    """Download the approved current model and metrics from Drive to local paths.

    Returns True on success.
    """
    if service is None:
        service = get_drive_service()

    gcc_id = _get_gcc_folder_id(service)

    # Download model
    current_models_id = _get_subfolder_id(service, gcc_id, "models", "current")
    model_file_id = get_file_in_folder(service, "hourly_lgbm.pkl", current_models_id)
    if not model_file_id:
        raise FileNotFoundError("No current model found in Drive")

    local_model = Path(__file__).resolve().parents[2] / "app" / "models" / "hourly_lgbm.pkl"
    download_file(service, model_file_id, local_model)

    # Download metrics
    current_metrics_id = _get_subfolder_id(service, gcc_id, "metrics", "current")
    metrics_file_id = get_file_in_folder(service, "metrics_current.json", current_metrics_id)
    if metrics_file_id:
        local_metrics = Path(__file__).resolve().parents[2] / "app" / "models" / "metrics_hourly.json"
        download_file(service, metrics_file_id, local_metrics)

    print("[Drive] Current model downloaded to local")
    return True
