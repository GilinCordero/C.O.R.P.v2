"""Google Drive API client using GCP Service Account."""
import io
import os
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = Path(__file__).resolve().parent / "gccc-498819-28eb8af3eb04.json"


def get_drive_service():
    """Authenticate and return Google Drive API service."""
    if not SERVICE_ACCOUNT_FILE.exists():
        raise FileNotFoundError(f"Service account key not found: {SERVICE_ACCOUNT_FILE}")

    credentials = service_account.Credentials.from_service_account_file(
        str(SERVICE_ACCOUNT_FILE),
        scopes=SCOPES,
    )
    return build("drive", "v3", credentials=credentials)


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
def find_folder(service, name, parent_id=None):
    """Find a folder by name. Returns folder ID or None."""
    query = f"mimeType='application/vnd.google-apps.folder' and name='{name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def create_folder(service, name, parent_id=None):
    """Create a folder. Returns folder ID."""
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id] if parent_id else [],
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    return folder["id"]


def get_or_create_folder(service, name, parent_id=None):
    """Find folder or create if not exists. Returns folder ID."""
    folder_id = find_folder(service, name, parent_id)
    if folder_id:
        return folder_id
    return create_folder(service, name, parent_id)


# ---------------------------------------------------------------------------
# Files: download
# ---------------------------------------------------------------------------
def find_file(service, name, parent_id=None):
    """Find a file by name. Returns file ID or None."""
    query = f"name='{name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    files = results.get("files", [])
    return files[0]["id"] if files else None


def download_file(service, file_id, local_path: str | Path):
    """Download a file from Drive to local path."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    request = service.files().get_media(fileId=file_id)
    with open(local_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%")

    print(f"Downloaded: {local_path}")


# ---------------------------------------------------------------------------
# Files: upload
# ---------------------------------------------------------------------------
def upload_file(service, local_path: str | Path, parent_id=None, mime_type=None):
    """Upload a local file to Drive."""
    local_path = Path(local_path)

    if mime_type is None:
        # Guess mime type from extension
        ext = local_path.suffix.lower()
        mime_map = {
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".csv": "text/csv",
            ".parquet": "application/octet-stream",
            ".pkl": "application/octet-stream",
            ".json": "application/json",
        }
        mime_type = mime_map.get(ext, "application/octet-stream")

    metadata = {
        "name": local_path.name,
        "parents": [parent_id] if parent_id else [],
    }

    media = MediaFileUpload(str(local_path), mimetype=mime_type, resumable=True)
    file = service.files().create(body=metadata, media_body=media, fields="id").execute()

    print(f"Uploaded: {local_path.name} → ID {file['id']}")
    return file["id"]


# ---------------------------------------------------------------------------
# Convenience: full pipeline helpers
# ---------------------------------------------------------------------------
def download_from_drive(drive_path: str, local_path: str | Path, parent_id=None):
    """High-level: download by Drive path."""
    service = get_drive_service()
    file_id = find_file(service, drive_path, parent_id)
    if not file_id:
        raise FileNotFoundError(f"File not found in Drive: {drive_path}")
    download_file(service, file_id, local_path)


def upload_to_drive(local_path: str | Path, drive_folder_id=None):
    """High-level: upload to Drive folder."""
    service = get_drive_service()
    return upload_file(service, local_path, parent_id=drive_folder_id)
