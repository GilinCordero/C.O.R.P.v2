"""Google Drive API client.

Supports two auth methods (in priority order):
1. OAuth 2.0 refresh token (GitHub Actions / CI / personal Drive)
2. Service Account JSON (legacy / Workspace)
"""
import io
import json
import os
import tempfile
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2 import credentials as oauth_credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = Path(__file__).resolve().parent / "gccc-498819-28eb8af3eb04.json"


def _get_oauth_credentials():
    """Build credentials from OAuth refresh token (CI mode)."""
    client_id = os.environ.get("GCP_CLIENT_ID")
    client_secret = os.environ.get("GCP_CLIENT_SECRET")
    refresh_token = os.environ.get("GCP_REFRESH_TOKEN")

    if not all([client_id, client_secret, refresh_token]):
        return None

    creds = oauth_credentials.Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    creds.refresh(Request())
    return creds


def _get_service_account_credentials():
    """Build credentials from service account JSON."""
    sa_key_env = os.environ.get("GCP_SERVICE_ACCOUNT_KEY")

    if sa_key_env:
        key_info = json.loads(sa_key_env)
        return service_account.Credentials.from_service_account_info(
            key_info,
            scopes=SCOPES,
        )

    if SERVICE_ACCOUNT_FILE.exists():
        return service_account.Credentials.from_service_account_file(
            str(SERVICE_ACCOUNT_FILE),
            scopes=SCOPES,
        )

    return None


def get_drive_service():
    """Authenticate and return Google Drive API service."""
    # 1. Try OAuth (preferred for personal Drive)
    credentials = _get_oauth_credentials()
    if credentials:
        print("[Drive] Using OAuth 2.0 (refresh token)")
        return build("drive", "v3", credentials=credentials)

    # 2. Fallback to Service Account
    credentials = _get_service_account_credentials()
    if credentials:
        print("[Drive] Using Service Account")
        return build("drive", "v3", credentials=credentials)

    raise FileNotFoundError(
        "No Google Drive credentials found. "
        "Set either (GCP_CLIENT_ID + GCP_CLIENT_SECRET + GCP_REFRESH_TOKEN) "
        "or GCP_SERVICE_ACCOUNT_KEY."
    )


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
