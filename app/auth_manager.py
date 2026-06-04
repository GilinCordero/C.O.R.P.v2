"""Simple SHA-256 auth manager."""
import hashlib

# Demo user — replace with proper DB in production
_USERS = {
    "gcc_corp_user": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # "password"
}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_credentials(username: str, password: str) -> bool:
    if username not in _USERS:
        return False
    return _USERS[username] == hash_password(password)
