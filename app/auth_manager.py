"""Simple SHA-256 auth manager."""
import hashlib

# Demo user — replace with proper DB in production
_USERS = {
    "gcc_corp_user": "b9f6ca8ed76ac3a4f5fac6132e825e059bb3004b0234a2e4eb20943bc64171c5",  # "password"
}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_credentials(username: str, password: str) -> bool:
    if username not in _USERS:
        return False
    return _USERS[username] == hash_password(password)
