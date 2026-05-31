"""
JWT auth helpers — password hashing, token creation/verification.
"""
import os
from datetime import datetime, timedelta

import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from src.database import get_user_by_id

# ── Config ────────────────────────────────────────────────────────────────────
def _get_secret():
    try:
        from src.config.settings import get_settings
        return get_settings().JWT_SECRET
    except Exception:
        return os.getenv("JWT_SECRET", "neuroresearch-change-this-in-production")

SECRET_KEY  = None  # resolved lazily
ALGORITHM   = "HS256"
TOKEN_EXPIRE_DAYS = 30

bearer = HTTPBearer(auto_error=False)


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    return jwt.encode({"sub": str(user_id), "exp": expire}, _get_secret(), algorithm=ALGORITHM)


def decode_token(token: str) -> int:
    try:
        payload = jwt.decode(token, _get_secret(), algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if not creds:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    user_id = decode_token(creds.credentials)
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return dict(user)
