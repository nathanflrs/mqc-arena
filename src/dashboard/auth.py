"""
Milan Capital — Authentification
Comptes stockés soit dans src/dashboard/users.json (local, jamais commité),
soit dans la variable d'env MILAN_USERS_JSON (Railway) — même format :
  [{"username": "nathan", "password_hash": "$2b$..."}]
"""
from __future__ import annotations

import json
import os
import pathlib

import bcrypt
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

USERS_PATH = pathlib.Path(__file__).parent / "users.json"
USERS_ENV_VAR = "MILAN_USERS_JSON"

SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-insecure-secret-change-me")
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 jours

_serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="milan-session")


def _load_users() -> dict[str, str]:
    raw = os.getenv(USERS_ENV_VAR)
    if raw:
        data = json.loads(raw)
    elif USERS_PATH.exists():
        data = json.loads(USERS_PATH.read_text())
    else:
        data = []
    return {u["username"]: u["password_hash"] for u in data}


def verify_login(username: str, password: str) -> bool:
    hashed = _load_users().get(username)
    if not hashed:
        return False
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_session_token(username: str) -> str:
    return _serializer.dumps({"u": username})


def verify_session_token(token: str | None) -> str | None:
    if not token:
        return None
    try:
        data = _serializer.loads(token, max_age=SESSION_MAX_AGE)
        return data.get("u")
    except (BadSignature, SignatureExpired):
        return None
