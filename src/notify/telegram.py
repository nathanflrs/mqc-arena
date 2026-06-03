# src/notify/telegram.py
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TgUpdate:
    update_id: int
    chat_id: int
    text: str


def _token() -> str:
    t = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not t:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN env var")
    return t


def _chat_id() -> int:
    s = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not s:
        raise RuntimeError("Missing TELEGRAM_CHAT_ID env var")
    return int(s)


def send_message(text: str) -> None:
    url = f"https://api.telegram.org/bot{_token()}/sendMessage"
    payload = {"chat_id": _chat_id(), "text": text}
    r = requests.post(url, json=payload, timeout=15)
    r.raise_for_status()


def get_latest_update(after_update_id: Optional[int] = None) -> Optional[TgUpdate]:
    url = f"https://api.telegram.org/bot{_token()}/getUpdates"
    params = {"timeout": 0}
    if after_update_id is not None:
        params["offset"] = after_update_id + 1

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    results = data.get("result", [])
    if not results:
        return None

    last = results[-1]
    msg = last.get("message") or last.get("edited_message")
    if not msg:
        return None

    chat = msg.get("chat", {})
    text = (msg.get("text") or "").strip()
    return TgUpdate(
        update_id=int(last["update_id"]),
        chat_id=int(chat.get("id")),
        text=text,
    )


def wait_for_approval(
    *,
    plan_id: str,
    timeout_seconds: int = 900,
    poll_seconds: int = 3,
    last_update_id: Optional[int] = None,
) -> tuple[bool, Optional[int]]:
    deadline = time.time() + timeout_seconds
    target_chat = _chat_id()

    while time.time() < deadline:
        upd = get_latest_update(after_update_id=last_update_id)
        if upd is None:
            time.sleep(poll_seconds)
            continue

        last_update_id = upd.update_id

        if upd.chat_id != target_chat:
            continue

        txt = upd.text.upper()
        if txt == "APPROVE":
            return True, last_update_id
        if txt == "REJECT":
            return False, last_update_id

        time.sleep(poll_seconds)

    return False, last_update_id


def drain_updates() -> int | None:
    last_id = None
    while True:
        upd = get_latest_update(after_update_id=last_id)
        if upd is None:
            return last_id
        last_id = upd.update_id