# src/dashboard/server.py
"""
Milan Capital — Dashboard Server
- Local mode  : boutons → subprocess local + SSE streaming
- Cloud mode  : boutons → GitHub Actions API (GITHUB_TOKEN requis)

Usage:
    make dashboard           # local
    railway up               # cloud (Railway)
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import uuid
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import requests as _requests
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, Response
import uvicorn

from src.dashboard import auth as auth_mod

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent.parent
LOGS = ROOT / "logs"
HTML = pathlib.Path(__file__).parent / "index.html"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_OWNER = os.getenv("GITHUB_OWNER", "")
GITHUB_REPO  = os.getenv("GITHUB_REPO", "")
GITHUB_REF   = os.getenv("GITHUB_REF", "main")
IS_CLOUD     = bool(GITHUB_TOKEN and GITHUB_OWNER and GITHUB_REPO)

PORT = int(os.getenv("PORT", "8000"))

# ── Helpers ───────────────────────────────────────────────────────────────────
def _df_json(df: pd.DataFrame) -> JSONResponse:
    return JSONResponse(content=json.loads(df.to_json(orient="records")))


# ── Auth ──────────────────────────────────────────────────────────────────────
SESSION_COOKIE = "milan_session"


def _current_user(request: Request) -> str | None:
    return auth_mod.verify_session_token(request.cookies.get(SESSION_COOKIE))


def require_auth(request: Request) -> str:
    user = _current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Non authentifié")
    return user


# ── Job registry (local mode only) ───────────────────────────────────────────
JOBS: Dict[str, Dict[str, Any]] = {}
LOG_QUEUES: Dict[str, asyncio.Queue] = {}

LOCAL_COMMANDS = {
    "run":         [sys.executable, "-m", "src.arena.runner"],
    "shadow":      [sys.executable, "-m", "src.backtest.shadow_mode"],
    "backtest":    [sys.executable, "-m", "src.backtest.portfolio_backtest"],
    "walkforward": [sys.executable, "-m", "src.backtest.run_walkforward"],
}

async def _execute_local(job_id: str, command: str) -> None:
    queue = LOG_QUEUES[job_id]
    cmd = LOCAL_COMMANDS.get(command)
    if not cmd:
        await queue.put(f"❌ Commande inconnue: {command}")
        await queue.put(None)
        return
    JOBS[job_id]["started_at"] = datetime.now().isoformat()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
        )
        JOBS[job_id]["pid"] = proc.pid
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            JOBS[job_id].setdefault("logs", []).append(line)
            await queue.put(line)
        await proc.wait()
        JOBS[job_id]["status"]     = "done" if proc.returncode == 0 else "error"
        JOBS[job_id]["returncode"] = proc.returncode
    except Exception as exc:
        await queue.put(f"❌ Erreur: {exc}")
        JOBS[job_id]["status"] = "error"
    finally:
        await queue.put(None)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Milan Capital", docs_url=None, redoc_url=None)


@app.get("/", response_class=HTMLResponse)
def root():
    return HTML.read_text(encoding="utf-8")


# ── PWA assets ───────────────────────────────────────────────────────────────
@app.get("/manifest.json")
def manifest():
    return JSONResponse(content={
        "name": "Milan Capital",
        "short_name": "MilCap",
        "description": "Multi-Agent Quantitative Fund — by Nathan Floiras",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#05080f",
        "theme_color": "#DEAA3D",
        "orientation": "portrait-primary",
        "icons": [
            {"src": "/icon-192.png?v=3", "sizes": "192x192", "type": "image/png", "purpose": "any"},
            {"src": "/icon-512.png?v=3", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
        ]
    })


ICONS = pathlib.Path(__file__).parent / "icons"

@app.get("/icon-{size}.png")
def icon_png(size: int):
    path = ICONS / f"icon-{size}.png"
    if not path.exists():
        return Response(status_code=404)
    return Response(content=path.read_bytes(), media_type="image/png")


@app.get("/sw.js")
def service_worker():
    js = """
const CACHE = 'milan-v1';
const ASSETS = ['/'];

self.addEventListener('install', e =>
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)))
);

self.addEventListener('fetch', e => {
  if (e.request.url.includes('/api/')) return;
  e.respondWith(
    fetch(e.request)
      .then(r => { caches.open(CACHE).then(c => c.put(e.request, r.clone())); return r; })
      .catch(() => caches.match(e.request))
  );
});
"""
    return Response(content=js, media_type="application/javascript")


# ── Auth routes ───────────────────────────────────────────────────────────────
@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))

    if not auth_mod.verify_login(username, password):
        return JSONResponse({"error": "Identifiant ou mot de passe incorrect."}, status_code=401)

    token = auth_mod.create_session_token(username)
    resp = JSONResponse({"ok": True, "username": username})
    resp.set_cookie(
        SESSION_COOKIE, token,
        # Pas de max_age : cookie de session pur — survit à la mise en arrière-plan
        # (process iOS suspendu) mais disparaît si l'app est fermée (swipe-up).
        httponly=True, samesite="lax", secure=IS_CLOUD, path="/",
    )
    return resp


@app.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/api/session")
def session_info(request: Request):
    user = _current_user(request)
    return {"authenticated": bool(user), "username": user}


# ── Mode ──────────────────────────────────────────────────────────────────────
@app.get("/api/mode")
def get_mode():
    return {
        "cloud": IS_CLOUD,
        "github_owner": GITHUB_OWNER,
        "github_repo": GITHUB_REPO,
    }


# ── Local: run subprocess ─────────────────────────────────────────────────────
@app.post("/api/run/{command}")
async def run_local(command: str, user: str = Depends(require_auth)):
    job_id = str(uuid.uuid4())[:8]
    queue: asyncio.Queue = asyncio.Queue()
    JOBS[job_id] = {"command": command, "status": "running"}
    LOG_QUEUES[job_id] = queue
    asyncio.create_task(_execute_local(job_id, command))
    return {"job_id": job_id, "status": "started"}


@app.get("/api/stream/{job_id}")
async def stream_logs(job_id: str, user: str = Depends(require_auth)):
    async def generator():
        queue = LOG_QUEUES.get(job_id)
        if not queue:
            yield "data: ❌ Job inconnu\n\n"
            return
        while True:
            try:
                line = await asyncio.wait_for(queue.get(), timeout=25.0)
            except asyncio.TimeoutError:
                yield "data: \n\n"
                continue
            if line is None:
                yield "data: __DONE__\n\n"
                break
            yield f"data: {line.replace(chr(10), ' ')}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Cloud: trigger GitHub Actions ─────────────────────────────────────────────
@app.post("/api/trigger/{command}")
def trigger_github(command: str, user: str = Depends(require_auth)):
    if not IS_CLOUD:
        return JSONResponse({"error": "GitHub env vars non configurés"}, status_code=400)

    url = (
        f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
        f"/actions/workflows/manual_trigger.yml/dispatches"
    )
    resp = _requests.post(
        url,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": GITHUB_REF, "inputs": {"command": command}},
        timeout=10,
    )
    if resp.status_code == 204:
        actions_url = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/actions"
        return {"status": "triggered", "command": command, "actions_url": actions_url}
    return JSONResponse({"error": resp.text, "code": resp.status_code}, status_code=502)


# ── Data routes ───────────────────────────────────────────────────────────────
@app.get("/api/status")
def get_status(user: str = Depends(require_auth)):
    cb: dict = {}
    cb_path = LOGS / "circuit_breaker.json"
    if cb_path.exists():
        cb = json.loads(cb_path.read_text())
    running = [jid for jid, j in JOBS.items() if j.get("status") == "running"]
    return {"circuit_breaker": cb, "running_jobs": running, "cloud": IS_CLOUD}


@app.get("/api/signals")
def get_signals(user: str = Depends(require_auth)):
    path = LOGS / "decisions.csv"
    if not path.exists():
        return JSONResponse(content=[])
    try:
        df = pd.read_csv(path)
        cols = [c for c in ["ts","symbol","regime","winner_agent","action","confidence","reason"]
                if c in df.columns]
        df = df[cols].drop_duplicates(subset=["symbol"], keep="last").sort_values("symbol")
        return _df_json(df)
    except Exception:
        return JSONResponse(content=[])


@app.get("/api/performance")
def get_performance(user: str = Depends(require_auth)):
    path = LOGS / "portfolio_by_symbol.csv"
    if not path.exists():
        return JSONResponse(content=[])
    try:
        return _df_json(pd.read_csv(path).sort_values("ret", ascending=False))
    except Exception:
        return JSONResponse(content=[])


@app.get("/api/equity")
def get_equity(user: str = Depends(require_auth)):
    path = LOGS / "portfolio_equity.csv"
    if not path.exists():
        return JSONResponse(content=[])
    try:
        df = pd.read_csv(path)
        step = max(1, len(df) // 120)
        return _df_json(df.iloc[::step])
    except Exception:
        return JSONResponse(content=[])


@app.get("/api/agents")
def get_agents(user: str = Depends(require_auth)):
    path = LOGS / "walkforward_summary.csv"
    if not path.exists():
        return JSONResponse(content=[])
    try:
        df = (pd.read_csv(path)
              .sort_values("avg_oos_sharpe", ascending=False)
              .drop_duplicates(subset=["symbol"], keep="first"))
        return _df_json(df)
    except Exception:
        return JSONResponse(content=[])


if __name__ == "__main__":
    print(f"🚀  Milan Capital Dashboard → http://localhost:{PORT}")
    print(f"   Mode: {'☁️  Cloud (GitHub Actions)' if IS_CLOUD else '💻 Local (subprocess)'}")
    uvicorn.run("src.dashboard.server:app", host="0.0.0.0", port=PORT, reload=not IS_CLOUD)
