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
import base64
import io
import json
import os
import pathlib
import sys
import uuid
from datetime import datetime
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import logging
import pandas as pd
import requests as _requests

logger = logging.getLogger(__name__)
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


def _github_file(rel_path: str) -> str | None:
    """Lit un fichier depuis le repo GitHub (contenu commité par les Actions)."""
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{rel_path}"
    resp = _requests.get(
        url,
        headers={
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        },
        params={"ref": GITHUB_REF},
        timeout=10,
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    if data.get("encoding") != "base64":
        return None
    return base64.b64decode(data["content"]).decode("utf-8")


def _read_text(rel_path: str) -> str | None:
    """rel_path relatif à la racine du repo, ex: 'logs/decisions.csv'."""
    if IS_CLOUD:
        return _github_file(rel_path)
    path = ROOT / rel_path
    return path.read_text(encoding="utf-8") if path.exists() else None


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

# ── Brute-force protection for /api/login ────────────────────────────────────
import time as _time
_LOGIN_ATTEMPTS: Dict[str, list] = {}   # ip → [timestamp, ...]
_MAX_ATTEMPTS = 5
_LOCKOUT_SECONDS = 300   # 5 minutes


def _check_login_rate(ip: str) -> bool:
    """Returns True if allowed, False if locked out."""
    now = _time.monotonic()
    attempts = [t for t in _LOGIN_ATTEMPTS.get(ip, []) if now - t < _LOCKOUT_SECONDS]
    _LOGIN_ATTEMPTS[ip] = attempts
    return len(attempts) < _MAX_ATTEMPTS


def _record_failed_login(ip: str) -> None:
    _LOGIN_ATTEMPTS.setdefault(ip, []).append(_time.monotonic())

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
        # Un run déclenché depuis le dashboard est une décision discrétionnaire :
        # quelqu'un a choisi ce moment-là. Sans étiquette, impossible de le
        # distinguer plus tard d'une séance produite par le planificateur, et le
        # track record devient inanalysable. Le verrou lit cette variable et la
        # conserve. Voir src/execution/run_lock.py.
        env = {**os.environ, "RUN_TRIGGER": "manual"}
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
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
    ip = request.client.host if request.client else "unknown"
    if not _check_login_rate(ip):
        return JSONResponse(
            {"error": "Trop de tentatives. Réessayez dans 5 minutes."},
            status_code=429,
        )

    body = await request.json()
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))

    if not auth_mod.verify_login(username, password):
        _record_failed_login(ip)
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
def get_mode(user: str = Depends(require_auth)):
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
_VALID_TRIGGER_COMMANDS = frozenset(LOCAL_COMMANDS.keys())  # {"run","shadow","backtest","walkforward"}


@app.post("/api/trigger/{command}")
def trigger_github(command: str, user: str = Depends(require_auth)):
    if command not in _VALID_TRIGGER_COMMANDS:
        raise HTTPException(status_code=400, detail=f"Unknown command: {command}")

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


# ── Risk Assistant IA ─────────────────────────────────────────────────────────
@app.post("/api/ask")
async def ask_assistant(request: Request, user: str = Depends(require_auth)):
    import asyncio
    body = await request.json()
    question = str(body.get("question", "")).strip()
    if not question:
        raise HTTPException(status_code=400, detail="question required")
    try:
        from src.risk.assistant import RiskAssistant, build_context
        ctx = build_context(LOGS)
        # L'appel Anthropic est synchrone/bloquant — on l'isole dans un thread
        # pour ne pas bloquer l'event loop FastAPI (sinon httpx ne peut pas
        # compléter son propre I/O → Connection error).
        answer = await asyncio.to_thread(RiskAssistant().ask, question, ctx)
        return {"answer": answer}
    except Exception as exc:
        logger.error("ask_assistant: %s", exc, exc_info=True)
        return JSONResponse({"answer": f"Erreur serveur : {exc}"}, status_code=500)


# ── Data routes ───────────────────────────────────────────────────────────────
# Au-delà de ce délai, les données affichées ne décrivent plus le marché
# d'aujourd'hui. Deux séances de bourse de marge : un week-end ou un jour férié
# ne doit pas déclencher l'alerte, trois jours de silence si.
FRESH_MAX_HOURS = 72.0


def _last_run_info() -> dict:
    """
    Âge réel des données affichées.

    Le badge « LIVE » du bandeau était une constante écrite en dur dans le HTML :
    il s'affichait vert quoi qu'il arrive. Le 13/08/2026, le dashboard annonçait
    LIVE avec des décisions du 23/07 — trois semaines de retard, invisibles.
    C'est le pire mode de défaillance d'un écran de pilotage : il n'affiche pas
    une erreur, il affiche une santé.
    """
    raw = _read_text("logs/decisions.csv")
    if not raw:
        return {"timestamp": None, "age_hours": None, "is_fresh": False}
    try:
        df = pd.read_csv(io.StringIO(raw))
        ts = pd.to_datetime(df["timestamp"], format="mixed", utc=True).max()
        age = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0
        return {
            "timestamp": ts.isoformat(),
            "age_hours": round(age, 1),
            "is_fresh": bool(age <= FRESH_MAX_HOURS),
        }
    except Exception:
        return {"timestamp": None, "age_hours": None, "is_fresh": False}


@app.get("/api/status")
def get_status(user: str = Depends(require_auth)):
    cb: dict = {}
    raw = _read_text("logs/circuit_breaker.json")
    if raw:
        cb = json.loads(raw)
    running = [jid for jid, j in JOBS.items() if j.get("status") == "running"]
    # Run en cours, y compris déclenché hors du dashboard (planificateur, ligne
    # de commande). `running_jobs` ne connaît que les travaux lancés par ce
    # processus : sur le serveur, le run planifié lui est invisible.
    try:
        from src.execution.run_lock import current_holder
        holder = current_holder()
        lock = ({"pid": holder.pid, "started_at": holder.started_at,
                 "trigger": holder.trigger} if holder else None)
    except Exception:
        lock = None

    return {
        "circuit_breaker": cb,
        "running_jobs": running,
        "cloud": IS_CLOUD,
        "last_run": _last_run_info(),
        "fresh_max_hours": FRESH_MAX_HOURS,
        "active_run": lock,
    }


def _data() -> "DashboardData":
    from src.dashboard.data import DashboardData
    return DashboardData(_read_text)


@app.get("/api/overview")
def get_overview(user: str = Depends(require_auth)):
    """
    Tout ce que la vue d'ensemble affiche, chaque chiffre avec sa provenance.

    Un seul appel plutôt que six, et surtout une seule règle : `as_of`,
    `source` et `kind` accompagnent systématiquement la valeur, si bien qu'une
    carte ne peut plus afficher un backtest de 2022 ou une décision de juin
    sans que l'écran ait de quoi le dire.
    """
    d = _data()
    return {
        "nav":           d.nav().to_dict(),
        "total_return":  d.total_return().to_dict(),
        "equity_curve":  d.equity_curve().to_dict(),
        "pnl":           d.pnl().to_dict(),
        "decisions":     d.latest_decisions().to_dict(),
        "strategies":    d.strategies().to_dict(),
        "circuit_breaker": d.circuit_breaker().to_dict(),
        "last_run":      d.last_run().to_dict(),
    }


@app.get("/api/signals")
def get_signals(user: str = Depends(require_auth)):
    """
    Décisions du dernier run **seulement**.

    L'implémentation précédente gardait le dernier gagnant de chaque symbole sur
    tout l'historique : BRK-B et JNJ, retirés de l'univers en juin, figuraient
    encore parmi les « signaux live » du 13 août, portant le compte à 16 au lieu
    de 12. Le filtrage se fait désormais sur le plan_id du dernier run.

    Le format de sortie est conservé pour ne pas casser l'écran existant.
    """
    fig = _data().latest_decisions()
    rows = [
        {
            "ts":           fig.as_of,
            "symbol":       r["symbol"],
            "regime":       r["regime"],
            "winner_agent": r["agent"],
            "action":       r["action"],
            "confidence":   r["confidence"],
            "reason":       r["reason"],
        }
        for r in (fig.value or [])
    ]
    return JSONResponse(content=rows)


@app.get("/api/equity")
def get_equity_live(user: str = Depends(require_auth)):
    """
    Capital jour après jour, **du compte réel**.

    L'ancienne version servait `logs/portfolio_equity.csv` — un backtest
    2022→juin 2026 sur base 100 000 $ — sous le titre « Equity curve vs SPY ».
    La courbe du vrai compte commence au premier run du serveur et se remplit
    une séance à la fois ; un seul point au départ est la réponse honnête.
    """
    return _data().equity_curve().to_dict()


@app.get("/api/backtest-equity")
def get_backtest_equity(user: str = Depends(require_auth)):
    """Courbe du backtest, servie séparément et jamais confondue avec le réel."""
    raw = _read_text("logs/portfolio_equity.csv")
    if not raw:
        return JSONResponse(content={"kind": "unavailable", "value": None})
    try:
        df = pd.read_csv(io.StringIO(raw))
        step = max(1, len(df) // 120)
        return JSONResponse(content={
            "kind": "simulated",
            "source": "logs/portfolio_equity.csv",
            "note": "backtest — ni le capital réel, ni la période en cours",
            "value": json.loads(df.iloc[::step].to_json(orient="records")),
        })
    except Exception:
        return JSONResponse(content={"kind": "unavailable", "value": None})


@app.get("/api/performance")
def get_performance(user: str = Depends(require_auth)):
    """
    Classement des agents, **avec l'origine des chiffres**.

    L'endpoint renvoyait une liste nue, sans dire si elle venait d'un backtest
    ou de trades réels. L'écran affichait donc « Buffett +196.28 % » sous une
    pastille LIVE, alors que le chiffre sort de `portfolio_by_symbol.csv`,
    un résultat de simulation — le commentaire du code le disait déjà, mais
    l'information s'arrêtait là et n'atteignait jamais l'utilisateur.

    Un rendement simulé et un rendement réalisé ne se lisent pas de la même
    façon : le premier est une borne haute obtenue en connaissant la période,
    le second est ce qui est arrivé. Les afficher sans distinction est la
    manière la plus simple de se mentir à soi-même — et de perdre toute
    crédibilité le jour où quelqu'un demande d'où sort le chiffre.

    La réponse porte désormais `source` : "backtest" ou "live".
    """
    raw = _read_text("logs/portfolio_by_symbol.csv")
    if raw:
        try:
            df = pd.read_csv(io.StringIO(raw)).sort_values("ret", ascending=False)
            return JSONResponse(content={
                "source": "backtest",
                "origin": "logs/portfolio_by_symbol.csv",
                "rows": json.loads(df.to_json(orient="records")),
            })
        except Exception:
            pass

    try:
        from src.risk.live_scorer import LiveScorer
        metrics = LiveScorer().compute_agent_metrics()
        if not metrics:
            return JSONResponse(content={"source": "live", "origin": "", "rows": []})
        rows = [
            {
                "sym":    "ALL",
                "agent":  m.agent,
                "ret":    round(m.total_pnl_pct, 4),
                "sharpe": round(m.sharpe, 4),
                "trades": m.n_trades,
            }
            for m in sorted(metrics.values(), key=lambda x: x.sharpe, reverse=True)
        ]
        return JSONResponse(content={
            "source": "live",
            "origin": "logs/executions.csv (round-trips réels)",
            "rows": rows,
        })
    except Exception:
        return JSONResponse(content={"source": "live", "origin": "", "rows": []})


@app.get("/api/portfolio-summary")
def get_portfolio_summary(user: str = Depends(require_auth)):
    try:
        from src.risk.live_scorer import LiveScorer
        perf = LiveScorer().compute_portfolio_performance()
        if perf is None:
            return JSONResponse(content={"available": False})
        return JSONResponse(content={"available": True, **perf.to_dict(),
                                     "equity_curve": perf.equity_curve})
    except Exception as e:
        return JSONResponse(content={"available": False, "error": str(e)})


@app.get("/api/drift-alerts")
def get_drift_alerts(user: str = Depends(require_auth)):
    try:
        from src.risk.live_scorer import LiveScorer
        alerts = LiveScorer().compute_drift_alerts()
        return JSONResponse(content=[a.to_dict() for a in alerts])
    except Exception:
        return JSONResponse(content=[])


@app.get("/api/regime-accuracy")
def get_regime_accuracy(user: str = Depends(require_auth)):
    try:
        from src.risk.live_scorer import LiveScorer
        stats = LiveScorer().compute_regime_accuracy()
        return JSONResponse(content=[s.to_dict() for s in stats])
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/fill-stats")
def get_fill_stats(user: str = Depends(require_auth)):
    try:
        from src.risk.live_scorer import LiveScorer
        stats = LiveScorer().compute_fill_stats()
        if stats is None:
            return JSONResponse(content={"available": False})
        return JSONResponse(content={"available": True, **stats.to_dict()})
    except Exception as e:
        return JSONResponse(content={"available": False, "error": str(e)})


@app.post("/api/reset-circuit-breaker")
def reset_circuit_breaker(user: str = Depends(require_auth)):
    try:
        from src.risk.manager import DrawdownCircuitBreaker
        DrawdownCircuitBreaker().reset()
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# L'ancien /api/equity servait logs/portfolio_equity.csv — un backtest — sous
# le titre « Equity curve vs SPY ». Il est remplacé plus haut par la courbe du
# compte réel, et le backtest est exposé séparément via /api/backtest-equity.
# FastAPI retient la PREMIÈRE route déclarée : laisser ce doublon en place
# n'avait pas d'effet visible, mais la moindre réorganisation du fichier aurait
# silencieusement remis le backtest à l'écran.


@app.get("/api/agents")
def get_agents(user: str = Depends(require_auth)):
    raw = _read_text("logs/walkforward_summary.csv")
    if not raw:
        return JSONResponse(content=[])
    try:
        df = (pd.read_csv(io.StringIO(raw))
              .sort_values("avg_oos_sharpe", ascending=False)
              .drop_duplicates(subset=["symbol"], keep="first"))
        return _df_json(df)
    except Exception:
        return JSONResponse(content=[])


@app.get("/api/monte-carlo")
def get_monte_carlo(user: str = Depends(require_auth)):
    """Returns the latest Monte Carlo simulation result from logs/monte_carlo_latest.json."""
    raw = _read_text("logs/monte_carlo_latest.json")
    if not raw:
        return JSONResponse(content={"available": False})
    try:
        data = json.loads(raw)
        return JSONResponse(content={"available": True, **data})
    except Exception as exc:
        return JSONResponse(content={"available": False, "error": str(exc)})


@app.get("/api/events")
def get_events(
    limit: int = 100,
    type: str | None = None,
    user: str = Depends(require_auth),
):
    """Return recent events from the event bus, newest first."""
    from src.events.bus import get_bus
    events = get_bus().get_recent(limit=max(1, min(limit, 500)), type_filter=type)
    return JSONResponse(content={"events": events})


@app.get("/api/events/stream")
async def stream_events(user: str = Depends(require_auth)):
    """SSE stream of live events. One JSON object per `data:` line."""
    import threading
    from src.events.bus import get_bus

    async def generator():
        bus  = get_bus()
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        stop = threading.Event()

        def safe_put(payload: str) -> None:
            # Called inside the event loop via call_soon_threadsafe.
            # If the asyncio queue is full the client has disconnected —
            # swallow the error and signal the feed thread to exit.
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                stop.set()

        def feed() -> None:
            # Bridge blocking bus generator → asyncio queue.
            # Explicitly close the generator so bus._unregister() always runs.
            gen = bus.subscribe_sse()
            try:
                for payload in gen:
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(safe_put, payload)
            finally:
                gen.close()

        t = threading.Thread(target=feed, daemon=True)
        t.start()

        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if payload == "__HEARTBEAT__":
                    yield ": heartbeat\n\n"
                else:
                    yield f"data: {payload}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            # Signal feed thread to stop on the next bus timeout (≤ 25 s).
            stop.set()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/events/{event_id}/acknowledge")
def acknowledge_event(event_id: str, user: str = Depends(require_auth)):
    from src.events.bus import get_bus
    get_bus().acknowledge(event_id)
    return JSONResponse(content={"ok": True})


@app.post("/api/monte-carlo/run")
def run_monte_carlo(request: Request, user: str = Depends(require_auth)):
    """Triggers a new Monte Carlo simulation (N=10,000, horizon=90j) and saves results."""
    if IS_CLOUD:
        return JSONResponse(
            {"error": "Monte Carlo run non disponible en mode cloud — utilisez le script local"},
            status_code=400,
        )
    try:
        from src.analytics.monte_carlo import run_simulation, MonteCarloReporter
        result = run_simulation(
            n_simulations=10_000,
            horizon_days=90,
            save_path=str(ROOT / "logs" / "monte_carlo_latest.json"),
        )
        reporter = MonteCarloReporter()
        return JSONResponse(content={
            "ok": True,
            "summary": reporter.format_tearsheet_section(result),
            "var_95": result.var_95,
            "median_return": result.median_return,
            "sharpe_ratio": result.sharpe_ratio,
            "prob_circuit_breaker": result.prob_circuit_breaker,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/cta-signals")
def get_cta_signals(user: str = Depends(require_auth)):
    """CTA Trend signals for the 6-ETF universe. Cached 30 min."""
    import time as _time

    CACHE_PATH = LOGS / "cta_signals_cache.json"
    CACHE_TTL  = 1800  # 30 min

    # Return fresh cache
    if CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text())
            if _time.time() - cached.pop("_ts", 0) < CACHE_TTL:
                return JSONResponse(content=cached)
        except Exception:
            pass

    try:
        from src.agents.cta_trend_agent import CTATrendAgent, CTA_UNIVERSE
        from src.agents.base import MarketState
        from src.data.market_data import download_ohlcv
        from datetime import timezone

        agent = CTATrendAgent()

        # Best-effort portfolio from executions log
        portfolio: dict[str, float] = {}
        try:
            raw = _read_text("logs/executions.csv")
            if raw:
                df_exec = pd.read_csv(io.StringIO(raw))
                for _, row in df_exec.iterrows():
                    sym = str(row.get("symbol", ""))
                    if sym in CTA_UNIVERSE:
                        dq = float(row.get("delta_qty", 0) or 0)
                        portfolio[sym] = portfolio.get(sym, 0.0) + dq
        except Exception:
            pass

        signals = []
        for sym in CTA_UNIVERSE:
            try:
                df = download_ohlcv(sym, period="1y")
                state = MarketState(
                    symbol=sym,
                    price=float(df["Close"].iloc[-1]),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                sig  = agent.generate_signal(state, portfolio, data=df)
                meta = sig.meta or {}
                cta_dir     = meta.get("cta_direction", "flat")
                current_qty = float(meta.get("current_qty", portfolio.get(sym, 0.0)))

                if cta_dir == "hold":
                    if current_qty > 0:
                        display_dir = "hold_long"
                    elif current_qty < 0:
                        display_dir = "hold_short"
                    else:
                        display_dir = "flat"
                else:
                    display_dir = cta_dir

                signals.append({
                    "symbol":        sym,
                    "direction":     cta_dir,
                    "display_dir":   display_dir,
                    "mom_fast":      round(float(meta.get("mom_fast",  0.0)), 4),
                    "mom_slow":      round(float(meta.get("mom_slow",  0.0)), 4),
                    "adx":           round(float(meta.get("adx",       0.0)), 1),
                    "realized_vol":  round(float(meta.get("realized_vol", 0.0)), 4),
                    "target_weight": round(float(sig.target_weight), 4),
                    "current_qty":   current_qty,
                })
            except Exception as exc:
                signals.append({
                    "symbol": sym, "direction": "flat", "display_dir": "flat",
                    "mom_fast": 0.0, "mom_slow": 0.0, "adx": 0.0,
                    "realized_vol": 0.0, "target_weight": 0.0, "current_qty": 0.0,
                    "error": str(exc),
                })

        n_long  = sum(1 for s in signals if s["direction"] == "long")
        n_short = sum(1 for s in signals if s["direction"] == "short")
        n_flat  = len(signals) - n_long - n_short
        gross   = round(sum(abs(s["target_weight"]) for s in signals), 4)
        net     = round(sum(s["target_weight"]      for s in signals), 4)

        result = {
            "signals":        signals,
            "gross_exposure": gross,
            "net_exposure":   net,
            "gross_cap":      0.60,
            "n_long":         n_long,
            "n_short":        n_short,
            "n_flat":         n_flat,
            "computed_at":    datetime.now(timezone.utc).isoformat(),
        }
        try:
            CACHE_PATH.write_text(json.dumps({**result, "_ts": _time.time()}))
        except Exception:
            pass
        return JSONResponse(content=result)

    except Exception as exc:
        return JSONResponse(content={
            "error": str(exc), "signals": [],
            "gross_exposure": 0, "net_exposure": 0, "gross_cap": 0.6,
            "n_long": 0, "n_short": 0, "n_flat": 6,
        })


# La collecte interroge une vingtaine de tickers en série sur le réseau. Sans
# borne, la requête du navigateur restait en attente indéfiniment : le bandeau
# « Chargement des actualités… » de la page d'accueil ne se résolvait jamais.
# Un écran de pilotage doit toujours rendre la main — quitte à dire qu'il n'a
# pas l'information.
NEWS_FETCH_TIMEOUT_SECONDS = 12.0


@app.get("/api/news-today")
async def get_news_today(user: str = Depends(require_auth)):
    """Scored top-5 news for today. At most 3 server-side Finnhub fetches per calendar day."""
    import time as _time
    from datetime import date, timezone

    CACHE_PATH          = LOGS / "news_today_cache.json"
    MAX_FETCHES_PER_DAY = 3
    CACHE_FRESH_SECS    = 3600   # re-use cache if < 1h old regardless of fetch count
    today_str           = date.today().isoformat()

    cached_data:    dict | None = None
    n_fetches_today: int        = 0

    if CACHE_PATH.exists():
        try:
            cached = json.loads(CACHE_PATH.read_text())
            if cached.get("date") == today_str:
                n_fetches_today = cached.get("n_fetches", 0)
                cached_data = cached
        except Exception:
            pass

    if cached_data and cached_data.get("items") is not None:
        cache_age = _time.time() - cached_data.get("_ts", 0)
        if cache_age < CACHE_FRESH_SECS or n_fetches_today >= MAX_FETCHES_PER_DAY:
            return JSONResponse(content={k: v for k, v in cached_data.items() if k != "_ts"})

    try:
        from src.news.collector import NewsCollector
        from src.news.selector  import NewsSelector
        from src.config         import WATCHLIST

        CTA_EXTRA   = ["TLT", "UUP", "DBC"]
        all_tickers = WATCHLIST + CTA_EXTRA

        # Best-effort portfolio from executions log
        portfolio: dict[str, float] = {}
        try:
            raw = _read_text("logs/executions.csv")
            if raw:
                df_exec = pd.read_csv(io.StringIO(raw))
                for _, row in df_exec.iterrows():
                    sym = str(row.get("symbol", ""))
                    dq  = float(row.get("delta_qty", 0) or 0)
                    portfolio[sym] = portfolio.get(sym, 0.0) + dq
        except Exception:
            pass

        collector = NewsCollector()
        selector  = NewsSelector()

        # Borne dure sur la collecte réseau. Au-delà, on rend la main avec le
        # cache s'il existe, sinon un état vide explicite — jamais une attente
        # sans fin. Le thread sous-jacent peut continuer et remplira le cache
        # pour la prochaine requête.
        try:
            items = await asyncio.wait_for(
                asyncio.to_thread(collector.fetch_all, all_tickers, 1),
                timeout=NEWS_FETCH_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            if cached_data and cached_data.get("items") is not None:
                payload = {k: v for k, v in cached_data.items() if k != "_ts"}
                payload["stale"] = True
                return JSONResponse(content=payload)
            return JSONResponse(content={
                "date": today_str, "items": [], "timeout": True,
                "error": f"collecte au-delà de {NEWS_FETCH_TIMEOUT_SECONDS:.0f}s",
            })

        top = selector.select_daily(items, portfolio=portfolio, watchlist=WATCHLIST)

        cat_label = {"earnings": "Earnings", "ma": "M&A", "guidance": "Guidance",
                     "analyst": "Analyst",   "general": "News"}
        cat_icon  = {"earnings": "📊", "ma": "🤝", "guidance": "📈",
                     "analyst": "🔍",  "general": "📰"}

        news_items = []
        for sn in top:
            age_h = (datetime.now(timezone.utc) - sn.item.datetime).total_seconds() / 3600
            news_items.append({
                "ticker":    sn.item.ticker,
                "headline":  sn.item.headline,
                "source":    sn.item.source,
                "url":       sn.item.url,
                "category":  sn.item.category,
                "cat_label": cat_label.get(sn.item.category, "News"),
                "icon":      cat_icon.get(sn.item.category, "📰"),
                "score":     round(sn.score, 1),
                "age_h":     round(age_h, 1),
            })

        result = {
            "date":      today_str,
            "items":     news_items,
            "n_fetches": n_fetches_today + 1,
            "_ts":       _time.time(),
        }
        try:
            CACHE_PATH.write_text(json.dumps(result))
        except Exception:
            pass

        return JSONResponse(content={k: v for k, v in result.items() if k != "_ts"})

    except Exception as exc:
        return JSONResponse(content={"error": str(exc), "items": [], "date": today_str})


if __name__ == "__main__":
    print(f"🚀  Milan Capital Dashboard → http://localhost:{PORT}")
    print(f"   Mode: {'☁️  Cloud (GitHub Actions)' if IS_CLOUD else '💻 Local (subprocess)'}")
    uvicorn.run("src.dashboard.server:app", host="0.0.0.0", port=PORT, reload=not IS_CLOUD)
