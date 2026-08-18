# src/notify/push.py
"""
Notifications web poussées vers le téléphone.

Pourquoi pas Telegram
---------------------
Le fonds passait par un bot Telegram. Deux raisons d'en sortir : Nathan a
demandé que tout le pilotage vive dans le dashboard plutôt qu'éclaté sur deux
surfaces, et le jeton du bot s'était retrouvé public dans l'historique Git du
dépôt le 2026-06-03. Le Web Push n'a pas de secret partagé de ce genre : chaque
appareil détient sa propre clé, et le serveur ne conserve qu'un abonnement
révocable.

Comment ça marche, en une phrase
--------------------------------
Le navigateur crée un abonnement auprès de son propre service de notification
(Apple pour Safari, Google pour Chrome) et nous confie une URL. Pour notifier,
on envoie un message chiffré à cette URL, signé par notre clé VAPID. Le message
arrive même si le dashboard est fermé.

Deux conditions, toutes deux remplies :
  - HTTPS valide — obtenu le 2026-08-13 via Caddy
  - un service worker avec un gestionnaire `push` — voir server.py

Ce que ce module ne fait pas
----------------------------
Il n'envoie jamais de contenu sensible. Une notification traverse
l'infrastructure d'Apple ou de Google : elle annonce qu'une décision a été
prise, pas le détail des positions. Le détail se consulte dans le dashboard,
derrière l'authentification.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

KEYS_PATH = Path("logs/vapid_keys.json")
SUBS_PATH = Path("logs/push_subscriptions.json")

# Adresse de contact exigée par la spécification VAPID : elle permet à Apple ou
# Google de signaler un problème plutôt que de couper l'envoi sans prévenir.
VAPID_SUBJECT = "mailto:nathan.floiras@gmail.com"

# Codes signalant un abonnement définitivement mort : l'utilisateur a désinstallé
# l'application, révoqué l'autorisation, ou changé d'appareil. On le retire au
# lieu de réessayer indéfiniment.
_DEAD_CODES = (404, 410)


@dataclass(frozen=True)
class PushResult:
    sent: int
    failed: int
    pruned: int

    def render(self) -> str:
        s = f"{self.sent} envoyée(s)"
        if self.failed:
            s += f", {self.failed} échec(s)"
        if self.pruned:
            s += f", {self.pruned} abonnement(s) périmé(s) retiré(s)"
        return s


# ── Clés VAPID ────────────────────────────────────────────────────────────────

def get_or_create_keys(path: Path = KEYS_PATH) -> Dict[str, str]:
    """
    Paire de clés VAPID, créée à la première utilisation puis réutilisée.

    Les régénérer invaliderait tous les abonnements existants : les appareils
    déjà inscrits cesseraient de recevoir sans message d'erreur. Le fichier est
    donc écrit une fois et n'est jamais remplacé automatiquement.
    """
    if path.exists():
        try:
            d = json.loads(path.read_text())
            if d.get("public_key") and d.get("private_key"):
                # Les clés écrites avant le correctif du 2026-08-13 sont au
                # format PEM, que pywebpush refuse. On les régénère — les
                # abonnements existants deviennent caducs et les appareils
                # doivent se réinscrire, mais ils ne recevaient rien de toute
                # façon.
                if d["private_key"].lstrip().startswith("-----BEGIN"):
                    logger.warning(
                        "clé VAPID au format PEM (obsolète) — régénération. "
                        "Les appareils déjà inscrits doivent se réabonner.")
                else:
                    return d
        except Exception:
            logger.warning("clés VAPID illisibles, régénération : %s", path)

    from py_vapid import Vapid02
    from cryptography.hazmat.primitives import serialization
    import base64

    v = Vapid02()
    v.generate_keys()

    # Clé privée au format BRUT encodé en base64url, pas en PEM.
    #
    # pywebpush appelle `Vapid.from_string()` dès que la valeur n'est pas un
    # chemin de fichier existant. Un PEM multi-ligne passé comme chaîne échoue
    # sur « Could not deserialize key data » — constaté au premier run réel du
    # 2026-08-13 : la séance s'est terminée normalement, mais la notification
    # n'est jamais partie.
    priv = base64.urlsafe_b64encode(
        v.private_key.private_numbers().private_value.to_bytes(32, "big")
    ).decode().rstrip("=")

    raw_pub = v.public_key.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint,
    )
    pub = base64.urlsafe_b64encode(raw_pub).decode().rstrip("=")

    keys = {"public_key": pub, "private_key": priv,
            "created_at": datetime.now(timezone.utc).isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(keys, indent=2))
    try:
        path.chmod(0o600)
    except Exception:
        pass
    return keys


def public_key(path: Path = KEYS_PATH) -> str:
    """Clé publique, seule valeur transmise au navigateur."""
    return get_or_create_keys(path)["public_key"]


# ── Abonnements ───────────────────────────────────────────────────────────────

def _load_subs(path: Path = SUBS_PATH) -> List[dict]:
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text())
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _save_subs(subs: List[dict], path: Path = SUBS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(subs, indent=2))
    try:
        path.chmod(0o600)
    except Exception:
        pass


def add_subscription(sub: dict, path: Path = SUBS_PATH) -> int:
    """
    Enregistre un appareil. L'`endpoint` sert de clé : réinstaller
    l'application depuis le même appareil met à jour au lieu de dupliquer, sans
    quoi chaque réinstallation aurait produit une notification supplémentaire.
    """
    if not isinstance(sub, dict) or not sub.get("endpoint"):
        raise ValueError("abonnement invalide : endpoint manquant")
    subs = [s for s in _load_subs(path) if s.get("endpoint") != sub["endpoint"]]
    subs.append({**sub, "added_at": datetime.now(timezone.utc).isoformat()})
    _save_subs(subs, path)
    return len(subs)


def remove_subscription(endpoint: str, path: Path = SUBS_PATH) -> int:
    subs = [s for s in _load_subs(path) if s.get("endpoint") != endpoint]
    _save_subs(subs, path)
    return len(subs)


def count_subscriptions(path: Path = SUBS_PATH) -> int:
    return len(_load_subs(path))


# ── Envoi ─────────────────────────────────────────────────────────────────────

def send_push(
    title: str,
    body: str,
    url: str = "/",
    tag: Optional[str] = None,
    subs_path: Optional[Path] = None,
    keys_path: Optional[Path] = None,
) -> PushResult:
    """
    Notifie tous les appareils inscrits.

    Ne lève jamais : une notification est un confort, pas une étape du trading.
    Si Apple est injoignable, le run doit se poursuivre — d'où le fait que
    l'appelant n'ait rien à protéger.

    `tag` permet au téléphone de remplacer une notification par la suivante
    plutôt que de les empiler : deux runs le même jour ne doivent pas laisser
    deux lignes contradictoires dans le centre de notifications.
    """
    # Résolus à l'appel, pas à la définition. Des valeurs par défaut nommées
    # (`subs_path: Path = SUBS_PATH`) sont évaluées une seule fois, au
    # chargement du module : redéfinir SUBS_PATH ensuite n'avait plus aucun
    # effet, et `notify_run_complete` écrivait toujours dans le chemin d'origine.
    subs_path = subs_path or SUBS_PATH
    keys_path = keys_path or KEYS_PATH

    subs = _load_subs(subs_path)
    if not subs:
        return PushResult(0, 0, 0)

    try:
        from pywebpush import WebPushException, webpush
    except ImportError:
        logger.warning("pywebpush absent — notification non envoyée")
        return PushResult(0, len(subs), 0)

    keys = get_or_create_keys(keys_path)
    payload = json.dumps({"title": title, "body": body, "url": url,
                          "tag": tag or "milan"})

    sent = failed = 0
    dead: List[str] = []
    for sub in subs:
        try:
            webpush(
                subscription_info={k: sub[k] for k in ("endpoint", "keys")
                                   if k in sub},
                data=payload,
                vapid_private_key=keys["private_key"],
                vapid_claims={"sub": VAPID_SUBJECT},
                timeout=10,
            )
            sent += 1
        except WebPushException as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code in _DEAD_CODES:
                dead.append(sub.get("endpoint", ""))
            else:
                failed += 1
                logger.warning("push échoué (%s) : %s", code, str(exc)[:120])
        except Exception as exc:
            failed += 1
            logger.warning("push échoué : %s", str(exc)[:120])

    if dead:
        _save_subs([s for s in subs if s.get("endpoint") not in dead], subs_path)

    return PushResult(sent=sent, failed=failed, pruned=len(dead))


def notify_run_complete(
    n_orders: int,
    n_rejected: int,
    netliq: float,
    regime: str,
    executed: bool,
    broker_ok: bool = True,
) -> PushResult:
    """
    Résumé d'un run, poussé sur le téléphone.

    Volontairement bref et sans détail de position : le message transite par
    l'infrastructure d'Apple ou de Google. Il dit qu'une décision a été prise et
    invite à ouvrir le dashboard, qui lui est authentifié.

    `broker_ok=False` n'est pas une variante du message : c'est une panne
    ------------------------------------------------------------------
    Avant le 2026-08-18, cette fonction ne savait pas distinguer « le risque a
    tout écarté » de « le courtier était injoignable ». Les deux produisaient
    le même titre, « aucun ordre ».

    IB Gateway s'est déconnecté le samedi 2026-08-15 à 23h45 — son
    redémarrage quotidien obligatoire, suivi d'un « Unrecognized Username or
    Password ». Le fonds a continué d'analyser, de décider et d'appliquer ses
    garde-fous dans le vide pendant trois jours. La notification du lundi
    annonçait « aucun ordre · 5 plan(s) écarté(s) par le risque », ce qui est
    exactement ce qu'affiche une journée calme normale.

    Une panne déguisée en fonctionnement normal est pire qu'une absence de
    notification : elle fabrique une fausse confiance. Le titre doit donc être
    impossible à confondre avec un run réussi.
    """
    if not broker_ok:
        return send_push(
            "🔴 Milan Capital — COURTIER INJOIGNABLE",
            (f"Aucun ordre n'a pu partir. {n_orders} décision(s) perdue(s). "
             f"Le fonds tourne à vide tant que la passerelle IBKR n'est pas "
             f"reconnectée."),
            url="/", tag="run",
        )

    if n_orders == 0:
        title = "Milan Capital — aucun ordre"
        body = f"Régime {regime.upper()} · {n_rejected} plan(s) écarté(s) par le risque"
    else:
        verbe = "envoyé" if executed else "préparé"
        title = f"Milan Capital — {n_orders} ordre{'s' if n_orders > 1 else ''} {verbe}"
        body = f"Régime {regime.upper()} · capital ${netliq:,.0f}"
    return send_push(title, body, url="/", tag="run")
