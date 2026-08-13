# src/execution/run_lock.py
"""
Verrou d'exécution — un seul run du fonds à la fois.

Le risque, concrètement
-----------------------
`execute_plans_paper_ibkr` refuse de rejouer un plan déjà exécuté en comparant
les `plan_id` déjà présents dans `logs/executions.csv`. Cette garde protège
contre le fait de **rejouer le même plan**. Elle ne protège pas du tout contre
**deux plans différents produits en même temps** : chaque run tire un
`plan_id` neuf (`uuid4().hex[:8]`), donc deux runs simultanés produisent deux
identifiants distincts, et les deux passent.

Deux occasions banales de déclencher ça une fois l'exécution automatique
activée :
  - cliquer « run » dans le dashboard pendant que le run planifié tourne ;
  - un run planifié qui déborde sur le suivant après un incident réseau.

Résultat : deux plans calculés sur le même portefeuille de départ, chacun
croyant partir de zéro, et une position ouverte deux fois.

Pourquoi `flock` plutôt qu'un fichier PID
------------------------------------------
Le noyau libère un `flock` dès que le processus qui le détient disparaît, quelle
qu'en soit la raison — arrêt brutal, machine qui redémarre, processus tué. Il
n'existe donc pas de verrou « resté coincé » après un crash, et donc aucune
logique de péremption à écrire ni à tester. Un fichier PID, lui, survit au
crash et finit toujours par exiger un nettoyage manuel un jour de panne.

Le fichier porte aussi un contenu lisible (pid, date, origine du
déclenchement), pour que `logs/run.lock` réponde à la question « qui tourne en
ce moment ? » sans outil supplémentaire.
"""
from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_LOCK_PATH = Path("logs/run.lock")

# Origines possibles d'un run. Conservée dans le verrou puis dans le journal :
# sans elle, impossible de distinguer six mois plus tard une séance produite par
# le planificateur d'une séance forcée à la main — et un run manuel est une
# décision discrétionnaire, qui n'a pas le même statut dans un track record.
TRIGGER_SCHEDULED = "scheduled"
TRIGGER_MANUAL = "manual"
TRIGGER_CLI = "cli"


@dataclass(frozen=True)
class LockInfo:
    """Qui détient le verrou."""
    pid: int
    started_at: str
    trigger: str

    @classmethod
    def from_json(cls, raw: str) -> Optional["LockInfo"]:
        try:
            d = json.loads(raw)
            return cls(int(d["pid"]), str(d["started_at"]), str(d["trigger"]))
        except Exception:
            return None

    def render(self) -> str:
        return (f"pid={self.pid} démarré={self.started_at} "
                f"origine={self.trigger}")


class RunLockBusy(RuntimeError):
    """Un run est déjà en cours. Le second doit renoncer, pas attendre."""

    def __init__(self, holder: Optional[LockInfo]) -> None:
        self.holder = holder
        detail = holder.render() if holder else "détenteur inconnu"
        super().__init__(
            f"Un run est déjà en cours ({detail}). "
            f"Exécution refusée pour éviter deux plans simultanés."
        )


@contextmanager
def run_lock(
    path: str | Path = DEFAULT_LOCK_PATH,
    trigger: str = TRIGGER_CLI,
) -> Iterator[LockInfo]:
    """
    Prend le verrou, ou lève `RunLockBusy` immédiatement.

    Volontairement **non bloquant** : un run qui attendrait son tour finirait
    par s'exécuter sur des prix périmés et un portefeuille qui a changé entre
    temps. Mieux vaut renoncer bruyamment et laisser le planificateur reprendre
    au créneau suivant.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # `a+` n'écrase pas le contenu existant : si l'acquisition échoue, on veut
    # pouvoir lire qui détient le verrou.
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            holder = None
            try:
                with open(str(p)) as fh:
                    holder = LockInfo.from_json(fh.read())
            except Exception:
                pass
            raise RunLockBusy(holder)

        info = LockInfo(
            pid=os.getpid(),
            started_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            trigger=trigger,
        )
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, json.dumps({
            "pid": info.pid,
            "started_at": info.started_at,
            "trigger": info.trigger,
        }).encode())
        os.fsync(fd)

        try:
            yield info
        finally:
            # Le contenu est vidé pour que le fichier ne décrive pas un run
            # terminé. Le verrou lui-même tombe à la fermeture du descripteur,
            # y compris si le processus est tué avant d'arriver ici.
            try:
                os.ftruncate(fd, 0)
            except Exception:
                pass
    finally:
        os.close(fd)


def current_holder(path: str | Path = DEFAULT_LOCK_PATH) -> Optional[LockInfo]:
    """
    Le run en cours, ou None.

    Utilisé par le dashboard pour griser le bouton plutôt que de laisser
    l'utilisateur déclencher un refus. Ne prend aucun verrou.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        raw = p.read_text()
    except Exception:
        return None
    if not raw.strip():
        return None
    return LockInfo.from_json(raw)
