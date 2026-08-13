"""
Verrou d'exécution (src/execution/run_lock.py).

Le scénario visé n'est pas théorique : une fois l'exécution automatique
activée, cliquer « run » dans le dashboard pendant que le run planifié tourne
produit deux plans avec des `plan_id` distincts. La garde anti-rejeu existante
compare les `plan_id` — elle ne voit donc rien, et la position s'ouvre deux
fois.

Le test central est `test_second_run_is_refused_while_first_holds`. Les autres
vérifient que le verrou ne se retourne pas contre nous : libéré à la sortie,
libéré même sur exception, et jamais coincé après un arrêt brutal.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.execution.run_lock import (
    TRIGGER_MANUAL, TRIGGER_SCHEDULED, LockInfo, RunLockBusy, current_holder,
    run_lock,
)


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / "run.lock"


# ── Le test qui justifie le module ───────────────────────────────────────────

class TestMutualExclusion:

    def test_second_run_is_refused_while_first_holds(self, lock_path):
        with run_lock(lock_path, trigger=TRIGGER_SCHEDULED):
            with pytest.raises(RunLockBusy) as exc:
                with run_lock(lock_path, trigger=TRIGGER_MANUAL):
                    pytest.fail("le second run n'aurait jamais dû démarrer")
        assert "déjà en cours" in str(exc.value)

    def test_refusal_names_the_holder(self, lock_path):
        """
        Un refus muet enverrait chercher la cause dans les logs. Le message doit
        dire qui occupe la place et depuis quand.
        """
        with run_lock(lock_path, trigger=TRIGGER_SCHEDULED):
            with pytest.raises(RunLockBusy) as exc:
                with run_lock(lock_path):
                    pass
        holder = exc.value.holder
        assert holder is not None
        assert holder.pid == os.getpid()
        assert holder.trigger == TRIGGER_SCHEDULED
        assert str(holder.pid) in str(exc.value)

    def test_refusal_is_immediate_not_queued(self, lock_path):
        """
        Attendre son tour ferait s'exécuter le second run sur des prix périmés
        et un portefeuille modifié entre temps. Il doit renoncer, pas patienter.
        """
        import time
        with run_lock(lock_path):
            t0 = time.monotonic()
            with pytest.raises(RunLockBusy):
                with run_lock(lock_path):
                    pass
            assert time.monotonic() - t0 < 0.5, "le second run a attendu"


class TestRelease:

    def test_lock_is_released_after_normal_exit(self, lock_path):
        with run_lock(lock_path):
            pass
        with run_lock(lock_path) as info:      # ne doit pas lever
            assert info.pid == os.getpid()

    def test_lock_is_released_when_the_run_crashes(self, lock_path):
        """
        Un run qui plante ne doit pas condamner tous les suivants. C'est le cas
        le plus fréquent en production : un agent qui lève, un timeout réseau.
        """
        with pytest.raises(ValueError):
            with run_lock(lock_path):
                raise ValueError("panne simulée")
        with run_lock(lock_path):              # ne doit pas lever
            pass

    def test_kill_minus_nine_does_not_strand_the_lock(self, lock_path, tmp_path):
        """
        Le point qui distingue `flock` d'un fichier PID : le noyau libère le
        verrou quand le processus disparaît, même tué sans ménagement. Sans ça,
        une coupure de courant en pleine séance exigerait un nettoyage manuel
        avant le run suivant.
        """
        root = Path(__file__).resolve().parents[1]
        ready = tmp_path / "ready"
        child = textwrap.dedent(f"""
            import sys, time, pathlib
            sys.path.insert(0, {str(root)!r})
            from src.execution.run_lock import run_lock
            with run_lock({str(lock_path)!r}):
                pathlib.Path({str(ready)!r}).write_text("held")
                time.sleep(60)
        """)
        proc = subprocess.Popen([sys.executable, "-c", child])
        try:
            for _ in range(100):
                if ready.exists():
                    break
                import time as _t; _t.sleep(0.05)
            assert ready.exists(), "le processus fils n'a pas pris le verrou"

            with pytest.raises(RunLockBusy):
                with run_lock(lock_path):
                    pass

            proc.kill()
            proc.wait(timeout=10)
        finally:
            if proc.poll() is None:
                proc.kill()

        with run_lock(lock_path):              # doit passer immédiatement
            pass


class TestProvenance:

    def test_trigger_is_recorded(self, lock_path):
        """
        Un run manuel est une décision discrétionnaire. Ne pas l'étiqueter rend
        le track record inanalysable : impossible de séparer plus tard ce que le
        planificateur a fait de ce qui a été forcé à la main.
        """
        with run_lock(lock_path, trigger=TRIGGER_MANUAL):
            assert json.loads(lock_path.read_text())["trigger"] == TRIGGER_MANUAL

    def test_current_holder_reports_the_running_job(self, lock_path):
        assert current_holder(lock_path) is None
        with run_lock(lock_path, trigger=TRIGGER_SCHEDULED) as info:
            seen = current_holder(lock_path)
            assert seen is not None and seen.pid == info.pid
        assert current_holder(lock_path) is None, \
            "après la sortie, plus personne ne tourne"

    def test_current_holder_takes_no_lock(self, lock_path):
        """Consulter l'état ne doit jamais empêcher un run de démarrer."""
        assert current_holder(lock_path) is None
        with run_lock(lock_path):
            pass

    def test_corrupt_lock_file_is_not_fatal(self, lock_path):
        lock_path.write_text("{ pas du json")
        assert current_holder(lock_path) is None
        with run_lock(lock_path):              # doit rester acquérable
            pass

    def test_lockinfo_rejects_garbage(self):
        assert LockInfo.from_json("") is None
        assert LockInfo.from_json('{"pid": "x"}') is None
