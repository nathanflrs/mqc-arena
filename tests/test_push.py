"""
Notifications web push (src/notify/push.py).

Trois exigences, toutes issues d'incidents ou de décisions réelles :

- Une notification ne doit JAMAIS interrompre un run. C'est un confort, pas une
  étape du trading : si Apple est injoignable, la séance continue.
- Régénérer les clés VAPID invaliderait tous les abonnements sans message
  d'erreur — les appareils cesseraient simplement de recevoir. Elles sont donc
  créées une fois et jamais remplacées automatiquement.
- Un abonnement mort (application désinstallée, autorisation révoquée) doit être
  retiré, pas réessayé indéfiniment.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.notify import push as P


@pytest.fixture
def keys_path(tmp_path) -> Path:
    return tmp_path / "vapid.json"


@pytest.fixture
def subs_path(tmp_path) -> Path:
    return tmp_path / "subs.json"


def _sub(endpoint: str) -> dict:
    return {"endpoint": endpoint,
            "keys": {"p256dh": "fake-p256dh", "auth": "fake-auth"}}


# ── Clés ─────────────────────────────────────────────────────────────────────

class TestVapidKeys:

    def test_keys_are_created_once_and_reused(self, keys_path):
        """
        Les régénérer couperait silencieusement tous les appareils déjà
        inscrits : ils ne recevraient plus rien, sans erreur visible.
        """
        first = P.get_or_create_keys(keys_path)
        second = P.get_or_create_keys(keys_path)
        assert first["public_key"] == second["public_key"]
        assert first["private_key"] == second["private_key"]

    def test_public_key_is_url_safe_base64_without_padding(self, keys_path):
        """Format imposé par l'API du navigateur, qui refuse tout autre."""
        k = P.public_key(keys_path)
        assert not k.endswith("=")
        assert "+" not in k and "/" not in k
        assert len(k) > 80

    def test_corrupt_key_file_is_regenerated(self, keys_path):
        keys_path.write_text("{ pas du json")
        assert P.get_or_create_keys(keys_path)["public_key"]

    def test_public_key_does_not_contain_the_private_one(self, keys_path):
        """
        `public_key()` est la seule valeur transmise au navigateur. Elle ne doit
        rien laisser filtrer de la clé privée, qui signe les notifications.
        """
        k = P.get_or_create_keys(keys_path)
        pub = P.public_key(keys_path)
        assert k["private_key"] not in pub
        assert pub == k["public_key"]
        # 32 octets encodés en base64url sans remplissage.
        assert len(k["private_key"]) == 43


    def test_private_key_is_loadable_by_pywebpush(self, keys_path):
        """
        Régression du 2026-08-13. La clé était écrite en PEM ; pywebpush appelle
        `Vapid.from_string()` dès que la valeur n'est pas un chemin de fichier,
        et un PEM multi-ligne y échoue sur « Could not deserialize key data ».
        Le premier run réel s'est terminé normalement, mais la notification
        n'est jamais partie — l'échec n'apparaissait que dans les journaux.
        """
        from py_vapid import Vapid02 as Vapid
        k = P.get_or_create_keys(keys_path)
        assert not k["private_key"].startswith("-----BEGIN"), \
            "le PEM n'est pas accepté par pywebpush"
        Vapid.from_string(private_key=k["private_key"])   # ne doit pas lever

    def test_legacy_pem_key_is_migrated(self, keys_path):
        """Une clé PEM déjà écrite doit être remplacée, pas conservée."""
        import json
        keys_path.write_text(json.dumps({
            "public_key": "ancienne",
            "private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
        }))
        k = P.get_or_create_keys(keys_path)
        assert not k["private_key"].startswith("-----BEGIN")
        assert k["public_key"] != "ancienne"


# ── Abonnements ──────────────────────────────────────────────────────────────

class TestSubscriptions:

    def test_same_device_updates_instead_of_duplicating(self, subs_path):
        """
        Sans déduplication sur l'endpoint, chaque réinstallation de
        l'application aurait ajouté un abonnement — et donc une notification
        supplémentaire par run.
        """
        P.add_subscription(_sub("https://push.apple.com/AAA"), subs_path)
        P.add_subscription(_sub("https://push.apple.com/AAA"), subs_path)
        assert P.count_subscriptions(subs_path) == 1

    def test_two_devices_are_both_kept(self, subs_path):
        P.add_subscription(_sub("https://push.apple.com/AAA"), subs_path)
        P.add_subscription(_sub("https://fcm.google.com/BBB"), subs_path)
        assert P.count_subscriptions(subs_path) == 2

    def test_subscription_without_endpoint_is_rejected(self, subs_path):
        with pytest.raises(ValueError):
            P.add_subscription({"keys": {}}, subs_path)

    def test_removal(self, subs_path):
        P.add_subscription(_sub("https://push.apple.com/AAA"), subs_path)
        assert P.remove_subscription("https://push.apple.com/AAA", subs_path) == 0

    def test_file_is_not_world_readable(self, subs_path):
        P.add_subscription(_sub("https://push.apple.com/AAA"), subs_path)
        assert subs_path.stat().st_mode & 0o077 == 0, \
            "un abonnement permet d'écrire sur le téléphone : pas lisible par tous"


# ── Envoi ────────────────────────────────────────────────────────────────────

class TestSending:

    def test_no_subscriber_is_not_an_error(self, subs_path, keys_path):
        r = P.send_push("t", "b", subs_path=subs_path, keys_path=keys_path)
        assert (r.sent, r.failed, r.pruned) == (0, 0, 0)

    def test_transport_failure_never_raises(self, subs_path, keys_path, monkeypatch):
        """
        Le point le plus important du module : `send_push` est appelé depuis le
        runner. S'il levait, une panne d'Apple interromprait une séance de
        trading.
        """
        P.add_subscription(_sub("https://push.apple.com/AAA"), subs_path)

        import pywebpush
        def boom(*a, **k):
            raise RuntimeError("réseau indisponible")
        monkeypatch.setattr(pywebpush, "webpush", boom)

        r = P.send_push("t", "b", subs_path=subs_path, keys_path=keys_path)
        assert r.failed == 1 and r.sent == 0
        assert P.count_subscriptions(subs_path) == 1, \
            "une panne réseau n'est pas un abonnement mort"

    def test_dead_subscription_is_pruned(self, subs_path, keys_path, monkeypatch):
        """404 / 410 = appareil définitivement parti. Réessayer serait vain."""
        P.add_subscription(_sub("https://push.apple.com/GONE"), subs_path)

        import pywebpush

        class FakeResponse:
            status_code = 410

        def gone(*a, **k):
            raise pywebpush.WebPushException("parti", response=FakeResponse())

        monkeypatch.setattr(pywebpush, "webpush", gone)
        r = P.send_push("t", "b", subs_path=subs_path, keys_path=keys_path)
        assert r.pruned == 1
        assert P.count_subscriptions(subs_path) == 0

    def test_payload_carries_title_body_and_tag(self, subs_path, keys_path, monkeypatch):
        P.add_subscription(_sub("https://push.apple.com/AAA"), subs_path)
        captured = {}

        import pywebpush
        def capture(subscription_info, data, **kw):
            captured.update(json.loads(data))
        monkeypatch.setattr(pywebpush, "webpush", capture)

        P.send_push("Titre", "Corps", url="/perf", tag="run",
                    subs_path=subs_path, keys_path=keys_path)
        assert captured == {"title": "Titre", "body": "Corps",
                            "url": "/perf", "tag": "run"}

    def test_run_summary_mentions_no_position_detail(self, subs_path, keys_path, monkeypatch):
        """
        Le message transite par Apple ou Google. Il annonce qu'une décision a
        été prise ; le détail reste derrière l'authentification du dashboard.
        """
        P.add_subscription(_sub("https://push.apple.com/AAA"), subs_path)
        captured = {}

        import pywebpush
        def capture(subscription_info, data, **kw):
            captured.update(json.loads(data))
        monkeypatch.setattr(pywebpush, "webpush", capture)
        monkeypatch.setattr(P, "SUBS_PATH", subs_path)
        monkeypatch.setattr(P, "KEYS_PATH", keys_path)

        P.notify_run_complete(n_orders=3, n_rejected=8, netliq=1_020_809.57,
                              regime="bull", executed=True)
        texte = captured["title"] + captured["body"]
        for interdit in ("AAPL", "NVDA", "TLT", "action", "titre "):
            assert interdit not in texte
        assert "3 ordres" in captured["title"]

    def test_zero_orders_says_so_explicitly(self, subs_path, keys_path, monkeypatch):
        P.add_subscription(_sub("https://push.apple.com/AAA"), subs_path)
        captured = {}
        import pywebpush
        monkeypatch.setattr(pywebpush, "webpush",
                            lambda subscription_info, data, **kw: captured.update(json.loads(data)))
        monkeypatch.setattr(P, "SUBS_PATH", subs_path)
        monkeypatch.setattr(P, "KEYS_PATH", keys_path)

        P.notify_run_complete(0, 12, 1_000_000, "bear", False)
        assert "aucun ordre" in captured["title"].lower()
        assert "12" in captured["body"]



class TestPanneCourtier:
    """
    Régression du 2026-08-18.

    IB Gateway s'est déconnecté le samedi 2026-08-15 à 23h45 lors de son
    redémarrage quotidien : « Unrecognized Username or Password ». Le fonds a
    analysé, décidé et appliqué ses garde-fous dans le vide pendant trois jours.

    La notification du lundi disait « Milan Capital — aucun ordre · Régime BULL
    · 5 plan(s) écarté(s) par le risque » — c'est-à-dire exactement ce
    qu'affiche une journée calme réussie. Une panne déguisée en fonctionnement
    normal est pire que pas de notification du tout.
    """

    def _capture(self, monkeypatch):
        envoyes = []
        from src.notify import push as p
        monkeypatch.setattr(p, "send_push",
                            lambda t, b, **k: envoyes.append((t, b)) or p.PushResult(1, 0, 0))
        return envoyes

    def test_une_panne_de_courtier_ne_ressemble_pas_a_un_jour_calme(self, monkeypatch):
        from src.notify.push import notify_run_complete
        envoyes = self._capture(monkeypatch)

        notify_run_complete(n_orders=0, n_rejected=5, netliq=1_000_000,
                            regime="bull", executed=False, broker_ok=True)
        notify_run_complete(n_orders=4, n_rejected=5, netliq=1_000_000,
                            regime="bull", executed=False, broker_ok=False)

        calme, panne = envoyes[0][0], envoyes[1][0]
        assert calme != panne
        assert "COURTIER INJOIGNABLE" in panne
        assert "aucun ordre" in calme.lower()

    def test_la_panne_dit_combien_de_decisions_sont_perdues(self, monkeypatch):
        from src.notify.push import notify_run_complete
        envoyes = self._capture(monkeypatch)
        notify_run_complete(n_orders=7, n_rejected=2, netliq=1_000_000,
                            regime="bull", executed=False, broker_ok=False)
        assert "7" in envoyes[0][1], "le nombre de décisions perdues doit apparaître"

    def test_le_comportement_normal_est_inchange(self, monkeypatch):
        """broker_ok vaut True par défaut : aucun appel existant ne change."""
        from src.notify.push import notify_run_complete
        envoyes = self._capture(monkeypatch)
        notify_run_complete(n_orders=3, n_rejected=1, netliq=1_021_514,
                            regime="bull", executed=True)
        titre, corps = envoyes[0]
        assert "3 ordres envoyé" in titre and "COURTIER" not in titre
        assert "1,021,514" in corps
