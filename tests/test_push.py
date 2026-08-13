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

    def test_private_key_never_leaves_through_public_key(self, keys_path):
        k = P.get_or_create_keys(keys_path)
        assert "PRIVATE" in k["private_key"]
        assert "PRIVATE" not in P.public_key(keys_path)


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
