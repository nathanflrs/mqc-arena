"""
Tests for InsiderBuyAgent + SECInsiderClient (Form 4).

All HTTP calls are mocked — no network access.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import MarketState
from src.agents.insider_buy import InsiderBuyAgent, InsiderBuyConfig
from src.data.sec_insider import (
    SECInsiderClient,
    InsiderTransaction,
    _parse_form4_xml,
)


# ── Fixtures & helpers ────────────────────────────────────────────────────────

def _state(symbol: str = "AAPL", price: float = 200.0) -> MarketState:
    return MarketState(symbol=symbol, price=price, timestamp="2026-07-15T09:30:00Z")


def _make_transaction(
    ticker: str = "AAPL",
    reporter: str = "Cook Timothy",
    is_officer: bool = True,
    total_value: float = 150_000.0,
    days_ago: int = 5,
) -> InsiderTransaction:
    txn_date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).date().isoformat()
    return InsiderTransaction(
        ticker=ticker,
        reporter_name=reporter,
        is_officer=is_officer,
        is_director=not is_officer,
        transaction_date=txn_date,
        shares=total_value / 200.0,
        price_per_share=200.0,
        total_value=total_value,
        transaction_code="P",
    )


def _form4_xml(
    reporter: str = "Cook Timothy",
    is_officer: bool = True,
    is_director: bool = False,
    shares: float = 1000.0,
    price: float = 200.0,
    txn_date: str = "2026-07-10",
    code: str = "P",
    ad_code: str = "A",
) -> bytes:
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerName>{reporter}</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>{'1' if is_director else '0'}</isDirector>
      <isOfficer>{'1' if is_officer else '0'}</isOfficer>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>{txn_date}</value></transactionDate>
      <transactionCoding>
        <transactionCode>{code}</transactionCode>
      </transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{ad_code}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>""".encode()


def _client_with_transactions(txns: list) -> SECInsiderClient:
    """Return a SECInsiderClient whose get_recent_purchases() is pre-set."""
    client = MagicMock(spec=SECInsiderClient)
    client.get_recent_purchases.return_value = txns
    return client


# ── XML parser tests ──────────────────────────────────────────────────────────

class TestParseForm4XML:

    def test_parses_officer_purchase(self):
        xml = _form4_xml(reporter="Cook Timothy", is_officer=True, shares=750, price=200.0)
        txns = _parse_form4_xml(xml, "AAPL")
        assert len(txns) == 1
        t = txns[0]
        assert t.reporter_name == "Cook Timothy"
        assert t.is_officer is True
        assert t.shares == pytest.approx(750.0)
        assert t.price_per_share == pytest.approx(200.0)
        assert t.total_value == pytest.approx(150_000.0)
        assert t.transaction_code == "P"

    def test_parses_director_purchase(self):
        xml = _form4_xml(is_officer=False, is_director=True)
        txns = _parse_form4_xml(xml, "AAPL")
        assert len(txns) == 1
        assert txns[0].is_director is True

    def test_sale_excluded(self):
        """Transaction code S (sale) must not be returned."""
        xml = _form4_xml(code="S", ad_code="D")
        txns = _parse_form4_xml(xml, "AAPL")
        assert txns == []

    def test_disposed_excluded(self):
        """Acquired/Disposed code D means disposal — excluded even if code=P."""
        xml = _form4_xml(code="P", ad_code="D")
        txns = _parse_form4_xml(xml, "AAPL")
        assert txns == []

    def test_non_officer_non_director_excluded(self):
        """Only officers and directors qualify."""
        xml = _form4_xml(is_officer=False, is_director=False)
        txns = _parse_form4_xml(xml, "AAPL")
        assert txns == []

    def test_award_excluded(self):
        """Transaction code A (award/grant) is excluded."""
        xml = _form4_xml(code="A", ad_code="A")
        txns = _parse_form4_xml(xml, "AAPL")
        assert txns == []

    def test_malformed_xml_returns_empty(self):
        txns = _parse_form4_xml(b"this is not xml", "AAPL")
        assert txns == []

    def test_empty_xml_returns_empty(self):
        txns = _parse_form4_xml(b"<ownershipDocument/>", "AAPL")
        assert txns == []

    def test_ticker_propagated(self):
        xml = _form4_xml()
        txns = _parse_form4_xml(xml, "MSFT")
        assert txns[0].ticker == "MSFT"


# ── SECInsiderClient unit tests ───────────────────────────────────────────────

class TestSECInsiderClientCIK:

    def test_cik_lookup_hit(self, tmp_path, monkeypatch):
        """CIK map loaded from cache returns correct CIK."""
        monkeypatch.chdir(tmp_path)
        cache_dir = tmp_path / "logs" / "insider_cache"
        cache_dir.mkdir(parents=True)

        import src.data.sec_insider as mod
        monkeypatch.setattr(mod, "_CACHE_DIR", cache_dir)

        cik_data = {
            "AAPL": 320193,
            "_cached_at": datetime.now(timezone.utc).isoformat(),
        }
        (cache_dir / "cik_map.json").write_text(json.dumps(cik_data))

        client = SECInsiderClient()
        cik = client._get_cik("AAPL")
        assert cik == 320193

    def test_cik_missing_returns_none(self, tmp_path, monkeypatch):
        """Ticker not in SEC universe (ETF) returns None."""
        import src.data.sec_insider as mod
        cache_dir = tmp_path / "logs" / "insider_cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(mod, "_CACHE_DIR", cache_dir)

        cik_data = {"AAPL": 320193, "_cached_at": datetime.now(timezone.utc).isoformat()}
        (cache_dir / "cik_map.json").write_text(json.dumps(cik_data))

        client = SECInsiderClient()
        assert client._get_cik("SPY") is None

    def test_cik_alias_brk_b(self, tmp_path, monkeypatch):
        """BRK-B → BRKB normalisation."""
        import src.data.sec_insider as mod
        cache_dir = tmp_path / "logs" / "insider_cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(mod, "_CACHE_DIR", cache_dir)

        cik_data = {"BRKB": 1067983, "_cached_at": datetime.now(timezone.utc).isoformat()}
        (cache_dir / "cik_map.json").write_text(json.dumps(cik_data))

        client = SECInsiderClient()
        assert client._get_cik("BRK-B") == 1067983


class TestSECInsiderClientFetch:

    def _setup(self, tmp_path, monkeypatch):
        import src.data.sec_insider as mod
        cache_dir = tmp_path / "logs" / "insider_cache"
        cache_dir.mkdir(parents=True)
        monkeypatch.setattr(mod, "_CACHE_DIR", cache_dir)
        return mod, cache_dir

    def test_etf_returns_empty(self, tmp_path, monkeypatch):
        """SPY has no CIK → get_recent_purchases returns []."""
        mod, cache_dir = self._setup(tmp_path, monkeypatch)
        cik_data = {"AAPL": 320193, "_cached_at": datetime.now(timezone.utc).isoformat()}
        (cache_dir / "cik_map.json").write_text(json.dumps(cik_data))

        client = SECInsiderClient()
        result = client.get_recent_purchases("SPY")
        assert result == []

    def test_purchases_returned_within_window(self, tmp_path, monkeypatch):
        """Recent filings within lookback_days are returned."""
        mod, cache_dir = self._setup(tmp_path, monkeypatch)

        # Pre-seed CIK cache
        today = date.today().isoformat()
        (cache_dir / "cik_map.json").write_text(json.dumps(
            {"AAPL": 320193, "_cached_at": datetime.now(timezone.utc).isoformat()}
        ))

        # Pre-seed submissions cache with one Form 4 filed today
        acc = "0001234567-26-000001"
        (cache_dir / "320193_subs.json").write_text(json.dumps({
            "_cached_at": datetime.now(timezone.utc).isoformat(),
            "filings": [{"accessionNumber": acc, "filingDate": today, "primaryDocument": "doc4.xml"}],
        }))

        # Pre-seed filing XML cache
        acc_no_dash = acc.replace("-", "")
        xml = _form4_xml(reporter="Cook T", shares=750, price=200.0, txn_date=today)
        (cache_dir / f"{acc_no_dash}_filing.json").write_text(json.dumps({
            "_cached_at": datetime.now(timezone.utc).isoformat(),
            "xml_content": xml.decode("latin-1"),
        }))

        client = SECInsiderClient()
        txns = client.get_recent_purchases("AAPL", days_back=30)
        assert len(txns) == 1
        assert txns[0].reporter_name == "Cook T"
        assert txns[0].total_value == pytest.approx(150_000.0)

    def test_old_filings_excluded(self, tmp_path, monkeypatch):
        """Filings older than lookback_days are not fetched or returned."""
        mod, cache_dir = self._setup(tmp_path, monkeypatch)
        (cache_dir / "cik_map.json").write_text(json.dumps(
            {"AAPL": 320193, "_cached_at": datetime.now(timezone.utc).isoformat()}
        ))

        old_date = (date.today() - timedelta(days=45)).isoformat()
        (cache_dir / "320193_subs.json").write_text(json.dumps({
            "_cached_at": datetime.now(timezone.utc).isoformat(),
            "filings": [{"accessionNumber": "0001-26-01", "filingDate": old_date, "primaryDocument": "doc4.xml"}],
        }))

        client = SECInsiderClient()
        txns = client.get_recent_purchases("AAPL", days_back=30)
        assert txns == []


# ── InsiderBuyAgent signal tests ──────────────────────────────────────────────

class TestInsiderBuyAgentSignals:

    def test_buy_when_cluster_met(self):
        """2+ insiders each ≥$100K within 30d → BUY."""
        txns = [
            _make_transaction("AAPL", "Cook T",   is_officer=True,  total_value=150_000, days_ago=5),
            _make_transaction("AAPL", "Maestri L", is_officer=True, total_value=120_000, days_ago=8),
        ]
        agent = InsiderBuyAgent(client=_client_with_transactions(txns))
        sig = agent.generate_signal(_state(), {})
        assert sig.action == "BUY"
        assert sig.confidence == pytest.approx(0.60)
        assert sig.target_weight == pytest.approx(0.06)
        assert "InsiderBuy" in sig.reason

    def test_hold_when_only_one_insider(self):
        """Only 1 qualifying insider → HOLD (cluster not formed)."""
        txns = [_make_transaction("AAPL", "Cook T", total_value=500_000, days_ago=3)]
        agent = InsiderBuyAgent(client=_client_with_transactions(txns))
        sig = agent.generate_signal(_state(), {})
        assert sig.action == "HOLD"

    def test_hold_when_purchase_below_threshold(self):
        """2 insiders but each buys < $100K → HOLD."""
        txns = [
            _make_transaction("AAPL", "Insider A", total_value=50_000, days_ago=5),
            _make_transaction("AAPL", "Insider B", total_value=80_000, days_ago=6),
        ]
        agent = InsiderBuyAgent(client=_client_with_transactions(txns))
        sig = agent.generate_signal(_state(), {})
        assert sig.action == "HOLD"

    def test_hold_for_etf(self):
        """ETF (SPY) → no CIK → empty transactions → HOLD."""
        client = _client_with_transactions([])
        agent  = InsiderBuyAgent(client=client)
        sig = agent.generate_signal(_state("SPY"), {})
        assert sig.action == "HOLD"

    def test_confidence_scales_with_extra_insiders(self):
        """3 insiders (1 extra beyond min=2) → confidence += 0.08."""
        txns = [
            _make_transaction("AAPL", f"Insider {i}", total_value=150_000, days_ago=i + 1)
            for i in range(3)
        ]
        agent = InsiderBuyAgent(client=_client_with_transactions(txns))
        sig = agent.generate_signal(_state(), {})
        assert sig.action == "BUY"
        assert sig.confidence == pytest.approx(0.68)  # 0.60 + 1×0.08

    def test_confidence_capped_at_max(self):
        """10 insiders → confidence stays at 0.85 (hard cap)."""
        txns = [
            _make_transaction("AAPL", f"Insider {i}", total_value=150_000, days_ago=i + 1)
            for i in range(10)
        ]
        agent = InsiderBuyAgent(client=_client_with_transactions(txns))
        sig = agent.generate_signal(_state(), {})
        assert sig.action == "BUY"
        assert sig.confidence == pytest.approx(0.85)

    def test_duplicate_insider_counts_once(self):
        """Same insider buying twice counts as 1 distinct insider."""
        txns = [
            _make_transaction("AAPL", "Cook T", total_value=150_000, days_ago=3),
            _make_transaction("AAPL", "Cook T", total_value=200_000, days_ago=10),
        ]
        agent = InsiderBuyAgent(client=_client_with_transactions(txns))
        sig = agent.generate_signal(_state(), {})
        assert sig.action == "HOLD"   # only 1 distinct insider

    def test_meta_populated(self):
        """Signal meta contains all diagnostic fields."""
        txns = [
            _make_transaction("AAPL", "Cook T",    total_value=150_000, days_ago=5),
            _make_transaction("AAPL", "Maestri L", total_value=120_000, days_ago=8),
        ]
        agent = InsiderBuyAgent(client=_client_with_transactions(txns))
        sig = agent.generate_signal(_state(), {})
        assert "distinct_insiders" in sig.meta
        assert "total_amount"      in sig.meta
        assert "lookback_days"     in sig.meta
        assert sig.meta["distinct_insiders"] == 2

    def test_fetch_failure_returns_hold(self):
        """If the client raises an exception, the agent returns HOLD without crashing."""
        client = MagicMock(spec=SECInsiderClient)
        client.get_recent_purchases.side_effect = Exception("network error")
        agent = InsiderBuyAgent(client=client)
        sig = agent.generate_signal(_state(), {})
        assert sig.action == "HOLD"
        assert sig.agent_name == "InsiderBuyAgent"

    def test_custom_thresholds(self):
        """Custom min_purchase_amount and min_distinct_insiders are respected."""
        cfg = InsiderBuyConfig(min_distinct_insiders=3, min_purchase_amount=50_000)
        txns = [
            _make_transaction("AAPL", f"Insider {i}", total_value=60_000, days_ago=i + 1)
            for i in range(3)
        ]
        agent = InsiderBuyAgent(config=cfg, client=_client_with_transactions(txns))
        sig = agent.generate_signal(_state(), {})
        assert sig.action == "BUY"

    def test_director_counts_as_qualifying(self):
        """Directors (not officers) also qualify."""
        txns = [
            _make_transaction("AAPL", "Director A", is_officer=False, total_value=150_000, days_ago=3),
            _make_transaction("AAPL", "Director B", is_officer=False, total_value=120_000, days_ago=7),
        ]
        for t in txns:
            object.__setattr__(t, "is_director", True)
        agent = InsiderBuyAgent(client=_client_with_transactions(txns))
        sig = agent.generate_signal(_state(), {})
        assert sig.action == "BUY"
