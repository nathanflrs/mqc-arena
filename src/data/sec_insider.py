# src/data/sec_insider.py
"""
SEC EDGAR Form 4 client — insider purchase tracker.

No API key required. Respects the SEC 10 req/s rate limit via 0.15s sleeps.

Flow per ticker:
  1. CIK lookup  → logs/insider_cache/cik_map.json          (7-day TTL)
  2. Submissions → logs/insider_cache/{cik}_subs.json       (24h TTL)
  3. Form 4 XML  → logs/insider_cache/{acc_no}_filing.json  (30-day TTL; filings never change)

Signal criteria (configurable via InsiderBuyConfig):
  - transactionCode == "P"  (open-market purchase, not award/grant)
  - acquiredDisposed  == "A" (acquired)
  - reporterRole in {officer, director}
  - totalValue >= min_purchase_amount
  - transactionDate within lookback_days
"""
from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

_CACHE_DIR      = Path("logs/insider_cache")
_CIK_TTL        = timedelta(days=7)
_SUBS_TTL       = timedelta(hours=24)
_FILING_TTL     = timedelta(days=30)
_RATE_SLEEP     = 0.15            # well under 10 req/s SEC limit
_USER_AGENT     = "MQC_ARENA research@milancapital.io"

_TICKERS_URL    = "https://www.sec.gov/files/company_tickers.json"
_SUBS_URL       = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_FILING_URL     = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


# ── Domain objects ─────────────────────────────────────────────────────────────

@dataclass
class InsiderTransaction:
    ticker: str
    reporter_name: str
    is_officer: bool
    is_director: bool
    transaction_date: str   # YYYY-MM-DD
    shares: float
    price_per_share: float
    total_value: float      # shares × price
    transaction_code: str   # "P" = open-market purchase


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(name: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / name


def _cache_valid(path: Path, ttl: timedelta) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
        cached_at = datetime.fromisoformat(data["_cached_at"])
        return datetime.now(timezone.utc) - cached_at < ttl
    except Exception:
        return False


def _cache_read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _cache_write(path: Path, payload: dict) -> None:
    payload["_cached_at"] = datetime.now(timezone.utc).isoformat()
    try:
        path.write_text(json.dumps(payload, default=str))
    except Exception as exc:
        logger.debug("sec_insider: cache write failed — %s", exc)


# ── HTTP helper ────────────────────────────────────────────────────────────────

def _sec_get(url: str) -> bytes:
    """GET a SEC EDGAR URL with proper User-Agent and rate-limit sleep."""
    time.sleep(_RATE_SLEEP)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


# ── XML parser for Form 4 ──────────────────────────────────────────────────────

def _parse_form4_xml(xml_bytes: bytes, ticker: str) -> List[InsiderTransaction]:
    """
    Parse a Form 4 XML document and return insider transactions.
    Handles both the old schema (direct text) and the newer <value> wrapper.
    """
    def _val(el: Optional[ET.Element]) -> str:
        if el is None:
            return ""
        # Some Form 4 XML wraps values in <value> children
        v = el.find("value")
        return (v.text or "").strip() if v is not None else (el.text or "").strip()

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        logger.debug("sec_insider: XML parse error — %s", exc)
        return []

    # Reporter identity
    owner_el       = root.find(".//reportingOwner")
    reporter_name  = _val(owner_el.find("reportingOwnerId/rptOwnerName")) if owner_el is not None else "Unknown"
    rel            = owner_el.find("reportingOwnerRelationship") if owner_el is not None else None
    is_officer     = _val(rel.find("isOfficer"))  == "1" if rel is not None else False
    is_director    = _val(rel.find("isDirector")) == "1" if rel is not None else False

    if not is_officer and not is_director:
        return []

    results: List[InsiderTransaction] = []

    for txn in root.findall(".//nonDerivativeTransaction"):
        code    = _val(txn.find("transactionCoding/transactionCode"))
        ad_code = _val(txn.find("transactionAmounts/transactionAcquiredDisposedCode"))
        if code != "P" or ad_code != "A":
            continue

        txn_date  = _val(txn.find("transactionDate"))
        shares_s  = _val(txn.find("transactionAmounts/transactionShares"))
        price_s   = _val(txn.find("transactionAmounts/transactionPricePerShare"))

        try:
            shares = float(shares_s)
            price  = float(price_s) if price_s else 0.0
        except ValueError:
            continue

        results.append(InsiderTransaction(
            ticker          = ticker,
            reporter_name   = reporter_name,
            is_officer      = is_officer,
            is_director     = is_director,
            transaction_date= txn_date[:10] if txn_date else "",
            shares          = shares,
            price_per_share = price,
            total_value     = shares * price,
            transaction_code= code,
        ))

    return results


# ── Client ─────────────────────────────────────────────────────────────────────

class SECInsiderClient:
    """Fetches and caches Form 4 insider purchase data from SEC EDGAR."""

    def __init__(self) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._cik_map: Optional[Dict[str, int]] = None

    # ── CIK lookup ─────────────────────────────────────────────────────────────

    def _load_cik_map(self) -> Dict[str, int]:
        if self._cik_map is not None:
            return self._cik_map
        p = _cache_path("cik_map.json")
        if _cache_valid(p, _CIK_TTL):
            data = _cache_read(p)
            if data:
                self._cik_map = {k: v for k, v in data.items() if k != "_cached_at"}
                return self._cik_map
        try:
            raw  = _sec_get(_TICKERS_URL)
            data = json.loads(raw)
            # SEC returns {"0": {"cik_str": 320193, "ticker": "AAPL", "title": ...}, ...}
            cik_map: Dict[str, int] = {}
            for entry in data.values():
                t = str(entry.get("ticker", "")).upper()
                c = int(entry.get("cik_str", 0))
                if t and c:
                    cik_map[t] = c
            _cache_write(p, dict(cik_map))
            self._cik_map = cik_map
        except Exception as exc:
            logger.warning("sec_insider: CIK map fetch failed — %s", exc)
            self._cik_map = {}
        return self._cik_map

    def _get_cik(self, ticker: str) -> Optional[int]:
        m = self._load_cik_map()
        # Handle aliases (BRK-B → BRKB, etc.)
        for candidate in [ticker, ticker.replace("-", ""), ticker.replace(".", "")]:
            if candidate in m:
                return m[candidate]
        return None

    # ── Submissions ────────────────────────────────────────────────────────────

    def _get_recent_form4_filings(self, cik: int) -> List[dict]:
        """
        Return list of dicts: {accessionNumber, filingDate, primaryDocument}
        for recent Form 4 filings of the given CIK.
        """
        p = _cache_path(f"{cik}_subs.json")
        if _cache_valid(p, _SUBS_TTL):
            data = _cache_read(p)
            if data:
                return data.get("filings", [])

        url = _SUBS_URL.format(cik=cik)
        try:
            raw  = _sec_get(url)
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("sec_insider: submissions fetch failed for CIK %d — %s", cik, exc)
            return []

        recent = data.get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        acc_nums   = recent.get("accessionNumber", [])
        dates      = recent.get("filingDate", [])
        primary    = recent.get("primaryDocument", [])

        filings = [
            {
                "accessionNumber": acc_nums[i],
                "filingDate":      dates[i],
                "primaryDocument": primary[i],
            }
            for i, form in enumerate(forms)
            if form == "4"
            and i < len(acc_nums) and i < len(dates) and i < len(primary)
        ]

        _cache_write(p, {"filings": filings})
        return filings

    # ── Filing XML ─────────────────────────────────────────────────────────────

    def _fetch_filing_xml(self, cik: int, filing: dict) -> Optional[bytes]:
        acc   = filing["accessionNumber"].replace("-", "")
        doc   = filing["primaryDocument"]
        p     = _cache_path(f"{acc}_filing.json")

        if _cache_valid(p, _FILING_TTL):
            data = _cache_read(p)
            if data and "xml_content" in data:
                return data["xml_content"].encode("latin-1")

        # Only attempt XML files — HTML primary docs are form wrappers
        if not doc.lower().endswith(".xml"):
            # Try replacing .htm with .xml as a common pattern
            doc_xml = doc.rsplit(".", 1)[0] + ".xml"
        else:
            doc_xml = doc

        url = _FILING_URL.format(cik=cik, acc=acc, doc=doc_xml)
        try:
            xml_bytes = _sec_get(url)
            # Sanity-check: must contain ownershipDocument tag
            if b"ownershipDocument" not in xml_bytes[:2000]:
                return None
            _cache_write(p, {"xml_content": xml_bytes.decode("latin-1", errors="replace")})
            return xml_bytes
        except Exception as exc:
            logger.debug("sec_insider: filing XML fetch failed %s — %s", acc, exc)
            return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_recent_purchases(
        self,
        ticker: str,
        days_back: int = 30,
    ) -> List[InsiderTransaction]:
        """
        Return open-market insider purchases (code P, acquired A) by
        officers/directors for *ticker* within the last *days_back* days.
        Returns an empty list on any fetch failure.
        """
        cik = self._get_cik(ticker)
        if cik is None:
            # ETFs (SPY, QQQ, GLD …) or tickers not in SEC universe
            logger.debug("sec_insider: no CIK for %s — likely an ETF", ticker)
            return []

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_back)).date()
        filings = self._get_recent_form4_filings(cik)

        results: List[InsiderTransaction] = []
        for filing in filings:
            try:
                filing_date = date.fromisoformat(filing["filingDate"])
            except ValueError:
                continue
            if filing_date < cutoff:
                break  # filings are newest-first in EDGAR submissions

            xml_bytes = self._fetch_filing_xml(cik, filing)
            if xml_bytes is None:
                continue
            txns = _parse_form4_xml(xml_bytes, ticker)
            results.extend(t for t in txns if t.transaction_date >= cutoff.isoformat())

        return results
