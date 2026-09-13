"""
Couche marché (2026-09-13) : collecte du S&P 500, vue pour les agents, flux
Form 4 de la SEC, agents d'univers en observation.

Ce qui est verrouillé ici, dans l'ordre où ça ferait le plus de dégâts :
- une collecte ratée n'écrase jamais la bonne ;
- un agent ne voit rien après sa date ;
- un achat déclaré par une société sur une autre n'est pas attribué au déclarant ;
- les agents d'univers gardent exactement les seuils des agents historiques ;
- une panne d'agent n'arrête pas les autres et ne se confond pas avec un jour calme.
"""
import urllib.error
from datetime import date

import numpy as np
import pandas as pd

from src.agents.base import AgentSignal
from src.agents.insider_cluster import InsiderClusterAgent
from src.agents.pairs_universe import PairsUniverseAgent
from src.agents.universe_base import UniverseAgent
from src.data.form4_feed import Form4Feed, extract_ownership_xml, parse_daily_index
from src.data.prices import YFinanceSource
from src.data.sec_insider import InsiderTransaction
from src.data.universe import Constituent, UniverseSnapshot
from src.market import collect as collect_mod
from src.market import observe
from src.market.view import MarketView, load_market_view

AS_OF = date(2026, 9, 11)


def _ohlcv(close, start="2023-01-02") -> pd.DataFrame:
    c = pd.Series(np.asarray(close, dtype=float),
                  index=pd.bdate_range(start, periods=len(close)))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                         "Close": c, "Volume": 1e6})


def _members(rows) -> pd.DataFrame:
    return pd.DataFrame([{"ticker": t, "sector": "S", "sub_industry": sub, "cik": cik}
                         for t, sub, cik in rows]).set_index("ticker")


def _view(ohlcv, rows, as_of=AS_OF) -> MarketView:
    return MarketView(as_of=as_of, members=_members(rows), ohlcv=ohlcv)


# ── Source de prix ────────────────────────────────────────────────────────────

def test_source_par_lots_ticker_absent_non_invente():
    calls = []

    def fake(lot, **kw):
        calls.append(list(lot))
        return pd.concat({t: _ohlcv(np.linspace(10, 20, 100)) for t in lot if t != "XXX"}, axis=1)

    out = YFinanceSource(chunk=2, pause=0, downloader=fake).daily(["A", "B", "XXX", "C"])
    assert calls == [["A", "B"], ["XXX", "C"]]
    assert set(out) == {"A", "B", "C"}


def test_source_serie_trop_courte_ecartee():
    fake = lambda lot, **kw: pd.concat({"A": _ohlcv(np.linspace(10, 20, 30))}, axis=1)
    assert YFinanceSource(pause=0, downloader=fake).daily(["A"]) == {}


# ── Vue du marché ─────────────────────────────────────────────────────────────

def test_until_ne_montre_rien_apres_la_date():
    v = _view({t: _ohlcv(np.linspace(10, 20, 300)) for t in "AB"},
              [("A", "Banks", 1), ("B", "Banks", 2)])
    d = v.ohlcv["A"].index[99]
    u = v.until(d)
    assert u.as_of == d.date()
    assert u.close().index.max() == d
    assert all(df.index[-1] <= d for df in u.ohlcv.values())


def test_pairs_d_une_meme_sous_industrie():
    v = _view({t: _ohlcv(np.linspace(10, 20, 300)) for t in "ABC"},
              [("A", "Banks", 1), ("B", "Banks", 2), ("C", "Oil", 3)])
    assert v.peers("A") == ["B"]
    assert v.peers("C") == []


# ── Collecte ──────────────────────────────────────────────────────────────────

class FakeSource:
    name = "fake"

    def __init__(self, data):
        self.data = data

    def daily(self, tickers, period="3y"):
        return {t: self.data[t] for t in tickers if t in self.data}


def _patch_universe(monkeypatch, tickers, sub="Banks"):
    members = [Constituent(t, "S", sub, i) for i, t in enumerate(tickers, 1)]
    snap = UniverseSnapshot(as_of=AS_OF, tickers=list(tickers), revision_id=1,
                            revision_date="2026-09-10T00:00:00Z")
    monkeypatch.setattr(collect_mod, "constituents_at", lambda as_of: (snap, members))


def test_collecte_ecrit_une_vue_lisible(tmp_path, monkeypatch):
    _patch_universe(monkeypatch, "AB")
    rep = collect_mod.collect(FakeSource({t: _ohlcv(np.linspace(10, 20, 300)) for t in "AB"}),
                              as_of=AS_OF, root=tmp_path)
    assert rep.ok and rep.coverage == 1.0
    v = load_market_view(tmp_path)
    assert v.tickers == ["A", "B"]
    assert v.peers("A") == ["B"]


def test_collecte_ratee_conserve_la_precedente(tmp_path, monkeypatch):
    tickers = "ABCDEFGHIJ"
    _patch_universe(monkeypatch, tickers)
    full = {t: _ohlcv(np.linspace(10, 20, 300)) for t in tickers}
    assert collect_mod.collect(FakeSource(full), as_of=AS_OF, root=tmp_path).ok

    rep = collect_mod.collect(FakeSource({"A": full["A"]}), as_of=AS_OF, root=tmp_path)
    assert not rep.ok
    assert "conservée" in rep.message
    assert len(load_market_view(tmp_path).tickers) == 10
    assert (tmp_path / collect_mod.FAILED_FILE).exists()


def test_serie_corrompue_rejetee_et_chiffree(tmp_path, monkeypatch):
    tickers = "ABCDEFGHIJ"
    _patch_universe(monkeypatch, tickers)
    data = {t: _ohlcv(np.linspace(10, 20, 300)) for t in tickers}
    data["J"] = _ohlcv(np.tile([10.0, 100.0], 150))    # des centaines de sauts impossibles
    rep = collect_mod.collect(FakeSource(data), as_of=AS_OF, root=tmp_path)
    assert rep.ok and rep.n_prices == 9
    assert [r["symbol"] for r in rep.rejected] == ["J"]
    assert "J" not in load_market_view(tmp_path).tickers


# ── Flux Form 4 ───────────────────────────────────────────────────────────────

def _index(*lines) -> str:
    return ("Description:  Daily Index of EDGAR Dissemination Feed by Form Type\n\n"
            "Form Type   Company Name   CIK   Date Filed  File Name\n"
            + "-" * 80 + "\n" + "\n".join(lines) + "\n")


ACC = "0000000001-26-000001"
LINE_ISSUER = f"4                ACME CORP                  100     20260911    edgar/data/100/{ACC}.txt"
LINE_HOLDER = f"4                BIG HOLDER INC             200     20260911    edgar/data/200/{ACC}.txt"


def _form4(issuer_cik, owner="Jane Doe", code="P", shares=1000, price=150.0) -> bytes:
    return f"""<SEC-DOCUMENT>header
<XML>
<ownershipDocument>
<issuer><issuerCik>{issuer_cik:010d}</issuerCik><issuerTradingSymbol>X</issuerTradingSymbol></issuer>
<reportingOwner><reportingOwnerId><rptOwnerName>{owner}</rptOwnerName></reportingOwnerId>
<reportingOwnerRelationship><isOfficer>1</isOfficer><isDirector>0</isDirector></reportingOwnerRelationship>
</reportingOwner>
<nonDerivativeTable><nonDerivativeTransaction>
<transactionDate><value>2026-09-10</value></transactionDate>
<transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
<transactionAmounts><transactionShares><value>{shares}</value></transactionShares>
<transactionPricePerShare><value>{price}</value></transactionPricePerShare>
<transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
</nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>
</XML></SEC-DOCUMENT>""".encode()


class FakeSEC:
    def __init__(self, indexes, filings):
        self.indexes, self.filings, self.calls = indexes, filings, []

    def __call__(self, url):
        self.calls.append(url)
        if "daily-index" in url:
            day = url[-12:-4]
            if day in self.indexes:
                return self.indexes[day].encode()
        else:
            for acc, raw in self.filings.items():
                if url.endswith(f"{acc}.txt"):
                    return raw
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)


def test_index_quotidien_lu_par_la_droite():
    idx = _index(LINE_ISSUER, "10-K             ACME CORP                  100     20260911    edgar/data/100/x.txt")
    e = parse_daily_index(idx)
    assert [(x.form, x.company, x.cik) for x in e] == [("4", "ACME CORP", 100), ("10-K", "ACME CORP", 100)]
    assert e[0].accession == ACC


def test_document_xml_extrait_du_depot_complet():
    assert extract_ownership_xml(_form4(100)).startswith(b"<ownershipDocument")
    assert extract_ownership_xml(b"pas de document") is None


def test_achat_attribue_a_l_emetteur_pas_au_declarant(tmp_path):
    # Le déclarant (200) est lui aussi dans l'univers, et sa ligne vient en premier.
    for i, lines in enumerate([(LINE_HOLDER, LINE_ISSUER), (LINE_ISSUER, LINE_HOLDER)]):
        sec = FakeSEC({"20260911": _index(*lines)}, {ACC: _form4(100)})
        feed = Form4Feed(get=sec, cache_dir=tmp_path / f"ordre{i}", today=date(2026, 9, 13))
        txns = feed.purchases({100: "ACME", 200: "BIGH"}, AS_OF, AS_OF)
        assert [(t.ticker, t.reporter_name, t.total_value) for t in txns] == [("ACME", "Jane Doe", 150_000.0)]


def test_un_depot_n_est_telecharge_qu_une_fois(tmp_path):
    sec = FakeSEC({"20260911": _index(LINE_ISSUER)}, {ACC: _form4(100)})
    feed = Form4Feed(get=sec, cache_dir=tmp_path, today=date(2026, 9, 13))
    feed.purchases({100: "ACME"}, AS_OF, AS_OF)
    n = len(sec.calls)
    assert feed.purchases({100: "ACME"}, AS_OF, AS_OF)[0].ticker == "ACME"
    assert len(sec.calls) == n          # index passé et dépôt : tous deux en cache


def test_index_absent_recent_n_est_pas_fige(tmp_path):
    # Un index manquant d'hier est peut-être seulement en retard de publication.
    sec = FakeSEC({}, {})
    feed = Form4Feed(get=sec, cache_dir=tmp_path, today=date(2026, 9, 12))
    assert feed.purchases({100: "ACME"}, AS_OF, AS_OF) == []
    assert not (tmp_path / "index" / "20260911.json").exists()


# ── Agents d'univers ──────────────────────────────────────────────────────────

class StubFeed:
    failures = 0

    def __init__(self, txns):
        self.txns = txns

    def purchases(self, cik_to_ticker, start, end):
        self.args = (cik_to_ticker, start, end)
        return self.txns


def _txn(ticker, who, value):
    return InsiderTransaction(ticker, who, True, False, "2026-09-10", value / 100, 100.0, value, "P")


def test_insider_meme_regle_sur_tout_l_univers():
    v = _view({t: _ohlcv(np.linspace(10, 20, 300)) for t in "ABC"},
              [("A", "Banks", 1), ("B", "Banks", 2), ("C", "Oil", 3)])
    feed = StubFeed([_txn("A", "x", 150_000), _txn("A", "y", 120_000),
                     _txn("B", "z", 500_000),                       # un seul dirigeant
                     _txn("C", "u", 150_000), _txn("C", "w", 99_999)])  # sous le seuil
    props = InsiderClusterAgent(feed=feed).propose(v)
    assert [(p.symbol, p.action) for p in props] == [("A", "BUY")]
    assert feed.args[0] == {1: "A", 2: "B", 3: "C"}
    assert feed.args[2] == AS_OF


def _cointegrated_pair(seed=11, n=800, spike_sd=3.0):
    """
    Une paire cointégrée par construction : B marche au hasard, A suit B à un
    écart qui revient vers zéro (demi-vie de 6 séances).

    La graine n'est pas arbitraire, et c'est une information : sur 100 paires
    construites ainsi, les filtres de PairsTradingAgent n'en acceptent que 54
    — surtout parce que la demi-vie estimée passe sous 5 jours. Les filtres
    sont prudents par conception ; ce test vérifie la chaîne, pas leur puissance.
    """
    rng = np.random.default_rng(seed)
    log_b = np.log(50) + np.cumsum(rng.normal(0, 0.02, n))
    phi = 0.5 ** (1 / 6)
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = phi * s[i - 1] + rng.normal(0, 0.01)
    s[-1] = spike_sd * 0.01 / np.sqrt(1 - phi ** 2)   # A devient cher face à B
    return np.exp(0.2 + log_b + s), np.exp(log_b)


def test_paires_candidates_par_sous_industrie():
    v = _view({t: _ohlcv(np.linspace(10, 20, 300)) for t in "ABCD"},
              [("A", "Banks", 1), ("B", "Banks", 2), ("C", "Oil", 3), ("D", "Banks", 4)])
    assert PairsUniverseAgent().candidates(v) == [("A", "B"), ("A", "D"), ("B", "D")]


def test_paire_cointegree_detectee_dans_le_bon_sens():
    a, b = _cointegrated_pair()
    rng = np.random.default_rng(11)
    c = 40 * np.exp(np.cumsum(rng.normal(0, 0.015, 800)))
    d = 60 * np.exp(np.cumsum(rng.normal(0, 0.015, 800)))
    v = _view({"A": _ohlcv(a), "B": _ohlcv(b), "C": _ohlcv(c), "D": _ohlcv(d)},
              [("A", "Banks", 1), ("B", "Banks", 2), ("C", "Oil", 3), ("D", "Oil", 4)])
    agent = PairsUniverseAgent()
    props = agent.propose(v)
    assert agent.last_stats["validees"] == ["A/B"]
    (p,) = props
    assert p.meta["direction"] == "short_a_long_b"
    assert p.meta["zscore"] > 2


# ── Passage nocturne ──────────────────────────────────────────────────────────

class Boom(UniverseAgent):
    name = "Boom"

    def propose(self, view):
        raise RuntimeError("source injoignable")


class One(UniverseAgent):
    name = "One"

    def propose(self, view):
        return [AgentSignal("One", "A", "BUY", 0.7, 0.05, "raison", {"k": 1})]


def test_une_panne_n_arrete_pas_les_autres(tmp_path, monkeypatch):
    _patch_universe(monkeypatch, "AB")
    collect_mod.collect(FakeSource({t: _ohlcv(np.linspace(10, 20, 300)) for t in "AB"}),
                        as_of=AS_OF, root=tmp_path)
    out = tmp_path / "proposals.csv"
    summary = observe.run([Boom(), One()], collect_first=False, root=tmp_path,
                          proposals_path=out, notify=False)
    assert summary["pannes"] == ["Boom"]
    assert summary["agents"]["One"]["propositions"] == 1
    df = pd.read_csv(out)
    assert list(df["agent"]) == ["One"] and list(df["symbol"]) == ["A"]
