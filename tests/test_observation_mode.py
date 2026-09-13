"""
Mode observation (2026-09-13) : le fonds solde ce qu'il détient, et rien d'autre.

Le risque à verrouiller est double : qu'une position reste ouverte faute d'ordre
(quota, plafond unitaire), ou qu'un ordre d'agent passe malgré le mode.
"""
from src.execution.guards import build_execution_plan
from src.execution.planner import liquidation_plans

# Portefeuille réel du 2026-09-11 (journal du serveur).
POSITIONS = {"AAPL": 437.0, "JPM": 227.0, "GS": 39.0, "LLY": 18.0, "AMZN": 81.0, "META": 31.0}
PRICES = {"AAPL": 332.27, "JPM": 300.0, "GS": 1036.5, "LLY": 1130.0, "AMZN": 256.78, "META": 655.36}


def test_chaque_long_est_vendu_en_totalite():
    plans = liquidation_plans(POSITIONS, PRICES)
    assert {p.symbol for p in plans} == set(POSITIONS)
    for p in plans:
        assert p.action == "SELL"
        assert p.delta_qty == -POSITIONS[p.symbol]
        assert p.target_qty == 0.0
        assert p.confidence == 1.0


def test_un_short_est_rachete():
    (p,) = liquidation_plans({"XOM": -50.0}, {"XOM": 110.0})
    assert p.action == "BUY"
    assert p.delta_qty == 50.0


def test_position_nulle_ignoree():
    assert liquidation_plans({"AAPL": 0.0}, {"AAPL": 332.0}) == []


def test_sans_prix_la_position_reste_en_place():
    # Liquider à l'aveugle serait pire qu'attendre le run suivant.
    plans = liquidation_plans({"AAPL": 10.0, "JPM": 5.0, "GS": 3.0},
                              {"AAPL": float("nan"), "GS": 0.0})
    assert plans == []


def test_la_garde_ne_rationne_aucune_liquidation():
    # MAX_ORDERS_PER_RUN=1 et un plafond de 2 % du NetLiq : AAPL pèse ~145 k$,
    # sept fois le plafond. Une liquidation rognée laisserait le fonds exposé.
    ex = build_execution_plan(
        liquidation_plans(POSITIONS, PRICES),
        net_liquidation=1_027_051.81,
        max_notional_pct=0.02,
        max_orders=1,
        limit_buffer_bps=10,
    )
    assert ex.n_dropped == 0 and ex.n_resized == 0
    assert {c.plan.symbol: c.qty for c in ex.candidates} == {s: int(q) for s, q in POSITIONS.items()}
    assert all(c.side == "SELL" and c.risk_reducing for c in ex.candidates)
