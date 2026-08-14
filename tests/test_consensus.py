"""
Agrégation des avis d'agents (src/arena/consensus.py).

Le point de ces tests n'est pas que la fonction renvoie un nombre : c'est
qu'elle corrige les trois défauts mesurés de `selector.select_best` — le
gagnant qui emporte tout, la confiance auto-déclarée comme arbitre, et l'avis
isolé qui vaut une unanimité.
"""
from __future__ import annotations

import math

import pytest

from src.agents.base import AgentSignal
from src.arena.consensus import (
    MIN_CONVICTION, NON_VOTANTS, ConsensusSignal, aggregate, equal_weights,
)

AGENTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]


def _sig(agent, action, conf=0.8, weight=0.10, symbol="AAPL"):
    return AgentSignal(agent_name=agent, symbol=symbol, action=action,
                       confidence=conf, target_weight=weight)


def _tous(action_par_agent, symbol="AAPL"):
    return [_sig(a, act, symbol=symbol) for a, act in action_par_agent.items()]


# ── Défaut 1 : le gagnant emportait tout ──────────────────────────────────────

class TestAgregationPlutotQueSelection:
    def test_une_voix_contre_dix_silences_ne_fait_pas_un_achat_plein(self):
        """
        Le cas qui condamnait `select_best` : un seul BUY parmi dix HOLD
        produisait un achat à pleine taille. Ici la direction est bien BUY —
        les HOLD s'abstiennent — mais la TAILLE est réduite, parce qu'un avis
        sur onze n'est pas un consensus.
        """
        votes = {a: "HOLD" for a in AGENTS}
        votes["A"] = "BUY"
        c = aggregate(_tous(votes), base_weight=0.10)
        assert c is not None and c.action == "BUY"
        assert c.score == pytest.approx(1.0)
        assert c.target_weight < 0.10 * 0.35, \
            "un avis isolé ne doit pas obtenir la taille d'une unanimité"

    def test_unanimite_obtient_la_taille_pleine(self):
        c = aggregate(_tous({a: "BUY" for a in AGENTS}), base_weight=0.10)
        assert c is not None
        assert c.conviction == pytest.approx(1.0)
        assert c.target_weight == pytest.approx(0.10)

    def test_un_desaccord_franc_ne_declenche_rien(self):
        """Cinq acheteurs contre cinq vendeurs : le système doit s'abstenir."""
        votes = {a: "HOLD" for a in AGENTS}
        for a in AGENTS[:5]:
            votes[a] = "BUY"
        for a in AGENTS[5:10]:
            votes[a] = "SELL"
        assert aggregate(_tous(votes)) is None

    def test_une_majorite_nette_passe_mais_amortie(self):
        votes = {a: "HOLD" for a in AGENTS}
        for a in AGENTS[:6]:
            votes[a] = "BUY"
        votes[AGENTS[6]] = "SELL"
        c = aggregate(_tous(votes), base_weight=0.10)
        assert c is not None and c.action == "BUY"
        # 6 pour, 1 contre sur 7 qui parlent → score = 5/7
        assert c.score == pytest.approx(5 / 7)
        assert c.conviction < abs(c.score), "la participation partielle amortit"


# ── Défaut 2 : la confiance auto-déclarée arbitrait ───────────────────────────

class TestLaConfianceNArbitrePlus:
    def test_un_agent_qui_se_declare_sur_de_lui_ne_pese_pas_plus(self):
        """
        Le cœur du problème. BuffettAgent remportait 128 décisions sur 200
        parce que `confidence=0.9` est écrit dans son fichier, alors que sa
        calibration mesurée est plate (53.0 % à 0.70, 53.5 % à 0.90).

        Ici, un agent qui se déclare certain et demande une grosse position ne
        pèse pas davantage qu'un agent modeste : seule sa DIRECTION compte.
        """
        criard = [_sig("A", "BUY", conf=0.99, weight=0.50)]
        criard += [_sig(a, "SELL", conf=0.51, weight=0.01) for a in AGENTS[1:4]]
        c = aggregate(criard)
        assert c is not None and c.action == "SELL", \
            "trois avis modestes doivent l'emporter sur un avis tonitruant"

    def test_le_resultat_ne_depend_pas_des_valeurs_de_confiance(self):
        votes = {a: "BUY" for a in AGENTS[:5]}
        votes.update({a: "SELL" for a in AGENTS[5:7]})
        faible = [_sig(a, act, conf=0.51) for a, act in votes.items()]
        forte = [_sig(a, act, conf=0.99) for a, act in votes.items()]
        assert aggregate(faible).score == pytest.approx(aggregate(forte).score)
        assert aggregate(faible).target_weight == \
               pytest.approx(aggregate(forte).target_weight)

    def test_une_majorite_trop_courte_ne_declenche_rien(self):
        """
        Quatre voix contre trois donne une conviction de 0.14, sous le seuil.
        C'est voulu : une majorité d'une voix sur un vote serré n'est pas une
        information, et `select_best` aurait pourtant produit un ordre plein.
        """
        votes = {a: "BUY" for a in AGENTS[:4]}
        votes.update({a: "SELL" for a in AGENTS[4:7]})
        c = aggregate(_tous(votes))
        assert c is None

    def test_les_poids_egaux_sont_bien_egaux(self):
        w = equal_weights(AGENTS)
        assert len(w) == len(AGENTS)
        assert all(v == pytest.approx(1 / len(AGENTS)) for v in w.values())
        assert sum(w.values()) == pytest.approx(1.0)

    def test_un_poids_mesure_peut_remplacer_l_egalite(self):
        """Le mécanisme doit accepter une pondération issue de la mesure."""
        votes = {"A": "BUY", "B": "SELL", "C": "SELL"}
        mesure = {"A": 0.90, "B": 0.05, "C": 0.05}
        c = aggregate(_tous(votes), weights=mesure)
        assert c is not None and c.action == "BUY", \
            "un agent lourdement pondéré par la MESURE doit l'emporter"


# ── Défaut 3 : HOLD, abstention et agents muets ───────────────────────────────

class TestAbstention:
    def test_hold_ne_compte_pas_comme_vote_contre(self):
        """
        Décision du 2026-08-14. Trois agents sont structurellement muets sur
        des mégacaps : compter leur silence comme une opposition noierait
        InsiderBuy le jour où il détecte réellement un achat d'initié.
        """
        deux = {"A": "BUY", "B": "BUY"}
        avec_muets = dict(deux, **{a: "HOLD" for a in AGENTS[2:]})
        assert aggregate(_tous(deux)).score == \
               pytest.approx(aggregate(_tous(avec_muets)).score), \
               "les HOLD ne doivent pas déplacer la direction"

    def test_mais_les_muets_reduisent_bien_la_taille(self):
        """L'abstention n'affecte pas la direction, mais bien la participation."""
        deux = aggregate(_tous({"A": "BUY", "B": "BUY"}))
        avec_muets = aggregate(_tous(
            dict({"A": "BUY", "B": "BUY"}, **{a: "HOLD" for a in AGENTS[2:]})))
        assert avec_muets.target_weight < deux.target_weight

    def test_tout_le_monde_se_tait_ne_produit_rien(self):
        assert aggregate(_tous({a: "HOLD" for a in AGENTS})) is None

    def test_liste_vide(self):
        assert aggregate([]) is None

    def test_le_temoin_ne_vote_pas(self):
        """DummyHoldAgent est un point de comparaison, pas une opinion."""
        assert "DummyHoldAgent" in NON_VOTANTS
        sigs = [_sig("DummyHoldAgent", "BUY"), _sig("A", "SELL")]
        c = aggregate(sigs)
        assert c is not None and c.action == "SELL"
        assert c.n_eligible == 1


# ── Dimensionnement ───────────────────────────────────────────────────────────

class TestDimensionnement:
    def test_la_conviction_croit_avec_le_nombre_de_voix_concordantes(self):
        tailles = []
        for n in (1, 3, 6, 11):
            votes = {a: "HOLD" for a in AGENTS}
            for a in AGENTS[:n]:
                votes[a] = "BUY"
            tailles.append(aggregate(_tous(votes)).target_weight)
        assert tailles == sorted(tailles), "plus d'accord ⇒ position plus grande"

    def test_amortissement_en_racine_de_la_participation(self):
        votes = {a: "HOLD" for a in AGENTS}
        for a in AGENTS[:4]:
            votes[a] = "BUY"
        c = aggregate(_tous(votes))
        assert c.conviction == pytest.approx(math.sqrt(4 / 11))

    def test_seuil_de_conviction_minimal(self):
        """Un avis trop divisé ou trop isolé ne doit pas produire d'ordre."""
        votes = {a: "HOLD" for a in AGENTS}
        votes["A"], votes["B"], votes["C"] = "BUY", "SELL", "SELL"
        c = aggregate(_tous(votes), min_conviction=0.90)
        assert c is None

    def test_la_taille_ne_depasse_jamais_la_base(self):
        for n in range(1, 12):
            votes = {a: "HOLD" for a in AGENTS}
            for a in AGENTS[:n]:
                votes[a] = "BUY"
            c = aggregate(_tous(votes), base_weight=0.10)
            if c is not None:
                assert c.target_weight <= 0.10 + 1e-12


# ── Traçabilité ───────────────────────────────────────────────────────────────

class TestTracabilite:
    def test_le_detail_du_vote_est_conserve(self):
        votes = {"A": "BUY", "B": "BUY", "C": "SELL", "D": "HOLD"}
        c = aggregate(_tous(votes))
        assert c.votes == votes, "on doit pouvoir rejouer la décision"
        assert c.n_speaking == 3 and c.n_eligible == 4

    def test_le_resume_nomme_les_agents_qui_ont_porte_la_decision(self):
        c = aggregate(_tous({"A": "BUY", "B": "BUY", "C": "SELL"}))
        assert "A" in c.reason and "B" in c.reason
        assert "AAPL" in c.render()


# ── Journalisation en observation ─────────────────────────────────────────────

class TestJournalObservation:
    """
    Le mécanisme agrégé n'exécute rien : il écrit. Ces tests protègent la seule
    chose qui compte pour la comparaison à venir — que les lignes soient
    écrites, y compris quand il s'abstient.
    """

    def test_une_abstention_est_journalisee_aussi(self, tmp_path, monkeypatch):
        """
        Sans cette ligne, la comparaison ne retiendrait que les séances où le
        nouveau mécanisme agit, et paraîtrait systématiquement plus décisif
        qu'il ne l'est.
        """
        import pandas as pd
        from src.execution import logger as lg
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(lg, "CONSENSUS_PATH", str(tmp_path / "c.csv"))

        lg.log_consensus(None, symbol="AAPL", plan_id="p1", regime="bull",
                         winner_agent="BuffettAgent", winner_action="BUY")
        d = pd.read_csv(tmp_path / "c.csv")
        assert len(d) == 1
        assert d.consensus_action.iloc[0] == "NONE"
        assert bool(d.diverge.iloc[0]) is True

    def test_les_lignes_s_accumulent(self, tmp_path, monkeypatch):
        import pandas as pd
        from src.execution import logger as lg
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(lg, "CONSENSUS_PATH", str(tmp_path / "c.csv"))

        c = aggregate(_tous({"A": "BUY", "B": "BUY", "C": "BUY"}))
        for i in range(3):
            lg.log_consensus(c, symbol="AAPL", plan_id=f"p{i}", regime="bull",
                             winner_agent="A", winner_action="BUY")
        d = pd.read_csv(tmp_path / "c.csv")
        assert len(d) == 3
        assert (d.diverge == False).all()          # noqa: E712
        assert d.n_speaking.iloc[0] == 3
