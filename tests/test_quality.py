"""
Détection des séries de prix corrompues (src/data/quality.py).

Le module est né d'un incident précis, le 2026-08-14 : le test hors échantillon
de MeanReversion affichait un skew de +43,6, et 16 observations d'un seul
ticker (TIE, +758 % de rendement journalier) portaient la moitié du rendement
moyen.

Ces tests protègent les deux propriétés qui comptent :
  - une série corrompue doit être écartée ;
  - une action réellement performante ne doit PAS l'être. Le premier critère
    essayé — l'amplitude max/min — retirait NVDA sur 2010-2019, dont la hausse
    d'un facteur mille est authentique. Un filtre qui supprime des données
    correctes est pire que pas de filtre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.quality import (
    JUMP_THRESHOLD, MAX_JUMPS, assess, filter_universe, render_rejects,
)


def serie(valeurs) -> pd.Series:
    idx = pd.date_range("2015-01-01", periods=len(valeurs), freq="B")
    return pd.Series(valeurs, index=idx)


def croissance(n=500, depart=10.0, taux=0.002) -> pd.Series:
    """Une action qui monte régulièrement, sans aucun saut aberrant."""
    return serie([depart * (1 + taux) ** i for i in range(n)])


class TestCorruption:

    def test_series_with_many_impossible_jumps_is_rejected(self):
        """Le cas TIE : 197 sauts de plus de ±50 % en dix ans."""
        v = croissance(300).tolist()
        for i in range(20, 300, 10):          # un saut toutes les dix séances
            v[i] = v[i] * 5
        q = assess("TIE", serie(v))
        assert not q.ok
        assert "sauts" in q.reason

    def test_isolated_jump_with_absurd_range_is_rejected(self):
        """
        Le cas PARA : seulement 4 sauts, mais une amplitude de 68 614×.
        Un saut isolé peut être réel ; conjugué à une amplitude impossible,
        il trahit un ajustement cassé.
        """
        v = croissance(400, depart=1.0, taux=0.0).tolist()
        v[200] = 0.001                         # effondrement impossible
        v[201] = 5000.0                        # et remontée absurde
        q = assess("PARA", serie(v))
        assert not q.ok
        assert "amplitude" in q.reason

    def test_too_short_history_is_rejected(self):
        assert not assess("X", serie([1, 2, 3])).ok


class TestFauxPositifs:

    def test_a_thousandfold_riser_without_jumps_is_kept(self):
        """
        NVDA sur 2010-2019 : amplitude ~1 000×, zéro saut aberrant. C'est le
        contre-exemple qui a fait rejeter le critère d'amplitude seule.
        """
        q = assess("NVDA", croissance(2500, depart=1.0, taux=0.003))
        assert q.price_range > 1000
        assert q.ok, "une hausse réelle sur dix ans n'est pas une corruption"

    def test_one_genuine_crash_is_kept(self):
        """
        Un titre peut perdre 60 % en une séance — résultats catastrophiques,
        échec d'essai clinique. Un événement isolé n'est pas un défaut.
        """
        v = croissance(500).tolist()
        v[250] = v[250] * 0.4
        q = assess("BIOTECH", serie(v))
        assert q.n_jumps >= 1
        assert q.ok

    def test_an_isolated_spike_counts_as_two_jumps(self):
        """
        Propriété du critère, à connaître avant de régler le seuil : un pic
        d'une séance produit DEUX sauts — la montée puis le retour. MAX_JUMPS=5
        tolère donc environ deux pics aller-retour, ou cinq mouvements
        durables, sur toute la période.
        """
        v = croissance(600).tolist()
        v[100] = v[100] * 2
        assert assess("UN_PIC", serie(v)).n_jumps == 2

    def test_two_spikes_stay_under_the_limit(self):
        v = croissance(600).tolist()
        for i in (100, 300):
            v[i] = v[i] * 2
        q = assess("VOLATILE", serie(v))
        assert q.n_jumps == 4 <= MAX_JUMPS
        assert q.ok


class TestUnivers:

    def test_rejects_are_returned_not_swallowed(self):
        """
        Une exclusion silencieuse de données est aussi dangereuse qu'une donnée
        fausse : les rejets doivent remonter jusqu'au rapport.
        """
        v = croissance(300).tolist()
        for i in range(20, 300, 10):
            v[i] = v[i] * 5
        data = {"BON": pd.DataFrame({"Close": croissance(300)}),
                "CASSE": pd.DataFrame({"Close": serie(v)})}
        gardees, rejets = filter_universe(data)
        assert set(gardees) == {"BON"}
        assert len(rejets) == 1 and rejets[0].symbol == "CASSE"

    def test_missing_column_is_a_rejection_not_a_crash(self):
        data = {"X": pd.DataFrame({"Open": [1, 2, 3]})}
        gardees, rejets = filter_universe(data)
        assert not gardees and rejets[0].reason.startswith("colonne")

    def test_report_names_the_worst_offenders_first(self):
        v = croissance(300).tolist()
        for i in range(20, 300, 10):
            v[i] = v[i] * 5
        w = croissance(300).tolist()
        for i in range(20, 300, 40):
            w[i] = w[i] * 5
        _, rejets = filter_universe({
            "PIRE": pd.DataFrame({"Close": serie(v)}),
            "MOINS": pd.DataFrame({"Close": serie(w)}),
        })
        txt = render_rejects(rejets)
        assert txt.index("PIRE") < txt.index("MOINS")

    def test_empty_rejects_says_so(self):
        assert "aucune" in render_rejects([])
