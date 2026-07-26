# Edge Audit — MQC Arena

*Généré le 2026-07-26 — P0(d), mesure pure, zéro modification du système.*

---

## ⚠ Puissance statistique

**Minimum requis pour toute conclusion d'edge : 60 dates indépendantes.**

| Métrique | Valeur actuelle | Minimum requis | Statut |
|---|---|---|---|
| Dates de marché distinctes (H1) | 11 | 60 | 🔴 SOUS SEUIL |
| Fenêtres H5 non chevauchantes   | 7 | 60 | 🔴 SOUS SEUIL |

> **Statut global : SOUS SEUIL.** Aucune conclusion d'edge n'est rendue. Le pipeline est validé sur données réelles. Les résultats se rempliront aux prochains runs — revenir à ce rapport à ~60 dates de marché (environ 3 mois de runs quotidiens).

### Corrélation inter-actifs

Un run où BuffettAgent dit BUY sur 12 actifs le même jour représente **1 observation de marché**, pas 12. N_effectif ≪ N_lignes. Avec 13 runs sur 9 jours de bourse, les actifs se déplacent ensemble — un hit rate calculé sur cette fenêtre mesure la direction du marché de juillet 2026, pas l'edge des agents.

---

## Définition du succès

- **Seuil de matérialité** : μ = 0.30% (30 bps)
  - *Représente environ la moitié d'un aller-retour IBKR large-cap (commission + slippage estimé).*
- **BUY correct** : forward_return(H) > +0.30%
- **SELL correct** : forward_return(H) < −0.30%
- **HOLD correct** : |forward_return(H)| ≤ 0.30%
  - *Note : un marché immobile à ±30 bps sur 5 jours est rarissime pour des actions individuelles.*
  *Les résultats HOLD seront quasi-systématiquement 'échec' — ce n'est pas un bug.*

- **Anti-look-ahead** : forward return = log(close_{t+H} / close_t), t = close du jour de décision. Aucune donnée intra-journalière antérieure au signal n'est utilisée.

---

## Résultats par agent

### Horizon H1

| Agent | N signaux | N dates | Hit rate | IC 95% | Verdict |
|---|---|---|---|---|---|
| MacroAgent | 146 | 7 | 0.308% | [0.239, 0.387] | échantillon temporel insuffisant (7/60 dates) |
| BuffettAgent | 125 | 10 | 0.368% | [0.289, 0.455] | échantillon temporel insuffisant (10/60 dates) |
| TrendFollowingAgent | 39 | 9 | 0.359% | [0.227, 0.516] | échantillon temporel insuffisant (9/60 dates) |
| CitadelAgent | 25 | 6 | 0.16% | [0.064, 0.347] | échantillon temporel insuffisant (6/60 dates) |
| PairsTradingAgent | 7 | 4 | 0.571% | [0.25, 0.842] | échantillon temporel insuffisant (4/60 dates) |
| MeanReversionAgent | 1 | 1 | 0.0% | [0.0, 0.793] | échantillon temporel insuffisant (1/60 dates) |
| CrossSectionalMomentumAgent | 0 | 4 | — | [—, —] | forward return non disponible (4 signaux directionnels sans données prix) |
| DividendArbitrageAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |
| DummyHoldAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |
| EarningsSentimentAgent | 0 | 10 | — | [—, —] | forward return non disponible (10 signaux directionnels sans données prix) |
| InsiderBuyAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |
| VolatilityAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |

### Horizon H5

| Agent | N signaux | N dates | Hit rate | IC 95% | Verdict |
|---|---|---|---|---|---|
| MacroAgent | 146 | 7 | 0.253% | [0.19, 0.33] | échantillon temporel insuffisant (7/60 dates) |
| BuffettAgent | 125 | 10 | 0.288% | [0.216, 0.373] | échantillon temporel insuffisant (10/60 dates) |
| TrendFollowingAgent | 39 | 9 | 0.179% | [0.09, 0.327] | échantillon temporel insuffisant (9/60 dates) |
| CitadelAgent | 25 | 6 | 0.12% | [0.042, 0.3] | échantillon temporel insuffisant (6/60 dates) |
| PairsTradingAgent | 7 | 4 | 0.571% | [0.25, 0.842] | échantillon temporel insuffisant (4/60 dates) |
| MeanReversionAgent | 1 | 1 | 0.0% | [0.0, 0.793] | échantillon temporel insuffisant (1/60 dates) |
| CrossSectionalMomentumAgent | 0 | 4 | — | [—, —] | forward return non disponible (4 signaux directionnels sans données prix) |
| DividendArbitrageAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |
| DummyHoldAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |
| EarningsSentimentAgent | 0 | 10 | — | [—, —] | forward return non disponible (10 signaux directionnels sans données prix) |
| InsiderBuyAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |
| VolatilityAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |

### Horizon H20

| Agent | N signaux | N dates | Hit rate | IC 95% | Verdict |
|---|---|---|---|---|---|
| MacroAgent | 132 | 6 | 0.462% | [0.379, 0.547] | échantillon temporel insuffisant (6/60 dates) |
| BuffettAgent | 114 | 9 | 0.544% | [0.452, 0.632] | échantillon temporel insuffisant (9/60 dates) |
| TrendFollowingAgent | 37 | 8 | 0.459% | [0.31, 0.616] | échantillon temporel insuffisant (8/60 dates) |
| CitadelAgent | 24 | 5 | 0.375% | [0.212, 0.573] | échantillon temporel insuffisant (5/60 dates) |
| PairsTradingAgent | 7 | 4 | 0.571% | [0.25, 0.842] | échantillon temporel insuffisant (4/60 dates) |
| MeanReversionAgent | 1 | 1 | 0.0% | [0.0, 0.793] | échantillon temporel insuffisant (1/60 dates) |
| CrossSectionalMomentumAgent | 0 | 4 | — | [—, —] | forward return non disponible (4 signaux directionnels sans données prix) |
| DividendArbitrageAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |
| DummyHoldAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |
| EarningsSentimentAgent | 0 | 10 | — | [—, —] | forward return non disponible (10 signaux directionnels sans données prix) |
| InsiderBuyAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |
| VolatilityAgent | 0 | 0 | — | [—, —] | aucun signal directionnel émis |

---

## Courbes de fiabilité

*Confidence émise (abscisse) vs taux de succès directionnel réalisé (ordonnée).*

*Un agent calibré a une courbe croissante. Flat ou aléatoire = pas d'edge.*


### CitadelAgent — H1

![CitadelAgent_H1_reliability](docs/charts/CitadelAgent_H1_reliability.png)


### CitadelAgent — H5

![CitadelAgent_H5_reliability](docs/charts/CitadelAgent_H5_reliability.png)


### PairsTradingAgent — H1

![PairsTradingAgent_H1_reliability](docs/charts/PairsTradingAgent_H1_reliability.png)


### PairsTradingAgent — H5

![PairsTradingAgent_H5_reliability](docs/charts/PairsTradingAgent_H5_reliability.png)

---

## Verdicts par agent

- **BuffettAgent** : échantillon temporel insuffisant (10/60 dates)
- **CitadelAgent** : échantillon temporel insuffisant (6/60 dates)
- **CrossSectionalMomentumAgent** : forward return non disponible (4 signaux directionnels sans données prix)
- **DividendArbitrageAgent** : aucun signal directionnel émis
- **DummyHoldAgent** : aucun signal directionnel émis
- **EarningsSentimentAgent** : forward return non disponible (10 signaux directionnels sans données prix)
- **InsiderBuyAgent** : aucun signal directionnel émis
- **MacroAgent** : échantillon temporel insuffisant (7/60 dates)
- **MeanReversionAgent** : échantillon temporel insuffisant (1/60 dates)
- **PairsTradingAgent** : échantillon temporel insuffisant (4/60 dates)
- **TrendFollowingAgent** : échantillon temporel insuffisant (9/60 dates)
- **VolatilityAgent** : aucun signal directionnel émis

---

## Quand revenir

Ce rapport devient actionnable à **60 dates de marché indépendantes**. Avec des runs quotidiens (GH Actions 9h35 ET), cela représente environ **3 mois** à partir du lancement effectif.

Relancer : `python -m src.analysis.edge_audit`
