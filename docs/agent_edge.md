# Edge par agent — Milan Capital

*Généré le 2026-08-14 — snapshot du 2026-08-02.*

## Méthode

L'arène est rejouée jour par jour sur **955 séances**, et le signal de chaque agent est collecté — pas seulement celui du gagnant. Aucun agent ne voit de données postérieures à sa date de décision.

Un signal est **correct** si le rendement forward dépasse ±0.30% dans le sens annoncé (seuil de matérialité : environ la moitié d'un aller-retour IBKR large-cap).

### Trois corrections successives

**Hypothèse nulle.** L'audit précédent (`docs/edge_audit.md`) testait contre une pièce équilibrée. Sur un marché haussier, un agent qui dit toujours BUY obtient bien plus de 50 % sans contenir la moindre information. La référence retenue ici est le **taux de base inconditionnel** de la même action sur le même univers et la même période. La colonne `excès` est ce que l'agent apporte au-delà.

**Corrélation entre actifs.** L'audit précédent calculait un intervalle de Wilson sur le nombre de signaux, alors que son propre texte reconnaissait qu'un run où un agent dit BUY sur 12 actifs vaut une observation et non douze. On regroupe donc par **date** avant de bootstrapper.

**Chevauchement des fenêtres — correction du 2026-08-14.** Regrouper par date ne suffisait pas. Un rendement à 20 jours mesuré chaque séance partage 19 jours avec le précédent : tirer les dates indépendamment revenait à compter la même information vingt fois, et divisait l'intervalle par la racine d'un effectif fictif. Les intervalles ci-dessous viennent d'un **bootstrap par blocs mobiles** de longueur H (2 000 tirages), qui préserve cette dépendance.

Deux résultats publiés auparavant n'y ont pas survécu : le momentum transversal « significativement perdant » à H20, et le rendement par signal du CTA à H20, seul intervalle strictement positif du document. Les deux traversent désormais zéro. **Aucun résultat n'a été renforcé par la correction** — c'est le signe attendu quand on cesse de surestimer sa propre information.

Seuil de puissance : 60 dates minimum. Atteint (955) — mais le bootstrap par blocs rappelle que ces séances ne valent pas autant d'observations indépendantes.

## Horizon H1

```
── Edge par agent — horizon H1 (succès = |rendement| > 0.30%) ──
Agent                              N  dates    taux    base    excès            IC 95%
MeanReversionAgent               210    148   52.4%   45.8%    +6.6%   [-0.7%, +14.0%]
CrossSectionalMomentumAgent     3551    954   46.8%   45.8%    +1.0%    [-1.1%, +3.1%]
BuffettAgent                    7680    902   44.8%   45.8%    -1.0%    [-2.9%, +0.7%]
CitadelAgent                    2694    773   44.1%   45.8%    -1.7%    [-4.1%, +0.9%]
TrendFollowingAgent             2520    759   43.6%   45.8%    -2.2%    [-4.8%, +0.5%]
DummyHoldAgent                     —      —       —       —        —                 —
```

## Horizon H5

```
── Edge par agent — horizon H5 (succès = |rendement| > 0.30%) ──
Agent                              N  dates    taux    base    excès            IC 95%
MeanReversionAgent               209    147   59.3%   54.3%    +5.1%   [-5.4%, +16.2%]
CitadelAgent                    2686    769   55.7%   54.3%    +1.4%    [-2.1%, +5.2%]
CrossSectionalMomentumAgent     3535    950   54.7%   54.3%    +0.5%    [-3.1%, +4.1%]
BuffettAgent                    7655    898   53.4%   54.3%    -0.9%    [-3.9%, +2.2%]
TrendFollowingAgent             2516    755   53.0%   54.3%    -1.3%    [-5.3%, +2.9%]
DummyHoldAgent                     —      —       —       —        —                 —
```

## Horizon H20

```
── Edge par agent — horizon H20 (succès = |rendement| > 0.30%) ──
Agent                              N  dates    taux    base    excès            IC 95%
MeanReversionAgent               206    145   68.9%   60.2%    +8.7%   [-1.5%, +21.3%]
TrendFollowingAgent             2495    740   60.1%   60.2%    -0.1%    [-8.0%, +7.7%]
BuffettAgent                    7537    883   59.2%   60.2%    -1.0%    [-7.4%, +5.0%]
CrossSectionalMomentumAgent     3475    935   58.9%   60.2%    -1.2%    [-8.1%, +6.0%]
CitadelAgent                    2655    755   58.0%   60.2%    -2.2%    [-9.4%, +5.0%]
DummyHoldAgent                     —      —       —       —        —                 —
```

## CTATrendAgent — univers séparé

CTA ne passe pas par l'arène : le runner l'appelle sur un chemin parallèle. Il était pour cette seule raison absent de toute mesure d'edge, alors qu'il porte la plus grosse allocation de risque du fonds — `max_gross_cta_pct = 60 %` du NAV, dans une catégorie explicitement exclue du budget net-long.

**Les tableaux ci-dessous ne sont pas comparables à ceux de l'arène.** Le taux de base est calculé sur l'univers CTA (SPY, QQQ, TLT, GLD, UUP, DBC), dont la dérive inconditionnelle n'a rien à voir avec celle des mégacaps. Replay sur 955 séances, portefeuille vide à chaque date : on mesure le signal directionnel émis, pas la gestion de position.

### Horizon H1

```
── Edge par agent — horizon H1 (succès = |rendement| > 0.30%) ──
Agent                              N  dates    taux    base    excès            IC 95%
CTATrendAgent                   3222    953   36.2%   36.0%    +0.2%    [-1.7%, +2.0%]

── Rendement par signal — horizon H1 (log, sens annoncé) ──
Agent                              N  dates    moyen   passif                IC 95%   skew   G/P
CTATrendAgent                   3222    953  +0.008%  +0.051%    [-0.039%, +0.055%]  -1.00  0.91
```

### Horizon H5

```
── Edge par agent — horizon H5 (succès = |rendement| > 0.30%) ──
Agent                              N  dates    taux    base    excès            IC 95%
CTATrendAgent                   3213    949   46.4%   46.1%    +0.3%    [-2.6%, +3.2%]

── Rendement par signal — horizon H5 (log, sens annoncé) ──
Agent                              N  dates    moyen   passif                IC 95%   skew   G/P
CTATrendAgent                   3213    949  +0.083%  +0.258%    [-0.056%, +0.235%]  -0.10  0.94
```

### Horizon H20

```
── Edge par agent — horizon H20 (succès = |rendement| > 0.30%) ──
Agent                              N  dates    taux    base    excès            IC 95%
CTATrendAgent                   3149    934   53.2%   52.4%    +0.8%    [-3.6%, +6.2%]

── Rendement par signal — horizon H20 (log, sens annoncé) ──
Agent                              N  dates    moyen   passif                IC 95%   skew   G/P
CTATrendAgent                   3149    934  +0.453%  +1.056%    [-0.016%, +1.034%]  -0.21  1.04
```

### Pourquoi deux tableaux

Le taux de réussite suppose que tous les succès se valent. C'est faux pour un suiveur de tendance, qui a classiquement raison 35-40 % du temps et gagne quand même parce que ses gains dépassent largement ses pertes. Juger un CTA au seul taux de réussite reviendrait à le condamner sur le mauvais critère — d'où la mesure du rendement par signal, avec le skew et le ratio gain/perte qui rendent la forme du gain visible.

La colonne `passif` est la référence honnête : le rendement d'un dollar simplement investi long sur le même univers et la même période. Une espérance positive mais inférieure à cette référence ne crée pas de valeur par unité d'exposition.

### Le vol targeting ne s'est jamais déclenché

Sur les 3,224 signaux directionnels du replay, **100 % sortent exactement au plafond de 15 %** (écart-type des poids : `2.8e-17`, la constante machine).

La cause est arithmétique. Le poids vaut `min(vol_target / vol, max_position)` = `min(0.10 / vol, 0.15)` : le plafond ne cède que si la volatilité annualisée dépasse **66.7 %**. Le maximum jamais observé sur les six ETF en cinq ans est 61.1 % (QQQ), et UUP plafonne à 15.1 %.

Conséquence : UUP (vol médiane 6.5 %) reçoit le même poids que QQQ (18.6 %), soit environ trois fois plus de risque sur le second — l'inverse exact de ce que le vol targeting est censé produire. La fonctionnalité annoncée dans le docstring de l'agent (« vol targeting, style Winton / Man AHL ») est inerte.

## Calibration de la confiance

Un agent calibré a un taux de succès croissant avec la confiance qu'il émet. Une courbe plate signifie que sa confiance ne porte aucune information : l'agent peut avoir un edge global tout en étant incapable de dire *quand* il est fiable — ce qui rend toute pondération par la confiance illusoire (le blending Kelly, notamment).

### BuffettAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 1545 | 783 | 0.70 | 53.0% |
| [0.85, 1.01) | 6110 | 889 | 0.90 | 53.5% |

### CitadelAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 153 | 138 | 0.70 | 53.6% |
| [0.75, 0.85) | 1536 | 668 | 0.80 | 56.0% |
| [0.85, 1.01) | 997 | 505 | 0.90 | 55.6% |

### CrossSectionalMomentumAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 1750 | 950 | 0.68 | 55.6% |
| [0.75, 0.85) | 1785 | 950 | 0.77 | 53.9% |

### MeanReversionAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 47 | 44 | 0.70 | 53.2% |
| [0.75, 0.85) | 113 | 88 | 0.80 | 59.3% |
| [0.85, 1.01) | 49 | 43 | 0.90 | 65.3% |

### TrendFollowingAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 62 | 60 | 0.69 | 61.3% |
| [0.75, 0.85) | 465 | 371 | 0.79 | 49.0% |
| [0.85, 1.01) | 1989 | 707 | 0.90 | 53.6% |

---

## Limites

- Les seuils des agents (RSI, ADX, fenêtres) ont été choisis en regardant ces mêmes marchés. Un edge mesuré ici est une **borne haute**, pas une promesse hors échantillon.
- 6 agents sont exclus du replay de l'arène (DividendArbitrageAgent, EarningsSentimentAgent, InsiderBuyAgent, MacroAgent, PairsTradingAgent, VolatilityAgent) : leurs données ne sont pas reconstituables point-in-time. CTATrendAgent, lui, ne lit que des prix d'ETF — son absence des mesures antérieures était un oubli, pas une impossibilité ; il est désormais mesuré ci-dessus.
- Prix ajustés : l'historique est réécrit rétroactivement à chaque dividende. Le snapshot fige les données, mais un `--refresh` change la base de mesure.
- L'univers de l'arène compte 11 titres. Les ETF en ont été retirés (un fonds de fonds ne teste pas la sélection de titres) et servent désormais de contexte seul : SPY, GLD. Les effectifs `N` ne sont donc pas comparables à ceux des versions antérieures de ce document.
- Onze titres ne suffisent à établir aucun edge. Ce document sert à **réfuter**, pas à valider : un agent qui n'y ressort pas est écarté, un agent qui y ressort demande une confirmation sur l'univers élargi (S&P 500, `logs/universe_snapshot`).