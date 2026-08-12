# Edge par agent — Milan Capital

*Généré le 2026-08-02 — snapshot du 2026-08-02.*

## Méthode

L'arène est rejouée jour par jour sur **955 séances**, et le signal de chaque agent est collecté — pas seulement celui du gagnant. Aucun agent ne voit de données postérieures à sa date de décision.

Un signal est **correct** si le rendement forward dépasse ±0.30% dans le sens annoncé (seuil de matérialité : environ la moitié d'un aller-retour IBKR large-cap).

### Deux corrections par rapport à `docs/edge_audit.md`

**Hypothèse nulle.** L'audit précédent testait contre une pièce équilibrée. Sur un marché haussier, un agent qui dit toujours BUY obtient bien plus de 50 % sans contenir la moindre information. La référence retenue ici est le **taux de base inconditionnel** de la même action sur le même univers et la même période. La colonne `excès` est ce que l'agent apporte au-delà.

**Intervalles de confiance.** L'audit précédent calculait un intervalle de Wilson sur le nombre de signaux, alors que son propre texte reconnaissait qu'un run où un agent dit BUY sur 12 actifs vaut une observation et non douze. On utilise ici un bootstrap sur les **dates** (2 000 tirages), qui conserve la corrélation entre actifs d'une même journée.

Seuil de puissance : 60 dates indépendantes minimum. Atteint (955).

## Horizon H1

```
── Edge par agent — horizon H1 (succès = |rendement| > 0.30%) ──
Agent                              N  dates    taux    base    excès            IC 95%
MeanReversionAgent               272    168   50.4%   44.8%    +5.6%   [-1.5%, +13.0%]
CrossSectionalMomentumAgent     3816    954   46.4%   44.8%    +1.6%    [-0.4%, +3.6%]
BuffettAgent                   10271    911   43.7%   44.8%    -1.0%    [-2.8%, +0.7%]
CitadelAgent                    3586    813   43.5%   44.8%    -1.3%    [-3.7%, +1.2%]
TrendFollowingAgent             3339    801   42.7%   44.8%    -2.1%    [-4.6%, +0.3%]
DummyHoldAgent                     —      —       —       —        —                 —
```

## Horizon H5

```
── Edge par agent — horizon H5 (succès = |rendement| > 0.30%) ──
Agent                              N  dates    taux    base    excès            IC 95%
MeanReversionAgent               268    165   59.0%   54.3%    +4.6%   [-4.1%, +13.0%]
CitadelAgent                    3578    809   55.6%   54.3%    +1.3%    [-1.0%, +3.6%]
CrossSectionalMomentumAgent     3800    950   54.3%   54.3%    +0.0%    [-2.0%, +2.0%]
BuffettAgent                   10238    907   53.5%   54.3%    -0.8%    [-2.7%, +1.0%]
TrendFollowingAgent             3335    797   53.4%   54.3%    -0.9%    [-3.3%, +1.5%]
DummyHoldAgent                     —      —       —       —        —                 —
```

## Horizon H20

```
── Edge par agent — horizon H20 (succès = |rendement| > 0.30%) ──
Agent                              N  dates    taux    base    excès            IC 95%
MeanReversionAgent               265    163   69.8%   61.5%    +8.3%   [-0.3%, +16.3%]
TrendFollowingAgent             3308    782   62.6%   61.5%    +1.1%    [-1.4%, +3.4%]
CitadelAgent                    3546    795   60.9%   61.5%    -0.6%    [-3.0%, +1.6%]
BuffettAgent                   10090    892   60.9%   61.5%    -0.7%    [-2.5%, +1.0%]
CrossSectionalMomentumAgent     3740    935   58.5%   61.5%    -3.0%    [-5.0%, -1.1%] ❌
DummyHoldAgent                     —      —       —       —        —                 —
```

## Calibration de la confiance

Un agent calibré a un taux de succès croissant avec la confiance qu'il émet. Une courbe plate signifie que sa confiance ne porte aucune information : l'agent peut avoir un edge global tout en étant incapable de dire *quand* il est fiable — ce qui rend toute pondération par la confiance illusoire (le blending Kelly, notamment).

### BuffettAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 1751 | 793 | 0.70 | 54.0% |
| [0.85, 1.01) | 8487 | 906 | 0.90 | 53.5% |

### CitadelAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 170 | 151 | 0.70 | 52.4% |
| [0.75, 0.85) | 1710 | 694 | 0.80 | 56.5% |
| [0.85, 1.01) | 1698 | 621 | 0.89 | 55.1% |

### CrossSectionalMomentumAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 1900 | 950 | 0.68 | 55.2% |
| [0.75, 0.85) | 1900 | 950 | 0.78 | 53.5% |

### MeanReversionAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 79 | 65 | 0.70 | 49.4% |
| [0.75, 0.85) | 128 | 91 | 0.80 | 59.4% |
| [0.85, 1.01) | 61 | 50 | 0.90 | 70.5% |

### TrendFollowingAgent

| tranche | N | dates | conf. moyenne | taux de succès |
|---|---|---|---|---|
| [0.65, 0.75) | 90 | 80 | 0.68 | 54.4% |
| [0.75, 0.85) | 708 | 478 | 0.79 | 50.3% |
| [0.85, 1.01) | 2537 | 746 | 0.90 | 54.2% |

---

## Limites

- Les seuils des agents (RSI, ADX, fenêtres) ont été choisis en regardant ces mêmes marchés. Un edge mesuré ici est une **borne haute**, pas une promesse hors échantillon.
- Six agents sont exclus du replay (DividendArbitrageAgent, EarningsSentimentAgent, InsiderBuyAgent, MacroAgent, PairsTradingAgent, VolatilityAgent) : leurs données ne sont pas reconstituables point-in-time.
- Prix ajustés : l'historique est réécrit rétroactivement à chaque dividende. Le snapshot fige les données, mais un `--refresh` change la base de mesure.