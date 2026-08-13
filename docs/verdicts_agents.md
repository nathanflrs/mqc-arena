# Verdicts par agent

*Registre des décisions prises sur chaque agent, une par séance de travail.*

Trois questions, un verdict écrit et daté. Pas de « on verra ».

1. **Qu'affirme l'agent exactement ?** — sa règle réelle, pas son nom.
2. **Cette affirmation tient-elle**, mesurée honnêtement ?
3. **Verdict** : garder, corriger, ou retirer.

> **Pourquoi trancher plutôt qu'« améliorer ».** Ajuster les seuils d'un agent
> jusqu'à ce que le backtest sourie fonctionne toujours — et uniquement sur les
> données qui ont servi à les ajuster. La trace en est visible dans le dépôt :
> `buffett.py` porte encore `near_high_252_threshold: float = 0.85  # assoupli
> de 0.90 -> 0.85`. Avec treize agents, c'est treize occasions de se tromper
> soi-même. Et surtout : on ne sort pas d'une absence d'avantage en réglant des
> seuils. Un avantage vient d'une information différente, pas d'un meilleur
> dosage de la même.

---

## 1. CrossSectionalMomentumAgent — **RETIRÉ** (2026-08-13)

### Ce qu'il affirme

Momentum transversal de Jegadeesh-Titman (1993) : classer l'univers par
rendement moyen sur 3, 6 et 12 mois — en sautant le dernier mois pour éviter
l'effet de retournement court terme — puis acheter le quartile supérieur.

C'est une stratégie académique sérieuse, l'une des mieux documentées de la
littérature. L'idée n'est pas en cause.

### Ce que dit la mesure

955 séances rejouées, bootstrap par dates, référence = taux de base
inconditionnel du même univers (`docs/agent_edge.md`) :

| Horizon | N | dates | taux | base | excès | IC 95 % |
|---|---|---|---|---|---|---|
| H1 | 3 816 | 954 | 46.4 % | 44.8 % | +1.6 % | [−0.4 %, +3.6 %] |
| H5 | 3 800 | 950 | 54.3 % | 54.3 % | +0.0 % | [−2.0 %, +2.0 %] |
| **H20** | 3 740 | 935 | 58.5 % | 61.5 % | **−3.0 %** | **[−5.0 %, −1.1 %]** |

À vingt jours, l'intervalle **exclut zéro par le bas**. Ce n'est pas « on ne
sait pas » : c'est le seul résultat statistiquement solide de tout l'audit
d'agents, et il est défavorable. L'agent fait moins bien que d'acheter au
hasard dans le même univers.

### Pourquoi il échoue — le diagnostic

Ce n'est pas la stratégie qui est mauvaise, c'est son implémentation ici. Deux
écarts avec ce que décrit Jegadeesh-Titman, et chacun suffirait :

**L'univers est trop petit.** JT classe plusieurs centaines de titres.
`WATCHLIST` en compte 14 : le quartile supérieur, ce sont **3 titres**. Un
classement sur 14 valeurs très corrélées ne sépare pas des gagnants de des
perdants, il départage du bruit.

**La jambe vendeuse manque.** Le momentum transversal tire son rendement de
l'**écart** entre le décile haut et le décile bas. Cet agent est long
seulement. On garde la moitié qui coûte — la concentration — en supprimant
celle qui rapporte.

Résultat : « acheter les 3 mégacaps qui ont le plus monté », sur un panier qui
monte ensemble. C'est de la poursuite de tendance concentrée, pas du momentum
transversal.

### Ce que ça coûtait

Au dernier run avant retrait (2026-08-13), l'agent pilotait **2 décisions sur
11 actives, soit 18 %** du portefeuille — pour un rendement mesuré
significativement négatif.

### Verdict

**Retiré de l'arène.** Le module reste dans le dépôt : il est utilisé par le
replay historique (`scripts/measure_agent_edge.py`, `system_backtest.py`), et
supprimer le code effacerait la trace de ce qui a été essayé.

**Ce qui pourrait le ressusciter**, et rien d'autre : un univers de plusieurs
centaines de titres **et** une jambe vendeuse. C'est-à-dire une stratégie
différente, à valider par une hypothèse écrite avant d'être codée — pas un
réglage de celle-ci.

---

*Prochains verdicts prévus : MeanReversion, CTATrend, la fusion
Buffett/Citadel/TrendFollowing, Macro et Volatility, DividendArb et InsiderBuy,
puis Pairs et EarningsSentiment.*
