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

## 2. MeanReversionAgent — **CONSERVÉ, sous condition** (2026-08-13)

### Ce qu'il affirme

Acheter après une baisse marquée, en pariant sur le rebond. Trois conditions
simultanées :

- RSI 14 jours < 35 (< 30 en régime baissier)
- prix sous la bande de Bollinger basse (20 jours, 2 écarts-types)
- volume du jour > 1,2 × la moyenne 20 jours

Le docstring invoque Renaissance Technologies. C'est une décoration : la règle
est un croisement RSI-Bollinger de manuel. Mais contrairement à
`BuffettAgent`, elle fait bien ce qu'elle annonce — acheter des baisses.

### Ce que dit la mesure

| Horizon | excès vs base | IC 95 % | signaux / dates |
|---|---|---|---|
| H1 | **+5.6 %** | [−1.5 %, +13.0 %] | 272 / 168 |
| H5 | **+4.6 %** | [−4.1 %, +13.0 %] | — |
| H20 | **+8.3 %** | [−0.3 %, +16.3 %] | 265 / 163 |

**Meilleur excès des douze agents aux trois horizons.** Et pourtant aucun
intervalle n'exclut zéro — à H20 il s'en faut de 0,3 point.

Ce n'est pas la même situation que CrossSectionalMomentum, dont l'intervalle
excluait zéro *par le bas*. Ici on ne sait pas, et la raison est identifiée.

### Pourquoi on ne sait pas : il se tait

Fréquence de chaque condition sur 13 370 observations :

| Condition | Fréquence |
|---|---|
| RSI < 35 | 12,8 % |
| + prix sous Bollinger | **3,1 %** — retire 76 % |
| + volume élevé | **2,1 %** — retire encore 33 % |

La bande de Bollinger étrangle l'agent. Elle mesure largement la même chose que
le RSI — une baisse récente — mais bien plus strictement.

Conséquence : **163 dates sur 955 portent un signal**, contre 907 pour
BuffettAgent. Le bootstrap rééchantillonne les dates : c'est leur nombre, pas
celui des signaux, qui fixe la largeur de l'intervalle.

### La correction qu'il ne faut PAS faire

Desserrer les seuils. Passer Bollinger à 1,5 écart-type multiplierait les
signaux et resserrerait l'intervalle — mais le seuil aurait été choisi en
regardant ce même historique, et le résultat ne vaudrait plus rien. C'est
exactement ce qui a produit `near_high_252_threshold: 0.85  # assoupli de 0.90`
dans `buffett.py`.

### La correction légitime : le même agent, plus d'observations

Appliquer la règle **inchangée**, aux **mêmes seuils**, à un univers plus
large. On ne modifie pas l'affirmation, on la teste sur plus de données.

Avec 2,1 % de déclenchement par titre et par jour, presque chaque séance
porterait un signal :

| Univers | Dates avec signal | Effet sur l'IC | IC projeté à H20 |
|---|---|---|---|
| 14 titres (aujourd'hui) | 163 | — | [−0.3 %, +16.3 %] |
| 100 titres | ~841 | ÷2,3 | **[+4.6 %, +12.0 %]** |
| 500 titres | ~955 | ÷2,4 | **[+4.9 %, +11.7 %]** |

### Verdict

**Conservé dans l'arène, et promu au rang de seul candidat sérieux.**

**Prédiction falsifiable, posée avant le test :** si l'effet est réel à cette
amplitude, un univers de 100 titres ou plus doit produire un intervalle
excluant zéro à H20. **S'il s'évanouit en s'élargissant, c'était du bruit** —
et l'agent sera retiré comme le précédent.

Deux réserves à ne pas oublier au moment de conclure :

- Les seuils (RSI 35, Bollinger 2σ) ont été choisis en regardant ces mêmes
  marchés. Le +8,3 % est une **borne haute**, pas une promesse.
- Le retour à la moyenne sur petites capitalisations n'est pas celui des
  mégacaps : écarts achat-vente plus larges, liquidité moindre. Le test devra
  être **net de coûts**, sans quoi il ne prouvera rien d'exploitable.

Ce test rejoint le chantier d'univers de `docs/hypothese_01_accruals.md` : les
deux ont besoin de la même chose — un univers large, reconstitué à chaque date
sans biais du survivant.

---

## 3. CTATrendAgent — **RETIRÉ**, et les 6 ETF avec lui (2026-08-13)

### Ce qu'il affirmait

Suivi de tendance multi-actifs, façon Winton ou Man AHL : momentum 3 et 6 mois
filtré par moyenne longue et force de tendance, sur six fonds indiciels
couvrant actions américaines, technologie, obligations, or, dollar et matières
premières. Position longue **ou vendeuse**, taille ajustée à la volatilité.

L'idée est légitime, et la diversification entre classes d'actifs a une vraie
valeur théorique.

### Quatre raisons de le retirer, dont une rédhibitoire

**1. Aucun avantage mesurable.** Sur 955 séances : excès +0,2 % / +0,3 % /
+0,8 % à H1, H5, H20, tous les intervalles traversant zéro.

**2. Il rapporte moins que ne rien faire.** Sharpe 0,21 contre 1,59 pour un
simple achat équipondéré des mêmes six fonds. En cumul : +4 % contre +67 %.
Ajouté à un portefeuille passif, il **dégrade** le Sharpe à toutes les
allocations testées.

**3. Le dosage du risque n'a jamais fonctionné.** 100 % des signaux sortent au
plafond de 15 %, écart-type des poids `2.8e-17`. La formule `min(0.10 / vol,
0.15)` ne cède le plafond qu'au-delà de 66,7 % de volatilité annualisée ; le
maximum observé en cinq ans sur ces fonds est 61,1 %. UUP, à 6,5 % de
volatilité, recevait le même poids que QQQ à 18,6 % — l'inverse exact de ce
que le dosage annonce.

**4. Rédhibitoire : les positions ne peuvent pas exister.** Au premier run
réel, IBKR a refusé l'ordre :

```
Error 201: Order rejected — Customer Ineligible
This product does not have a KID in a language approved for your country
```

Réglementation européenne **PRIIPs** : un particulier résidant dans l'UE ne
peut pas acheter d'ETF américain, faute de document d'information
réglementaire. L'univers CTA en est composé à 100 %.

> **Aucun backtest ne pouvait révéler ce point.** Il fallait un ordre réel. Ce
> qui valide exactement la raison d'être des runs quotidiens : ils ne mesurent
> pas l'avantage, ils vérifient que la machine peut fonctionner.

### Conséquence sur l'univers

`SPY`, `QQQ` et `GLD` quittent la liste tradable ; `TLT`, `UUP` et `DBC` ne
sont plus téléchargés. **L'univers passe de 14 à 11 titres, tous des actions.**

`SPY` et `GLD` restent téléchargés comme **données** — le premier détecte le
régime de marché, les deux alimentent MacroAgent. D'où la distinction
introduite dans `config.py` entre `WATCHLIST` (ce qu'on peut acheter) et
`DATA_ONLY` (ce qu'on lit sans jamais le posséder).

### Découverte annexe : PairsTradingAgent est amputé de moitié

Sept de ses quatorze paires candidates sont des ETF — `SPY/QQQ`, `SPY/IWM`,
`GLD/SLV`, `QQQ/XLK`, `XLF/XLK`, `XLE/XLU`, `XLV/XLP` — donc inexécutables. Les
sept autres sont des paires d'actions (`XOM/CVX`, `SO/DUK`, `AEP/EXC`,
`MET/PRU`, `CB/TRV`, `RF/FITB`, `CFG/HBAN`) et restent valides. À traiter au
verdict 7.

### Verdict

**Retiré de la production.** Le module reste dans le dépôt pour le replay
historique. Rien ne peut le ressusciter tant que le compte est européen et
l'univers composé d'ETF américains.

---

*Prochains verdicts prévus : la fusion
Buffett/Citadel/TrendFollowing, Macro et Volatility, DividendArb et InsiderBuy,
puis Pairs et EarningsSentiment.*
