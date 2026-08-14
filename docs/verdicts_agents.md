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

## ⚠️ Correction statistique du 2026-08-14 — lire avant les verdicts

Tous les intervalles de confiance de ce registre antérieurs au 2026-08-14
étaient **trop étroits**. Ils ont été recalculés ; les tableaux ci-dessous
portent les valeurs corrigées, et chaque endroit où la conclusion change le
dit explicitement.

**Le défaut.** Un rendement à 20 jours mesuré chaque séance partage 19 jours de
marché avec celui de la veille. Le bootstrap tirait ces séances indépendamment,
comme si chacune apportait une information neuve : il comptait la même
information vingt fois et divisait l'intervalle par la racine d'un effectif
fictif. Sur le momentum long/short, 1 364 « observations » n'en valaient que 69.

**Le remède**, appliqué en un seul endroit (`block_bootstrap_indices` dans
`src/analysis/agent_edge.py`, appelé par les quatre scripts de mesure) : tirer
des **blocs contigus** de la longueur de la fenêtre plutôt que des séances
isolées. La dépendance est conservée à l'intérieur d'un bloc, deux blocs
éloignés restent indépendants. Sept tests de non-régression le verrouillent.

> **Le remède avait lui-même un défaut, corrigé le même jour.** La première
> version tirait les débuts de bloc de façon à ne jamais déborder de la série,
> ce qui sous-échantillonnait gravement ses extrémités — la première séance
> pesait 23 fois moins que celles du milieu. Le symptôme était visible sans
> ambiguïté : dans le test de régime sur 2010-2019, la moyenne mesurée
> (+1.32 %) tombait **hors de son propre intervalle** [−5.00 %, +1.19 %], ce
> qu'un bootstrap correct ne peut pas produire. La série est désormais refermée
> en anneau (bootstrap circulaire, Politis-Romano 1992) : chaque séance pèse
> exactement autant que les autres. Les chiffres de ce document sont ceux
> d'après cette seconde correction.

**Ce que la correction a changé.** Deux résultats « significatifs » tombent :
le momentum transversal perdant à H20 (verdict 1), et le rendement par signal
du CTA à H20, seul intervalle strictement positif de `docs/agent_edge.md`
(verdict 3). L'hypothèse 01 sur les régularisations comptables est rejetée pour
la même raison (`docs/hypothese_01_accruals.md`).

**Aucun résultat n'a été renforcé.** C'est attendu : la correction ne fait que
retirer de la certitude qu'on n'avait pas. Les verdicts de retrait, eux, ne
changent pas — ils reposaient sur l'absence d'avantage démontré, et une
absence de preuve reste une absence de preuve avec un intervalle plus large.

---

## 1. CrossSectionalMomentumAgent — **RETIRÉ** (2026-08-13)

### Ce qu'il affirme

Momentum transversal de Jegadeesh-Titman (1993) : classer l'univers par
rendement moyen sur 3, 6 et 12 mois — en sautant le dernier mois pour éviter
l'effet de retournement court terme — puis acheter le quartile supérieur.

C'est une stratégie académique sérieuse, l'une des mieux documentées de la
littérature. L'idée n'est pas en cause.

### Ce que dit la mesure

955 séances rejouées, bootstrap par blocs, référence = taux de base
inconditionnel du même univers (`docs/agent_edge.md`, remesuré le 2026-08-14) :

| Horizon | N | dates | taux | base | excès | IC 95 % |
|---|---|---|---|---|---|---|
| H1 | 3 551 | 954 | 46.8 % | 45.8 % | +1.0 % | [−1.1 %, +3.1 %] |
| H5 | 3 535 | 950 | 54.7 % | 54.3 % | +0.5 % | [−3.2 %, +4.1 %] |
| H20 | 3 475 | 935 | 58.9 % | 60.2 % | −1.2 % | [−8.1 %, +5.4 %] |

Aux trois horizons, l'intervalle contient zéro : l'agent est **indistinguable
du hasard** sur cet univers.

> **Ce que disait cette section avant le 2026-08-14.** Elle affirmait qu'à
> vingt jours l'excès valait −3.0 % avec un intervalle [−5.0 %, −1.1 %]
> excluant zéro, et en tirait « le seul résultat statistiquement solide de tout
> l'audit d'agents ». C'était faux : l'intervalle ignorait le chevauchement des
> fenêtres de vingt jours (voir la correction en tête de document), et l'excès
> lui-même était mesuré sur un univers de 14 titres incluant des ETF, depuis
> ramené à 11 actions.
>
> **Le verdict de retrait ne change pas, mais son fondement si.** On ne peut
> plus dire que l'agent perd de l'argent de façon démontrée ; on peut seulement
> dire qu'il n'a jamais démontré en gagner. C'est suffisant pour le retirer —
> un agent doit justifier sa place, pas attendre qu'on prouve sa nocivité — et
> c'est moins que ce que ce document affirmait.

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

### RÉEXAMEN AVEC LES DEUX JAMBES — 2026-08-14

Les deux conditions posées ci-dessus ont été réunies : 500 titres disponibles,
et vente à découvert vérifiée comme autorisée sur le compte. Score de momentum
inchangé, déciles au lieu de quartiles — retour à la construction de
Jegadeesh-Titman, le quartile n'ayant été qu'une concession à un univers de 14
titres.

| | séances | écart long/short à 20 j | IC 95 % | |
|---|---|---|---|---|
| 2020-2026 | 1 364 | +0.53 % | [−0.47 %, +1.55 %] | traverse zéro |
| 2010-2019 | 3 821 | +0.42 % | [−0.19 %, +1.06 %] | traverse zéro |

Le signe est le même sur les deux époques et va dans le sens prédit par
Jegadeesh-Titman. L'ampleur, elle, n'est pas distinguable de zéro.

> **Correction du même jour, quelques heures plus tard.** La première version de
> cette section donnait [+0.22 %, +0.82 %] et [+0.24 %, +0.60 %], deux
> intervalles strictement positifs, et concluait : « c'est le premier résultat
> du projet qui se reproduit sur deux époques ». C'était un artefact de
> comptage. Le classement est refait chaque séance alors que le rendement court
> sur vingt jours : les 1 364 lignes de 2020-2026 ne valent que 69 observations
> indépendantes. Avec le bootstrap par blocs, les deux intervalles traversent
> zéro.
>
> Le diagnostic de départ (univers trop petit, jambe vendeuse absente) reste
> plausible — le signe est constant — mais il n'est **pas démontré**. La
> distinction compte : « vrai et trop petit » était une affirmation, c'est
> redevenu une hypothèse.

Deux constats survivent à la correction, parce qu'ils ne dépendent d'aucun
intervalle de confiance — ils décrivent la distribution elle-même.

**Le signal est dix fois plus petit que le bruit.**

| | 2020-2026 | 2010-2019 |
|---|---|---|
| moyenne | +0.53 % | +0.42 % |
| écart-type | **5.63 %** | **5.72 %** |
| séances négatives | 42 % | 44 % |

**La queue est catastrophique.** Pire fenêtre de 20 jours : **−31,4 %** sur la
première période, **−34,5 %** sur la seconde. Une seule efface soixante séances
de gain moyen. C'est le « krach de momentum », mode de défaillance documenté de
cette stratégie — elle perd le plus violemment quand les perdants rebondissent
brutalement.

**Les coûts finissent le travail** : net de 40 bps d'aller-retour, il reste
+0,13 % et +0,02 % ; à 60 bps, les deux périodes deviennent négatives. Or un
long/short en déciles rebalancé toutes les vingt séances a une rotation double,
plus des frais d'emprunt sur la jambe vendeuse.

**Verdict inchangé : reste retiré.** Et l'argument est plus court qu'il n'y
paraissait. Il n'est même pas nécessaire de trancher si l'effet existe : un
écart moyen de 0,4 à 0,5 % par vingt séances, noyé dans 5,7 % d'écart-type,
exposé à des pertes de 30 % et effacé par 60 bps de frais, ne serait pas
exploitable **même s'il était démontré**. La correction statistique retire une
certitude qu'on n'avait pas ; elle ne retire rien à cette conclusion-là.

Ce qui pourrait changer cela, et rien d'autre : une exécution nettement moins
chère, ou une couverture du risque de krach de momentum. Les deux sortent du
périmètre actuel du fonds.

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

### RÉSULTAT DU TEST — 2026-08-13, quelques heures plus tard

**La prédiction est tenue.** 570 sociétés, appartenance à l'indice reconstituée
à chaque date, seuils strictement inchangés.

| Horizon | N | dates | excès | IC 95 % | |
|---|---|---|---|---|---|
| H1 | 14 587 | 1 270 | +3.3 % | [−0.3 %, +6.8 %] | |
| **H5** | 14 570 | 1 266 | **+5.4 %** | **[+2.0 %, +8.8 %]** | ✅ |
| **H20** | 14 453 | 1 252 | **+5.1 %** | **[+2.1 %, +8.0 %]** | ✅ |

Et sur le rendement, pas seulement la fréquence :

| Horizon | rendement/signal | passif | IC 95 % | |
|---|---|---|---|---|
| H20 | **+2.194 %** | +1.137 % | [+1.443 %, +2.962 %] | ✅ |

**Il bat la référence passive**, ce que CTA ne faisait pas. De 163 dates à
1 252 : la puissance manquante est là.

**Net de coûts**, comme le verdict l'exigeait :

| Aller-retour | net | vs passif |
|---|---|---|
| 10 bps | +2.09 % | +0.96 % |
| 20 bps | +1.99 % | +0.86 % |
| 40 bps (pessimiste) | +1.79 % | +0.66 % |

L'équivalence entre le calcul vectorisé et l'agent réel a été **prouvée avant
le test**, sur 40 points tirés au hasard : 40/40 identiques. Sans cette
vérification, on aurait testé autre chose que MeanReversion.

### Ce que ce résultat NE prouve PAS

Trois réserves, et la première est la plus sérieuse.

**1. Ce n'est pas un test hors échantillon dans le TEMPS.** Les seuils
(RSI 35, Bollinger 2σ) ont été choisis en regardant les mêmes marchés, sur la
même période 2020-2026. Élargir l'univers ajoute des observations
**transversales**, pas temporelles. Le test décisif reste à faire : rejouer la
règle telle quelle sur une période antérieure, 2010-2019, jamais consultée.

**2. Le profil de gain est asymétrique dans le mauvais sens.** Skew −0,63 à
H20, ratio gain/perte 1,17. La stratégie gagne par la **fréquence**, pas par
l'ampleur : beaucoup de petits gains, quelques pertes lourdes. C'est le profil
qui ramasse des pièces devant un rouleau compresseur, et il exige une gestion
du risque stricte.

**3. Les 11 % de sociétés manquantes sont précisément les mauvaises pour cette
stratégie.** La couverture est de 88,9 %, et parmi les 71 absentes figurent
**SIVB et FRC** — Silicon Valley Bank et First Republic, deux faillites
bancaires de 2023. Une stratégie qui achète les baisses les aurait ramassées
en chute libre. Leur absence flatte donc le résultat, contrairement à la
plupart des autres manquantes qui sont des rachats.

### Verdict mis à jour

**Conservé. Premier agent du fonds à franchir un test réel.**

Ce n'est pas encore un avantage démontré hors échantillon — c'est un effet
robuste à l'élargissement de l'univers, qui survit aux coûts, et dont les
limites sont nommées. La prochaine étape est écrite : **la période 2010-2019,
seuils inchangés.** Si l'effet y tient, on aura quelque chose. S'il disparaît,
il aura été une propriété de 2020-2026.

### TEST HORS ÉCHANTILLON — 2026-08-14 : **il ne se reproduit pas**

535 sociétés, 2010-2019, appartenance reconstituée à chaque date, seuils
strictement inchangés. Une décennie qu'aucun réglage n'avait jamais vue.

| | 2020-2026 (conception) | 2010-2019 (validation) |
|---|---|---|
| Taux de réussite H1 | +3.2 % [−0.4, +7.0] | +4.7 % [+1.8, +7.7] ✅ |
| Taux de réussite H5 | +5.3 % [+1.9, +8.7] ✅ | +3.4 % [+0.9, +5.5] ✅ |
| Taux de réussite H20 | +5.1 % [+2.0, +8.0] ✅ | +2.0 % [−0.5, +4.3] ❌ |
| **Rendement H5** | **+0.855 % [+0.46, +1.25] ✅** | **+0.254 % [−0.22, +0.67] ❌** |
| **Rendement H20** | **+2.254 % [+1.48, +3.00] ✅** | **+0.856 % [−0.09, +1.72] ❌** |
| vs passif à H20 | +2.25 % contre +1.17 % ✅ | +0.86 % contre +0.73 % |
| Ratio gain/perte H20 | 1.20 | **0.87** |

**La prédiction écrite le matin même n'est pas tenue.** L'intervalle à H20
inclut zéro sur la période de validation.

Et le point décisif est ailleurs, dans les lignes de rendement : **les trois
intervalles traversent zéro.** Le taux de réussite conserve une part de
persistance à H1 et H5 — ce n'est pas rien — mais **un taux de réussite sans
avantage de rendement n'est pas exploitable.** Le ratio gain/perte de 0,87 le
dit crûment : on a raison plus souvent, et on perd davantage quand on a tort.

### L'incident qui a failli inverser la conclusion

Le premier passage de ce test affichait un rendement significatif à tous les
horizons — et un **skew de +43,6**. Sur 35 000 observations, ce chiffre ne
décrit pas un marché mais quelques valeurs aberrantes.

Elles venaient d'un seul ticker. `TIE` affichait +758 % de rendement journalier
— +197 000 % en prix. **Seize observations portaient la moitié du rendement
moyen.** L'inspection a révélé 37 séries corrompues sur 547 : tickers
réattribués, facteurs d'ajustement cassés.

Après filtrage (`src/data/quality.py`, 12 séries écartées), le skew passe de
+43,6 à −2,07 et **tous les rendements deviennent non significatifs**.

> Sans ce contrôle, ce document annoncerait aujourd'hui que MeanReversion se
> reproduit hors échantillon. C'était faux, et rien dans les chiffres bruts ne
> le signalait.

Le résultat de conception, lui, **survit au même filtrage** — une seule série
écartée sur 2020-2026, et les chiffres bougent à peine. Il n'était donc pas
contaminé.

### Verdict mis à jour — **CONSERVÉ EN SURSIS**

L'effet mesuré sur 2020-2026 était réel et robuste à l'élargissement de
l'univers. **Il n'est pas robuste au changement d'époque.** C'était une
propriété de cette période, pas une régularité du marché.

Il reste dans l'arène pour deux raisons, et pas une de plus :

1. La persistance du taux de réussite à H1 et H5 sur les deux périodes n'est
   pas rien. Quelque chose existe, ce n'est simplement pas exploitable en
   l'état.
2. C'est le dernier agent du fonds dont une propriété résiste à un test.

**Ce qu'il faudrait pour le confirmer**, et qui ne relève plus du réglage :
comprendre *pourquoi* le rendement se dissipe alors que le taux de réussite
tient. L'hypothèse la plus simple est que 2020-2026 a connu des rebonds
d'ampleur inhabituelle — le creux de mars 2020, puis 2022 — que la décennie
précédente n'offrait pas. Si c'est vrai, l'agent ne capte pas une inefficience
mais un régime.

**Ce qui le condamnerait :** un troisième test sur une autre période, avec le
même échec. Ou la démonstration que la persistance du taux de réussite
s'explique entièrement par la structure des données.

### TEST DE L'HYPOTHÈSE DE RÉGIME — 2026-08-14 : **réfutée**

Protocole fixé avant exécution (`scripts/test_regime_hypothesis.py`) : chaque
signal classé selon la baisse de SPY depuis son plus haut glissant sur un an,
en quatre tranches, sur les deux périodes. La référence est le rendement
inconditionnel de l'univers **dans la même tranche** — comparer à la moyenne
globale aurait mélangé effet de régime et effet d'agent.

| Régime | part 20-26 | part 10-19 | excès 20-26 | excès 10-19 |
|---|---|---|---|---|
| 0-5 % sommets | 55.4 % | 57.1 % | +0.51 % | +0.10 % |
| 5-10 % correction | 17.7 % | 19.5 % | **+3.14 %** ✅ | −0.21 % |
| 10-20 % marquée | 21.7 % | 19.3 % | **+2.84 %** ✅ | −1.91 % |
| > 20 % baissier | 5.1 % | 4.0 % | **−2.62 %** ❌ | +1.32 % |

L'hypothèse avait deux jambes. La première tient, la seconde s'effondre.

**Confirmé :** l'avantage se concentre bien dans les corrections. Tout se joue
entre 5 et 20 % de baisse ; ailleurs, rien.

**Réfuté :** la composition des régimes est quasi identique entre les deux
périodes. Marché tendu (baisse > 10 %) : **26,8 % contre 23,4 %**. Trois points
d'écart ne peuvent expliquer un renversement de signe.

### Ce que le test révèle à la place

Même régime, résultats opposés :

```
correction  5-10 %    2020-2026 : +3.14 %      2010-2019 : −0.21 %
correction 10-20 %    2020-2026 : +2.84 %      2010-2019 : −1.91 %
```

Ce n'est donc pas la **fréquence** des corrections qui a changé, c'est **leur
comportement**. Les creux de 2020-2026 se sont rattrapés en V, soutenus par une
liquidité exceptionnelle ; ceux de la décennie précédente se sont traînés.

L'agent n'exploite ni une inefficience persistante, ni une composition de
régimes. Il exploite **une caractéristique d'une époque** — ce qui est plus
difficile à défendre, et impossible à projeter.

### Le seul résultat immédiatement exploitable

```
> 20 % baissier    −2.62 %    IC [−4.08 %, −1.15 %]
```

**En marché franchement baissier, l'agent perd de façon significative.** C'est
le couteau qui tombe : acheter des baisses pendant un krach. Le résultat est
cohérent avec la théorie, significatif, et donne une règle applicable
directement — cet agent ne doit pas acheter au-delà de 20 % de baisse du
marché.

C'est aussi, symétriquement, la seule piste sérieuse de **vente à découvert**
identifiée à ce jour : ce que l'agent perd à l'achat dans ce régime, il
pourrait le gagner à la vente. À tester séparément — le constat d'une perte à
l'achat ne démontre pas un gain à la vente.

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

> **Le seul argument à décharge est tombé le 2026-08-14.** Juger un suiveur de
> tendance au taux de réussite est injuste : il a classiquement raison 35 à
> 40 % du temps et gagne quand même, parce que ses gains dépassent ses pertes.
> D'où la seconde mesure, le rendement moyen par signal — la seule du projet à
> porter un intervalle strictement positif : **[+0,284 %, +0,628 %]** à vingt
> jours. Elle souffrait exactement du même défaut de chevauchement que les
> autres. Recalculée par blocs : **[−0,084 %, +0,985 %]**, qui traverse zéro.
>
> Le retrait ne dépendait pas de ce point — la raison 4 est réglementaire et
> sans appel — mais il ne subsiste désormais plus rien du côté favorable.

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
