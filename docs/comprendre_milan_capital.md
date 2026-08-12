# Comprendre Milan Capital

*Écrit le 2026-08-12. Ce document explique le fonds en français ordinaire.*

---

## Pourquoi ce document existe

Un fonds que son gérant ne comprend pas est un problème plus grave qu'une
stratégie qui ne marche pas. Une stratégie, ça se remplace. La compréhension,
non — sans elle on ne peut ni décider, ni corriger, ni défendre publiquement ce
qu'on affirme.

Tout ce qui suit est vérifiable dans le code. Quand un chiffre est donné, sa
source est indiquée. Quand quelque chose n'est pas mesuré, c'est écrit.

**Si une phrase de ce document est incompréhensible, c'est le document qu'il
faut corriger.**

---

## 1. Ce que fait le fonds, chaque jour

Tous les jours de bourse, la même séquence :

**1. On regarde le portefeuille.** Combien d'argent, quelles positions ouvertes.

**2. On télécharge les prix** des 14 actions et fonds suivis, sur 2 ans
d'historique.

**3. On détermine l'ambiance du marché** — hausse, baisse, ou agitée. C'est ce
qu'on appelle le *régime*.

**4. Les 13 agents donnent leur avis**, action par action. Un agent est un
programme qui suit une règle fixe et répond : acheter, vendre, ou ne rien faire.
Chacun accompagne son avis d'un chiffre de « confiance » entre 0 et 1.

**5. Un seul avis est retenu par action.** C'est ce qu'on appelle *l'arène* :
les agents sont mis en concurrence, un seul gagne, les autres sont ignorés.

**6. Les garde-fous filtrent.** Position trop grosse ? Trop de risque déjà pris ?
Pas assez de liquidités ? Résultats trimestriels dans trois jours ? L'ordre est
réduit ou annulé.

**7. Les ordres partent** chez le courtier (Interactive Brokers), *si*
l'exécution est activée. Aujourd'hui elle ne l'est que sur ta machine, jamais
depuis le cloud.

**8. Tout est enregistré** — chaque avis d'agent, chaque ordre, chaque refus.

---

## 2. Les treize cuisiniers

Voici ce que chaque agent fait **réellement**, indépendamment de son nom.

> ⚠️ **Les noms sont trompeurs.** Plusieurs agents portent le nom d'un
> investisseur célèbre sans faire ce que cet investisseur fait. C'est un défaut
> connu du projet, pas une subtilité : il a masqué pendant des mois le fait que
> trois agents font la même chose.

### Les trois qui font le même métier

**BuffettAgent** — Regarde trois choses : le prix est-il au-dessus de sa moyenne
sur 200 jours ; l'action est-elle calme (peu de variations) ; le prix est-il
proche de son plus haut de l'année. Si deux conditions sur trois sont remplies,
il achète.

> **Il ne regarde aucun chiffre d'entreprise.** Ni bénéfices, ni dettes, ni
> valorisation. Le vrai Buffett achète une entreprise **moins cher** qu'elle ne
> vaut. Cet agent achète ce qui est **proche de son plus haut** — c'est-à-dire
> l'inverse. Le nom est une décoration.

**CitadelAgent** — Le prix doit être au-dessus de sa moyenne 200 jours, la
moyenne 50 jours au-dessus de la moyenne 200 jours, le volume correct, et soit
le prix casse son plus haut du mois, soit il a progressé de plus de 5 % en trois
mois.

**TrendFollowingAgent** — Les moyennes mobiles alignées dans le bon ordre, une
mesure de force de tendance (l'ADX) au-dessus d'un seuil, et une progression sur
20 jours.

> **Ces trois agents disent tous : « le prix monte, donc j'achète ».** Formulé
> avec des indicateurs différents, mais c'est la même idée. C'est pour ça qu'ils
> sont presque toujours d'accord, et pour ça que l'arène n'arbitre rien.

### Celui qui fait l'inverse

**MeanReversionAgent** — Achète quand ça vient de baisser fort : indicateur de
survente (RSI) sous 35, prix sous sa bande basse habituelle, volume élevé.
Il parie sur le rebond.

> C'est le seul agent dont la mesure suggère un avantage aux trois horizons
> testés. Mais il se déclenche si rarement (272 signaux contre 10 271 pour
> Buffett) qu'on ne peut pas conclure. **À étudier en priorité.**

### Celui qui parle du monde, pas des actions

**MacroAgent** — Regarde l'économie globale : écart des taux d'intérêt, stress
du crédit, indice de la peur, politique de la banque centrale. Il en tire un
score « ambiance favorable / défavorable ».

> Ce score est **le même pour les 14 actions** d'une même journée. Seule une
> petite condition sur la progression du titre le rend légèrement spécifique.
> En pratique, il donne un avis global recopié partout.

### Ceux qui ne se déclenchent jamais

**VolatilityAgent** — Surveille le VIX, l'indice de la peur des marchés. N'agit
qu'aux extrêmes (peur > 25, panique > 35). Ces seuils n'ont jamais été atteints
sur la période. **166 avis, tous « ne rien faire ».** Et son signal ne dépend pas
de l'action examinée : c'est un avis unique recopié 14 fois.

**InsiderBuyAgent** — Cherche les dirigeants qui achètent les actions de leur
propre entreprise (déclarations obligatoires à l'autorité américaine). Il faut au
moins **2 dirigeants différents achetant chacun 100 000 $ ou plus en 30 jours**.
Sur des géants comme Apple ou Microsoft, ça n'arrive jamais. **0 signal.**

**DividendArbitrageAgent** — Achète juste avant qu'une entreprise verse son
dividende, revend après. Ne s'est jamais déclenché sur la période.

> L'idée de ces trois agents n'est pas mauvaise. **C'est l'univers qui ne
> convient pas.** Chercher des achats de dirigeants ou des dividendes
> exploitables sur les 14 plus grosses entreprises du monde, c'est chercher là
> où tout le monde regarde déjà.

### Les cas particuliers

**PairsTradingAgent** — Trouve deux actifs qui bougent historiquement ensemble.
Quand leur écart se creuse anormalement, il achète le retardataire et vend
l'autre, en pariant que l'écart se refermera. Pas encore mesuré.

**EarningsSentimentAgent** — **Le seul agent qui utilise une IA.** Il récupère
les actualités récentes d'une entreprise et demande à Claude de trancher :
acheter, vendre, ou ne rien faire.

> Problème : sa confiance est un chiffre que l'IA s'attribue elle-même en lisant
> un article. Les autres agents calculent la leur à partir de mesures de marché.
> **Ces deux chiffres ne vivent pas sur la même échelle**, et pourtant ils sont
> comparés directement dans l'arène.

**CrossSectionalMomentumAgent** — Classe les 14 actions par progression récente
et achète les 4 premières.

> **Le seul agent dont on a prouvé qu'il fait perdre de l'argent** : à 20 jours,
> il fait 3 % de moins que le simple hasard, et cette fois le résultat est
> statistiquement solide. À retirer.

**CTATrendAgent** — Achète *ou vend à découvert* 6 fonds indiciels (actions
américaines, technologie, obligations, or, dollar, matières premières) selon leur
tendance sur 3 et 6 mois.

> **C'est lui qui prend les plus grosses positions du fonds** : 45 % du
> portefeuille, contre 21 % pour les douze autres réunis. Mesuré le 2026-08-12 :
> il gagne un peu d'argent, mais **acheter simplement ces 6 fonds et ne rien
> faire rapporte bien plus** (+67 % contre +4 % sur la période). Et son système
> de dosage du risque n'a jamais fonctionné — un calcul erroné fait que toutes
> ses positions sortent à la taille maximale. Voir `docs/agent_edge.md`.

**DummyHoldAgent** — Ne fait jamais rien, exprès. C'est le point de comparaison :
si un agent ne bat pas « ne rien faire », il ne sert à rien.

---

## 3. Les garde-fous — la meilleure partie du fonds

C'est ici que Milan Capital est réellement sérieux. Ces protections
fonctionnent, sont testées, et valent mieux que ce qu'on trouve dans la plupart
des projets amateurs.

| Protection | Ce qu'elle empêche |
|---|---|
| **Position maximale** | Jamais plus de 20 % du capital sur une seule ligne |
| **Exposition maximale** | Jamais plus de 60 % du capital investi en actions |
| **Réserve de liquidités** | Toujours au moins 30 % en argent disponible |
| **Filtre de liquidité** | Ne jamais peser trop lourd sur le volume quotidien d'un titre |
| **Garde de corrélation** | Ne pas acheter un titre qui bouge déjà comme ceux détenus |
| **Stop-loss** | Vente automatique si une position perd 7 % |
| **Filtre résultats** | Pas d'achat dans les 3 jours avant des résultats trimestriels |
| **Disjoncteur** | Si le fonds perd trop depuis son sommet, il se restreint puis se bloque, en 4 niveaux |
| **Estimation de perte (VaR)** | Chiffre chaque jour la perte plausible du lendemain |
| **Garde d'exécution** | Réduit un ordre trop gros au lieu de l'annuler |
| **Réconciliation** | Vérifie que le courtier a fait exactement ce qui était demandé |
| **Isolation des pannes** | Un agent qui plante n'annule plus la journée entière |

---

## 4. Ce qu'on sait, et ce qu'on ne sait pas

### Mesuré, et solide

- **Aucun agent de l'arène n'a d'avantage démontrable.** Testés sur 955 jours de
  bourse. Ce n'est pas « ils sont mauvais » — c'est « on ne peut pas les
  distinguer du hasard ».
- **CrossSectionalMomentum fait activement perdre** à 20 jours.
- **CTATrendAgent n'a pas d'avantage** et rapporte moins que ne rien faire, alors
  qu'il porte le plus gros risque du fonds.
- **La « confiance » des agents ne veut rien dire.** Quand Buffett annonce 0,90
  de confiance, il a raison 53,5 % du temps ; quand il annonce 0,70, il a raison
  54,0 %. Le chiffre n'apporte aucune information — or cinq mécanismes du fonds
  s'en servent pour décider.
- **L'arène n'arbitre presque jamais.** Sur 185 décisions, il n'y a eu qu'un vrai
  désaccord (un agent veut acheter, un autre vendre) **5 fois**.

### Pas encore mesuré

- PairsTradingAgent
- EarningsSentimentAgent (l'agent IA)
- MacroAgent, InsiderBuyAgent, DividendArbitrageAgent, VolatilityAgent — pour
  ceux-là, la mesure est techniquement impossible : leurs données d'époque ne
  sont pas reconstituables aujourd'hui.

### Su, et non résolu

- **Le fonds n'a pas de véritable historique de performance.** Le calcul
  quotidien automatique repart chaque jour d'un compte fictif vide. Le vrai
  compte n'a bougé que les jours où le programme a été lancé à la main.

---

## 5. Petit dictionnaire

**Agent** — Un programme qui suit une règle fixe et propose une décision. Il ne
trade jamais lui-même ; il propose.

**Arène** — Le mécanisme qui met les agents en concurrence et n'en retient qu'un
par action.

**Avantage / *edge*** — Le fait de faire mieux que le hasard, de façon prouvée.
C'est la seule chose qui justifie de gérer activement plutôt que d'acheter et
d'attendre.

**Backtest** — Rejouer une stratégie sur le passé pour voir ce qu'elle aurait
donné. Piège majeur : il est très facile de tricher sans le vouloir, en
utilisant des informations qui n'étaient pas connues à l'époque.

**Moyenne mobile (SMA)** — Le prix moyen des N derniers jours. « Au-dessus de la
moyenne 200 jours » est la façon la plus courante de dire « la tendance est
haussière ».

**Momentum** — La progression récente. Parier dessus, c'est parier que ce qui
monte continue de monter.

**Retour à la moyenne** — L'idée inverse : ce qui a beaucoup baissé va rebondir.

**RSI** — Un indicateur entre 0 et 100 qui mesure si un titre a beaucoup monté ou
beaucoup baissé récemment. Sous 30, on parle de « survendu ».

**ADX** — Un indicateur qui mesure la *force* d'une tendance, sans dire son sens.

**Vente à découvert (*short*)** — Parier à la baisse : on vend un actif qu'on ne
possède pas pour le racheter moins cher.

**Volatilité** — L'ampleur des variations. Une forte volatilité signifie un
risque élevé.

**Sharpe** — Le rendement rapporté au risque pris. Gagner 10 % calmement vaut
mieux que gagner 10 % en montagnes russes. Au-dessus de 1, c'est bien.

**Drawdown** — La perte depuis le point le plus haut atteint. Un drawdown de
12 % veut dire qu'on a perdu 12 % depuis le sommet.

**VaR** — Une estimation de la perte plausible sur une journée mauvaise.

**Intervalle de confiance** — La fourchette dans laquelle se trouve
vraisemblablement la vraie valeur. **S'il contient zéro, on ne peut rien
conclure.** C'est la phrase la plus importante de ce document : c'est ce qui
sépare « ça marche » de « on n'en sait rien ».

**Taux de base** — La performance qu'on obtiendrait sans réfléchir. Dans un
marché qui monte, dire « acheter » tout le temps donne un bon score sans
contenir la moindre information. C'est **contre ça** qu'il faut comparer un
agent, jamais contre pile ou face.

---

## 6. La direction prise

Le 2026-08-12, la thèse du fonds a été arrêtée : **chercher un avantage
structurel**, c'est-à-dire quelque chose qu'un acteur comme Milan Capital peut
faire et que les grandes maisons ne peuvent pas.

Sur les courbes de prix d'Apple ou Nvidia, cet avantage n'existe pas : des
milliers de gens mieux équipés regardent exactement les mêmes courbes. Il ne peut
exister que là où les autres ne vont pas :

- des entreprises que peu d'analystes suivent ;
- un horizon long — personne n'est payé pour attendre cinq ans ;
- des situations particulières et peu suivies ;
- **les chiffres réels des entreprises**, que le fonds n'utilise aujourd'hui pas
  du tout.

Ce dernier point est la piste principale. **Aucun agent de Milan Capital ne
regarde aujourd'hui les bénéfices, les dettes ou la valorisation d'une
entreprise.** Tous ne lisent que des courbes de prix.

Et une règle en découle, qui s'applique désormais à tout nouvel agent :

> **Hypothèse économique → données → validation sur des périodes non utilisées
> pour la conception → *ensuite seulement* le code.**

L'ordre inverse — écrire l'agent, mesurer après — est celui qui a produit la
situation actuelle.

---

*Documents liés : `docs/agent_edge.md` (mesures détaillées),
`docs/audit_2026-07-23.md` (audit technique de l'arbitrage).*
