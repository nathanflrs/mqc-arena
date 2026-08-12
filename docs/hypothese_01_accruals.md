# Hypothèse 01 — La qualité des bénéfices, là où personne ne regarde

*Écrite le 2026-08-12, **avant** tout test.*

---

## Pourquoi ce document est daté et figé

Ce document est un **engagement pris à l'avance**. Il décrit ce qu'on va
tester, comment, et ce qui compterait comme un échec — le tout avant d'avoir vu
le moindre résultat.

C'est la seule protection connue contre la manière la plus courante de se
tromper soi-même : essayer trente variantes, garder celle qui brille, et
raconter après coup pourquoi c'était la bonne. Avec assez d'essais, on trouve
toujours quelque chose. Ce quelque chose ne survit jamais au contact du réel.

**Toute modification de ce protocole après avoir vu les résultats doit être
écrite ici, datée, et justifiée.** Un protocole qu'on ajuste en silence ne vaut
rien.

---

## 1. L'hypothèse, en une phrase

> Les entreprises dont le bénéfice comptable est mal adossé à de la trésorerie
> réelle sous-performent ensuite — et cet écart est plus fort là où peu
> d'analystes regardent.

## 2. Le mécanisme économique

Une entreprise peut afficher un beau bénéfice sans encaisser un centime.

C'est parfaitement légal. Une vente à crédit compte comme un bénéfice le jour
de la facture, pas le jour du paiement. Un stock qui gonfle réduit les charges
comptables. Ces écarts entre le bénéfice annoncé et l'argent réellement entré
s'appellent les **régularisations comptables** (*accruals*).

Deux raisons de penser qu'ils prédisent une sous-performance :

**La raison mécanique.** Un bénéfice non encaissé finit par se corriger — la
créance impayée devient une perte, le stock invendu est déprécié. Le bénéfice
d'aujourd'hui est emprunté à demain.

**La raison comportementale.** Le bénéfice par action est le chiffre que tout
le monde regarde. L'état des flux de trésorerie est en page 6 et demande du
travail. Les investisseurs réagissent donc au premier et négligent le second.

C'est une anomalie documentée depuis 1996 (Sloan), et c'est précisément ce qui
impose de la tester plutôt que d'y croire : **une anomalie publiée est une
anomalie que d'autres exploitent.** Elle a pu disparaître. Le test doit dire
laquelle des deux situations est vraie aujourd'hui.

## 3. Pourquoi cette hypothèse-là, pour Milan Capital

| Critère | Vérification |
|---|---|
| Mécanisme économique explicite | Oui — pas un motif trouvé dans les données |
| Données disponibles et honnêtes | Oui — `src/data/sec_fundamentals.py`, point-in-time |
| Terrain non encombré | Oui — l'effet est documenté comme plus fort sur les petites capitalisations peu suivies |
| Horizon compatible | Oui — révision annuelle, personne n'est payé pour attendre |
| Falsifiable | Oui — voir §6 |

Et surtout : **elle exige exactement le travail qu'on vient de faire.** Comparer
un bénéfice à une trésorerie suppose que les deux couvrent la même durée. C'est
le bug corrigé aujourd'hui (trimestre contre exercice, marge apparente de 44 %
chez UFPT). Sans ce correctif, cette hypothèse aurait été intestable.

## 4. La mesure

Pour chaque société, à chaque date de décision :

```
régularisations = (résultat net − trésorerie d'exploitation) / actif total
```

Les trois grandeurs sont ramenées à douze mois glissants et rapportées à
l'actif total, pour qu'une société de 100 M$ et une de 5 Md$ soient comparables.

- **Régularisations élevées** → bénéfice peu adossé à du cash → candidat à la
  sous-performance.
- **Régularisations faibles ou négatives** → bénéfice encaissé, voire plus →
  candidat à la surperformance.

## 5. L'univers, et ses pièges

**Définition :** sociétés américaines cotées déposant auprès de la SEC, avec au
moins huit trimestres d'historique, hors sociétés financières et immobilières
(leur bilan ne se lit pas de la même façon : chez une banque, la dette *est*
l'activité).

Trois pièges à traiter explicitement, faute de quoi le résultat ne vaut rien :

**Le biais du survivant.** Construire l'univers à partir de la liste des
sociétés cotées *aujourd'hui* exclut toutes celles qui ont fait faillite. On
mesurerait alors la performance d'un portefeuille dont on aurait retiré les
faillites à l'avance. L'univers doit être reconstitué à chaque date à partir des
dépôts existants **à cette date**.

**La liquidité.** Une société de 50 M$ peut ne pas s'échanger assez pour qu'on
y entre. Filtre : volume quotidien moyen suffisant pour que la position visée
reste sous 1 % du volume — la règle déjà appliquée par le gestionnaire de
risque.

**Les coûts.** Sur des petites capitalisations, l'écart entre prix d'achat et de
vente est large. Toute performance sera présentée **nette de coûts estimés**,
jamais brute. Une stratégie qui ne survit pas à ses frais n'existe pas.

## 6. Ce qui compterait comme un échec

Défini maintenant, pour ne pas être renégocié plus tard.

L'hypothèse est **rejetée** si, sur la période de validation :

1. l'écart de rendement entre le décile de régularisations les plus faibles et
   celui des plus élevées a un intervalle de confiance à 95 % **contenant
   zéro** ; ou
2. cet écart devient nul ou négatif une fois les coûts de transaction déduits ;
   ou
3. l'effet ne survit pas au retrait des 1 % de sociétés les plus extrêmes —
   auquel cas il s'agit de quelques accidents comptables, pas d'un phénomène.

**Si l'hypothèse est rejetée, ce document reste dans le dépôt avec son verdict.**
Les hypothèses mortes ne s'effacent pas : elles disent où on a déjà cherché.

## 7. Le protocole de validation

**Découpage temporel, décidé maintenant :**

| Période | Usage |
|---|---|
| 2011 → 2019 | Conception. On a le droit de regarder, d'ajuster, de se tromper. |
| 2020 → 2023 | Contrôle intermédiaire. Un seul passage. |
| 2024 → 2026 | **Validation finale. Interdite jusqu'à la toute fin.** |

**Nombre d'essais :** au plus **trois** variantes testées sur la période de
validation. Chaque essai supplémentaire augmente la probabilité de trouver un
faux positif ; s'autoriser trente essais garantit de « trouver » quelque chose
d'inexistant.

**Référence de comparaison :** le rendement moyen de l'univers lui-même, pas
zéro et pas le S&P 500. Si les petites capitalisations montent de 15 % dans
l'année, une stratégie qui fait 12 % sur ce terrain a détruit de la valeur.

**Intervalles de confiance :** bootstrap sur les **dates**, comme dans
`src/analysis/agent_edge.py`. Les sociétés d'un même jour bougent ensemble ;
les traiter comme indépendantes divise l'intervalle par la racine du nombre de
titres et fabrique une fausse certitude.

## 8. Ce qui n'est PAS promis

- Que l'anomalie existe encore. Elle est publiée depuis 1996.
- Qu'elle soit exploitable après frais sur des titres peu liquides.
- Que Milan Capital puisse l'exécuter à sa taille.

**Ce document ne promet qu'une chose : on saura.** Et le jour où on saura, le
résultat sera écrit ici, qu'il soit favorable ou non.

## 9. Ce qui vient après, dans l'ordre

1. Construire l'univers point-in-time (le morceau le plus délicat — le biais du
   survivant est un travail en soi).
2. Calculer les régularisations sur la période de conception.
3. Regarder. Ajuster. Se tromper. **Sans jamais toucher à 2024-2026.**
4. Un passage sur 2020-2023.
5. Si ça tient : le passage final. Puis, **et seulement alors**, écrire l'agent.

---

*Règle du projet, posée dans `docs/comprendre_milan_capital.md` :*
*hypothèse → données → validation hors échantillon → ensuite le code.*
*Ce document est l'étape 1.*
