# Couche marché — le S&P 500 entier, chaque nuit

*Mise en place le 2026-09-13, à la demande de Nathan : « je ne comprends pas
pourquoi on devrait se limiter à 11 stocks ».*

## Pourquoi

La liste de 11 titres était un accident historique, et elle a coûté cher :

- **Des agents affamés.** InsiderBuy et DividendArb ont émis 100 % de HOLD sur
  les mégacaps : leurs dirigeants n'achètent presque pas, leurs dividendes sont
  minuscules. Un agent d'événements a besoin de largeur.
- **Un faux avantage.** Choisis en connaissant la fin de l'histoire, ces 11
  titres ont battu les facteurs de +14 %/an sur 2022-2026. Tout agent qui les
  détenait semblait avoir de l'alpha ; aucun n'en avait face au simple panier.

## Ce qui tourne

Chaque soir de semaine à 18 h 30 (New York), `milan-market.timer` lance
`python -m src.market.observe` :

1. **Collecte** (`src/market/collect.py`) — les membres du S&P 500 du jour
   (Wikipédia, avec sous-industrie GICS et CIK), trois ans de prix via
   `src/data/prices.py`, passés au filtre de `src/data/quality.py`.
2. **Vue du marché** (`src/market/view.py`) — ce que les agents reçoivent :
   tout l'univers, rien après la date `as_of`.
3. **Agents d'univers** (`src/agents/universe_base.py`) — chacun parcourt les
   500 titres et renvoie ses propositions du jour, écrites dans
   `logs/universe_proposals.csv`.

**Aucun ordre ne peut partir de ce service** : il n'importe pas le courtier.

## Trois garanties de la collecte

- **Rien n'est remplacé par pire.** Sous 90 % de couverture, la collecte
  précédente reste en place et l'échec est écrit dans `failed_attempt.json`.
- **Aucune collecte à moitié écrite n'est visible** : écriture à côté, puis
  substitution par renommage.
- **Ce qui manque est chiffré** : membres sans prix et séries rejetées sont
  listés dans `report.json`.

## Les agents branchés, en observation

| Agent | Règle | Ce qui change |
|---|---|---|
| `InsiderClusterAgent` | Celle d'InsiderBuy : ≥ 2 dirigeants distincts, ≥ 100 k$ chacun, 30 jours | Seul l'univers. Form 4 lus depuis l'index quotidien d'EDGAR (`src/data/form4_feed.py`) |
| `PairsUniverseAgent` | Les filtres de PairsTrading, seuils inchangés ; entrée à \|z\| > 2 | Candidats : toutes les paires d'une même sous-industrie GICS (~1 450) |

Aucun seuil n'a été modifié en passant de 11 titres à 500. Ce n'est pas un
détail : régler les seuils sur ce nouvel univers, c'est recommencer l'erreur
de `near_high_252_threshold: 0.85  # assoupli de 0.90`.

## Ce que l'observation ne prouve pas

**Observer n'est pas valider.** Les propositions s'accumulent pour montrer que
les agents trouvent des événements, et combien. Qu'elles rapportent se mesurera
hors échantillon, avec un protocole écrit avant, comme pour tous les agents.

**Les paires sont exposées au hasard des tests multiples.** Tester ~1 450
paires au seuil de 5 %, c'est en déclarer environ 70 cointégrées par pur
hasard à chaque fenêtre. Les filtres successifs en éliminent la plupart, pas
toutes : une paire validée est un candidat, pas une preuve.

**Les filtres de paires sont prudents — mesuré.** Sur 100 paires réellement
cointégrées par construction (demi-vie de 6 séances, 800 séances d'historique),
ils n'en acceptent que **54**. Le premier motif de rejet est la demi-vie
estimée sous 5 jours (38 cas) : l'estimation est bruitée, et le seuil coupe
aussi des paires vraies. C'est un choix de conception (les paires trop rapides
coûtent en frais), pas un défaut à corriger — mais il faut le savoir en lisant
un nombre de paires validées.

## Premier passage réel — 2026-09-13

- **Collecte** : 503 membres sur 503, 67 s en local, aucune série rejetée.
- **Paires** : 1 453 candidates examinées, **4 validées** — ACGL/CB et
  ALL/TRV (assurance dommages), MSCI/NDAQ et MSCI/SPGI (indices et données
  financières). Aucune assez écartée ce jour-là pour une proposition.
- **Form 4** : le premier passage s'est arrêté au Labor Day (7 septembre).
  EDGAR répond **403**, et non 404, à un fichier absent : le flux prenait un
  jour férié pour un blocage. Corrigé — un jour sans index est ignoré, et un
  vrai blocage se reconnaît à sa durée (trois jours ouvrés consécutifs sans
  index) ou à plus de 10 % de dépôts illisibles. Dans les deux cas l'agent se
  déclare en panne au lieu de rendre une liste vide, qui passerait pour un
  mois sans achat.
- **InsiderCluster, après correction** : 500 sociétés suivies, 2 111 Form 4
  lus sur 22 jours ouvrés, aucun illisible. **20 achats sur le marché par des
  dirigeants, dans 16 sociétés — et aucun groupe** (deux dirigeants distincts
  à 100 000 $ ou plus). Premier enseignement : sur les grandes capitalisations,
  l'événement que cherche l'agent est rare. La validation devra mesurer sa
  fréquence réelle avant son rendement — la littérature trouve d'ailleurs cet
  effet surtout dans les petites capitalisations.
- **Isolation vérifiée en production** : pendant cette panne, l'agent des
  paires a tourné normalement, et le service s'est marqué en échec au lieu
  de sortir en silence.

## Changer de fournisseur de prix

yfinance a été retenu « pour le moment ». Le jour où l'équipe paiera un
fournisseur, il suffira d'écrire une classe respectant `PriceSource` dans
`src/data/prices.py`.
