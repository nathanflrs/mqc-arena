# Mettre Milan Capital sur un serveur

*Guide pas à pas. Chaque étape se vérifie avant de passer à la suivante.*

---

## Ce qu'on installe, et pourquoi

Une seule machine porte tout :

```
Serveur Ubuntu 24.04
├── Xvfb + IB Gateway + IBC ... la connexion permanente à IBKR
├── milan-run.timer ........... lance le fonds chaque jour de bourse
├── milan-dashboard ........... l'écran, consultable depuis le téléphone
└── Caddy ..................... HTTPS automatique
```

**Pourquoi une seule machine.** Aujourd'hui le calcul tourne sur GitHub, les
données sont lues via l'API GitHub, et le portefeuille vit ailleurs. Trois
systèmes à garder synchronisés, et c'est précisément ce qui produit les
incohérences. Ici, une seule source de vérité.

**Pourquoi Xvfb.** IB Gateway est une application Java avec une interface
graphique. Elle refuse de démarrer sans écran, même quand personne ne la
regarde. Xvfb fournit un écran qui n'existe que dans la mémoire de la machine.

**Pourquoi IBC.** IBKR déconnecte le Gateway une fois par jour et redemande
les identifiants. IBC (logiciel libre) saisit la connexion et gère ce
redémarrage, sans quoi le fonds s'arrêterait chaque nuit.

---

## Règle de sécurité, à ne pas contourner

> **Compte IBKR *paper* uniquement sur ce serveur. Jamais le compte réel.**

Les identifiants doivent être écrits dans un fichier lisible par le Gateway,
donc présents en clair sur la machine. Sur un compte paper, le pire scénario
est une perte d'argent fictif. Sur un compte réel, ce serait tout autre chose.

Aucun identifiant ne doit jamais être commité. `deploy/` ne contient que des
gabarits.

---

## Étape 0 — Commander la machine

Chez [Hetzner Cloud](https://console.hetzner.com) :

| Réglage | Valeur |
|---|---|
| Type | **CX22** (2 vCPU, 4 Go RAM) — IB Gateway est gourmand en mémoire |
| Image | **Ubuntu 24.04** |
| Localisation | Allemagne ou Finlande |
| Clé SSH | La tienne (voir ci-dessous) |

Si tu n'as pas de clé SSH, sur ton Mac :

```bash
ssh-keygen -t ed25519 -C "milan-capital" -f ~/.ssh/milan_capital
```

La clé **publique** à coller dans Hetzner :

```bash
cat ~/.ssh/milan_capital.pub
```

Ne colle jamais l'autre fichier (celui sans `.pub`) — c'est la clé privée, elle
ne quitte jamais ta machine.

**Vérification :** tu dois pouvoir te connecter.

```bash
ssh -i ~/.ssh/milan_capital root@<IP_DU_SERVEUR>
```

---

## Étape 1 — Durcir la machine

Sur le serveur, en `root` :

```bash
curl -fsSL https://raw.githubusercontent.com/nathanflrs/mqc-arena/main/deploy/bootstrap.sh | bash
```

Ou, si tu préfères lire avant d'exécuter — ce qui est le bon réflexe :

```bash
git clone https://github.com/nathanflrs/mqc-arena.git /opt/milan
bash /opt/milan/deploy/bootstrap.sh
```

Ce script crée un utilisateur `milan` sans privilèges, ferme tout sauf SSH et
HTTPS, désactive la connexion par mot de passe, et installe les dépendances.

**Vérification :**

```bash
sudo -u milan whoami     # → milan
ufw status               # → 22, 80, 443 seulement
```

---

## Étape 2 — IB Gateway et IBC

```bash
bash /opt/milan/deploy/install_ibgateway.sh
```

Le script installe le Gateway et IBC, puis crée
`/home/milan/ibc/config.ini` avec des champs vides.

**C'est toi qui remplis les identifiants**, jamais le script, jamais moi :

```bash
sudo -u milan nano /home/milan/ibc/config.ini
```

Deux lignes à compléter :

```ini
IbLoginId=TON_IDENTIFIANT_PAPER
IbPassword=TON_MOT_DE_PASSE_PAPER
```

Puis verrouille le fichier :

```bash
chmod 600 /home/milan/ibc/config.ini
```

> **Avant cette étape**, désactive la double authentification **sur le compte
> paper uniquement**, depuis le portail IBKR. Sans ça, la connexion automatique
> restera bloquée en attente d'une validation sur ton téléphone. Ne touche
> jamais à celle du compte réel.

**Vérification :**

```bash
systemctl start ibgateway && sleep 60
systemctl status ibgateway            # → active (running)
ss -tlnp | grep 4002                  # → le port d'écoute répond
```

---

## Étape 3 — L'application

```bash
bash /opt/milan/deploy/install_app.sh <ton-domaine.com>
```

Installe le code, l'environnement Python, les services, et Caddy qui obtient
un certificat HTTPS automatiquement.

Le script crée `/opt/milan/.env` vide. À compléter avec tes clés
(`ANTHROPIC_API_KEY`, `FINNHUB_API_KEY`, `SESSION_SECRET`…) :

```bash
sudo -u milan nano /opt/milan/.env
chmod 600 /opt/milan/.env
```

Pour créer ton compte du dashboard :

```bash
cd /opt/milan && sudo -u milan .venv/bin/python -m src.dashboard.create_user
```

**Vérification :**

```bash
systemctl status milan-dashboard      # → active (running)
systemctl list-timers milan-run       # → prochaine exécution affichée
curl -sI https://<ton-domaine.com>    # → HTTP/2 200
```

---

## Étape 4 — Le premier run, à la main

Avant de laisser le planificateur travailler seul, un run manuel, exécution
**désactivée** :

```bash
cd /opt/milan
sudo -u milan EXECUTION_ENABLED=false RUN_TRIGGER=manual .venv/bin/python -m src.arena.runner
```

Il doit afficher `✅ IBKR connected | NetLiq=…` avec ton vrai solde paper. Si
c'est le cas, la chaîne complète fonctionne.

---

## Étape 5 — Armer le système

**Seulement une fois l'étape 4 concluante.**

Il y a **deux verrous distincts**, volontairement séparés. Tant que l'un des
deux est fermé, aucun ordre ne peut partir.

**Verrou 1 — le courtier.** Dans `/home/milan/ibc/config.ini` :

```ini
ReadOnlyApi=no
```

Tant qu'il vaut `yes`, IB Gateway refuse tout ordre quoi que demande le code.
C'est une sécurité côté courtier : un bug, une erreur de configuration ou un
run inattendu ne peuvent rien envoyer.

```bash
systemctl restart ibgateway
```

**Verrou 2 — le code.** Dans `/opt/milan/.env` :

```ini
EXECUTION_ENABLED=true
```

Puis :

```bash
systemctl restart milan-dashboard
systemctl enable --now milan-run.timer
```

À partir de là, le fonds trade seul.

> Pour tout arrêter en urgence, il suffit de refermer **un seul** des deux.
> Le plus rapide : `ReadOnlyApi=yes` puis `systemctl restart ibgateway`.

---

## Vérifier que ça tourne, tous les jours

Le meilleur indicateur est le dashboard lui-même : le bandeau doit afficher
**LIVE en vert**. S'il passe en ambre (`DONNÉES · N j`), c'est que plus rien ne
tourne depuis plus de 72 h — et tu le vois d'un coup d'œil depuis ton
téléphone.

En ligne de commande :

```bash
systemctl status milan-run.timer
journalctl -u milan-run --since today
```

---

## Arrêter, et arrêter de payer

```bash
systemctl disable --now milan-run.timer   # stoppe le trading, garde le reste
```

Pour arrêter la facturation, il faut **supprimer** le serveur depuis la console
Hetzner — l'éteindre ne suffit pas, le stockage reste facturé.
