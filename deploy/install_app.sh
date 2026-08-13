#!/usr/bin/env bash
# deploy/install_app.sh — le code, le planificateur, le dashboard, le HTTPS.
#
#   bash install_app.sh mon-domaine.com
#
# À lancer après bootstrap.sh et install_ibgateway.sh. Idempotent.
# N'écrit aucun secret : crée un .env vide, à toi de le remplir.
set -euo pipefail

APP_USER=milan
APP_DIR=/opt/milan
DOMAIN="${1:-}"

log()  { printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }

[[ $EUID -eq 0 ]] || { echo "À lancer en root." >&2; exit 1; }
[[ -n "$DOMAIN" ]] || { echo "Usage: bash install_app.sh <domaine>" >&2; exit 1; }

log "Code source"
if [[ -d "$APP_DIR/.git" ]]; then
  sudo -u "$APP_USER" git -C "$APP_DIR" pull --ff-only
else
  git clone https://github.com/nathanflrs/mqc-arena.git "$APP_DIR"
  chown -R "$APP_USER:$APP_USER" "$APP_DIR"
fi
ok "à jour"

log "Environnement Python"
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv" 2>/dev/null || true
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"
ok "dépendances installées"

log "Fichier d'environnement"
ENV_FILE="$APP_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
  ok ".env existe déjà — laissé intact"
else
  cat > "$ENV_FILE" <<'EOF'
# ─── Milan Capital — configuration serveur ───────────────────────────────────
# Ce fichier contient des secrets. Il n'est jamais commité (voir .gitignore).

# ─── Exécution ───────────────────────────────────────────────────────────────
# Laisser à false jusqu'au premier run manuel concluant (étape 4 du README).
EXECUTION_ENABLED=false
MAX_ORDERS_PER_RUN=3
MAX_NOTIONAL_PCT=0.02
LIMIT_BUFFER_BPS=10

# ─── Courtier ────────────────────────────────────────────────────────────────
# 4002 = port API d'IB Gateway en mode paper. Le Gateway tourne sur cette même
# machine ; ce port n'est pas ouvert sur internet, et ne doit jamais l'être.
IBKR_PORT=4002
IBKR_CLIENT_ID=1

# ─── Risque ──────────────────────────────────────────────────────────────────
RISK_MAX_NET_LONG_PCT=0.60
RISK_MAX_SINGLE_POSITION_PCT=0.20
RISK_MIN_CASH_PCT=0.30
RISK_SELL_ONLY_MODE=false
STOP_LOSS_PCT=0.07
MAX_LEVERAGE=1.0
MIN_SCORE_THRESHOLD=0.02

# ─── Dashboard ───────────────────────────────────────────────────────────────
# Générer : python -c "import secrets; print(secrets.token_hex(32))"
SESSION_SECRET=
PORT=8000

# ─── Clés externes ───────────────────────────────────────────────────────────
ANTHROPIC_API_KEY=
FINNHUB_API_KEY=
FRED_API_KEY=
EOF
  chown "$APP_USER:$APP_USER" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  warn ".env créé, vide. À compléter : nano $ENV_FILE"
fi

log "Services systemd"
install -m 644 "$APP_DIR/deploy/systemd/milan-dashboard.service" /etc/systemd/system/
install -m 644 "$APP_DIR/deploy/systemd/milan-run.service"       /etc/systemd/system/
install -m 644 "$APP_DIR/deploy/systemd/milan-run.timer"         /etc/systemd/system/
systemctl daemon-reload
systemctl enable milan-dashboard >/dev/null
ok "dashboard activé ; le planificateur reste éteint (étape 5)"

log "Caddy — HTTPS automatique"
if ! command -v caddy &>/dev/null; then
  apt-get install -y -qq debian-keyring debian-archive-keyring apt-transport-https
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -qq && apt-get install -y -qq caddy
fi

# Caddy obtient et renouvelle seul le certificat Let's Encrypt. C'est ce qui
# rend possibles les notifications web sur le téléphone : un navigateur refuse
# de les activer sur une connexion non chiffrée.
cat > /etc/caddy/Caddyfile <<EOF
$DOMAIN {
    encode zstd gzip
    reverse_proxy 127.0.0.1:8000

    # Le dashboard expose un flux d'événements en continu (SSE). Sans ce
    # réglage, Caddy attend la fin de la réponse avant de la transmettre — et
    # le flux n'arrive jamais.
    reverse_proxy /api/events/stream 127.0.0.1:8000 {
        flush_interval -1
    }

    header {
        Strict-Transport-Security "max-age=31536000;"
        X-Content-Type-Options    "nosniff"
        X-Frame-Options           "DENY"
        Referrer-Policy           "no-referrer"
    }
}
EOF
systemctl reload caddy 2>/dev/null || systemctl restart caddy
ok "HTTPS configuré pour $DOMAIN"

printf '\n\033[1;32m✅ Application installée.\033[0m\n\n'
cat <<EOF
Étapes suivantes :

  1. Compléter les secrets
       nano $ENV_FILE

  2. Créer ton compte du dashboard
       cd $APP_DIR && sudo -u $APP_USER .venv/bin/python -m src.dashboard.create_user

  3. Démarrer le dashboard
       systemctl start milan-dashboard
       curl -sI https://$DOMAIN

  4. Un run manuel, SANS exécution, pour valider la chaîne complète
       cd $APP_DIR
       sudo -u $APP_USER EXECUTION_ENABLED=false RUN_TRIGGER=manual \\
            .venv/bin/python -m src.arena.runner

     Doit afficher : ✅ IBKR connected | NetLiq=... avec ton vrai solde paper.

  5. SEULEMENT si l'étape 4 est concluante — activer l'automatique
       nano $ENV_FILE          # EXECUTION_ENABLED=true
       systemctl enable --now milan-run.timer
       systemctl list-timers milan-run

EOF
