#!/usr/bin/env bash
# deploy/bootstrap.sh — durcissement initial du serveur.
#
# À lancer une fois, en root, sur une machine Ubuntu 24.04 neuve.
# Idempotent : le relancer ne casse rien.
#
# Ce qu'il fait :
#   - crée l'utilisateur `milan`, sans privilèges (rien ne tourne en root)
#   - ferme tout sauf SSH, HTTP et HTTPS
#   - désactive la connexion SSH par mot de passe (clé uniquement)
#   - installe les dépendances système
#
# Ce qu'il ne fait PAS : toucher à un identifiant. Aucun secret ici.
set -euo pipefail

APP_USER=milan
APP_DIR=/opt/milan

log()  { printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
  echo "À lancer en root." >&2
  exit 1
fi

log "Paquets système"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  python3 python3-venv python3-pip python3-dev build-essential \
  git curl unzip \
  xvfb x11-utils \
  openjdk-21-jre-headless \
  ufw fail2ban \
  ca-certificates
ok "installés"

# `python3` plutôt qu'une version figée : le paquet `python3.12` n'existe pas
# sur une Ubuntu plus récente, et le script échouerait dès la première ligne
# pour une raison sans rapport avec le projet. Le contrôle de version se fait
# juste en dessous, sur ce qui est réellement installé.
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)'; then
  ok "Python $PY_VER"
else
  echo "Python $PY_VER trop ancien — le projet exige 3.12 ou plus." >&2
  exit 1
fi

# IB Gateway est une application Java avec interface graphique : sans écran,
# elle refuse de démarrer. Xvfb en fournit un qui n'existe qu'en mémoire.
# openjdk-21-jre-headless suffit — le Gateway embarque sa propre JVM, mais IBC
# a besoin d'un java pour son lanceur.

log "Utilisateur applicatif"
if id "$APP_USER" &>/dev/null; then
  ok "$APP_USER existe déjà"
else
  # --disabled-password : ce compte ne sert qu'aux services, on ne s'y connecte
  # jamais directement.
  adduser --system --group --shell /bin/bash --home "/home/$APP_USER" \
          --disabled-password "$APP_USER"
  ok "$APP_USER créé"
fi

log "Répertoire applicatif"
mkdir -p "$APP_DIR"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
ok "$APP_DIR"

log "Fichier d'échange (swap)"
# 4 Go de RAM suffisent en régime normal, mais le pic est proche : IB Gateway
# occupe ~1,5 Go en permanence, et un run charge pandas/numpy par-dessus.
# Sans swap, un dépassement ponctuel fait tuer un processus par le noyau — en
# pratique le Gateway, c'est-à-dire la connexion au courtier, en pleine séance.
# 2 Go de swap transforment ce cas en simple ralentissement.
if swapon --show | grep -q '/swapfile'; then
  ok "déjà actif"
else
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile >/dev/null
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  # Par défaut Linux échange trop tôt (60). À 10, il n'y touche qu'en cas de
  # vraie pression mémoire — on veut un filet, pas un usage courant du disque.
  sysctl -qw vm.swappiness=10
  grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
  ok "2 Go actifs (swappiness=10)"
fi

log "Pare-feu"
ufw --force reset >/dev/null
ufw default deny incoming  >/dev/null
ufw default allow outgoing >/dev/null
ufw allow 22/tcp  comment 'SSH'   >/dev/null
ufw allow 80/tcp  comment 'HTTP — redirection et renouvellement du certificat' >/dev/null
ufw allow 443/tcp comment 'HTTPS' >/dev/null
ufw --force enable >/dev/null
ok "seuls 22, 80 et 443 sont ouverts"

# Le port 4002 (API IB Gateway) n'est délibérément PAS ouvert : le runner tourne
# sur la même machine et s'y connecte via 127.0.0.1. L'exposer reviendrait à
# offrir un accès au compte de courtage à internet entier.

log "SSH"
SSHD_CONF=/etc/ssh/sshd_config.d/99-milan.conf
cat > "$SSHD_CONF" <<'EOF'
# Connexion par clé uniquement. Un mot de passe SSH exposé à internet finit
# toujours par être testé en force brute.
PasswordAuthentication no
PermitRootLogin prohibit-password
KbdInteractiveAuthentication no
EOF

if ssh-keygen -lf /root/.ssh/authorized_keys &>/dev/null; then
  systemctl reload ssh 2>/dev/null || systemctl reload sshd
  ok "mot de passe désactivé, clé requise"
else
  rm -f "$SSHD_CONF"
  warn "AUCUNE clé SSH trouvée dans /root/.ssh/authorized_keys."
  warn "Configuration SSH laissée intacte — la durcir maintenant te"
  warn "verrouillerait dehors. Ajoute ta clé, puis relance ce script."
fi

log "fail2ban"
systemctl enable --now fail2ban >/dev/null 2>&1 || true
ok "actif"

log "Horloge en UTC"
timedatectl set-timezone UTC
ok "$(date -u '+%Y-%m-%d %H:%M UTC')"

# La machine reste en UTC ; c'est le planificateur qui raisonne en heure de
# New York, pour suivre l'heure d'été américaine sans intervention.

printf '\n\033[1;32m✅ Machine prête.\033[0m\n\n'
echo "Étape suivante :"
echo "  bash $APP_DIR/deploy/install_ibgateway.sh"
echo
