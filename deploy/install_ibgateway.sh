#!/usr/bin/env bash
# deploy/install_ibgateway.sh — IB Gateway + IBC, en service qui se relance seul.
#
# Pourquoi c'est nécessaire : IBKR n'expose pas d'API web pour les comptes
# particuliers. L'authentification passe obligatoirement par une application de
# bureau qui maintient la session — d'où une application Java sur un serveur,
# et Xvfb pour lui donner l'écran qu'elle exige.
#
# IBC (IbcAlpha/IBC, logiciel libre) saisit la connexion et gère le
# redémarrage quotidien qu'IBKR impose. Sans lui, le fonds s'arrêterait chaque
# nuit en attendant que quelqu'un tape un mot de passe.
#
# Ce script n'écrit AUCUN identifiant. Il crée un gabarit vide, à toi de le
# remplir.
set -euo pipefail

APP_USER=milan
HOME_DIR="/home/$APP_USER"
# Version figée plutôt que « dernière en date » : une installation doit donner
# le même résultat aujourd'hui et dans six mois. Pour changer :
#   IBC_VERSION=3.25.0 bash install_ibgateway.sh
# Versions publiées : https://github.com/IbcAlpha/IBC/releases
IBC_VERSION="${IBC_VERSION:-3.24.1}"
IBC_DIR="$HOME_DIR/ibc"
GW_DIR="$HOME_DIR/Jts"

log()  { printf '\n\033[1;34m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }

[[ $EUID -eq 0 ]] || { echo "À lancer en root." >&2; exit 1; }

# Emplacement exigé par IBC. Lu dans son propre code (scripts/ibcstart.sh) :
#
#     gateway_program_path="${tws_path}/ibgateway/${tws_version}"
#
# soit $TWS_PATH/ibgateway/<version sans point>/jars. L'installateur autonome,
# lui, produit une arborescence PLATE dans ~/ibgateway, sans dossier de
# version — la version n'apparaît que dans le nom du raccourci « IB Gateway
# 10.45.desktop ». Les deux conventions ne coïncident pas, et IBC s'arrête
# alors sur « can't find jars folder ».
#
# On installe donc dans un emplacement provisoire, on lit la version dans le
# nom du raccourci, puis on range à l'endroit attendu. La version n'est
# connaissable qu'après l'installation : c'est pour ça que l'ordre est
# installer → détecter → déplacer, et pas l'inverse.
log "IB Gateway"
if compgen -G "$GW_DIR/ibgateway/*/jars" >/dev/null; then
  ok "déjà installé"
else
  # Dossier temporaire créé PAR milan : `mktemp -d` lancé en root produit un
  # dossier en 700 root, que `sudo -u milan` ne peut ensuite ni lire ni
  # exécuter — l'installateur échouait sur « Permission denied ».
  TMP=$(sudo -u "$APP_USER" mktemp -d)
  # URL « stable-standalone » : toujours la dernière version stable, pas de
  # numéro à mettre à jour dans ce script.
  curl -fsSL -o "$TMP/gw.sh" \
    https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh
  chown "$APP_USER:$APP_USER" "$TMP/gw.sh"
  chmod +x "$TMP/gw.sh"

  STAGING="$HOME_DIR/.gw-staging"
  rm -rf "$STAGING"
  sudo -u "$APP_USER" HOME="$HOME_DIR" "$TMP/gw.sh" -q -dir "$STAGING"
  rm -rf "$TMP"

  # « IB Gateway 10.45.desktop » → 10.45 → 1045, la forme attendue par IBC
  # (son propre exemple utilise 1019).
  DESKTOP=$(find "$STAGING" -maxdepth 1 -name 'IB Gateway *.desktop' | head -1)
  DOTTED=$(basename "$DESKTOP" .desktop | sed -E 's/^IB Gateway //')
  if [[ -z "$DOTTED" ]]; then
    echo "Version illisible : aucun raccourci 'IB Gateway *.desktop' dans $STAGING" >&2
    ls -la "$STAGING" >&2
    exit 1
  fi
  VER="${DOTTED//./}"

  TARGET="$GW_DIR/ibgateway/$VER"
  sudo -u "$APP_USER" mkdir -p "$GW_DIR/ibgateway"
  rm -rf "$TARGET"
  mv "$STAGING" "$TARGET"
  chown -R "$APP_USER:$APP_USER" "$GW_DIR"
  ok "installé — version $DOTTED dans $TARGET"
fi

log "IBC $IBC_VERSION"
if [[ -f "$IBC_DIR/gatewaystart.sh" ]]; then
  ok "déjà installé"
else
  TMP=$(mktemp -d)
  curl -fsSL -o "$TMP/ibc.zip" \
    "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip"
  mkdir -p "$IBC_DIR"
  unzip -oq "$TMP/ibc.zip" -d "$IBC_DIR"
  chmod +x "$IBC_DIR"/*.sh "$IBC_DIR"/scripts/*.sh 2>/dev/null || true
  chown -R "$APP_USER:$APP_USER" "$IBC_DIR"
  rm -rf "$TMP"
  ok "installé dans $IBC_DIR"
fi

log "Configuration IBC"
CONF="$IBC_DIR/config.ini"

# IBC livre son propre config.ini d'exemple, avec `IbLoginId=edemo`,
# `IbPassword=demouser` et surtout **TradingMode=live**.
#
# Une simple condition « IbLoginId a-t-il une valeur ? » voyait `edemo`,
# concluait « déjà configuré », et laissait le mode `live` en place. Les
# identifiants de démonstration ne se connectent à rien, mais le mode, lui,
# serait resté — et une fois de vrais identifiants saisis, IBC aurait visé le
# compte réel. On considère donc explicitement `edemo` comme « non configuré ».
if [[ -f "$CONF" ]] \
   && grep -qE '^IbLoginId=.+' "$CONF" \
   && ! grep -qE '^IbLoginId=edemo\s*$' "$CONF"; then
  ok "config.ini déjà renseigné — laissé intact"
  if ! grep -qE '^TradingMode=paper\s*$' "$CONF"; then
    warn "ATTENTION : TradingMode n'est pas 'paper' dans $CONF"
    warn "Corrige-le avant de démarrer le service."
  fi
else
  [[ -f "$CONF" ]] && mv "$CONF" "$CONF.ibc-default.bak"
  cat > "$CONF" <<'EOF'
# ─────────────────────────────────────────────────────────────────────────────
# IBC — configuration Milan Capital
#
# ⚠️  COMPTE PAPER UNIQUEMENT. Jamais les identifiants du compte réel.
#     Ce fichier contient un mot de passe en clair : c'est la contrainte
#     d'IBKR, qui n'offre pas d'authentification par jeton aux particuliers.
#     Sur un compte paper, le pire cas est une perte d'argent fictif.
#
# Remplis les deux lignes ci-dessous, puis : chmod 600 sur ce fichier.
# ─────────────────────────────────────────────────────────────────────────────

IbLoginId=
IbPassword=

# 'paper' refuse de se connecter à un compte réel même si les identifiants en
# étaient un. C'est une ceinture de sécurité, pas un simple réglage.
TradingMode=paper

# Gateway plutôt que TWS : pas d'interface complète, beaucoup moins de mémoire.
FIX=no

# Accepte la connexion du runner, qui tourne sur cette même machine.
AcceptIncomingConnectionAction=accept
AcceptNonBrokerageAccountWarning=yes
AllowBlindTrading=yes

# IBKR force une déconnexion quotidienne. On la place à 23:45 New York, après
# la clôture et bien avant le run du lendemain matin.
ClosedownAt=
RestartAfterHalt=yes
AutoRestartTime=11:45 PM

# Ferme les boîtes de dialogue qui, sinon, bloqueraient le démarrage sans que
# personne ne soit là pour cliquer.
DismissPasswordExpiryWarning=yes
DismissNSEComplianceNotice=yes
ExistingSessionDetectedAction=primary
EOF
  chown "$APP_USER:$APP_USER" "$CONF"
  chmod 600 "$CONF"
  warn "config.ini créé SANS identifiants."
  warn "À remplir : sudo -u $APP_USER nano $CONF"
fi

log "Service systemd"

# IBC doit savoir quelle version majeure du Gateway lancer. Coder « 1030 » en
# dur serait faux dès la prochaine mise à jour d'IBKR, et l'échec serait
# obscur : IBC démarrerait puis ne trouverait rien. On lit donc la version
# réellement installée.
# Le dossier de version EST la valeur attendue par IBC : c'est ainsi qu'il
# reconstruit le chemin du programme.
IBC_MAJOR=$(find "$GW_DIR/ibgateway" -maxdepth 1 -mindepth 1 -type d \
            -printf '%f\n' 2>/dev/null | sort -V | tail -1)
if [[ -z "$IBC_MAJOR" || ! -d "$GW_DIR/ibgateway/$IBC_MAJOR/jars" ]]; then
  echo "Installation incomplète : $GW_DIR/ibgateway/<version>/jars introuvable." >&2
  find "$GW_DIR" -maxdepth 3 2>/dev/null | head -30 >&2
  echo "Supprime $GW_DIR et relance ce script." >&2
  exit 1
fi
ok "version détectée : $IBC_MAJOR (jars présents)"

cat > /etc/systemd/system/ibgateway.service <<EOF
[Unit]
Description=IB Gateway (via IBC, écran virtuel Xvfb)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
Environment=DISPLAY=:1
Environment=TWS_MAJOR_VRSN=$IBC_MAJOR
Environment=IBC_INI=$IBC_DIR/config.ini
Environment=IBC_PATH=$IBC_DIR
Environment=TWS_PATH=$GW_DIR

# Xvfb fournit l'écran que le Gateway exige, sans carte graphique ni personne
# devant. La taille n'a pas d'importance en soi, mais trop petite, certaines
# boîtes de dialogue se dessinent mal et IBC ne les retrouve plus pour les
# fermer — le démarrage reste alors bloqué sur une fenêtre invisible.
#
# Le préfixe '-' rend l'échec non fatal : au premier démarrage aucun Xvfb ne
# tourne, et pkill sort en erreur. Sans lui, le service refusait de démarrer.
# (systemd n'interprète pas le shell : un ';' aurait été passé comme argument.)
ExecStartPre=-/usr/bin/pkill -f "Xvfb :1"
ExecStartPre=/bin/bash -c '/usr/bin/Xvfb :1 -screen 0 1024x768x24 -nolisten tcp & sleep 3'
ExecStart=$IBC_DIR/gatewaystart.sh

# Le Gateway peut tomber : session expirée, coupure réseau, maintenance IBKR.
# On relance toujours, sans limite de tentatives — un fonds qui reste
# déconnecté jusqu'au lendemain matin ne remplit pas son office.
Restart=always
RestartSec=30

# Durcissement : le service ne peut écrire que dans son propre dossier.
NoNewPrivileges=yes
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$HOME_DIR

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ibgateway >/dev/null
ok "ibgateway.service activé (non démarré)"

printf '\n\033[1;32m✅ IB Gateway installé.\033[0m\n\n'
echo "Avant de démarrer :"
echo "  1. Désactive la double authentification sur ton compte PAPER"
echo "  2. sudo -u $APP_USER nano $CONF     # identifiants"
echo "  3. chmod 600 $CONF"
echo "  4. systemctl start ibgateway && sleep 60 && systemctl status ibgateway"
echo "  5. ss -tlnp | grep 4002              # le port doit écouter"
echo
