#!/usr/bin/env bash
# Chien de garde de la passerelle IBKR.
#
# Pourquoi il existe
# ------------------
# Le 2026-08-15 à 23h45, IB Gateway a fait son redémarrage quotidien
# obligatoire. IBKR a refusé la reconnexion une fois — un incident transitoire,
# pendant leur fenêtre de maintenance. IBC a affiché « Unrecognized Username or
# Password » et n'a JAMAIS réessayé : son log n'a plus bougé pendant cinq jours.
#
# Pendant ce temps, `systemctl is-active ibgateway` répondait « active », parce
# que le script enveloppe et le processus Java vivaient — figés sur une boîte de
# dialogue modale, sur un écran virtuel que personne ne regarde. Quatre séances
# ont été perdues (17, 18, 19, 20 août). Un simple `systemctl restart` a suffi à
# tout rétablir : les identifiants étaient bons depuis le début.
#
# La leçon n'est pas « surveiller IBC ». C'est que **la présence d'un processus
# ne prouve pas la disponibilité d'un service**. On teste donc la capacité
# réelle — le port API accepte-t-il une connexion — et non l'état systemd.
set -uo pipefail

PORT="${IBGW_PORT:-4002}"
ETAT=/var/lib/milan/ibgateway-watchdog.state
COOLDOWN=1200          # 20 min entre deux redémarrages : évite l'emballement
                       # et laisse à IBKR le temps de sa maintenance quotidienne.

mkdir -p "$(dirname "$ETAT")"

if timeout 5 bash -c "</dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then
    [ -f "$ETAT" ] && rm -f "$ETAT"
    echo "ibgateway-watchdog: port $PORT accepte les connexions"
    exit 0
fi

maintenant=$(date +%s)
dernier=$(cat "$ETAT" 2>/dev/null || echo 0)
depuis=$(( maintenant - dernier ))

if [ "$depuis" -lt "$COOLDOWN" ]; then
    echo "ibgateway-watchdog: port $PORT muet, mais redémarrage il y a ${depuis}s — on attend"
    exit 0
fi

echo "ibgateway-watchdog: port $PORT injoignable → redémarrage de ibgateway"
echo "$maintenant" > "$ETAT"
systemctl restart ibgateway

sleep 100
if timeout 5 bash -c "</dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then
    echo "ibgateway-watchdog: rétabli"
else
    # Deux causes possibles : identifiants réellement invalides, ou IBKR
    # indisponible. Le cooldown empêche de marteler le compte et de le faire
    # verrouiller ; le prochain passage réessaiera.
    echo "ibgateway-watchdog: TOUJOURS injoignable après redémarrage — intervention humaine requise"
    exit 1
fi
