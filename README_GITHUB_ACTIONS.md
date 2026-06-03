# GitHub Actions — Daily Runner Setup

Ce guide explique comment déployer le runner Milan Capital sur GitHub Actions pour qu'il tourne automatiquement du lundi au vendredi à 13h15 UTC.

## Comportement sur GitHub Actions

Le runner détecte que TWS IBKR n'est pas disponible et bascule automatiquement en **mode analyse** :
- Il télécharge les données de marché via yfinance
- Il génère l'order plan complet avec risk manager et allocateur dynamique
- Il envoie le résultat sur Telegram
- **Aucun ordre n'est envoyé** (`EXECUTION_ENABLED=false` forcé côté CI)

## 1. Pousser le projet sur GitHub

```bash
git init
git add .
git commit -m "Initial commit — Milan Capital MQC_ARENA"
git remote add origin https://github.com/<ton-username>/mqc-arena.git
git push -u origin main
```

## 2. Configurer les secrets GitHub

Dans ton dépôt GitHub : **Settings → Secrets and variables → Actions → New repository secret**

Ajoute chacun de ces secrets :

| Secret | Description | Valeur par défaut |
|--------|-------------|-------------------|
| `TELEGRAM_BOT_TOKEN` | Token du bot Telegram | — (obligatoire) |
| `TELEGRAM_CHAT_ID` | ID du chat Telegram | — (obligatoire) |
| `TELEGRAM_APPROVAL_TIMEOUT` | Délai d'approbation en secondes | `900` |
| `IBKR_PORT` | Port TWS (non utilisé sur CI) | `7497` |
| `IBKR_CLIENT_ID` | Client ID IBKR (non utilisé sur CI) | `1` |
| `EXECUTION_ENABLED` | Laisser à `false` sur GitHub Actions | `false` |
| `MAX_ORDERS_PER_RUN` | Nombre max d'ordres par run | `1` |
| `MAX_NOTIONAL_PCT` | Notionnel max par ordre (% NetLiq) | `0.02` |
| `LIMIT_BUFFER_BPS` | Buffer limit order en bps | `10` |
| `RISK_MAX_NET_LONG_PCT` | Exposition long max | `0.40` |
| `RISK_MAX_SINGLE_POSITION_PCT` | Position max par symbole | `0.20` |
| `RISK_MIN_CASH_PCT` | Cash minimum requis | `0.30` |
| `RISK_SELL_ONLY_MODE` | Kill switch ventes uniquement | `false` |
| `MAX_LEVERAGE` | Levier max | `1.0` |
| `MIN_SCORE_THRESHOLD` | Score minimum sélecteur | `0.02` |

## 3. Activer le workflow

Le fichier `.github/workflows/daily_runner.yml` est déjà configuré. Une fois poussé, le workflow s'activera automatiquement.

Pour tester manuellement : **Actions → Daily Runner → Run workflow**

## 4. Déclencher les ordres depuis local

Quand le runner CI envoie le plan sur Telegram et que tu réponds `APPROVE`, comme `EXECUTION_ENABLED=false` sur CI, **aucun ordre n'est envoyé automatiquement**.

Pour exécuter les ordres :
1. Lancer TWS IBKR en local
2. Copier le `plan_id` reçu sur Telegram
3. Lancer le runner localement avec `EXECUTION_ENABLED=true` et le même `plan_id`

Ou laisser le runner local tourner en parallèle — il détecte le `plan_id` déjà généré et ne le double pas (guard dans `execute_plans_paper_ibkr`).

## 5. Structure des logs

Les logs sont générés dans `logs/` :
- `logs/decisions.csv` — signaux de chaque agent par run
- `logs/executions.csv` — exécutions réelles
- `logs/order_plan.csv` — plans d'ordre générés
- `logs/allocator_cache.json` — cache Sharpe rolling (TTL 24h)

Sur GitHub Actions, ces fichiers sont perdus à la fin du job. Pour les persister, ajoute un step `actions/upload-artifact` après le runner.
