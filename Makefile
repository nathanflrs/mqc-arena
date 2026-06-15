# Milan Capital — Makefile
# Usage: make <target>

.PHONY: dashboard run shadow backtest walkforward install

# ── One-button launch ──────────────────────────────────────────────────────────
dashboard:
	@echo "🚀  Milan Capital Dashboard → http://localhost:8000"
	uvicorn src.dashboard.server:app --reload --port 8000 --log-level warning

# ── Fund operations ────────────────────────────────────────────────────────────
run:
	python -m src.arena.runner

shadow:
	python -m src.backtest.shadow_mode

backtest:
	python -m src.backtest.portfolio_backtest

walkforward:
	python -m src.backtest.run_walkforward

# ── Setup ──────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
