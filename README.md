# WC2026 Bolão Predictor

Prediction engine for a 2026 FIFA World Cup prediction pool ("bolão") on
[Dacopa](https://www.dacopa.com/bolao/). Goal: **maximize total pool points**
under Dacopa's specific 6-tier scoring system, with all knockout matches weighted 2×.

The full strategy, the confirmed scoring rules, and the modeling spec live in
[`CLAUDE.md`](./CLAUDE.md) — read that first. It's also the context file
Claude Code reads automatically when you work in this repo. Vetted datasets and
reference implementations are listed in [`data_sources.md`](./data_sources.md).

## How it works

1. **`scoring.py`** — the Dacopa 6-tier scoring ladder, encoded and exhaustively
   tested against the official examples. The most safety-critical file.
2. **`poisson_model.py`** — Dixon-Coles model (independent Poisson + low-score
   correction) producing a scoreline probability distribution per match; a
   bivariate Poisson alternative is also included.
3. **`optimizer.py`** — picks the prediction that maximizes *expected points*
   under the scoring table. This is usually NOT the most likely scoreline.
4. **`ratings.py` / `data_loader.py`** — Elo strength + form from historical
   international results (wire up to your Kaggle CSV in `data/raw/`).
5. **`markets.py`** — optional prediction-market/bookmaker probabilities as a
   benchmark.
6. **`pipeline.py`** — ties it together into per-match prediction records.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place your Kaggle international-results CSV in `data/raw/` (gitignored).

## Run

```bash
# Demo a single match prediction (hand-set expected goals)
PYTHONPATH=src python -m wc2026.pipeline

# Run the tests (scoring.py must always pass)
PYTHONPATH=src pytest tests/
```

## Working with Claude Code

This repo is set up for [Claude Code](https://docs.claude.com/en/docs/claude-code).
Open it in your terminal and Claude Code will read `CLAUDE.md` automatically.
Good first prompts:

- "Wire up `data_loader.py` to the CSV in data/raw and show me the schema."
- "Build the ratings table in `ratings.py` from the historical results."
- "Generate EV-optimal predictions for all 72 group-stage fixtures to outputs/."
- "Before the Round of 16, re-run predictions for the confirmed matchups."

## Scoring rules — verified

The full 6-tier ladder has been verified against Dacopa's live scoring simulator
(see `CLAUDE.md` and `tests/test_scoring.py`). The key correction: **draws score the
goal-difference tier (15/30) on any draw guess** — they are not exact-or-zero. Knockouts
are scored on the **90-minute** result (extra time and penalties don't count, so a game
level at 90' is a draw even if later decided in ET/pens), and weighted 2×.
