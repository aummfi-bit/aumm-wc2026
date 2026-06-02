# Data sources & reference implementations

A vetted list of where to get data and which public work is worth borrowing
from. Read the assessments — not everything here should be used the same way.

## Use these for DATA (download → `data/raw/`)

### Historical results (for fitting team strength)
- **jfjelstul/worldcup** — https://github.com/jfjelstul/worldcup
  Clean, well-structured World Cup data 1930–2022 (matches, tournaments,
  standings) as tidy CSVs. Higher quality and better documented than most
  Kaggle dumps. Best historical World Cup source.
- **International football results 1872–present** (Kaggle, "martj42" is the
  canonical version) — the standard international-results dataset. Needed
  because World-Cup-only data is too sparse to fit team strength; this gives
  every international (incl. friendlies and qualifiers). Expected columns:
  date, home_team, away_team, home_score, away_score, tournament, neutral.
  Filter to neutral-ground and competitive matches when fitting (see CLAUDE.md).

### 2026 tournament structure (for the fixture list to predict)
- **FIFA World Cup 2026 Match Data (Unofficial)** (Kaggle, areezvisram12) —
  https://www.kaggle.com/datasets/areezvisram12/fifa-world-cup-2026-match-data-unofficial
  Complete 2026 schedule, venues, team data in CSV + SQLite. This is the
  FIXTURE LIST the pipeline predicts over. Verify match dates/venues against an
  official source before locking, since it's unofficial.
- **FIFA World Cup Team Dataset** (Kaggle, harrachimustapha) —
  https://www.kaggle.com/datasets/harrachimustapha/fifa-world-cup-team-dataset
  2002–2026 team-level features. Useful supplementary team features.

## Use these as BENCHMARKS only (not ground truth, not to copy)

These are other people's predictions/models. Treat them exactly like market
prices: one more crowd estimate to compare against. None of them is optimized
for the Dacopa 6-tier scoring system, so none maximizes pool points.
- **WC2026 Match Probability Baseline Dataset** (Kaggle, sarazahran1) —
  https://www.kaggle.com/datasets/sarazahran1/wc2026-match-probability-baseline-dataset
- **World Cup 2026 Match Predictor notebook** (Kaggle, sarazahran1) —
  https://www.kaggle.com/code/sarazahran1/world-cup-2026-match-predictor/notebook

## Reference implementations (borrow TECHNIQUE, do not fork wholesale)

- **zvizdo/fifa-wc-2026-simulation** — https://github.com/zvizdo/fifa-wc-2026-simulation
  The most relevant 2026 repo. It's a tournament simulator + Streamlit
  dashboard, NOT a per-match pool predictor, so don't adopt its structure. But
  two ideas were worth taking and are now in our code/notes:
    * **Dixon-Coles correction** on low-scoring outcomes → implemented in
      `poisson_model.py` (`dixon_coles_grid`).
    * **Mean reversion on Elo-style ranks** for the short 3-game group stage:
      after each update, pull the dynamic rank partway back toward the
      pre-tournament FIFA ranking so one fluky result doesn't distort the rest.
      → to implement in `ratings.py`.
  Also uses explicit host advantage: eff_rank = rank × (1 - host_discount × is_host).
- **danielguerreros/WC-Model** — https://github.com/danielguerreros/WC-Model
  (+ Medium write-up: how-to-create-an-international-soccer-match-prediction-model)
  Clean walkthrough of Elo-as-Poisson-covariate. Good technique reference.
- **thezane/soccer-predictions** — https://github.com/thezane/soccer-predictions
  Bivariate Poisson with attack/defense ratings for international tournaments.

## Academic anchors

- Gilch & Müller (2018), "On Elo based prediction models for the FIFA World Cup
  2018" — arxiv.org/pdf/1806.01930. Poisson regression with Elo covariates,
  fit on matches on NEUTRAL ground since 2010. The neutral-ground filter is the
  right call for a World Cup; reflected in `ratings.py` guidance.
- Dixon & Coles (1997) — the original low-score correction now in our model.
- Karlis & Ntzoufras (2003) — bivariate Poisson for football scores.

## Reality check

Every 2026-specific model here is UNPROVEN — the tournament hasn't happened.
Published accuracy figures, even good ones, consistently show draws predicted
far worse than wins/losses. For our pool, draws only ever score on an exact
low-score guess, so that weakness is a real risk to manage, not smooth over.
Triangulate across these sources; treat none as authority.
