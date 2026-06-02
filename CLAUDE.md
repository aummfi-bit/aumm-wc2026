# CLAUDE.md — WC2026 Bolão Predictor

Context and standing instructions for Claude Code working in this repo. Read this
fully before writing or changing code.

## What this project is

A prediction engine for a **2026 FIFA World Cup prediction pool ("bolão")** hosted on
Dacopa (`app.dacopa.com`). The goal is to **maximize total points in the pool**, not to
build an academically perfect forecaster. Every modeling decision should be justified by
"does this raise expected pool points?" — not by forecast accuracy in the abstract.

The tournament: 48 teams, 104 matches, hosted across USA, Canada, and Mexico. Group
stage (72 matches) then a knockout bracket of 32 matches (Round of 32 → Round of 16 →
Quarters → Semis → Third-place → Final).

## The scoring system (CONFIRMED against the live Dacopa scoring simulator + the dacopa.com/bolao/regras rules page — do not re-derive)

Per match you predict a **scoreline** (e.g. 2–1). Points are awarded on a 6-tier ladder
based on how close the prediction is. Tiers are checked top-down; you get the highest one
that matches.

| Tier | Condition | Group pts | Knockout pts (2×) |
|------|-----------|-----------|-------------------|
| 1 | Exact score | 25 | 50 |
| 2 | Correct winner + winner's goal count correct | 18 | 36 |
| 3 | Correct **signed** goal difference (right winner + right margin — *or any draw guess on a real draw*) | 15 | 30 |
| 4 | Correct winner + loser's goal count correct | 12 | 24 |
| 5 | Correct winner only | 10 | 20 |
| 6 | Nothing correct | 0 | 0 |

### Critical scoring subtleties (these drive strategy — verified against the official examples)

1. **Draws score the goal-difference tier — they are NOT exact-or-zero.** A real draw has
   no winner, so tiers 2/4/5 (which all require a winner) can't apply. But every draw
   prediction has goal difference 0, which matches a real draw's 0, so **any draw guess on
   a real draw scores tier 3 = 15** (30 in knockouts), and **25/50 if the score is also
   exact**. Only a *decisive* prediction on a real draw scores 0. (Dacopa simulator, real
   3–3: guess 3–3 → 25; 2–2 / 1–1 / 0–0 → 15; 3–0 / 3–1 / 2–1 → 0.) This makes predicting a
   draw a **low-variance, solid-floor play**, not the gamble an earlier draft assumed —
   strategically important, especially in knockouts (see #3).

   **Tier 3 is the SIGNED goal difference**, which resolves the old "ambiguity": real 2–1
   vs guess 1–0 scores **15**, because 1–0's diff +1 equals the real +1 (it is *not* 10).
   `scoring.py` implements exactly this and is simulator-verified — no config flag needed.

2. **All knockout matches are weight 2×** (Round of 32 through Final, including
   third-place). 32 knockout matches × 2 ≈ the weight of 64 group matches, nearly equal to
   all 72 group games combined. **Knockout precision is where the pool is won.**

3. **Knockouts are scored at the end of regulation (90 min) ONLY.** Extra-time goals and
   the penalty shootout DO NOT count. A game that is 1–1 after 90, then won 2–1 in extra
   time (or decided on pens), is scored as **1–1** for prediction purposes. → Every KO game
   that reaches extra time is scored as its (usually level) 90' result, so **regulation
   draws are far more common in KO scoring than intuition suggests** — up-weight KO draw
   probability accordingly, and a correct KO draw call pays a **30-point floor** (15 × 2).
   Modeling implication: the KO model should predict the 90-minute score, and historical KO
   results must be stored as 90-minute scores (ET goals and pens stripped). (Confirmed
   against dacopa.com/bolao/regras: *"Vale o placar do tempo regulamentar (90 minutos). Gols
   na prorrogação não contam. Pênaltis não contam."*)

4. **WO/walkover** counts as the real 3–0 result. **Postponed** games keep predictions
   valid until the rescheduled kickoff; **cancelled** games void predictions.

### Tiebreakers (in order)
1. Total points
2. Most exact scorelines (tier-1 hits)
3. Earliest join date

→ Exact scores have **hidden value** as the primary tiebreaker. In a tight end-game,
shift toward chasing exact scores in spots where the field plays safe.

### Prediction mechanics
- Predict each match's scoreline individually; locks at that match's kickoff.
- Editable unlimited times until lock; future matches can be predicted days ahead.
- **No pre-tournament bracket commitment.** You predict each knockout game once the two
  teams are known. You are never penalized for a wrong bracket — only wrong scorelines.
  This means knockout predictions are made with full information, one match at a time.
- Empty/missing prediction = 0 points for that match.

## Strategy (derived from the scoring system above)

1. **Winner accuracy is the foundation.** Tiers 2–5 all require the correct winner first.
   Maximize win/draw/loss calibration before optimizing scoreline precision.
2. **The model outputs a full scoreline probability distribution per match**, then a
   separate optimizer picks the prediction that **maximizes expected points under the
   exact 6-tier table for that phase's weight.** The highest-EV prediction is usually NOT
   the single most-likely scoreline — it's typically a common favorite scoreline (1–0,
   2–1, 2–0) that maximizes the chance of landing somewhere on the partial-credit ladder.
3. **Goal-difference tier (tier 3) is the high-value safe play** for decisive games: pick
   the right winner and the most likely margin (margins cluster at 1 and 2). Banks 15/30
   reliably with tier 1 as upside.
4. **Draws are a floor play, not a gamble.** A correctly-called draw banks 15/30 off *any*
   draw scoreline (25/50 if exact), so predict a draw whenever P(draw) is high enough that
   its EV beats the best decisive pick — not only when chasing an exact 0–0/1–1. 1–1 and
   0–0 remain the best draw guesses because they also carry the exact-score upside and the
   tiebreaker. In knockouts a game is scored as a draw whenever it is level at 90' — which
   includes every game that goes to extra time — so KO draws are *more* common, not less;
   lean into them, with a 30-point floor on a correct call.
5. **Neutral venues, not home advantage.** Almost all matches are at neutral sites. Apply
   a host boost ONLY to USA, Canada, Mexico when playing in their own country. Do NOT
   apply generic home advantage — this is a common mistake the field will make.
6. **World Cup-specific calibration:** favorites win ~55% of WC matches (lower than club
   leagues); upsets more frequent; knockouts more conservative (lower-scoring); already-
   qualified teams may rest players in the 3rd group game (dead-rubber effect).
7. **Opponent modeling.** The field is largely reading Dacopa's generic tip sheet
   (anchor on 1–0/2–1/1–1, "pick the home team"). Edge comes from (a) better-calibrated
   favorite/margin picks via a fitted model, and (b) deliberate differentiation on matches
   the crowd will misprice: neutral-venue games where they wrongly apply home advantage,
   knockout draws they wrongly call decisive, and dead rubbers.

## Modeling approach

- **Core model: independent Poisson + Dixon-Coles correction** for scorelines
  (`poisson_model.py:dixon_coles_grid`). Dixon-Coles re-weights the low-scoring cells
  (0-0, 1-0, 0-1, 1-1) where independent Poisson is empirically wrong. This is the default
  because those low-score cells are precisely the high-value ones for us: any correct draw
  call already banks the goal-difference tier (15/30), and nailing the *exact* low-score
  draw (0–0, 1–1) adds the jump to 25/50 plus the exact-score tiebreaker. A bivariate Poisson is
  also implemented (`bivariate_poisson_grid`) as an alternative correlation mechanism. Use
  ONE, not both — stacking double-counts the correlation. The `rho` parameter (~ -0.05 to
  -0.15 for international football) should be fit on historical data.
- **Strength ratings:** Elo (or world-football Elo) as the primary strength input, plus a
  recency-weighted form term. National teams play infrequently, so weight tournament and
  competitive matches over friendlies, and prefer fitting on NEUTRAL-ground matches (the
  World Cup is almost entirely neutral venues; see Gilch & Müller in data_sources.md).
- **Short-tournament mean reversion (TODO in ratings.py):** groups play only 3 matches
  before knockout, so a single fluky result can distort a team's rating for the rest of the
  run. After each in-tournament rating update, pull the dynamic rating partway back toward
  the pre-tournament prior: `new = (1 - r) * dynamic + r * base`. Borrowed from the zvizdo
  repo (see data_sources.md). Tune the reversion rate.
- **Inputs/benchmarks:** prediction-market prices (Kalshi, Polymarket), bookmaker odds, AND
  other public WC2026 prediction models/datasets are all **crowd probability estimates** —
  use as a calibration benchmark and optional input, NOT as ground truth. None of the
  public models is optimized for the Dacopa scoring table, so none maximizes pool points.
  When the model and a benchmark disagree sharply, surface it explicitly.
- **Expected-points optimizer:** for each match, enumerate candidate scorelines (0–0 up to
  ~6–6), compute P(scoreline) from the model, then compute expected points over the 6-tier
  table (phase-weighted) for each candidate prediction, and select the argmax. This is the
  heart of the system — the model proposes, the optimizer disposes. The EV-optimal pick is
  usually NOT the modal scoreline.

## Data

See **data_sources.md** for the full vetted list of datasets, reference repos, and
academic anchors, with assessments of how to use each.

- Historical results for fitting: jfjelstul/worldcup (clean WC data) + a broad
  international-results dataset (martj42 on Kaggle). Filter to neutral/competitive when
  fitting. Expected normalized columns: date, home_team, away_team, home_score, away_score,
  tournament, neutral (venue flag).
- 2026 fixtures to predict over: the "FIFA World Cup 2026 Match Data (Unofficial)" Kaggle
  dataset (verify against an official source before locking).
- `data/raw/` = untouched source files (gitignored). `data/processed/` = cleaned/derived.
- Code execution environment may have no network; data is provided locally.

## Repo layout

```
src/wc2026/
  data_loader.py     # load + clean historical results, schema normalization
  ratings.py         # Elo / strength rating computation with recency decay
  poisson_model.py   # bivariate Poisson: fit strengths, output scoreline grid
  scoring.py         # the 6-tier Dacopa scoring function (pure, fully tested)
  optimizer.py       # expected-points optimizer: scoreline grid -> best prediction
  markets.py         # optional: ingest/compare market & bookmaker probabilities
  pipeline.py        # end-to-end: data -> ratings -> model -> predictions -> outputs
tests/               # scoring.py especially must be exhaustively tested
outputs/             # generated predictions (csv), one row per match
notebooks/           # exploration
```

## Engineering conventions

- Python 3.11+. Use `uv` or `pip` with the provided `requirements.txt`.
- `scoring.py` is the most safety-critical file — it encodes the rules above. It must be a
  pure function with an exhaustive test suite covering every tier and the draw/knockout
  edge cases. **If you change scoring.py, run the tests and re-verify against the
  simulator-verified examples in this file and in tests/test_scoring.py.** A wrong scoring
  function silently corrupts every prediction.
- Keep the model and the optimizer decoupled: the model only produces probabilities; all
  rule-specific logic lives in scoring.py + optimizer.py. This lets us re-tune strategy
  without touching the model.
- Determinism: seed any randomness; predictions should be reproducible from fixed inputs.
- Don't hardcode the fixture list; load it from data so it updates as the bracket fills.

## Workflow notes

- Group-stage predictions lock progressively; the full slate is locked by the tournament
  start (~June 11, 2026, 09:00 UTC). Knockout predictions are made round by round as teams
  are confirmed — re-run the pipeline with updated fixtures before each round.
- When asked for predictions, output BOTH the chosen scoreline AND the underlying win/
  draw/loss probabilities and expected points, so the human can sanity-check and override.
- Be direct about uncertainty. Tournament football is high-variance; flag low-confidence
  matches rather than implying false precision.
