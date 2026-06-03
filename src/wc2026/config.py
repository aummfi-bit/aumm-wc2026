"""Calibrated model configuration (from the Stage 3 walk-forward backtest).

These come from backtesting over the 2010–2022 World Cups (256 matches),
scoring under the corrected Dacopa table.

READ THIS CAVEAT before trusting the numbers:

  * The points surface is FLAT. The EV-optimal pick, a naive "favorite wins
    1-0", and a trivial "favorite wins 2-1" all land at ~10.4–10.8 mean
    points/match — differences within ~1 standard error on 256 matches. We have
    NOT demonstrated a statistically significant edge over the field's simple
    anchors. Treat these as sensible defaults, not a proven optimum, and do not
    over-tune to them.

  * The two NON-noisy signals from the backtest:
      1. The goal regression UNDER-predicts World Cup scoring (mean total 2.37
         predicted vs 2.62 actual), which justifies GOAL_SCALE > 1.
      2. Predicting draws/low scores indiscriminately (e.g. always 1-1) is
         clearly bad (~5.6 pts/match) — draws pay only when they actually land.

  * Real pool edge is RELATIVE (beating other players), which an absolute-points
    backtest does not capture. The differentiation strategy (deferred) and
    better calibration are where edge has to come from — see CLAUDE.md #7.
"""

HOME_ADVANTAGE = 65.0   # Elo home bump used when fitting ratings (backtest mildly prefers < 100)
RHO = -0.12             # Dixon-Coles low-score correction
KO_FACTOR = 0.85        # knockout goal-suppression multiplier
GOAL_SCALE = 1.10       # corrects the WC goal under-prediction (2.37 -> ~2.61)

# Per-host home advantage as a FRACTION of the fitted home term b_home. Almost
# every WC match is neutral (factor 0 -> no boost); only the three hosts get one,
# and not equally: Mexico's Azteca-altitude + crowd edge is strongest, the USA's
# big-but-often-neutral crowds less, Canada's the least. These are judgment
# priors (not fitted) and are tunable.
HOST_FACTORS = {
    "Mexico": 1.15,
    "United States": 0.90,
    "Canada": 0.50,
}
