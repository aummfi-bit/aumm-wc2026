"""Dacopa bolão scoring — the 6-tier ladder.

This module encodes the pool's official rules. It is the most safety-critical
file in the repo: a wrong scoring function silently corrupts every prediction
and every expected-points optimization. Keep it a pure function and keep the
test suite exhaustive. Verified against the official examples on
dacopa.com/bolao/regras (see tests/test_scoring.py and CLAUDE.md).
"""

from __future__ import annotations


# Base (group-stage, weight 1x) points per tier.
EXACT_SCORE = 25          # tier 1
WINNER_AND_WINNER_GOALS = 18   # tier 2
WINNER_AND_GOAL_DIFF = 15      # tier 3
WINNER_AND_LOSER_GOALS = 12    # tier 4
WINNER_ONLY = 10               # tier 5
NOTHING = 0                    # tier 6

KNOCKOUT_MULTIPLIER = 2


def score_prediction(
    pred_home: int,
    pred_away: int,
    real_home: int,
    real_away: int,
    knockout: bool = False,
) -> int:
    """Return the points for one prediction against one real result.

    Scores are evaluated top-down; the highest matching tier wins.

    Knockout games are scored at the END OF REGULATION (90 min) only — the
    caller is responsible for passing the regulation score (extra-time goals
    and penalties must already be stripped out). `knockout=True` only applies
    the 2x weight.

    Draws are all-or-nothing: only an exact-score match scores. The
    intermediate tiers require a real winner, and goal difference carries no
    information when the game is tied, so a non-exact draw prediction scores 0
    even if you correctly called it a draw. (Official example: real 0-0,
    guess 1-1 -> 0.)
    """
    points = _base_points(pred_home, pred_away, real_home, real_away)
    if knockout:
        points *= KNOCKOUT_MULTIPLIER
    return points


def _base_points(ph: int, pa: int, rh: int, ra: int) -> int:
    # Tier 1: exact score. Covers exact draws too.
    if ph == rh and pa == ra:
        return EXACT_SCORE

    real_draw = rh == ra
    pred_draw = ph == pa

    if real_draw:
        # On a real draw, the only scoring tier is exact (handled above).
        # A non-exact draw prediction, or any decisive prediction, scores 0.
        return NOTHING

    # From here the real result is decisive.
    real_home_won = rh > ra

    # You must at least have the correct winner to score anything.
    pred_home_won = ph > pa
    if pred_draw or pred_home_won != real_home_won:
        return NOTHING

    # Correct winner established. Now ladder down tiers 2 -> 5.
    real_winner_goals = rh if real_home_won else ra
    real_loser_goals = ra if real_home_won else rh
    pred_winner_goals = ph if real_home_won else pa
    pred_loser_goals = pa if real_home_won else ph

    real_diff = abs(rh - ra)
    pred_diff = abs(ph - pa)

    # Tier 2: correct winner + winner's goal count.
    if pred_winner_goals == real_winner_goals:
        return WINNER_AND_WINNER_GOALS
    # Tier 3: correct winner + goal difference.
    if pred_diff == real_diff:
        return WINNER_AND_GOAL_DIFF
    # Tier 4: correct winner + loser's goal count.
    if pred_loser_goals == real_loser_goals:
        return WINNER_AND_LOSER_GOALS
    # Tier 5: correct winner only.
    return WINNER_ONLY


def expected_points(
    prediction: tuple[int, int],
    score_probabilities: dict[tuple[int, int], float],
    knockout: bool = False,
) -> float:
    """Expected points of a single prediction over a scoreline distribution.

    `score_probabilities` maps (home_goals, away_goals) -> probability. It need
    not be exhaustive, but should cover enough mass to be meaningful; any
    missing mass is treated as scoring 0 for this prediction.
    """
    ph, pa = prediction
    total = 0.0
    for (rh, ra), p in score_probabilities.items():
        total += p * score_prediction(ph, pa, rh, ra, knockout=knockout)
    return total
