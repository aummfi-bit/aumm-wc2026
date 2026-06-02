"""Tests for the Dacopa scoring ladder.

These cases are verified one-by-one against Dacopa's official scoring
SIMULATOR (the live tool that scores a prediction against a real result). The
simulator is the authority; where the older marketing examples / a prior draft
of CLAUDE.md disagreed (it claimed draws were "exact-or-zero"), the simulator
wins and they were corrected. If any of these fail, the scoring function no
longer matches the pool's rules.

The unified ladder (top-down, highest matching tier wins):
    1  exact score ..................................... 25
    2  (decisive) winner + winner's goal count ......... 18
    3  signed goal difference equal .................... 15   <- also covers
    4  (decisive) winner + loser's goal count .......... 12      ANY draw guess
    5  (decisive) winner only .......................... 10      on a real draw
    6  nothing ..........................................  0      (diff 0 == 0)

Knockout matches are scored on the 90-minute result (extra time and the penalty
shootout don't count, so a game level at 90' is a draw), at 2x.
"""

from wc2026.scoring import score_prediction, expected_points


# --- Simulator battery 1: real result A 4-2 B (decisive home win) ---

def test_sim_4_2_exact():
    assert score_prediction(4, 2, 4, 2) == 25

def test_sim_4_2_winner_and_winner_goals():
    assert score_prediction(4, 1, 4, 2) == 18   # winner + winner's goals (4)

def test_sim_4_2_goal_diff_2_0():
    assert score_prediction(2, 0, 4, 2) == 15   # winner + goal diff (+2)

def test_sim_4_2_goal_diff_3_1():
    assert score_prediction(3, 1, 4, 2) == 15   # winner + goal diff (+2)

def test_sim_4_2_loser_goals():
    assert score_prediction(3, 2, 4, 2) == 12   # winner + loser's goals (2)

def test_sim_4_2_winner_only():
    assert score_prediction(1, 0, 4, 2) == 10   # winner only

def test_sim_4_2_predicted_draw_scores_zero():
    assert score_prediction(2, 2, 4, 2) == 0
    assert score_prediction(4, 4, 4, 2) == 0

def test_sim_4_2_wrong_winner_scores_zero():
    assert score_prediction(2, 4, 4, 2) == 0    # picked B (mirror score)
    assert score_prediction(0, 2, 4, 2) == 0    # picked B


# --- Simulator battery 2: real result A 3-3 B (draw) ---
# THE key correction: a correctly-called draw scores the goal-difference tier
# (15), NOT zero. Only the exact score scores 25; any decisive guess scores 0.

def test_sim_3_3_exact():
    assert score_prediction(3, 3, 3, 3) == 25

def test_sim_3_3_any_draw_scores_15():
    assert score_prediction(2, 2, 3, 3) == 15
    assert score_prediction(1, 1, 3, 3) == 15
    assert score_prediction(0, 0, 3, 3) == 15

def test_sim_3_3_decisive_scores_zero():
    # Even guesses that match a goal count (3-0, 3-1) score 0: a real draw has
    # no winner, so the winner-based tiers can't apply.
    assert score_prediction(3, 0, 3, 3) == 0
    assert score_prediction(3, 1, 3, 3) == 0
    assert score_prediction(2, 1, 3, 3) == 0


# --- The goal-difference tier is the SIGNED difference ---
# This resolves the old "ambiguity": real 2-1 vs guess 1-0. 1-0 has diff +1,
# equal to the real +1, so it's the goal-difference tier = 15 (confirmed by the
# structurally identical 2-0 / 3-1 cases in the 4-2 battery above).

def test_signed_goal_difference_tier():
    assert score_prediction(1, 0, 2, 1) == 15    # diff +1 == +1
    assert score_prediction(0, 1, 1, 2) == 15    # diff -1 == -1 (away win)
    assert score_prediction(0, 2, 2, 1) == 0     # wrong direction -> 0


# --- Draw symmetry: 0-0 and 1-1 reals ---

def test_real_draw_low_scores():
    assert score_prediction(0, 0, 0, 0) == 25    # exact
    assert score_prediction(1, 1, 0, 0) == 15    # any draw guess on a real draw
    assert score_prediction(2, 2, 1, 1) == 15
    assert score_prediction(0, 0, 1, 1) == 15
    assert score_prediction(1, 0, 0, 0) == 0     # decisive guess on a real draw


# --- Away-winner symmetry (real 1-3) ---

def test_away_winner_tiers():
    assert score_prediction(1, 3, 1, 3) == 25          # exact
    assert score_prediction(0, 3, 1, 3) == 18          # winner goals 3
    assert score_prediction(0, 2, 1, 3) == 15          # diff -2
    assert score_prediction(0, 4, 1, 3) == 10          # winner only
    assert score_prediction(1, 4, 1, 3) == 12          # loser goals 1


# --- Knockout doubling + 90-minute scoring (a level-at-90' game is a draw) ---

def test_knockout_doubles():
    assert score_prediction(2, 1, 2, 1, knockout=True) == 50   # exact
    assert score_prediction(1, 0, 3, 1, knockout=True) == 20   # winner only x2
    assert score_prediction(0, 0, 0, 0, knockout=True) == 50   # exact draw

def test_knockout_correct_draw_call_scores_30():
    # A KO match level at 90' (even if later decided in ET or on penalties) is
    # scored as a draw. Any correctly-called draw is the goal-difference tier:
    # 15 x2 = 30.
    assert score_prediction(1, 1, 0, 0, knockout=True) == 30
    assert score_prediction(2, 2, 1, 1, knockout=True) == 30


# --- Expected points ---

def test_expected_points_basic():
    # 50% 1-0, 50% 2-1. Predict 1-0. vs 1-0: exact 25. vs 2-1: diff +1 == +1
    # -> 15. EV = 0.5*25 + 0.5*15 = 20.
    dist = {(1, 0): 0.5, (2, 1): 0.5}
    assert expected_points((1, 0), dist) == 20.0

def test_expected_points_knockout_scales():
    dist = {(1, 0): 1.0}
    assert expected_points((1, 0), dist, knockout=True) == 50.0

def test_expected_points_missing_mass_scores_zero():
    dist = {(1, 0): 0.3}
    assert expected_points((1, 0), dist) == 7.5

def test_expected_points_values_draw_floor():
    # Predicting a draw banks 15 whenever the game is ANY draw, not only on the
    # exact score. 60% chance of some draw; predict 1-1:
    #   vs 1-1 -> 25 (exact), vs 0-0 / 2-2 -> 15 (draw tier), vs 2-1 -> 0.
    dist = {(1, 1): 0.2, (0, 0): 0.2, (2, 2): 0.2, (2, 1): 0.4}
    assert expected_points((1, 1), dist) == 0.2 * 25 + 0.2 * 15 + 0.2 * 15
