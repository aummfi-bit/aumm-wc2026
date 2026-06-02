"""Tests for the Dacopa scoring ladder.

The cases marked "official" come directly from the worked examples on
dacopa.com/bolao/regras and must always pass. If any of these fail, the
scoring function no longer matches the pool's rules.
"""

from wc2026.scoring import score_prediction, expected_points


# --- Official examples: real result Palmeiras 2-1 Flamengo (decisive) ---

def test_official_exact_score():
    assert score_prediction(2, 1, 2, 1) == 25

def test_official_winner_and_winner_goals():
    # guess 2-0: right winner + winner scored 2
    assert score_prediction(2, 0, 2, 1) == 18

def test_official_winner_and_goal_diff():
    # guess 3-2: right winner + margin +1
    assert score_prediction(3, 2, 2, 1) == 15

def test_official_winner_and_loser_goals():
    # guess 3-1: right winner + loser scored 1
    assert score_prediction(3, 1, 2, 1) == 12

def test_winner_only_clean_case():
    # A guess that gets ONLY the winner right and matches no other feature.
    # Real 3-1 (win by 2, winner 3, loser 1). Guess 1-0: winner ok, winner
    # goals 1!=3, loser goals 0!=1, diff 1!=2 -> winner only = 10.
    assert score_prediction(1, 0, 3, 1) == 10

# NOTE ON A KNOWN AMBIGUITY (see CLAUDE.md):
# Dacopa's marketing table lists guess 1-0 vs real 2-1 as "winner only = 10".
# But 1-0 has goal difference +1, which equals the real difference +1, so a
# strict top-down ladder scores it as the goal-difference tier = 15. The
# rule TEXT ("any score where the winner is right but you missed winner's
# goals, loser's goals AND goal difference") implies the strict ladder. We
# default to the strict, internally-consistent ladder. Confirm against the
# live app before the tournament and flip STRICT_LADDER if the app disagrees.
def test_known_ambiguity_strict_ladder():
    # Under the strict ladder this is the goal-difference tier.
    assert score_prediction(1, 0, 2, 1) == 15

def test_official_predicted_draw_but_decisive():
    # guess 1-1: predicted a draw, real was decisive -> 0
    assert score_prediction(1, 1, 2, 1) == 0

def test_official_wrong_winner():
    # guess 0-2 (away win): wrong winner -> 0
    assert score_prediction(0, 2, 2, 1) == 0


# --- Official examples: real result 0-0 (draw) ---

def test_official_draw_exact():
    assert score_prediction(0, 0, 0, 0) == 25

def test_official_draw_wrong_score():
    # guess 1-1, real 0-0: correctly a draw but wrong score -> 0
    assert score_prediction(1, 1, 0, 0) == 0

def test_official_draw_predicted_decisive():
    # guess 1-0, real 0-0 -> 0
    assert score_prediction(1, 0, 0, 0) == 0


# --- Draw edge cases (the binary rule) ---

def test_real_draw_only_exact_scores():
    # real 1-1: only an exact 1-1 guess scores
    assert score_prediction(1, 1, 1, 1) == 25
    assert score_prediction(2, 2, 1, 1) == 0
    assert score_prediction(0, 0, 1, 1) == 0
    assert score_prediction(2, 1, 1, 1) == 0  # decisive guess on a real draw


# --- Symmetry: away winners ---

def test_away_winner_tiers():
    # real 1-3 (away win by 2; winner scored 3, loser scored 1)
    assert score_prediction(1, 3, 1, 3) == 25          # exact
    assert score_prediction(0, 3, 1, 3) == 18          # winner goals 3 ok
    assert score_prediction(0, 2, 1, 3) == 15          # diff 2 ok (winner goals 2!=3)


def test_away_winner_loser_goals_tier():
    # real 1-3: loser (home) scored 1. guess 0-4: winner=away,
    # winner goals 4 != 3, diff 4 != 2, loser goals 0 != 1 -> winner only = 10
    assert score_prediction(0, 4, 1, 3) == 10
    # guess 1-4: loser goals 1 == 1, winner goals 4 !=3, diff 3 !=2 -> t4 = 12
    assert score_prediction(1, 4, 1, 3) == 12


# --- Knockout doubling ---

def test_knockout_doubles():
    assert score_prediction(2, 1, 2, 1, knockout=True) == 50   # exact
    assert score_prediction(1, 0, 3, 1, knockout=True) == 20   # winner only x2
    assert score_prediction(0, 0, 0, 0, knockout=True) == 50   # exact draw
    assert score_prediction(1, 1, 0, 0, knockout=True) == 0    # wrong draw score


# --- Expected points ---

def test_expected_points_basic():
    # Distribution: 50% 1-0, 50% 2-1. Predict 1-0.
    dist = {(1, 0): 0.5, (2, 1): 0.5}
    # vs 1-0: exact (25). vs 2-1: right winner, winner goals 1!=2,
    #   diff 1==1 -> 15. EV = 0.5*25 + 0.5*15 = 20
    assert expected_points((1, 0), dist) == 20.0

def test_expected_points_knockout_scales():
    dist = {(1, 0): 1.0}
    assert expected_points((1, 0), dist, knockout=True) == 50.0

def test_expected_points_missing_mass_scores_zero():
    # Only partial distribution; uncovered outcomes contribute nothing.
    dist = {(1, 0): 0.3}
    assert expected_points((1, 0), dist) == 7.5
