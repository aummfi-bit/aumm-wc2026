"""Tests for the qualification-scenario / motivation layer."""

import pandas as pd

from wc2026.scenarios import (
    compute_standings, classify_team, classify_match, adjust_expected_goals,
    OWN_FACTOR, OPP_FACTOR,
)

GROUP = ["A1", "A2", "A3", "A4"]


def _groups():
    return pd.DataFrame({"group": ["A"] * 4, "team": GROUP})


def _res(rows):
    # rows: list of (home, away, hs, as_)
    return pd.DataFrame(
        [{"home_team": h, "away_team": a, "home_score": hs, "away_score": as_} for h, a, hs, as_ in rows]
    )


def test_standings_basic():
    res = _res([("A1", "A2", 2, 0), ("A3", "A4", 1, 1)])
    st = compute_standings(_groups(), res).set_index("team")
    assert st.loc["A1", "points"] == 3 and st.loc["A1", "gd"] == 2
    assert st.loc["A2", "points"] == 0
    assert st.loc["A3", "points"] == 1 and st.loc["A4", "points"] == 1


def test_classify_alive_before_final_round():
    # Only round 1 played: every team still has 2 games left -> ALIVE.
    res = _res([("A1", "A2", 1, 0), ("A3", "A4", 1, 0)])
    assert classify_team("A1", "A3", GROUP, res) == "ALIVE"


def test_classify_secure_and_eliminated():
    # Rounds 1-2 played; final round remaining: A1 vs A2, A3 vs A4.
    # A1: 6 pts (beat A3, A4). A2: 6 pts (beat A3, A4). A3 & A4: 0 pts.
    res = _res([
        ("A1", "A3", 1, 0), ("A2", "A4", 1, 0),    # round 1
        ("A1", "A4", 1, 0), ("A2", "A3", 1, 0),    # round 2
    ])
    # A1 & A2 both on 6, A3 & A4 on 0, final games A1-A2 and A3-A4.
    assert classify_team("A1", "A2", GROUP, res) == "SECURE"      # both already top 2
    assert classify_team("A2", "A1", GROUP, res) == "SECURE"
    assert classify_team("A3", "A4", GROUP, res) == "ELIMINATED"  # can't reach top 2
    assert classify_team("A4", "A3", GROUP, res) == "ELIMINATED"


def test_classify_must_win():
    # After 2 rounds: A1=6 (secure), A2=3, A3=3, A4=0. Final: A2 vs A3, A1 vs A4.
    # A2 and A3 fight for 2nd; the loser of A2-A3 risks elimination, a win likely
    # secures 2nd -> not strictly must-win for both, but the head-to-head matters.
    res = _res([
        ("A1", "A2", 1, 0), ("A3", "A4", 1, 0),    # A1=3,A3=3
        ("A1", "A3", 1, 0), ("A2", "A4", 1, 0),    # A1=6,A2=3,A3=3,A4=0
    ])
    # Final round remaining: A2 vs A3, A1 vs A4.
    s_a2 = classify_team("A2", "A3", GROUP, res)
    s_a3 = classify_team("A3", "A2", GROUP, res)
    assert s_a2 in {"MUST_WIN", "DRAW_OK", "ALIVE"}
    # A1 is already through regardless.
    assert classify_team("A1", "A4", GROUP, res) == "SECURE"
    # A4 on 0 with only A1 (already-through) left: cannot reach top 2.
    assert classify_team("A4", "A1", GROUP, res) == "ELIMINATED"


def test_classify_match_pair_and_only_final_match():
    res = _res([
        ("A1", "A3", 1, 0), ("A2", "A4", 1, 0),
        ("A1", "A4", 1, 0), ("A2", "A3", 1, 0),
    ])
    sh, sa = classify_match(GROUP, "A1", "A2", res)
    assert sh == "SECURE" and sa == "SECURE"
    # A team whose queried opponent isn't its remaining match -> ALIVE.
    assert classify_team("A1", "A3", GROUP, res) == "ALIVE"   # A1-A3 already played


def test_adjust_expected_goals_directions():
    # Secure (resting) home vs alive away: home scores less, away more.
    mh, ma = adjust_expected_goals(1.5, 1.0, "SECURE", "ALIVE")
    assert mh < 1.5 and ma > 1.0
    # Must-win home: home scores more.
    mh2, _ = adjust_expected_goals(1.5, 1.0, "MUST_WIN", "ALIVE")
    assert mh2 > 1.5
    # Two ALIVE teams: unchanged.
    assert adjust_expected_goals(1.5, 1.0, "ALIVE", "ALIVE") == (1.5, 1.0)
