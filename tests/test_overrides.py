"""Tests for manual lineup/team-news overrides."""

import pandas as pd

from wc2026.overrides import load_overrides, elo_delta_for, OVERRIDES_CSV


def test_empty_overrides_zero_delta():
    assert elo_delta_for(pd.DataFrame(), "Spain", "Japan") == 0.0
    assert elo_delta_for(None, "Spain", "Japan") == 0.0


def test_match_level_override():
    ov = pd.DataFrame([{"team": "Spain", "opponent": "Uruguay", "elo_delta": -150, "note": "rotation"}])
    assert elo_delta_for(ov, "Spain", "Uruguay") == -150.0   # applies to this match
    assert elo_delta_for(ov, "Spain", "Japan") == 0.0        # not other matches
    assert elo_delta_for(ov, "Uruguay", "Spain") == 0.0      # only the named team


def test_team_level_override_applies_to_all():
    for opp in ("", "*", float("nan")):
        ov = pd.DataFrame([{"team": "Ghana", "opponent": opp, "elo_delta": -80, "note": "injury"}])
        assert elo_delta_for(ov, "Ghana", "Panama") == -80.0
        assert elo_delta_for(ov, "Ghana", "England") == -80.0


def test_multiple_overrides_sum():
    ov = pd.DataFrame([
        {"team": "Brazil", "opponent": "*", "elo_delta": -40, "note": "key injury"},
        {"team": "Brazil", "opponent": "Haiti", "elo_delta": -60, "note": "rest for dead rubber"},
    ])
    assert elo_delta_for(ov, "Brazil", "Haiti") == -100.0    # both apply
    assert elo_delta_for(ov, "Brazil", "Morocco") == -40.0   # only the team-level one


def test_overrides_csv_exists_and_loads():
    if OVERRIDES_CSV.exists():
        df = load_overrides()
        assert list(df.columns) == ["team", "opponent", "elo_delta", "note"]
