"""Tests for knockout prediction (2x weight, 90-minute draws, no scenarios)."""

import pandas as pd
import pytest

from wc2026.pipeline import generate_knockout_slate, generate_group_slate
from wc2026.fixtures import load_knockout_fixtures
from wc2026.data_loader import MARTJ42_DIR

DATA = (MARTJ42_DIR / "results.csv").exists()


@pytest.mark.skipif(not DATA, reason="raw data not present (gitignored)")
def test_knockout_slate_structure_and_double_weight():
    ko = pd.DataFrame([
        {"round": "R32", "home_team": "Brazil", "away_team": "Norway"},
        {"round": "R32", "home_team": "Spain", "away_team": "Croatia"},
    ])
    slate = generate_knockout_slate(ko)
    assert len(slate) == 2
    assert {"round", "home", "away", "prediction", "draw_pick", "p_draw",
            "exp_points"} <= set(slate.columns)
    # Knockout EV is 2x-weighted, so it should exceed the same matchup's group EV.
    grp = generate_group_slate()
    g = grp[(grp.home == "Spain") & (grp.away == "Croatia")]
    if len(g):
        ko_ev = slate.loc[slate.home == "Spain", "exp_points"].iloc[0]
        assert ko_ev > float(g["exp_points"].iloc[0])


@pytest.mark.skipif(not DATA, reason="raw data not present (gitignored)")
def test_knockout_overrides_apply():
    ko = pd.DataFrame([{"round": "R32", "home_team": "Spain", "away_team": "Uruguay"}])
    base = generate_knockout_slate(ko)
    ovr = pd.DataFrame([{"team": "Spain", "opponent": "Uruguay", "elo_delta": -250, "note": "x"}])
    weakened = generate_knockout_slate(ko, overrides=ovr)
    assert weakened["mu_home"].iloc[0] < base["mu_home"].iloc[0]   # Spain weaker
    assert weakened["ovr_home"].iloc[0] == -250


def test_empty_knockout_fixtures_loads():
    ko = load_knockout_fixtures()
    assert list(ko.columns) == ["round", "home_team", "away_team"] or ko.empty
