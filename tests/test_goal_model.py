"""Tests for the Elo→goals bridge."""

import math

import pandas as pd
import pytest

from wc2026.goal_model import (
    GoalModelParams, build_training_data, fit_goal_model, expected_goals,
)
from wc2026.data_loader import MARTJ42_DIR, load_results
from wc2026.ratings import rate_matches


def _matches(n, eh, ea, hs, as_, neutral, comp=True, date="2020-01-01"):
    return [{
        "date": pd.Timestamp(date), "home_team": "H", "away_team": "A",
        "home_score": hs, "away_score": as_, "tournament": "X",
        "neutral": neutral, "competitive": comp,
        "elo_home_pre": eh, "elo_away_pre": ea,
    } for _ in range(n)]


def test_build_training_data_shape_and_signs():
    df = pd.DataFrame(_matches(2, 1600, 1500, 2, 1, neutral=False))
    data = build_training_data(df, min_date=None)
    assert len(data) == 4                                  # 2 matches x 2 sides
    assert set(data.columns) == {"goals", "elo_diff", "has_venue"}
    home, away = data.iloc[:2], data.iloc[2:]
    assert (home["elo_diff"] == 1.0).all()                 # (1600-1500)/100
    assert (home["has_venue"] == 1.0).all()                # non-neutral home
    assert (away["elo_diff"] == -1.0).all()                # mirrored for away
    assert (away["has_venue"] == 0.0).all()


def test_build_training_data_competitive_filter():
    df = pd.DataFrame(
        _matches(2, 1500, 1500, 1, 1, neutral=True, comp=True)
        + _matches(2, 1500, 1500, 1, 1, neutral=True, comp=False)
    )
    assert len(build_training_data(df, min_date=None)) == 4    # only the 2 competitive x2


def test_fit_recovers_positive_slopes():
    rows = (
        _matches(150, 1700, 1500, 2, 1, neutral=True)     # stronger home scores more
        + _matches(150, 1500, 1500, 1, 1, neutral=True)   # even -> level
        + _matches(150, 1500, 1500, 2, 1, neutral=False)  # venue -> home scores more
    )
    params = fit_goal_model(pd.DataFrame(rows), min_date=None)
    assert params.b_elo > 0
    assert params.b_home > 0


def test_expected_goals_behaviour():
    p = GoalModelParams(b0=0.2, b_elo=0.15, b_home=0.25)
    mh, ma = expected_goals(1500, 1500, p)
    assert abs(mh - ma) < 1e-9
    assert abs(mh - math.exp(0.2)) < 1e-9                  # baseline = exp(b0)

    mh2, ma2 = expected_goals(1800, 1500, p)              # stronger home
    assert mh2 > mh and ma2 < ma

    _, ma3 = expected_goals(1500, 1500, p, host_away=True)  # host boost lifts away
    assert ma3 > ma

    mhk, mak = expected_goals(1500, 1500, p, knockout=True)  # KO scales down
    assert mhk < mh and mak < ma


def test_expected_goals_symmetry():
    p = GoalModelParams(b0=0.2, b_elo=0.15, b_home=0.25)
    mh_ab, ma_ab = expected_goals(1700, 1400, p)
    mh_ba, ma_ba = expected_goals(1400, 1700, p)
    assert abs(mh_ab - ma_ba) < 1e-9
    assert abs(ma_ab - mh_ba) < 1e-9


@pytest.mark.skipif(
    not (MARTJ42_DIR / "results.csv").exists(),
    reason="raw data not present (gitignored)",
)
def test_real_fit_is_sane():
    params = fit_goal_model(rate_matches(load_results()))
    assert params.b_elo > 0                                # stronger -> more goals
    assert params.b_home > 0                               # home venue -> more goals
    assert 0.8 < math.exp(params.b0) < 2.0                 # plausible baseline
