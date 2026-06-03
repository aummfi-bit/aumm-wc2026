"""Tests for the Elo ratings engine.

Pure-function and synthetic-frame tests always run; the real-data check skips
when the (gitignored) raw data is absent.
"""

import pandas as pd
import pytest

from wc2026.ratings import (
    expected_score, update_elo, margin_multiplier, match_importance,
    revert_to_prior, compute_ratings, rate_matches, DEFAULT_ELO,
    update_ratings_in_tournament,
)
from wc2026.data_loader import MARTJ42_DIR, load_results


def test_expected_score_symmetry():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1700, 1500) > 0.5
    assert expected_score(1500, 1700) < 0.5
    assert abs(expected_score(1700, 1500) + expected_score(1500, 1700) - 1.0) < 1e-9


def test_update_direction():
    assert update_elo(1500, 0.5, 1.0) > 1500   # better than expected -> up
    assert update_elo(1500, 0.5, 0.0) < 1500   # worse than expected -> down


def test_margin_multiplier():
    assert margin_multiplier(0) == 1.0
    assert margin_multiplier(1) == 1.0
    assert margin_multiplier(-1) == 1.0        # uses absolute margin
    assert margin_multiplier(2) == 1.5
    assert margin_multiplier(3) == (11 + 3) / 8
    assert margin_multiplier(5) == (11 + 5) / 8


def test_match_importance_tiers():
    assert match_importance("Friendly") == 20.0
    assert match_importance("FIFA World Cup") == 60.0
    assert match_importance("FIFA World Cup qualification") == 40.0   # qualifier wins
    assert match_importance("UEFA Euro") == 50.0
    assert match_importance("Some Minor Cup") == 30.0
    assert match_importance("") == 30.0


def test_revert_to_prior():
    assert revert_to_prior(1600, 1500, 0.0) == 1600   # no reversion
    assert revert_to_prior(1600, 1500, 1.0) == 1500   # full reversion
    assert revert_to_prior(1600, 1500, 0.5) == 1550


def _repeated_wins(n=10):
    return pd.DataFrame([{
        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
        "home_team": "A", "away_team": "B",
        "home_score": 2, "away_score": 0,
        "tournament": "Friendly", "neutral": True, "competitive": False,
    } for i in range(n)])


def test_compute_ratings_winner_above_loser_and_zero_sum():
    table = compute_ratings(_repeated_wins())
    r = dict(zip(table["team"], table["rating"]))
    assert r["A"] > DEFAULT_ELO > r["B"]
    assert abs((r["A"] + r["B"]) - 2 * DEFAULT_ELO) < 1e-6     # zero-sum update
    assert list(table.columns) == ["team", "rating", "matches", "last_date"]
    assert table.iloc[0]["team"] == "A"                        # sorted descending
    assert int(table.loc[table.team == "A", "matches"].iloc[0]) == 10


def test_home_advantage_only_off_neutral():
    # Same single match, neutral vs non-neutral: the non-neutral home win moves
    # the home team LESS (it was expected to win more, so beats expectation by
    # less). Both teams start equal.
    base = {"date": pd.Timestamp("2020-01-01"), "home_team": "A", "away_team": "B",
            "home_score": 1, "away_score": 0, "tournament": "Friendly",
            "competitive": False}
    neutral = compute_ratings(pd.DataFrame([{**base, "neutral": True}]))
    home = compute_ratings(pd.DataFrame([{**base, "neutral": False}]))
    a_neutral = neutral.loc[neutral.team == "A", "rating"].iloc[0]
    a_home = home.loc[home.team == "A", "rating"].iloc[0]
    assert a_neutral > a_home


def _played(home, away, hs, as_):
    return pd.DataFrame([{
        "date": pd.Timestamp("2026-06-11"), "home_team": home, "away_team": away,
        "home_score": hs, "away_score": as_, "neutral": True,
    }])


def test_in_tournament_empty_returns_base():
    base = {"A": 1800.0, "B": 1500.0}
    out = update_ratings_in_tournament(base, pd.DataFrame())
    assert out == base
    assert out is not base                              # a copy


def test_in_tournament_full_reversion_stays_at_base():
    base = {"A": 1800.0, "B": 1500.0}
    out = update_ratings_in_tournament(base, _played("A", "B", 0, 3), reversion=1.0)
    assert out["A"] == 1800.0 and out["B"] == 1500.0    # update fully reverted away


def test_in_tournament_update_moves_and_reversion_dampens():
    base = {"A": 1500.0, "B": 1800.0}
    # Underdog A beats favourite B -> A rises, B falls.
    full = update_ratings_in_tournament(base, _played("A", "B", 2, 0), reversion=0.0)
    damped = update_ratings_in_tournament(base, _played("A", "B", 2, 0), reversion=0.5)
    assert full["A"] > base["A"] > 1500.0 - 1            # A gained
    assert full["B"] < base["B"]                         # B dropped
    assert base["A"] < damped["A"] < full["A"]           # reversion pulls back toward base
    assert base["B"] > damped["B"] > full["B"]
    assert {"A", "B"} == set(full)                        # base not mutated elsewhere


def test_rate_matches_adds_prematch_elo():
    out = rate_matches(_repeated_wins(3))
    assert "elo_home_pre" in out.columns and "elo_away_pre" in out.columns
    assert len(out) == 3
    assert out.iloc[0]["elo_home_pre"] == DEFAULT_ELO      # first match: cold start
    assert out.iloc[0]["elo_away_pre"] == DEFAULT_ELO
    assert out.iloc[2]["elo_home_pre"] > out.iloc[0]["elo_home_pre"]  # winner climbs


@pytest.mark.skipif(
    not (MARTJ42_DIR / "results.csv").exists(),
    reason="raw data not present (gitignored)",
)
def test_real_ratings_rank_powers_at_top():
    table = compute_ratings(load_results())
    assert len(table) > 200
    assert abs(table["rating"].mean() - DEFAULT_ELO) < 1e-6    # conserved
    top25 = set(table.head(25)["team"])
    assert {"Brazil", "Argentina", "France", "Spain"}.issubset(top25)
