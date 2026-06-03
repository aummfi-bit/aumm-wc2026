"""Tests for the backtest scoring helpers and a fast integration run."""

from types import SimpleNamespace

import pytest

from wc2026.backtest import (
    score_match, real_90, evaluate, load_wc_matches, load_results, JFJELSTUL_MATCHES,
)


def test_real_90_strips_extra_time():
    # ET game -> None (draw at 90, exact unknown); regulation game -> its score.
    assert real_90(SimpleNamespace(extra_time=1, home_team_score=2, away_team_score=1)) is None
    assert real_90(SimpleNamespace(extra_time=0, home_team_score=2, away_team_score=1)) == (2, 1)


def test_score_match_regulation_uses_table():
    assert score_match((2, 1), (2, 1), False) == 25     # exact
    assert score_match((1, 0), (2, 1), False) == 15     # signed goal diff
    assert score_match((1, 1), (0, 0), False) == 15     # draw tier
    assert score_match((2, 1), (0, 0), False) == 0      # decisive on a draw


def test_score_match_extra_time_draw():
    # ET game: 90' outcome is a draw, exact unknown. Draw guess -> 15 (x2 in KO),
    # decisive guess -> 0; never the exact-score bonus.
    assert score_match((1, 1), None, True) == 30
    assert score_match((0, 0), None, True) == 30
    assert score_match((2, 1), None, True) == 0
    assert score_match((1, 1), None, False) == 15


@pytest.mark.skipif(not JFJELSTUL_MATCHES.exists(), reason="raw data not present (gitignored)")
def test_evaluate_runs_and_is_sane():
    # One tournament keeps it fast (a single pre-tournament fit).
    res = evaluate(load_results(), load_wc_matches(years=("2022",)))
    assert res.n > 50
    assert 5.0 < res.ev_points < 25.0
    assert 0.0 <= res.winner_rate <= 1.0
    assert res.ev_points > 0
