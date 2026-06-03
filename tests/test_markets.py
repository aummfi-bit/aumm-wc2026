"""Tests for the field model + differentiation analysis and odds helpers."""

import pandas as pd

from wc2026.markets import (
    implied_prob_from_decimal_odds, devig_two_way,
    field_prediction, differentiation, annotate_slate,
)


def test_odds_helpers():
    assert implied_prob_from_decimal_odds(2.0) == 0.5
    a, b = devig_two_way(0.55, 0.55)        # 10% overround -> 0.5/0.5
    assert abs(a - 0.5) < 1e-9 and abs(b - 0.5) < 1e-9


def test_field_prediction_anchors():
    assert field_prediction(0.70, 0.20, 0.10) == (2, 1)   # clear home favorite
    assert field_prediction(0.10, 0.20, 0.70) == (1, 2)   # clear away favorite
    assert field_prediction(0.33, 0.34, 0.33) == (1, 1)   # even -> 1-1


def test_differentiation_categories():
    # We predict a draw the field calls decisive, with a real draw chance.
    d = differentiation(0.55, 0.32, 0.13, our_pred=(1, 1))
    assert d["field_pick"] == "2-1" and not d["agrees"]
    assert d["edge"] == "draw_we_see"

    # Same winner, different margin -> low-value nuance.
    d = differentiation(0.70, 0.20, 0.10, our_pred=(2, 0))
    assert d["edge"] == "margin" and not d["agrees"]

    # Agreement with the field.
    d = differentiation(0.70, 0.20, 0.10, our_pred=(2, 1))
    assert d["agrees"] and d["edge"] == "none"

    # Backing the other side.
    d = differentiation(0.55, 0.20, 0.25, our_pred=(0, 1))
    assert d["edge"] == "winner_contrarian"


def test_annotate_slate():
    slate = pd.DataFrame({
        "home": ["A", "B"], "away": ["C", "D"],
        "prediction": ["2-1", "1-1"],
        "p_home": [0.70, 0.40], "p_draw": [0.20, 0.33], "p_away": [0.10, 0.27],
    })
    out = annotate_slate(slate)
    assert list(out["field_pick"]) == ["2-1", "1-1"]
    assert out.loc[0, "agrees"]                       # 2-1 matches field
    assert out.loc[1, "agrees"]                       # both 1-1 (even game)
