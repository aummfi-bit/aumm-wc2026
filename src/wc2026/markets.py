"""Field model + differentiation analysis, plus market-odds helpers.

In a bolão you win by RANK, not absolute points — your edge comes from being
more right than the FIELD on matches it misprices, not from points everyone
earns alike. The Stage 3 backtest showed our EV pick rarely beats a trivial
"2-1 favorite" anchor on absolute points, so the lever is differentiation: find
where our model diverges from the likely field pick, and flag the CLAUDE.md
mispricing categories for human override.

The field model is a transparent proxy for the Dacopa tip-sheet crowd — anchor
on 2-1 for a clear favorite, 1-1 for a near-even game. It is an ASSUMPTION, not
fitted data; treat its output as a prompt to think, not ground truth.

Market/bookmaker helpers (devig etc.) are kept here too: market prices are one
more crowd estimate, a calibration check, never ground truth.
"""
from __future__ import annotations

import pandas as pd

# A game with no side above this probability is "even" → the field anchors 1-1.
FIELD_EVEN_THRESHOLD = 0.45
# The field's confident draw call is rare; it under-calls draws (key edge).
DRAW_EDGE_THRESHOLD = 0.30


# --------------------------------------------------------------------------- #
# Market-odds helpers (optional benchmark input)
# --------------------------------------------------------------------------- #

def implied_prob_from_decimal_odds(odds: float) -> float:
    """Convert decimal odds to an implied probability (with bookmaker margin)."""
    return 1.0 / odds


def devig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    """Remove the vig from a two-way market by normalizing to 1."""
    total = p_a + p_b
    return p_a / total, p_b / total


# --------------------------------------------------------------------------- #
# Field model + differentiation
# --------------------------------------------------------------------------- #

def field_prediction(
    p_home: float,
    p_draw: float,
    p_away: float,
    even_threshold: float = FIELD_EVEN_THRESHOLD,
) -> tuple[int, int]:
    """The crowd's likely scoreline: 2-1 for a clear favorite, else 1-1.

    Encodes the Dacopa tip-sheet behavior (anchor on 2-1 / 1-1, back the favored
    side). A transparent proxy, not a fitted model.
    """
    if max(p_home, p_away) < even_threshold:
        return (1, 1)
    return (2, 1) if p_home >= p_away else (1, 2)


def differentiation(
    p_home: float,
    p_draw: float,
    p_away: float,
    our_pred: tuple[int, int],
    even_threshold: float = FIELD_EVEN_THRESHOLD,
    draw_edge_threshold: float = DRAW_EDGE_THRESHOLD,
) -> dict:
    """Compare our pick to the likely field pick; classify the divergence.

    Returns field_pick, whether we agree, and an `edge` category:
      * draw_we_see       — we predict a draw the field calls decisive (the
                            highest-value edge: a correct draw banks 15/30 the
                            field misses).
      * draw_field_only   — field predicts a draw, we go decisive.
      * winner_contrarian — we back the other side (high risk / high reward).
      * margin            — same winner, different scoreline (low-value nuance).
      * none              — we agree with the field (no differentiation).
    """
    field = field_prediction(p_home, p_draw, p_away, even_threshold)
    agree = tuple(our_pred) == field
    our_draw = our_pred[0] == our_pred[1]
    field_draw = field[0] == field[1]
    our_winner = "H" if our_pred[0] > our_pred[1] else ("A" if our_pred[0] < our_pred[1] else "D")
    field_winner = "H" if field[0] > field[1] else ("A" if field[0] < field[1] else "D")

    if agree:
        edge = "none"
    elif our_draw and not field_draw:
        edge = "draw_we_see" if p_draw >= draw_edge_threshold else "margin"
    elif field_draw and not our_draw:
        edge = "draw_field_only"
    elif our_winner != field_winner:
        edge = "winner_contrarian"
    else:
        edge = "margin"

    return {"field_pick": f"{field[0]}-{field[1]}", "agrees": agree, "edge": edge}


def annotate_slate(slate: pd.DataFrame) -> pd.DataFrame:
    """Add field_pick / agrees / edge columns to a generated prediction slate.

    Expects the columns from pipeline.generate_group_slate: prediction (e.g.
    "2-1"), p_home, p_draw, p_away.
    """
    out = slate.copy()
    recs = []
    for row in out.itertuples(index=False):
        h, a = (int(x) for x in row.prediction.split("-"))
        recs.append(differentiation(row.p_home, row.p_draw, row.p_away, (h, a)))
    out["field_pick"] = [r["field_pick"] for r in recs]
    out["agrees"] = [r["agrees"] for r in recs]
    out["edge"] = [r["edge"] for r in recs]
    return out
