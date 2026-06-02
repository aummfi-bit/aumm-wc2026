"""Expected-points optimizer.

Given a scoreline probability distribution for a match (from poisson_model.py),
choose the prediction that maximizes expected points under the Dacopa 6-tier
table for the match's phase weight. The optimum is usually NOT the modal
scoreline — it's whatever maximizes expected partial credit. See CLAUDE.md.
"""

from __future__ import annotations

from .scoring import expected_points


def best_prediction(
    score_probabilities: dict[tuple[int, int], float],
    knockout: bool = False,
    max_goals: int = 6,
) -> tuple[tuple[int, int], float]:
    """Return (best_scoreline, expected_points) over all candidate scorelines.

    Candidates are all (h, a) with 0 <= h, a <= max_goals. We evaluate the EV
    of *predicting* each candidate against the full probability distribution.
    """
    best_pred: tuple[int, int] | None = None
    best_ev = -1.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            ev = expected_points((h, a), score_probabilities, knockout=knockout)
            if ev > best_ev:
                best_ev = ev
                best_pred = (h, a)
    assert best_pred is not None
    return best_pred, best_ev


def ranked_predictions(
    score_probabilities: dict[tuple[int, int], float],
    knockout: bool = False,
    max_goals: int = 6,
    top_n: int = 5,
) -> list[tuple[tuple[int, int], float]]:
    """Return the top_n candidate predictions by expected points.

    Useful for human review and for the tiebreaker end-game, where you may
    deliberately trade EV for a higher chance of an exact-score tiebreaker hit.
    """
    scored = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            ev = expected_points((h, a), score_probabilities, knockout=knockout)
            scored.append(((h, a), ev))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]
