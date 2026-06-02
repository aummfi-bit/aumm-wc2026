"""Team strength ratings (Elo) with recency-weighted form.

National teams play infrequently and in disjoint windows, so weight
competitive/tournament matches above friendlies and decay older results.
Output feeds expected-goals estimation in pipeline.py.
"""
from __future__ import annotations

DEFAULT_ELO = 1500.0
K_FACTOR = 30.0  # tune; higher reacts faster to recent results


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected result for A in [0,1]."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating: float, expected: float, actual: float, k: float = K_FACTOR) -> float:
    return rating + k * (actual - expected)

# TODO: iterate over historical results to build current ratings per team,
# apply match-importance weights and time decay, and expose a ratings table.
