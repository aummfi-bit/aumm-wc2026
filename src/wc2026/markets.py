"""Optional: ingest prediction-market / bookmaker probabilities as a benchmark.

Market prices are crowd probability estimates — a calibration check and an
optional model input, NOT ground truth. When the model and the market disagree
sharply, surface it explicitly (see CLAUDE.md).

No live API access in the offline code environment; ingest via pasted/exported
data or a CSV the human provides.
"""
from __future__ import annotations


def implied_prob_from_decimal_odds(odds: float) -> float:
    """Convert decimal odds to an implied probability (with bookmaker margin)."""
    return 1.0 / odds


def devig_two_way(p_a: float, p_b: float) -> tuple[float, float]:
    """Remove the vig from a two-way market by normalizing to 1."""
    total = p_a + p_b
    return p_a / total, p_b / total
