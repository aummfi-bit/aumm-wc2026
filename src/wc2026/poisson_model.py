"""Scoreline models: independent Poisson + Dixon-Coles, and bivariate Poisson.

Produces a probability distribution over scorelines for a match, given each
team's expected goals (mu_home, mu_away). Two ways to capture the dependence
between the two teams' goals that a naive independent Poisson misses:

  1. Dixon-Coles (recommended): independent Poisson with a low-score correction
     (parameter rho) that re-weights the 0-0, 1-0, 0-1, 1-1 cells. This is the
     classic Dixon & Coles (1997) approach and is well validated for football.
     It specifically fixes the cells that matter most to us — low scores
     dominate, and for our pool the only draws that EVER score points are exact
     low-score draws (0-0, 1-1). Getting those probabilities right is high value.

  2. Bivariate Poisson: induces positive correlation via a shared component
     lambda_cov. A different mechanism for the same goal.

Use ONE of them, not both — they are alternative ways to model the same
correlation, and stacking them double-counts it. Default to Dixon-Coles.

Keep this module free of any scoring-rule logic — that lives in scoring.py /
optimizer.py. Expected-goals inputs come from ratings.py (Elo/strength + form,
host adjustment for USA/CAN/MEX, knockout-conservatism factor).
"""

from __future__ import annotations

import math


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


# --------------------------------------------------------------------------- #
# Dixon-Coles (recommended default)
# --------------------------------------------------------------------------- #

def _dc_tau(x: int, y: int, mu_h: float, mu_a: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor for cell (x, y).

    Only the four lowest-scoring cells are adjusted; everything else is
    unchanged (tau = 1). rho < 0 (typical for football) lifts 0-0 and 1-1 and
    lowers 1-0 and 0-1, matching the empirical excess of low-scoring draws.
    Stability requires roughly rho in [-1, 0]; values that drive a cell
    non-positive should be avoided when fitting.
    """
    if x == 0 and y == 0:
        return 1.0 - mu_h * mu_a * rho
    if x == 0 and y == 1:
        return 1.0 + mu_h * rho
    if x == 1 and y == 0:
        return 1.0 + mu_a * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_grid(
    mu_home: float,
    mu_away: float,
    rho: float = -0.05,
    max_goals: int = 8,
) -> dict[tuple[int, int], float]:
    """Return P(home=x, away=y) under independent Poisson + Dixon-Coles.

    mu_home, mu_away are the marginal expected goals for each team. rho is the
    low-score dependence parameter (fit it on historical data; -0.05 to -0.15
    is a reasonable starting range for international football). Negative tau
    values from an extreme rho are clamped to 0 before normalization.
    """
    grid: dict[tuple[int, int], float] = {}
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            base = _poisson_pmf(x, mu_home) * _poisson_pmf(y, mu_away)
            adj = base * _dc_tau(x, y, mu_home, mu_away, rho)
            grid[(x, y)] = max(adj, 0.0)
    total = sum(grid.values())
    return {k: v / total for k, v in grid.items()}


# --------------------------------------------------------------------------- #
# Bivariate Poisson (alternative correlation mechanism)
# --------------------------------------------------------------------------- #

def bivariate_poisson_grid(
    lambda_home: float,
    lambda_away: float,
    lambda_cov: float = 0.0,
    max_goals: int = 8,
) -> dict[tuple[int, int], float]:
    """Return P(home=x, away=y) under a bivariate Poisson.

    Parameters (l1, l2, l3): marginal means are (l1 + l3) and (l2 + l3).
    lambda_cov=0 reduces to independent Poisson. Callers usually have marginal
    expected goals mu_home, mu_away; here we treat lambda_home/lambda_away as
    those marginals and subtract the shared component.

    NOTE: do not also apply Dixon-Coles on top of this — pick one correlation
    mechanism. See module docstring.
    """
    l3 = lambda_cov
    l1 = max(lambda_home - l3, 1e-9)
    l2 = max(lambda_away - l3, 1e-9)

    grid: dict[tuple[int, int], float] = {}
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            s = 0.0
            for k in range(min(x, y) + 1):
                s += (
                    _poisson_pmf(x - k, l1)
                    * _poisson_pmf(y - k, l2)
                    * _poisson_pmf(k, l3)
                )
            grid[(x, y)] = s
    total = sum(grid.values())
    return {k: v / total for k, v in grid.items()}


# --------------------------------------------------------------------------- #
# Shared helper
# --------------------------------------------------------------------------- #

def outcome_probabilities(
    grid: dict[tuple[int, int], float]
) -> dict[str, float]:
    """Collapse a scoreline grid into P(home win), P(draw), P(away win)."""
    home = draw = away = 0.0
    for (h, a), p in grid.items():
        if h > a:
            home += p
        elif h == a:
            draw += p
        else:
            away += p
    return {"home": home, "draw": draw, "away": away}
