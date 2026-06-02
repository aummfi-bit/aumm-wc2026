"""End-to-end pipeline: data -> ratings -> model -> optimized predictions.

This is a skeleton. The data_loader and ratings steps depend on the actual
Kaggle schema, which is filled in once the CSVs are in data/raw/. The model and
optimizer below are functional today.

Intended flow:
    1. load_results()            historical international matches (data_loader)
    2. compute_ratings()         Elo/strength + recency-weighted form (ratings)
    3. expected_goals(a, b)      map ratings -> (mu_home, mu_away), with host
                                 boost for USA/CAN/MEX and KO conservatism
    4. bivariate_poisson_grid()  scoreline distribution
    5. best_prediction()         EV-optimal prediction under Dacopa scoring
    6. write outputs/predictions.csv  (one row per fixture, with probs + EV)
"""

from __future__ import annotations

from .poisson_model import dixon_coles_grid, outcome_probabilities
from .optimizer import best_prediction
from .goal_model import GoalModelParams, expected_goals


def predict_match(
    mu_home: float,
    mu_away: float,
    rho: float = -0.05,
    knockout: bool = False,
) -> dict:
    """Produce a full prediction record for a single match.

    Uses the Dixon-Coles model (independent Poisson + low-score correction).
    Returns the EV-optimal scoreline plus the supporting probabilities and EV,
    so a human can sanity-check and override (see CLAUDE.md workflow notes).
    """
    grid = dixon_coles_grid(mu_home, mu_away, rho)
    outcome = outcome_probabilities(grid)
    pred, ev = best_prediction(grid, knockout=knockout)
    modal = max(grid, key=grid.get)
    return {
        "prediction": pred,
        "expected_points": round(ev, 2),
        "modal_scoreline": modal,
        "p_home": round(outcome["home"], 3),
        "p_draw": round(outcome["draw"], 3),
        "p_away": round(outcome["away"], 3),
        "knockout": knockout,
    }


def predict_match_from_elo(
    elo_home: float,
    elo_away: float,
    params: GoalModelParams,
    *,
    host_home: bool = False,
    host_away: bool = False,
    knockout: bool = False,
    rho: float = -0.05,
) -> dict:
    """Full chain for one match: two Elo ratings -> EV-optimal prediction.

    Composes expected_goals (Elo -> mu, with host/KO adjustments) and
    predict_match (Dixon-Coles grid -> optimizer). Returns the same record as
    predict_match plus the expected goals used.
    """
    mu_home, mu_away = expected_goals(
        elo_home, elo_away, params,
        host_home=host_home, host_away=host_away, knockout=knockout,
    )
    record = predict_match(mu_home, mu_away, rho=rho, knockout=knockout)
    record["mu_home"] = round(mu_home, 2)
    record["mu_away"] = round(mu_away, 2)
    return record


if __name__ == "__main__":
    # Demo with hand-set expected goals (no data needed).
    demo = predict_match(mu_home=2.1, mu_away=0.7)
    print(demo)
