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
    ko_factor: float = 0.90,
    goal_scale: float = 1.0,
) -> dict:
    """Full chain for one match: two Elo ratings -> EV-optimal prediction.

    Composes expected_goals (Elo -> mu, with host/KO/calibration adjustments)
    and predict_match (Dixon-Coles grid -> optimizer). Returns the same record
    as predict_match plus the expected goals used.
    """
    mu_home, mu_away = expected_goals(
        elo_home, elo_away, params,
        host_home=host_home, host_away=host_away, knockout=knockout,
        ko_factor=ko_factor, goal_scale=goal_scale,
    )
    record = predict_match(mu_home, mu_away, rho=rho, knockout=knockout)
    record["mu_home"] = round(mu_home, 2)
    record["mu_away"] = round(mu_away, 2)
    return record


def generate_group_slate(results_2026: "pd.DataFrame | None" = None) -> "pd.DataFrame":
    """Produce EV-optimal predictions for all 72 group-stage matches.

    Fits ratings + goal model on the full history (calibrated config), loads the
    2026 group draw, and predicts each round-robin fixture. One row per match
    with the pick, win/draw/loss probabilities, expected goals, and EV.

    If `results_2026` (played matches so far) is given, the ratings are updated
    in-tournament with mean reversion before predicting — so re-running before
    MD2/MD3 reflects what has actually happened.
    """
    import pandas as pd

    from . import config
    from .data_loader import load_results
    from .ratings import compute_ratings, rate_matches, update_ratings_in_tournament
    from .goal_model import fit_goal_model
    from .fixtures import load_groups, group_stage_fixtures

    results = load_results()
    elo = compute_ratings(
        results, home_advantage=config.HOME_ADVANTAGE
    ).set_index("team")["rating"]
    params = fit_goal_model(rate_matches(results, home_advantage=config.HOME_ADVANTAGE))

    if results_2026 is not None and len(results_2026):
        updated = update_ratings_in_tournament(
            elo.to_dict(), results_2026, home_advantage=config.HOME_ADVANTAGE
        )
        elo = pd.Series(updated)

    from .scenarios import classify_match, adjust_expected_goals

    groups = load_groups()
    fixtures = group_stage_fixtures(groups)
    group_teams = {g: list(sub["team"]) for g, sub in groups.groupby("group")}
    use_scenarios = results_2026 is not None and len(results_2026)

    rows = []
    for fx in fixtures.itertuples(index=False):
        mu_h, mu_a = expected_goals(
            elo[fx.home_team], elo[fx.away_team], params,
            host_home=fx.host_home, host_away=fx.host_away, knockout=False,
            ko_factor=config.KO_FACTOR, goal_scale=config.GOAL_SCALE,
        )
        scen_h = scen_a = "ALIVE"
        if use_scenarios:
            scen_h, scen_a = classify_match(
                group_teams[fx.group], fx.home_team, fx.away_team, results_2026
            )
            mu_h, mu_a = adjust_expected_goals(mu_h, mu_a, scen_h, scen_a)

        rec = predict_match(mu_h, mu_a, rho=config.RHO, knockout=False)
        h, a = rec["prediction"]
        rows.append({
            "group": fx.group, "home": fx.home_team, "away": fx.away_team,
            "prediction": f"{h}-{a}",
            "p_home": rec["p_home"], "p_draw": rec["p_draw"], "p_away": rec["p_away"],
            "mu_home": round(mu_h, 2), "mu_away": round(mu_a, 2),
            "exp_points": rec["expected_points"],
            "scen_home": scen_h, "scen_away": scen_a,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path

    from .markets import load_tipsheet, annotate_slate
    from .fixtures import load_results_2026

    played = load_results_2026()
    if len(played):
        print(f"Updating ratings with {len(played)} played 2026 result(s).")
    slate = annotate_slate(
        generate_group_slate(results_2026=played), tipsheet=load_tipsheet()
    )
    out_path = Path(__file__).resolve().parents[2] / "outputs" / "group_predictions.csv"
    slate.to_csv(out_path, index=False)
    print(f"Wrote {len(slate)} group-stage predictions (with field flags) to {out_path}")

    print(f"\nAgreement with the field tip sheet: {slate['agrees'].mean():.0%}")
    print("Edge breakdown:")
    print(slate["edge"].value_counts().to_string())

    flagged = slate[slate["edge"].isin(["winner_contrarian", "draw_we_see"])]
    if len(flagged):
        print("\nHIGH-VALUE DIVERGENCES (review these):")
        cols = ["group", "home", "away", "prediction", "field_pick",
                "p_home", "p_draw", "p_away", "edge"]
        print(flagged[cols].to_string(index=False))
