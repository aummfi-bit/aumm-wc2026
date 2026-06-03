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


def _current_ratings_and_params(results_2026: "pd.DataFrame | None" = None):
    """Current team ratings (base history + in-tournament updates) and goal params."""
    import pandas as pd

    from . import config
    from .data_loader import load_results
    from .ratings import compute_ratings, rate_matches, update_ratings_in_tournament
    from .goal_model import fit_goal_model

    results = load_results()
    elo = compute_ratings(
        results, home_advantage=config.HOME_ADVANTAGE
    ).set_index("team")["rating"]
    params = fit_goal_model(rate_matches(results, home_advantage=config.HOME_ADVANTAGE))
    if results_2026 is not None and len(results_2026):
        elo = pd.Series(update_ratings_in_tournament(
            elo.to_dict(), results_2026, home_advantage=config.HOME_ADVANTAGE
        ))
    return elo, params


def generate_group_slate(
    results_2026: "pd.DataFrame | None" = None,
    overrides: "pd.DataFrame | None" = None,
) -> "pd.DataFrame":
    """Produce EV-optimal predictions for all 72 group-stage matches.

    Fits ratings + goal model on the full history (calibrated config), loads the
    2026 group draw, and predicts each round-robin fixture. One row per match
    with the pick, win/draw/loss probabilities, expected goals, and EV.

    If `results_2026` (played matches so far) is given, the ratings are updated
    in-tournament with mean reversion before predicting. `overrides` applies
    manual lineup/team-news Elo deltas, which take precedence over the scenario
    motivation prior for any team that has one (observed beats guess).
    """
    import pandas as pd

    from . import config
    from .fixtures import load_groups, group_stage_fixtures
    from .overrides import elo_delta_for
    from .scenarios import classify_match, adjust_expected_goals

    elo, params = _current_ratings_and_params(results_2026)

    groups = load_groups()
    fixtures = group_stage_fixtures(groups)
    group_teams = {g: list(sub["team"]) for g, sub in groups.groupby("group")}
    use_scenarios = results_2026 is not None and len(results_2026)

    rows = []
    for fx in fixtures.itertuples(index=False):
        ovr_h = elo_delta_for(overrides, fx.home_team, fx.away_team)
        ovr_a = elo_delta_for(overrides, fx.away_team, fx.home_team)
        mu_h, mu_a = expected_goals(
            elo[fx.home_team] + ovr_h, elo[fx.away_team] + ovr_a, params,
            host_home=fx.host_home, host_away=fx.host_away, knockout=False,
            ko_factor=config.KO_FACTOR, goal_scale=config.GOAL_SCALE,
        )
        scen_h = scen_a = "ALIVE"
        if use_scenarios:
            scen_h, scen_a = classify_match(
                group_teams[fx.group], fx.home_team, fx.away_team, results_2026
            )
            # A manual override supersedes the motivation prior for that team.
            eff_h = "ALIVE" if ovr_h else scen_h
            eff_a = "ALIVE" if ovr_a else scen_a
            mu_h, mu_a = adjust_expected_goals(mu_h, mu_a, eff_h, eff_a)

        rec = predict_match(mu_h, mu_a, rho=config.RHO, knockout=False)
        h, a = rec["prediction"]
        rows.append({
            "group": fx.group, "home": fx.home_team, "away": fx.away_team,
            "prediction": f"{h}-{a}",
            "p_home": rec["p_home"], "p_draw": rec["p_draw"], "p_away": rec["p_away"],
            "mu_home": round(mu_h, 2), "mu_away": round(mu_a, 2),
            "exp_points": rec["expected_points"],
            "scen_home": scen_h, "scen_away": scen_a,
            "ovr_home": ovr_h, "ovr_away": ovr_a,
        })
    return pd.DataFrame(rows)


def generate_knockout_slate(
    ko_fixtures: "pd.DataFrame",
    results_2026: "pd.DataFrame | None" = None,
    overrides: "pd.DataFrame | None" = None,
) -> "pd.DataFrame":
    """EV-optimal predictions for confirmed knockout matchups (each weight 2x).

    Knockouts are scored on the 90-minute result, so the model predicts the 90'
    scoreline and the optimizer (knockout=True) banks the doubled tiers — a
    correct draw call is worth 30 off ANY draw scoreline, which the ko_factor
    (lower-scoring KO games) makes more likely. There is NO scenario layer in
    knockouts (win-or-go-home — everyone is maximally motivated), but manual
    overrides (injuries/suspensions/rotation) still apply.

    `ko_fixtures` needs round, home_team, away_team, and optionally
    host_home/host_away (else derived from host-nation membership).
    """
    import pandas as pd

    from . import config
    from .fixtures import HOSTS
    from .overrides import elo_delta_for

    elo, params = _current_ratings_and_params(results_2026)
    has_host_cols = {"host_home", "host_away"} <= set(ko_fixtures.columns)

    rows = []
    for fx in ko_fixtures.itertuples(index=False):
        home, away = fx.home_team, fx.away_team
        ovr_h = elo_delta_for(overrides, home, away)
        ovr_a = elo_delta_for(overrides, away, home)
        host_home = bool(fx.host_home) if has_host_cols else (home in HOSTS)
        host_away = bool(fx.host_away) if has_host_cols else (away in HOSTS)

        rec = predict_match_from_elo(
            elo[home] + ovr_h, elo[away] + ovr_a, params,
            host_home=host_home, host_away=host_away, knockout=True,
            rho=config.RHO, ko_factor=config.KO_FACTOR, goal_scale=config.GOAL_SCALE,
        )
        h, a = rec["prediction"]
        rows.append({
            "round": getattr(fx, "round", "KO"),
            "home": home, "away": away,
            "prediction": f"{h}-{a}", "draw_pick": h == a,
            "p_home": rec["p_home"], "p_draw": rec["p_draw"], "p_away": rec["p_away"],
            "mu_home": rec["mu_home"], "mu_away": rec["mu_away"],
            "exp_points": rec["expected_points"],   # already 2x-weighted
            "ovr_home": ovr_h, "ovr_away": ovr_a,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path

    from .markets import load_tipsheet, annotate_slate
    from .fixtures import load_results_2026
    from .overrides import load_overrides

    played = load_results_2026()
    overrides = load_overrides()
    if len(played):
        print(f"Updating ratings with {len(played)} played 2026 result(s).")
    if len(overrides):
        print(f"Applying {len(overrides)} manual override(s).")
    slate = annotate_slate(
        generate_group_slate(results_2026=played, overrides=overrides),
        tipsheet=load_tipsheet(),
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

    # Knockout predictions, once matchups are entered in the file.
    from .fixtures import load_knockout_fixtures

    ko = load_knockout_fixtures()
    if len(ko):
        ko_slate = generate_knockout_slate(ko, results_2026=played, overrides=overrides)
        ko_path = Path(__file__).resolve().parents[2] / "outputs" / "knockout_predictions.csv"
        ko_slate.to_csv(ko_path, index=False)
        print(f"\nWrote {len(ko_slate)} knockout prediction(s) (2x weight) to {ko_path}")
        print(ko_slate.to_string(index=False))
