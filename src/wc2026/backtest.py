"""Walk-forward backtest + calibration on past World Cups.

For each target World Cup we fit ratings and the goal model on ONLY the matches
that predate the tournament (no leakage), predict every match with those fixed
pre-tournament ratings, and score the prediction under the corrected Dacopa
table — including the 90-minute knockout rule.

90-minute scoring of history: jfjelstul records the post-extra-time score, so a
match with `extra_time == 1` was level at 90' (its 90-minute *outcome* is a
draw, exact scoreline unknown). We score those as the draw tier for any draw
prediction and 0 otherwise — never the exact-score bonus (see data/SOURCES.md).

The objective is POOL POINTS under the real table, not generic accuracy. We
compare the EV-optimal pick against two baselines: the model's modal scoreline
(isolates the optimizer's value) and a naive "favorite wins 1-0" (proxies the
field). `tune()` sweeps the calibration knobs against backtested points.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data_loader import load_results, MARTJ42_DIR
from .ratings import compute_ratings, rate_matches, HOME_ADVANTAGE
from .goal_model import fit_goal_model, expected_goals, KO_FACTOR
from .poisson_model import dixon_coles_grid
from .optimizer import best_prediction
from .scoring import score_prediction, WINNER_AND_GOAL_DIFF, KNOCKOUT_MULTIPLIER

JFJELSTUL_MATCHES = MARTJ42_DIR.parent / "jfjelstul" / "matches.csv"
TARGET_WCS = ("2010", "2014", "2018", "2022")


def load_wc_matches(path=JFJELSTUL_MATCHES, years=TARGET_WCS) -> pd.DataFrame:
    """Load target World Cup matches with the fields the backtest needs."""
    m = pd.read_csv(path)
    m = m[m["tournament_name"].str.contains("|".join(years))].copy()
    m["match_date"] = pd.to_datetime(m["match_date"])
    m["knockout"] = m["knockout_stage"] == 1
    # A team playing in its own country is the host (boost); on neutral WC
    # matches neither team's name equals the venue country.
    m["host_home"] = m["home_team_name"] == m["country_name"]
    m["host_away"] = m["away_team_name"] == m["country_name"]
    return m[[
        "tournament_name", "match_date", "home_team_name", "away_team_name",
        "home_team_score", "away_team_score", "knockout", "extra_time",
        "host_home", "host_away",
    ]].reset_index(drop=True)


def real_90(row) -> tuple[int, int] | None:
    """90-minute real score, or None when the game went to extra time.

    None signals "a draw at 90', exact scoreline unknown" — handled by
    score_match.
    """
    if int(row.extra_time) == 1:
        return None
    return (int(row.home_team_score), int(row.away_team_score))


def score_match(pred: tuple[int, int], real: tuple[int, int] | None, knockout: bool) -> int:
    """Score a prediction against a 90-minute result (real=None ⇒ ET draw)."""
    if real is None:
        mult = KNOCKOUT_MULTIPLIER if knockout else 1
        return WINNER_AND_GOAL_DIFF * mult if pred[0] == pred[1] else 0
    return score_prediction(pred[0], pred[1], real[0], real[1], knockout=knockout)


def _naive_pick(mu_home: float, mu_away: float) -> tuple[int, int]:
    """Field proxy: favorite wins 1-0 (1-1 if the means are level)."""
    if abs(mu_home - mu_away) < 1e-9:
        return (1, 1)
    return (1, 0) if mu_home > mu_away else (0, 1)


def _fit_pre_tournament(results: pd.DataFrame, start_date, home_advantage: float):
    """Ratings + goal-model params from matches strictly before `start_date`."""
    train = results[results["date"] < start_date]
    elo = compute_ratings(train, home_advantage=home_advantage).set_index("team")["rating"]
    params = fit_goal_model(rate_matches(train, home_advantage=home_advantage))
    return elo, params


@dataclass
class BacktestResult:
    n: int
    ev_points: float          # mean pool points per match, EV-optimal pick
    modal_points: float       # mean, model's modal scoreline
    naive_points: float       # mean, naive favorite 1-0
    exact_rate: float         # share of EV picks that are exact (non-ET games)
    winner_rate: float        # share of EV picks with the right outcome
    ev_total: float

    def __str__(self) -> str:
        return (
            f"n={self.n}  EV={self.ev_points:.2f}  modal={self.modal_points:.2f}  "
            f"naive={self.naive_points:.2f}  | exact={self.exact_rate:.1%}  "
            f"winner={self.winner_rate:.1%}  total={self.ev_total:.0f}"
        )


def evaluate(
    results: pd.DataFrame,
    wc_matches: pd.DataFrame,
    home_advantage: float = HOME_ADVANTAGE,
    rho: float = -0.05,
    ko_factor: float = KO_FACTOR,
    goal_scale: float = 1.0,
    max_goals: int = 6,
) -> BacktestResult:
    """Backtest all target World Cups under one set of calibration knobs."""
    ev_total = modal_total = naive_total = 0.0
    n = exact = winner = non_et = 0

    for _, wc in wc_matches.groupby("tournament_name"):
        elo, params = _fit_pre_tournament(results, wc["match_date"].min(), home_advantage)
        for row in wc.itertuples(index=False):
            if row.home_team_name not in elo.index or row.away_team_name not in elo.index:
                continue
            mu_h, mu_a = expected_goals(
                elo[row.home_team_name], elo[row.away_team_name], params,
                host_home=row.host_home, host_away=row.host_away,
                knockout=row.knockout, ko_factor=ko_factor, goal_scale=goal_scale,
            )
            grid = dixon_coles_grid(mu_h, mu_a, rho=rho, max_goals=max_goals)
            ev_pred, _ = best_prediction(grid, knockout=row.knockout, max_goals=max_goals)
            modal = max(grid, key=grid.get)
            naive = _naive_pick(mu_h, mu_a)
            real = real_90(row)

            ev_total += score_match(ev_pred, real, row.knockout)
            modal_total += score_match(modal, real, row.knockout)
            naive_total += score_match(naive, real, row.knockout)
            n += 1

            # Diagnostics on the EV pick.
            ev_draw = ev_pred[0] == ev_pred[1]
            if real is None:                      # ET: outcome is a draw
                winner += ev_draw
            else:
                non_et += 1
                if ev_pred == real:
                    exact += 1
                real_draw = real[0] == real[1]
                if ev_draw == real_draw and (real_draw or (ev_pred[0] > ev_pred[1]) == (real[0] > real[1])):
                    winner += 1

    return BacktestResult(
        n=n,
        ev_points=ev_total / n,
        modal_points=modal_total / n,
        naive_points=naive_total / n,
        exact_rate=exact / non_et if non_et else 0.0,
        winner_rate=winner / n if n else 0.0,
        ev_total=ev_total,
    )


def tune(
    results: pd.DataFrame,
    wc_matches: pd.DataFrame,
    home_advs=(60.0, 80.0, 100.0, 120.0),
    rhos=(-0.15, -0.10, -0.05, 0.0),
    ko_factors=(0.80, 0.90, 1.00),
    max_goals: int = 6,
) -> pd.DataFrame:
    """Coordinate sweep over the calibration knobs, ranked by mean EV points.

    Ratings/goal-model fits depend only on home_advantage, so we fit once per
    (home_advantage, tournament) and sweep rho x ko_factor cheaply on top.
    """
    rows = []
    for ha in home_advs:
        cache = {
            name: _fit_pre_tournament(results, wc["match_date"].min(), ha)
            for name, wc in wc_matches.groupby("tournament_name")
        }
        for rho in rhos:
            for kf in ko_factors:
                total = 0.0
                n = 0
                for name, wc in wc_matches.groupby("tournament_name"):
                    elo, params = cache[name]
                    for row in wc.itertuples(index=False):
                        if row.home_team_name not in elo.index or row.away_team_name not in elo.index:
                            continue
                        mu_h, mu_a = expected_goals(
                            elo[row.home_team_name], elo[row.away_team_name], params,
                            host_home=row.host_home, host_away=row.host_away,
                            knockout=row.knockout, ko_factor=kf,
                        )
                        grid = dixon_coles_grid(mu_h, mu_a, rho=rho, max_goals=max_goals)
                        pred, _ = best_prediction(grid, knockout=row.knockout, max_goals=max_goals)
                        total += score_match(pred, real_90(row), row.knockout)
                        n += 1
                rows.append({
                    "home_adv": ha, "rho": rho, "ko_factor": kf,
                    "mean_points": total / n, "n": n,
                })
    return pd.DataFrame(rows).sort_values("mean_points", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from . import config

    results = load_results()
    wc = load_wc_matches()
    print("Backtest over 2010-2022 World Cups (mean pool points / match):")
    print("  uncalibrated:", evaluate(results, wc))
    print("  calibrated:  ", evaluate(
        results, wc,
        home_advantage=config.HOME_ADVANTAGE, rho=config.RHO,
        ko_factor=config.KO_FACTOR, goal_scale=config.GOAL_SCALE,
    ))
