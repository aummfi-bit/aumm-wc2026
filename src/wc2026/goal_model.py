"""Elo → expected-goals bridge (the model's core mapping).

Turns two team Elo ratings into a pair of expected goals (mu_home, mu_away) for
the Dixon-Coles scoreline grid. The relationship is fit on history as an
"Elo-as-Poisson-covariate" regression (cf. danielguerreros/WC-Model):

    log(mu_team) = b0 + b_elo * (elo_team - elo_opp)/100 + b_home * has_venue

fit by Poisson maximum likelihood. Two rows per historical match (one per team's
perspective). `has_venue` is 1 for the home side of a non-neutral match and 0
otherwise — almost every World Cup match is neutral, so the venue term is what
lets non-neutral history still inform the rating→goals slope without biasing the
neutral WC prediction.

At prediction time, the host boost (USA/CAN/MEX in their own country) reuses the
fitted venue term, and knockout conservatism scales both means down.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ELO_SCALE = 100.0          # elo difference is expressed per-100 points
KO_FACTOR = 0.90           # knockout games are lower-scoring (tune in Stage 3)
DEFAULT_MIN_DATE = "2006-01-01"   # fit on the modern era to reflect current scoring


@dataclass(frozen=True)
class GoalModelParams:
    b0: float        # intercept: log baseline goals for an even, neutral match
    b_elo: float     # per-100-Elo log-goal slope
    b_home: float    # venue (home / host) log-goal bump


def build_training_data(
    rated_df: pd.DataFrame,
    competitive_only: bool = True,
    min_date: str | None = DEFAULT_MIN_DATE,
) -> pd.DataFrame:
    """Long-format training rows (one per team-perspective) from rated matches.

    `rated_df` must come from ratings.rate_matches (has elo_home_pre /
    elo_away_pre). Returns columns: goals, elo_diff (scaled, signed for the
    team), has_venue.
    """
    df = rated_df
    if competitive_only and "competitive" in df.columns:
        df = df[df["competitive"]]
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]

    diff = (df["elo_home_pre"] - df["elo_away_pre"]) / ELO_SCALE
    has_venue = (~df["neutral"].to_numpy()).astype(float)

    home = pd.DataFrame({
        "goals": df["home_score"].astype(float).to_numpy(),
        "elo_diff": diff.to_numpy(),
        "has_venue": has_venue,
    })
    away = pd.DataFrame({
        "goals": df["away_score"].astype(float).to_numpy(),
        "elo_diff": -diff.to_numpy(),
        "has_venue": np.zeros(len(df)),
    })
    return pd.concat([home, away], ignore_index=True)


def fit_goal_model(
    rated_df: pd.DataFrame,
    competitive_only: bool = True,
    min_date: str | None = DEFAULT_MIN_DATE,
) -> GoalModelParams:
    """Fit the Poisson Elo→goals regression by maximum likelihood."""
    data = build_training_data(rated_df, competitive_only, min_date)
    y = data["goals"].to_numpy()
    X = np.column_stack([
        np.ones(len(data)),
        data["elo_diff"].to_numpy(),
        data["has_venue"].to_numpy(),
    ])

    def nll(beta: np.ndarray) -> float:
        mu = np.exp(X @ beta)
        return float(np.sum(mu - y * (X @ beta)))

    def grad(beta: np.ndarray) -> np.ndarray:
        mu = np.exp(X @ beta)
        return X.T @ (mu - y)

    res = minimize(nll, x0=np.array([0.3, 0.0, 0.2]), jac=grad, method="L-BFGS-B")
    b0, b_elo, b_home = res.x
    return GoalModelParams(b0=float(b0), b_elo=float(b_elo), b_home=float(b_home))


def expected_goals(
    elo_home: float,
    elo_away: float,
    params: GoalModelParams,
    host_home: bool = False,
    host_away: bool = False,
    knockout: bool = False,
    ko_factor: float = KO_FACTOR,
    goal_scale: float = 1.0,
) -> tuple[float, float]:
    """Map two Elo ratings to (mu_home, mu_away).

    Venue/host: most WC matches are neutral, so host_home/host_away default to
    False; set host_home=True when the home-listed team is a host nation playing
    in its own country (USA/CAN/MEX). Knockout scales both means by ko_factor.
    `goal_scale` is a calibration multiplier on both means: the regression is
    fit on all competitive internationals, which score lower than World Cup
    matches, so a scale > 1 corrects the under-prediction (see config.py /
    backtest).
    """
    diff = (elo_home - elo_away) / ELO_SCALE
    eta_home = params.b0 + params.b_elo * diff + (params.b_home if host_home else 0.0)
    eta_away = params.b0 - params.b_elo * diff + (params.b_home if host_away else 0.0)
    mu_home = float(np.exp(eta_home)) * goal_scale
    mu_away = float(np.exp(eta_away)) * goal_scale
    if knockout:
        mu_home *= ko_factor
        mu_away *= ko_factor
    return mu_home, mu_away
