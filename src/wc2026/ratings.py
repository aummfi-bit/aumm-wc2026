"""Team strength ratings via an Elo engine fit on historical results.

World-football-Elo-style, with three refinements over textbook Elo:
  * a margin-of-victory multiplier (a 3-0 moves ratings more than a 1-0),
  * match-importance weighting (friendlies move ratings far less than World Cup
    games), and
  * a home-field adjustment that is switched OFF for neutral-ground matches —
    almost every World Cup match is at a neutral site.

Ratings update chronologically over the full history; the latest value per team
is its current strength. Because recent matches act on already-moved ratings,
Elo is inherently recency-weighted. The host boost for USA/CAN/MEX and the
in-tournament mean reversion (`revert_to_prior`) are applied downstream, not
baked into these base ratings.

Feeds expected-goals estimation in pipeline.py.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_ELO = 1500.0
HOME_ADVANTAGE = 100.0  # Elo points added to the home team on non-neutral grounds

# Match-importance K (the base step size per match). World-football-Elo-style
# tiers; these are tunable in the Stage 3 calibration pass.
IMPORTANCE_DEFAULT = 30.0
IMPORTANCE_FRIENDLY = 20.0
IMPORTANCE_QUALIFIER = 40.0
IMPORTANCE_CONTINENTAL = 50.0
IMPORTANCE_WORLD_CUP = 60.0

_CONTINENTAL_KEYS = (
    "euro", "copa américa", "copa america", "african cup", "asian cup",
    "gold cup", "nations league", "confederations",
)


def match_importance(tournament: str) -> float:
    """Map a tournament name to its Elo importance weight (base K)."""
    t = (tournament or "").lower()
    if "friendly" in t:
        return IMPORTANCE_FRIENDLY
    if "qualif" in t:
        return IMPORTANCE_QUALIFIER
    if "world cup" in t:
        return IMPORTANCE_WORLD_CUP
    if any(key in t for key in _CONTINENTAL_KEYS):
        return IMPORTANCE_CONTINENTAL
    return IMPORTANCE_DEFAULT


def margin_multiplier(goal_diff: int) -> float:
    """World-football-Elo margin-of-victory multiplier."""
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11.0 + g) / 8.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo expected result for A in [0, 1]."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating: float, expected: float, actual: float,
               k: float = IMPORTANCE_DEFAULT) -> float:
    return rating + k * (actual - expected)


def revert_to_prior(dynamic: float, base: float, r: float) -> float:
    """Pull a rating partway back toward a prior (in-tournament mean reversion).

    `new = (1 - r) * dynamic + r * base`. Applied between matches during the
    short group stage so one fluky result doesn't distort the rest of the run.
    """
    return (1.0 - r) * dynamic + r * base


def _run_elo(df: pd.DataFrame, home_advantage: float, default_elo: float):
    """Single chronological Elo pass.

    Returns (sorted_df, pre_home, pre_away, ratings, matches, last_date), where
    pre_home/pre_away are each match's PRE-match ratings (aligned to sorted_df)
    and ratings/matches/last_date are the final per-team state. Shared by
    compute_ratings (final table) and rate_matches (training snapshots).
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}
    matches: dict[str, int] = {}
    last_date: dict[str, object] = {}
    pre_home: list[float] = []
    pre_away: list[float] = []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        rh = ratings.get(home, default_elo)
        ra = ratings.get(away, default_elo)
        pre_home.append(rh)
        pre_away.append(ra)

        adv = 0.0 if row.neutral else home_advantage
        exp_home = expected_score(rh + adv, ra)
        goal_diff = int(row.home_score) - int(row.away_score)
        actual_home = 1.0 if goal_diff > 0 else (0.5 if goal_diff == 0 else 0.0)

        k = match_importance(row.tournament) * margin_multiplier(goal_diff)
        delta = k * (actual_home - exp_home)
        ratings[home] = rh + delta
        ratings[away] = ra - delta

        for team in (home, away):
            matches[team] = matches.get(team, 0) + 1
            last_date[team] = row.date

    return df, pre_home, pre_away, ratings, matches, last_date


def compute_ratings(
    df: pd.DataFrame,
    home_advantage: float = HOME_ADVANTAGE,
    default_elo: float = DEFAULT_ELO,
) -> pd.DataFrame:
    """Run chronological Elo over normalized results; return a ratings table.

    `df` must carry the data_loader canonical columns (date, home_team,
    away_team, home_score, away_score, tournament, neutral). Returns one row per
    team — team, rating, matches, last_date — sorted by rating descending.
    """
    _, _, _, ratings, matches, last_date = _run_elo(df, home_advantage, default_elo)
    table = pd.DataFrame({
        "team": list(ratings.keys()),
        "rating": list(ratings.values()),
        "matches": [matches[t] for t in ratings],
        "last_date": [last_date[t] for t in ratings],
    })
    return table.sort_values("rating", ascending=False).reset_index(drop=True)


def rate_matches(
    df: pd.DataFrame,
    home_advantage: float = HOME_ADVANTAGE,
    default_elo: float = DEFAULT_ELO,
) -> pd.DataFrame:
    """Return `df` (sorted by date) with each match's PRE-match Elo attached.

    Adds `elo_home_pre` and `elo_away_pre` columns — the ratings going into the
    match, before its result is applied. This is the training covariate for the
    Elo→goals regression in goal_model.py.
    """
    sdf, pre_home, pre_away, *_ = _run_elo(df, home_advantage, default_elo)
    out = sdf.copy()
    out["elo_home_pre"] = pre_home
    out["elo_away_pre"] = pre_away
    return out
