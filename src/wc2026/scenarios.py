"""Qualification-scenario / motivation layer for the group stage.

Reads the live standings + remaining fixtures, classifies each team's situation
going into its final group match, and nudges expected goals for motivation
effects the field ignores: dead-rubber rotation, must-win intensity, play-it-
safe caginess.

IMPORTANT HONESTY: the goal multipliers below are DIRECTIONAL PRIORS, not fitted
coefficients. Motivation/rotation effects are real but noisy and situation-
specific, and there is little clean data to fit them. The highest-confidence
signal is an OBSERVED lineup (a qualified team resting six starters) — treat
these as a structured default to be overridden by team news near kickoff.

Scope: we only classify a team's FINAL group match (when exactly one game
remains), where scenarios are determinable and motivation effects concentrate.
Earlier rounds return ALIVE (no adjustment).

Third-place caveat: in the 48-team format the 8 best third-placed teams also
advance, which is cross-group and not knowable from one group alone. We classify
TOP-2 scenarios within the group; "ELIMINATED" means "cannot reach top 2 here",
and a third-place lifeline may still exist — so its adjustment is kept mild.
"""
from __future__ import annotations

from itertools import combinations, product

import pandas as pd

# Per-team goal multiplier for the team's OWN scenario, and the bump its
# OPPONENT gets from facing a team in that scenario. Tunable priors.
OWN_FACTOR = {
    "SECURE": 0.85,      # locked into top 2 -> rotation/dead-rubber risk
    "ELIMINATED": 0.90,  # nothing to play for (third-place caveat -> mild)
    "MUST_WIN": 1.10,    # must win -> pushes forward
    "DRAW_OK": 0.97,     # a draw advances -> slightly cagey
    "ALIVE": 1.00,
}
OPP_FACTOR = {
    "SECURE": 1.08,      # exploit their rested XI
    "ELIMINATED": 1.04,
    "MUST_WIN": 1.05,    # they leave space -> more open game
    "DRAW_OK": 1.00,
    "ALIVE": 1.00,
}


def compute_standings(groups: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Per-team group table (played, points, gf, ga, gd) from played results."""
    rows = {
        t: {"group": g, "played": 0, "points": 0, "gf": 0, "ga": 0}
        for g, t in zip(groups["group"], groups["team"])
    }
    if results is not None and len(results):
        for r in results.itertuples(index=False):
            if r.home_team not in rows or r.away_team not in rows:
                continue
            hs, as_ = int(r.home_score), int(r.away_score)
            for team, gf, ga in ((r.home_team, hs, as_), (r.away_team, as_, hs)):
                rows[team]["played"] += 1
                rows[team]["gf"] += gf
                rows[team]["ga"] += ga
                rows[team]["points"] += 3 if gf > ga else (1 if gf == ga else 0)
    df = pd.DataFrame([{"team": t, **v} for t, v in rows.items()])
    df["gd"] = df["gf"] - df["ga"]
    return df.sort_values(["group", "points", "gd", "gf"], ascending=[True, False, False, False]).reset_index(drop=True)


def _remaining_in_group(group_teams: list[str], results: pd.DataFrame) -> list[tuple[str, str]]:
    played = set()
    if results is not None and len(results):
        for r in results.itertuples(index=False):
            if r.home_team in group_teams and r.away_team in group_teams:
                played.add(frozenset((r.home_team, r.away_team)))
    return [pair for pair in combinations(group_teams, 2) if frozenset(pair) not in played]


def _base_table(group_teams, results):
    pts = {t: 0 for t in group_teams}
    gd = {t: 0 for t in group_teams}
    gf = {t: 0 for t in group_teams}
    if results is not None and len(results):
        for r in results.itertuples(index=False):
            if r.home_team in pts and r.away_team in pts:
                hs, as_ = int(r.home_score), int(r.away_score)
                gf[r.home_team] += hs
                gf[r.away_team] += as_
                gd[r.home_team] += hs - as_
                gd[r.away_team] += as_ - hs
                pts[r.home_team] += 3 if hs > as_ else (1 if hs == as_ else 0)
                pts[r.away_team] += 3 if as_ > hs else (1 if as_ == hs else 0)
    return pts, gd, gf


def _top2(group_teams, pts0, gd0, gf0, rem, combo):
    pts, gd, gf = dict(pts0), dict(gd0), dict(gf0)
    for (a, b), o in zip(rem, combo):
        if o == "H":
            pts[a] += 3; gd[a] += 1; gd[b] -= 1; gf[a] += 1
        elif o == "A":
            pts[b] += 3; gd[b] += 1; gd[a] -= 1; gf[b] += 1
        else:
            pts[a] += 1; pts[b] += 1
    order = sorted(group_teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
    return set(order[:2])


def classify_team(team: str, opponent: str, group_teams: list[str], results: pd.DataFrame) -> str:
    """Classify `team`'s scenario for its match vs `opponent`.

    Returns ALIVE unless this is the team's final group match (exactly one game
    left for it). Otherwise enumerates all outcomes of the remaining group games
    and returns SECURE / ELIMINATED / MUST_WIN / DRAW_OK / ALIVE (top-2 basis).
    """
    rem = _remaining_in_group(group_teams, results)
    team_rem = [m for m in rem if team in m]
    if len(team_rem) != 1 or frozenset((team, opponent)) != frozenset(team_rem[0]):
        return "ALIVE"

    pts0, gd0, gf0 = _base_table(group_teams, results)
    match = team_rem[0]
    idx = rem.index(match)
    a, b = match
    team_is_a = team == a

    records = []  # (team_result in {'W','D','L'}, qualified bool)
    for combo in product("HDA", repeat=len(rem)):
        top2 = _top2(group_teams, pts0, gd0, gf0, rem, combo)
        o = combo[idx]
        won = (o == "H" and team_is_a) or (o == "A" and not team_is_a)
        res = "W" if won else ("D" if o == "D" else "L")
        records.append((res, team in top2))

    if all(q for _, q in records):
        return "SECURE"
    if not any(q for _, q in records):
        return "ELIMINATED"
    if not any(q for res, q in records if res != "W"):
        return "MUST_WIN"          # only ever qualifies by winning
    if all(q for res, q in records if res != "L"):
        return "DRAW_OK"           # a draw is always enough
    return "ALIVE"


def classify_match(group_teams: list[str], home: str, away: str, results: pd.DataFrame) -> tuple[str, str]:
    """(scenario_home, scenario_away) for an upcoming group match."""
    return (
        classify_team(home, away, group_teams, results),
        classify_team(away, home, group_teams, results),
    )


def adjust_expected_goals(
    mu_home: float, mu_away: float, scen_home: str, scen_away: str
) -> tuple[float, float]:
    """Apply motivation multipliers to expected goals given both scenarios."""
    mh = mu_home * OWN_FACTOR[scen_home] * OPP_FACTOR[scen_away]
    ma = mu_away * OWN_FACTOR[scen_away] * OPP_FACTOR[scen_home]
    return mh, ma
