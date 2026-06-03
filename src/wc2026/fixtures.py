"""2026 World Cup fixtures: load the group draw and build the match list.

The group draw lives in data/groups_2026.csv (source: the 2026 FIFA World Cup
final draw, 5 Dec 2025). Group-stage fixtures are the round-robin within each
group. Host nations (USA/CAN/MEX) get the venue boost on their group matches —
they play those in their own country.

Knockout fixtures are NOT pre-built: under Dacopa you predict each KO match once
the two teams are known, so those are generated round-by-round later from the
actual bracket.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
GROUPS_CSV = DATA_DIR / "groups_2026.csv"
RESULTS_2026_CSV = DATA_DIR / "results_2026.csv"

# Host nations get the home/host boost (only these three, per CLAUDE.md).
HOSTS = {"United States", "Canada", "Mexico"}


def load_groups(path: Path = GROUPS_CSV) -> pd.DataFrame:
    """Load the group draw as a (group, team) table."""
    return pd.read_csv(path)


def load_knockout_fixtures(path: Path = DATA_DIR / "knockout_fixtures_2026.csv") -> pd.DataFrame:
    """Load confirmed knockout matchups (empty until the bracket fills).

    Columns: round, home_team, away_team, and optionally host_home/host_away.
    Filled in round by round as the two teams of each KO match are known —
    Dacopa has no pre-bracket commitment, so we predict each once confirmed.
    """
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_results_2026(path: Path = RESULTS_2026_CSV) -> pd.DataFrame:
    """Load played 2026 results so far (empty until the tournament starts).

    Columns: date, home_team, away_team, home_score, away_score, neutral,
    matchday, group. Appended to as matches finish; feeds in-tournament rating
    updates and (later) the qualification-scenario layer.
    """
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns and len(df):
        df["date"] = pd.to_datetime(df["date"])
    return df


def group_stage_fixtures(groups: pd.DataFrame) -> pd.DataFrame:
    """Round-robin fixtures within each group: 6 per group, 72 total.

    Columns: group, home_team, away_team, host_home, host_away. The home/away
    label is only meaningful for the host boost (matches are otherwise neutral).
    """
    rows = []
    for group, sub in groups.groupby("group"):
        teams = list(sub["team"])
        for home, away in combinations(teams, 2):
            rows.append({
                "group": group,
                "home_team": home,
                "away_team": away,
                "host_home": home in HOSTS,
                "host_away": away in HOSTS,
            })
    return pd.DataFrame(rows)
