"""Manual lineup / team-news overrides — the highest-confidence in-tournament input.

Near kickoff (~1h before, when we bet) the team news is out: a qualified side
resting six starters, or a key injury, is a bigger and far more CERTAIN strength
change than any scenario prior. This layer lets you hand-apply an Elo delta to a
team — for a specific match (rotation) or all its remaining matches (injury) —
which takes precedence over the model's motivation guess.

data/overrides_2026.csv columns:
    team       team to adjust
    opponent   specific opponent (match-level), or blank / * for all matches
    elo_delta  Elo points to add (negative = weaker; e.g. -150 for heavy rotation)
    note       free text (why)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OVERRIDES_CSV = Path(__file__).resolve().parents[2] / "data" / "overrides_2026.csv"


def load_overrides(path: Path = OVERRIDES_CSV) -> pd.DataFrame:
    """Load manual overrides (empty if the file is absent or has no rows)."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _is_team_level(opponent) -> bool:
    return not isinstance(opponent, str) or opponent.strip() in ("", "*")


def elo_delta_for(overrides: pd.DataFrame, team: str, opponent: str) -> float:
    """Total Elo delta to apply to `team` for its match vs `opponent`.

    Sums every override row for `team` whose opponent is blank/* (team-level) or
    matches `opponent` (match-level). Returns 0.0 if none apply.
    """
    if overrides is None or len(overrides) == 0:
        return 0.0
    total = 0.0
    for r in overrides.itertuples(index=False):
        if r.team != team:
            continue
        if _is_team_level(r.opponent) or str(r.opponent).strip() == opponent:
            total += float(r.elo_delta)
    return total
