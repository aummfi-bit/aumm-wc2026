"""Load and normalize historical international match results.

Primary source: martj42/international_results (`data/raw/martj42/`), whose
schema is already close to our target. See `data/SOURCES.md` for provenance.

Target normalized schema (one row per played match):
    date          datetime64
    home_team     str   (former names canonicalized to current)
    away_team     str
    home_score    int
    away_score    int
    tournament    str
    neutral       bool  (played on neutral ground)
    competitive   bool  (not a friendly)

The normalization logic is a pure function (`normalize_results`) so it can be
unit-tested on a synthetic frame without the (gitignored) raw data present.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MARTJ42_DIR = RAW_DIR / "martj42"

CANON_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "tournament", "neutral", "competitive",
]

# Tournaments treated as non-competitive (down-weighted or filtered when
# fitting strength). Kept as a set so it is easy to extend.
FRIENDLY_TOURNAMENTS = {"Friendly"}


def load_former_names(path: Path | None = None) -> dict[str, str]:
    """Return a {former_name: current_name} map from former_names.csv.

    Names are mapped by string only (the date window in the source is ignored);
    former international names are distinct enough that this is unambiguous.
    """
    path = path or (MARTJ42_DIR / "former_names.csv")
    fn = pd.read_csv(path)
    return dict(zip(fn["former"], fn["current"]))


def canonicalize_teams(df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    """Replace former team names with current names in home/away columns."""
    if not name_map:
        return df
    df = df.copy()
    df["home_team"] = df["home_team"].replace(name_map)
    df["away_team"] = df["away_team"].replace(name_map)
    return df


def normalize_results(
    df: pd.DataFrame,
    name_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Normalize raw martj42 results into the canonical fitting schema.

    Drops unplayed matches (missing score — this also removes future scheduled
    rows), coerces types, adds the `competitive` flag, optionally canonicalizes
    former team names, and returns only CANON_COLUMNS sorted by date. Pure: no
    file IO.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Keep only played matches; unplayed/future rows have no score.
    df = df[df["home_score"].notna() & df["away_score"].notna()]
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    neutral = df["neutral"]
    if not pd.api.types.is_bool_dtype(neutral):
        # martj42 stores TRUE/FALSE; in pandas 3.0 these read as string dtype
        # (not object), so coerce via the literal strings rather than truthiness.
        neutral = neutral.astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    df["neutral"] = neutral.astype(bool)

    df["competitive"] = ~df["tournament"].isin(FRIENDLY_TOURNAMENTS)

    if name_map:
        df = canonicalize_teams(df, name_map)

    return df[CANON_COLUMNS].sort_values("date").reset_index(drop=True)


def load_results(
    filename: str = "results.csv",
    canonicalize: bool = True,
) -> pd.DataFrame:
    """Load and normalize the martj42 results CSV from data/raw/martj42/."""
    df = pd.read_csv(MARTJ42_DIR / filename)
    name_map = load_former_names() if canonicalize else None
    return normalize_results(df, name_map)
