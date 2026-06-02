"""Load and normalize historical international match results.

Fill in to match the actual Kaggle dataset schema once the CSV is in
data/raw/. Target normalized columns:
    date, home_team, away_team, home_score, away_score, tournament, neutral

Common Kaggle international-results datasets (e.g. "International football
results 1872-present") already use these names — adapt as needed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_results(filename: str) -> pd.DataFrame:
    """Load a results CSV from data/raw and normalize columns."""
    df = pd.read_csv(RAW_DIR / filename, parse_dates=["date"])
    # TODO: rename/clean to the normalized schema once the real columns are known.
    return df
