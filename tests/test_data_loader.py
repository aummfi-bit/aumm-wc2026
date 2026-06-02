"""Tests for data_loader normalization.

The pure-function tests run on a synthetic frame so they don't depend on the
(gitignored) raw data. The integration test skips when the real file is absent.
"""

import pandas as pd
import pytest

from wc2026 import data_loader as dl
from wc2026.data_loader import normalize_results, MARTJ42_DIR, CANON_COLUMNS


def _raw() -> pd.DataFrame:
    # Mirrors the martj42 results.csv shape, including a future/unplayed row.
    return pd.DataFrame({
        "date": ["2010-06-11", "1970-06-21", "2024-03-01", "2030-01-01"],
        "home_team": ["South Africa", "Brazil", "Upper Volta", "France"],
        "away_team": ["Mexico", "Italy", "Ghana", "Spain"],
        "home_score": [1.0, 4.0, 2.0, None],
        "away_score": [1.0, 1.0, 0.0, None],
        "tournament": ["FIFA World Cup", "FIFA World Cup", "Friendly", "Friendly"],
        "neutral": [False, True, True, False],
        "city": ["Johannesburg", "Guadalajara", "Accra", "Paris"],
        "country": ["South Africa", "Mexico", "Ghana", "France"],
    })


def test_drops_unplayed_and_sets_schema():
    out = normalize_results(_raw())
    assert len(out) == 3                          # the scoreless future row is dropped
    assert list(out.columns) == CANON_COLUMNS
    assert out["home_score"].dtype.kind == "i"
    assert out["away_score"].dtype.kind == "i"
    assert out["neutral"].dtype.kind == "b"
    assert out["date"].is_monotonic_increasing    # sorted by date


def test_competitive_flag():
    out = normalize_results(_raw())
    comp = dict(zip(out["home_team"], out["competitive"]))
    assert comp["South Africa"] is True or comp["South Africa"]   # World Cup
    assert not comp["Upper Volta"]                                # Friendly


def test_neutral_coerces_from_strings():
    raw = _raw()
    raw["neutral"] = ["TRUE", "FALSE", "TRUE", "FALSE"]
    out = normalize_results(raw)
    assert out["neutral"].dtype.kind == "b"
    assert out["neutral"].sum() == 2


def test_canonicalize_former_names():
    out = normalize_results(_raw(), name_map={"Upper Volta": "Burkina Faso"})
    names = set(out["home_team"]) | set(out["away_team"])
    assert "Upper Volta" not in names
    assert "Burkina Faso" in names


@pytest.mark.skipif(
    not (MARTJ42_DIR / "results.csv").exists(),
    reason="raw data not present (gitignored)",
)
def test_load_results_real_file():
    df = dl.load_results()
    assert len(df) > 40000
    assert list(df.columns) == CANON_COLUMNS
    assert df["home_score"].dtype.kind == "i"
    # No unplayed rows survive, and no future matches past the data's intent.
    assert df[["home_score", "away_score"]].notna().all().all()
    assert df["date"].max().year <= 2026
