"""Tests for fixture construction and the 2026 group draw."""

from collections import Counter

import pandas as pd

from wc2026.fixtures import (
    load_groups, group_stage_fixtures, GROUPS_CSV,
    load_schedule, scheduled_fixtures, SCHEDULE_CSV, HOSTS,
)


def test_round_robin_counts_and_hosts():
    groups = pd.DataFrame({
        "group": ["A"] * 4 + ["B"] * 4,
        "team": ["Mexico", "T2", "T3", "T4", "Canada", "U2", "U3", "U4"],
    })
    fx = group_stage_fixtures(groups)
    assert len(fx) == 12                              # 6 per group of 4
    plays = Counter(list(fx["home_team"]) + list(fx["away_team"]))
    assert all(v == 3 for v in plays.values())        # each team plays 3
    # Mexico is a host -> flagged whenever it appears.
    mex = fx[(fx.home_team == "Mexico") | (fx.away_team == "Mexico")]
    assert (mex["host_home"] | mex["host_away"]).all()
    # A non-host group-B match has no host flag.
    plain = fx[(fx.group == "B") & (fx.home_team != "Canada") & (fx.away_team != "Canada")]
    assert not plain["host_home"].any() and not plain["host_away"].any()


def test_2026_groups_well_formed():
    g = load_groups()
    assert len(g) == 48
    assert g["group"].nunique() == 12
    assert (g.groupby("group").size() == 4).all()     # exactly 4 per group
    assert g["team"].nunique() == 48                   # no duplicate teams


def test_2026_full_slate_is_72():
    assert len(group_stage_fixtures(load_groups())) == 72


def test_groups_csv_exists():
    assert GROUPS_CSV.exists()


def test_schedule_is_complete_and_consistent():
    if not SCHEDULE_CSV.exists():
        return
    sched = load_schedule()
    assert len(sched) == 72
    assert sorted(sched["matchday"].unique().tolist()) == [1, 2, 3]
    assert (sched.groupby("matchday").size() == 24).all()        # 24 per round
    # The schedule's fixtures must be exactly the round-robin pairs of the draw.
    sched_pairs = {frozenset((r.home_team, r.away_team)) for r in sched.itertuples()}
    rr_pairs = {frozenset((r.home_team, r.away_team))
                for r in group_stage_fixtures(load_groups()).itertuples()}
    assert sched_pairs == rr_pairs


def test_scheduled_fixtures_columns_and_hosts():
    if not SCHEDULE_CSV.exists():
        return
    fx = scheduled_fixtures(load_schedule())
    assert {"date", "matchday", "group", "home_team", "away_team",
            "host_home", "host_away"} == set(fx.columns)
    # A host nation is flagged whenever it appears on either side.
    mex = fx[(fx.home_team == "Mexico") | (fx.away_team == "Mexico")]
    assert (mex["host_home"] | mex["host_away"]).all()
