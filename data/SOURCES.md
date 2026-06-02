# Data provenance

Raw files live in `data/raw/` (gitignored). This file records exactly what was
fetched and from where, so the inputs are reproducible. Re-fetch by checking out
the pinned commit SHA below.

## martj42/international_results

- Repo: https://github.com/martj42/international_results
- Pinned commit: `6637d6d7d6ae164a9732f845f64037e03edced26`
- Fetched: 2026-06-02
- Files → `data/raw/martj42/`:
  - `results.csv` — 49,363 international matches, 1872-11-30 → 2026-06-27.
    Columns: date, home_team, away_team, home_score, away_score, tournament,
    city, country, neutral. ~13k neutral. **Contains future/unplayed rows
    (date up to 2026-06-27) that must be filtered out before fitting.**
  - `shootouts.csv` — penalty-shootout records (which matches went to pens).
  - `former_names.csv` — historical team renames, for name canonicalization.

## jfjelstul/worldcup

- Repo: https://github.com/jfjelstul/worldcup
- Pinned commit: `f41e9437a007498bdbf3751305818101f96cb6fb`
- Fetched: 2026-06-02
- Files → `data/raw/jfjelstul/` (from the repo's `data-csv/`):
  - `matches.csv` — 1,248 World Cup matches. Has `group_stage` / `knockout_stage`
    phase flags, `extra_time` and `penalty_shootout` flags, and separate penalty
    scores. **`home_team_score`/`away_team_score` is the full-time score INCLUDING
    extra time** — see the 90-minute caveat below.
  - `tournaments.csv`, `teams.csv`, `host_countries.csv`, `stadiums.csv` —
    supporting tables (hosts matter for the USA/CAN/MEX boost).

## 90-minute scoring caveat (affects KO backtesting)

The pool scores knockouts on the **90-minute** result (extra-time goals and
penalties stripped; see CLAUDE.md). jfjelstul records the post-ET score for KO
games, so for any match with `extra_time == 1` the recorded score is NOT the
90-minute score. Useful fact: a game that reached extra time was, by definition,
**level at 90 minutes**, so its 90-minute *outcome* is a draw — only the exact
90-minute draw scoreline is unknown and would need separate sourcing/approximation.
