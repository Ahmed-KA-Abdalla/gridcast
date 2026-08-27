from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gridcast.revisions import (
    error_by_lead,
    path_summary,
    revision_autocorrelation,
    revision_paths,
    revision_predicts_error,
)
from gridcast.storage import write_snapshot

UTC = dt.UTC
TARGET = dt.datetime(2026, 8, 25, 12, 0, tzinfo=UTC)


def intensity_payload(entries):
    return {
        "data": [
            {
                "from": start,
                "to": end,
                "intensity": {"forecast": forecast, "actual": actual, "index": "moderate"},
            }
            for start, end, forecast, actual in entries
        ]
    }


def stamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%MZ")


def write_issue(root, issue: dt.datetime, entries: list[tuple[dt.datetime, float]]) -> None:
    payload = intensity_payload(
        [
            (stamp(period), stamp(period + dt.timedelta(minutes=30)), value, None)
            for period, value in entries
        ]
    )
    write_snapshot(payload, "forecast_fw48h", "/intensity/issue/fw48h", issue, root=root)


def write_outcomes(root, entries: list[tuple[dt.datetime, float]]) -> None:
    payload = intensity_payload(
        [
            (stamp(period), stamp(period + dt.timedelta(minutes=30)), value, value)
            for period, value in entries
        ]
    )
    write_snapshot(
        payload,
        "outcomes_pt24h",
        "/intensity/pt24h",
        TARGET + dt.timedelta(hours=3),
        root=root,
    )


@pytest.fixture
def drifting_store(tmp_path):
    """A forecast revised steadily upwards towards the eventual outcome."""
    for step, value in enumerate([100.0, 110.0, 120.0, 130.0]):
        issue = TARGET - dt.timedelta(hours=8 - 2 * step)
        write_issue(tmp_path, issue, [(TARGET, value)])
    write_outcomes(tmp_path, [(TARGET, 150.0)])
    return tmp_path


def test_revision_paths_orders_issues_and_differences_them(drifting_store):
    paths = revision_paths(drifting_store)

    assert paths["forecast"].tolist() == [100.0, 110.0, 120.0, 130.0]
    assert paths["issue_index"].tolist() == [0, 1, 2, 3]
    # The first issue revised nothing.
    assert np.isnan(paths["revision"].iloc[0])
    assert paths["revision"].iloc[1:].tolist() == [10.0, 10.0, 10.0]


def test_horizon_shrinks_along_a_path(drifting_store):
    paths = revision_paths(drifting_store)
    assert paths["horizon_hours"].is_monotonic_decreasing


def test_path_summary_separates_wandering_from_converging(drifting_store):
    summary = path_summary(revision_paths(drifting_store)).iloc[0]

    assert summary["issues"] == 4
    assert summary["net_movement"] == pytest.approx(30.0)
    assert summary["total_movement"] == pytest.approx(30.0)
    # Moved monotonically, so nothing was retraced.
    assert summary["wander"] == pytest.approx(0.0)


def test_wander_counts_movement_that_was_later_undone(tmp_path):
    for step, value in enumerate([100.0, 160.0, 100.0]):
        write_issue(tmp_path, TARGET - dt.timedelta(hours=6 - 2 * step), [(TARGET, value)])
    write_outcomes(tmp_path, [(TARGET, 100.0)])

    summary = path_summary(revision_paths(tmp_path)).iloc[0]
    assert summary["net_movement"] == pytest.approx(0.0)
    assert summary["total_movement"] == pytest.approx(120.0)
    assert summary["wander"] == pytest.approx(120.0)


def test_a_steadily_drifting_forecast_has_positive_autocorrelation(tmp_path):
    # Under-reaction: each revision continues in the direction of the last, so
    # part of the next one is already implied.
    rng = np.random.default_rng(0)
    for index in range(30):
        target = TARGET + dt.timedelta(minutes=30 * index)
        base = 100.0 + rng.normal(0, 5)
        # The size of the drift varies between periods; within a period it
        # persists. A constant drift would leave the revisions with no variance
        # and the correlation undefined.
        drift = rng.normal(10.0, 5.0)
        for step in range(4):
            write_issue(
                tmp_path,
                target - dt.timedelta(hours=8 - 2 * step),
                [(target, base + drift * step)],
            )

    scored = revision_autocorrelation(revision_paths(tmp_path))
    overall = scored[scored["lead"] == "all"].iloc[0]
    assert overall["autocorrelation"] > 0.5


def test_an_overshooting_forecast_has_negative_autocorrelation(tmp_path):
    for index in range(30):
        target = TARGET + dt.timedelta(minutes=30 * index)
        for step, value in enumerate([100.0, 140.0, 100.0, 140.0]):
            write_issue(tmp_path, target - dt.timedelta(hours=8 - 2 * step), [(target, value)])

    scored = revision_autocorrelation(revision_paths(tmp_path))
    overall = scored[scored["lead"] == "all"].iloc[0]
    assert overall["autocorrelation"] < -0.5


def test_autocorrelation_is_reported_by_lead_as_well_as_overall(tmp_path):
    rng = np.random.default_rng(1)
    for index in range(40):
        target = TARGET + dt.timedelta(minutes=30 * index)
        for hours in (30.0, 20.0, 8.0, 2.0):
            write_issue(
                tmp_path,
                target - dt.timedelta(hours=hours),
                [(target, 100.0 + rng.normal(0, 10))],
            )

    scored = revision_autocorrelation(revision_paths(tmp_path))
    assert len(scored) > 1
    assert "all" in scored["lead"].tolist()


def test_error_by_lead_can_restrict_to_periods_seen_at_every_lead(tmp_path):
    # One period forecast at two leads, another at one. Matched mode keeps only
    # the first, so the bands describe the same days.
    both = TARGET
    one = TARGET + dt.timedelta(minutes=30)

    write_issue(tmp_path, both - dt.timedelta(hours=30), [(both, 100.0)])
    write_issue(tmp_path, both - dt.timedelta(hours=2), [(both, 120.0)])
    write_issue(tmp_path, one - dt.timedelta(hours=2), [(one, 200.0)])
    write_outcomes(tmp_path, [(both, 130.0), (one, 210.0)])

    matched = error_by_lead(tmp_path, matched=True)
    pooled = error_by_lead(tmp_path, matched=False)

    assert matched["periods"].max() == 1
    assert pooled["n"].sum() > matched["n"].sum()


def test_revision_predicts_error_detects_a_forecast_that_under_reacts(drifting_store):
    scored = revision_predicts_error(drifting_store)
    overall = scored[scored["lead"] == "all"].iloc[0]
    # Revised upwards each time and still finished below the outcome, so a
    # revision upwards implied more error remaining in the same direction.
    assert overall["n"] == 3


def test_a_forecast_revised_by_a_constant_has_no_defined_autocorrelation(tmp_path):
    # Every revision identical leaves the series with no variance, so the
    # correlation is undefined rather than zero. Reporting it as zero would
    # claim efficiency on the strength of a degenerate sample.
    for index in range(20):
        target = TARGET + dt.timedelta(minutes=30 * index)
        for step in range(4):
            write_issue(
                tmp_path,
                target - dt.timedelta(hours=8 - 2 * step),
                [(target, 100.0 + 10.0 * step)],
            )

    scored = revision_autocorrelation(revision_paths(tmp_path))
    overall = scored[scored["lead"] == "all"].iloc[0]
    assert np.isnan(overall["autocorrelation"])
    assert overall["n"] > 0


def test_everything_is_empty_on_an_empty_store(tmp_path):
    assert revision_paths(tmp_path).empty
    assert path_summary(pd.DataFrame()).empty
    assert revision_autocorrelation(pd.DataFrame()).empty
    assert error_by_lead(tmp_path).empty
    assert revision_predicts_error(tmp_path).empty


def test_revisions_command_reports_every_section(drifting_store, capsys):
    from gridcast.cli import main

    assert main(["--root", str(drifting_store), "revisions"]) == 0
    out = capsys.readouterr().out
    assert "captures of them" in out
    assert "captures finding no change" in out
    assert "median movement retraced" in out


def test_revisions_command_says_so_on_an_empty_store(tmp_path, capsys):
    from gridcast.cli import main

    assert main(["--root", str(tmp_path), "revisions"]) == 0
    assert "no captured forecasts" in capsys.readouterr().out


# -- distinct revisions ---------------------------------------------------


def build_paths(period_values: dict[str, list[float]]) -> pd.DataFrame:
    """A captured path per period, one capture an hour."""
    rows = []
    for period, values in period_values.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "period_start": pd.Timestamp(period),
                    "captured_at": pd.Timestamp("2026-08-25T00:00Z") + pd.Timedelta(hours=index),
                    "forecast": value,
                    "horizon_hours": 12.0 - index,
                }
            )
    paths = pd.DataFrame(rows)
    grouped = paths.groupby("period_start", sort=False)
    paths["revision"] = grouped["forecast"].diff()
    paths["previous_revision"] = grouped["revision"].shift()
    paths["issue_index"] = grouped.cumcount()
    return paths


def test_distinct_revisions_collapses_repeated_values():
    from gridcast.revisions import distinct_revisions

    paths = build_paths({"2026-08-25T12:00Z": [100.0, 100.0, 110.0, 110.0, 110.0, 105.0]})
    distinct = distinct_revisions(paths)

    assert distinct["forecast"].tolist() == [100.0, 110.0, 105.0]
    assert distinct["revision"].iloc[1:].tolist() == [10.0, -5.0]
    assert distinct["change_index"].tolist() == [0, 1, 2]


def test_held_captures_counts_how_long_a_value_stood():
    from gridcast.revisions import distinct_revisions

    paths = build_paths({"2026-08-25T12:00Z": [100.0, 100.0, 110.0, 110.0, 110.0, 105.0]})
    distinct = distinct_revisions(paths)
    # Two captures saw 100, three saw 110, one saw 105.
    assert distinct["held_captures"].tolist() == [2, 3, 1]


def test_hours_since_previous_measures_the_gap_between_changes():
    from gridcast.revisions import distinct_revisions

    paths = build_paths({"2026-08-25T12:00Z": [100.0, 100.0, 110.0, 110.0, 110.0, 105.0]})
    distinct = distinct_revisions(paths)
    assert distinct["hours_since_previous"].iloc[1:].tolist() == [2.0, 3.0]


def test_distinct_revisions_keeps_periods_separate():
    from gridcast.revisions import distinct_revisions

    paths = build_paths(
        {"2026-08-25T12:00Z": [100.0, 100.0, 110.0], "2026-08-25T12:30Z": [200.0, 205.0, 205.0]}
    )
    distinct = distinct_revisions(paths)

    assert len(distinct) == 4
    assert distinct.groupby("period_start")["change_index"].max().tolist() == [1, 1]
    # The first value of each period revised nothing.
    assert distinct[distinct["change_index"] == 0]["revision"].isna().all()


def test_a_stable_forecast_that_never_changes_yields_one_row():
    from gridcast.revisions import distinct_revisions

    paths = build_paths({"2026-08-25T12:00Z": [100.0] * 8})
    distinct = distinct_revisions(paths)

    assert len(distinct) == 1
    assert distinct["held_captures"].iloc[0] == 8


def test_repeated_captures_pull_the_correlation_towards_zero_not_minus_a_half():
    # Worth stating because the opposite is easy to assume. A real change
    # followed by captures that repeat it inserts zeros into the differenced
    # series, and zeros dilute the correlation towards nothing. Repeats are a
    # reason to distrust the magnitude, not a source of negative structure.
    from gridcast.revisions import distinct_revisions

    rng = np.random.default_rng(3)
    values = {}
    for index in range(60):
        period = pd.Timestamp("2026-08-25T12:00Z") + pd.Timedelta(minutes=30 * index)
        level = 100.0
        series = []
        for _ in range(4):
            level = level + rng.normal(0, 10)
            series.extend([level, level, level])
        values[str(period)] = series

    paths = build_paths(values)
    captured = revision_autocorrelation(paths)
    captured_value = captured[captured["lead"] == "all"]["autocorrelation"].iloc[0]

    assert abs(captured_value) < 0.2
    # The underlying walk is a random walk, whose revisions are uncorrelated.
    collapsed = revision_autocorrelation(distinct_revisions(paths))
    assert abs(collapsed[collapsed["lead"] == "all"]["autocorrelation"].iloc[0]) < 0.3


def test_a_forecast_that_jitters_around_a_level_gives_minus_a_half():
    # The structure that does produce minus a half. If each forecast is a stable
    # level plus independent noise, consecutive differences share one noise term
    # with opposite sign, and the correlation goes to exactly -0.5. A forecast
    # behaving this way retraces most of its own movement rather than
    # converging.
    rng = np.random.default_rng(5)
    values = {}
    for index in range(80):
        period = pd.Timestamp("2026-08-25T12:00Z") + pd.Timedelta(minutes=30 * index)
        values[str(period)] = list(120.0 + rng.normal(0, 8, 10))

    scored = revision_autocorrelation(build_paths(values))
    overall = scored[scored["lead"] == "all"]["autocorrelation"].iloc[0]
    assert overall == pytest.approx(-0.5, abs=0.1)


def test_wander_distinguishes_oscillation_from_convergence():
    # The statistic that corroborates the interpretation: a forecast that
    # oscillates moves far in total while ending near where it began.
    rng = np.random.default_rng(7)
    jittery = build_paths({"2026-08-25T12:00Z": list(120.0 + rng.normal(0, 8, 10))})
    converging = build_paths({"2026-08-25T12:00Z": [100.0 + 5.0 * step for step in range(10)]})

    jittery_summary = path_summary(jittery).iloc[0]
    converging_summary = path_summary(converging).iloc[0]

    assert jittery_summary["wander"] > jittery_summary["net_movement"].__abs__()
    assert converging_summary["wander"] == pytest.approx(0.0)


def test_refresh_cadence_reports_the_interval_between_changes():
    from gridcast.revisions import distinct_revisions, refresh_cadence

    paths = build_paths({"2026-08-25T12:00Z": [100.0, 100.0, 110.0, 110.0, 110.0, 105.0]})
    cadence = refresh_cadence(distinct_revisions(paths))

    assert not cadence.empty
    assert cadence["changes"].sum() == 2
    assert cadence["median_hours_between"].iloc[0] > 0


def test_refresh_cadence_is_empty_when_nothing_ever_changed():
    from gridcast.revisions import distinct_revisions, refresh_cadence

    paths = build_paths({"2026-08-25T12:00Z": [100.0] * 6})
    assert refresh_cadence(distinct_revisions(paths)).empty
