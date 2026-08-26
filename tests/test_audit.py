from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gridcast.audit import (
    cost_of_a_near_miss,
    decision_detail,
    ordering_quality,
    reference_availability,
)
from gridcast.scheduling import Load

UTC = dt.UTC


def test_every_seasonal_reference_settles_well_before_any_issue():
    # The claim that rules the availability gate out as an explanation. The
    # tightest case is the shortest lag at the longest horizon: seven days back,
    # forecast forty-eight hours ahead, which still leaves 118.5 hours between
    # the reference settling and the forecast being issued.
    frame = reference_availability(Load(periods=4, window_hours=48.0))
    assert (frame["settled_before_issue_hours"] > 0).all()
    assert frame["settled_before_issue_hours"].min() == pytest.approx(118.5)


def test_availability_margin_shrinks_with_horizon_but_stays_positive():
    frame = reference_availability(Load(periods=4, window_hours=48.0))
    seven_day = frame[frame["lag_days"] == 7].sort_values("horizon_hours")
    assert seven_day["settled_before_issue_hours"].is_monotonic_decreasing
    assert seven_day["settled_before_issue_hours"].iloc[-1] > 0


def detail_frame(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "published_start": [r[0] for r in rows],
            "baseline_start": [r[1] for r in rows],
            "oracle_start": [r[2] for r in rows],
            "window_spread": [100.0] * len(rows),
            "placements": [45] * len(rows),
            "quartile_excess": [5.0] * len(rows),
            "mean_excess": [20.0] * len(rows),
        }
    )


def test_ordering_quality_measures_distance_not_cost():
    # Two failures a cost figure conflates: an adjacent window, and one on the
    # wrong side of the day.
    frame = detail_frame([(11, 0, 12), (12, 40, 12)])
    scored = ordering_quality(frame)

    assert scored.loc["published", "mean_periods_from_optimum"] == pytest.approx(0.5)
    assert scored.loc["published", "within_one_period"] == pytest.approx(1.0)
    assert scored.loc["baseline", "worse_than_twelve_periods"] == pytest.approx(0.5)


def test_excess_relative_to_spread_says_whether_precision_matters():
    frame = detail_frame([(0, 0, 0)])
    summary = cost_of_a_near_miss(frame)
    # Five gCO2/kWh between the best placement and the better quarter of them,
    # in a window spanning a hundred: near-misses are nearly free.
    assert summary.loc["mean", "quartile_over_spread"] == pytest.approx(0.05)
    assert summary.loc["mean", "mean_over_spread"] == pytest.approx(0.20)


def test_a_window_with_no_spread_does_not_divide_by_zero():
    frame = detail_frame([(0, 0, 0)])
    frame["window_spread"] = 0.0
    summary = cost_of_a_near_miss(frame)
    assert np.isnan(summary.loc["mean", "quartile_over_spread"])


def test_excess_is_taken_at_a_rank_fraction_so_windows_compare():
    # A fixed rank counts a different part of the distribution in a six-hour
    # window with eleven placements than in a day with forty-five, and would
    # report the window length as though it were the shape of the grid.
    short = detail_frame([(0, 0, 0)])
    short["placements"] = 11
    long = detail_frame([(0, 0, 0)])

    assert cost_of_a_near_miss(short).loc["mean", "quartile_over_spread"] == pytest.approx(
        cost_of_a_near_miss(long).loc["mean", "quartile_over_spread"]
    )


def test_diagnostics_are_empty_on_an_empty_store(tmp_path):
    assert decision_detail(tmp_path).empty
    assert ordering_quality(pd.DataFrame()).empty
    assert cost_of_a_near_miss(pd.DataFrame()).empty


def test_audit_command_runs_without_matched_decisions(tmp_path, capsys):
    from gridcast.cli import main

    assert main(["--root", str(tmp_path), "audit"]) == 0
    out = capsys.readouterr().out
    assert "settled before the issue time" in out
    assert "no matched decisions" in out


# -- integration against a store ------------------------------------------

BASE = dt.datetime(2026, 6, 1, tzinfo=UTC)


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


@pytest.fixture
def audited_store(tmp_path):
    """History with a daily cycle, plus a forward forecast that errs late."""
    from gridcast.storage import write_snapshot

    history = []
    for step in range(48 * 40):
        moment = BASE + dt.timedelta(minutes=30 * step)
        value = 200.0 + 60.0 * np.sin(2 * np.pi * (step % 48) / 48)
        history.append(
            (
                moment.strftime("%Y-%m-%dT%H:%MZ"),
                (moment + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                value,
                value,
            )
        )
    write_snapshot(
        intensity_payload(history),
        "intensity_range_20260601",
        "/intensity/range",
        BASE + dt.timedelta(days=41),
        root=tmp_path,
    )

    issue = BASE + dt.timedelta(days=35)
    forward = []
    for step in range(12):
        moment = issue + dt.timedelta(minutes=30 * step)
        index = int((moment - BASE) / dt.timedelta(minutes=30))
        truth = 200.0 + 60.0 * np.sin(2 * np.pi * (index % 48) / 48)
        # A forecast displaced by one period: right shape, wrong phase.
        shifted = 200.0 + 60.0 * np.sin(2 * np.pi * ((index + 1) % 48) / 48)
        forward.append(
            (
                moment.strftime("%Y-%m-%dT%H:%MZ"),
                (moment + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                shifted,
                None,
            )
        )
        del truth
    write_snapshot(
        intensity_payload(forward),
        "forecast_fw48h",
        "/intensity/issue/fw48h",
        issue - dt.timedelta(minutes=30),
        root=tmp_path,
    )
    return tmp_path


def test_decision_detail_reports_each_method_and_the_window_shape(audited_store):
    detail = decision_detail(audited_store, Load(periods=2, window_hours=5.0))

    assert not detail.empty
    for column in (
        "published_start",
        "baseline_start",
        "oracle_start",
        "window_spread",
        "placements",
        "quartile_excess",
        "mean_excess",
        "forecast_bias",
    ):
        assert column in detail.columns
    assert (detail["window_spread"] > 0).all()


def test_detail_costs_are_ordered_with_hindsight_cheapest(audited_store):
    detail = decision_detail(audited_store, Load(periods=2, window_hours=5.0))
    assert (detail["published_cost"] >= detail["oracle_cost"] - 1e-9).all()
    assert (detail["baseline_cost"] >= detail["oracle_cost"] - 1e-9).all()


def test_audit_command_prints_every_section(audited_store, capsys):
    from gridcast.cli import main

    exit_code = main(["--root", str(audited_store), "audit", "--periods", "2", "--window", "5"])
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "settled before the issue time" in out
    assert "distance from the true optimum" in out
    assert "window shape" in out
    assert "costliest published decisions" in out
