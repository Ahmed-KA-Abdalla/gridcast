from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gridcast.baseline import add_baselines, seasonal_mean, seasonal_naive

UTC = dt.UTC


def outcomes_frame(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    """Build an outcome record from (period, actual, settled_at) triples."""
    return pd.DataFrame(
        {
            "period_start": pd.to_datetime([r[0] for r in rows], utc=True),
            "actual": [r[1] for r in rows],
            "settled_at": pd.to_datetime([r[2] for r in rows], utc=True),
        }
    )


def targets_frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """Build a target frame from (period, captured_at) pairs."""
    return pd.DataFrame(
        {
            "period_start": pd.to_datetime([r[0] for r in rows], utc=True),
            "captured_at": pd.to_datetime([r[1] for r in rows], utc=True),
        }
    )


@pytest.fixture
def history() -> pd.DataFrame:
    # The same half-hour on four successive Thursdays.
    return outcomes_frame(
        [
            ("2026-07-30T18:00Z", 100.0, "2026-07-30T19:00Z"),
            ("2026-08-06T18:00Z", 200.0, "2026-08-06T19:00Z"),
            ("2026-08-13T18:00Z", 300.0, "2026-08-13T19:00Z"),
        ]
    )


def test_seasonal_naive_takes_the_previous_week(history):
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-20T12:00Z")])
    assert seasonal_naive(targets, history).tolist() == [300.0]


def test_seasonal_mean_averages_the_available_lags(history):
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-20T12:00Z")])
    assert seasonal_mean(targets, history).tolist() == [200.0]


def test_seasonal_naive_falls_back_when_a_week_is_missing(history):
    # The 13 August observation is dropped, as a gap in the record would.
    gapped = history[history["period_start"] != pd.Timestamp("2026-08-13T18:00Z")]
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-20T12:00Z")])
    assert seasonal_naive(targets, gapped).tolist() == [200.0]


def test_a_reference_that_had_not_settled_is_not_used(history):
    # Predicting as of 18:45 on 13 August: that day's 18:00 period has ended but
    # the settlement allowance has not elapsed, so the prediction must fall back
    # to the week before.
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-13T18:45Z")])
    assert seasonal_naive(targets, history).tolist() == [200.0]


def test_a_reference_becomes_available_once_the_allowance_has_elapsed(history):
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-13T19:45Z")])
    assert seasonal_naive(targets, history).tolist() == [300.0]


def test_availability_ignores_when_the_project_happened_to_fetch_the_value():
    # A backfilled observation carries today's capture time. Were that used as
    # the availability test, two years of settled history would be treated as
    # unknown to any forecast issued before the backfill ran.
    backfilled = outcomes_frame(
        [("2026-08-13T18:00Z", 300.0, "2026-08-21T02:00Z")],
    )
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-20T12:00Z")])
    assert seasonal_naive(targets, backfilled).tolist() == [300.0]


def test_the_settlement_check_can_be_disabled_for_a_retrospective_backtest(history):
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-13T18:15Z")])
    assert seasonal_naive(targets, history, as_of=None).tolist() == [300.0]


def test_prediction_is_missing_when_no_lag_is_available(history):
    # A target 200 days on has no reference within the configured lags.
    targets = targets_frame([("2027-03-11T18:00Z", "2027-03-11T12:00Z")])
    assert np.isnan(seasonal_naive(targets, history).iloc[0])


def test_seasonal_mean_ignores_unavailable_lags_rather_than_zeroing_them(history):
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-13T18:45Z")])
    # Only the 30 July and 6 August values had settled: mean of 100 and 200.
    assert seasonal_mean(targets, history).tolist() == [150.0]


def test_the_lag_matches_the_half_hour_not_merely_the_day(history):
    extra = pd.concat(
        [history, outcomes_frame([("2026-08-13T18:30Z", 999.0, "2026-08-13T19:30Z")])],
        ignore_index=True,
    )
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-20T12:00Z")])
    # 18:30 a week earlier must not be used to predict 18:00.
    assert seasonal_naive(targets, extra).tolist() == [300.0]


def test_add_baselines_leaves_the_input_untouched(history):
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-20T12:00Z")])
    result = add_baselines(targets, history)

    assert list(result.columns[-2:]) == ["seasonal_naive", "seasonal_mean"]
    assert "seasonal_naive" not in targets.columns
