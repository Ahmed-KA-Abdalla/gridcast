from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gridcast.evaluate import (
    backtest_baselines,
    compare_at_matched_leads,
    metrics,
    score_columns,
    skill,
)
from gridcast.storage import write_snapshot

UTC = dt.UTC


def test_metrics_computes_error_bias_and_spread():
    prediction = pd.Series([10.0, 20.0, 30.0])
    actual = pd.Series([12.0, 18.0, 30.0])

    result = metrics(prediction, actual)
    assert result["n"] == 3
    assert result["mae"] == pytest.approx(4 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(8 / 3))
    # Errors of -2, +2 and 0 cancel: no systematic bias despite real error.
    assert result["bias"] == pytest.approx(0.0)


def test_metrics_excludes_missing_predictions_rather_than_scoring_them():
    prediction = pd.Series([10.0, np.nan, 30.0])
    actual = pd.Series([12.0, 18.0, 30.0])

    result = metrics(prediction, actual)
    assert result["n"] == 2
    assert result["mae"] == pytest.approx(1.0)


def test_metrics_reports_nothing_rather_than_zero_on_an_empty_input():
    result = metrics(pd.Series(dtype=float), pd.Series(dtype=float))
    assert result["n"] == 0
    assert np.isnan(result["mae"])


def test_skill_is_positive_when_the_candidate_is_better():
    assert skill({"mae": 8.0}, {"mae": 10.0, "n": 5}) == pytest.approx(0.2)


def test_skill_is_negative_when_the_candidate_is_worse():
    assert skill({"mae": 12.0}, {"mae": 10.0, "n": 5}) == pytest.approx(-0.2)


def test_skill_is_undefined_against_an_empty_reference():
    assert np.isnan(skill({"mae": 8.0}, {"mae": np.nan, "n": 0}))


def test_score_columns_scores_each_model_on_the_same_rows():
    frame = pd.DataFrame(
        {
            "actual": [100.0, 200.0],
            "good": [101.0, 199.0],
            "poor": [150.0, 150.0],
        }
    )
    scored = score_columns(frame, ["good", "poor"])
    assert scored.loc["good", "mae"] == pytest.approx(1.0)
    assert scored.loc["poor", "mae"] == pytest.approx(50.0)


# -- integration against a small store ------------------------------------

BASE = dt.datetime(2026, 8, 20, 18, 0, tzinfo=UTC)


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
def store(tmp_path):
    """Four Thursdays of history plus one captured forecast for the fifth."""
    history = []
    for week in range(4):
        day = BASE - dt.timedelta(days=7 * (4 - week))
        history.append(
            (
                day.strftime("%Y-%m-%dT%H:%MZ"),
                (day + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                200.0,
                200.0,
            )
        )
    # The target period itself, settled.
    history.append(("2026-08-20T18:00Z", "2026-08-20T18:30Z", 210.0, 220.0))

    write_snapshot(
        intensity_payload(history),
        "intensity_range_20260701",
        "/intensity/range",
        BASE + dt.timedelta(hours=2),
        root=tmp_path,
    )
    # A forecast for the target, issued six hours ahead.
    write_snapshot(
        intensity_payload([("2026-08-20T18:00Z", "2026-08-20T18:30Z", 210.0, None)]),
        "forecast_fw48h",
        "/intensity/2026-08-20T12:00Z/fw48h",
        BASE - dt.timedelta(hours=6),
        root=tmp_path,
    )
    return tmp_path


def test_backtest_baselines_scores_the_whole_outcome_record(store):
    scored = backtest_baselines(store)
    assert set(scored.index) == {"seasonal_naive", "seasonal_mean"}
    # Four of the five periods are predictable from an earlier week. Three of
    # them read 200 against a reference of 200; the target read 220 against 200.
    assert scored.loc["seasonal_naive", "n"] == 4
    assert scored.loc["seasonal_naive", "mae"] == pytest.approx(5.0)


def test_compare_scores_the_published_forecast_beside_the_baselines(store):
    scored = compare_at_matched_leads(store)
    assert not scored.empty

    bucket = scored.index[0]
    # Published forecast 210 against an outcome of 220; baseline 200.
    assert scored.loc[bucket, ("mae", "forecast")] == pytest.approx(10.0)
    assert scored.loc[bucket, ("mae", "seasonal_naive")] == pytest.approx(20.0)


def test_comparison_is_empty_when_nothing_has_been_captured(tmp_path):
    assert compare_at_matched_leads(tmp_path).empty
    assert backtest_baselines(tmp_path).empty


def test_compare_command_prints_both_tables(store, capsys):
    from gridcast.cli import main

    assert main(["--root", str(store), "compare"]) == 0
    out = capsys.readouterr().out
    assert "whole outcome record" in out
    assert "matched rows only" in out


def test_compare_command_says_so_on_an_empty_store(tmp_path, capsys):
    from gridcast.cli import main

    assert main(["--root", str(tmp_path), "compare"]) == 0
    assert "no outcomes stored" in capsys.readouterr().out
