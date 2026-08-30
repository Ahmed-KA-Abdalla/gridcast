from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gridcast.correction import (
    Damping,
    apply_damping,
    evaluate_correction,
    fit_damping,
    revision_frame,
    split_by_date,
)

UTC = dt.UTC


def frame_from(
    revisions: list[float], errors: list[float], band: str = "(0, 3]", dates=None
) -> pd.DataFrame:
    """A revision frame with the columns the fitter needs."""
    count = len(revisions)
    if dates is None:
        dates = [dt.date(2026, 8, 20)] * count
    return pd.DataFrame(
        {
            "revision": revisions,
            "remaining_error": errors,
            "forecast": [100.0] * count,
            "actual": [100.0 + e for e in errors],
            "band": [band] * count,
            "date": dates,
            "horizon_hours": [1.0] * count,
        }
    )


def test_a_forecast_that_overshoots_gives_positive_damping():
    # Revision up, outcome below the revised forecast: part of the revision
    # should be undone.
    rng = np.random.default_rng(0)
    revisions = rng.normal(0, 10, 200)
    errors = -0.4 * revisions + rng.normal(0, 1, 200)

    damping = fit_damping(frame_from(list(revisions), list(errors)))
    assert damping.for_band("(0, 3]") == pytest.approx(0.4, abs=0.05)


def test_a_forecast_that_under_reacts_gives_negative_damping():
    # The opposite sign: the revision did not go far enough, so undoing part of
    # it would make matters worse and the fitted coefficient says so.
    rng = np.random.default_rng(1)
    revisions = rng.normal(0, 10, 200)
    errors = 0.3 * revisions + rng.normal(0, 1, 200)

    damping = fit_damping(frame_from(list(revisions), list(errors)))
    assert damping.for_band("(0, 3]") < 0


def test_an_efficient_forecast_gives_damping_near_zero():
    rng = np.random.default_rng(2)
    revisions = rng.normal(0, 10, 400)
    errors = rng.normal(0, 10, 400)

    damping = fit_damping(frame_from(list(revisions), list(errors)))
    assert abs(damping.for_band("(0, 3]")) < 0.15


def test_a_band_with_too_few_observations_is_left_alone():
    # Passing the published forecast through unchanged is the safe default; a
    # coefficient fitted on a handful of rows would be noise.
    damping = fit_damping(frame_from([1.0, -1.0, 2.0], [1.0, -1.0, 2.0]))
    assert damping.for_band("(0, 3]") == 0.0
    assert damping.counts["(0, 3]"] == 3


def test_a_band_whose_revisions_never_vary_is_left_alone():
    damping = fit_damping(frame_from([5.0] * 50, list(np.arange(50.0))))
    assert damping.for_band("(0, 3]") == 0.0


def test_each_band_is_fitted_separately():
    rng = np.random.default_rng(3)
    short = frame_from(list(rng.normal(0, 10, 200)), [0.0] * 200, band="(0, 3]")
    short["remaining_error"] = -0.5 * short["revision"]
    long = frame_from(list(rng.normal(0, 10, 200)), [0.0] * 200, band="(24, 48]")
    long["remaining_error"] = 0.0

    damping = fit_damping(pd.concat([short, long], ignore_index=True))
    assert damping.for_band("(0, 3]") == pytest.approx(0.5, abs=0.05)
    assert damping.for_band("(24, 48]") == pytest.approx(0.0, abs=0.05)


def test_apply_damping_subtracts_the_fitted_share_of_the_revision():
    frame = frame_from([10.0], [0.0])
    corrected = apply_damping(frame, Damping({"(0, 3]": 0.5}, {"(0, 3]": 100}))

    assert corrected["corrected"].iloc[0] == pytest.approx(95.0)
    assert corrected["published_abs_error"].iloc[0] == pytest.approx(0.0)
    assert corrected["corrected_abs_error"].iloc[0] == pytest.approx(5.0)


def test_a_band_without_a_coefficient_passes_through_unchanged():
    frame = frame_from([10.0], [0.0], band="(6, 12]")
    corrected = apply_damping(frame, Damping({"(0, 3]": 0.5}, {"(0, 3]": 100}))

    assert corrected["corrected"].iloc[0] == corrected["forecast"].iloc[0]


def test_the_split_is_by_date_not_by_row():
    # Rows sharing a target period are not independent, so a random split would
    # put a period's early revisions in training and its later ones in test.
    dates = [dt.date(2026, 8, day) for day in (1, 1, 2, 2, 3, 3, 4, 4, 5, 5)]
    frame = frame_from([1.0] * 10, [1.0] * 10, dates=dates)

    train, test = split_by_date(frame, train_fraction=0.6)
    assert set(train["date"]) & set(test["date"]) == set()
    assert max(train["date"]) < min(test["date"])


def test_the_split_balances_rows_when_days_are_uneven():
    # The fault this replaced: a fixed fraction of dates is not a fixed fraction
    # of rows. Here four dense days precede four sparse ones, so taking the
    # first 60% of dates would hand almost everything to training.
    dates = []
    for day in range(1, 5):
        dates.extend([dt.date(2026, 8, day)] * 100)
    for day in range(5, 9):
        dates.extend([dt.date(2026, 8, day)] * 5)

    frame = frame_from([1.0] * len(dates), [1.0] * len(dates), dates=dates)
    train, test = split_by_date(frame, train_fraction=0.6)

    share = len(train) / len(frame)
    assert 0.5 < share < 0.75
    # Still a clean date boundary, so no period straddles the split.
    assert max(train["date"]) < min(test["date"])


def test_the_split_lands_on_the_closest_achievable_boundary():
    # Days are indivisible, so the requested fraction is rarely exactly
    # attainable; the boundary chosen should be the nearest one.
    dates = []
    for day in range(1, 5):
        dates.extend([dt.date(2026, 8, day)] * 10)

    frame = frame_from([1.0] * 40, [1.0] * 40, dates=dates)
    train, _ = split_by_date(frame, train_fraction=0.5)
    assert len(train) == 20


def test_the_split_still_divides_evenly_when_days_are_equal():
    dates = []
    for day in range(1, 11):
        dates.extend([dt.date(2026, 8, day)] * 10)

    frame = frame_from([1.0] * 100, [1.0] * 100, dates=dates)
    train, test = split_by_date(frame, train_fraction=0.6)
    assert len(train) == 60
    assert len(test) == 40


def test_the_split_always_holds_something_out():
    dates = [dt.date(2026, 8, day) for day in (1, 2)]
    frame = frame_from([1.0, 2.0], [1.0, 2.0], dates=dates)

    train, test = split_by_date(frame, train_fraction=0.99)
    assert not train.empty
    assert not test.empty


def test_a_single_date_cannot_be_split():
    frame = frame_from([1.0, 2.0], [1.0, 2.0])
    train, test = split_by_date(frame)
    assert test.empty


# -- end to end -----------------------------------------------------------


def test_revision_frame_carries_the_error_remaining_after_each_revision(overshooting_store):
    frame = revision_frame(overshooting_store)

    assert not frame.empty
    assert {"revision", "remaining_error", "band", "date"} <= set(frame.columns)
    # Signed as outcome minus forecast, so positive means the forecast was low.
    reconstructed = frame["actual"] - frame["forecast"]
    assert np.allclose(frame["remaining_error"], reconstructed)


def test_evaluation_fits_on_earlier_dates_and_scores_on_later_ones(overshooting_store):
    summary, damping, note = evaluate_correction(overshooting_store)

    assert not summary.empty
    assert note["train_dates"] > 0
    assert note["test_dates"] > 0
    # No date appears in both.
    assert note["train_rows"] + note["test_rows"] == len(revision_frame(overshooting_store))


def test_an_overshooting_forecast_is_improved_out_of_sample(overshooting_store):
    summary, _, _ = evaluate_correction(overshooting_store)
    scored = summary[summary["n"] >= 30]

    assert not scored.empty
    assert (scored["improvement"] > 0).all()


def test_everything_is_empty_on_an_empty_store(tmp_path):
    summary, damping, note = evaluate_correction(tmp_path)
    assert summary.empty
    assert damping.coefficients == {}


def test_correct_command_reports_the_split_and_the_result(overshooting_store, capsys):
    from gridcast.cli import main

    assert main(["--root", str(overshooting_store), "correct"]) == 0
    out = capsys.readouterr().out
    assert "fitted on" in out
    assert "out-of-sample error" in out


def test_correct_command_says_so_when_there_is_nothing_to_fit(tmp_path, capsys):
    from gridcast.cli import main

    assert main(["--root", str(tmp_path), "correct"]) == 0
    assert "not enough captured revisions" in capsys.readouterr().out


# -- intervals ------------------------------------------------------------


def scored_frame(published: list[float], corrected: list[float], periods=None) -> pd.DataFrame:
    count = len(published)
    if periods is None:
        # One distinct period per row, so the resampling has as many clusters
        # as observations and behaves like an ordinary paired bootstrap.
        periods = pd.date_range("2026-08-20", periods=count, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "published_abs_error": published,
            "corrected_abs_error": corrected,
            "period_start": pd.to_datetime(periods, utc=True),
        }
    )


def test_a_clear_improvement_has_an_interval_above_zero():
    from gridcast.correction import bootstrap_improvement

    rng = np.random.default_rng(0)
    published = list(rng.normal(20, 3, 200))
    corrected = [value - 5.0 for value in published]

    result = bootstrap_improvement(scored_frame(published, corrected), resamples=500)
    assert result["improvement_low"] > 0
    assert result["worse_fraction"] == 0.0


def test_a_negligible_improvement_has_an_interval_spanning_zero():
    # The case the point estimate hides: a small positive mean that a resample
    # turns negative about as often as not.
    from gridcast.correction import bootstrap_improvement

    rng = np.random.default_rng(1)
    published = list(rng.normal(20, 8, 200))
    corrected = list(rng.normal(20, 8, 200))

    result = bootstrap_improvement(scored_frame(published, corrected), resamples=500)
    assert result["improvement_low"] < 0 < result["improvement_high"]


def test_a_correction_that_hurts_shows_a_negative_interval():
    from gridcast.correction import bootstrap_improvement

    rng = np.random.default_rng(2)
    published = list(rng.normal(20, 3, 200))
    corrected = [value + 5.0 for value in published]

    result = bootstrap_improvement(scored_frame(published, corrected), resamples=500)
    assert result["improvement_high"] < 0
    assert result["worse_fraction"] == 1.0


def test_resampling_is_by_period_not_by_row():
    # Rows sharing a target period move together. Resampling rows would treat
    # forty correlated observations as forty independent ones and give an
    # interval far too narrow.
    from gridcast.correction import bootstrap_improvement

    rng = np.random.default_rng(3)
    periods, published, corrected = [], [], []
    for index in range(10):
        # Each period contributes twenty rows sharing one common offset.
        offset = rng.normal(0, 6)
        for _ in range(20):
            periods.append(pd.Timestamp("2026-08-20T00:00Z") + pd.Timedelta(days=index))
            base = rng.normal(20, 1)
            published.append(base)
            corrected.append(base - offset)

    frame = scored_frame(published, corrected, periods)
    clustered = bootstrap_improvement(frame, resamples=500)

    assert clustered["periods"] == 10
    width = clustered["improvement_high"] - clustered["improvement_low"]
    # Ten clusters of correlated rows cannot support a tight interval.
    assert width > 1.0


def test_an_interval_needs_more_than_one_period():
    from gridcast.correction import bootstrap_improvement

    frame = scored_frame([20.0, 21.0], [15.0, 16.0], [pd.Timestamp("2026-08-20T00:00Z")] * 2)
    assert bootstrap_improvement(frame) == {}


def test_evaluation_with_intervals_marks_which_bands_are_significant(overshooting_store):
    from gridcast.correction import evaluate_with_intervals

    summary, _, note = evaluate_with_intervals(overshooting_store, resamples=300)

    assert not summary.empty
    assert "improvement_low" in summary.columns
    assert "significant" in summary.columns
    assert note["resamples"] == 300
    # The fixture overshoots by construction, so at least one band should hold up.
    assert summary["significant"].any()
