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


@pytest.fixture
def overshooting_store(tmp_path):
    """A store whose forecasts overshoot: each revision partly undone by reality.

    One snapshot per issue time carrying every period it forecasts, which is the
    shape the real capture produces. Writing one period per snapshot would put
    several files at the same capture minute, and the later ones would overwrite
    the earlier.
    """
    from gridcast.storage import write_snapshot

    rng = np.random.default_rng(11)
    start = dt.datetime(2026, 8, 1, tzinfo=UTC)
    periods = 480

    truth = {
        start + dt.timedelta(minutes=30 * index): 200.0 + rng.normal(0, 20)
        for index in range(periods)
    }

    write_snapshot(
        intensity_payload(
            [
                (stamp(target), stamp(target + dt.timedelta(minutes=30)), value, value)
                for target, value in truth.items()
            ]
        ),
        "intensity_range_20260801",
        "/intensity/range",
        start + dt.timedelta(days=20),
        root=tmp_path,
    )

    # Hourly issues, each forecasting the next six hours. The forecast sits at
    # the truth plus twice its latest jolt, so half of every revision is undone.
    for step in range(periods // 2):
        issue = start + dt.timedelta(hours=step)
        entries = []
        for ahead in range(1, 13):
            target = issue + dt.timedelta(minutes=30 * ahead)
            if target not in truth:
                continue
            value = truth[target] + 2.0 * rng.normal(0, 12)
            entries.append((stamp(target), stamp(target + dt.timedelta(minutes=30)), value, None))
        if entries:
            write_snapshot(
                intensity_payload(entries),
                "forecast_fw48h",
                "/intensity/issue/fw48h",
                issue,
                root=tmp_path,
            )

    return tmp_path


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
