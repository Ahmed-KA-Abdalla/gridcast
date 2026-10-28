from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gridcast.correction import (
    BandDamping,
    Damping,
    apply_damping,
    evaluate_rolling,
    fit_damping,
    revision_frame,
    rolling_blocks,
    rolling_evaluation,
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
    fitted = damping.for_band("(0, 3]")
    # The overshoot is the same in both directions here, so both coefficients
    # should find it.
    assert fitted.up == pytest.approx(0.4, abs=0.05)
    assert fitted.down == pytest.approx(0.4, abs=0.05)


def test_a_forecast_that_under_reacts_gives_negative_damping():
    # The opposite sign: the revision did not go far enough, so undoing part of
    # it would make matters worse and the fitted coefficient says so.
    rng = np.random.default_rng(1)
    revisions = rng.normal(0, 10, 200)
    errors = 0.3 * revisions + rng.normal(0, 1, 200)

    damping = fit_damping(frame_from(list(revisions), list(errors)))
    assert damping.for_band("(0, 3]").up < 0
    assert damping.for_band("(0, 3]").down < 0


def test_an_efficient_forecast_gives_damping_near_zero():
    rng = np.random.default_rng(2)
    revisions = rng.normal(0, 10, 400)
    errors = rng.normal(0, 10, 400)

    damping = fit_damping(frame_from(list(revisions), list(errors)))
    assert abs(damping.for_band("(0, 3]").up) < 0.15
    assert abs(damping.for_band("(0, 3]").down) < 0.15


def test_a_band_with_too_few_observations_is_left_alone():
    # Passing the published forecast through unchanged is the safe default; a
    # coefficient fitted on a handful of rows would be noise.
    damping = fit_damping(frame_from([1.0, -1.0, 2.0], [1.0, -1.0, 2.0]))
    assert damping.for_band("(0, 3]").is_zero
    assert damping.counts["(0, 3]"] == 3


def test_a_band_whose_revisions_never_vary_is_left_alone():
    # Through the origin a constant revision still yields a number, but it is
    # only the mean error rescaled: a bias correction labelled as damping.
    damping = fit_damping(frame_from([5.0] * 50, list(np.arange(50.0))))
    assert damping.for_band("(0, 3]").is_zero


def test_each_band_is_fitted_separately():
    rng = np.random.default_rng(3)
    short = frame_from(list(rng.normal(0, 10, 200)), [0.0] * 200, band="(0, 3]")
    short["remaining_error"] = -0.5 * short["revision"]
    long = frame_from(list(rng.normal(0, 10, 200)), [0.0] * 200, band="(24, 48]")
    long["remaining_error"] = 0.0

    damping = fit_damping(pd.concat([short, long], ignore_index=True))
    assert damping.for_band("(0, 3]").up == pytest.approx(0.5, abs=0.05)
    assert damping.for_band("(0, 3]").down == pytest.approx(0.5, abs=0.05)
    assert damping.for_band("(24, 48]").up == pytest.approx(0.0, abs=0.05)
    assert damping.for_band("(24, 48]").down == pytest.approx(0.0, abs=0.05)


def test_upward_and_downward_revisions_are_damped_separately():
    # The shape the real record shows: a downward revision mostly retraced, an
    # upward one mostly believed.
    rng = np.random.default_rng(4)
    revisions = rng.normal(0, 10, 600)
    errors = np.where(revisions > 0, -0.1, -0.8) * revisions + rng.normal(0, 1, 600)

    frame = frame_from(list(revisions), list(errors))
    asymmetric = fit_damping(frame, "asymmetric").for_band("(0, 3]")
    symmetric = fit_damping(frame, "symmetric").for_band("(0, 3]")

    assert asymmetric.up == pytest.approx(0.1, abs=0.05)
    assert asymmetric.down == pytest.approx(0.8, abs=0.05)
    # One coefficient splits the difference and is wrong for both directions.
    assert symmetric.up == symmetric.down
    assert 0.2 < symmetric.up < 0.7


def test_the_coefficient_is_the_least_squares_fit_of_the_model_applied():
    # The fault this replaced. The correction is forecast - damping * revision,
    # with no intercept, so the coefficient must minimise that model's squared
    # error. The centred slope, -cov/var, fits a model with an intercept, and
    # within one direction of revision, where every value has the same sign,
    # the two disagree badly.
    rng = np.random.default_rng(5)
    revisions = rng.uniform(1, 20, 300)
    errors = 3.0 - 0.2 * revisions + rng.normal(0, 1, 300)
    frame = frame_from(list(revisions), list(errors))

    fitted = fit_damping(frame, "asymmetric").for_band("(0, 3]").up

    grid = np.linspace(-1.0, 1.0, 20001)
    squared = [((errors + c * revisions) ** 2).sum() for c in grid]
    assert fitted == pytest.approx(grid[int(np.argmin(squared))], abs=1e-3)

    centred = -np.cov(errors, revisions, bias=True)[0, 1] / np.var(revisions)
    assert abs(centred - fitted) > 0.1


def test_the_symmetric_model_gives_equal_coefficients():
    rng = np.random.default_rng(6)
    revisions = rng.normal(0, 10, 200)
    errors = -0.4 * revisions + rng.normal(0, 1, 200)

    fitted = fit_damping(frame_from(list(revisions), list(errors)), "symmetric")
    assert fitted.for_band("(0, 3]").up == fitted.for_band("(0, 3]").down
    assert fitted.model == "symmetric"


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="unknown model"):
        fit_damping(frame_from([1.0], [1.0]), "quadratic")


def test_apply_damping_subtracts_the_fitted_share_of_the_revision():
    frame = frame_from([10.0], [0.0])
    corrected = apply_damping(frame, Damping({"(0, 3]": 0.5}, {"(0, 3]": 100}))

    assert corrected["corrected"].iloc[0] == pytest.approx(95.0)
    assert corrected["published_abs_error"].iloc[0] == pytest.approx(0.0)
    assert corrected["corrected_abs_error"].iloc[0] == pytest.approx(5.0)


def test_the_coefficient_applied_follows_the_direction_of_the_revision():
    frame = frame_from([10.0, -10.0], [0.0, 0.0])
    corrected = apply_damping(frame, Damping({"(0, 3]": BandDamping(up=0.5, down=0.2)}, {}))

    # Up by 10, half undone; down by 10, a fifth undone.
    assert corrected["corrected"].tolist() == pytest.approx([95.0, 102.0])


def test_a_bare_number_reads_as_a_symmetric_coefficient():
    # How every record written before the asymmetric model stores a band.
    assert Damping({"(0, 3]": 0.46}, {}).for_band("(0, 3]") == BandDamping(0.46, 0.46)
    assert BandDamping.coerce({"up": 0.1, "down": 0.8}) == BandDamping(0.1, 0.8)


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


def dated_frame(rows_per_day: list[int]) -> pd.DataFrame:
    """A revision frame with the given number of rows on consecutive days."""
    dates = []
    for offset, count in enumerate(rows_per_day):
        dates.extend([dt.date(2026, 8, 1) + dt.timedelta(days=offset)] * count)
    return frame_from([1.0] * len(dates), [1.0] * len(dates), dates=dates)


def test_rolling_blocks_are_consecutive_and_never_start_on_the_first_date():
    frame = dated_frame([10] * 20)
    blocks = rolling_blocks(frame, folds=5, first_fit_fraction=0.4)

    assert len(blocks) == 5
    assert blocks[0][0] > min(frame["date"])
    for (_, last), (first, _) in zip(blocks[:-1], blocks[1:], strict=True):
        assert first == last + dt.timedelta(days=1)
    assert blocks[-1][1] == max(frame["date"])


def test_rolling_blocks_balance_rows_when_days_are_uneven():
    # Dense early days and sparse late ones, the shape of this record after
    # capture collapsed. Blocks placed by date position would leave the first
    # fit with almost every row and the blocks with almost none.
    frame = dated_frame([100] * 4 + [5] * 40)
    first, _ = rolling_blocks(frame, folds=5, first_fit_fraction=0.4)[0]

    share = (frame["date"] < first).mean()
    assert 0.3 < share < 0.5


def test_a_short_record_yields_fewer_blocks_rather_than_empty_ones():
    blocks = rolling_blocks(dated_frame([10, 10, 10]), folds=5)
    assert 1 <= len(blocks) < 5
    assert all(first <= last for first, last in blocks)


def test_a_record_of_one_date_cannot_be_held_out():
    assert rolling_blocks(dated_frame([10])) == []
    assert rolling_blocks(pd.DataFrame()) == []


def test_no_block_is_fitted_on_its_own_dates_or_its_own_periods(overshooting_store, monkeypatch):
    # The two leaks a rolling evaluation can have. A fit that sees the block's
    # dates is in sample. A fit that sees an earlier revision of a period the
    # block scores has seen that period's outcome, which a date boundary alone
    # does not prevent: a period's first revisions can fall the day before.
    import gridcast.correction as correction

    frame = revision_frame(overshooting_store)
    calls = []
    original = correction.fit_damping

    def spy(fit_frame, model=correction.DEFAULT_MODEL):
        calls.append(fit_frame)
        return original(fit_frame, model)

    monkeypatch.setattr(correction, "fit_damping", spy)
    blocks = rolling_blocks(frame)
    correction.rolling_evaluation(frame, resamples=50)

    # One fit per block, then one on the whole record for promotion.
    assert len(calls) == len(blocks) + 1
    for fitted, (first, last) in zip(calls[:-1], blocks, strict=True):
        held = frame[(frame["date"] >= first) & (frame["date"] <= last)]
        assert fitted["date"].max() < first
        assert not set(fitted["period_start"]) & set(held["period_start"])
    assert len(calls[-1]) == len(frame)


def test_an_overshooting_forecast_is_improved_in_every_block(overshooting_store):
    summary, damping, per_fold, note = evaluate_rolling(overshooting_store, resamples=300)
    scored = summary[summary["n"] >= 30]

    assert not scored.empty
    assert note["folds"] >= 3
    assert (scored["improvement"] > 0).all()
    assert (scored["folds_positive"] == scored["folds"]).all()
    assert scored["significant"].all()
    # The fixture overshoots by the same amount either way, so both directions
    # find about a half and the fits barely move between blocks.
    assert np.allclose(scored["damping_up"], 0.5, atol=0.1)
    assert np.allclose(scored["damping_down"], 0.5, atol=0.1)
    assert (scored[["up_range", "down_range"]] < 0.25).all().all()


def test_the_promoted_coefficients_are_fitted_on_the_whole_record(overshooting_store):
    frame = revision_frame(overshooting_store)
    _, damping, _, _ = rolling_evaluation(frame, resamples=50)
    assert damping == fit_damping(frame)


def test_everything_is_empty_on_an_empty_store(tmp_path):
    summary, damping, per_fold, note = evaluate_rolling(tmp_path)
    assert summary.empty
    assert per_fold.empty
    assert damping.coefficients == {}
    assert "reason" in note


def test_correct_command_reports_the_split_and_the_result(overshooting_store, capsys):
    from gridcast.cli import main

    assert main(["--root", str(overshooting_store), "correct"]) == 0
    out = capsys.readouterr().out
    assert "fitted on" in out
    assert "out-of-sample error" in out
    # Both models are scored the same way, so the comparison is between models.
    assert "symmetric" in out
    assert "asymmetric" in out
    assert "held-out block" in out


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


# -- the corrected record -------------------------------------------------


def test_corrected_record_keeps_the_shape_of_the_published_one(overshooting_store):
    from gridcast.correction import Damping, corrected_forecast_record
    from gridcast.load import forecast_record

    published = forecast_record(overshooting_store)
    corrected = corrected_forecast_record(
        overshooting_store, Damping({"(0, 3]": 0.5}, {"(0, 3]": 100})
    )

    assert {"period_start", "captured_at", "forecast", "horizon_hours"} <= set(corrected.columns)
    assert len(corrected) == len(published)


def test_the_correction_moves_the_forecast_against_its_latest_revision():
    # Built directly rather than through a store, to isolate the arithmetic.
    from gridcast.correction import (
        LEAD_BINS,  # noqa: F401
        Damping,
    )

    damping = Damping({"(0, 3]": 0.5}, {"(0, 3]": 100})
    frame = pd.DataFrame(
        {
            "period_start": pd.to_datetime(["2026-08-25T12:00Z"] * 2, utc=True),
            "captured_at": pd.to_datetime(["2026-08-25T10:00Z", "2026-08-25T11:00Z"], utc=True),
            "forecast": [100.0, 120.0],
            "horizon_hours": [2.0, 1.0],
        }
    )
    frame["revision"] = [np.nan, 20.0]
    frame["previous_revision"] = [np.nan, np.nan]
    frame["issue_index"] = [0, 1]

    band = pd.cut(frame["horizon_hours"], bins=list(LEAD_BINS), right=True).astype(str)
    revision = frame["revision"].fillna(0.0)
    corrected = frame["forecast"] - damping.factors(band, revision) * revision

    # First capture revised nothing, so it passes through; the second revised
    # upwards by 20 and so is pulled back by half of it.
    assert corrected.tolist() == [100.0, 110.0]


def test_a_band_without_a_promoted_coefficient_is_left_alone(overshooting_store):
    from gridcast.correction import Damping, corrected_forecast_record
    from gridcast.load import forecast_record

    published = forecast_record(overshooting_store)
    untouched = corrected_forecast_record(overshooting_store, Damping({}, {}))

    merged = published.merge(
        untouched, on=["period_start", "captured_at"], suffixes=("_published", "_corrected")
    )
    assert np.allclose(merged["forecast_published"], merged["forecast_corrected"])
