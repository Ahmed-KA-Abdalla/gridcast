from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from gridcast.drift import (
    PSI_LARGE,
    PSI_MODEST,
    coefficient_drift,
    compare_distributions,
    describe,
    error_drift,
    label_psi,
    population_stability_index,
    seasonal_drift,
)

UTC = dt.UTC


def test_an_unmoved_distribution_has_an_index_of_zero():
    rng = np.random.default_rng(0)
    sample = rng.normal(100, 20, 5000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_a_shifted_distribution_registers():
    rng = np.random.default_rng(1)
    reference = rng.normal(100, 20, 5000)
    shifted = rng.normal(140, 20, 5000)

    assert population_stability_index(reference, shifted) > PSI_LARGE


def test_a_small_shift_reads_as_modest_rather_than_large():
    rng = np.random.default_rng(2)
    reference = rng.normal(100, 20, 20000)
    nudged = rng.normal(105, 20, 20000)

    index = population_stability_index(reference, nudged)
    assert PSI_MODEST / 4 < index < PSI_LARGE


def test_a_distribution_that_never_reaches_a_bin_is_counted_not_dropped():
    # The logarithm is undefined for an empty bin, and a bin the current sample
    # never reaches is exactly the case worth counting.
    rng = np.random.default_rng(3)
    reference = rng.normal(100, 20, 5000)
    truncated = reference[reference < 100]

    index = population_stability_index(reference, truncated)
    assert np.isfinite(index)
    assert index > PSI_LARGE


def test_the_index_is_not_computable_on_a_tiny_sample():
    assert np.isnan(population_stability_index(np.arange(3.0), np.arange(3.0)))


def test_the_index_is_not_computable_when_the_reference_is_constant():
    constant = np.full(500, 42.0)
    assert np.isnan(population_stability_index(constant, np.arange(500.0)))


def test_missing_values_are_excluded_rather_than_binned():
    rng = np.random.default_rng(4)
    sample = rng.normal(100, 20, 2000)
    with_gaps = np.concatenate([sample, np.full(500, np.nan)])

    assert population_stability_index(sample, with_gaps) == pytest.approx(0.0, abs=0.02)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.02, "stable"), (0.15, "moderate shift"), (0.4, "large shift"), (np.nan, "not computable")],
)
def test_the_reading_follows_the_conventional_thresholds(value, expected):
    assert label_psi(value) == expected


def test_describe_summarises_a_distribution():
    described = describe(pd.Series(np.arange(101.0)))
    assert described["n"] == 101
    assert described["q50"] == pytest.approx(50.0)
    assert described["q05"] == pytest.approx(5.0)


def test_describe_reports_nothing_for_an_empty_series():
    assert describe(pd.Series(dtype=float))["n"] == 0


def test_comparing_distributions_ranks_the_most_moved_column_first():
    rng = np.random.default_rng(5)
    reference = pd.DataFrame({"steady": rng.normal(0, 1, 4000), "moved": rng.normal(0, 1, 4000)})
    current = pd.DataFrame({"steady": rng.normal(0, 1, 4000), "moved": rng.normal(3, 1, 4000)})

    scored = compare_distributions(reference, current, ["steady", "moved"])
    assert scored.iloc[0]["column"] == "moved"
    assert scored.iloc[0]["reading"] == "large shift"


def test_comparing_distributions_skips_a_column_that_is_absent():
    frame = pd.DataFrame({"present": np.arange(500.0)})
    scored = compare_distributions(frame, frame, ["present", "absent"])
    assert list(scored["column"]) == ["present"]


# -- against a store -------------------------------------------------------


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
def seasonal_store(tmp_path):
    """A year whose winter intensity sits well above its summer."""
    from gridcast.storage import write_snapshot

    rng = np.random.default_rng(7)
    start = dt.datetime(2025, 1, 1, tzinfo=UTC)
    entries = []
    for step in range(48 * 360):
        moment = start + dt.timedelta(minutes=30 * step)
        winter = moment.month in (12, 1, 2)
        level = 200.0 if winter else 110.0
        value = level + rng.normal(0, 15)
        entries.append(
            (
                moment.strftime("%Y-%m-%dT%H:%MZ"),
                (moment + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                value,
                value,
            )
        )
    write_snapshot(
        intensity_payload(entries),
        "intensity_range_20250101",
        "/intensity/range",
        start + dt.timedelta(days=365),
        root=tmp_path,
    )
    return tmp_path


def test_seasonal_drift_detects_a_winter_unlike_the_summer(seasonal_store):
    scored = seasonal_drift(seasonal_store)

    winter = scored[scored["season"] == "winter"].iloc[0]
    summer = scored[scored["season"] == "summer"].iloc[0]

    assert summer["psi_against_summer"] == pytest.approx(0.0, abs=1e-9)
    assert winter["psi_against_summer"] > PSI_LARGE
    assert winter["mean"] > summer["mean"]


def test_seasonal_drift_needs_a_summer_to_compare_against(tmp_path):
    from gridcast.storage import write_snapshot

    start = dt.datetime(2025, 1, 1, tzinfo=UTC)
    entries = [
        (
            (start + dt.timedelta(minutes=30 * step)).strftime("%Y-%m-%dT%H:%MZ"),
            (start + dt.timedelta(minutes=30 * (step + 1))).strftime("%Y-%m-%dT%H:%MZ"),
            200.0,
            200.0,
        )
        for step in range(48 * 20)
    ]
    write_snapshot(
        intensity_payload(entries),
        "intensity_range_20250101",
        "/intensity/range",
        start + dt.timedelta(days=30),
        root=tmp_path,
    )
    assert seasonal_drift(tmp_path).empty


def test_drift_views_are_empty_on_an_empty_store(tmp_path):
    assert seasonal_drift(tmp_path).empty
    assert error_drift(tmp_path).empty


def test_coefficient_drift_reports_the_spread_across_runs(record_path):
    from gridcast.gate import BandVerdict, write_record

    def verdict(damping: float) -> BandVerdict:
        return BandVerdict(
            band="(6, 12]",
            promoted=True,
            damping=damping,
            improvement=1.0,
            improvement_low=0.5,
            n=500,
            periods=150,
        )

    for index, value in enumerate((0.46, 0.44, 0.52)):
        write_record(
            [verdict(value)], path=record_path, generated=f"2026-09-0{index + 1}T00:00:00Z"
        )

    scored = coefficient_drift(record_path)
    row = scored.iloc[0]

    assert row["runs"] == 3
    assert row["first"] == pytest.approx(0.46)
    assert row["latest"] == pytest.approx(0.52)
    assert row["range"] == pytest.approx(0.08)


def test_coefficient_drift_is_empty_without_a_record(record_path):
    assert coefficient_drift(record_path).empty


def test_drift_command_reports_the_seasonal_comparison(seasonal_store, capsys):
    from gridcast.cli import main

    assert main(["--root", str(seasonal_store), "drift"]) == 0
    out = capsys.readouterr().out
    assert "distribution by season" in out
    assert "conventions, not derived quantities" in out


def test_drift_command_says_so_when_the_record_is_too_short(tmp_path, capsys):
    from gridcast.cli import main

    assert main(["--root", str(tmp_path), "drift"]) == 0
    assert "not enough of the record" in capsys.readouterr().out


def test_error_drift_reports_the_forecast_error_by_week_and_band(tmp_path):
    # Performance drift, the only one of the three that is a fault by itself.
    from gridcast.storage import write_snapshot

    start = dt.datetime(2026, 8, 1, tzinfo=UTC)
    rng = np.random.default_rng(9)

    outcomes, forecasts = [], []
    for step in range(48 * 30):
        target = start + dt.timedelta(minutes=30 * step)
        truth = 200.0 + rng.normal(0, 20)
        outcomes.append(
            (
                target.strftime("%Y-%m-%dT%H:%MZ"),
                (target + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                truth,
                truth,
            )
        )

    for day in range(28):
        issue = start + dt.timedelta(days=day)
        entries = []
        for ahead in range(1, 13):
            target = issue + dt.timedelta(minutes=30 * ahead)
            entries.append(
                (
                    target.strftime("%Y-%m-%dT%H:%MZ"),
                    (target + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                    200.0 + rng.normal(0, 25),
                    None,
                )
            )
        forecasts.append((issue, entries))

    write_snapshot(
        intensity_payload(outcomes),
        "intensity_range_20260801",
        "/intensity/range",
        start + dt.timedelta(days=31),
        root=tmp_path,
    )
    for issue, entries in forecasts:
        write_snapshot(
            intensity_payload(entries),
            "forecast_fw48h",
            "/intensity/issue/fw48h",
            issue,
            root=tmp_path,
        )

    scored = error_drift(tmp_path)
    assert not scored.empty
    assert {"bucket", "band", "n", "mae", "bias"} <= set(scored.columns)
    # Weekly buckets, so a month of captures gives several.
    assert scored["bucket"].nunique() > 1
    # Buckets keep their timezone rather than being converted to periods.
    assert scored["bucket"].dt.tz is not None
