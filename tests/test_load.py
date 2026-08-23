from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from gridcast.load import (
    coverage,
    evaluation_frame,
    forecast_record,
    generation_record,
    kind_family,
    outcome_record,
)
from gridcast.storage import write_snapshot

UTC = dt.UTC
BASE = dt.datetime(2026, 8, 20, 12, 0, tzinfo=UTC)


def period(offset_periods: int) -> str:
    moment = BASE + dt.timedelta(minutes=30 * offset_periods)
    return moment.strftime("%Y-%m-%dT%H:%MZ")


def intensity_payload(entries: list[tuple[int, float | None, float | None]]) -> dict:
    return {
        "data": [
            {
                "from": period(offset),
                "to": period(offset + 1),
                "intensity": {"forecast": forecast, "actual": actual, "index": "moderate"},
            }
            for offset, forecast, actual in entries
        ]
    }


def generation_payload(offsets: list[int], gas: float) -> dict:
    return {
        "data": [
            {
                "from": period(offset),
                "to": period(offset + 1),
                "generationmix": [
                    {"fuel": "gas", "perc": gas},
                    {"fuel": "wind", "perc": 100.0 - gas},
                ],
            }
            for offset in offsets
        ]
    }


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("intensity_range_20240101", "intensity_range"),
        ("generation_range_20261231", "generation_range"),
        ("forecast_fw48h", "forecast_fw48h"),
        ("intensity", "intensity"),
    ],
)
def test_kind_family_strips_the_backfill_window(kind, expected):
    assert kind_family(kind) == expected


@pytest.fixture
def store(tmp_path):
    """A store holding one forward forecast, one settled period and a backfill."""
    # Issued at 12:00, covering the period in progress plus the next two.
    write_snapshot(
        intensity_payload([(0, 200.0, 198.0), (1, 210.0, None), (2, 220.0, None)]),
        "forecast_fw48h",
        "/intensity/2026-08-20T12:00Z/fw48h",
        BASE,
        root=tmp_path,
    )
    # Reissued half an hour later with revised numbers for the same periods.
    write_snapshot(
        intensity_payload([(1, 205.0, 203.0), (2, 215.0, None)]),
        "forecast_fw48h",
        "/intensity/2026-08-20T12:30Z/fw48h",
        BASE + dt.timedelta(minutes=30),
        root=tmp_path,
    )
    # The settled outcomes, observed later still.
    write_snapshot(
        intensity_payload([(1, 205.0, 190.0), (2, 215.0, 230.0)]),
        "intensity_range_20260820",
        "/intensity/2026-08-20T12:30Z/2026-08-20T13:30Z",
        BASE + dt.timedelta(hours=3),
        root=tmp_path,
    )
    write_snapshot(
        generation_payload([0, 1, 2], gas=55.0),
        "generation_range_20260820",
        "/generation/2026-08-20T12:00Z/2026-08-20T13:30Z",
        BASE + dt.timedelta(hours=3),
        root=tmp_path,
    )
    return tmp_path


def test_forecast_record_keeps_every_issue_of_a_period(store):
    frame = forecast_record(store)
    period = frame[frame["period_start"] == pd.Timestamp("2026-08-20T13:00Z")]

    # Forecast at one hour's lead in the 12:00 issue and at half an hour's lead
    # in the 12:30 issue, with the number revised between the two.
    assert len(period) == 2
    assert sorted(period["forecast"].tolist()) == [215.0, 220.0]
    assert sorted(period["horizon_hours"].tolist()) == [0.5, 1.0]


def test_forecast_record_keeps_a_single_issue_where_only_one_exists(store):
    frame = forecast_record(store)
    period = frame[frame["period_start"] == pd.Timestamp("2026-08-20T12:30Z")]

    # The 12:30 issue reports this period at zero lead, which is dropped, so
    # only the 12:00 issue survives.
    assert len(period) == 1
    assert period["forecast"].iloc[0] == 210.0


def test_forecast_record_drops_the_period_already_in_progress(store):
    frame = forecast_record(store)
    # The 12:00 period appears in the 12:00 issue at zero lead time; it is an
    # observation of the present, not a forecast.
    assert (frame["horizon_hours"] > 0).all()
    assert pd.Timestamp("2026-08-20T12:00Z") not in set(frame["period_start"])


def test_forecast_record_excludes_backfilled_revised_forecasts(store):
    frame = forecast_record(store)
    # The backfill snapshot carries a forecast column of revised values. Were it
    # included, the 13:00 period would show a forecast captured three hours
    # after the fact, at a negative lead time.
    assert frame["captured_at"].max() == pd.Timestamp("2026-08-20T12:30Z")


def test_forecast_record_carries_no_outcome_column(store):
    # The outcome must arrive through the join, not ride along with the
    # predictor.
    assert "actual" not in forecast_record(store).columns


def test_outcome_record_takes_the_latest_observation_of_a_period(store):
    frame = outcome_record(store)
    settled = frame.set_index("period_start")["actual"]

    # 12:30 was reported as 203 at 12:30 and revised to 190 three hours later.
    assert settled[pd.Timestamp("2026-08-20T12:30Z")] == 190.0
    assert settled[pd.Timestamp("2026-08-20T13:00Z")] == 230.0


def test_outcome_record_has_one_row_per_period(store):
    frame = outcome_record(store)
    assert not frame["period_start"].duplicated().any()


def test_evaluation_frame_scores_each_issue_against_the_outcome(store):
    frame = evaluation_frame(store)
    row = frame[
        (frame["period_start"] == pd.Timestamp("2026-08-20T12:30Z"))
        & (frame["captured_at"] == pd.Timestamp("2026-08-20T12:00Z"))
    ].iloc[0]

    assert row["forecast"] == 210.0
    assert row["actual"] == 190.0
    # Signed forecast minus outcome: positive means the forecast was too high.
    assert row["error"] == pytest.approx(20.0)
    assert row["abs_error"] == pytest.approx(20.0)
    assert row["horizon_hours"] == pytest.approx(0.5)


def test_evaluation_frame_reports_a_forecast_that_was_too_low_as_negative(store):
    frame = evaluation_frame(store)
    row = frame[
        (frame["period_start"] == pd.Timestamp("2026-08-20T13:00Z"))
        & (frame["captured_at"] == pd.Timestamp("2026-08-20T12:00Z"))
    ].iloc[0]
    assert row["error"] == pytest.approx(220.0 - 230.0)


def test_evaluation_frame_drops_forecasts_whose_period_has_not_settled(tmp_path):
    write_snapshot(
        intensity_payload([(1, 210.0, None), (2, 220.0, None)]),
        "forecast_fw48h",
        "/intensity/2026-08-20T12:00Z/fw48h",
        BASE,
        root=tmp_path,
    )
    assert evaluation_frame(tmp_path).empty


def test_generation_record_deduplicates_on_the_latest_observation(store):
    frame = generation_record(store)
    assert len(frame) == 3
    assert frame["gas"].iloc[0] == pytest.approx(55.0)


def test_records_are_empty_on_an_empty_store(tmp_path):
    assert forecast_record(tmp_path).empty
    assert outcome_record(tmp_path).empty
    assert generation_record(tmp_path).empty
    assert evaluation_frame(tmp_path).empty


def test_coverage_counts_issues_and_periods(store):
    summary = coverage(store)
    assert summary["outcome_periods"] == 2
    assert summary["forecast_periods"] == 2
    assert summary["issues"] == 2
    assert summary["outcome_missing"] == 0


def test_coverage_reports_a_gap_in_the_outcome_record(tmp_path):
    # Two periods an hour apart, with the one between them never observed.
    write_snapshot(
        intensity_payload([(0, 200.0, 198.0), (2, 220.0, 225.0)]),
        "intensity_range_20260820",
        "/intensity/2026-08-20T12:00Z/2026-08-20T13:30Z",
        BASE + dt.timedelta(hours=3),
        root=tmp_path,
    )
    assert coverage(tmp_path)["outcome_missing"] == 1


def test_report_command_runs_on_a_populated_store(store, capsys):
    from gridcast.cli import main

    assert main(["--root", str(store), "report"]) == 0
    out = capsys.readouterr().out
    assert "settled periods:" in out
    assert "error by lead time" in out


def test_report_command_says_so_when_nothing_is_scoreable(tmp_path, capsys):
    from gridcast.cli import main

    assert main(["--root", str(tmp_path), "report"]) == 0
    assert "nothing scoreable yet" in capsys.readouterr().out


def test_outcomes_from_the_past_24h_capture_are_used(tmp_path):
    # A run reports the past 24 hours, so a period is observed by many
    # successive runs and survives dropped schedules.
    write_snapshot(
        intensity_payload([(0, 200.0, 198.0), (1, 210.0, 205.0)]),
        "outcomes_pt24h",
        "/intensity/2026-08-20T14:00Z/pt24h",
        BASE + dt.timedelta(hours=2),
        root=tmp_path,
    )
    frame = outcome_record(tmp_path)
    assert len(frame) == 2
    assert frame["actual"].tolist() == [198.0, 205.0]


def test_a_later_pt24h_capture_supersedes_an_earlier_one(tmp_path):
    write_snapshot(
        intensity_payload([(0, 200.0, 198.0)]),
        "outcomes_pt24h",
        "/intensity/2026-08-20T13:00Z/pt24h",
        BASE + dt.timedelta(hours=1),
        root=tmp_path,
    )
    write_snapshot(
        intensity_payload([(0, 200.0, 191.0)]),
        "outcomes_pt24h",
        "/intensity/2026-08-20T18:00Z/pt24h",
        BASE + dt.timedelta(hours=6),
        root=tmp_path,
    )
    assert outcome_record(tmp_path)["actual"].tolist() == [191.0]


def test_generation_from_the_past_24h_capture_is_used(tmp_path):
    # The mix needs the same redundancy as intensity: /generation reports only
    # the period in progress, so a run captures one mix period while the
    # schedule advances by two or three.
    from gridcast.load import generation_record

    payload = {
        "data": [
            {
                "from": period(offset),
                "to": period(offset + 1),
                "generationmix": [
                    {"fuel": "gas", "perc": 55.0},
                    {"fuel": "wind", "perc": 45.0},
                ],
            }
            for offset in range(3)
        ]
    }
    write_snapshot(
        payload,
        "generation_pt24h",
        "/generation/2026-08-19T14:00Z/2026-08-20T14:00Z",
        BASE + dt.timedelta(hours=2),
        root=tmp_path,
    )
    frame = generation_record(tmp_path)
    assert len(frame) == 3
    assert frame["gas"].tolist() == [55.0, 55.0, 55.0]
