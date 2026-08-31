from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from gridcast.decisions import (
    decision_windows,
    split_decisions,
    to_forecast_frame,
    window_relative,
)
from gridcast.scheduling import Load

UTC = dt.UTC


def outcomes(start: str, count: int, values=None) -> pd.DataFrame:
    index = pd.date_range(start, periods=count, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "period_start": index,
            "actual": values if values is not None else np.arange(count, dtype=float),
        }
    )


def test_every_decision_covers_a_complete_window():
    frame = decision_windows(
        outcomes("2026-06-01T00:00Z", 48 * 10), Load(periods=2, window_hours=6.0)
    )
    sizes = frame.groupby("decision_id").size()
    assert (sizes == 12).all()


def test_position_indexes_the_period_within_its_window():
    frame = decision_windows(
        outcomes("2026-06-01T00:00Z", 48 * 10), Load(periods=2, window_hours=6.0)
    )
    first = frame[frame["decision_id"] == frame["decision_id"].min()]
    assert first["position"].tolist() == list(range(12))


def test_a_window_with_a_gap_is_dropped_rather_than_filled():
    # Interpolating would invent the quantity the scheduler is scored against.
    settled = outcomes("2026-06-01T00:00Z", 48 * 10)
    gapped = settled.drop(index=100)

    complete = decision_windows(settled, Load(periods=2, window_hours=6.0))
    reduced = decision_windows(gapped, Load(periods=2, window_hours=6.0))
    assert reduced["decision_id"].nunique() < complete["decision_id"].nunique()


def test_decisions_are_generated_at_each_requested_hour():
    frame = decision_windows(
        outcomes("2026-06-01T00:00Z", 48 * 10),
        Load(periods=2, window_hours=6.0),
        issue_hours=(2, 14),
    )
    hours = sorted(pd.DatetimeIndex(frame["decision_id"].unique()).hour.unique())
    assert hours == [2, 14]


def test_the_record_yields_many_more_decisions_than_the_captured_sample():
    # The point of the module: two and a half years at four issue times a day
    # is thousands of decisions, against under two hundred captured ones.
    frame = decision_windows(
        outcomes("2024-01-01T00:00Z", 48 * 400), Load(periods=4, window_hours=24.0)
    )
    assert frame["decision_id"].nunique() > 1500


def test_window_relative_subtracts_each_window_s_own_mean():
    frame = pd.DataFrame(
        {
            "decision_id": ["a", "a", "b", "b"],
            "value": [10.0, 20.0, 110.0, 120.0],
        }
    )
    relative = window_relative(frame, ["value"])
    # Two windows a hundred apart in level become identical in relative terms.
    assert relative["value_rel"].tolist() == [-5.0, 5.0, -5.0, 5.0]


def test_window_relative_ignores_a_column_that_is_absent():
    frame = pd.DataFrame({"decision_id": ["a"], "value": [1.0]})
    assert "missing_rel" not in window_relative(frame, ["missing"]).columns


def test_to_forecast_frame_matches_what_the_scorer_expects():
    from gridcast.scheduling import evaluate_decisions

    settled = outcomes("2026-06-01T00:00Z", 48 * 10, values=None)
    load = Load(periods=1, window_hours=6.0)
    windows = decision_windows(settled, load, issue_hours=(2,))

    # A perfect forecast must produce no regret, which also checks the frame is
    # accepted by the scorer unchanged.
    forecasts = to_forecast_frame(windows, windows["actual"])
    scored = evaluate_decisions(forecasts, settled, load)

    assert not scored.empty
    assert np.allclose(scored["chosen"], scored["oracle"])


def test_split_is_by_date_and_no_decision_straddles_it():
    frame = decision_windows(
        outcomes("2026-06-01T00:00Z", 48 * 20), Load(periods=2, window_hours=6.0)
    )
    frame["date"] = frame["captured_at"].dt.date

    train, test = split_decisions(frame, train_fraction=0.6)
    assert max(train["date"]) < min(test["date"])
    assert set(train["decision_id"]) & set(test["decision_id"]) == set()


def test_split_balances_decisions_between_the_halves():
    frame = decision_windows(
        outcomes("2026-06-01T00:00Z", 48 * 20), Load(periods=2, window_hours=6.0)
    )
    frame["date"] = frame["captured_at"].dt.date

    train, test = split_decisions(frame, train_fraction=0.6)
    share = train["decision_id"].nunique() / frame["decision_id"].nunique()
    assert 0.5 < share < 0.7


def test_everything_is_empty_without_outcomes():
    empty = pd.DataFrame(columns=["period_start", "actual"])
    assert decision_windows(empty, Load()).empty
    assert split_decisions(pd.DataFrame())[0].empty


def test_the_two_paths_generate_the_same_decisions():
    # The harness and the older scheduling path derive issue times separately.
    # They must agree, or a model fitted through one and compared against the
    # other would be measuring the difference between them.
    from gridcast.scheduling import issue_times

    settled = outcomes("2026-06-01T00:00Z", 48 * 30, values=None)
    load = Load(periods=2, window_hours=6.0)

    harness = set(decision_windows(settled, load, issue_hours=(18,))["decision_id"].unique())
    existing = set(issue_times(settled, load, hour=18))

    # The harness keeps only complete windows, so it may hold fewer; it must
    # never hold one the other path would not have generated.
    assert harness <= existing


def test_the_harness_agrees_with_the_existing_scoring_path():
    # The check the harness exists to pass. Scoring the same predictions through
    # both paths must give the same answer, or a model fitted on one and
    # compared against the other would be measuring the difference between them.
    from gridcast.decisions import score_through_harness
    from gridcast.scheduling import baseline_forecasts, evaluate_decisions, summarise

    settled = outcomes("2026-06-01T00:00Z", 48 * 30, values=None)
    load = Load(periods=2, window_hours=6.0)
    windows = decision_windows(settled, load, issue_hours=(18,))

    flat = pd.Series(np.arange(len(windows), dtype=float) % 7)
    through_harness = score_through_harness(windows, settled, load, flat)

    forecasts = to_forecast_frame(windows, flat)
    direct = summarise(evaluate_decisions(forecasts, settled, load))

    assert through_harness == direct
    del baseline_forecasts


def test_decision_dataset_carries_features_and_the_relative_target(tmp_path):
    from gridcast.decisions import decision_dataset
    from gridcast.storage import write_snapshot

    entries = []
    start = dt.datetime(2026, 6, 1, tzinfo=UTC)
    for step in range(48 * 40):
        moment = start + dt.timedelta(minutes=30 * step)
        value = 200.0 + 50.0 * np.sin(2 * np.pi * (step % 48) / 48)
        entries.append(
            {
                "from": moment.strftime("%Y-%m-%dT%H:%MZ"),
                "to": (moment + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                "intensity": {"forecast": value, "actual": value, "index": "moderate"},
            }
        )
    write_snapshot(
        {"data": entries},
        "intensity_range_20260601",
        "/intensity/range",
        start + dt.timedelta(days=41),
        root=tmp_path,
    )

    frame = decision_dataset(tmp_path, Load(periods=2, window_hours=6.0), issue_hours=(2, 14))
    assert not frame.empty
    assert "actual_rel" in frame.columns
    assert "intensity_lag_7d_rel" in frame.columns
    assert "horizon_hours" in frame.columns

    # The relative target sums to zero within every window by construction.
    sums = frame.groupby("decision_id")["actual_rel"].sum()
    assert np.allclose(sums.to_numpy(), 0.0, atol=1e-9)


def test_decisions_command_reports_the_dataset_and_the_baselines(tmp_path, capsys):
    from gridcast.cli import main
    from gridcast.storage import write_snapshot

    entries = []
    start = dt.datetime(2026, 6, 1, tzinfo=UTC)
    for step in range(48 * 40):
        moment = start + dt.timedelta(minutes=30 * step)
        value = 200.0 + 50.0 * np.sin(2 * np.pi * (step % 48) / 48)
        entries.append(
            {
                "from": moment.strftime("%Y-%m-%dT%H:%MZ"),
                "to": (moment + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                "intensity": {"forecast": value, "actual": value, "index": "moderate"},
            }
        )
    write_snapshot(
        {"data": entries},
        "intensity_range_20260601",
        "/intensity/range",
        start + dt.timedelta(days=41),
        root=tmp_path,
    )

    exit_code = main(
        [
            "--root",
            str(tmp_path),
            "decisions",
            "--periods",
            "2",
            "--window",
            "6",
            "--issue-hours",
            "18",
        ]
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "decisions," in out
    assert "seasonal_mean" in out
    assert "must match what 'gridcast schedule' reports" in out


def test_decisions_command_says_so_on_an_empty_store(tmp_path, capsys):
    from gridcast.cli import main

    assert main(["--root", str(tmp_path), "decisions"]) == 0
    assert "no complete windows" in capsys.readouterr().out


def test_a_decision_the_predictor_cannot_cover_is_dropped_not_filled():
    # A seasonal baseline has no reference for the first weeks of a record.
    # Filling those predictions with a number would tell the scheduler those
    # periods were the cheapest available and have it schedule into them.
    from gridcast.decisions import score_through_harness

    settled = outcomes("2026-06-01T00:00Z", 48 * 30, values=None)
    load = Load(periods=2, window_hours=6.0)
    windows = decision_windows(settled, load, issue_hours=(18,))

    partial = pd.Series(np.arange(len(windows), dtype=float) % 7)
    first_decision = windows["decision_id"] == windows["decision_id"].min()
    partial[first_decision.to_numpy()] = np.nan

    scored = score_through_harness(windows, settled, load, partial)
    complete = score_through_harness(
        windows, settled, load, pd.Series(np.arange(len(windows), dtype=float) % 7)
    )
    assert scored["n"] == complete["n"] - 1
