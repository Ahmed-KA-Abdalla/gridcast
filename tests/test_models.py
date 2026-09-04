from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.models import (
    DEFAULT_PARAMS,
    accuracy,
    compare_level_model,
    fit_level_model,
    model_features,
    permutation_importance,
    predict,
    restore_level,
    score_predictions,
)
from gridcast.scheduling import Load


def synthetic_dataset(days: int = 120, window_periods: int = 12, seed: int = 0):
    """Decisions whose windows have a learnable shape plus noise.

    Built directly rather than through a store so the tests stay fast and the
    signal is known: intensity follows the position within the window, which a
    model can learn from the position feature alone.
    """
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-01-01T18:00Z")

    for day in range(days):
        issue = start + pd.Timedelta(days=day)
        level = 200.0 + rng.normal(0, 30)
        for position in range(window_periods):
            shape = 40.0 * np.sin(2 * np.pi * position / window_periods)
            rows.append(
                {
                    "decision_id": issue,
                    "captured_at": issue,
                    "period_start": issue + pd.Timedelta(minutes=30 * position),
                    "position": position,
                    "actual": level + shape + rng.normal(0, 3),
                    "horizon_hours": position * 0.5,
                    "sin_day": np.sin(2 * np.pi * position / window_periods),
                    "cos_day": np.cos(2 * np.pi * position / window_periods),
                    "date": issue.date(),
                }
            )

    frame = pd.DataFrame(rows)
    frame["actual_rel"] = frame["actual"] - frame.groupby("decision_id")["actual"].transform("mean")
    frame["sin_day_rel"] = frame["sin_day"] - frame.groupby("decision_id")["sin_day"].transform(
        "mean"
    )

    outcomes = frame[["period_start", "actual"]].drop_duplicates("period_start")
    return frame, outcomes.reset_index(drop=True)


FAST = {**DEFAULT_PARAMS, "max_iter": 40}


def test_model_features_include_absolute_and_relative_columns():
    frame, _ = synthetic_dataset(days=5)
    columns = model_features(frame)

    assert "sin_day" in columns
    assert "sin_day_rel" in columns


def test_the_target_is_never_offered_as_a_feature():
    # actual_rel is the answer expressed relative to the window.
    frame, _ = synthetic_dataset(days=5)
    assert "actual_rel" not in model_features(frame)
    assert "actual" not in model_features(frame)


def test_accuracy_reports_error_over_covered_rows_only():
    prediction = pd.Series([10.0, np.nan, 30.0])
    actual = pd.Series([12.0, 18.0, 30.0])

    mae, rmse = accuracy(prediction, actual)
    assert mae == pytest.approx(1.0)
    assert rmse == pytest.approx(np.sqrt(2.0))


def test_accuracy_is_undefined_with_nothing_to_score():
    mae, rmse = accuracy(pd.Series(dtype=float), pd.Series(dtype=float))
    assert np.isnan(mae) and np.isnan(rmse)


def test_a_level_model_learns_the_window_shape():
    # Measured on the deviation from each window's mean rather than on the
    # level. The synthetic level moves with a standard deviation of 30 and no
    # feature predicts it, so absolute error is floored near that whatever the
    # model learns — and the shape is the part a scheduler consumes.
    frame, outcomes = synthetic_dataset(days=120)
    model, columns = fit_level_model(frame, FAST)
    prediction = predict(model, frame, columns)

    predicted_shape = prediction - prediction.groupby(frame["decision_id"]).transform("mean")
    assert predicted_shape.corr(frame["actual_rel"]) > 0.9


def test_absolute_error_is_dominated_by_a_level_the_model_cannot_see():
    # Worth stating: a large mean absolute error here is not a model that failed
    # to learn. It is a model that learned everything available and still cannot
    # know the level, which is precisely the quantity the decision ignores.
    frame, outcomes = synthetic_dataset(days=120)
    model, columns = fit_level_model(frame, FAST)

    mae, _ = accuracy(predict(model, frame, columns), frame["actual"])
    level_spread = frame.groupby("decision_id")["actual"].mean().std()

    assert mae == pytest.approx(level_spread * np.sqrt(2 / np.pi), rel=0.35)


def test_scoring_reports_accuracy_and_decision_quality_together():
    # The distinction the whole project rests on: one number cannot show a
    # model that wins on error and loses on the choice.
    frame, outcomes = synthetic_dataset(days=60)
    load = Load(periods=2, window_hours=6.0)

    scored = score_predictions("perfect", frame, frame["actual"], outcomes, load)
    assert scored.mae == pytest.approx(0.0)
    assert scored.decision["n"] > 0
    assert set(scored.row()) >= {"mae", "captured_fraction", "hit_rate"}


def test_a_constant_offset_costs_accuracy_and_nothing_else():
    # Stated again here because it is the reason both metrics are reported: a
    # forecast wrong by a constant has large error and schedules perfectly.
    frame, outcomes = synthetic_dataset(days=60)
    load = Load(periods=2, window_hours=6.0)

    exact = score_predictions("exact", frame, frame["actual"], outcomes, load)
    shifted = score_predictions("shifted", frame, frame["actual"] + 40.0, outcomes, load)

    assert shifted.mae == pytest.approx(40.0, abs=1e-6)
    assert shifted.decision["mean_regret"] == pytest.approx(exact.decision["mean_regret"])


def test_comparison_scores_model_and_baselines_on_the_held_out_half():
    frame, outcomes = synthetic_dataset(days=150)
    summary, note = compare_level_model(
        frame, outcomes, Load(periods=2, window_hours=6.0), params=FAST
    )

    assert "gradient_boosting" in summary.index
    assert {"seasonal_naive", "seasonal_mean"} <= set(summary.index)
    assert note["test_decisions"] > 0
    assert note["train_decisions"] + note["test_decisions"] == frame["decision_id"].nunique()


def test_the_comparison_needs_more_than_one_date():
    frame, outcomes = synthetic_dataset(days=1)
    summary, note = compare_level_model(frame, outcomes, Load(periods=2, window_hours=6.0))
    assert summary.empty
    assert "reason" in note


def test_permutation_importance_separates_the_two_metrics():
    # A feature can matter to the level and not to the choice, which is exactly
    # what this project is about, so importance is reported against both.
    frame, outcomes = synthetic_dataset(days=100)
    model, columns = fit_level_model(frame, FAST)

    importance = permutation_importance(
        model, frame, columns, outcomes, Load(periods=2, window_hours=6.0), repeats=1
    )
    assert set(importance.columns) == {"feature", "mae_increase", "captured_fraction_loss"}
    assert len(importance) == len(columns)


def test_restoring_the_level_leaves_the_ordering_untouched():
    frame, outcomes = synthetic_dataset(days=40)
    load = Load(periods=2, window_hours=6.0)

    relative = frame["actual_rel"]
    restored = restore_level(frame, relative)

    from gridcast.decisions import score_through_harness

    assert score_through_harness(frame, outcomes, load, relative) == score_through_harness(
        frame, outcomes, load, restored
    )


def test_model_command_says_so_on_an_empty_store(tmp_path, capsys):
    from gridcast.cli import main

    assert main(["--root", str(tmp_path), "model"]) == 0
    assert "no complete windows" in capsys.readouterr().out


def test_a_feature_that_is_wholly_missing_is_dropped():
    # A store without generation data leaves the nine fuel features entirely
    # absent. The histogram binner raises on such a column rather than ignoring
    # it, so fitting would fail outright.
    frame, _ = synthetic_dataset(days=20)
    frame["recent_gas"] = np.nan

    assert "recent_gas" not in model_features(frame)


def test_a_constant_feature_is_dropped():
    frame, _ = synthetic_dataset(days=20)
    frame["recent_staleness_hours"] = 1.5

    assert "recent_staleness_hours" not in model_features(frame)


def test_a_model_fits_when_several_features_are_absent():
    frame, outcomes = synthetic_dataset(days=60)
    for column in ("recent_gas", "recent_wind", "intensity_lag_7d"):
        frame[column] = np.nan

    model, columns = fit_level_model(frame, FAST)
    prediction = predict(model, frame, columns)
    assert prediction.notna().all()


# -- intervals on the difference ------------------------------------------


def test_paired_regret_keeps_only_decisions_every_predictor_covered():
    # Pairing requires the same decisions on both sides. Comparing a model
    # scored on one sample against a baseline scored on another is the error
    # this project has already made twice.
    from gridcast.models import paired_decision_regret

    frame, outcomes = synthetic_dataset(days=40)
    load = Load(periods=2, window_hours=6.0)

    partial = frame["actual"].copy()
    partial[frame["decision_id"] == frame["decision_id"].min()] = np.nan

    paired = paired_decision_regret(
        frame, outcomes, load, {"complete": frame["actual"], "partial": partial}
    )
    assert len(paired) == frame["decision_id"].nunique() - 1
    assert set(paired.columns) == {"complete", "partial"}


def test_a_clear_advantage_gives_an_interval_above_zero():
    from gridcast.models import bootstrap_difference

    rng = np.random.default_rng(0)
    paired = pd.DataFrame({"candidate": rng.normal(5, 2, 300), "reference": rng.normal(12, 2, 300)})
    result = bootstrap_difference(paired, "candidate", "reference", resamples=400)

    assert result["regret_reduction"] > 0
    assert result["low"] > 0
    assert result["worse_fraction"] == 0.0


def test_a_marginal_advantage_gives_an_interval_spanning_zero():
    # The case this exists for: the summary shows one predictor ahead, and the
    # interval says the lead is smaller than the noise.
    from gridcast.models import bootstrap_difference

    rng = np.random.default_rng(1)
    candidate = rng.normal(10, 8, 200)
    # A tenth of a unit better on average, against a spread of twelve: far
    # smaller than the sampling error on two hundred decisions.
    reference = candidate + rng.normal(0.1, 12, 200)
    paired = pd.DataFrame({"candidate": candidate, "reference": reference})

    result = bootstrap_difference(paired, "candidate", "reference", resamples=400)
    assert result["low"] < 0 < result["high"]
    assert 0.1 < result["worse_fraction"] < 0.9


def test_a_difference_needs_a_predictor_that_is_present():
    from gridcast.models import bootstrap_difference

    paired = pd.DataFrame({"candidate": [1.0, 2.0]})
    assert bootstrap_difference(paired, "candidate", "absent") == {}


def test_comparison_with_intervals_reports_the_split_span():
    # The span matters: a model trained on winter and tested on summer is being
    # asked a different question from one trained and tested across both.
    from gridcast.models import compare_with_intervals

    frame, outcomes = synthetic_dataset(days=200)
    summary, differences, note = compare_with_intervals(
        frame, outcomes, Load(periods=2, window_hours=6.0), params=FAST, resamples=200
    )

    assert "gradient_boosting" in summary.index
    assert set(differences["against"]) == {"seasonal_mean", "seasonal_naive"}
    assert "significant" in differences.columns
    assert note["train_span"][1] < note["test_span"][0]


def test_model_command_reports_the_interval_on_the_difference(tmp_path, capsys):
    import datetime as dt

    from gridcast.cli import main
    from gridcast.storage import write_snapshot

    entries = []
    start = dt.datetime(2026, 3, 1, tzinfo=dt.UTC)
    rng = np.random.default_rng(2)
    for step in range(48 * 120):
        moment = start + dt.timedelta(minutes=30 * step)
        value = 200.0 + 50.0 * np.sin(2 * np.pi * (step % 48) / 48) + rng.normal(0, 5)
        entries.append(
            {
                "from": moment.strftime("%Y-%m-%dT%H:%MZ"),
                "to": (moment + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                "intensity": {"forecast": value, "actual": value, "index": "moderate"},
            }
        )
    write_snapshot(
        {"data": entries},
        "intensity_range_20260301",
        "/intensity/range",
        start + dt.timedelta(days=121),
        root=tmp_path,
    )

    assert (
        main(
            [
                "--root",
                str(tmp_path),
                "model",
                "--periods",
                "2",
                "--window",
                "6",
                "--issue-hours",
                "18",
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "reduction in mean regret" in out
    assert "not a result until this one is" in out
