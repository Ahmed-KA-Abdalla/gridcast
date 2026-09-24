from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tests.test_models import FAST, synthetic_dataset

from gridcast.ranking import (
    fit_pairwise_model,
    fit_relative_model,
    pairwise_score,
    sample_pairs,
)
from gridcast.scheduling import Load


def test_sampled_pairs_never_cross_a_window():
    # Comparing a period in January with one in July is not a comparison any
    # scheduler makes.
    frame, _ = synthetic_dataset(days=30)
    columns = ["sin_day", "cos_day"]

    # A feature that identifies the window: if pairs crossed windows, the
    # difference in this column would be non-zero somewhere.
    frame["window_tag"] = frame.groupby("decision_id").ngroup().astype(float)
    features, _ = sample_pairs(frame, [*columns, "window_tag"], pairs_per_window=20)

    assert np.allclose(features[:, -1], 0.0)


def test_each_pair_is_emitted_in_both_orders():
    # Otherwise the labels inherit the sampling's bias and a scorer can look
    # accurate from the sign convention alone.
    frame, _ = synthetic_dataset(days=20)
    features, labels = sample_pairs(frame, ["sin_day", "cos_day"], pairs_per_window=10)

    assert len(labels) % 2 == 0
    assert labels.mean() == pytest.approx(0.5)
    assert np.allclose(features[0::2], -features[1::2])


def test_pairs_too_close_to_matter_are_dropped():
    # Two periods a fraction apart decide nothing, and learning to order them
    # fits noise.
    frame, _ = synthetic_dataset(days=20)
    frame["actual"] = 100.0  # every pair identical

    features, labels = sample_pairs(frame, ["sin_day"], pairs_per_window=20)
    assert len(features) == 0


def test_the_label_says_which_period_was_cheaper():
    frame = pd.DataFrame(
        {
            "decision_id": ["a"] * 2,
            "actual": [10.0, 200.0],
            "feature": [1.0, 0.0],
        }
    )
    features, labels = sample_pairs(frame, ["feature"], pairs_per_window=40, min_gap=1.0)

    # Wherever the feature difference is positive the first period is the one
    # with feature 1.0, which is the cheaper of the two.
    positive = features[:, 0] > 0
    assert labels[positive].mean() == pytest.approx(1.0)
    assert labels[~positive].mean() == pytest.approx(0.0)


def test_a_relative_model_learns_the_shape_and_not_the_level():
    frame, _ = synthetic_dataset(days=120)
    model, columns = fit_relative_model(frame, FAST)
    prediction = pd.Series(model.predict(frame[columns]), index=frame.index)

    assert prediction.corr(frame["actual_rel"]) > 0.9
    # Trained on a demeaned target, so its predictions centre near zero and are
    # not intensities.
    assert abs(prediction.mean()) < 10.0


def test_a_pairwise_scorer_orders_periods_correctly():
    frame, _ = synthetic_dataset(days=150)
    classifier, pipeline, columns = fit_pairwise_model(frame, pairs_per_window=40)
    score = pairwise_score(classifier, pipeline, frame, columns)

    # Ordering only: the score has no unit, so it is judged by rank agreement.
    assert score.corr(frame["actual_rel"], method="spearman") > 0.8


def test_the_pairwise_score_is_not_an_intensity():
    # Stated as a test because it governs how the model may be reported: its
    # error against realised values is meaningless.
    frame, _ = synthetic_dataset(days=100)
    classifier, pipeline, columns = fit_pairwise_model(frame, pairs_per_window=30)
    score = pairwise_score(classifier, pipeline, frame, columns)

    assert abs(score.mean() - frame["actual"].mean()) > 50.0


def test_the_pairwise_scorer_schedules_as_well_as_it_ranks():
    from gridcast.decisions import score_through_harness

    frame, outcomes = synthetic_dataset(days=150)
    load = Load(periods=2, window_hours=6.0)

    classifier, pipeline, columns = fit_pairwise_model(frame, pairs_per_window=40)
    score = pairwise_score(classifier, pipeline, frame, columns)

    scored = score_through_harness(frame, outcomes, load, score)
    perfect = score_through_harness(frame, outcomes, load, frame["actual"])

    assert scored["n"] == perfect["n"]
    # A scorer carrying only the ordering still secures most of the saving.
    assert scored["captured_fraction"] > 0.5


def test_fitting_pairwise_needs_usable_pairs():
    frame, _ = synthetic_dataset(days=5)
    frame["actual"] = 100.0

    with pytest.raises(ValueError, match="no usable pairs"):
        fit_pairwise_model(frame)


def test_a_constant_added_within_a_window_leaves_the_pairwise_score_unchanged():
    # The invariance the objective is meant to have: a level shift changes no
    # ordering, so it must change no score difference.
    frame, _ = synthetic_dataset(days=80)
    classifier, pipeline, columns = fit_pairwise_model(frame, pairs_per_window=30)

    base = pairwise_score(classifier, pipeline, frame, columns)
    shifted = frame.copy()
    shifted["actual"] = shifted["actual"] + 100.0
    moved = pairwise_score(classifier, pipeline, shifted, columns)

    # The target moved; the features did not, so the score cannot have.
    assert np.allclose(base.to_numpy(), moved.to_numpy())


def test_comparing_objectives_scores_them_on_the_same_decisions():
    from gridcast.ranking import compare_objectives

    frame, outcomes = synthetic_dataset(days=200)
    summary, differences, note = compare_objectives(
        frame,
        outcomes,
        Load(periods=2, window_hours=6.0),
        params=FAST,
        resamples=200,
        pairs_per_window=20,
    )

    assert {"level_squared_error", "relative_squared_error", "pairwise_ranking"} <= set(
        summary.index
    )
    # Same held-out decisions for every objective.
    assert summary["n_decisions"].nunique() == 1
    assert note["train_span"][1] < note["test_span"][0]


def test_accuracy_is_left_blank_where_it_is_undefined():
    from gridcast.ranking import compare_objectives

    frame, outcomes = synthetic_dataset(days=200)
    summary, _, _ = compare_objectives(
        frame,
        outcomes,
        Load(periods=2, window_hours=6.0),
        params=FAST,
        resamples=100,
        pairs_per_window=20,
    )

    assert np.isnan(summary.loc["pairwise_ranking", "mae"])
    assert np.isnan(summary.loc["relative_squared_error", "mae"])
    assert np.isfinite(summary.loc["level_squared_error", "mae"])


def test_differences_are_reported_against_every_reference():
    from gridcast.ranking import compare_objectives

    frame, outcomes = synthetic_dataset(days=200)
    _, differences, _ = compare_objectives(
        frame,
        outcomes,
        Load(periods=2, window_hours=6.0),
        params=FAST,
        resamples=200,
        pairs_per_window=20,
    )

    assert set(differences["candidate"]) == {"relative_squared_error", "pairwise_ranking"}
    assert set(differences["against"]) == {
        "level_squared_error",
        "seasonal_mean",
        "linear_relative",
    }
    assert "significant" in differences.columns
