"""Fitting on the ordering of a window rather than on the level of intensity.

The decision a scheduler makes depends only on which periods of a window are
cheapest, so a forecast that is wrong by a constant schedules perfectly and one
that is accurate but inverts two adjacent periods does not. Every forecaster in
this repository so far, including the published one, is fitted to minimise
squared error on the level. This module fits the ordering instead.

Two candidates, of increasing distance from the conventional objective.

The *relative* model keeps squared-error loss but changes the target to each
period's deviation from its own window's mean. It cannot predict the level at
all, by construction, and it is included to separate two effects that are
otherwise confounded: removing the level from the target, and abandoning
squared error.

The *pairwise* model changes the loss. For pairs of periods within a window it
learns which of the two is cheaper, from the difference between their feature
vectors. A linear scorer is used deliberately: the ordering a model induces on
single periods is recoverable from a model fitted on differences only when the
scorer is additive, since a tree fitted on differences has no corresponding
per-period score.

Both are scored on regret against the same decisions as the conventional model,
so the comparison isolates the objective.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .models import DEFAULT_PARAMS, model_features

#: Period pairs drawn from each window for the pairwise model. A window of
#: forty-eight periods admits over a thousand pairs, and using all of them would
#: weight the fit towards windows rather than towards the pairs that decide a
#: schedule.
PAIRS_PER_WINDOW = 80

#: Pairs whose realised intensities differ by less than this contribute nothing
#: to a decision and are dropped. Learning to order two periods that are
#: effectively identical fits noise.
MIN_PAIR_GAP = 1.0


def fit_relative_model(train: pd.DataFrame, params: dict | None = None):
    """Gradient boosting on each period's deviation from its window's mean.

    Squared-error loss still, but on a target the scheduler can use. The model
    cannot predict the level and is not asked to.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    columns = model_features(train)
    model = HistGradientBoostingRegressor(**(params or DEFAULT_PARAMS))
    model.fit(train[columns], train["actual_rel"])
    return model, columns


def sample_pairs(
    frame: pd.DataFrame,
    columns: list[str],
    pairs_per_window: int = PAIRS_PER_WINDOW,
    min_gap: float = MIN_PAIR_GAP,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature differences and labels for pairs of periods within a window.

    Pairs are drawn within a window and never across windows, because comparing
    a period in January with one in July is not a comparison any scheduler
    makes. The label is whether the first period of the pair was realised
    cheaper than the second.

    Each pair is emitted in both orders. Otherwise the labels inherit whatever
    bias the sampling has, and a scorer can achieve apparent accuracy from the
    sign convention alone.
    """
    rng = np.random.default_rng(seed)
    differences, labels = [], []

    for _, window in frame.groupby("decision_id", sort=False):
        values = window["actual"].to_numpy(dtype=float)
        features = window[columns].to_numpy(dtype=float)
        size = len(window)
        if size < 2:
            continue

        left = rng.integers(0, size, pairs_per_window)
        right = rng.integers(0, size, pairs_per_window)
        keep = np.abs(values[left] - values[right]) >= min_gap
        left, right = left[keep], right[keep]
        if len(left) == 0:
            continue

        difference = features[left] - features[right]
        cheaper = (values[left] < values[right]).astype(int)

        # Each pair immediately followed by its reverse, so that every even row
        # is the negation of the odd row after it. Emitting both orders keeps
        # the sign convention from carrying signal; interleaving them makes the
        # property checkable without knowing how windows were batched.
        paired = np.empty((2 * len(difference), difference.shape[1]))
        paired[0::2] = difference
        paired[1::2] = -difference

        both = np.empty(2 * len(cheaper), dtype=int)
        both[0::2] = cheaper
        both[1::2] = 1 - cheaper

        differences.append(paired)
        labels.append(both)

    if not differences:
        return np.empty((0, len(columns))), np.empty(0)
    return np.vstack(differences), np.concatenate(labels)


def fit_linear_relative_model(train: pd.DataFrame):
    """Ordinary least squares on the window-relative target.

    The control that separates two things the comparison would otherwise
    confound. The pairwise model is linear because a scorer fitted on feature
    differences must be additive to induce an ordering on single periods, while
    the other candidates are trees. Without a linear model trained on a
    conventional loss, a pairwise result could be attributed to its objective
    when it belongs to its functional form.

    Imputation and scaling as for the pairwise model, for the same reason and
    fitted on the training rows only.
    """
    from sklearn.linear_model import Ridge

    columns = model_features(train)
    imputer = SimpleImputer(strategy="median").fit(train[columns])
    scaler = StandardScaler().fit(imputer.transform(train[columns]))
    pipeline = Pipeline([("impute", imputer), ("scale", scaler)])

    model = Ridge(alpha=1.0)
    model.fit(pipeline.transform(train[columns]), train["actual_rel"])
    return model, pipeline, columns


def linear_relative_score(model, pipeline: Pipeline, frame: pd.DataFrame, columns: list[str]):
    """Predictions from the linear control, aligned to the frame."""
    return pd.Series(model.predict(pipeline.transform(frame[columns])), index=frame.index)


def fit_pairwise_model(
    train: pd.DataFrame,
    pairs_per_window: int = PAIRS_PER_WINDOW,
    seed: int = 0,
):
    """A linear scorer fitted on which of two periods is cheaper.

    Fitted without an intercept: a constant added to every period of a window
    changes no ordering, so an intercept is unidentifiable from differences and
    would only absorb noise.

    Missing features are imputed with the training median rather than handled
    natively, because a linear model has no equivalent of a tree's missing-value
    branch. This is a cost of the linear form and is stated rather than hidden:
    the seasonal lag features are genuinely absent early in the record.
    """
    columns = model_features(train)
    imputer = SimpleImputer(strategy="median").fit(train[columns])
    scaler = StandardScaler().fit(imputer.transform(train[columns]))

    prepared = train.copy()
    prepared[columns] = scaler.transform(imputer.transform(train[columns]))

    features, labels = sample_pairs(prepared, columns, pairs_per_window, seed=seed)
    if len(features) == 0:
        raise ValueError("no usable pairs in the training window")

    classifier = LogisticRegression(fit_intercept=False, max_iter=2000, C=1.0)
    classifier.fit(features, labels)

    pipeline = Pipeline([("impute", imputer), ("scale", scaler)])
    return classifier, pipeline, columns


def pairwise_score(
    classifier: LogisticRegression,
    pipeline: Pipeline,
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """Score each period so that cheaper periods score lower.

    The classifier's coefficients give the probability that the first period of
    a pair is cheaper, so their dot product with a period's features increases
    with cheapness. The sign is flipped here so the result orders like an
    intensity and can go through the existing scheduler unchanged.

    The returned score is not an intensity and has no unit. Its accuracy against
    realised values is meaningless, which is the point: it carries ordering and
    nothing else.
    """
    prepared = pipeline.transform(frame[columns])
    return pd.Series(-(prepared @ classifier.coef_.ravel()), index=frame.index)


def compare_objectives(
    dataset: pd.DataFrame,
    outcomes: pd.DataFrame,
    load,
    train_fraction: float = 0.6,
    params: dict | None = None,
    resamples: int = 1000,
    pairs_per_window: int = PAIRS_PER_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit every objective on the same data and score them on the same decisions.

    The question is whether the objective matters once a model is involved at
    all. The conventional model minimises squared error on the level; the
    relative model minimises squared error on the deviation from the window
    mean; the pairwise model minimises a classification loss on which of two
    periods is cheaper. They share features, training rows and held-out
    decisions, so a difference between them is a difference in objective.

    Accuracy is reported where it is defined. The pairwise scorer produces no
    intensity, so its error against realised values is not reported rather than
    being reported as a large number: a meaningless figure in that column would
    invite the comparison the whole exercise argues against.
    """
    from .decisions import split_decisions
    from .models import (
        baseline_predictions,
        bootstrap_difference,
        fit_level_model,
        paired_decision_regret,
        predict,
        score_predictions,
    )

    train, test = split_decisions(dataset, train_fraction)
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame(), {"reason": "not enough dates to hold any out"}

    level_model, level_columns = fit_level_model(train, params)
    relative_model, relative_columns = fit_relative_model(train, params)
    classifier, pipeline, pair_columns = fit_pairwise_model(
        train, pairs_per_window=pairs_per_window
    )
    linear_model, linear_pipeline, linear_columns = fit_linear_relative_model(train)

    predictions = {
        "level_squared_error": predict(level_model, test, level_columns),
        "relative_squared_error": pd.Series(
            relative_model.predict(test[relative_columns]), index=test.index
        ),
        "linear_relative": linear_relative_score(
            linear_model, linear_pipeline, test, linear_columns
        ),
        "pairwise_ranking": pairwise_score(classifier, pipeline, test, pair_columns),
    }
    predictions.update(baseline_predictions(test, outcomes))

    rows = {}
    for name, prediction in predictions.items():
        scored = score_predictions(name, test, prediction, outcomes, load)
        row = scored.row()
        if name in ("relative_squared_error", "linear_relative", "pairwise_ranking"):
            # Neither predicts an intensity, so neither has an error against one.
            row["mae"] = np.nan
            row["rmse"] = np.nan
        rows[name] = row

    summary = pd.DataFrame(rows).T

    paired = paired_decision_regret(test, outcomes, load, predictions)
    differences = []
    for candidate in ("relative_squared_error", "pairwise_ranking"):
        for reference in ("level_squared_error", "seasonal_mean", "linear_relative"):
            result = bootstrap_difference(paired, candidate, reference, resamples=resamples)
            if result:
                differences.append({"candidate": candidate, "against": reference, **result})

    table = pd.DataFrame(differences)
    if not table.empty:
        table["significant"] = table["low"] > 0

    note = {
        "train_decisions": int(train["decision_id"].nunique()),
        "test_decisions": int(test["decision_id"].nunique()),
        "train_span": (str(train["date"].min()), str(train["date"].max())),
        "test_span": (str(test["date"].min()), str(test["date"].max())),
        "features": len(level_columns),
        "pairs": int(pairs_per_window),
    }
    return summary, table, note
