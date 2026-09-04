"""Models fitted on the decision dataset, and the control they are judged against.

This module holds the conventional approach first: gradient boosting trained to
minimise squared error on carbon intensity, which is what forecasting this
quantity normally means. It exists to be the control.

The reason for a control rather than a straight attempt at improvement is that
this project has already found that accuracy and decision quality come apart. A
model that predicts the level well may schedule no better than a weekly average,
and unless the conventional model is fitted and scored on both metrics there is
no way to tell whether a later ranking model is better because of its objective
or merely because it is a model at all.

Both metrics are reported for every model. A model that wins on one and loses on
the other is the outcome this project expects, and reporting a single number
would hide it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .decisions import score_through_harness, split_decisions, to_forecast_frame
from .features import FEATURE_COLUMNS
from .scheduling import Load

#: Relative features are added by the decision dataset alongside the absolute
#: ones. A model given both can use whichever helps; which it uses is visible
#: in the permutation importances rather than assumed here.
RELATIVE_SUFFIX = "_rel"

#: Kept small deliberately. The decision dataset has a few thousand independent
#: situations, not a few hundred thousand, and a deeper model would fit the
#: weeks rather than the structure.
DEFAULT_PARAMS = {
    "max_iter": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}


def model_features(frame: pd.DataFrame) -> list[str]:
    """Feature columns present in the frame, absolute and window-relative.

    Columns with fewer than two distinct values are dropped. A feature that is
    wholly missing or constant over the training window carries no information,
    and the histogram binner cannot build a threshold from it: it raises rather
    than ignoring the column, so a store without generation data — where the
    nine fuel features are entirely absent — would fail to fit at all.
    """
    columns = [name for name in FEATURE_COLUMNS if name in frame.columns]
    columns += [name for name in frame.columns if name.endswith(RELATIVE_SUFFIX)]
    # The target's own relative form would be the answer.
    columns = [name for name in columns if name != f"actual{RELATIVE_SUFFIX}"]

    return [name for name in columns if frame[name].nunique(dropna=True) > 1]


@dataclass
class Scored:
    """A model's accuracy and its decision quality, side by side."""

    name: str
    mae: float
    rmse: float
    decision: dict[str, float]

    def row(self) -> dict[str, float]:
        return {
            "mae": self.mae,
            "rmse": self.rmse,
            "n_decisions": self.decision.get("n", 0),
            "mean_regret": self.decision.get("mean_regret", np.nan),
            "hit_rate": self.decision.get("hit_rate", np.nan),
            "captured_fraction": self.decision.get("captured_fraction", np.nan),
        }


def accuracy(prediction: pd.Series, actual: pd.Series) -> tuple[float, float]:
    """Mean absolute and root mean square error, over rows the model covered."""
    mask = prediction.notna() & actual.notna()
    error = (prediction[mask] - actual[mask]).astype(float)
    if error.empty:
        return float("nan"), float("nan")
    return float(error.abs().mean()), float(np.sqrt((error**2).mean()))


def score_predictions(
    name: str,
    frame: pd.DataFrame,
    prediction: pd.Series,
    outcomes: pd.DataFrame,
    load: Load,
) -> Scored:
    """Score one set of predictions on accuracy and on the decision it produces.

    ``prediction`` is an intensity in gCO2/kWh for the accuracy figures. Where a
    model predicts a window-relative quantity instead, the caller restores the
    level before scoring, since a relative prediction cannot be compared with a
    realised intensity — though it may schedule identically.
    """
    mae, rmse = accuracy(prediction, frame["actual"])
    decision = score_through_harness(frame, outcomes, load, prediction)
    return Scored(name=name, mae=mae, rmse=rmse, decision=decision)


def fit_level_model(
    train: pd.DataFrame, params: dict | None = None
) -> tuple[HistGradientBoostingRegressor, list[str]]:
    """Gradient boosting trained on squared error of the intensity itself.

    The conventional objective, and the control. Missing features are handled
    natively rather than imputed, which matters because the seasonal lag columns
    are absent for the first weeks of the record and for periods the source
    never published.
    """
    columns = model_features(train)
    model = HistGradientBoostingRegressor(**(params or DEFAULT_PARAMS))
    model.fit(train[columns], train["actual"])
    return model, columns


def predict(
    model: HistGradientBoostingRegressor, frame: pd.DataFrame, columns: list[str]
) -> pd.Series:
    """Predictions aligned to the frame's index."""
    return pd.Series(model.predict(frame[columns]), index=frame.index)


def baseline_predictions(frame: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, pd.Series]:
    """The seasonal baselines, as the standard the model must beat.

    Computed through the same code the rest of the project uses, so a difference
    between a model and a baseline cannot come from a difference in how their
    predictions were produced.
    """
    from .baseline import seasonal_mean, seasonal_naive

    return {
        "seasonal_naive": seasonal_naive(frame, outcomes, as_of="captured_at"),
        "seasonal_mean": seasonal_mean(frame, outcomes, as_of="captured_at"),
    }


def compare_level_model(
    dataset: pd.DataFrame,
    outcomes: pd.DataFrame,
    load: Load,
    train_fraction: float = 0.6,
    params: dict | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit the control on earlier dates and score it against the baselines.

    Everything is scored on the held-out half only, and the baselines are scored
    there too rather than over the whole record, so the comparison is on one
    sample.
    """
    train, test = split_decisions(dataset, train_fraction)
    if train.empty or test.empty:
        return pd.DataFrame(), {"reason": "not enough dates to hold any out"}

    model, columns = fit_level_model(train, params)
    scored = [
        score_predictions("gradient_boosting", test, predict(model, test, columns), outcomes, load)
    ]
    for name, prediction in baseline_predictions(test, outcomes).items():
        scored.append(score_predictions(name, test, prediction, outcomes, load))

    summary = pd.DataFrame({item.name: item.row() for item in scored}).T
    note = {
        "train_decisions": int(train["decision_id"].nunique()),
        "test_decisions": int(test["decision_id"].nunique()),
        "train_dates": int(train["date"].nunique()),
        "test_dates": int(test["date"].nunique()),
        "features": len(columns),
    }
    return summary, note


def permutation_importance(
    model: HistGradientBoostingRegressor,
    frame: pd.DataFrame,
    columns: list[str],
    outcomes: pd.DataFrame,
    load: Load,
    repeats: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """How much each feature is worth to accuracy and to the decision separately.

    A feature can matter a great deal to the level and not at all to the choice,
    which is the whole subject of this project. Measuring importance against one
    metric alone would obscure exactly the distinction being investigated.
    """
    rng = np.random.default_rng(seed)
    base_prediction = predict(model, frame, columns)
    base_mae, _ = accuracy(base_prediction, frame["actual"])
    base_decision = score_through_harness(frame, outcomes, load, base_prediction)
    base_captured = base_decision.get("captured_fraction", np.nan)

    records = []
    for column in columns:
        mae_losses, captured_losses = [], []
        for _ in range(repeats):
            shuffled = frame.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            prediction = predict(model, shuffled, columns)

            mae, _ = accuracy(prediction, frame["actual"])
            mae_losses.append(mae - base_mae)

            decision = score_through_harness(frame, outcomes, load, prediction)
            captured_losses.append(base_captured - decision.get("captured_fraction", np.nan))

        records.append(
            {
                "feature": column,
                "mae_increase": float(np.mean(mae_losses)),
                "captured_fraction_loss": float(np.mean(captured_losses)),
            }
        )

    return pd.DataFrame(records).sort_values("mae_increase", ascending=False)


def restore_level(frame: pd.DataFrame, relative: pd.Series) -> pd.Series:
    """Turn a window-relative prediction back into an intensity.

    Adds each window's mean realised intensity. This is not something a
    forecaster could do at issue time, so it is used only to put a relative
    model's accuracy on the same scale as the others for reporting. The decision
    is unaffected either way, since adding a constant within a window changes no
    ordering.
    """
    means = frame.groupby("decision_id", sort=False)["actual"].transform("mean")
    return relative + means


#: Resamples for the interval on a difference between two scorers.
BOOTSTRAP_RESAMPLES = 1000


def paired_decision_regret(
    frame: pd.DataFrame,
    outcomes: pd.DataFrame,
    load: Load,
    predictions: dict[str, pd.Series],
) -> pd.DataFrame:
    """Regret per decision for several predictors, on the decisions all covered.

    An interval on a difference has to be paired, and pairing requires the same
    decisions on both sides. A predictor that cannot cover a decision drops it
    for every predictor, so no comparison is made across different samples —
    which is the error this project has already made twice, once with lead-time
    buckets and once with the scheduling baselines.
    """
    from .scheduling import evaluate_decisions

    scored = {}
    for name, prediction in predictions.items():
        decisions = evaluate_decisions(to_forecast_frame(frame, prediction), outcomes, load)
        if decisions.empty:
            return pd.DataFrame()
        regret = (
            decisions.set_index("window_start")["chosen"]
            - decisions.set_index("window_start")["oracle"]
        )
        scored[name] = regret

    merged = pd.DataFrame(scored).dropna()
    return merged


def bootstrap_difference(
    paired: pd.DataFrame,
    candidate: str,
    reference: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """Interval for the reduction in mean regret of one predictor over another.

    Positive means the candidate incurs less regret. Resampled by decision,
    which is the independent unit here: unlike the revision frames, each
    decision contributes one row.
    """
    if paired.empty or candidate not in paired or reference not in paired:
        return {}

    difference = (paired[reference] - paired[candidate]).to_numpy(dtype=float)
    if len(difference) < 2:
        return {}

    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for index in range(resamples):
        drawn = rng.integers(0, len(difference), len(difference))
        means[index] = difference[drawn].mean()

    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "n_decisions": int(len(difference)),
        "regret_reduction": float(difference.mean()),
        "low": float(low),
        "high": float(high),
        "worse_fraction": float((means <= 0).mean()),
    }


def compare_with_intervals(
    dataset: pd.DataFrame,
    outcomes: pd.DataFrame,
    load: Load,
    train_fraction: float = 0.6,
    params: dict | None = None,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Score the model and the baselines, and interval the differences.

    Returns the per-predictor summary, a table of differences against each
    baseline, and a note describing the split. The summary alone cannot settle
    whether a model beats a baseline: the baselines themselves move by more
    between samples than a model typically beats them by.
    """
    train, test = split_decisions(dataset, train_fraction)
    if train.empty or test.empty:
        return pd.DataFrame(), pd.DataFrame(), {"reason": "not enough dates to hold any out"}

    model, columns = fit_level_model(train, params)
    predictions = {"gradient_boosting": predict(model, test, columns)}
    predictions.update(baseline_predictions(test, outcomes))

    scored = [
        score_predictions(name, test, prediction, outcomes, load)
        for name, prediction in predictions.items()
    ]
    summary = pd.DataFrame({item.name: item.row() for item in scored}).T

    paired = paired_decision_regret(test, outcomes, load, predictions)
    rows = []
    for reference in ("seasonal_mean", "seasonal_naive"):
        result = bootstrap_difference(paired, "gradient_boosting", reference, resamples=resamples)
        if result:
            rows.append({"against": reference, **result})
    differences = pd.DataFrame(rows)
    if not differences.empty:
        differences["significant"] = differences["low"] > 0

    note = {
        "train_decisions": int(train["decision_id"].nunique()),
        "test_decisions": int(test["decision_id"].nunique()),
        "train_dates": int(train["date"].nunique()),
        "test_dates": int(test["date"].nunique()),
        "train_span": (str(train["date"].min()), str(train["date"].max())),
        "test_span": (str(test["date"].min()), str(test["date"].max())),
        "features": len(columns),
        "resamples": resamples,
    }
    return summary, differences, note
