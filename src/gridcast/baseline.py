"""Seasonal baselines, and the machinery that keeps them causal.

Carbon intensity has strong daily and weekly structure: a Tuesday at 18:00
resembles the previous Tuesday at 18:00 more than it resembles the hour before
it. A baseline exploiting only that structure is the standard against which any
learned model must justify itself, and it is a genuinely hard one to beat.

Every prediction here is made from observations that were available before the
moment the prediction is attributed to. Comparing a baseline against the
published forecast is meaningless unless both were restricted to the same
information.

Availability is decided by when a period's value could have been known, not by
when this project happened to fetch it. A backfilled observation carries the
capture time of the backfill run, which is today; using that as the availability
test would declare two years of settled history unavailable to any forecast
issued before the backfill, and the matched comparison would silently score
nothing. The rule is therefore that a period is available once it has ended and
a settlement allowance has passed.
"""

from __future__ import annotations

import pandas as pd

#: Reference points, in days before the target period. Seven-day multiples
#: preserve both the weekday and the position in the day.
SEASONAL_LAGS = (7, 14, 21)

DAY = pd.Timedelta(days=1)

PERIOD = pd.Timedelta(minutes=30)

#: How long after a period ends before its realised value can be relied upon.
#: Deliberately generous: an allowance that is too short would let a prediction
#: use a value that was not yet published, which is the failure that matters.
SETTLEMENT_LAG = pd.Timedelta(hours=1)


def _reference_columns(
    targets: pd.DataFrame,
    outcomes: pd.DataFrame,
    lags: tuple[int, ...],
    as_of: str | None,
) -> pd.DataFrame:
    """Look up the outcome at each seasonal lag, blanking unavailable ones.

    A reference is unavailable when the period was never observed (a gap in the
    record) or when it had not yet settled at the ``as_of`` time. Both cases
    yield NaN, which the callers below treat identically: fall back to a longer
    lag, or decline to predict.
    """
    actual = outcomes.set_index("period_start")["actual"]

    columns = {}
    for lag in lags:
        reference_period = targets["period_start"] - lag * DAY
        values = reference_period.map(actual)

        if as_of is not None:
            knowable_from = reference_period + PERIOD + SETTLEMENT_LAG
            values = values.where(knowable_from <= targets[as_of])

        columns[f"lag_{lag}d"] = values

    return pd.DataFrame(columns, index=targets.index)


def seasonal_naive(
    targets: pd.DataFrame,
    outcomes: pd.DataFrame,
    as_of: str | None = "captured_at",
    lags: tuple[int, ...] = SEASONAL_LAGS,
) -> pd.Series:
    """The value at the same half-hour of the most recent available same weekday.

    Falls back through the lags in order, so a period whose previous week is
    missing is predicted from a fortnight earlier rather than not at all.
    Returns NaN where no lag is available.
    """
    references = _reference_columns(targets, outcomes, lags, as_of)
    prediction = references.iloc[:, 0]
    for column in references.columns[1:]:
        prediction = prediction.fillna(references[column])
    return prediction.rename("seasonal_naive")


def seasonal_mean(
    targets: pd.DataFrame,
    outcomes: pd.DataFrame,
    as_of: str | None = "captured_at",
    lags: tuple[int, ...] = SEASONAL_LAGS,
) -> pd.Series:
    """The mean of every available same-weekday, same-half-hour observation.

    Averaging several weeks suppresses the influence of one unusual day at the
    cost of responding more slowly to a genuine shift in the generation mix.
    Which of the two baselines is harder to beat is an empirical question, and
    the point of having both.
    """
    references = _reference_columns(targets, outcomes, lags, as_of)
    return references.mean(axis=1, skipna=True).rename("seasonal_mean")


BASELINES = {
    "seasonal_naive": seasonal_naive,
    "seasonal_mean": seasonal_mean,
}


def add_baselines(
    targets: pd.DataFrame,
    outcomes: pd.DataFrame,
    as_of: str | None = "captured_at",
) -> pd.DataFrame:
    """Return ``targets`` with a prediction column for each baseline."""
    frame = targets.copy()
    for name, predictor in BASELINES.items():
        frame[name] = predictor(targets, outcomes, as_of=as_of)
    return frame
