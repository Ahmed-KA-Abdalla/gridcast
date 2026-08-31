"""A decision-shaped dataset over the whole settled record.

The scheduling evaluation so far has been limited to decisions the published
forecast faced, which is fewer than two hundred and all from one fortnight. That
is enough to compare forecasters and nowhere near enough to fit one.

This module builds the alternative: a decision at every issue time across the
settled record, each carrying the window's periods, features computed as of that
issue time, and the realised values. Two and a half years and both seasons,
without needing the published forecast at all.

The frame is deliberately long rather than wide — one row per period per
decision, not one row per decision. A model predicting the ordering within a
window needs the periods as separate rows to rank them, and a decision is
recovered by grouping on ``decision_id``.

Two things this module does not do. It does not choose or score anything, which
belongs in ``scheduling``. And it does not know what a model is; it produces the
same frame whether the predictions come from a baseline, a fitted model, or the
published forecast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features
from .load import generation_record, outcome_record
from .scheduling import PERIOD, Load
from .storage import DEFAULT_ROOT

#: Issue times per day for the generated decisions. Four is enough to cover the
#: daily cycle without successive decisions overlapping so heavily that the
#: sample looks larger than the number of distinct situations in it.
DEFAULT_ISSUE_HOURS = (2, 8, 14, 20)


def decision_windows(
    outcomes: pd.DataFrame,
    load: Load,
    issue_hours: tuple[int, ...] = DEFAULT_ISSUE_HOURS,
) -> pd.DataFrame:
    """One row per period per decision, over every complete window in the record.

    A decision is kept only where every period of its window has a settled
    value. Interpolating across a gap would invent the very quantity the
    scheduler is scored against.

    ``position`` is the period's index within its window, which is what a
    scheduler chooses among and what a ranking objective orders.
    """
    if outcomes.empty:
        return pd.DataFrame()

    settled = outcomes.set_index("period_start")["actual"].sort_index()
    first = settled.index.min().normalize()
    last = settled.index.max() - pd.Timedelta(hours=load.window_hours)
    if last <= first:
        return pd.DataFrame()

    starts = []
    for hour in sorted(issue_hours):
        starts.append(pd.date_range(first + pd.Timedelta(hours=hour), last, freq="D", tz="UTC"))
    issue_times = pd.DatetimeIndex(np.concatenate([s.to_numpy() for s in starts])).sort_values()

    frames = []
    for issue in issue_times:
        index = pd.date_range(issue, periods=load.window_periods, freq=PERIOD, tz="UTC")
        actual = settled.reindex(index)
        if actual.isna().any():
            continue

        frames.append(
            pd.DataFrame(
                {
                    "decision_id": issue,
                    "captured_at": issue,
                    "period_start": index,
                    "position": np.arange(load.window_periods),
                    "actual": actual.to_numpy(dtype=float),
                }
            )
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def window_relative(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Express columns as deviations from their own window's mean.

    A scheduler compares periods within one window and never across windows, so
    the level of a window carries no information it can use. Leaving features in
    absolute terms invites a model to spend its capacity predicting the level —
    which is the objective this project has already shown does not help.
    """
    relative = frame.copy()
    grouped = relative.groupby("decision_id", sort=False)
    for column in columns:
        if column in relative:
            relative[f"{column}_rel"] = relative[column] - grouped[column].transform("mean")
    return relative


def decision_dataset(
    root: Path = DEFAULT_ROOT,
    load: Load | None = None,
    issue_hours: tuple[int, ...] = DEFAULT_ISSUE_HOURS,
    relative_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Decisions with features attached, ready to train or score on.

    Features are computed as of each decision's issue time by the same code the
    published-forecast path uses, so a model trained here cannot see anything a
    forecaster at that moment could not.
    """
    load = load or Load()
    outcomes = outcome_record(root)
    generation = generation_record(root)

    windows = decision_windows(outcomes, load, issue_hours)
    if windows.empty:
        return windows

    features = build_features(windows, outcomes, generation)
    frame = pd.concat([windows, features], axis=1)

    columns = relative_columns or [
        "intensity_lag_7d",
        "intensity_lag_14d",
        "intensity_lag_21d",
    ]
    frame = window_relative(frame, columns)

    # The target a ranking objective orders: how far this period sits from its
    # window's mean. Positive is dirtier than the window average.
    frame["actual_rel"] = frame["actual"] - frame.groupby("decision_id", sort=False)[
        "actual"
    ].transform("mean")

    frame["date"] = frame["captured_at"].dt.date
    return frame


def to_forecast_frame(frame: pd.DataFrame, prediction: pd.Series) -> pd.DataFrame:
    """Shape predictions for ``scheduling.evaluate_decisions``.

    That function expects the loader's forecast record — ``captured_at``,
    ``period_start`` and ``forecast`` — so a model, a baseline and the published
    forecast all reach the scorer by the same path and cannot be treated
    differently by accident.
    """
    return pd.DataFrame(
        {
            "captured_at": frame["captured_at"].to_numpy(),
            "period_start": frame["period_start"].to_numpy(),
            "forecast": np.asarray(prediction, dtype=float),
        }
    )


def split_decisions(
    frame: pd.DataFrame, train_fraction: float = 0.6
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide by date, balanced on decisions rather than rows.

    Every window contributes the same number of rows here, unlike the revision
    frames, so rows and decisions balance together. The split is by date all the
    same: two decisions on one day share the periods their windows overlap on.
    """
    if frame.empty:
        return frame, frame

    counts = frame.groupby("date")["decision_id"].nunique().sort_index()
    if len(counts) < 2:
        return frame, frame.iloc[0:0]

    shares = counts.cumsum() / counts.sum()
    candidates = shares.iloc[:-1]
    boundary_index = int((candidates - train_fraction).abs().to_numpy().argmin())
    boundary = counts.index[boundary_index + 1]

    return frame[frame["date"] < boundary], frame[frame["date"] >= boundary]


def score_through_harness(
    frame: pd.DataFrame,
    outcomes: pd.DataFrame,
    load: Load,
    prediction: pd.Series,
) -> dict[str, float]:
    """Score a set of predictions using the same scorer the rest of the project uses.

    Kept here so that a model, a baseline and the published forecast all reach
    ``scheduling.summarise`` by one path. A second scoring implementation would
    be the easiest place for a difference between them to hide.
    """
    from .scheduling import evaluate_decisions, summarise

    forecasts = to_forecast_frame(frame, prediction)
    return summarise(evaluate_decisions(forecasts, outcomes, load))
