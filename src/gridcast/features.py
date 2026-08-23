"""Feature construction for the candidate model.

The unit of prediction is a (period, issue time) pair, not a period. Carbon
intensity forty-eight hours ahead and thirty minutes ahead are different
problems: the second may lean on an observation made an hour ago, the first may
not, and a model given one column of "recent intensity" without regard to lead
time would be using information that did not exist when the prediction was
supposedly made.

Every feature here is therefore computed as of an explicit ``captured_at``, and
the training set is the cross product of observed periods with a set of
hypothetical issue times. A period observed once contributes one row per lead,
each seeing only what was knowable at that lead.

One consequence worth stating: rows are correlated within a period, so any split
for validation must be by time and not by row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import PERIOD, SETTLEMENT_LAG
from .parse import FUELS

#: Hypothetical lead times, in hours, at which the training rows are generated.
#: Denser at the short end because that is where the information available
#: changes fastest.
TRAINING_HORIZONS = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 36.0, 48.0)

#: Same-half-hour references, in days. All exceed the 48-hour maximum lead, so
#: each is available at every horizon and needs no availability mask.
WEEKLY_LAGS = (7, 14, 21)

DAY = pd.Timedelta(days=1)
HOUR = pd.Timedelta(hours=1)


def knowable_from(period_start: pd.Series) -> pd.Series:
    """The moment a period's realised value can be relied upon.

    The period must have ended and the settlement allowance elapsed. This is a
    property of the data rather than of when this project fetched it.
    """
    return period_start + PERIOD + SETTLEMENT_LAG


def calendar_features(period_start: pd.Series) -> pd.DataFrame:
    """Time-of-day, weekday and seasonal position.

    Cyclical quantities are encoded as sine and cosine pairs so that 23:30 and
    00:00 are adjacent rather than maximally distant, which a raw hour index
    would imply. Local clock time drives demand behaviour, so the daily cycle is
    taken in Europe/London while the underlying index stays UTC.
    """
    local = period_start.dt.tz_convert("Europe/London")
    minutes = local.dt.hour * 60 + local.dt.minute
    day_fraction = minutes / (24 * 60)
    year_fraction = local.dt.dayofyear / 365.25

    return pd.DataFrame(
        {
            "sin_day": np.sin(2 * np.pi * day_fraction),
            "cos_day": np.cos(2 * np.pi * day_fraction),
            "sin_year": np.sin(2 * np.pi * year_fraction),
            "cos_year": np.cos(2 * np.pi * year_fraction),
            "day_of_week": local.dt.dayofweek,
            "is_weekend": (local.dt.dayofweek >= 5).astype(int),
        },
        index=period_start.index,
    )


def weekly_reference_features(
    targets: pd.DataFrame, outcomes: pd.DataFrame, lags: tuple[int, ...] = WEEKLY_LAGS
) -> pd.DataFrame:
    """Intensity at the same half-hour of earlier weeks.

    No availability mask is applied because every lag exceeds the maximum lead:
    a value from seven days ago has settled before any forecast issued within
    forty-eight hours of the target.
    """
    actual = outcomes.set_index("period_start")["actual"]
    columns = {
        f"intensity_lag_{lag}d": (targets["period_start"] - lag * DAY).map(actual) for lag in lags
    }
    return pd.DataFrame(columns, index=targets.index)


def latest_observation_features(
    targets: pd.DataFrame,
    outcomes: pd.DataFrame,
    generation: pd.DataFrame,
) -> pd.DataFrame:
    """The most recent intensity and fuel mix knowable at each issue time.

    Implemented as a backward as-of join on the knowable time, which is what
    makes the result honest: a row can only match an observation whose value had
    become reliable before the prediction was made.

    ``recent_staleness_hours`` is measured from the issue time, not from the
    target period. Measured that way it is normally constant at the settlement
    allowance, because there is almost always an observation that has just
    become knowable. Its informational content is therefore the exceptions: it
    rises above the floor only where the record has a gap, which is precisely
    when the recent-observation features deserve less weight. Distance from the
    observation to the target is not carried separately, being the sum of this
    quantity and the lead time the model already has.
    """
    left = targets[["captured_at"]].sort_values("captured_at")

    observations = outcomes[["period_start", "actual"]].copy()
    observations["knowable_at"] = knowable_from(observations["period_start"])
    if not generation.empty:
        mix = generation[["period_start", *FUELS]]
        observations = observations.merge(mix, on="period_start", how="left")
    observations = observations.sort_values("knowable_at")

    merged = pd.merge_asof(
        left,
        observations,
        left_on="captured_at",
        right_on="knowable_at",
        direction="backward",
    )
    merged.index = left.index

    frame = pd.DataFrame(index=left.index)
    frame["recent_intensity"] = merged["actual"]
    frame["recent_staleness_hours"] = (merged["captured_at"] - merged["period_start"]) / HOUR
    for fuel in FUELS:
        frame[f"recent_{fuel}"] = merged.get(fuel, np.nan)

    return frame.reindex(targets.index)


def build_features(
    targets: pd.DataFrame,
    outcomes: pd.DataFrame,
    generation: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble every feature for a frame of (period, issue time) rows.

    ``targets`` must carry ``period_start`` and ``captured_at``. The returned
    frame carries the features only; the outcome is joined by the caller, so
    that a prediction path and a training path cannot diverge in what they see.
    """
    if targets.empty:
        return pd.DataFrame()

    horizon = (targets["period_start"] - targets["captured_at"]) / HOUR

    return pd.concat(
        [
            pd.DataFrame({"horizon_hours": horizon}, index=targets.index),
            calendar_features(targets["period_start"]),
            weekly_reference_features(targets, outcomes),
            latest_observation_features(targets, outcomes, generation),
        ],
        axis=1,
    )


def issue_grid(
    outcomes: pd.DataFrame, horizons: tuple[float, ...] = TRAINING_HORIZONS
) -> pd.DataFrame:
    """Pair every observed period with each hypothetical issue time.

    The result is the training frame's index: one row per period per lead, with
    the outcome attached. Rows sharing a period are not independent, which is
    why validation splits by date rather than by row.
    """
    if outcomes.empty:
        return pd.DataFrame()

    parts = []
    for horizon in horizons:
        part = outcomes[["period_start", "actual"]].copy()
        part["captured_at"] = part["period_start"] - pd.Timedelta(hours=horizon)
        parts.append(part)

    grid = pd.concat(parts, ignore_index=True)
    return grid.sort_values(["period_start", "captured_at"]).reset_index(drop=True)


def training_frame(
    outcomes: pd.DataFrame,
    generation: pd.DataFrame,
    horizons: tuple[float, ...] = TRAINING_HORIZONS,
) -> pd.DataFrame:
    """Features and outcome for every (period, hypothetical issue time) pair."""
    grid = issue_grid(outcomes, horizons)
    if grid.empty:
        return pd.DataFrame()

    features = build_features(grid, outcomes, generation)
    return pd.concat([grid[["period_start", "captured_at", "actual"]], features], axis=1)


FEATURE_COLUMNS = [
    "horizon_hours",
    "sin_day",
    "cos_day",
    "sin_year",
    "cos_year",
    "day_of_week",
    "is_weekend",
    *[f"intensity_lag_{lag}d" for lag in WEEKLY_LAGS],
    "recent_intensity",
    "recent_staleness_hours",
    *[f"recent_{fuel}" for fuel in FUELS],
]
