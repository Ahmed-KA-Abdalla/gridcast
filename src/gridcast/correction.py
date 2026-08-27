"""A damped-revision correction to the published forecast.

The revision analysis found that within roughly twelve hours the published
forecast retraces most of its own movement: successive revisions are
anticorrelated at about minus a half, and a revision is negatively correlated
with the error that remains after it. Read together, those say the forecast
overshoots — when it revises upwards it tends to end up too high.

If that holds, part of the most recent revision should be subtracted rather than
believed. The corrected forecast is

    corrected = published - damping * most_recent_revision

with one coefficient per lead band. The coefficient that minimises squared error
is available in closed form, being the least-squares slope of the remaining
error on the revision, so no search is needed and nothing is tuned by hand.

The whole exercise is only meaningful out of sample. The coefficient is fitted
on earlier dates and applied to later ones, split by date rather than by row,
because rows sharing a target period are not independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .load import outcome_record
from .revisions import LEAD_BINS, distinct_revisions, revision_paths
from .storage import DEFAULT_ROOT

#: Below this many observations a band's coefficient is not estimated, and the
#: published forecast is passed through unchanged.
MIN_OBSERVATIONS = 30


@dataclass(frozen=True)
class Damping:
    """Coefficients by lead band, with the sample each was fitted on."""

    coefficients: dict[str, float]
    counts: dict[str, int]

    def for_band(self, band: str) -> float:
        return self.coefficients.get(band, 0.0)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "band": list(self.coefficients),
                "damping": [self.coefficients[b] for b in self.coefficients],
                "fitted_on": [self.counts.get(b, 0) for b in self.coefficients],
            }
        )


def revision_frame(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Distinct revisions joined to the outcome, with the error that remained.

    ``remaining_error`` is signed as outcome minus forecast, so a positive value
    means the forecast was too low at that moment.
    """
    paths = distinct_revisions(revision_paths(root))
    outcomes = outcome_record(root)
    if paths.empty or outcomes.empty:
        return pd.DataFrame()

    joined = paths.merge(outcomes[["period_start", "actual"]], on="period_start", how="inner")
    joined = joined.dropna(subset=["revision", "actual"])
    if joined.empty:
        return pd.DataFrame()

    joined["remaining_error"] = joined["actual"] - joined["forecast"]
    joined["band"] = pd.cut(joined["horizon_hours"], bins=list(LEAD_BINS), right=True).astype(str)
    joined["date"] = joined["captured_at"].dt.date
    return joined.reset_index(drop=True)


def fit_damping(frame: pd.DataFrame) -> Damping:
    """Least-squares damping coefficient per lead band.

    Minimising the squared error of ``forecast - damping * revision`` against
    the outcome gives ``-cov(remaining_error, revision) / var(revision)``. A
    positive coefficient means part of the revision should be undone; a negative
    one would mean the forecast under-reacts and the revision should be
    amplified.
    """
    coefficients: dict[str, float] = {}
    counts: dict[str, int] = {}

    if frame.empty:
        return Damping(coefficients, counts)

    for band, part in frame.groupby("band", observed=True):
        revision = part["revision"].to_numpy(dtype=float)
        error = part["remaining_error"].to_numpy(dtype=float)
        counts[band] = int(len(part))

        variance = float(np.var(revision))
        if len(part) < MIN_OBSERVATIONS or variance == 0.0:
            coefficients[band] = 0.0
            continue

        covariance = float(np.cov(error, revision, bias=True)[0, 1])
        coefficients[band] = -covariance / variance

    return Damping(coefficients, counts)


def apply_damping(frame: pd.DataFrame, damping: Damping) -> pd.DataFrame:
    """Add the corrected forecast and both errors to a revision frame."""
    corrected = frame.copy()
    factors = corrected["band"].map(damping.for_band).fillna(0.0)
    corrected["damping"] = factors
    corrected["corrected"] = corrected["forecast"] - factors * corrected["revision"]
    corrected["published_abs_error"] = (corrected["forecast"] - corrected["actual"]).abs()
    corrected["corrected_abs_error"] = (corrected["corrected"] - corrected["actual"]).abs()
    return corrected


def split_by_date(frame: pd.DataFrame, train_fraction: float = 0.6) -> tuple[pd.DataFrame, ...]:
    """Divide into earlier and later dates.

    By date rather than by row: rows sharing a target period are not
    independent, and a random split would put a period's early revisions in
    training and its later ones in test.
    """
    if frame.empty:
        return frame, frame

    dates = np.sort(frame["date"].unique())
    if len(dates) < 2:
        return frame, frame.iloc[0:0]

    cut = max(1, int(len(dates) * train_fraction))
    cut = min(cut, len(dates) - 1)
    boundary = dates[cut]
    return frame[frame["date"] < boundary], frame[frame["date"] >= boundary]


def evaluate_correction(
    root: Path = DEFAULT_ROOT, train_fraction: float = 0.6
) -> tuple[pd.DataFrame, Damping, dict[str, object]]:
    """Fit on earlier dates, score on later ones, and report per band.

    Returns the per-band scores, the fitted coefficients, and a note of the
    split so the result cannot be read without knowing what it rests on.
    """
    frame = revision_frame(root)
    if frame.empty:
        return pd.DataFrame(), Damping({}, {}), {}

    train, test = split_by_date(frame, train_fraction)
    if test.empty:
        return pd.DataFrame(), Damping({}, {}), {"reason": "not enough dates to hold any out"}

    damping = fit_damping(train)
    scored = apply_damping(test, damping)

    summary = scored.groupby("band", observed=True).agg(
        n=("corrected_abs_error", "size"),
        damping=("damping", "first"),
        published_mae=("published_abs_error", "mean"),
        corrected_mae=("corrected_abs_error", "mean"),
    )
    summary["improvement"] = summary["published_mae"] - summary["corrected_mae"]
    summary["improvement_percent"] = 100.0 * summary["improvement"] / summary["published_mae"]

    note = {
        "train_dates": int(train["date"].nunique()),
        "test_dates": int(test["date"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
    return summary.reset_index(), damping, note
