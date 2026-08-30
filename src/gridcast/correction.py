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

#: Resamples used for the interval around an improvement. Enough for a
#: percentile interval at the 95% level to be stable to a few hundredths.
BOOTSTRAP_RESAMPLES = 2000


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
    """Divide into earlier and later dates, balanced by rows rather than by date.

    The boundary is still a date, because rows sharing a target period are not
    independent and a row-wise split would put a period's early revisions in
    training and its later ones in test.

    Which date, though, is chosen so that the training side holds about
    ``train_fraction`` of the rows. Taking the date at a fixed position instead
    assumes every day contributes equally, and this record's days do not: when
    capture fell from around 38 runs a day to 2, sixty per cent of the dates
    became ninety-one per cent of the rows and the held-out half was starved to
    the point where nothing could be detected in it. A date-position split makes
    every downstream result depend on the capture schedule.
    """
    if frame.empty:
        return frame, frame

    counts = frame.groupby("date").size().sort_index()
    if len(counts) < 2:
        return frame, frame.iloc[0:0]

    # Rows on the training side for each candidate boundary, as a share of all.
    shares = counts.cumsum() / counts.sum()
    # Every candidate but the last, so the test side is never empty.
    candidates = shares.iloc[:-1]
    boundary_index = int((candidates - train_fraction).abs().to_numpy().argmin())
    boundary = counts.index[boundary_index + 1]

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


def bootstrap_improvement(
    scored: pd.DataFrame,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = 0,
) -> dict[str, float]:
    """A percentile interval for the improvement, resampled by target period.

    Paired: each resample takes the same rows for both forecasts, so the
    interval describes the difference between them rather than the variability
    of either. Without pairing the two errors would be resampled independently
    and the interval would be far too wide.

    Resampled by target period rather than by row. Several revisions of the same
    period appear in the frame and their errors move together, so treating rows
    as independent would understate the interval — the same reason the
    train-test split is by date.
    """
    if scored.empty:
        return {}

    difference = scored["published_abs_error"] - scored["corrected_abs_error"]
    clusters = scored["period_start"].to_numpy()
    unique = np.unique(clusters)
    if len(unique) < 2:
        return {}

    grouped = {period: difference[clusters == period].to_numpy() for period in unique}
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)

    for index in range(resamples):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        means[index] = np.concatenate([grouped[period] for period in drawn]).mean()

    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "improvement_low": float(low),
        "improvement_high": float(high),
        # The share of resamples in which the correction made matters worse.
        "worse_fraction": float((means <= 0).mean()),
        "periods": int(len(unique)),
    }


def evaluate_with_intervals(
    root: Path = DEFAULT_ROOT,
    train_fraction: float = 0.6,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[pd.DataFrame, Damping, dict[str, object]]:
    """As ``evaluate_correction``, with a confidence interval per band.

    An improvement without an interval cannot be read: a gain of 0.05 gCO2/kWh
    on a few hundred rows is indistinguishable from none, and nothing in the
    point estimate says so.
    """
    frame = revision_frame(root)
    if frame.empty:
        return pd.DataFrame(), Damping({}, {}), {}

    train, test = split_by_date(frame, train_fraction)
    if test.empty:
        return pd.DataFrame(), Damping({}, {}), {"reason": "not enough dates to hold any out"}

    damping = fit_damping(train)
    scored = apply_damping(test, damping)

    records = []
    for band, part in scored.groupby("band", observed=True):
        entry = {
            "band": band,
            "n": int(len(part)),
            "damping": float(part["damping"].iloc[0]),
            "published_mae": float(part["published_abs_error"].mean()),
            "corrected_mae": float(part["corrected_abs_error"].mean()),
        }
        entry["improvement"] = entry["published_mae"] - entry["corrected_mae"]
        entry.update(bootstrap_improvement(part, resamples=resamples))
        records.append(entry)

    summary = pd.DataFrame(records)
    if not summary.empty:
        summary["significant"] = (
            summary.get("improvement_low", pd.Series(np.nan, index=summary.index)) > 0
        )

    note = {
        "train_dates": int(train["date"].nunique()),
        "test_dates": int(test["date"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "resamples": resamples,
    }
    return summary, damping, note


def corrected_forecast_record(
    root: Path = DEFAULT_ROOT, damping: Damping | None = None
) -> pd.DataFrame:
    """The forecast record with the damping applied, ready to be scheduled on.

    Returned in the same shape as ``load.forecast_record`` so that a corrected
    forecast and the published one go through one scoring path. Rows with no
    revision — the first capture of a period, and captures that changed nothing
    — are passed through unchanged, since there is nothing to damp.

    ``damping`` defaults to whatever the gate has promoted. Applying an
    unpromoted coefficient would be scoring a correction the project does not
    claim.
    """
    from .gate import load_record  # imported here to keep the module acyclic

    paths = revision_paths(root)
    if paths.empty:
        return paths

    if damping is None:
        promoted = load_record()
        damping = Damping(promoted, dict.fromkeys(promoted, 0))

    frame = paths.copy()
    frame["band"] = pd.cut(frame["horizon_hours"], bins=list(LEAD_BINS), right=True).astype(str)
    factors = frame["band"].map(damping.for_band).fillna(0.0)
    frame["forecast"] = frame["forecast"] - factors * frame["revision"].fillna(0.0)
    return frame.drop(columns=["revision", "previous_revision", "issue_index", "band"])
