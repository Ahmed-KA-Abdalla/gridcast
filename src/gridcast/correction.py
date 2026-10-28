"""A damped-revision correction to the published forecast.

Within roughly twelve hours the published forecast retraces part of its own
movement: successive revisions are anticorrelated, and a revision is negatively
correlated with the error that remains after it. Part of the most recent
revision should therefore be subtracted rather than believed. The corrected
forecast is

    corrected = published - damping * most_recent_revision

with the coefficient chosen by lead band and by the direction of the revision.
Upward and downward revisions are damped separately because they behave
differently: a downward revision is mostly retraced, an upward one mostly is
not. A single coefficient splits the difference and gets both wrong, which is
what made the symmetric correction fail once autumn brought a run of large
upward revisions.

The coefficient is the least-squares fit of that model, which has no intercept:
``-sum(error * revision) / sum(revision ** 2)``. An earlier version used the
centred slope, ``-cov / var``, which is the fit of a model *with* an intercept,
and then applied it without one. For revisions that average near zero the two
barely differ. Fitted separately to upward and downward revisions they differ a
great deal, since each subset is one-signed by construction, and the centred
version produced coefficients above one that no model actually applied would
choose. Models that keep the intercept and apply it were also tried; intercepts
are level corrections, and they failed out of sample even between August and
September.

The whole exercise is only meaningful out of sample, and one held-out split has
not been enough. The symmetric correction at short lead was significant on one
split and significantly harmful on the next. ``rolling_evaluation`` scores
several consecutive held-out blocks, each with a coefficient fitted on every
earlier date, so a result must hold repeatedly rather than once, and a
coefficient that swings between refits shows up within a single run.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .load import outcome_record
from .revisions import LEAD_BINS, distinct_revisions, revision_paths
from .storage import DEFAULT_ROOT

#: Below this many observations a coefficient is not estimated, and the
#: published forecast is passed through unchanged. Applies to each direction of
#: an asymmetric fit separately.
MIN_OBSERVATIONS = 30

#: Resamples used for the interval around an improvement. Enough for a
#: percentile interval at the 95% level to be stable to a few hundredths.
BOOTSTRAP_RESAMPLES = 2000

#: The models that can be fitted. ``symmetric`` damps every revision by one
#: coefficient per band; ``asymmetric`` fits upward and downward separately.
MODELS = ("symmetric", "asymmetric")
DEFAULT_MODEL = "asymmetric"

#: Held-out blocks in a rolling evaluation, and the share of rows the first
#: block's fit is given. Five blocks over the last sixty per cent of the record
#: is enough to see whether an improvement repeats, while leaving each block
#: several days of data.
FOLDS = 5
FIRST_FIT_FRACTION = 0.4


@dataclass(frozen=True)
class BandDamping:
    """The coefficients for one lead band: one per direction of revision.

    A symmetric correction is the special case of equal coefficients, so one
    type describes both models and a record written under either reads back the
    same way.
    """

    up: float = 0.0
    down: float = 0.0

    @classmethod
    def symmetric(cls, value: float) -> BandDamping:
        return cls(float(value), float(value))

    @classmethod
    def coerce(cls, value: BandDamping | dict | float) -> BandDamping:
        """From a stored or hand-written value. A bare number is symmetric."""
        if isinstance(value, BandDamping):
            return value
        if isinstance(value, dict):
            return cls(float(value.get("up", 0.0)), float(value.get("down", 0.0)))
        return cls.symmetric(float(value))

    @property
    def is_zero(self) -> bool:
        return self.up == 0.0 and self.down == 0.0

    def to_dict(self) -> dict[str, float]:
        return {"up": self.up, "down": self.down}

    def __str__(self) -> str:
        return f"up {self.up:.2f}, down {self.down:.2f}"


@dataclass(frozen=True)
class Damping:
    """Coefficients by lead band, with the sample each was fitted on."""

    coefficients: dict[str, BandDamping]
    counts: dict[str, int]
    model: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        # Accept bare numbers, as an older record or a test may supply them.
        coerced = {band: BandDamping.coerce(value) for band, value in self.coefficients.items()}
        object.__setattr__(self, "coefficients", coerced)

    def for_band(self, band: str) -> BandDamping:
        return self.coefficients.get(band, BandDamping())

    def factors(self, bands: pd.Series, revisions: pd.Series) -> np.ndarray:
        """The coefficient applying to each row, chosen by band and direction.

        A missing revision — the first capture of a period — is treated as
        downward, which does not matter: there is nothing to multiply.
        """
        revision = pd.to_numeric(revisions, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        up = bands.map(lambda band: self.for_band(band).up).to_numpy(dtype=float)
        down = bands.map(lambda band: self.for_band(band).down).to_numpy(dtype=float)
        return np.where(revision > 0, up, down)

    def to_frame(self) -> pd.DataFrame:
        bands = list(self.coefficients)
        return pd.DataFrame(
            {
                "band": bands,
                "damping_up": [self.coefficients[b].up for b in bands],
                "damping_down": [self.coefficients[b].down for b in bands],
                "fitted_on": [self.counts.get(b, 0) for b in bands],
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


def _through_origin(revision: np.ndarray, error: np.ndarray) -> float:
    """Least-squares damping for ``corrected = forecast - damping * revision``.

    Minimising ``sum((error + damping * revision) ** 2)`` gives
    ``-sum(error * revision) / sum(revision ** 2)``. No intercept, because the
    correction applies none.

    Revisions that never vary are left alone. Through the origin the slope is
    still defined for them, but it is then nothing but the mean error divided
    by a constant — a bias correction labelled as damping.
    """
    if len(revision) < MIN_OBSERVATIONS or float(np.var(revision)) == 0.0:
        return 0.0
    return -float(error @ revision) / float(revision @ revision)


def fit_damping(frame: pd.DataFrame, model: str = DEFAULT_MODEL) -> Damping:
    """Least-squares damping per lead band, under the chosen model.

    A positive coefficient means part of the revision should be undone; a
    negative one would mean the forecast under-reacts and the revision should be
    amplified.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {MODELS}")

    coefficients: dict[str, BandDamping] = {}
    counts: dict[str, int] = {}
    if frame.empty:
        return Damping(coefficients, counts, model)

    for band, part in frame.groupby("band", observed=True):
        revision = part["revision"].to_numpy(dtype=float)
        error = part["remaining_error"].to_numpy(dtype=float)
        counts[band] = int(len(part))

        if model == "symmetric":
            coefficients[band] = BandDamping.symmetric(_through_origin(revision, error))
            continue

        upward = revision > 0
        coefficients[band] = BandDamping(
            up=_through_origin(revision[upward], error[upward]),
            down=_through_origin(revision[~upward], error[~upward]),
        )

    return Damping(coefficients, counts, model)


def apply_damping(frame: pd.DataFrame, damping: Damping) -> pd.DataFrame:
    """Add the corrected forecast and both errors to a revision frame."""
    corrected = frame.copy()
    factors = damping.factors(corrected["band"], corrected["revision"])
    corrected["damping"] = factors
    corrected["corrected"] = corrected["forecast"] - factors * corrected["revision"]
    corrected["published_abs_error"] = (corrected["forecast"] - corrected["actual"]).abs()
    corrected["corrected_abs_error"] = (corrected["corrected"] - corrected["actual"]).abs()
    return corrected


def _share_before(frame: pd.DataFrame) -> tuple[list[dt.date], np.ndarray]:
    """Sorted dates, and the share of rows falling strictly before each."""
    counts = frame.groupby("date").size().sort_index()
    before = (counts.cumsum() - counts) / counts.sum()
    return list(counts.index), before.to_numpy(dtype=float)


def split_by_date(frame: pd.DataFrame, train_fraction: float = 0.6) -> tuple[pd.DataFrame, ...]:
    """Divide into earlier and later dates, balanced by rows rather than by date.

    The boundary is a date, because rows sharing a target period are not
    independent and a row-wise split would put a period's early revisions in
    training and its later ones in test.

    Which date, though, is chosen so that the training side holds about
    ``train_fraction`` of the rows. Taking the date at a fixed position instead
    assumes every day contributes equally, and this record's days do not: when
    capture fell from around 38 runs a day to 2, sixty per cent of the dates
    became ninety-one per cent of the rows and the held-out half was starved to
    the point where nothing could be detected in it.
    """
    if frame.empty:
        return frame, frame

    dates, before = _share_before(frame)
    if len(dates) < 2:
        return frame, frame.iloc[0:0]

    # Every candidate but the first, so neither side is ever empty.
    boundary = 1 + int(np.abs(before[1:] - train_fraction).argmin())
    return frame[frame["date"] < dates[boundary]], frame[frame["date"] >= dates[boundary]]


def rolling_blocks(
    frame: pd.DataFrame,
    folds: int = FOLDS,
    first_fit_fraction: float = FIRST_FIT_FRACTION,
) -> list[tuple[dt.date, dt.date]]:
    """Consecutive held-out blocks, as inclusive (first, last) date pairs.

    The record after ``first_fit_fraction`` of its rows is divided into blocks
    of roughly equal rows, on date boundaries for the reason ``split_by_date``
    gives. A short record yields fewer blocks than asked for rather than empty
    ones.
    """
    if frame.empty:
        return []

    dates, before = _share_before(frame)
    if len(dates) < 2:
        return []

    targets = first_fit_fraction + (1.0 - first_fit_fraction) * np.arange(folds) / folds
    # Index 0 would leave the first block nothing to be fitted on.
    starts = sorted({1 + int(np.abs(before[1:] - target).argmin()) for target in targets})
    ends = [*starts[1:], len(dates)]
    return [(dates[start], dates[end - 1]) for start, end in zip(starts, ends, strict=True)]


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
    evaluation splits by date.
    """
    if scored.empty:
        return {}

    difference = (scored["published_abs_error"] - scored["corrected_abs_error"]).to_numpy(float)
    grouped = pd.DataFrame({"period": scored["period_start"].to_numpy(), "d": difference})
    per_period = grouped.groupby("period")["d"].agg(["sum", "size"])
    if len(per_period) < 2:
        return {}

    sums = per_period["sum"].to_numpy(dtype=float)
    sizes = per_period["size"].to_numpy(dtype=float)
    count = len(sums)
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)

    # A resample's mean over rows is the drawn periods' summed difference over
    # their summed size, which is what concatenating them would give. Drawn in
    # chunks so a long record does not need one very large index array.
    chunk = max(1, 2_000_000 // max(count, 1))
    for offset in range(0, resamples, chunk):
        size = min(chunk, resamples - offset)
        drawn = rng.integers(0, count, size=(size, count))
        means[offset : offset + size] = sums[drawn].sum(axis=1) / sizes[drawn].sum(axis=1)

    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "improvement_low": float(low),
        "improvement_high": float(high),
        # The share of resamples in which the correction made matters worse.
        "worse_fraction": float((means <= 0).mean()),
        "periods": int(count),
    }


def rolling_evaluation(
    frame: pd.DataFrame,
    model: str = DEFAULT_MODEL,
    folds: int = FOLDS,
    first_fit_fraction: float = FIRST_FIT_FRACTION,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[pd.DataFrame, Damping, pd.DataFrame, dict[str, object]]:
    """Score the correction on consecutive held-out blocks.

    Each block is corrected with coefficients fitted on every earlier date, the
    way the correction would actually be used: fitted on the past, applied to
    what comes next. Target periods scored in a block are purged from its fit,
    since a period's early revisions can fall on the date before the boundary
    and its outcome would then inform the coefficient that is scored on it.

    Returns four things. A summary per band, pooling every held-out row, with a
    paired bootstrap interval and two measures of consistency: how many blocks
    the correction helped in, and how far each coefficient moved between the
    blocks' fits. The coefficients fitted on the whole record, which are the
    ones a promotion would apply going forward. The per-block results. And a
    note of what the evaluation rests on.
    """
    empty = (pd.DataFrame(), Damping({}, {}, model), pd.DataFrame())
    if frame.empty:
        return (*empty, {"reason": "not enough captured revisions to fit a correction"})

    blocks = rolling_blocks(frame, folds, first_fit_fraction)
    if not blocks:
        return (*empty, {"reason": "not enough dates to hold any out"})

    scored_parts = []
    fold_rows = []
    for index, (first, last) in enumerate(blocks):
        held = frame[(frame["date"] >= first) & (frame["date"] <= last)]
        fit = frame[frame["date"] < first]
        fit = fit[~fit["period_start"].isin(held["period_start"].unique())]

        damping = fit_damping(fit, model)
        scored = apply_damping(held, damping).assign(fold=index)
        scored_parts.append(scored)

        for band, part in scored.groupby("band", observed=True):
            coefficient = damping.for_band(band)
            gain = part["published_abs_error"] - part["corrected_abs_error"]
            fold_rows.append(
                {
                    "fold": index,
                    "first": first,
                    "last": last,
                    "band": band,
                    "n": int(len(part)),
                    "improvement": float(gain.mean()),
                    "damping_up": coefficient.up,
                    "damping_down": coefficient.down,
                }
            )

    pooled = pd.concat(scored_parts, ignore_index=True)
    per_fold = pd.DataFrame(fold_rows)
    final = fit_damping(frame, model)

    records = []
    for band, part in pooled.groupby("band", observed=True):
        coefficient = final.for_band(band)
        entry: dict[str, object] = {
            "band": band,
            "n": int(len(part)),
            "damping_up": coefficient.up,
            "damping_down": coefficient.down,
            "published_mae": float(part["published_abs_error"].mean()),
            "corrected_mae": float(part["corrected_abs_error"].mean()),
        }
        entry["improvement"] = entry["published_mae"] - entry["corrected_mae"]
        entry.update(bootstrap_improvement(part, resamples=resamples))

        # Consistency across blocks, counting only blocks where the band had
        # enough rows for its result to mean something.
        counted = per_fold[(per_fold["band"] == band) & (per_fold["n"] >= MIN_OBSERVATIONS)]
        entry["folds"] = int(len(counted))
        entry["folds_positive"] = int((counted["improvement"] > 0).sum())
        if len(counted) >= 2:
            entry["up_range"] = float(counted["damping_up"].max() - counted["damping_up"].min())
            entry["down_range"] = float(
                counted["damping_down"].max() - counted["damping_down"].min()
            )
        else:
            entry["up_range"] = entry["down_range"] = float("nan")
        records.append(entry)

    summary = pd.DataFrame(records)
    summary["significant"] = summary.get(
        "improvement_low", pd.Series(np.nan, index=summary.index)
    ).gt(0)

    note = {
        "model": model,
        "folds": len(blocks),
        "first_held_out": blocks[0][0],
        "last_held_out": blocks[-1][1],
        "first_fit_dates": int(frame.loc[frame["date"] < blocks[0][0], "date"].nunique()),
        "held_out_rows": int(len(pooled)),
        "total_rows": int(len(frame)),
        "resamples": resamples,
    }
    return summary, final, per_fold, note


def evaluate_rolling(
    root: Path = DEFAULT_ROOT,
    model: str = DEFAULT_MODEL,
    folds: int = FOLDS,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[pd.DataFrame, Damping, pd.DataFrame, dict[str, object]]:
    """``rolling_evaluation`` on the stored record."""
    return rolling_evaluation(revision_frame(root), model, folds=folds, resamples=resamples)


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
    revision = frame["revision"].fillna(0.0)
    frame["forecast"] = frame["forecast"] - damping.factors(frame["band"], revision) * revision
    return frame.drop(columns=["revision", "previous_revision", "issue_index", "band"])
