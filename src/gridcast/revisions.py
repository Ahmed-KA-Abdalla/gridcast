"""How the published forecast changes as its target approaches.

Each half-hour period is forecast repeatedly — once per capture, up to
forty-eight hours ahead — and the API keeps no history of those forecasts. The
sequence for a single period is therefore an object this project can examine and
the source cannot reproduce.

Two questions are asked of it.

Does the forecast improve as the target nears? Obvious in principle, but it must
be measured on periods observed at every lead. Pooling all rows compares the
forecast's performance on one set of days at long lead against a different set at
short lead, and the weather differs between them.

Are the revisions unpredictable? A forecast that uses all available information
should revise in a way that cannot be anticipated: the change from one issue to
the next should be uncorrelated with the change before it. Positive correlation
means the forecast adjusts gradually towards news it has already received, so
part of the next revision is predictable from the last. That is a testable
efficiency claim, and it needs no data beyond what has already been captured.

The efficiency question must be asked of the forecast's own revisions, not of
this project's sampling of them. Captures run on a schedule of their own, and
about half of them find the value unchanged since the last. Those repeats insert
zeros into the differenced series, which dilutes any correlation towards nothing
and makes the magnitude untrustworthy. Consecutive identical values are
therefore collapsed before anything is inferred, leaving one entry per distinct
forecast with the interval between them recorded.

Note what repeats do not cause. A correlation of exactly minus a half is the
signature of a different structure: a stable level observed with independent
noise, where consecutive differences share one noise term with opposite sign.
Repeats pull towards zero, not towards minus a half, and the two are tested
separately so the distinction cannot be lost.

Collapsing them yields a quantity in its own right: how often the published
forecast is actually revised, by lead time. The API does not expose it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .load import forecast_record, outcome_record
from .storage import DEFAULT_ROOT

#: Lead-time bands, in hours, for reporting quantities that vary with horizon.
LEAD_BINS = (0, 3, 6, 12, 24, 48)


def revision_paths(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Every forecast issued for every period, ordered by issue time.

    Adds ``revision``, the change from the previous issue for the same period,
    and ``previous_revision``, the one before that. The first issue for a period
    has no revision, since there is nothing it revised.

    This is the captured series, one row per capture, including captures that
    found nothing changed. Use ``distinct_revisions`` for the forecaster's own
    revision sequence; drawing conclusions about efficiency from this frame
    measures the capture schedule.
    """
    forecasts = forecast_record(root)
    if forecasts.empty:
        return pd.DataFrame()

    paths = forecasts.sort_values(["period_start", "captured_at"]).copy()
    grouped = paths.groupby("period_start", sort=False)["forecast"]
    paths["revision"] = grouped.diff()
    paths["previous_revision"] = paths.groupby("period_start", sort=False)["revision"].shift()
    paths["issue_index"] = paths.groupby("period_start", sort=False).cumcount()
    return paths.reset_index(drop=True)


def distinct_revisions(paths: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive identical forecasts, keeping one row per change.

    The first capture of each period is kept as the starting value, with no
    revision attached. Every later row is a capture whose forecast differed from
    the one before it, carrying the size of the change, the hours since the
    previous distinct value, and the lead time at which it happened.

    ``held_captures`` counts how many captures saw the previous value before it
    moved, which is what makes the interval interpretable: a long interval
    observed by many captures is a forecast genuinely left alone, whereas one
    observed by few may only be a gap in the record.
    """
    if paths.empty:
        return pd.DataFrame()

    changed = paths["revision"].ne(0) | paths["revision"].isna()
    distinct = paths[changed].copy()

    run_lengths = (
        paths.assign(_changed=changed)
        .groupby(["period_start", changed.cumsum()], sort=False)
        .size()
        .rename("held_captures")
        .reset_index(drop=True)
    )
    distinct = distinct.reset_index(drop=True)
    distinct["held_captures"] = run_lengths.to_numpy()[: len(distinct)]

    grouped = distinct.groupby("period_start", sort=False)
    distinct["hours_since_previous"] = grouped["captured_at"].diff().dt.total_seconds() / 3600.0
    distinct["previous_revision"] = grouped["revision"].shift()
    distinct["change_index"] = grouped.cumcount()
    return distinct.reset_index(drop=True)


def refresh_cadence(distinct: pd.DataFrame) -> pd.DataFrame:
    """How often the published forecast is actually revised, by lead time.

    Not something the API reports. A short interval means the forecast for that
    horizon is being reworked frequently; a long one means it is settled and
    further captures add nothing.
    """
    if distinct.empty:
        return pd.DataFrame()

    usable = distinct.dropna(subset=["hours_since_previous"])
    if usable.empty:
        return pd.DataFrame()

    buckets = pd.cut(usable["horizon_hours"], bins=list(LEAD_BINS), right=True)
    scored = usable.groupby(buckets, observed=True).agg(
        changes=("revision", "size"),
        median_hours_between=("hours_since_previous", "median"),
        median_captures_held=("held_captures", "median"),
        mean_abs_change=("revision", lambda r: float(r.abs().mean())),
    )
    return scored.reset_index()


def path_summary(paths: pd.DataFrame) -> pd.DataFrame:
    """One row per period: how often it was forecast and how far it moved.

    ``total_movement`` is the sum of absolute revisions and ``net_movement`` the
    difference between the last forecast and the first. A total far exceeding
    the net means the forecast wandered rather than converged.
    """
    if paths.empty:
        return pd.DataFrame()

    grouped = paths.groupby("period_start")
    summary = pd.DataFrame(
        {
            "issues": grouped.size(),
            "first_forecast": grouped["forecast"].first(),
            "last_forecast": grouped["forecast"].last(),
            "longest_lead_hours": grouped["horizon_hours"].max(),
            "shortest_lead_hours": grouped["horizon_hours"].min(),
            "total_movement": grouped["revision"].apply(lambda r: r.abs().sum()),
        }
    )
    summary["net_movement"] = summary["last_forecast"] - summary["first_forecast"]
    summary["wander"] = summary["total_movement"] - summary["net_movement"].abs()
    return summary.reset_index()


def revision_autocorrelation(paths: pd.DataFrame) -> pd.DataFrame:
    """Correlation between a revision and the one before it.

    Near zero is what an efficient forecast produces: each revision reflects
    genuinely new information and says nothing about the next. A positive value
    means the forecast moves gradually in one direction, so some of the coming
    revision is already implied by the last one. A negative value means it
    overshoots and corrects.

    Reported overall and by lead time, because a forecast may be efficient at
    short range and sluggish at long range.

    Pass the frame from ``distinct_revisions``. Passing the captured series
    instead measures the capture schedule as much as the forecaster: roughly
    half of captures find the value unchanged, and those zeros dilute whatever
    structure is present.
    """
    if paths.empty:
        return pd.DataFrame()

    usable = paths.dropna(subset=["revision", "previous_revision"])
    if usable.empty:
        return pd.DataFrame()

    # A band whose revisions are all identical has no variance, so the
    # correlation is genuinely undefined and NaN is the answer wanted. NumPy
    # warns about the division that produces it; the warning is expected here
    # and silencing it deliberately keeps real ones visible.
    with np.errstate(invalid="ignore", divide="ignore"):
        records = [
            {
                "lead": "all",
                "n": int(len(usable)),
                "autocorrelation": float(usable["revision"].corr(usable["previous_revision"])),
                "mean_abs_revision": float(usable["revision"].abs().mean()),
            }
        ]

        buckets = pd.cut(usable["horizon_hours"], bins=list(LEAD_BINS), right=True)
        for bucket, part in usable.groupby(buckets, observed=True):
            if len(part) < 3:
                continue
            records.append(
                {
                    "lead": str(bucket),
                    "n": int(len(part)),
                    "autocorrelation": float(part["revision"].corr(part["previous_revision"])),
                    "mean_abs_revision": float(part["revision"].abs().mean()),
                }
            )

    return pd.DataFrame(records)


def error_by_lead(root: Path = DEFAULT_ROOT, matched: bool = True) -> pd.DataFrame:
    """Absolute forecast error against lead time.

    With ``matched`` set, only periods forecast in every lead band are counted,
    so the bands describe the same days and the comparison isolates lead time.
    Without it the bands cover different periods and a difference between them
    may be weather rather than horizon — the error that made an earlier version
    of this project's lead-time table non-monotonic.
    """
    forecasts = forecast_record(root)
    outcomes = outcome_record(root)
    if forecasts.empty or outcomes.empty:
        return pd.DataFrame()

    joined = forecasts.merge(outcomes[["period_start", "actual"]], on="period_start", how="inner")
    if joined.empty:
        return pd.DataFrame()

    joined["abs_error"] = (joined["forecast"] - joined["actual"]).abs()
    joined["bucket"] = pd.cut(joined["horizon_hours"], bins=list(LEAD_BINS), right=True)
    joined = joined.dropna(subset=["bucket"])

    if matched:
        counts = joined.groupby("period_start")["bucket"].nunique()
        complete = counts[counts == joined["bucket"].nunique()].index
        joined = joined[joined["period_start"].isin(complete)]
        if joined.empty:
            return pd.DataFrame()

    scored = joined.groupby("bucket", observed=True).agg(
        n=("abs_error", "size"),
        periods=("period_start", "nunique"),
        mae=("abs_error", "mean"),
        bias=("forecast", lambda s: float(np.mean(s - joined.loc[s.index, "actual"]))),
    )
    return scored.reset_index()


def revision_predicts_error(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Whether the last revision says anything about the error that remains.

    If a forecast under-reacts, a revision upwards implies the next one will
    also be upwards, and so the current forecast is still too low. That would
    show as a positive correlation between a revision and the error remaining
    after it, and it would be directly exploitable.

    Computed over distinct revisions, since a capture that changed nothing
    carries no information about what remains.
    """
    paths = distinct_revisions(revision_paths(root))
    outcomes = outcome_record(root)
    if paths.empty or outcomes.empty:
        return pd.DataFrame()

    joined = paths.merge(outcomes[["period_start", "actual"]], on="period_start", how="inner")
    joined = joined.dropna(subset=["revision"])
    if joined.empty:
        return pd.DataFrame()

    # Signed, so that a positive value means the forecast is still too low.
    joined["remaining_error"] = joined["actual"] - joined["forecast"]

    buckets = pd.cut(joined["horizon_hours"], bins=list(LEAD_BINS), right=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        records = [
            {
                "lead": "all",
                "n": int(len(joined)),
                "correlation": float(joined["revision"].corr(joined["remaining_error"])),
            }
        ]
        for bucket, part in joined.groupby(buckets, observed=True):
            if len(part) < 3:
                continue
            records.append(
                {
                    "lead": str(bucket),
                    "n": int(len(part)),
                    "correlation": float(part["revision"].corr(part["remaining_error"])),
                }
            )
    return pd.DataFrame(records)
