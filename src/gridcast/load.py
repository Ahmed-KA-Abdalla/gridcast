"""Assemble the stored snapshots into frames suitable for modelling.

Three records come out of the raw store.

``forecast_record`` is one row per (period, issue time): what the published
forecast said about a period at a given lead time. It is built exclusively from
``forecast_fw48h`` snapshots. This restriction is the point of the module and is
enforced rather than assumed — see ``FORECAST_FAMILIES`` below.

``outcome_record`` is one row per period: the realised intensity, taken from
whichever observation of that period settled last.

``evaluation_frame`` joins the two and computes the published forecast's error.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .parse import parse_generation, parse_intensity
from .storage import DEFAULT_ROOT, iter_snapshots, read_snapshot

#: Backfill kinds carry the window start, e.g. ``intensity_range_20240101``.
WINDOW_SUFFIX = re.compile(r"_\d{8}$")

#: Only forecasts captured at issue time may be used as forecasts. The
#: ``forecast`` field of a historical range response is a revised value: it was
#: produced with information that was not available at the lead time it appears
#: to belong to. Training or scoring on it would leak the outcome into the
#: predictor and would not announce itself in any metric.
FORECAST_FAMILIES = frozenset({"forecast_fw48h"})

#: Realised values are trustworthy from any source that reports them. The
#: ``outcomes_pt24h`` family is the working source: each run reports the past 24
#: hours, so a period is observed by 48 successive runs and survives any
#: plausible run of dropped schedules.
OUTCOME_FAMILIES = frozenset({"intensity", "intensity_range", "outcomes_pt24h"})

GENERATION_FAMILIES = frozenset({"generation", "generation_range"})


def kind_family(kind: str) -> str:
    """Strip the window suffix a backfill kind carries.

    ``intensity_range_20240101`` and ``intensity_range_20240115`` are the same
    family; the suffix exists only to keep filenames distinct within a run.
    """
    return WINDOW_SUFFIX.sub("", kind)


def _concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_family(root: Path, families: frozenset[str], parser) -> pd.DataFrame:
    frames = []
    for path in iter_snapshots(root):
        payload, kind, _endpoint, captured_at = read_snapshot(path)
        if kind_family(kind) in families:
            frames.append(parser(payload, captured_at))
    return _concat(frames)


def forecast_record(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Every published forecast, as issued, one row per period and issue time.

    Rows with a non-positive ``horizon_hours`` are dropped: the first period of
    a forward response is the one already in progress, which is an observation
    of the present rather than a forecast of the future.
    """
    frame = _load_family(root, FORECAST_FAMILIES, parse_intensity)
    if frame.empty:
        return frame

    frame = frame.drop(columns=["actual", "index"])
    frame = frame[frame["horizon_hours"] > 0]
    frame = frame.dropna(subset=["forecast"])

    # Two captures in the same minute would otherwise duplicate a row.
    frame = frame.drop_duplicates(subset=["period_start", "captured_at"], keep="last")
    return frame.sort_values(["period_start", "captured_at"]).reset_index(drop=True)


def outcome_record(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """The realised intensity of each period.

    A period may be observed several times. The API settles a value some minutes
    after the period ends and may revise it afterwards, so the observation with
    the latest capture time is taken as final.
    """
    frame = _load_family(root, OUTCOME_FAMILIES, parse_intensity)
    if frame.empty:
        return frame

    frame = frame.dropna(subset=["actual"])
    frame = frame.sort_values("captured_at")
    frame = frame.drop_duplicates(subset=["period_start"], keep="last")

    columns = [
        "period_start",
        "actual",
        "index",
        "captured_at",
        "utc_half_hour",
        "local_hour",
        "day_of_week",
    ]
    frame = frame[columns].rename(columns={"captured_at": "settled_at"})
    return frame.sort_values("period_start").reset_index(drop=True)


def generation_record(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """The fuel mix of each period, deduplicated on the latest observation."""
    frame = _load_family(root, GENERATION_FAMILIES, parse_generation)
    if frame.empty:
        return frame

    frame = frame.sort_values("captured_at")
    frame = frame.drop_duplicates(subset=["period_start"], keep="last")
    return frame.sort_values("period_start").reset_index(drop=True)


def evaluation_frame(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Published forecasts joined to the outcomes they were forecasting.

    An inner join: a forecast whose period has not yet settled has nothing to be
    scored against, and an outcome nobody forecast contributes nothing. Both are
    expected in normal operation and neither is an error.

    ``error`` is signed as forecast minus outcome, so a positive value means the
    published forecast was too high.
    """
    forecasts = forecast_record(root)
    outcomes = outcome_record(root)
    if forecasts.empty or outcomes.empty:
        return pd.DataFrame()

    merged = forecasts.merge(
        outcomes[["period_start", "actual", "index", "settled_at"]],
        on="period_start",
        how="inner",
        validate="many_to_one",
    )
    merged["error"] = merged["forecast"] - merged["actual"]
    merged["abs_error"] = merged["error"].abs()
    return merged.sort_values(["period_start", "captured_at"]).reset_index(drop=True)


def coverage(root: Path = DEFAULT_ROOT) -> dict[str, object]:
    """Summarise what the store currently holds.

    Reported rather than asserted: gaps in the capture record are expected,
    since scheduled runs are delayed and occasionally dropped.
    """
    outcomes = outcome_record(root)
    forecasts = forecast_record(root)

    summary: dict[str, object] = {
        "outcome_periods": len(outcomes),
        "forecast_rows": len(forecasts),
        "forecast_periods": int(forecasts["period_start"].nunique()) if len(forecasts) else 0,
        "issues": int(forecasts["captured_at"].nunique()) if len(forecasts) else 0,
    }

    if len(outcomes):
        span = outcomes["period_start"]
        expected = int((span.max() - span.min()) / pd.Timedelta(minutes=30)) + 1
        summary["outcome_span"] = (span.min(), span.max())
        summary["outcome_missing"] = expected - len(outcomes)

    if len(forecasts):
        summary["max_horizon_hours"] = float(forecasts["horizon_hours"].max())

    return summary
