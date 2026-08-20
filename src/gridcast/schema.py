"""Validation applied to parsed frames before they are used or stored.

Every check here corresponds to an assumption made downstream. Failures are
collected rather than raised one at a time, so a malformed payload reports
everything wrong with it in a single message.
"""

from __future__ import annotations

import pandas as pd

from .parse import FUELS

HALF_HOUR = pd.Timedelta(minutes=30)

#: gCO2/kWh. The GB system has never approached either bound; a value outside
#: them indicates a unit change or a corrupted payload, not a remarkable day.
INTENSITY_BOUNDS = (0.0, 1000.0)

#: Fuel shares are rounded to one decimal place before publication, so nine
#: fuels admit a rounding error of up to 0.45 in the sum.
MIX_TOLERANCE = 1.0


class ValidationError(ValueError):
    """Raised when a frame violates an assumption the pipeline relies on."""


def _check_grid(frame: pd.DataFrame, problems: list[str]) -> None:
    starts = frame["period_start"]
    if starts.duplicated().any():
        problems.append(f"{int(starts.duplicated().sum())} duplicated period_start values")
    if not starts.is_monotonic_increasing:
        problems.append("period_start is not sorted")

    lengths = frame["period_end"] - frame["period_start"]
    if not (lengths == HALF_HOUR).all():
        problems.append("some periods are not thirty minutes long")

    gaps = starts.diff().dropna()
    if len(gaps) and not (gaps == HALF_HOUR).all():
        missing = int(((gaps / HALF_HOUR) - 1).sum())
        problems.append(f"{missing} half-hour periods missing from the sequence")


def validate_intensity(frame: pd.DataFrame, *, require_actual: bool = False) -> pd.DataFrame:
    """Validate a parsed intensity frame and return it unchanged.

    ``require_actual`` should be set when the frame is meant to be historical:
    forward forecasts legitimately have no realised value yet.
    """
    problems: list[str] = []
    if frame.empty:
        raise ValidationError("intensity frame is empty")

    _check_grid(frame, problems)

    low, high = INTENSITY_BOUNDS
    for column in ("forecast", "actual"):
        values = frame[column].dropna()
        if len(values) and ((values < low) | (values > high)).any():
            problems.append(f"{column} has values outside [{low:.0f}, {high:.0f}] gCO2/kWh")

    if frame["forecast"].isna().all():
        problems.append("no forecast values present")

    if require_actual and frame["actual"].isna().any():
        problems.append(f"{int(frame['actual'].isna().sum())} periods lack a realised value")

    if problems:
        raise ValidationError("; ".join(problems))
    return frame


def validate_generation(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a parsed generation frame and return it unchanged."""
    problems: list[str] = []
    if frame.empty:
        raise ValidationError("generation frame is empty")

    _check_grid(frame, problems)

    shares = frame[list(FUELS)]
    if (shares < 0).any().any() or (shares > 100).any().any():
        problems.append("fuel shares outside [0, 100] per cent")

    totals = shares.sum(axis=1)
    off = (totals - 100.0).abs() > MIX_TOLERANCE
    if off.any():
        worst = (totals - 100.0).abs().max()
        problems.append(
            f"{int(off.sum())} periods where the mix does not sum to 100 per cent "
            f"(worst deviation {worst:.2f})"
        )

    if problems:
        raise ValidationError("; ".join(problems))
    return frame
