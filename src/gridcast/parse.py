"""Convert API payloads into tidy frames.

Two conventions are enforced here.

First, every row carries ``captured_at``: the moment the payload was fetched.
Without it a forecast cannot be distinguished from a later revision of itself,
and the horizon at which it was issued is unrecoverable.

Second, half-hour positions are labelled ``utc_half_hour`` (0-47 from midnight
UTC) rather than settlement period. GB settlement periods are numbered against
the local clock day, which has 46 or 50 of them at the daylight-saving
transitions. The two coincide for most of the year and diverge exactly when the
grid behaves unusually, so conflating them would corrupt the periods that matter
most.
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from .client import parse_timestamp

LONDON = ZoneInfo("Europe/London")

FUELS = (
    "biomass",
    "coal",
    "gas",
    "hydro",
    "imports",
    "nuclear",
    "other",
    "solar",
    "wind",
)

INTENSITY_COLUMNS = [
    "period_start",
    "period_end",
    "captured_at",
    "horizon_hours",
    "utc_half_hour",
    "local_hour",
    "day_of_week",
    "forecast",
    "actual",
    "index",
]


class ParseError(ValueError):
    """Raised when a payload does not have the structure the parsers expect."""


def as_records(payload: dict) -> list[dict]:
    """Return the ``data`` field as a list of period records.

    The published schema says ``data`` is always an array. The live
    ``/generation`` endpoint returns a bare object for the current period
    instead. Iterating that object yields its keys, so the deviation surfaces as
    a confusing type error deep in a parser rather than as a clear failure here.
    """
    data = payload.get("data")
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ParseError(f"data field is {type(data).__name__}, expected object or array")


def _time_features(frame: pd.DataFrame) -> pd.DataFrame:
    local = frame["period_start"].dt.tz_convert(LONDON)
    frame["utc_half_hour"] = (
        frame["period_start"].dt.hour * 2 + frame["period_start"].dt.minute // 30
    )
    frame["local_hour"] = local.dt.hour + local.dt.minute / 60
    frame["day_of_week"] = local.dt.dayofweek
    return frame


def parse_intensity(payload: dict, captured_at: dt.datetime) -> pd.DataFrame:
    """Flatten an intensity payload into one row per settlement period.

    ``horizon_hours`` is the lead time of the row relative to ``captured_at``:
    positive for a forecast of a future period, negative for an observation of a
    past one.
    """
    records = []
    for item in as_records(payload):
        intensity = item.get("intensity") or {}
        records.append(
            {
                "period_start": parse_timestamp(item["from"]),
                "period_end": parse_timestamp(item["to"]),
                "forecast": intensity.get("forecast"),
                "actual": intensity.get("actual"),
                "index": intensity.get("index"),
            }
        )

    frame = pd.DataFrame.from_records(
        records, columns=["period_start", "period_end", "forecast", "actual", "index"]
    )
    if frame.empty:
        return pd.DataFrame(columns=INTENSITY_COLUMNS)

    frame["period_start"] = pd.to_datetime(frame["period_start"], utc=True)
    frame["period_end"] = pd.to_datetime(frame["period_end"], utc=True)
    frame["captured_at"] = pd.Timestamp(captured_at).tz_convert("UTC")
    frame["horizon_hours"] = (
        frame["period_start"] - frame["captured_at"]
    ).dt.total_seconds() / 3600.0
    frame = _time_features(frame)

    for column in ("forecast", "actual"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame[INTENSITY_COLUMNS].sort_values("period_start").reset_index(drop=True)


def parse_generation(payload: dict, captured_at: dt.datetime) -> pd.DataFrame:
    """Flatten a generation payload, one row per period and one column per fuel.

    Fuels absent from a payload are filled with zero rather than left missing:
    the API omits a fuel when its share is nil, and a missing share is not the
    same kind of fact as an unknown one.
    """
    records = []
    for item in as_records(payload):
        row: dict[str, object] = {
            "period_start": parse_timestamp(item["from"]),
            "period_end": parse_timestamp(item["to"]),
        }
        for entry in item.get("generationmix", []):
            row[entry["fuel"]] = entry["perc"]
        records.append(row)

    columns = ["period_start", "period_end", *FUELS]
    frame = pd.DataFrame.from_records(records, columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=["period_start", "period_end", "captured_at", *FUELS])

    frame["period_start"] = pd.to_datetime(frame["period_start"], utc=True)
    frame["period_end"] = pd.to_datetime(frame["period_end"], utc=True)
    frame["captured_at"] = pd.Timestamp(captured_at).tz_convert("UTC")

    for fuel in FUELS:
        frame[fuel] = pd.to_numeric(frame[fuel], errors="coerce").fillna(0.0)

    ordered = ["period_start", "period_end", "captured_at", *FUELS]
    return frame[ordered].sort_values("period_start").reset_index(drop=True)
