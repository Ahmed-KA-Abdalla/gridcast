from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from gridcast.parse import FUELS, ParseError, as_records, parse_generation, parse_intensity
from gridcast.schema import (
    ValidationError,
    validate_generation,
    validate_intensity,
)

UTC = dt.UTC
DATA = Path(__file__).parent / "data"
CAPTURED = dt.datetime(2026, 8, 20, 12, 5, tzinfo=UTC)


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture
def forecast_frame() -> pd.DataFrame:
    return parse_intensity(load("intensity_forecast.json"), CAPTURED)


@pytest.fixture
def generation_frame() -> pd.DataFrame:
    return parse_generation(load("generation_current.json"), CAPTURED)


def test_parse_intensity_reads_values_and_periods(forecast_frame):
    assert len(forecast_frame) == 3
    assert forecast_frame["forecast"].tolist() == [266.0, 254.0, 241.0]
    assert forecast_frame["period_start"].iloc[0] == pd.Timestamp("2026-08-20T12:00Z")


def test_parse_intensity_records_capture_time_and_horizon(forecast_frame):
    # The first period began five minutes before capture, so its horizon is negative.
    assert forecast_frame["horizon_hours"].iloc[0] == pytest.approx(-5 / 60)
    assert forecast_frame["horizon_hours"].iloc[2] == pytest.approx(55 / 60)
    assert (forecast_frame["captured_at"] == pd.Timestamp(CAPTURED)).all()


def test_parse_intensity_keeps_a_missing_actual_missing(forecast_frame):
    assert forecast_frame["actual"].isna().all()


def test_parse_intensity_derives_time_features(forecast_frame):
    assert forecast_frame["utc_half_hour"].tolist() == [24, 25, 26]
    # August, so London is one hour ahead of UTC.
    assert forecast_frame["local_hour"].iloc[0] == pytest.approx(13.0)
    assert forecast_frame["day_of_week"].iloc[0] == 3  # a Thursday


def test_parse_intensity_handles_an_empty_payload():
    frame = parse_intensity({"data": []}, CAPTURED)
    assert frame.empty
    assert "horizon_hours" in frame.columns


def test_local_hour_follows_the_clock_across_the_spring_transition():
    payload = {
        "data": [
            {
                "from": "2026-03-29T00:30Z",
                "to": "2026-03-29T01:00Z",
                "intensity": {"forecast": 100, "actual": 98, "index": "low"},
            },
            {
                "from": "2026-03-29T01:00Z",
                "to": "2026-03-29T01:30Z",
                "intensity": {"forecast": 101, "actual": 99, "index": "low"},
            },
        ]
    }
    frame = parse_intensity(payload, dt.datetime(2026, 3, 29, tzinfo=UTC))
    # Clocks go forward at 01:00 UTC, so local time jumps from 00:30 to 02:00
    # while the UTC half-hour index advances by one as usual.
    assert frame["local_hour"].tolist() == [0.5, 2.0]
    assert frame["utc_half_hour"].tolist() == [1, 2]


def test_parse_generation_pivots_fuels_into_columns(generation_frame):
    assert set(FUELS) <= set(generation_frame.columns)
    assert generation_frame["gas"].iloc[0] == pytest.approx(43.6)
    assert generation_frame[list(FUELS)].sum(axis=1).iloc[0] == pytest.approx(100.0)


def test_parse_generation_treats_an_absent_fuel_as_zero():
    payload = {
        "data": [
            {
                "from": "2026-08-20T12:00Z",
                "to": "2026-08-20T12:30Z",
                "generationmix": [
                    {"fuel": "wind", "perc": 60.0},
                    {"fuel": "gas", "perc": 40.0},
                ],
            }
        ]
    }
    frame = parse_generation(payload, CAPTURED)
    assert frame["coal"].iloc[0] == 0.0
    assert not frame[list(FUELS)].isna().any().any()


def test_validate_intensity_accepts_a_well_formed_frame(forecast_frame):
    assert validate_intensity(forecast_frame) is forecast_frame


def test_validate_intensity_rejects_a_gap_in_the_sequence(forecast_frame):
    gapped = forecast_frame.drop(index=1).reset_index(drop=True)
    with pytest.raises(ValidationError, match="missing"):
        validate_intensity(gapped)


def test_validate_intensity_rejects_an_implausible_value(forecast_frame):
    frame = forecast_frame.copy()
    frame.loc[0, "forecast"] = 5000.0
    with pytest.raises(ValidationError, match="outside"):
        validate_intensity(frame)


def test_validate_intensity_can_require_realised_values(forecast_frame):
    with pytest.raises(ValidationError, match="realised"):
        validate_intensity(forecast_frame, require_actual=True)


def test_validate_intensity_rejects_an_empty_frame():
    with pytest.raises(ValidationError, match="empty"):
        validate_intensity(parse_intensity({"data": []}, CAPTURED))


def test_validate_generation_rejects_a_mix_that_does_not_sum(generation_frame):
    frame = generation_frame.copy()
    frame.loc[0, "gas"] = 10.0
    with pytest.raises(ValidationError, match="sum to 100"):
        validate_generation(frame)


def test_validate_generation_tolerates_publication_rounding(generation_frame):
    frame = generation_frame.copy()
    frame.loc[0, "gas"] += 0.4
    assert validate_generation(frame) is frame


def test_as_records_accepts_the_documented_array_form():
    assert as_records({"data": [{"from": "x"}]}) == [{"from": "x"}]


def test_as_records_accepts_the_bare_object_the_live_api_returns():
    # /generation serves a single object for the current period, contrary to
    # the published schema. Iterating it directly yields its keys.
    assert as_records({"data": {"from": "x"}}) == [{"from": "x"}]


def test_as_records_rejects_anything_else():
    with pytest.raises(ParseError, match="expected object or array"):
        as_records({"data": "unexpected"})


def test_parse_generation_handles_the_bare_object_form():
    payload = {
        "data": {
            "from": "2026-08-20T12:00Z",
            "to": "2026-08-20T12:30Z",
            "generationmix": [
                {"fuel": "gas", "perc": 60.0},
                {"fuel": "wind", "perc": 40.0},
            ],
        }
    }
    frame = parse_generation(payload, CAPTURED)
    assert len(frame) == 1
    assert frame["gas"].iloc[0] == pytest.approx(60.0)


def test_parse_intensity_handles_the_bare_object_form():
    payload = {
        "data": {
            "from": "2026-08-20T12:00Z",
            "to": "2026-08-20T12:30Z",
            "intensity": {"forecast": 266, "actual": 263, "index": "moderate"},
        }
    }
    frame = parse_intensity(payload, CAPTURED)
    assert frame["forecast"].tolist() == [266.0]
