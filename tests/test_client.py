from __future__ import annotations

import datetime as dt

import pytest
import responses

from gridcast.client import (
    BASE_URL,
    CarbonIntensityClient,
    CarbonIntensityError,
    format_timestamp,
    parse_timestamp,
    window_range,
)

UTC = dt.UTC


def test_format_timestamp_assumes_utc_when_naive():
    naive = dt.datetime(2026, 8, 20, 12, 35)
    assert format_timestamp(naive) == "2026-08-20T12:35Z"


def test_format_timestamp_converts_from_other_zones():
    aware = dt.datetime(2026, 8, 20, 14, 35, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    assert format_timestamp(aware) == "2026-08-20T12:35Z"


def test_parse_timestamp_round_trips():
    text = "2026-08-20T12:30Z"
    assert format_timestamp(parse_timestamp(text)) == text


def test_window_range_respects_the_fourteen_day_cap():
    start = dt.datetime(2026, 1, 1, tzinfo=UTC)
    end = dt.datetime(2026, 3, 1, tzinfo=UTC)
    windows = window_range(start, end)

    assert windows[0][0] == start
    assert windows[-1][1] == end
    assert all(stop - begin <= dt.timedelta(days=14) for begin, stop in windows)
    # Windows abut exactly, leaving no gap and no overlap.
    assert all(a[1] == b[0] for a, b in zip(windows, windows[1:], strict=False))


def test_window_range_rejects_a_reversed_interval():
    start = dt.datetime(2026, 1, 2, tzinfo=UTC)
    with pytest.raises(ValueError):
        window_range(start, start - dt.timedelta(days=1))


@responses.activate
def test_current_intensity_returns_payload():
    responses.add(responses.GET, f"{BASE_URL}/intensity", json={"data": []}, status=200)
    assert CarbonIntensityClient().current_intensity() == {"data": []}


@responses.activate
def test_client_retries_transient_failures_then_succeeds():
    responses.add(responses.GET, f"{BASE_URL}/intensity", status=503)
    responses.add(responses.GET, f"{BASE_URL}/intensity", json={"data": [1]}, status=200)

    client = CarbonIntensityClient(backoff=1.0)
    assert client.current_intensity() == {"data": [1]}
    assert len(responses.calls) == 2


@responses.activate
def test_client_does_not_retry_a_client_error():
    responses.add(responses.GET, f"{BASE_URL}/intensity", status=400)

    with pytest.raises(CarbonIntensityError, match="400"):
        CarbonIntensityClient(backoff=1.0).current_intensity()
    assert len(responses.calls) == 1


@responses.activate
def test_client_gives_up_after_max_attempts():
    responses.add(responses.GET, f"{BASE_URL}/intensity", status=500)

    client = CarbonIntensityClient(max_attempts=3, backoff=1.0)
    with pytest.raises(CarbonIntensityError, match="giving up"):
        client.current_intensity()
    assert len(responses.calls) == 3


@responses.activate
def test_client_rejects_a_response_without_a_data_field():
    responses.add(responses.GET, f"{BASE_URL}/intensity", json={"error": {}}, status=200)

    with pytest.raises(CarbonIntensityError, match="no data field"):
        CarbonIntensityClient().current_intensity()


def test_intensity_range_refuses_an_over_long_window():
    client = CarbonIntensityClient()
    start = dt.datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="14 days"):
        client.intensity_range(start, start + dt.timedelta(days=15))


@responses.activate
def test_forecast_48h_builds_the_documented_path():
    moment = dt.datetime(2026, 8, 20, 12, 35, tzinfo=UTC)
    url = f"{BASE_URL}/intensity/2026-08-20T12:35Z/fw48h"
    responses.add(responses.GET, url, json={"data": []}, status=200)

    CarbonIntensityClient().forecast_48h(moment)
    assert responses.calls[0].request.url == url


@pytest.mark.network
def test_live_endpoint_matches_the_documented_shape():
    payload = CarbonIntensityClient().current_intensity()
    assert payload["data"]
    assert {"from", "to", "intensity"} <= set(payload["data"][0])
