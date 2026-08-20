from __future__ import annotations

import datetime as dt
import json
import re

import pytest
import responses

from gridcast.cli import main
from gridcast.client import BASE_URL, CarbonIntensityClient
from gridcast.storage import (
    iter_snapshots,
    read_snapshot,
    snapshot_path,
    write_snapshot,
)

UTC = dt.UTC
CAPTURED = dt.datetime(2026, 8, 20, 12, 5, tzinfo=UTC)

FORECAST_URL = re.compile(rf"{re.escape(BASE_URL)}/intensity/.+/fw48h")

INTENSITY_PAYLOAD = {
    "data": [
        {
            "from": "2026-08-20T12:00Z",
            "to": "2026-08-20T12:30Z",
            "intensity": {"forecast": 266, "actual": 263, "index": "moderate"},
        }
    ]
}

# The live /generation endpoint returns a bare object here, not an array.
GENERATION_PAYLOAD = {
    "data": {
        "from": "2026-08-20T12:00Z",
        "to": "2026-08-20T12:30Z",
        "generationmix": [
            {"fuel": "gas", "perc": 60.0},
            {"fuel": "wind", "perc": 40.0},
        ],
    }
}


@pytest.fixture
def impatient_client() -> CarbonIntensityClient:
    """A client that does not sleep between attempts, so tests stay quick."""
    return CarbonIntensityClient(max_attempts=2, backoff=1.0)


def test_snapshot_path_partitions_by_utc_date(tmp_path):
    path = snapshot_path(tmp_path, "intensity", CAPTURED)
    assert path.relative_to(tmp_path).parts[:3] == ("2026", "08", "20")
    assert path.name == "intensity_20260820T1205Z.json"


def test_snapshot_path_partitions_by_utc_not_local_time(tmp_path):
    # 00:30 in London during summer is 23:30 UTC the previous day.
    local_midnight = dt.datetime(2026, 8, 20, 0, 30, tzinfo=dt.timezone(dt.timedelta(hours=1)))
    path = snapshot_path(tmp_path, "intensity", local_midnight)
    assert path.relative_to(tmp_path).parts[:3] == ("2026", "08", "19")


@pytest.mark.parametrize(
    "kind",
    [
        "/intensity",
        "intensity/2026-08-20T12:05Z/fw48h",
        "2026-08-20T12:05Z",
        "Intensity",
        "intensity range",
        "",
    ],
)
def test_snapshot_path_rejects_a_kind_that_is_unsafe_as_a_filename(tmp_path, kind):
    # A colon on NTFS opens an alternate data stream rather than a file, so an
    # unsafe kind must fail loudly rather than write somewhere invisible.
    with pytest.raises(ValueError, match="invalid snapshot kind"):
        snapshot_path(tmp_path, kind, CAPTURED)


def test_written_filenames_are_valid_on_windows(tmp_path):
    forbidden = set('<>:"/\\|?*')
    for kind in ("intensity", "forecast_fw48h", "generation", "intensity_range"):
        path = write_snapshot({"data": []}, kind, "/anything:goes/here", CAPTURED, root=tmp_path)
        assert not forbidden & set(path.name)
        assert path.is_file()


def test_write_and_read_round_trips_a_payload(tmp_path):
    payload = {"data": [{"from": "2026-08-20T12:00Z"}]}
    endpoint = "/intensity/2026-08-20T12:05Z/fw48h"
    path = write_snapshot(payload, "forecast_fw48h", endpoint, CAPTURED, root=tmp_path)

    recovered, kind, recovered_endpoint, captured_at = read_snapshot(path)
    assert recovered == payload
    assert kind == "forecast_fw48h"
    # The punctuation the filename cannot carry survives in the envelope.
    assert recovered_endpoint == endpoint
    assert captured_at == CAPTURED


def test_envelope_records_provenance(tmp_path):
    path = write_snapshot({"data": []}, "generation", "/generation", CAPTURED, root=tmp_path)
    envelope = json.loads(path.read_text(encoding="utf-8"))

    assert envelope["licence"] == "CC BY 4.0"
    assert envelope["source"] == "NESO Carbon Intensity API"
    assert envelope["captured_at"].startswith("2026-08-20T12:05")


def test_read_snapshot_rejects_an_unknown_envelope_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"envelope_version": 99, "payload": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported envelope"):
        read_snapshot(path)


def test_iter_snapshots_filters_by_kind_and_orders_by_time(tmp_path):
    later = CAPTURED + dt.timedelta(minutes=30)
    write_snapshot({"data": []}, "generation", "/generation", CAPTURED, root=tmp_path)
    write_snapshot({"data": []}, "intensity", "/intensity", later, root=tmp_path)
    write_snapshot({"data": []}, "intensity", "/intensity", CAPTURED, root=tmp_path)

    found = list(iter_snapshots(tmp_path, kind="intensity"))
    assert [p.name for p in found] == [
        "intensity_20260820T1205Z.json",
        "intensity_20260820T1235Z.json",
    ]


@responses.activate
def test_snapshot_command_writes_all_three_endpoints(tmp_path, impatient_client):
    responses.add(responses.GET, f"{BASE_URL}/intensity", json=INTENSITY_PAYLOAD, status=200)
    responses.add(responses.GET, FORECAST_URL, json=INTENSITY_PAYLOAD, status=200)
    responses.add(responses.GET, f"{BASE_URL}/generation", json=GENERATION_PAYLOAD, status=200)

    assert main(["--root", str(tmp_path), "snapshot"], client=impatient_client) == 0

    stored = sorted(read_snapshot(path)[1] for path in iter_snapshots(tmp_path))
    assert stored == ["forecast_fw48h", "generation", "intensity"]


@responses.activate
def test_snapshot_command_reports_failure_without_writing(tmp_path, impatient_client):
    responses.add(responses.GET, f"{BASE_URL}/intensity", status=500)
    responses.add(responses.GET, FORECAST_URL, status=500)
    responses.add(responses.GET, f"{BASE_URL}/generation", status=500)

    assert main(["--root", str(tmp_path), "snapshot"], client=impatient_client) == 1
    assert list(iter_snapshots(tmp_path)) == []


@responses.activate
def test_snapshot_command_rejects_a_payload_that_fails_validation(tmp_path, impatient_client):
    broken = {
        "data": [
            {
                "from": "2026-08-20T12:00Z",
                "to": "2026-08-20T12:30Z",
                "intensity": {"forecast": 9999, "actual": None, "index": "moderate"},
            }
        ]
    }
    responses.add(responses.GET, f"{BASE_URL}/intensity", json=broken, status=200)
    responses.add(responses.GET, FORECAST_URL, json=INTENSITY_PAYLOAD, status=200)
    responses.add(responses.GET, f"{BASE_URL}/generation", json=GENERATION_PAYLOAD, status=200)

    assert main(["--root", str(tmp_path), "snapshot"], client=impatient_client) == 1
    # The two sound endpoints are still stored; only the offending one is dropped.
    stored = sorted(read_snapshot(path)[1] for path in iter_snapshots(tmp_path))
    assert stored == ["forecast_fw48h", "generation"]


@responses.activate
def test_snapshot_command_survives_a_payload_of_the_wrong_shape(tmp_path, impatient_client):
    responses.add(responses.GET, f"{BASE_URL}/intensity", json={"data": "unexpected"}, status=200)
    responses.add(responses.GET, FORECAST_URL, json=INTENSITY_PAYLOAD, status=200)
    responses.add(responses.GET, f"{BASE_URL}/generation", json=GENERATION_PAYLOAD, status=200)

    # A malformed payload must be reported, not raised out of the process: one
    # bad endpoint should not cost the run the other two.
    assert main(["--root", str(tmp_path), "snapshot"], client=impatient_client) == 1
    assert len(list(iter_snapshots(tmp_path))) == 2


@responses.activate
def test_backfill_splits_a_long_range_into_windows(tmp_path, impatient_client):
    responses.add(
        responses.GET,
        re.compile(rf"{re.escape(BASE_URL)}/intensity/.+"),
        json=INTENSITY_PAYLOAD,
        status=200,
    )
    responses.add(
        responses.GET,
        re.compile(rf"{re.escape(BASE_URL)}/generation/.+"),
        json=GENERATION_PAYLOAD,
        status=200,
    )

    exit_code = main(
        ["--root", str(tmp_path), "backfill", "--start", "2026-01-01", "--end", "2026-02-01"],
        client=impatient_client,
    )
    assert exit_code == 0
    # Thirty-one days is three windows, each fetched for two endpoints.
    assert len(responses.calls) == 6
    # Six requests must leave six files: every window in a run shares one
    # capture minute, so the window start has to distinguish them.
    kinds = sorted(read_snapshot(p)[1] for p in iter_snapshots(tmp_path))
    assert kinds == [
        "generation_range_20260101",
        "generation_range_20260115",
        "generation_range_20260129",
        "intensity_range_20260101",
        "intensity_range_20260115",
        "intensity_range_20260129",
    ]
    assert len(list(iter_snapshots(tmp_path, kind="intensity_range_20260101"))) == 1
