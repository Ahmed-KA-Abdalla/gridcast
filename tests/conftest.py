"""Fixtures shared across test modules.

``overshooting_store`` builds a store whose published forecasts overshoot, so
that the correction and the gate can both be exercised end to end against data
with a known property.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

UTC = dt.UTC


def intensity_payload(entries):
    return {
        "data": [
            {
                "from": start,
                "to": end,
                "intensity": {"forecast": forecast, "actual": actual, "index": "moderate"},
            }
            for start, end, forecast, actual in entries
        ]
    }


def stamp(moment: dt.datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%MZ")


@pytest.fixture
def overshooting_store(tmp_path):
    """A store whose forecasts overshoot: each revision partly undone by reality.

    One snapshot per issue time carrying every period it forecasts, which is the
    shape the real capture produces. Writing one period per snapshot would put
    several files at the same capture minute, and the later ones would overwrite
    the earlier.
    """
    from gridcast.storage import write_snapshot

    rng = np.random.default_rng(11)
    start = dt.datetime(2026, 8, 1, tzinfo=UTC)
    periods = 480

    truth = {
        start + dt.timedelta(minutes=30 * index): 200.0 + rng.normal(0, 20)
        for index in range(periods)
    }

    write_snapshot(
        intensity_payload(
            [
                (stamp(target), stamp(target + dt.timedelta(minutes=30)), value, value)
                for target, value in truth.items()
            ]
        ),
        "intensity_range_20260801",
        "/intensity/range",
        start + dt.timedelta(days=20),
        root=tmp_path,
    )

    # Hourly issues, each forecasting the next six hours. The forecast sits at
    # the truth plus twice its latest jolt, so half of every revision is undone.
    for step in range(periods // 2):
        issue = start + dt.timedelta(hours=step)
        entries = []
        for ahead in range(1, 13):
            target = issue + dt.timedelta(minutes=30 * ahead)
            if target not in truth:
                continue
            value = truth[target] + 2.0 * rng.normal(0, 12)
            entries.append((stamp(target), stamp(target + dt.timedelta(minutes=30)), value, None))
        if entries:
            write_snapshot(
                intensity_payload(entries),
                "forecast_fw48h",
                "/intensity/issue/fw48h",
                issue,
                root=tmp_path,
            )

    return tmp_path


@pytest.fixture
def record_path(tmp_path_factory):
    """A promotion record kept outside any snapshot root.

    The loader reads every JSON file beneath the root it is given, so a record
    written inside one would be picked up as a capture and rejected as a
    malformed envelope. In the repository the record lives under docs/.
    """
    return tmp_path_factory.mktemp("record") / "promoted.json"
