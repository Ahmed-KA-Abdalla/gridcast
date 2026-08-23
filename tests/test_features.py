from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.features import (
    FEATURE_COLUMNS,
    build_features,
    calendar_features,
    issue_grid,
    knowable_from,
    latest_observation_features,
    training_frame,
    weekly_reference_features,
)
from gridcast.parse import FUELS


def periods(start: str, count: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=count, freq="30min", tz="UTC")


def outcomes_frame(start: str, count: int, values=None) -> pd.DataFrame:
    index = periods(start, count)
    return pd.DataFrame(
        {
            "period_start": index,
            "actual": values if values is not None else np.arange(count, dtype=float),
        }
    )


def generation_frame(start: str, count: int, gas: float = 60.0) -> pd.DataFrame:
    index = periods(start, count)
    frame = pd.DataFrame({"period_start": index})
    for fuel in FUELS:
        frame[fuel] = 0.0
    frame["gas"] = gas
    frame["wind"] = 100.0 - gas
    return frame


def targets_frame(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_start": pd.to_datetime([p[0] for p in pairs], utc=True),
            "captured_at": pd.to_datetime([p[1] for p in pairs], utc=True),
        }
    )


def test_knowable_from_allows_the_period_to_end_and_settle():
    start = pd.Series(pd.to_datetime(["2026-08-20T12:00Z"], utc=True))
    assert knowable_from(start).iloc[0] == pd.Timestamp("2026-08-20T13:30Z")


def test_calendar_features_wrap_around_midnight():
    frame = calendar_features(
        pd.Series(pd.to_datetime(["2026-08-20T23:30Z", "2026-08-21T00:00Z"], utc=True))
    )
    # Adjacent half-hours must be adjacent in the encoding, which a raw hour
    # index would not make them.
    separation = np.hypot(
        frame["sin_day"].iloc[1] - frame["sin_day"].iloc[0],
        frame["cos_day"].iloc[1] - frame["cos_day"].iloc[0],
    )
    assert separation < 0.2


def test_calendar_features_use_local_clock_time():
    # 12:00 UTC in August is 13:00 in London.
    frame = calendar_features(pd.Series(pd.to_datetime(["2026-08-20T12:00Z"], utc=True)))
    assert frame["sin_day"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 13 / 24), abs=1e-9)


def test_weekend_flag_follows_the_local_day():
    frame = calendar_features(
        pd.Series(pd.to_datetime(["2026-08-21T12:00Z", "2026-08-22T12:00Z"], utc=True))
    )
    assert frame["is_weekend"].tolist() == [0, 1]


def test_weekly_reference_reads_the_same_half_hour_a_week_earlier():
    outcomes = pd.DataFrame(
        {
            "period_start": pd.to_datetime(["2026-08-13T18:00Z"], utc=True),
            "actual": [300.0],
        }
    )
    targets = targets_frame([("2026-08-20T18:00Z", "2026-08-20T12:00Z")])
    frame = weekly_reference_features(targets, outcomes, lags=(7,))
    assert frame["intensity_lag_7d"].tolist() == [300.0]


def test_latest_observation_respects_the_issue_time():
    outcomes = outcomes_frame("2026-08-20T00:00Z", 48)
    generation = generation_frame("2026-08-20T00:00Z", 48)

    # Issued at 06:00. The 04:00 period settled at 05:30 and is usable; the
    # 05:00 period settles at 06:30 and is not.
    targets = targets_frame([("2026-08-21T00:00Z", "2026-08-20T06:00Z")])
    frame = latest_observation_features(targets, outcomes, generation)

    assert frame["recent_intensity"].iloc[0] == 9.0  # the 04:30 period
    assert frame["recent_staleness_hours"].iloc[0] == pytest.approx(1.5)


def test_a_longer_lead_sees_a_staler_observation():
    outcomes = outcomes_frame("2026-08-20T00:00Z", 96)
    generation = generation_frame("2026-08-20T00:00Z", 96)

    target = "2026-08-21T12:00Z"
    near = targets_frame([(target, "2026-08-21T11:00Z")])
    far = targets_frame([(target, "2026-08-19T12:00Z")])

    near_staleness = latest_observation_features(near, outcomes, generation)[
        "recent_staleness_hours"
    ].iloc[0]
    far_features = latest_observation_features(far, outcomes, generation)

    assert near_staleness == pytest.approx(1.5)
    # Nothing had been observed before the far issue time at all.
    assert np.isnan(far_features["recent_intensity"].iloc[0])


def test_latest_observation_carries_the_fuel_mix():
    outcomes = outcomes_frame("2026-08-20T00:00Z", 48)
    generation = generation_frame("2026-08-20T00:00Z", 48, gas=70.0)
    targets = targets_frame([("2026-08-21T00:00Z", "2026-08-20T06:00Z")])

    frame = latest_observation_features(targets, outcomes, generation)
    assert frame["recent_gas"].iloc[0] == pytest.approx(70.0)
    assert frame["recent_wind"].iloc[0] == pytest.approx(30.0)


def test_no_feature_can_see_the_target_period_itself():
    # The decisive property: a feature computed for a target must never read an
    # observation from the target period or later.
    outcomes = outcomes_frame("2026-08-20T00:00Z", 200)
    generation = generation_frame("2026-08-20T00:00Z", 200)
    target = pd.Timestamp("2026-08-23T12:00Z")

    targets = pd.DataFrame(
        {"period_start": [target], "captured_at": [target - pd.Timedelta(hours=1)]}
    )
    frame = latest_observation_features(targets, outcomes, generation)

    observed_value = frame["recent_intensity"].iloc[0]
    index_of_target = outcomes.index[outcomes["period_start"] == target][0]
    assert observed_value < outcomes.loc[index_of_target, "actual"]
    assert frame["recent_staleness_hours"].iloc[0] >= 1.5


def test_build_features_produces_every_declared_column():
    outcomes = outcomes_frame("2026-08-01T00:00Z", 1200)
    generation = generation_frame("2026-08-01T00:00Z", 1200)
    targets = targets_frame([("2026-08-25T12:00Z", "2026-08-25T06:00Z")])

    frame = build_features(targets, outcomes, generation)
    assert list(frame.columns) == FEATURE_COLUMNS


def test_horizon_is_derived_from_the_pair_not_supplied():
    outcomes = outcomes_frame("2026-08-01T00:00Z", 1200)
    generation = generation_frame("2026-08-01T00:00Z", 1200)
    targets = targets_frame([("2026-08-25T12:00Z", "2026-08-24T12:00Z")])

    assert build_features(targets, outcomes, generation)["horizon_hours"].iloc[0] == 24.0


def test_issue_grid_pairs_every_period_with_every_horizon():
    outcomes = outcomes_frame("2026-08-20T00:00Z", 10)
    grid = issue_grid(outcomes, horizons=(1.0, 24.0))

    assert len(grid) == 20
    first = grid[grid["period_start"] == pd.Timestamp("2026-08-20T00:00Z")]
    leads = (first["period_start"] - first["captured_at"]) / pd.Timedelta(hours=1)
    assert sorted(leads.tolist()) == [1.0, 24.0]


def test_training_frame_carries_the_outcome_beside_the_features():
    outcomes = outcomes_frame("2026-08-01T00:00Z", 1200)
    generation = generation_frame("2026-08-01T00:00Z", 1200)

    frame = training_frame(outcomes, generation, horizons=(1.0, 24.0))
    assert {"period_start", "captured_at", "actual"} <= set(frame.columns)
    assert set(FEATURE_COLUMNS) <= set(frame.columns)
    assert len(frame) == 2400


def test_training_rows_for_one_period_differ_only_in_what_was_knowable():
    outcomes = outcomes_frame("2026-08-01T00:00Z", 1200)
    generation = generation_frame("2026-08-01T00:00Z", 1200)

    frame = training_frame(outcomes, generation, horizons=(1.0, 24.0))
    one_period = frame[frame["period_start"] == pd.Timestamp("2026-08-20T12:00Z")]

    assert len(one_period) == 2
    # Same target, same calendar position, same weekly references.
    assert one_period["actual"].nunique() == 1
    assert one_period["sin_day"].nunique() == 1
    assert one_period["intensity_lag_7d"].nunique() == 1
    # Different lead, and correspondingly different recency.
    assert one_period["horizon_hours"].nunique() == 2
    assert one_period["recent_intensity"].nunique() == 2


def test_empty_inputs_produce_empty_output():
    empty = pd.DataFrame(columns=["period_start", "actual"])
    assert issue_grid(empty).empty
    assert training_frame(empty, empty).empty


def test_staleness_is_measured_from_the_issue_time_not_the_target():
    # Normally constant at the settlement allowance: at any lead there is an
    # observation that has just become knowable. Distance from the observation
    # to the target grows with the lead, but that is the lead plus this
    # quantity, so it is not carried as a separate feature.
    outcomes = outcomes_frame("2026-08-01T00:00Z", 1500)
    generation = generation_frame("2026-08-01T00:00Z", 1500)
    target = "2026-08-25T12:00Z"

    near = latest_observation_features(
        targets_frame([(target, "2026-08-25T11:00Z")]), outcomes, generation
    )
    far = latest_observation_features(
        targets_frame([(target, "2026-08-23T12:00Z")]), outcomes, generation
    )

    assert near["recent_staleness_hours"].iloc[0] == pytest.approx(1.5)
    assert far["recent_staleness_hours"].iloc[0] == pytest.approx(1.5)
    # The two leads none the less see different observations.
    assert near["recent_intensity"].iloc[0] != far["recent_intensity"].iloc[0]


def test_staleness_rises_above_the_floor_where_the_record_has_a_gap():
    outcomes = outcomes_frame("2026-08-01T00:00Z", 1500)
    generation = generation_frame("2026-08-01T00:00Z", 1500)

    # Remove twelve hours immediately before the issue time.
    gap_start = pd.Timestamp("2026-08-25T00:00Z")
    gap_end = pd.Timestamp("2026-08-25T12:00Z")
    gapped = outcomes[
        (outcomes["period_start"] < gap_start) | (outcomes["period_start"] >= gap_end)
    ]

    frame = latest_observation_features(
        targets_frame([("2026-08-26T00:00Z", "2026-08-25T11:00Z")]), gapped, generation
    )
    assert frame["recent_staleness_hours"].iloc[0] > 10.0
