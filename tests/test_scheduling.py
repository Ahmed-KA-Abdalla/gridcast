from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast.scheduling import (
    Load,
    choose,
    decision_costs,
    evaluate_decisions,
    summarise,
)


def test_a_load_must_fit_inside_its_window():
    with pytest.raises(ValueError, match="too short"):
        Load(periods=8, window_hours=2.0)


def test_a_load_must_occupy_a_period():
    with pytest.raises(ValueError, match="at least one period"):
        Load(periods=0)


def test_contiguous_choice_takes_the_cheapest_block():
    values = np.array([10.0, 10.0, 1.0, 1.0, 10.0, 10.0])
    assert choose(values, Load(periods=2, window_hours=3.0)).tolist() == [2, 3]


def test_contiguous_choice_will_not_split_a_block():
    # The two cheapest periods are 0 and 5, but a contiguous load cannot take
    # both, so it must settle for the cheapest adjacent pair.
    values = np.array([1.0, 9.0, 4.0, 4.0, 9.0, 1.0])
    assert choose(values, Load(periods=2, window_hours=3.0)).tolist() == [2, 3]


def test_an_interruptible_load_may_split():
    values = np.array([1.0, 9.0, 4.0, 4.0, 9.0, 1.0])
    load = Load(periods=2, window_hours=3.0, contiguous=False)
    assert choose(values, load).tolist() == [0, 5]


def test_ties_resolve_to_the_earliest_option():
    values = np.array([1.0, 1.0, 1.0, 1.0])
    assert choose(values, Load(periods=2, window_hours=2.0)).tolist() == [0, 1]


def test_the_worst_choice_is_the_most_expensive_block():
    values = np.array([10.0, 10.0, 1.0, 1.0, 10.0, 10.0])
    load = Load(periods=2, window_hours=3.0)
    assert choose(values, load, worst=True).tolist() == [0, 1]


def test_a_constant_offset_in_the_forecast_costs_nothing():
    # The decisive property. A forecast uniformly wrong by a large margin has a
    # large mean absolute error and yet schedules perfectly, because adding a
    # constant changes no ordering.
    actual = np.array([300.0, 100.0, 120.0, 280.0])
    forecast = actual + 40.0
    load = Load(periods=1, window_hours=2.0)

    costs = decision_costs(forecast, actual, load)
    assert costs["chosen"] == costs["oracle"]


def test_a_small_error_that_inverts_two_windows_costs_a_great_deal():
    # The converse. A forecast accurate to within 8 gCO2/kWh picks the wrong
    # window because it reverses the order of the two cheapest periods.
    actual = np.array([300.0, 100.0, 108.0, 300.0])
    forecast = np.array([300.0, 108.0, 100.0, 300.0])
    load = Load(periods=1, window_hours=2.0)

    costs = decision_costs(forecast, actual, load)
    assert costs["chosen"] == 108.0
    assert costs["oracle"] == 100.0


def test_costs_are_all_evaluated_against_what_happened():
    # What a choice was expected to cost is not what it cost.
    actual = np.array([50.0, 200.0])
    forecast = np.array([500.0, 10.0])
    costs = decision_costs(forecast, actual, Load(periods=1, window_hours=1.0))

    assert costs["chosen"] == 200.0  # the forecast preferred the second period
    assert costs["oracle"] == 50.0
    assert costs["worst"] == 200.0


def test_immediate_is_the_cost_of_not_deferring():
    actual = np.array([300.0, 100.0, 100.0, 100.0])
    forecast = actual.copy()
    costs = decision_costs(forecast, actual, Load(periods=1, window_hours=2.0))
    assert costs["immediate"] == 300.0


def test_forecast_and_outcome_must_cover_the_same_periods():
    with pytest.raises(ValueError, match="same periods"):
        decision_costs(np.array([1.0, 2.0]), np.array([1.0]), Load(periods=1, window_hours=1.0))


# -- aggregation ----------------------------------------------------------


def decisions_frame(rows: list[tuple[float, float, float, float, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["chosen", "oracle", "worst", "immediate", "chosen_start", "oracle_start"]
    )


def test_captured_fraction_is_one_for_a_perfect_scheduler():
    frame = decisions_frame([(100.0, 100.0, 300.0, 200.0, 3, 3)])
    assert summarise(frame)["captured_fraction"] == pytest.approx(1.0)


def test_captured_fraction_is_zero_for_the_worst_possible_scheduler():
    frame = decisions_frame([(300.0, 100.0, 300.0, 200.0, 0, 3)])
    assert summarise(frame)["captured_fraction"] == pytest.approx(0.0)


def test_a_flat_window_is_excluded_rather_than_scored_as_success():
    # Nothing was at stake, so the scheduler deserves no credit. Counting it as
    # a perfect decision would let a windless week flatter any scheduler.
    frame = decisions_frame(
        [(100.0, 100.0, 100.0, 100.0, 0, 0), (150.0, 100.0, 300.0, 200.0, 1, 2)]
    )
    summary = summarise(frame)

    assert summary["flat_windows"] == 1
    assert summary["n"] == 2
    # Only the second decision contributes: regret 50 of an available 200.
    assert summary["captured_fraction"] == pytest.approx(0.75)


def test_hit_rate_counts_decisions_that_found_the_true_window():
    frame = decisions_frame(
        [(100.0, 100.0, 300.0, 200.0, 2, 2), (200.0, 100.0, 300.0, 200.0, 0, 2)]
    )
    assert summarise(frame)["hit_rate"] == pytest.approx(0.5)


def test_saving_against_not_deferring_is_reported():
    frame = decisions_frame([(120.0, 100.0, 300.0, 250.0, 1, 2)])
    assert summarise(frame)["mean_saving_vs_immediate"] == pytest.approx(130.0)


def test_summarise_reports_nothing_on_an_empty_frame():
    assert summarise(pd.DataFrame())["n"] == 0


# -- end to end -----------------------------------------------------------


def build(start: str, values: list[float]) -> pd.DataFrame:
    index = pd.date_range(start, periods=len(values), freq="30min", tz="UTC")
    return pd.DataFrame({"period_start": index, "actual": values})


def test_evaluate_decisions_scores_one_decision_per_issue():
    outcomes = build("2026-08-20T00:00Z", [300.0, 100.0, 110.0, 290.0])
    forecasts = pd.DataFrame(
        {
            "captured_at": pd.to_datetime(["2026-08-19T23:00Z"] * 4, utc=True),
            "period_start": outcomes["period_start"],
            "forecast": [300.0, 100.0, 110.0, 290.0],
        }
    )
    frame = evaluate_decisions(forecasts, outcomes, Load(periods=1, window_hours=2.0))

    assert len(frame) == 1
    assert frame["chosen"].iloc[0] == 100.0


def test_a_decision_whose_window_is_incomplete_is_skipped():
    # Interpolating across a gap would invent observations, and a scheduler
    # scored on invented data cannot be falsified.
    outcomes = build("2026-08-20T00:00Z", [300.0, 100.0, 110.0, 290.0])
    outcomes = outcomes.drop(index=2)

    forecasts = pd.DataFrame(
        {
            "captured_at": pd.to_datetime(["2026-08-19T23:00Z"] * 3, utc=True),
            "period_start": outcomes["period_start"],
            "forecast": [300.0, 100.0, 290.0],
        }
    )
    assert evaluate_decisions(forecasts, outcomes, Load(periods=1, window_hours=2.0)).empty


def test_evaluate_decisions_is_empty_without_input():
    empty = pd.DataFrame(columns=["captured_at", "period_start", "forecast"])
    assert evaluate_decisions(empty, empty, Load()).empty


def test_baseline_windows_can_be_aligned_to_a_set_of_decisions():
    # A captured forecast is issued partway through a period and its window
    # begins at the next one. Aligning a baseline to the issue minute instead
    # would shift it half an hour and score the two on different windows.
    from gridcast.scheduling import baseline_forecasts

    index = pd.date_range("2026-08-20T00:00Z", periods=200, freq="30min", tz="UTC")
    outcomes = pd.DataFrame({"period_start": index, "actual": np.arange(200.0)})

    issue = pd.Timestamp("2026-08-22T11:45Z")
    window_start = pd.Timestamp("2026-08-22T12:00Z")
    load = Load(periods=1, window_hours=2.0)

    aligned = baseline_forecasts(
        outcomes,
        load,
        lambda t, o, as_of: pd.Series(1.0, index=t.index),
        pd.DatetimeIndex([issue]),
        window_starts=pd.DatetimeIndex([window_start]),
    )
    assert aligned["period_start"].min() == window_start

    unaligned = baseline_forecasts(
        outcomes,
        load,
        lambda t, o, as_of: pd.Series(1.0, index=t.index),
        pd.DatetimeIndex([issue]),
    )
    assert unaligned["period_start"].min() == issue
