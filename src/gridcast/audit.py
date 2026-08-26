"""Diagnostics for the decision comparison.

The scheduling results raised a question the summary table cannot answer: the
seasonal baselines capture more of the available saving than the published
forecast, on identical decisions. The functions here open individual decisions
up so the mechanism can be seen rather than guessed at.

Nothing here computes a headline number. It exists to make a surprising result
falsifiable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import PERIOD, SETTLEMENT_LAG, seasonal_mean
from .load import forecast_record, outcome_record
from .scheduling import Load, baseline_forecasts, evaluate_decisions
from .storage import DEFAULT_ROOT

DAY = pd.Timedelta(days=1)


def reference_availability(load: Load, lags: tuple[int, ...] = (7, 14, 21)) -> pd.DataFrame:
    """How long before each issue time the seasonal references had settled.

    A seven-day lag sits at least 120 hours before any issue within a
    forty-eight-hour horizon, so the availability gate cannot be what lets a
    baseline appear to win. This tabulates the margin so the claim is checkable
    rather than asserted.
    """
    records = []
    for lag in lags:
        for horizon in (0.5, load.window_hours):
            reference_end = -lag * DAY + PERIOD + SETTLEMENT_LAG
            margin = -reference_end - pd.Timedelta(hours=horizon)
            records.append(
                {
                    "lag_days": lag,
                    "horizon_hours": horizon,
                    "settled_before_issue_hours": margin / pd.Timedelta(hours=1),
                }
            )
    return pd.DataFrame(records)


def decision_detail(
    root: Path = DEFAULT_ROOT,
    load: Load | None = None,
    predictor=seasonal_mean,
) -> pd.DataFrame:
    """One row per matched decision, with what each method chose and why.

    Carries the chosen start index for the published forecast, the baseline and
    hindsight, the realised cost of each, and the shape of the window: its
    spread, and how flat it is around its minimum. The last of these is the
    quantity that decides whether a near-miss is cheap.
    """
    load = load or Load()
    outcomes = outcome_record(root)
    published = forecast_record(root)
    if outcomes.empty or published.empty:
        return pd.DataFrame()

    decisions = evaluate_decisions(published, outcomes, load)
    if decisions.empty:
        return pd.DataFrame()

    issues = pd.DatetimeIndex(decisions["captured_at"])
    windows = pd.DatetimeIndex(decisions["window_start"])
    baseline = baseline_forecasts(outcomes, load, predictor, issues, window_starts=windows)
    baseline_decisions = evaluate_decisions(baseline, outcomes, load)

    settled = outcomes.set_index("period_start")["actual"].sort_index()
    published_by_issue = {
        moment: group.set_index("period_start")["forecast"]
        for moment, group in published.groupby("captured_at")
    }

    records = []
    for _, row in decisions.iterrows():
        index = pd.date_range(
            row["window_start"], periods=load.window_periods, freq=PERIOD, tz="UTC"
        )
        actual = settled.reindex(index).to_numpy(dtype=float)
        if np.isnan(actual).any():
            continue

        forecast = published_by_issue[row["captured_at"]].reindex(index).to_numpy(dtype=float)
        block = np.convolve(actual, np.ones(load.periods), mode="valid") / load.periods
        best = float(block.min())

        matching = baseline_decisions[baseline_decisions["window_start"] == row["window_start"]]
        records.append(
            {
                "window_start": row["window_start"],
                "published_start": row["chosen_start"],
                "baseline_start": int(matching["chosen_start"].iloc[0]) if len(matching) else -1,
                "oracle_start": row["oracle_start"],
                "published_cost": row["chosen"],
                "baseline_cost": float(matching["chosen"].iloc[0]) if len(matching) else np.nan,
                "oracle_cost": row["oracle"],
                "window_spread": float(actual.max() - actual.min()),
                "placements": int(len(block)),
                # How much worse a typical placement is than the best. Taken at
                # a rank fraction rather than a fixed rank: a fixed rank counts
                # a different part of the distribution in a six-hour window with
                # eleven placements than in a day with forty-five, and reads as
                # a property of the grid when it is a property of the window
                # length.
                "quartile_excess": float(np.quantile(block, 0.25) - best),
                "mean_excess": float(block.mean() - best),
                "forecast_bias": float(np.nanmean(forecast - actual)),
                "forecast_mae": float(np.nanmean(np.abs(forecast - actual))),
            }
        )

    return pd.DataFrame(records)


def ordering_quality(detail: pd.DataFrame) -> pd.DataFrame:
    """How far each method's choice sat from the true optimum, in periods.

    Distance in periods separates two failures a cost figure conflates: picking
    a window adjacent to the best one, and picking a window on the wrong side of
    the day.
    """
    if detail.empty:
        return pd.DataFrame()

    rows = {}
    for name in ("published", "baseline"):
        offset = (detail[f"{name}_start"] - detail["oracle_start"]).abs()
        rows[name] = {
            "mean_periods_from_optimum": float(offset.mean()),
            "median_periods_from_optimum": float(offset.median()),
            "within_one_period": float((offset <= 1).mean()),
            "within_four_periods": float((offset <= 4).mean()),
            "worse_than_twelve_periods": float((offset > 12).mean()),
        }
    return pd.DataFrame(rows).T


def cost_of_a_near_miss(detail: pd.DataFrame) -> pd.DataFrame:
    """What the window's shape implies about how much choice precision is worth.

    If ``quartile_excess`` is small relative to ``window_spread``, the better
    quarter of placements are all nearly as good as the best, so a forecast that
    ranks well but not perfectly loses very little. That would explain a low hit
    rate coexisting with a high captured fraction.

    Both excess measures are taken at rank fractions so that windows of
    different lengths can be compared. An earlier version used the tenth-best
    placement, which is a modest position among the forty-five a day offers and
    the second-worst among the eleven in six hours; the resulting figure
    described the window length rather than the shape of the grid.
    """
    if detail.empty:
        return pd.DataFrame()

    spread = detail["window_spread"].replace(0, np.nan)
    return pd.DataFrame(
        {
            "window_spread": detail["window_spread"].describe(),
            "placements": detail["placements"].describe(),
            "quartile_excess": detail["quartile_excess"].describe(),
            "quartile_over_spread": (detail["quartile_excess"] / spread).describe(),
            "mean_over_spread": (detail["mean_excess"] / spread).describe(),
        }
    )
