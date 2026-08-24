"""Scheduling decisions and the regret they incur.

Nobody consumes carbon intensity for its own sake. It is used to decide when to
run a deferrable load: charge a car, heat water, run a wash. The question this
module asks is therefore not how close a forecast was, but whether it led to the
right hours being chosen.

The two come apart. A forecast uniformly too high by 40 gCO2/kWh has a mean
absolute error of 40 and schedules perfectly, because adding a constant changes
no ordering. A forecast with an error of 8 that inverts the two cheapest windows
schedules badly. Only the ordering matters here, and mean error does not measure
ordering.

Four choices are scored for every decision:

``chosen``     the periods the forecast picked, costed at what actually happened
``oracle``     the true cheapest periods, knowable only afterwards
``worst``      the true most expensive periods, which bounds what was at stake
``immediate``  running straight away, the behaviour of a consumer who does not
               defer at all

Regret is ``chosen - oracle``. Reported alone it is misleading: on a flat day no
choice is much worse than any other, so a windless week would flatter any
scheduler. It is therefore normalised by ``worst - oracle``, the saving that was
actually available, giving the fraction of the achievable benefit that was
forgone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIOD = pd.Timedelta(minutes=30)


@dataclass(frozen=True)
class Load:
    """A deferrable load: how many half-hours it needs and by when.

    ``contiguous`` distinguishes a wash cycle, which must run in one block, from
    a car charger that can pause and resume.
    """

    periods: int = 4
    window_hours: float = 24.0
    contiguous: bool = True

    def __post_init__(self) -> None:
        if self.periods < 1:
            raise ValueError("a load must occupy at least one period")
        if self.window_hours * 2 < self.periods:
            raise ValueError("the window is too short to fit the load")

    @property
    def window_periods(self) -> int:
        return int(self.window_hours * 2)

    def describe(self) -> str:
        hours = self.periods / 2
        placement = "contiguous" if self.contiguous else "interruptible"
        return f"{hours:g}h {placement} load within {self.window_hours:g}h"


def choose(values: np.ndarray, load: Load, worst: bool = False) -> np.ndarray:
    """Return the indices a scheduler would select from ``values``.

    Ties are broken by taking the earliest qualifying option, which matches what
    a real scheduler does and keeps the result deterministic.
    """
    count = load.periods
    if len(values) < count:
        raise ValueError("fewer periods available than the load requires")

    if load.contiguous:
        # Sum over every placement of a block of the required length.
        window_sums = np.convolve(values, np.ones(count), mode="valid")
        start = int(np.argmax(window_sums) if worst else np.argmin(window_sums))
        return np.arange(start, start + count)

    order = np.argsort(-values if worst else values, kind="stable")
    return np.sort(order[:count])


def decision_costs(forecast: np.ndarray, actual: np.ndarray, load: Load) -> dict[str, float]:
    """Cost the four choices for one decision, in mean gCO2/kWh over the load.

    Every cost is evaluated against ``actual``: what a choice was expected to
    cost is not what it cost. Reporting a mean rather than a total keeps the
    figure independent of the load's power.
    """
    if len(forecast) != len(actual):
        raise ValueError("forecast and outcome must cover the same periods")

    chosen = choose(forecast, load)
    oracle = choose(actual, load)
    worst = choose(actual, load, worst=True)
    immediate = np.arange(load.periods)

    return {
        "chosen": float(actual[chosen].mean()),
        "oracle": float(actual[oracle].mean()),
        "worst": float(actual[worst].mean()),
        "immediate": float(actual[immediate].mean()),
        "chosen_start": int(chosen[0]),
        "oracle_start": int(oracle[0]),
    }


def summarise(decisions: pd.DataFrame) -> dict[str, float]:
    """Aggregate a frame of decisions into the headline figures.

    ``captured_fraction`` is the share of the available saving the scheduler
    secured: one means it chose as well as hindsight allowed, zero means it did
    no better than running immediately would have. It is undefined for decisions
    where nothing was at stake, and those are excluded and counted separately
    rather than being scored as successes.
    """
    if decisions.empty:
        return {"n": 0}

    regret = decisions["chosen"] - decisions["oracle"]
    available = decisions["worst"] - decisions["oracle"]
    flat = available <= 0

    summary = {
        "n": int(len(decisions)),
        "mean_regret": float(regret.mean()),
        "median_regret": float(regret.median()),
        "mean_available": float(available.mean()),
        "hit_rate": float((decisions["chosen_start"] == decisions["oracle_start"]).mean()),
        "flat_windows": int(flat.sum()),
    }

    usable = ~flat
    if usable.any():
        summary["normalised_regret"] = float((regret[usable] / available[usable]).mean())
        summary["captured_fraction"] = 1.0 - summary["normalised_regret"]

    # What deferring bought at all, against not deferring.
    summary["mean_saving_vs_immediate"] = float(
        (decisions["immediate"] - decisions["chosen"]).mean()
    )
    return summary


def _window(
    outcomes: pd.Series, start: pd.Timestamp, load: Load
) -> tuple[pd.DatetimeIndex, np.ndarray] | None:
    """The contiguous run of settled periods a decision may choose among.

    A decision is skipped where the window is incomplete. Interpolating across a
    gap would invent observations, and scoring a scheduler on invented data
    would make its regret unfalsifiable.
    """
    index = pd.date_range(start, periods=load.window_periods, freq=PERIOD, tz="UTC")
    values = outcomes.reindex(index)
    if values.isna().any():
        return None
    return index, values.to_numpy(dtype=float)


def evaluate_decisions(
    forecasts: pd.DataFrame,
    outcomes: pd.DataFrame,
    load: Load,
) -> pd.DataFrame:
    """Score one decision per issue time in ``forecasts``.

    ``forecasts`` carries ``captured_at``, ``period_start`` and ``forecast``:
    the loader's forecast record, or a baseline's predictions in the same shape.
    Decisions are scored one at a time rather than vectorised, since each is
    independent and clarity is worth more here than speed.
    """
    if forecasts.empty or outcomes.empty:
        return pd.DataFrame()

    settled = outcomes.set_index("period_start")["actual"].sort_index()
    records = []

    for captured_at, issue in forecasts.groupby("captured_at"):
        start = issue["period_start"].min()
        window = _window(settled, start, load)
        if window is None:
            continue

        index, actual = window
        predicted = issue.set_index("period_start")["forecast"].reindex(index)
        if predicted.isna().any():
            continue

        costs = decision_costs(predicted.to_numpy(dtype=float), actual, load)
        records.append({"captured_at": captured_at, "window_start": start, **costs})

    return pd.DataFrame(records)


def issue_times(
    outcomes: pd.DataFrame, load: Load, hour: int = 18, minute: int = 0
) -> pd.DatetimeIndex:
    """One decision per day, at a fixed hour, across the settled record.

    Sampling daily rather than at every half-hour keeps successive decisions
    from overlapping almost entirely, which would make the sample look far
    larger than the number of independent situations in it.
    """
    if outcomes.empty:
        return pd.DatetimeIndex([], tz="UTC")

    first = outcomes["period_start"].min().normalize() + pd.Timedelta(hours=hour, minutes=minute)
    last = outcomes["period_start"].max() - pd.Timedelta(hours=load.window_hours)
    if last < first:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.date_range(first, last, freq="D", tz="UTC")


def baseline_forecasts(
    outcomes: pd.DataFrame,
    load: Load,
    predictor,
    moments: pd.DatetimeIndex,
    window_starts: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Predictions from a baseline for the window following each issue time.

    Produced in the same shape as the loader's forecast record, so that a
    baseline and the published forecast go through one scoring path and cannot
    diverge in how they are treated.

    ``window_starts`` is separate from ``moments`` because the two are not the
    same thing. A captured forecast is issued partway through a period and its
    window begins at the next one, so aligning a baseline to the issue minute
    would shift it half an hour against the forecast it is being compared with
    and score the two on different windows. Supply the window starts explicitly
    to match a set of decisions; they default to the issue times.
    """
    if window_starts is None:
        window_starts = moments

    frames = []
    for moment, window_start in zip(moments, window_starts, strict=True):
        index = pd.date_range(window_start, periods=load.window_periods, freq=PERIOD, tz="UTC")
        targets = pd.DataFrame({"period_start": index, "captured_at": moment})
        targets["forecast"] = predictor(targets, outcomes, as_of="captured_at").to_numpy()
        frames.append(targets)

    if not frames:
        return pd.DataFrame(columns=["period_start", "captured_at", "forecast"])
    return pd.concat(frames, ignore_index=True)
