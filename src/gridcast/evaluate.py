"""Scoring: metrics, lead-time breakdown, and the head-to-head comparison.

Two evaluations are offered and they answer different questions.

``backtest_baselines`` scores the seasonal baselines over the whole outcome
record. A seasonal prediction does not depend on lead time, so it can be
evaluated on every period ever observed. This gives a usable number from the
first day.

``compare_at_matched_leads`` scores the published forecast against those same
baselines, restricted to periods where a forecast was captured and the baseline
was computable from information available at that forecast's issue time. This is
the honest head-to-head, and it grows only as the capture record grows.

Reporting one in place of the other would flatter whichever is favoured by the
difference in sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import add_baselines, seasonal_mean, seasonal_naive
from .load import evaluation_frame, forecast_record, outcome_record
from .scheduling import (
    Load,
    baseline_forecasts,
    evaluate_decisions,
    issue_times,
    summarise,
)
from .storage import DEFAULT_ROOT

#: Lead-time buckets in hours. Uneven by design: forecast error grows fastest in
#: the first few hours and then flattens, so equal-width bins would waste
#: resolution where it matters and pool it where it does not.
LEAD_BINS = (0, 1, 3, 6, 12, 24, 48)


def metrics(prediction: pd.Series, actual: pd.Series) -> dict[str, float]:
    """Mean absolute error, root mean square error and bias, in gCO2/kWh.

    Rows where the prediction is missing are excluded rather than counted as
    error, and the surviving count is reported so that a metric computed from
    very few rows is visible as such.
    """
    mask = prediction.notna() & actual.notna()
    error = (prediction[mask] - actual[mask]).astype(float)
    if error.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan}

    return {
        "n": int(error.size),
        "mae": float(error.abs().mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
    }


def _lead_buckets(frame: pd.DataFrame) -> pd.Series:
    return pd.cut(frame["horizon_hours"], bins=list(LEAD_BINS), right=True)


def score_columns(
    frame: pd.DataFrame,
    columns: list[str],
    actual: str = "actual",
    by: pd.Series | None = None,
) -> pd.DataFrame:
    """Score several prediction columns against the same outcome.

    With ``by`` supplied, the score is computed within each group; without it,
    over the frame as a whole.
    """
    if by is None:
        rows = {name: metrics(frame[name], frame[actual]) for name in columns}
        return pd.DataFrame(rows).T

    records = []
    for group, part in frame.groupby(by, observed=True):
        for name in columns:
            records.append({"group": group, "model": name, **metrics(part[name], part[actual])})
    if not records:
        return pd.DataFrame()

    scored = pd.DataFrame(records)
    return scored.pivot(index="group", columns="model", values=["n", "mae", "bias"])


def backtest_baselines(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Score the seasonal baselines over every observed period.

    ``as_of`` is None here: the whole record is treated as available, which is
    correct for a retrospective backtest over settled history and is not a
    licence the matched comparison below is given.
    """
    outcomes = outcome_record(root)
    if outcomes.empty:
        return pd.DataFrame()

    scored = add_baselines(outcomes, outcomes, as_of=None)
    return score_columns(scored, ["seasonal_naive", "seasonal_mean"])


def compare_at_matched_leads(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Score the published forecast against the baselines on identical rows.

    Every model is scored on the same periods with the same information cut-off,
    so a difference in the numbers is a difference in the models.
    """
    frame = evaluation_frame(root)
    if frame.empty:
        return pd.DataFrame()

    outcomes = outcome_record(root)
    frame = add_baselines(frame, outcomes, as_of="captured_at")

    models = ["forecast", "seasonal_naive", "seasonal_mean"]
    # A row is comparable only where every model produced a prediction.
    comparable = frame[models].notna().all(axis=1)
    return score_columns(frame[comparable], models, by=_lead_buckets(frame[comparable]))


def skill(candidate: dict[str, float], reference: dict[str, float]) -> float:
    """Fractional reduction in mean absolute error against a reference.

    Positive means the candidate is better. Reported as a fraction rather than a
    percentage to keep the sign unambiguous when it is negative.
    """
    if not reference.get("n") or not np.isfinite(reference.get("mae", np.nan)):
        return np.nan
    return 1.0 - candidate["mae"] / reference["mae"]


def compare_schedulers(root: Path = DEFAULT_ROOT, load: Load | None = None) -> pd.DataFrame:
    """Score the published forecast and the baselines as schedulers.

    Three rows, and the first two are the comparison. ``published`` and the
    ``_matched`` baselines are scored on the same decisions: the issue times
    where a forward snapshot was captured. Differences between those rows are
    differences between the forecasters.

    The remaining rows score the baselines across the whole settled record.
    They are a much larger sample and they are not comparable with the first
    two: the captured decisions come from a few weeks of one season, so the
    windows differ in how much saving was available in the first place. An
    earlier version of this function omitted the matched rows, and the
    difference in available spread was large enough to reverse the ranking.
    """
    load = load or Load()
    outcomes = outcome_record(root)
    if outcomes.empty:
        return pd.DataFrame()

    rows: dict[str, dict[str, float]] = {}
    baselines = (("seasonal_naive", seasonal_naive), ("seasonal_mean", seasonal_mean))

    published = forecast_record(root)
    if not published.empty:
        decisions = evaluate_decisions(published, outcomes, load)
        rows["published"] = summarise(decisions)

        # The same issue times, so the baselines face the same windows.
        if not decisions.empty:
            issues = pd.DatetimeIndex(decisions["captured_at"])
            windows = pd.DatetimeIndex(decisions["window_start"])
            for name, predictor in baselines:
                forecasts = baseline_forecasts(
                    outcomes, load, predictor, issues, window_starts=windows
                )
                rows[f"{name}_matched"] = summarise(evaluate_decisions(forecasts, outcomes, load))

    moments = issue_times(outcomes, load)
    for name, predictor in baselines:
        forecasts = baseline_forecasts(outcomes, load, predictor, moments)
        rows[f"{name}_full"] = summarise(evaluate_decisions(forecasts, outcomes, load))

    return pd.DataFrame(rows).T
