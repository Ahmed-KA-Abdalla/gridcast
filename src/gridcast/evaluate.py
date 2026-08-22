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

from .baseline import add_baselines
from .load import evaluation_frame, outcome_record
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
