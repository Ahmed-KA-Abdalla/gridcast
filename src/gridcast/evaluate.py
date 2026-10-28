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

``correction_decisions`` asks the project's central question of the correction:
does it change decisions for the better? It answers out of sample, block by
block, because a correction fitted on the record and scored on the same record
is being graded on its own homework.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baseline import add_baselines, seasonal_mean, seasonal_naive
from .correction import (
    DEFAULT_MODEL,
    Damping,
    fit_damping,
    revision_frame,
    rolling_blocks,
)
from .load import evaluation_frame, forecast_record, outcome_record
from .revisions import LEAD_BINS as CORRECTION_BANDS
from .revisions import revision_paths
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

    Three kinds of row, and the first two are the comparison. ``published`` and
    the ``_matched`` baselines are scored on the same decisions: the issue times
    where a forward snapshot was captured. Differences between those rows are
    differences between the forecasters.

    The corrected forecast is not among them. Its coefficients are fitted on
    this record, so scoring it here would be in sample, and as the correction
    improved the in-sample row flattered it more. ``correction_decisions`` scores
    it out of sample instead.

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


def _day_clustered_interval(
    values: np.ndarray, days: np.ndarray, resamples: int, seed: int = 0
) -> tuple[float, float]:
    """A percentile interval for a mean, resampling whole days.

    Several decisions are issued each day and their windows overlap, so they are
    not independent. Resampling days keeps each day's decisions together.
    """
    frame = pd.DataFrame({"day": days, "value": values}).groupby("day")["value"]
    sums = frame.sum().to_numpy(dtype=float)
    sizes = frame.size().to_numpy(dtype=float)
    if len(sums) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    drawn = rng.integers(0, len(sums), size=(resamples, len(sums)))
    means = sums[drawn].sum(axis=1) / sizes[drawn].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def correction_decisions(
    root: Path = DEFAULT_ROOT,
    load: Load | None = None,
    record: Path | None = None,
    resamples: int = 4000,
) -> tuple[dict[str, object], pd.DataFrame]:
    """The promoted correction's effect on decisions, out of sample.

    The correction the project claims — the promoted bands, under the model the
    record was written with — is applied to each held-out block with
    coefficients fitted only on earlier dates, purged of the block's target
    periods exactly as in the accuracy evaluation. Each corrected decision is
    paired with the published forecast's decision at the same issue time, so the
    two face identical windows and the oracle is shared.

    One qualification the interval cannot carry. Which bands are promoted was
    decided by the gate on the same record, so the choice of bands is informed
    by these blocks even though no coefficient is.
    """
    from .gate import DEFAULT_RECORD, load_record, record_model

    load = load or Load()
    record = record or DEFAULT_RECORD
    promoted = load_record(record)
    model = record_model(record) or DEFAULT_MODEL
    if not promoted:
        return {"reason": "no band is promoted, so there is no correction to apply"}, pd.DataFrame()

    frame = revision_frame(root)
    blocks = rolling_blocks(frame)
    if not blocks:
        return {"reason": "not enough captured revisions to hold any out"}, pd.DataFrame()

    paths = revision_paths(root)
    outcomes = outcome_record(root)
    published = forecast_record(root)
    paths = paths.assign(
        # The correction's bands, not this module's finer error buckets: a
        # promoted (0, 3] would otherwise never match and silently go unapplied.
        band=pd.cut(paths["horizon_hours"], bins=list(CORRECTION_BANDS), right=True).astype(str),
        date=paths["captured_at"].dt.date,
    )
    published = published.assign(date=published["captured_at"].dt.date)
    columns = ["period_start", "captured_at", "forecast", "horizon_hours"]

    parts = []
    for index, (first, last) in enumerate(blocks):
        held = frame[(frame["date"] >= first) & (frame["date"] <= last)]
        fit = frame[frame["date"] < first]
        fit = fit[~fit["period_start"].isin(held["period_start"].unique())]
        fitted = fit_damping(fit, model)
        damping = Damping({band: fitted.for_band(band) for band in promoted}, {}, model)

        block = paths[(paths["date"] >= first) & (paths["date"] <= last)]
        revision = block["revision"].fillna(0.0)
        corrected = block.assign(
            forecast=block["forecast"] - damping.factors(block["band"], revision) * revision
        )[columns]
        baseline = published[(published["date"] >= first) & (published["date"] <= last)][columns]

        scored_published = evaluate_decisions(baseline, outcomes, load)
        scored_corrected = evaluate_decisions(corrected, outcomes, load)
        if scored_published.empty or scored_corrected.empty:
            continue
        merged = scored_published.merge(
            scored_corrected,
            on=["captured_at", "window_start"],
            suffixes=("_published", "_corrected"),
        )
        parts.append(merged.assign(block=index, first=first, last=last))

    if not parts:
        return {"reason": "no held-out decision could be scored"}, pd.DataFrame()

    decisions = pd.concat(parts, ignore_index=True)
    # The oracle depends only on the outcome, so both sides share it.
    oracle = decisions["oracle_published"]
    decisions["regret_published"] = decisions["chosen_published"] - oracle
    decisions["regret_corrected"] = decisions["chosen_corrected"] - oracle
    decisions["gain"] = decisions["regret_published"] - decisions["regret_corrected"]
    days = decisions["captured_at"].dt.date.to_numpy()

    low, high = _day_clustered_interval(decisions["gain"].to_numpy(dtype=float), days, resamples)
    per_block = (
        decisions.groupby(["block", "first", "last"])["gain"]
        .agg(n="size", gain="mean")
        .reset_index()
    )
    changed = (decisions["chosen_published"] - decisions["chosen_corrected"]).abs() > 1e-9

    summary: dict[str, object] = {
        "model": model,
        "bands": sorted(promoted),
        "n": int(len(decisions)),
        "days": int(len(set(days))),
        "blocks": int(len(per_block)),
        "blocks_helped": int((per_block["gain"] > 0).sum()),
        "published_regret": float(decisions["regret_published"].mean()),
        "corrected_regret": float(decisions["regret_corrected"].mean()),
        "gain": float(decisions["gain"].mean()),
        "gain_low": low,
        "gain_high": high,
        "changed": float(changed.mean()),
    }
    return summary, per_block
