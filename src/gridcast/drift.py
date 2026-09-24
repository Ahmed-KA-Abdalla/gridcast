"""Monitoring whether the data a model was fitted on still resembles the data
it is being asked about.

Every model and coefficient in this repository was fitted on a particular
stretch of record. The captured forecast vintages begin in late August, so the
damping correction and the revision analysis rest on summer behaviour: solar is
large and predictable, demand is low, and the intensity range is narrow. In
winter there is no solar, wind sets the margin, and the forecast's errors are
wind forecast errors propagated through a merit order that looks nothing like
August's. Whether a coefficient fitted in summer holds in January is not
something summer data can answer.

Drift monitoring does not answer it either. What it does is say when the
question has become live, so that a coefficient still being applied to data
unlike anything it was fitted on is visible rather than assumed.

Three kinds of drift are reported, and they fail differently.

Feature drift: the inputs have moved. A model may still be accurate, since a
shift within its fitted range is not a problem.

Target drift: the quantity being predicted has moved. More serious, because a
model fitted on one regime's distribution has no reason to be calibrated on
another's.

Performance drift: the errors have grown. The only one that is a fault by
itself; the other two are warnings that it may be coming.

Nothing here is a hypothesis test. With one observation per period and strong
serial correlation, a p-value on these comparisons would be meaningless, and the
statistics are reported as magnitudes for a reader to judge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .load import outcome_record
from .storage import DEFAULT_ROOT

#: Quantiles used to describe a distribution compactly.
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

#: Population stability index thresholds in common use. They are conventions
#: rather than derived quantities, and are reported as labels, not verdicts.
PSI_MODEST = 0.1
PSI_LARGE = 0.25


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """How far one distribution has moved from another, in the usual index.

    Bin edges come from the reference's quantiles, so the reference is uniform
    across bins by construction and the index measures how unevenly the current
    sample falls into them. Empty bins are floored rather than dropped, since
    the logarithm is otherwise undefined and a bin the current sample never
    reaches is exactly the case worth counting.
    """
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins or len(current) < bins:
        return float("nan")

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)

    floor = 1e-6
    reference_share = np.maximum(reference_share, floor)
    current_share = np.maximum(current_share, floor)

    return float(
        np.sum((current_share - reference_share) * np.log(current_share / reference_share))
    )


def label_psi(value: float) -> str:
    """The conventional reading of an index value."""
    if not np.isfinite(value):
        return "not computable"
    if value < PSI_MODEST:
        return "stable"
    if value < PSI_LARGE:
        return "moderate shift"
    return "large shift"


def describe(values: pd.Series) -> dict[str, float]:
    """A distribution in a few numbers."""
    clean = values.dropna()
    if clean.empty:
        return {"n": 0}
    described = {"n": int(len(clean)), "mean": float(clean.mean()), "sd": float(clean.std())}
    for quantile in QUANTILES:
        described[f"q{int(quantile * 100):02d}"] = float(clean.quantile(quantile))
    return described


def compare_distributions(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Per-column drift between two stretches of the record."""
    rows = []
    for column in columns:
        if column not in reference or column not in current:
            continue
        index = population_stability_index(
            reference[column].to_numpy(dtype=float), current[column].to_numpy(dtype=float)
        )
        rows.append(
            {
                "column": column,
                "psi": index,
                "reading": label_psi(index),
                "reference_mean": float(reference[column].mean()),
                "current_mean": float(current[column].mean()),
                "reference_sd": float(reference[column].std()),
                "current_sd": float(current[column].std()),
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values("psi", ascending=False) if not frame.empty else frame


def seasonal_drift(root: Path = DEFAULT_ROOT, column: str = "actual") -> pd.DataFrame:
    """How the target's distribution differs between seasons.

    The comparison the correction's seasonal question rests on. If January's
    intensity distribution is far from August's, a coefficient fitted in August
    is being applied out of its range, whatever its interval said at the time.
    """
    outcomes = outcome_record(root)
    if outcomes.empty:
        return pd.DataFrame()

    months = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}
    period = outcomes["period_start"]

    by_season = {
        name: outcomes[period.dt.month.isin(group)][column].to_numpy(dtype=float)
        for name, group in months.items()
    }
    by_season = {name: values for name, values in by_season.items() if len(values) > 0}
    if "summer" not in by_season:
        return pd.DataFrame()

    rows = []
    for name, values in by_season.items():
        index = population_stability_index(by_season["summer"], values)
        described = describe(pd.Series(values))
        rows.append(
            {
                "season": name,
                "psi_against_summer": index,
                "reading": label_psi(index),
                **{key: described[key] for key in ("n", "mean", "sd", "q05", "q50", "q95")},
            }
        )
    return pd.DataFrame(rows)


def error_drift(root: Path = DEFAULT_ROOT, freq: str = "W") -> pd.DataFrame:
    """The published forecast's error over time, by lead band.

    Performance drift, and the only one of the three that is a fault by itself.
    Reported as a series rather than a single comparison, because an error that
    is growing steadily and one that jumped in a single week call for different
    responses.
    """
    from .load import evaluation_frame
    from .revisions import LEAD_BINS

    frame = evaluation_frame(root)
    if frame.empty:
        return pd.DataFrame()

    # Floored to the start of each bucket rather than converted to a period,
    # which drops the timezone and warns about doing so.
    frame = frame.assign(
        band=pd.cut(frame["horizon_hours"], bins=list(LEAD_BINS), right=True).astype(str),
        bucket=frame["period_start"].dt.tz_convert("UTC").dt.floor("D")
        - pd.to_timedelta(frame["period_start"].dt.dayofweek, unit="D")
        if freq == "W"
        else frame["period_start"].dt.tz_convert("UTC").dt.floor(freq),
    )
    grouped = frame.groupby(["bucket", "band"], observed=True).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
    )
    return grouped.reset_index()


def coefficient_drift(record: Path | None = None) -> pd.DataFrame:
    """The promoted coefficients across gate runs, with their spread.

    Concept drift as this project can observe it. A coefficient that stays put
    while the input distribution moves is evidence the correction is a property
    of the forecast; one that tracks the seasons is evidence it is a property of
    the weather it was fitted in.
    """
    from .gate import DEFAULT_RECORD, coefficient_history

    history = coefficient_history(record or DEFAULT_RECORD)
    if history.empty:
        return pd.DataFrame()

    summary = pd.DataFrame(
        {
            "runs": history.notna().sum(axis=1),
            "first": history.ffill(axis=1).iloc[:, 0],
            "latest": history.ffill(axis=1).iloc[:, -1],
            "min": history.min(axis=1),
            "max": history.max(axis=1),
        }
    )
    summary["range"] = summary["max"] - summary["min"]
    return summary.reset_index()


def drift_report(root: Path = DEFAULT_ROOT) -> dict[str, pd.DataFrame]:
    """Every drift view, for the command line to print."""
    return {
        "seasonal": seasonal_drift(root),
        "error": error_drift(root),
        "coefficients": coefficient_drift(),
    }
