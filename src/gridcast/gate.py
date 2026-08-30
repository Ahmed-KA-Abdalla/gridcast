"""The promotion gate.

A correction is not established by having been significant once. The coefficient
was fitted on four days of late August, and whether it describes the published
forecast or that week's weather is a question only time answers. The gate is
what asks it repeatedly: refit on the current record, score out of sample, and
refuse the correction unless it still clears the bar.

Three conditions, all of which must hold for a band to be promoted.

The improvement must be positive out of sample and its bootstrap interval must
exclude zero. A point estimate is not evidence.

The band must carry enough held-out observations and enough distinct target
periods for the interval to mean anything. Rows within a period are correlated,
so a band can have many rows and few independent situations.

The refitted coefficient must be of the same sign and the same rough size as the
one it replaces. A coefficient that swings from 0.46 to 0.93 between refits is
describing the sample rather than the forecast, whatever its interval says.

The gate reports rather than decides quietly: every band that fails carries the
reason, so a build that fails says what changed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .correction import evaluate_with_intervals
from .storage import DEFAULT_ROOT

#: Where the promoted coefficients are kept between runs.
DEFAULT_RECORD = Path("docs/promoted.json")

#: Runs of coefficient history retained. Enough to see whether a coefficient is
#: settling or drifting, without the file growing without bound.
HISTORY_LENGTH = 24


@dataclass(frozen=True)
class Thresholds:
    """What a band must clear to be promoted."""

    #: Held-out observations in the band.
    min_observations: int = 100
    #: Distinct target periods, which is the sample size the interval rests on.
    min_periods: int = 30
    #: The improvement's lower bound must exceed this, in gCO2/kWh.
    min_improvement_low: float = 0.0
    #: A refitted coefficient may differ from the promoted one by this much
    #: before it is treated as unstable rather than merely noisy.
    max_coefficient_drift: float = 0.25


@dataclass
class BandVerdict:
    """The outcome for one lead band, with the reasons it failed."""

    band: str
    promoted: bool
    damping: float
    improvement: float
    improvement_low: float
    n: int
    periods: int
    previous_damping: float | None = None
    reasons: list[str] = field(default_factory=list)


def load_record(path: Path = DEFAULT_RECORD) -> dict[str, float]:
    """The coefficients promoted by the last passing run, if any."""
    if not path.exists():
        return {}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {band: float(value) for band, value in stored.get("coefficients", {}).items()}


def load_history(path: Path = DEFAULT_RECORD) -> list[dict]:
    """Every retained run's coefficients, oldest first."""
    if not path.exists():
        return []
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    history = stored.get("history", [])
    return history if isinstance(history, list) else []


def coefficient_history(path: Path = DEFAULT_RECORD) -> pd.DataFrame:
    """Each band's fitted coefficient across runs.

    A coefficient settling near one value over successive refits is evidence no
    single run can give, since one run's interval speaks only to that run's
    sample. A coefficient wandering is the opposite, and it will not always trip
    the drift check, which compares consecutive runs rather than the trend.
    """
    history = load_history(path)
    if not history:
        return pd.DataFrame()

    rows = []
    for entry in history:
        for band, value in entry.get("fitted", {}).items():
            rows.append({"generated": entry.get("generated"), "band": band, "damping": value})
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    return frame.pivot_table(index="band", columns="generated", values="damping")


def write_record(
    verdicts: list[BandVerdict], path: Path = DEFAULT_RECORD, generated: str | None = None
) -> Path:
    """Store the promoted coefficients, and append this run to the history.

    Both the promoted coefficients and every band's fitted coefficient are
    kept. A band that fails its interval still yields a coefficient, and whether
    that coefficient is stable is exactly what tells you later whether the band
    was failing for want of data or for want of an effect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = generated or pd.Timestamp.now(tz="UTC").isoformat()

    history = load_history(path)
    history.append(
        {
            "generated": stamp,
            "fitted": {v.band: v.damping for v in verdicts},
            "promoted": [v.band for v in verdicts if v.promoted],
        }
    )
    history = history[-HISTORY_LENGTH:]

    payload = {
        "generated": stamp,
        "coefficients": {v.band: v.damping for v in verdicts if v.promoted},
        "verdicts": [asdict(v) for v in verdicts],
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _reasons(row: pd.Series, previous: float | None, thresholds: Thresholds) -> list[str]:
    reasons: list[str] = []

    if int(row["n"]) < thresholds.min_observations:
        reasons.append(f"{int(row['n'])} observations, needs {thresholds.min_observations}")

    periods = int(row.get("periods", 0) or 0)
    if periods < thresholds.min_periods:
        reasons.append(f"{periods} distinct periods, needs {thresholds.min_periods}")

    low = row.get("improvement_low")
    if low is None or not np.isfinite(low):
        reasons.append("no interval could be computed")
    elif low <= thresholds.min_improvement_low:
        reasons.append(f"interval includes zero (low {low:.3f})")

    if previous is not None:
        drift = abs(float(row["damping"]) - previous)
        if drift > thresholds.max_coefficient_drift:
            reasons.append(
                f"coefficient moved {drift:.2f} from {previous:.2f}, limit "
                f"{thresholds.max_coefficient_drift:.2f}"
            )
        elif np.sign(row["damping"]) != np.sign(previous):
            reasons.append("coefficient changed sign")

    return reasons


def evaluate_gate(
    root: Path = DEFAULT_ROOT,
    record: Path = DEFAULT_RECORD,
    thresholds: Thresholds | None = None,
    train_fraction: float = 0.6,
) -> tuple[list[BandVerdict], dict[str, object]]:
    """Refit, score, and decide which bands may be promoted."""
    thresholds = thresholds or Thresholds()
    summary, _, note = evaluate_with_intervals(root, train_fraction)
    if summary.empty:
        return [], note

    previous = load_record(record)
    verdicts = []
    for _, row in summary.iterrows():
        band = str(row["band"])
        prior = previous.get(band)
        reasons = _reasons(row, prior, thresholds)
        verdicts.append(
            BandVerdict(
                band=band,
                promoted=not reasons,
                damping=float(row["damping"]),
                improvement=float(row["improvement"]),
                improvement_low=float(row.get("improvement_low", np.nan)),
                n=int(row["n"]),
                periods=int(row.get("periods", 0) or 0),
                previous_damping=prior,
                reasons=reasons,
            )
        )

    return sorted(verdicts, key=lambda v: v.band), note


def gate_passes(verdicts: list[BandVerdict], previous: dict[str, float]) -> tuple[bool, str]:
    """Whether the build should pass.

    A band failing to promote is not itself a failure: most bands never clear
    the bar, and one that has never been promoted has nothing to regress from.
    The build fails only when a band that *was* promoted no longer is, since
    that is a claim in the repository ceasing to hold.

    A promoted band that disappears from the evaluation altogether counts as
    regressing. It usually means the band no longer has enough held-out data to
    be scored, which leaves the recorded coefficient standing on nothing —
    materially the same situation as failing outright, and easy to miss because
    there is no row reporting it.
    """
    if not verdicts:
        return True, "nothing to evaluate yet"

    failures = []
    evaluated = {v.band for v in verdicts}

    for item in verdicts:
        if item.band in previous and not item.promoted:
            failures.append(f"{item.band}: {', '.join(item.reasons)}")

    for band in previous:
        if band not in evaluated:
            failures.append(f"{band}: no longer evaluated at all")

    if failures:
        return False, "previously promoted bands no longer qualify — " + "; ".join(failures)

    promoted = [v.band for v in verdicts if v.promoted]
    if promoted:
        return True, f"promoted: {', '.join(promoted)}"
    return True, "no band qualifies yet, and none previously did"
