"""The promotion gate.

A correction is not established by having been significant once. The gate is
what asks repeatedly: refit on the current record, score out of sample, and
refuse the correction unless it still clears the bar.

Four conditions, all of which must hold for a band to be promoted.

The pooled improvement over the held-out blocks must clear a minimum worth
claiming, with the lower bound of its bootstrap interval above it. A point
estimate is not evidence.

The band must carry enough held-out observations and enough distinct target
periods for the interval to mean anything. Rows within a period are correlated,
so a band can have many rows and few independent situations.

The improvement must repeat: positive in most of the held-out blocks, not
carried by one. A single train-test split fooled this gate three times — the
symmetric correction at short lead was significant on one split and
significantly harmful on the next — and a pooled interval cannot see that,
because it measures sampling noise within the held-out data, not the variation
between one stretch of weeks and another.

The coefficient must be stable, both across the blocks' fits within this run
and against the coefficient it replaces from the last run. A coefficient that
swings between refits is describing the sample rather than the forecast,
whatever its interval says. The within-run check is the one that matters most:
it catches an unstable band on its first evaluation instead of weeks later.

The gate reports rather than decides quietly: every band that fails carries the
reason, so a build that fails says what changed.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .correction import DEFAULT_MODEL, BandDamping, evaluate_rolling
from .storage import DEFAULT_ROOT

#: Where the promoted coefficients are kept between runs.
DEFAULT_RECORD = Path("docs/promoted.json")

#: The model the gate evaluates and records.
MODEL = DEFAULT_MODEL

#: The model assumed for a record that names none, which is every record
#: written before the asymmetric correction existed.
LEGACY_MODEL = "symmetric"

#: Runs of coefficient history retained. Enough to see whether a coefficient is
#: settling or drifting, without the file growing without bound.
HISTORY_LENGTH = 24


@dataclass(frozen=True)
class Thresholds:
    """What a band must clear to be promoted. Conventions, not derived values."""

    #: Held-out observations in the band.
    min_observations: int = 100
    #: Distinct target periods, which is the sample size the interval rests on.
    min_periods: int = 30
    #: The improvement's lower bound must exceed this, in gCO2/kWh. Not zero:
    #: an interval clearing zero establishes that an effect exists, not that it
    #: is large enough to be worth claiming. A coefficient near zero applies
    #: almost no correction, so its improvement is tiny and its variance is
    #: tiny with it, and the interval clears zero while the effect is
    #: negligible. A tenth of a gCO2/kWh is under one per cent of the published
    #: forecast's error at any lead.
    min_improvement_low: float = 0.1
    #: A refitted coefficient may differ from the promoted one by this much
    #: before it is treated as unstable rather than merely noisy.
    max_coefficient_drift: float = 0.25
    #: Held-out blocks needed before consistency can be judged at all.
    min_folds: int = 3
    #: Share of blocks in which the correction must have helped. Four in five.
    min_positive_share: float = 0.8
    #: How far a coefficient may range across the blocks' fits in one run. The
    #: same tolerance as between runs, applied where it can see the problem
    #: immediately.
    max_fold_range: float = 0.25


@dataclass
class BandVerdict:
    """The outcome for one lead band, with the reasons it failed."""

    band: str
    promoted: bool
    damping_up: float
    damping_down: float
    improvement: float
    improvement_low: float
    improvement_high: float
    n: int
    periods: int
    folds: int = 0
    folds_positive: int = 0
    up_range: float = float("nan")
    down_range: float = float("nan")
    previous_up: float | None = None
    previous_down: float | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def damping(self) -> BandDamping:
        return BandDamping(self.damping_up, self.damping_down)

    @property
    def previous(self) -> BandDamping | None:
        if self.previous_up is None or self.previous_down is None:
            return None
        return BandDamping(self.previous_up, self.previous_down)


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return stored if isinstance(stored, dict) else {}


def load_record(path: Path = DEFAULT_RECORD) -> dict[str, BandDamping]:
    """The coefficients promoted by the last run, if any.

    A record written before the asymmetric correction stores one number per
    band. It reads back as equal upward and downward coefficients, which is
    exactly the correction that number described.
    """
    stored = _read(path).get("coefficients", {})
    if not isinstance(stored, dict):
        return {}
    return {band: BandDamping.coerce(value) for band, value in stored.items()}


def record_model(path: Path = DEFAULT_RECORD) -> str | None:
    """The model a record was written under, or None if there is no record."""
    stored = _read(path)
    if not stored:
        return None
    return str(stored.get("model", LEGACY_MODEL))


def load_history(path: Path = DEFAULT_RECORD) -> list[dict]:
    """Every retained run's coefficients, oldest first."""
    history = _read(path).get("history", [])
    return history if isinstance(history, list) else []


def coefficient_history(path: Path = DEFAULT_RECORD, model: str | None = None) -> pd.DataFrame:
    """Each band's fitted coefficients across runs, one row per direction.

    A coefficient settling near one value over successive refits is evidence no
    single run can give, since one run's interval speaks only to that run's
    sample. A coefficient wandering is the opposite.

    ``model`` restricts the history to runs made under one model. Mixing them
    would show the switch from a symmetric to an asymmetric fit as drift, which
    it is not.
    """
    rows = []
    for entry in load_history(path):
        entry_model = str(entry.get("model", LEGACY_MODEL))
        if model is not None and entry_model != model:
            continue
        for band, value in entry.get("fitted", {}).items():
            coefficient = BandDamping.coerce(value)
            for direction in ("up", "down"):
                rows.append(
                    {
                        "generated": entry.get("generated"),
                        "band": band,
                        "direction": direction,
                        "damping": getattr(coefficient, direction),
                    }
                )
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    return frame.pivot_table(index=["band", "direction"], columns="generated", values="damping")


def write_record(
    verdicts: list[BandVerdict],
    path: Path = DEFAULT_RECORD,
    generated: str | None = None,
    model: str = MODEL,
) -> Path:
    """Store the promoted coefficients, and append this run to the history.

    Both the promoted coefficients and every band's fitted coefficients are
    kept. A band that fails still yields coefficients, and whether they are
    stable is exactly what tells you later whether the band was failing for want
    of data or for want of an effect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = generated or pd.Timestamp.now(tz="UTC").isoformat()

    history = load_history(path)
    history.append(
        {
            "generated": stamp,
            "model": model,
            "fitted": {v.band: v.damping.to_dict() for v in verdicts},
            "promoted": [v.band for v in verdicts if v.promoted],
        }
    )
    history = history[-HISTORY_LENGTH:]

    payload = {
        "generated": stamp,
        "model": model,
        "coefficients": {v.band: v.damping.to_dict() for v in verdicts if v.promoted},
        "verdicts": [asdict(v) for v in verdicts],
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _finite(value: object) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _reasons(row: pd.Series, previous: BandDamping | None, thresholds: Thresholds) -> list[str]:
    reasons: list[str] = []
    up, down = float(row["damping_up"]), float(row["damping_down"])

    # Coefficients of exactly zero apply no correction at all, and the interval
    # collapses to a point. Checked separately from the effect-size threshold so
    # the reason given names the actual problem.
    if up == 0.0 and down == 0.0:
        reasons.append("coefficient is zero, so no correction is applied")

    if int(row["n"]) < thresholds.min_observations:
        reasons.append(f"{int(row['n'])} observations, needs {thresholds.min_observations}")

    periods = int(row.get("periods", 0) or 0)
    if periods < thresholds.min_periods:
        reasons.append(f"{periods} distinct periods, needs {thresholds.min_periods}")

    low = row.get("improvement_low")
    high = row.get("improvement_high")
    if not _finite(low):
        reasons.append("no interval could be computed")
    elif low <= thresholds.min_improvement_low:
        # Three distinct outcomes, which the reason has to tell apart. An
        # interval wholly below zero says the correction is reliably harmful,
        # not that it is unproven: the evidence is as strong as a promotion's
        # and points the other way. Reporting that as "includes zero" invites
        # the reader to wait for more data, when what is called for is
        # withdrawing the band.
        if _finite(high) and high < 0:
            reasons.append(f"significantly worse out of sample (interval {low:.3f} to {high:.3f})")
        elif low <= 0:
            reasons.append(f"interval includes zero (low {low:.3f})")
        else:
            reasons.append(
                f"improvement too small to claim (low {low:.3f}, "
                f"needs {thresholds.min_improvement_low:.2f})"
            )

    folds = int(row.get("folds", 0) or 0)
    if folds < thresholds.min_folds:
        reasons.append(
            f"{folds} held-out blocks, needs {thresholds.min_folds} to judge consistency"
        )
    else:
        needed = math.ceil(thresholds.min_positive_share * folds)
        positive = int(row.get("folds_positive", 0) or 0)
        if positive < needed:
            reasons.append(f"helped in only {positive} of {folds} held-out blocks, needs {needed}")

        spans = {
            direction: float(row.get(f"{direction}_range", np.nan)) for direction in ("up", "down")
        }
        unstable = {d: s for d, s in spans.items() if _finite(s) and s > thresholds.max_fold_range}
        if unstable:
            detail = ", ".join(f"{d} ranged {s:.2f}" for d, s in unstable.items())
            reasons.append(
                f"coefficient unstable across held-out blocks ({detail}, "
                f"limit {thresholds.max_fold_range:.2f})"
            )

    if previous is not None:
        for direction, now, before in (("up", up, previous.up), ("down", down, previous.down)):
            drift = abs(now - before)
            if drift > thresholds.max_coefficient_drift:
                reasons.append(
                    f"{direction} coefficient moved {drift:.2f} from {before:.2f}, limit "
                    f"{thresholds.max_coefficient_drift:.2f}"
                )
            # A zero coefficient has no sign to change: it means too little data
            # in that direction, which the other checks already speak to.
            elif now != 0.0 and before != 0.0 and np.sign(now) != np.sign(before):
                reasons.append(f"{direction} coefficient changed sign")

    return reasons


def evaluate_gate(
    root: Path = DEFAULT_ROOT,
    record: Path = DEFAULT_RECORD,
    thresholds: Thresholds | None = None,
    model: str = MODEL,
) -> tuple[list[BandVerdict], dict[str, object]]:
    """Refit, score, and decide which bands may be promoted."""
    thresholds = thresholds or Thresholds()
    summary, _, _, note = evaluate_rolling(root, model)
    if summary.empty:
        return [], note

    previous = load_record(record)
    previous_model = record_model(record)
    # Coefficients recorded under another model describe a different
    # correction, so there is nothing for this run's to have drifted from.
    comparable = previous_model in (None, model)
    note = {**note, "previous_model": previous_model}

    verdicts = []
    for _, row in summary.iterrows():
        band = str(row["band"])
        prior = previous.get(band) if comparable else None
        reasons = _reasons(row, prior, thresholds)
        verdicts.append(
            BandVerdict(
                band=band,
                promoted=not reasons,
                damping_up=float(row["damping_up"]),
                damping_down=float(row["damping_down"]),
                improvement=float(row["improvement"]),
                improvement_low=float(row.get("improvement_low", np.nan)),
                improvement_high=float(row.get("improvement_high", np.nan)),
                n=int(row["n"]),
                periods=int(row.get("periods", 0) or 0),
                folds=int(row.get("folds", 0) or 0),
                folds_positive=int(row.get("folds_positive", 0) or 0),
                up_range=float(row.get("up_range", np.nan)),
                down_range=float(row.get("down_range", np.nan)),
                previous_up=prior.up if prior is not None else None,
                previous_down=prior.down if prior is not None else None,
                reasons=reasons,
            )
        )

    return sorted(verdicts, key=lambda v: v.band), note


def gate_passes(
    verdicts: list[BandVerdict],
    previous: dict[str, BandDamping],
    previous_model: str | None = None,
    model: str = MODEL,
) -> tuple[bool, str]:
    """Whether the build should pass.

    A band failing to promote is not itself a failure: most bands never clear
    the bar, and one that has never been promoted has nothing to regress from.
    The build fails only when a band that *was* promoted no longer is, since
    that is a claim in the repository ceasing to hold.

    A promoted band that disappears from the evaluation altogether counts as
    regressing. It usually means the band no longer has enough held-out data to
    be scored, which leaves the recorded coefficient standing on nothing.

    A change of model is not a regression. The previous record's promotions
    were claims about a different correction; the change that replaced it is a
    commit, reviewed as one, and the bands it drops are listed in the message so
    the replacement is visible rather than silent. Failing the build on it would
    turn every deliberate model change into a red run that means nothing.
    """
    if not verdicts:
        return True, "nothing to evaluate yet"

    promoted = [v.band for v in verdicts if v.promoted]
    promoted_text = ", ".join(promoted) if promoted else "nothing"

    if previous and previous_model is not None and previous_model != model:
        dropped = sorted(set(previous) - set(promoted))
        dropped_text = f"; not carried over: {', '.join(dropped)}" if dropped else ""
        return True, (
            f"record was written under the {previous_model} model, so its promotions do not "
            f"carry over to the {model} one. Promoted under {model}: {promoted_text}"
            f"{dropped_text}"
        )

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

    if promoted:
        return True, f"promoted: {promoted_text}"
    return True, "no band qualifies yet, and none previously did"
