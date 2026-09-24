"""A static evaluation page, generated from the record rather than written.

The findings in this repository move. The matched sample grows with every
capture, the gate refits weekly, and two headline figures have already reversed
as the sample grew. A page written by hand goes stale silently; one generated
from the record goes stale visibly, because it carries the date it was built and
the counts it was built from.

Everything shown here is computed by the same functions the command line uses.
A second implementation for presentation would be the easiest place for the page
and the analysis to drift apart, and a reader has no way to tell which is right.

The output is a single self-contained HTML file with no external assets, so it
can be published as a GitHub Pages site or opened from disk.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .drift import seasonal_drift
from .evaluate import compare_schedulers
from .gate import DEFAULT_RECORD, coefficient_history, evaluate_gate, load_record
from .load import coverage
from .revisions import distinct_revisions, error_by_lead, revision_autocorrelation, revision_paths
from .scheduling import Load
from .storage import DEFAULT_ROOT

STYLE = """
:root { color-scheme: light dark; --edge: #8883; }
body { font-family: system-ui, sans-serif; line-height: 1.5; max-width: 60rem;
       margin: 2rem auto; padding: 0 1rem; }
h1 { margin-bottom: 0.2rem; }
.built { color: #888; font-size: 0.9rem; margin-top: 0; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid var(--edge); padding-bottom: 0.3rem; }
p.note { color: #666; font-size: 0.92rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums;
        font-size: 0.92rem; margin: 0.6rem 0 1rem; }
th, td { text-align: right; padding: 0.35rem 0.6rem; border-bottom: 1px solid var(--edge); }
th:first-child, td:first-child { text-align: left; }
thead th { border-bottom: 2px solid var(--edge); }
.promoted { color: #2a7; } .held { color: #a52; }
"""


def _table(frame: pd.DataFrame, decimals: int = 3) -> str:
    """A frame as HTML, with numeric columns rounded and nothing else touched."""
    if frame is None or frame.empty:
        return "<p class='note'>Nothing to report yet.</p>"

    numeric = frame.select_dtypes(include="number").columns
    rounded = frame.round({name: decimals for name in numeric})
    return rounded.to_html(index=False, border=0, na_rep="—", escape=True)


def _section(title: str, note: str, body: str) -> str:
    return f"<h2>{html.escape(title)}</h2><p class='note'>{html.escape(note)}</p>{body}"


def build_report(
    root: Path = DEFAULT_ROOT,
    record: Path = DEFAULT_RECORD,
    load: Load | None = None,
) -> str:
    """Assemble the page from the current record.

    Each section states what it rests on, because the counts are the part most
    likely to be out of date by the time anyone reads it.
    """
    load = load or Load()
    built = datetime.now(UTC)
    held = coverage(root)

    sections = []

    summary = (
        f"<p>Settled half-hours: <b>{held.get('outcome_periods', 0):,}</b>. "
        f"Captured forecasts: <b>{held.get('forecast_rows', 0):,}</b> "
        f"over <b>{held.get('issues', 0):,}</b> issue times. "
        f"Periods missing from the outcome record: <b>{held.get('outcome_missing', 0)}</b>.</p>"
    )
    if "outcome_span" in held:
        start, end = held["outcome_span"]
        summary += f"<p class='note'>Outcomes span {start:%Y-%m-%d} to {end:%Y-%m-%d}.</p>"
    sections.append(_section("The record", "What everything below is computed from.", summary))

    scheduled = compare_schedulers(root, load)
    sections.append(
        _section(
            f"Decision quality — {load.describe()}",
            "Regret is the excess emissions of the periods chosen over the periods hindsight "
            "would have chosen, normalised by the saving that was available. Rows ending "
            "_matched face the same decisions as the published forecast; rows ending _full "
            "cover the whole record and are a different sample.",
            _table(scheduled.reset_index().rename(columns={"index": "forecaster"})),
        )
    )

    verdicts, note = evaluate_gate(root, record=record)
    if verdicts:
        rows = pd.DataFrame(
            [
                {
                    "band": item.band,
                    "verdict": "promoted" if item.promoted else "held back",
                    "damping": item.damping,
                    "improvement": item.improvement,
                    "interval low": item.improvement_low,
                    "n": item.n,
                    "periods": item.periods,
                    "reasons": "; ".join(item.reasons) or "—",
                }
                for item in verdicts
            ]
        )
        gate_note = (
            f"Fitted on {note.get('train_dates', 0)} days, scored on {note.get('test_dates', 0)}. "
            "A band is promoted only where the improvement's lower bound clears a tenth of a "
            "gCO2/kWh, the band carries enough held-out data, and the refitted coefficient is "
            "close to the one it replaces."
        )
        sections.append(_section("The damping correction", gate_note, _table(rows)))

    history = coefficient_history(record)
    if not history.empty and history.shape[1] > 1:
        spread = (history.max(axis=1) - history.min(axis=1)).rename("range")
        combined = pd.concat([history, spread], axis=1).reset_index()
        sections.append(
            _section(
                "Coefficient stability",
                "The same coefficient refitted on each gate run. A coefficient that settles "
                "while the inputs move is evidence the correction belongs to the forecast "
                "rather than to the weather it was fitted in.",
                _table(combined),
            )
        )

    paths = revision_paths(root)
    if not paths.empty:
        autocorrelation = revision_autocorrelation(distinct_revisions(paths))
        sections.append(
            _section(
                "Forecast revisions",
                "Correlation between a revision and the one before it, over distinct revisions. "
                "Near zero is what an efficient forecast produces; a negative value means it "
                "retraces its own movement.",
                _table(autocorrelation),
            )
        )

    lead = error_by_lead(root, matched=True)
    if not lead.empty:
        sections.append(
            _section(
                "Error against lead time",
                "Restricted to periods forecast at every lead, so the bands describe the same "
                "days and the comparison isolates horizon rather than weather.",
                _table(lead.astype({"bucket": str})),
            )
        )

    drift = seasonal_drift(root)
    if not drift.empty:
        sections.append(
            _section(
                "Seasonal drift",
                "Population stability index of the intensity distribution against summer. Under "
                "0.1 is conventionally read as stable and over 0.25 as a large shift; the "
                "thresholds are conventions rather than derived quantities. The captured "
                "forecasts begin in late August, so anything fitted on them is fitted in summer.",
                _table(drift),
            )
        )

    promoted = load_record(record)
    footer = (
        "<p class='note'>Generated by <code>gridcast report-page</code> from the committed "
        "record. Every figure is computed by the same functions the command line uses. "
        f"Promoted coefficients at build time: "
        f"{html.escape(str(promoted) if promoted else 'none')}.</p>"
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>gridcast — evaluation</title>"
        f"<style>{STYLE}</style></head><body>"
        "<h1>gridcast</h1>"
        "<p class='built'>Measuring the GB carbon intensity forecast by the decisions it "
        f"produces. Built {built:%Y-%m-%d %H:%M} UTC.</p>"
        + "".join(sections)
        + footer
        + "</body></html>"
    )


def write_report(
    destination: Path,
    root: Path = DEFAULT_ROOT,
    record: Path = DEFAULT_RECORD,
    load: Load | None = None,
) -> Path:
    """Build the page and write it, creating the directory if needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_report(root, record, load), encoding="utf-8")
    return destination
