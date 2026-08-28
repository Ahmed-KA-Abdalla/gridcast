"""Command-line interface.

``snapshot`` is what the scheduled workflow runs. ``backfill`` is run once by
hand to recover historical realised values.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

from .audit import (
    cost_of_a_near_miss,
    decision_detail,
    ordering_quality,
    reference_availability,
)
from .client import CarbonIntensityClient, CarbonIntensityError, format_timestamp, window_range
from .correction import evaluate_with_intervals
from .evaluate import backtest_baselines, compare_at_matched_leads, compare_schedulers
from .load import coverage, evaluation_frame
from .parse import ParseError, parse_generation, parse_intensity
from .revisions import (
    distinct_revisions,
    error_by_lead,
    path_summary,
    refresh_cadence,
    revision_autocorrelation,
    revision_paths,
    revision_predicts_error,
)
from .scheduling import Load
from .schema import ValidationError, validate_generation, validate_intensity
from .storage import DEFAULT_ROOT, write_snapshot

#: Raised when a payload is structurally sound as JSON but not as data. Both
#: mean the same thing operationally: do not store it, and say why.
BAD_PAYLOAD = (ParseError, ValidationError, KeyError, TypeError, ValueError)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def snapshot(root: Path, client: CarbonIntensityClient | None = None) -> int:
    """Capture the current period, the forward forecast and the generation mix.

    The forward forecast is the reason this runs on a schedule: the API
    overwrites forecasts as they are revised and serves no history of them.

    A failure in one endpoint does not discard the others. The exit code is
    non-zero if any endpoint failed, so a scheduled run that captures two of
    three is still visibly degraded.
    """
    client = client or CarbonIntensityClient()
    captured_at = _now()
    stamp = format_timestamp(captured_at)

    tasks = (
        ("intensity", "/intensity", client.current_intensity, parse_intensity, validate_intensity),
        (
            "forecast_fw48h",
            f"/intensity/{stamp}/fw48h",
            lambda: client.forecast_48h(captured_at),
            parse_intensity,
            validate_intensity,
        ),
        (
            "outcomes_pt24h",
            f"/intensity/{stamp}/pt24h",
            lambda: client.past_24h(captured_at),
            parse_intensity,
            validate_intensity,
        ),
        (
            "generation",
            "/generation",
            client.current_generation,
            parse_generation,
            validate_generation,
        ),
        (
            "generation_pt24h",
            f"/generation/{format_timestamp(captured_at - dt.timedelta(hours=24))}/{stamp}",
            lambda: client.generation_range(captured_at - dt.timedelta(hours=24), captured_at),
            parse_generation,
            validate_generation,
        ),
    )

    failures = 0
    for kind, endpoint, fetch, parse, validate in tasks:
        try:
            payload = fetch()
        except CarbonIntensityError as exc:
            print(f"fetch failed for {kind}: {exc}", file=sys.stderr)
            failures += 1
            continue

        try:
            validate(parse(payload, captured_at))
        except BAD_PAYLOAD as exc:
            print(f"{kind} rejected: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue

        print(f"wrote {write_snapshot(payload, kind, endpoint, captured_at, root=root)}")

    return 1 if failures else 0


def backfill(
    root: Path,
    start: dt.datetime,
    end: dt.datetime,
    client: CarbonIntensityClient | None = None,
) -> int:
    """Fetch realised intensity and generation over a historical range.

    Requests are split into fourteen-day windows because the range endpoint
    refuses anything longer. This recovers realised values only; forecasts as
    issued cannot be recovered, for the reason given in ``snapshot``.
    """
    client = client or CarbonIntensityClient()
    captured_at = _now()

    for window_start, window_end in window_range(start, end):
        window = f"{format_timestamp(window_start)}/{format_timestamp(window_end)}"
        for stem_kind, stem, fetch in (
            ("intensity_range", "/intensity", client.intensity_range),
            ("generation_range", "/generation", client.generation_range),
        ):
            # Every window in a run shares one capture minute, so the window
            # start has to appear in the name or each window overwrites the last.
            kind = f"{stem_kind}_{window_start:%Y%m%d}"
            endpoint = f"{stem}/{window}"
            try:
                payload = fetch(window_start, window_end)
            except CarbonIntensityError as exc:
                print(f"fetch failed for {endpoint}: {exc}", file=sys.stderr)
                return 1
            print(f"wrote {write_snapshot(payload, kind, endpoint, captured_at, root=root)}")

    return 0


def report(root: Path) -> int:
    """Print what the store holds and how the published forecast has scored.

    Errors are broken down by lead time because a single average over all
    horizons is close to meaningless: a forecast half an hour ahead and one
    forty-eight hours ahead are different problems.
    """
    summary = coverage(root)
    print(f"settled periods:   {summary['outcome_periods']}")
    print(f"forecast rows:     {summary['forecast_rows']}")
    print(f"distinct issues:   {summary['issues']}")
    if "outcome_span" in summary:
        start, end = summary["outcome_span"]
        print(f"outcome span:      {start:%Y-%m-%d} to {end:%Y-%m-%d}")
        print(f"periods missing:   {summary['outcome_missing']}")
    if "max_horizon_hours" in summary:
        print(f"longest lead:      {summary['max_horizon_hours']:.1f} h")

    frame = evaluation_frame(root)
    if frame.empty:
        print("\nnothing scoreable yet: no forecast has a settled outcome")
        return 0

    bins = [0, 1, 3, 6, 12, 24, 48]
    frame = frame.assign(bucket=pd.cut(frame["horizon_hours"], bins=bins, right=True))
    scored = frame.groupby("bucket", observed=True).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        bias=("error", "mean"),
    )
    print("\npublished forecast error by lead time (gCO2/kWh)")
    print(scored.round(2).to_string())
    return 0


def compare(root: Path) -> int:
    """Score the published forecast against the seasonal baselines.

    Two tables, answering different questions. The backtest covers every period
    ever observed, which a seasonal prediction can do because it does not depend
    on lead time. The matched comparison covers only periods where a forecast
    was captured, which is the sample that permits a head-to-head.
    """
    backtest = backtest_baselines(root)
    if backtest.empty:
        print("no outcomes stored")
        return 0

    print("baselines over the whole outcome record (gCO2/kWh)")
    print(backtest.round(2).to_string())

    matched = compare_at_matched_leads(root)
    if matched.empty:
        print("\nno rows where a captured forecast and a computable baseline coincide")
        return 0

    print("\npublished forecast against the baselines, matched rows only")
    print(matched.round(2).to_string())
    return 0


def schedule(root: Path, load: Load) -> int:
    """Score forecasts by the scheduling decisions they produce.

    Regret is the excess emissions of the periods chosen over the periods
    hindsight would have chosen. ``captured_fraction`` normalises it by what was
    available to be saved, since on a flat day no choice is much worse than any
    other and an unnormalised figure would flatter any scheduler.
    """
    scored = compare_schedulers(root, load)
    if scored.empty:
        print("no settled outcomes to schedule against")
        return 0

    print(f"decision: {load.describe()}")
    print("all figures gCO2/kWh averaged over the load\n")
    print(scored.round(3).to_string())
    print(
        "\nRead the first rows together: 'published' and the '_matched' baselines "
        "face\nthe same decisions, so differences between them are differences "
        "between\nforecasters. The '_full' rows cover the whole record and are a "
        "different\nsample — compare mean_available before comparing anything else."
    )
    return 0


def audit(root: Path, load: Load) -> int:
    """Open the decision comparison up so its result can be checked.

    Prints, in order: the margin by which the seasonal references had settled
    before each issue, which rules the availability gate in or out as an
    explanation; how far each method's choice sat from the optimum; and the
    shape of the windows, which decides whether a near-miss is cheap.
    """
    print("seasonal references, hours settled before the issue time")
    print(reference_availability(load).round(1).to_string(index=False))
    print("\nA positive margin at every lag and horizon means no baseline")
    print("prediction could have used an unsettled reference.")

    detail = decision_detail(root, load)
    if detail.empty:
        print("\nno matched decisions to audit")
        return 0

    print(f"\n{len(detail)} matched decisions: {load.describe()}")
    print("\ndistance from the true optimum, in half-hour periods")
    print(ordering_quality(detail).round(3).to_string())

    print("\nwindow shape (gCO2/kWh)")
    print(cost_of_a_near_miss(detail).round(2).to_string())

    worst = detail.nlargest(5, "published_cost")
    print("\nthe five costliest published decisions")
    columns = [
        "window_start",
        "published_start",
        "baseline_start",
        "oracle_start",
        "published_cost",
        "baseline_cost",
        "oracle_cost",
        "forecast_bias",
    ]
    numeric = worst[columns].select_dtypes(include="number").columns
    print(worst[columns].round({name: 2 for name in numeric}).to_string(index=False))
    return 0


def revisions(root: Path) -> int:
    """Report how the published forecast moves as its target approaches.

    Three things. Whether accuracy improves with proximity, measured on periods
    forecast at every lead so that horizon rather than weather is what varies.
    Whether successive revisions are correlated, which an efficient forecast
    would not produce. And whether a revision anticipates the error still
    remaining, which would be directly exploitable.
    """
    paths = revision_paths(root)
    if paths.empty:
        print("no captured forecasts to examine")
        return 0

    distinct = distinct_revisions(paths)
    summary = path_summary(paths)
    unchanged = float(paths["revision"].eq(0).mean())

    print(f"{len(summary)} periods, {len(paths)} captures of them")
    print(f"captures finding no change:  {unchanged:.1%}")
    print(f"distinct forecast values:    {len(distinct)}")
    print(f"median captures per period:  {summary['issues'].median():.0f}")
    print(f"median total movement:       {summary['total_movement'].median():.1f} gCO2/kWh")
    print(f"median net movement:         {summary['net_movement'].abs().median():.1f}")
    print(f"median movement retraced:    {summary['wander'].median():.1f}")

    cadence = refresh_cadence(distinct)
    if not cadence.empty:
        print("\nhow often the forecast is actually revised")
        print(cadence.round(2).to_string(index=False))

    matched = error_by_lead(root, matched=True)
    if not matched.empty:
        print("\nerror against lead time, periods forecast at every lead")
        print(matched.round(2).to_string(index=False))

    scored = revision_autocorrelation(distinct)
    if not scored.empty:
        print("\ncorrelation between a revision and the one before it")
        print(scored.round(3).to_string(index=False))
        print("Computed over distinct revisions only; captures that changed nothing")
        print("dilute the figure towards zero. Near zero is what an efficient forecast")
        print("produces. Positive means it adjusts gradually towards news it already")
        print("has. Near -0.5 means it jitters around a level rather than converging,")
        print("which the retraced-movement figure above should corroborate.")

    predictive = revision_predicts_error(root)
    if not predictive.empty:
        print("\ncorrelation between a revision and the error still remaining")
        print(predictive.round(3).to_string(index=False))

    return 0


def correct(root: Path, train_fraction: float) -> int:
    """Test whether damping the published forecast's revisions improves it.

    The coefficient is fitted on earlier dates and scored on later ones. A
    positive coefficient means part of each revision is undone; the improvement
    column is what that bought out of sample, so a negative value there is a
    result and not a failure.
    """
    summary, damping, note = evaluate_with_intervals(root, train_fraction)
    if summary.empty:
        print(note.get("reason", "not enough captured revisions to fit a correction"))
        return 0

    print(
        f"fitted on {note['train_dates']} days ({note['train_rows']} revisions), "
        f"scored on {note['test_dates']} days ({note['test_rows']})"
    )
    print("\nout-of-sample error, gCO2/kWh, with 95% intervals")
    print(summary.round(3).to_string(index=False))
    print("\nA positive damping coefficient means the forecast overshoots and part")
    print("of each revision is better undone. Intervals come from a paired bootstrap")
    print("resampled by target period, so a band is only worth believing where")
    print("improvement_low is above zero.")
    return 0


def _date(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=dt.UTC)


def main(argv: list[str] | None = None, client: CarbonIntensityClient | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gridcast", description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT, help="directory for raw snapshots"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("snapshot", help="capture the current period and forward forecast")

    backfill_parser = sub.add_parser("backfill", help="fetch a historical range")
    backfill_parser.add_argument("--start", type=_date, required=True, help="YYYY-MM-DD")
    backfill_parser.add_argument("--end", type=_date, required=True, help="YYYY-MM-DD")

    sub.add_parser("report", help="summarise the store and score the published forecast")
    sub.add_parser("compare", help="score the published forecast against the baselines")

    schedule_parser = sub.add_parser("schedule", help="score forecasts by decision quality")
    schedule_parser.add_argument(
        "--periods", type=int, default=4, help="half-hours the load occupies"
    )
    schedule_parser.add_argument(
        "--window", type=float, default=24.0, help="hours within which it must run"
    )
    schedule_parser.add_argument(
        "--interruptible", action="store_true", help="allow the load to be split across the window"
    )

    sub.add_parser("revisions", help="examine how forecasts move as targets approach")

    correct_parser = sub.add_parser("correct", help="test a damped-revision correction")
    correct_parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.6,
        help="share of dates used to fit the coefficient",
    )

    audit_parser = sub.add_parser("audit", help="inspect the decision comparison")
    audit_parser.add_argument("--periods", type=int, default=4)
    audit_parser.add_argument("--window", type=float, default=24.0)

    args = parser.parse_args(argv)

    if args.command == "snapshot":
        return snapshot(args.root, client=client)
    if args.command == "report":
        return report(args.root)
    if args.command == "compare":
        return compare(args.root)
    if args.command == "revisions":
        return revisions(args.root)
    if args.command == "correct":
        return correct(args.root, args.train_fraction)
    if args.command == "audit":
        return audit(args.root, Load(periods=args.periods, window_hours=args.window))
    if args.command == "schedule":
        return schedule(
            args.root,
            Load(
                periods=args.periods,
                window_hours=args.window,
                contiguous=not args.interruptible,
            ),
        )
    return backfill(args.root, args.start, args.end, client=client)


if __name__ == "__main__":
    raise SystemExit(main())
