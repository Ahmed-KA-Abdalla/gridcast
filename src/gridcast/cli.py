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

from .client import CarbonIntensityClient, CarbonIntensityError, format_timestamp, window_range
from .evaluate import backtest_baselines, compare_at_matched_leads, compare_schedulers
from .load import coverage, evaluation_frame
from .parse import ParseError, parse_generation, parse_intensity
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

    args = parser.parse_args(argv)

    if args.command == "snapshot":
        return snapshot(args.root, client=client)
    if args.command == "report":
        return report(args.root)
    if args.command == "compare":
        return compare(args.root)
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
