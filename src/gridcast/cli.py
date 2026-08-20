"""Command-line interface.

``snapshot`` is what the scheduled workflow runs. ``backfill`` is run once by
hand to recover historical realised values.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from .client import CarbonIntensityClient, CarbonIntensityError, format_timestamp, window_range
from .parse import ParseError, parse_generation, parse_intensity
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
            "generation",
            "/generation",
            client.current_generation,
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

    args = parser.parse_args(argv)

    if args.command == "snapshot":
        return snapshot(args.root, client=client)
    return backfill(args.root, args.start, args.end, client=client)


if __name__ == "__main__":
    raise SystemExit(main())
