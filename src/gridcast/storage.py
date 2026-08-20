"""Persistence of raw API payloads.

Payloads are stored verbatim inside a small envelope recording what was
requested and when the response arrived. Parsing is deliberately not applied
before storage: the parser will change over the life of the project, and a
stored derivative cannot be re-derived under a corrected parser, whereas a
stored payload can.

Filenames are built from a short ``kind`` label rather than from the request
path. Request paths contain colons, which NTFS reads as an alternate data
stream separator: writing to such a path appears to succeed and produces a file
that directory listings do not show. The label is validated rather than
sanitised, so an unexpected value fails at the call site instead of being
quietly rewritten into something else.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Iterator
from pathlib import Path

DEFAULT_ROOT = Path("data/raw")

ENVELOPE_VERSION = 2

#: Kinds must be safe on every filesystem the project runs on, which in
#: practice means Windows: no colons, no path separators, no spaces.
KIND_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def _check_kind(kind: str) -> str:
    if not KIND_PATTERN.fullmatch(kind):
        raise ValueError(
            f"invalid snapshot kind {kind!r}: expected lower-case words joined by underscores"
        )
    return kind


def snapshot_path(root: Path, kind: str, captured_at: dt.datetime) -> Path:
    """Return the file a snapshot belongs in, partitioned by UTC date.

    The filename carries the capture time to the minute, which is finer than the
    half-hourly cadence and so cannot collide in normal operation.
    """
    _check_kind(kind)
    moment = captured_at.astimezone(dt.UTC)
    return (
        root
        / f"{moment:%Y}"
        / f"{moment:%m}"
        / f"{moment:%d}"
        / f"{kind}_{moment:%Y%m%dT%H%M}Z.json"
    )


def write_snapshot(
    payload: dict,
    kind: str,
    endpoint: str,
    captured_at: dt.datetime,
    root: Path = DEFAULT_ROOT,
) -> Path:
    """Write a payload with its provenance envelope, returning the path.

    ``kind`` names the file; ``endpoint`` is the request path and is recorded in
    the envelope, where its punctuation is harmless.
    """
    path = snapshot_path(root, kind, captured_at)
    path.parent.mkdir(parents=True, exist_ok=True)

    envelope = {
        "envelope_version": ENVELOPE_VERSION,
        "kind": kind,
        "endpoint": endpoint,
        "captured_at": captured_at.astimezone(dt.UTC).isoformat(),
        "source": "NESO Carbon Intensity API",
        "licence": "CC BY 4.0",
        "payload": payload,
    }
    path.write_text(json.dumps(envelope, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    # A path containing a colon on NTFS resolves to an alternate data stream:
    # the write succeeds and the file is invisible to a directory listing. The
    # kind is validated above, so this should be unreachable. The check is here
    # because the failure it guards against is silent.
    if not path.is_file():
        raise OSError(f"{path}: written but not present on disk")

    return path


def read_snapshot(path: Path) -> tuple[dict, str, str, dt.datetime]:
    """Read a snapshot, returning its payload, kind, endpoint and capture time."""
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if envelope.get("envelope_version") != ENVELOPE_VERSION:
        raise ValueError(f"{path}: unsupported envelope version")
    captured_at = dt.datetime.fromisoformat(envelope["captured_at"])
    return envelope["payload"], envelope["kind"], envelope["endpoint"], captured_at


def iter_snapshots(root: Path = DEFAULT_ROOT, kind: str | None = None) -> Iterator[Path]:
    """Yield snapshot paths in chronological order, optionally filtered by kind."""
    if kind is not None:
        _check_kind(kind)
    for path in sorted(root.rglob("*.json")):
        if kind is None or path.name.startswith(f"{kind}_"):
            yield path
