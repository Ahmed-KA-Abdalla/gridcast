from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from gridcast.report import build_report, write_report
from gridcast.scheduling import Load

UTC = dt.UTC


def intensity_payload(entries):
    return {
        "data": [
            {
                "from": start,
                "to": end,
                "intensity": {"forecast": forecast, "actual": actual, "index": "moderate"},
            }
            for start, end, forecast, actual in entries
        ]
    }


@pytest.fixture
def populated_store(tmp_path):
    """A year of outcomes with captured forecasts over the last month."""
    from gridcast.storage import write_snapshot

    rng = np.random.default_rng(3)
    start = dt.datetime(2025, 10, 1, tzinfo=UTC)

    outcomes = []
    for step in range(48 * 330):
        moment = start + dt.timedelta(minutes=30 * step)
        winter = moment.month in (12, 1, 2)
        value = (190.0 if winter else 115.0) + rng.normal(0, 18)
        outcomes.append(
            (
                moment.strftime("%Y-%m-%dT%H:%MZ"),
                (moment + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                value,
                value,
            )
        )
    write_snapshot(
        intensity_payload(outcomes),
        "intensity_range_20251001",
        "/intensity/range",
        start + dt.timedelta(days=331),
        root=tmp_path,
    )

    for day in range(25):
        issue = start + dt.timedelta(days=300 + day)
        entries = []
        for ahead in range(1, 49):
            target = issue + dt.timedelta(minutes=30 * ahead)
            entries.append(
                (
                    target.strftime("%Y-%m-%dT%H:%MZ"),
                    (target + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%MZ"),
                    115.0 + rng.normal(0, 22),
                    None,
                )
            )
        write_snapshot(
            intensity_payload(entries),
            "forecast_fw48h",
            "/intensity/issue/fw48h",
            issue,
            root=tmp_path,
        )

    return tmp_path


def test_the_page_is_self_contained(populated_store, record_path):
    # Published as a static site, so it must carry no external assets.
    page = build_report(populated_store, record=record_path, load=Load(periods=2, window_hours=6.0))

    assert page.startswith("<!doctype html>")
    assert "<link" not in page
    assert "<script" not in page
    assert "src=" not in page


def test_the_page_carries_the_counts_it_was_built_from(populated_store, record_path):
    # The findings move, so a page without its provenance goes stale silently.
    page = build_report(populated_store, record=record_path, load=Load(periods=2, window_hours=6.0))

    assert "Built " in page
    assert "Settled half-hours" in page
    assert "Outcomes span" in page


def test_the_page_reports_decision_quality_and_drift(populated_store, record_path):
    page = build_report(populated_store, record=record_path, load=Load(periods=2, window_hours=6.0))

    assert "Decision quality" in page
    assert "Seasonal drift" in page
    assert "seasonal_mean_full" in page


def test_a_section_with_nothing_to_show_says_so_rather_than_breaking(tmp_path, record_path):
    page = build_report(tmp_path, record=record_path)

    assert page.startswith("<!doctype html>")
    assert "Nothing to report yet." in page


def test_the_page_escapes_content_rather_than_interpolating_it(populated_store, record_path):
    page = build_report(populated_store, record=record_path, load=Load(periods=2, window_hours=6.0))

    # pandas escapes cell content; the surrounding prose is escaped explicitly.
    assert "<script>" not in page
    assert page.count("<html") == 1


def test_writing_the_page_creates_its_directory(populated_store, tmp_path_factory, record_path):
    destination = tmp_path_factory.mktemp("site") / "nested" / "index.html"
    written = write_report(
        destination,
        root=populated_store,
        record=record_path,
        load=Load(periods=2, window_hours=6.0),
    )

    assert written.exists()
    assert written.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_report_page_command_writes_the_file(populated_store, tmp_path_factory, capsys):
    from gridcast.cli import main

    destination = tmp_path_factory.mktemp("out") / "index.html"
    exit_code = main(
        [
            "--root",
            str(populated_store),
            "report-page",
            "--out",
            str(destination),
            "--periods",
            "2",
            "--window",
            "6",
        ]
    )
    assert exit_code == 0
    assert destination.exists()
    assert "wrote" in capsys.readouterr().out


# -- the README block ------------------------------------------------------


def test_splicing_replaces_only_the_marked_region():
    from gridcast.report import BLOCK_END, BLOCK_START, splice_block

    existing = f"before\n{BLOCK_START}\nold\n{BLOCK_END}\nafter\n"
    spliced = splice_block(existing, f"{BLOCK_START}\nnew\n{BLOCK_END}")

    assert spliced == f"before\n{BLOCK_START}\nnew\n{BLOCK_END}\nafter\n"
    assert "old" not in spliced


def test_splicing_refuses_a_document_without_markers():
    # A generator that rewrites a file it does not recognise is how a README
    # gets destroyed by a scheduled job.
    from gridcast.report import splice_block

    with pytest.raises(ValueError, match="not found"):
        splice_block("a README with no markers", "anything")


def test_splicing_refuses_markers_in_the_wrong_order():
    from gridcast.report import BLOCK_END, BLOCK_START, splice_block

    with pytest.raises(ValueError, match="out of order"):
        splice_block(f"{BLOCK_END}\n{BLOCK_START}", "anything")


def test_the_block_carries_the_date_and_counts_it_was_built_from(populated_store, record_path):
    from gridcast.report import readme_block

    block = readme_block(
        populated_store, record=record_path, load=Load(periods=2, window_hours=6.0)
    )

    assert "regenerated" in block
    assert "settled half-hours" in block
    assert "Decision quality" in block


def test_updating_a_readme_leaves_the_prose_untouched(
    populated_store, record_path, tmp_path_factory
):
    from gridcast.report import BLOCK_END, BLOCK_START, update_readme

    readme = tmp_path_factory.mktemp("doc") / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# title",
                "",
                "prose that explains the findings",
                "",
                BLOCK_START,
                BLOCK_END,
                "",
                "more prose",
                "",
            ]
        ),
        encoding="utf-8",
    )

    update_readme(
        readme, root=populated_store, record=record_path, load=Load(periods=2, window_hours=6.0)
    )
    updated = readme.read_text(encoding="utf-8")

    assert "prose that explains the findings" in updated
    assert "more prose" in updated
    assert "Decision quality" in updated


def test_updating_twice_is_idempotent_apart_from_the_date(
    populated_store, record_path, tmp_path_factory
):
    # The block is replaced wholesale, so repeated runs must not nest or
    # accumulate markers.
    from gridcast.report import BLOCK_END, BLOCK_START, update_readme

    readme = tmp_path_factory.mktemp("doc") / "README.md"
    readme.write_text(f"# title\n\n{BLOCK_START}\n{BLOCK_END}\n", encoding="utf-8")

    for _ in range(3):
        update_readme(
            readme,
            root=populated_store,
            record=record_path,
            load=Load(periods=2, window_hours=6.0),
        )

    updated = readme.read_text(encoding="utf-8")
    assert updated.count(BLOCK_START) == 1
    assert updated.count(BLOCK_END) == 1


def test_refresh_readme_command_reports_a_missing_marker(tmp_path_factory, capsys):
    from gridcast.cli import main

    readme = tmp_path_factory.mktemp("doc") / "README.md"
    readme.write_text("no markers here", encoding="utf-8")

    assert main(["refresh-readme", "--readme", str(readme)]) == 1
    assert "could not update" in capsys.readouterr().err
