from __future__ import annotations

import json

import numpy as np

from gridcast.correction import BandDamping
from gridcast.gate import (
    BandVerdict,
    Thresholds,
    _reasons,
    evaluate_gate,
    gate_passes,
    load_record,
    record_model,
    write_record,
)


def sym(value: float) -> BandDamping:
    return BandDamping.symmetric(value)


def row(
    damping: float = 0.45,
    improvement: float = 1.0,
    improvement_low: float = 0.5,
    improvement_high: float = 1.5,
    n: int = 500,
    periods: int = 150,
    up: float | None = None,
    down: float | None = None,
    folds: int = 5,
    folds_positive: int = 5,
    up_range: float = 0.05,
    down_range: float = 0.05,
):
    """A band that passes every check unless told otherwise.

    ``damping`` sets both directions at once; ``up`` and ``down`` override it.
    """
    import pandas as pd

    return pd.Series(
        {
            "band": "(6, 12]",
            "damping_up": damping if up is None else up,
            "damping_down": damping if down is None else down,
            "improvement": improvement,
            "improvement_low": improvement_low,
            "improvement_high": improvement_high,
            "n": n,
            "periods": periods,
            "folds": folds,
            "folds_positive": folds_positive,
            "up_range": up_range,
            "down_range": down_range,
        }
    )


def verdict(band: str, promoted: bool, reasons=None) -> BandVerdict:
    return BandVerdict(
        band=band,
        promoted=promoted,
        damping_up=0.45,
        damping_down=0.45,
        improvement=1.0,
        improvement_low=0.5,
        improvement_high=1.5,
        n=500,
        periods=150,
        reasons=reasons or [],
    )


def test_a_band_meeting_every_condition_has_no_reasons_against_it():
    assert _reasons(row(), previous=sym(0.44), thresholds=Thresholds()) == []


def test_an_interval_including_zero_blocks_promotion():
    reasons = _reasons(row(improvement_low=-0.1, improvement_high=0.8), None, Thresholds())
    assert any("includes zero" in reason for reason in reasons)


def test_an_interval_wholly_below_zero_is_reported_as_worse_not_as_unproven():
    # The distinction decides what the reader should do. An interval straddling
    # zero says wait for more data; one entirely below it says withdraw the
    # band. Both fail the gate, and for three weeks the gate called the second
    # of them the first, which read as a thin sample rather than a result.
    reasons = _reasons(
        row(improvement=-1.3, improvement_low=-1.9, improvement_high=-0.7), None, Thresholds()
    )
    assert any("significantly worse" in reason for reason in reasons)
    assert not any("includes zero" in reason for reason in reasons)


def test_a_missing_upper_bound_falls_back_to_the_includes_zero_reading():
    reasons = _reasons(row(improvement_low=-0.1, improvement_high=float("nan")), None, Thresholds())
    assert any("includes zero" in reason for reason in reasons)


def test_too_few_observations_blocks_promotion():
    reasons = _reasons(row(n=40), None, Thresholds())
    assert any("observations" in reason for reason in reasons)


def test_too_few_distinct_periods_blocks_promotion():
    # A band can carry many rows and few independent situations, since several
    # revisions of one period move together.
    reasons = _reasons(row(n=500, periods=5), None, Thresholds())
    assert any("distinct periods" in reason for reason in reasons)


def test_a_missing_interval_blocks_promotion():
    reasons = _reasons(row(improvement_low=np.nan), None, Thresholds())
    assert any("no interval" in reason for reason in reasons)


def test_a_coefficient_that_moves_too_far_blocks_promotion():
    # The condition an interval cannot supply: a coefficient swinging between
    # refits describes the sample, not the forecast.
    reasons = _reasons(row(damping=0.93), previous=sym(0.45), thresholds=Thresholds())
    assert any("moved" in reason for reason in reasons)


def test_a_coefficient_that_changes_sign_blocks_promotion():
    reasons = _reasons(row(damping=-0.10), previous=sym(0.10), thresholds=Thresholds())
    assert any("sign" in reason for reason in reasons)


def test_a_small_drift_is_tolerated():
    assert _reasons(row(damping=0.55), previous=sym(0.45), thresholds=Thresholds()) == []


def test_a_band_with_no_history_is_judged_on_its_own_merits():
    assert _reasons(row(damping=0.93), previous=None, thresholds=Thresholds()) == []


# -- the build decision ---------------------------------------------------


def test_losing_a_previously_promoted_band_fails_the_build():
    verdicts = [verdict("(6, 12]", promoted=False, reasons=["interval includes zero"])]
    passes, message = gate_passes(verdicts, previous={"(6, 12]": 0.46})

    assert not passes
    assert "no longer qualif" in message


def test_a_band_that_never_qualified_does_not_fail_the_build():
    # Most bands never clear the bar. Failing on that would mean the build is
    # red from the first run and stays red, which teaches everyone to ignore it.
    verdicts = [verdict("(0, 3]", promoted=False, reasons=["interval includes zero"])]
    passes, _ = gate_passes(verdicts, previous={})
    assert passes


def test_a_band_that_still_qualifies_passes():
    verdicts = [verdict("(6, 12]", promoted=True)]
    passes, message = gate_passes(verdicts, previous={"(6, 12]": 0.46})

    assert passes
    assert "(6, 12]" in message


def test_an_empty_evaluation_passes_rather_than_failing_on_no_data():
    passes, message = gate_passes([], previous={"(6, 12]": 0.46})
    assert passes
    assert "nothing to evaluate" in message


# -- the record -----------------------------------------------------------


def test_the_record_round_trips(tmp_path):
    path = tmp_path / "promoted.json"
    write_record([verdict("(6, 12]", True), verdict("(0, 3]", False)], path=path)

    stored = load_record(path)
    assert stored == {"(6, 12]": sym(0.45)}


def test_the_record_keeps_the_failing_bands_and_their_reasons(tmp_path):
    path = tmp_path / "promoted.json"
    write_record(
        [verdict("(0, 3]", False, reasons=["interval includes zero"])],
        path=path,
        generated="2026-08-29T12:00:00+00:00",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["coefficients"] == {}
    assert payload["verdicts"][0]["reasons"] == ["interval includes zero"]
    assert payload["generated"] == "2026-08-29T12:00:00+00:00"


def test_a_missing_record_reads_as_empty(tmp_path):
    assert load_record(tmp_path / "absent.json") == {}


def test_a_corrupt_record_reads_as_empty_rather_than_raising(tmp_path):
    # A malformed record should cost the drift check, not the whole run.
    path = tmp_path / "promoted.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_record(path) == {}


def test_gate_is_empty_on_a_store_with_nothing_to_fit(tmp_path):
    verdicts, _ = evaluate_gate(tmp_path, record=tmp_path / "promoted.json")
    assert verdicts == []


def test_gate_command_reports_each_band(overshooting_store, capsys, record_path):
    from gridcast.cli import main

    record = record_path
    assert main(["--root", str(overshooting_store), "gate", "--record", str(record)]) == 0

    out = capsys.readouterr().out
    assert "fitted on" in out
    assert "damping" in out


def test_gate_command_writes_the_record_only_when_asked(overshooting_store, record_path):
    from gridcast.cli import main

    record = record_path
    main(["--root", str(overshooting_store), "gate", "--record", str(record)])
    assert not record.exists()

    main(["--root", str(overshooting_store), "gate", "--record", str(record), "--update"])
    assert record.exists()


def test_gate_command_fails_when_a_promoted_band_regresses(overshooting_store, record_path):
    from gridcast.cli import main

    record = record_path
    # Claim every band was promoted last time, including ones that cannot
    # qualify, so the run must report a regression.
    record.write_text(
        json.dumps(
            {
                "model": "asymmetric",
                "coefficients": {"(0, 3]": 0.5, "(3, 6]": 0.5, "(6, 12]": 0.5},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(["--root", str(overshooting_store), "gate", "--record", str(record)])
    assert exit_code == 1


def test_a_failing_run_still_records_the_withdrawal(overshooting_store, record_path):
    # The withdrawal has to survive the failure. `--update` writes the record
    # and then exits non-zero, so a caller that only persists the record on a
    # zero exit discards the one thing that would let the next run recover: the
    # workflow did exactly that and the gate stayed red for three weeks,
    # re-deciding a verdict it could never commit. A gate needs a route back to
    # green that does not depend on someone noticing.
    from gridcast.cli import main

    record = record_path
    record.write_text(
        json.dumps(
            {
                "model": "asymmetric",
                "coefficients": {"(0, 3]": 0.5, "(3, 6]": 0.5, "(6, 12]": 0.5},
            }
        ),
        encoding="utf-8",
    )
    exit_code = main(
        ["--root", str(overshooting_store), "gate", "--record", str(record), "--update"]
    )
    assert exit_code == 1

    # The stale claim has been replaced by what this run actually found, so the
    # band it could not substantiate is no longer asserted anywhere.
    written = load_record(record)
    assert written != {band: sym(0.5) for band in ("(0, 3]", "(3, 6]", "(6, 12]")}
    assert "(6, 12]" not in written

    # And the second run, reading what the first wrote, has nothing to regress
    # from and passes. Without the write surviving the failure this run is
    # identical to the first and the gate never recovers.
    assert (
        main(["--root", str(overshooting_store), "gate", "--record", str(record), "--update"]) == 0
    )


def test_gate_command_says_so_with_nothing_to_evaluate(tmp_path, capsys):
    from gridcast.cli import main

    assert main(["--root", str(tmp_path), "gate", "--record", str(tmp_path / "r.json")]) == 0
    assert "not enough captured revisions" in capsys.readouterr().out


def test_a_promoted_band_that_vanishes_from_the_evaluation_fails_the_build():
    # Usually means the band no longer has enough held-out data to be scored,
    # which leaves the recorded coefficient standing on nothing. Easy to miss,
    # because no row reports it.
    verdicts = [verdict("(0, 3]", promoted=True)]
    passes, message = gate_passes(verdicts, previous={"(6, 12]": 0.46})

    assert not passes
    assert "no longer evaluated" in message


# -- coefficient history --------------------------------------------------


def test_the_history_accumulates_across_runs(record_path):
    from gridcast.gate import coefficient_history, write_record

    write_record([verdict("(6, 12]", True)], path=record_path, generated="2026-08-29T00:00:00Z")
    write_record([verdict("(6, 12]", True)], path=record_path, generated="2026-08-30T00:00:00Z")

    history = coefficient_history(record_path)
    # One row per direction, one column per run.
    assert history.shape == (2, 2)
    assert history.loc[("(6, 12]", "up")].tolist() == [0.45, 0.45]
    assert history.loc[("(6, 12]", "down")].tolist() == [0.45, 0.45]


def test_the_history_keeps_bands_that_were_not_promoted(record_path):
    # A failing band still yields a coefficient, and whether it is stable is
    # what later distinguishes too little data from no effect.
    from gridcast.gate import coefficient_history, write_record

    write_record(
        [verdict("(0, 3]", promoted=False, reasons=["interval includes zero"])],
        path=record_path,
        generated="2026-08-29T00:00:00Z",
    )
    history = coefficient_history(record_path)
    assert "(0, 3]" in history.index.get_level_values("band")


def test_the_history_is_capped(record_path):
    from gridcast.gate import HISTORY_LENGTH, coefficient_history, write_record

    for day in range(HISTORY_LENGTH + 6):
        write_record(
            [verdict("(6, 12]", True)],
            path=record_path,
            generated=f"2026-09-{day + 1:02d}T00:00:00Z",
        )

    assert coefficient_history(record_path).shape[1] == HISTORY_LENGTH


def test_the_history_is_empty_before_any_run(record_path):
    from gridcast.gate import coefficient_history

    assert coefficient_history(record_path).empty


def test_a_corrupt_record_yields_no_history_rather_than_raising(record_path):
    from gridcast.gate import coefficient_history

    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("{not json", encoding="utf-8")
    assert coefficient_history(record_path).empty


def test_writing_a_record_preserves_the_earlier_history(record_path):
    from gridcast.gate import load_record, write_record

    write_record([verdict("(6, 12]", True)], path=record_path, generated="2026-08-29T00:00:00Z")
    write_record(
        [verdict("(0, 3]", promoted=False)], path=record_path, generated="2026-08-30T00:00:00Z"
    )

    # The latest run promoted nothing, so the coefficients are empty, but the
    # history still carries both runs.
    assert load_record(record_path) == {}
    assert len(json.loads(record_path.read_text(encoding="utf-8"))["history"]) == 2


def test_a_zero_coefficient_is_never_promoted():
    # A zero coefficient leaves the forecast untouched, so the improvement is
    # identically zero and the interval collapses to a point at zero. Promoting
    # it would record a claim that the correction helps when it does nothing.
    from gridcast.gate import Thresholds, _reasons

    reasons = _reasons(row(damping=0.0, improvement=0.0, improvement_low=0.0), None, Thresholds())
    assert any("coefficient is zero" in reason for reason in reasons)


def test_a_zero_coefficient_is_rejected_even_with_a_clean_interval():
    from gridcast.gate import Thresholds, _reasons

    reasons = _reasons(
        row(damping=0.0, improvement=1e-18, improvement_low=1e-18), None, Thresholds()
    )
    assert any("coefficient is zero" in reason for reason in reasons)


def test_a_statistically_clear_but_negligible_improvement_is_rejected():
    # The case that exposed the threshold. A coefficient of 0.0008 applies
    # almost no correction, so the improvement is 0.0003 gCO2/kWh and its
    # variance is small in proportion; the interval clears zero while the
    # effect is nothing. Significance is not magnitude.
    from gridcast.gate import Thresholds, _reasons

    reasons = _reasons(
        row(damping=0.0008, improvement=0.00033, improvement_low=0.00008),
        None,
        Thresholds(),
    )
    assert any("too small to claim" in reason for reason in reasons)


def test_the_reason_distinguishes_a_negative_bound_from_a_negligible_one():
    from gridcast.gate import Thresholds, _reasons

    negative = _reasons(row(improvement_low=-0.5), None, Thresholds())
    negligible = _reasons(row(improvement_low=0.02), None, Thresholds())

    assert any("includes zero" in reason for reason in negative)
    assert any("too small to claim" in reason for reason in negligible)


# -- consistency across held-out blocks ------------------------------------


def test_an_improvement_carried_by_one_block_is_not_promoted():
    # What a single split could not see. The pooled interval here is clean, but
    # the correction helped in only three of five stretches of weeks; the
    # symmetric correction at short lead looked exactly like this before it
    # turned harmful.
    reasons = _reasons(row(folds=5, folds_positive=3), None, Thresholds())
    assert any("helped in only 3 of 5" in reason for reason in reasons)


def test_four_blocks_in_five_is_enough():
    assert _reasons(row(folds=5, folds_positive=4), None, Thresholds()) == []


def test_too_few_blocks_cannot_establish_consistency():
    reasons = _reasons(row(folds=2, folds_positive=2), None, Thresholds())
    assert any("held-out blocks, needs 3" in reason for reason in reasons)


def test_a_coefficient_that_swings_between_blocks_is_not_promoted():
    # The long-lead bands on the real record: a clean pooled interval, and an
    # upward coefficient ranging from -0.50 to 2.31 across the blocks' fits.
    reasons = _reasons(row(up_range=1.34, down_range=0.05), None, Thresholds())
    assert any("unstable across held-out blocks" in reason for reason in reasons)
    assert any("up ranged 1.34" in reason for reason in reasons)
    assert not any("down ranged" in reason for reason in reasons)


def test_each_direction_is_checked_for_drift_from_the_last_run():
    reasons = _reasons(
        row(up=0.10, down=0.80), previous=BandDamping(0.10, 0.40), thresholds=Thresholds()
    )
    assert any("down coefficient moved 0.40" in reason for reason in reasons)
    assert not any(reason.startswith("up coefficient") for reason in reasons)


def test_a_zero_coefficient_has_no_sign_to_change():
    # Zero means too little data in that direction last time, not a sign.
    reasons = _reasons(
        row(up=0.10, down=0.70), previous=BandDamping(0.0, 0.70), thresholds=Thresholds()
    )
    assert not any("changed sign" in reason for reason in reasons)


# -- the model ------------------------------------------------------------


def test_the_record_stores_the_model_and_both_directions(record_path):
    write_record(
        [
            BandVerdict(
                band="(6, 12]",
                promoted=True,
                damping_up=0.04,
                damping_down=0.86,
                improvement=1.6,
                improvement_low=1.3,
                improvement_high=1.9,
                n=2600,
                periods=2000,
            )
        ],
        path=record_path,
    )
    payload = json.loads(record_path.read_text(encoding="utf-8"))

    assert payload["model"] == "asymmetric"
    assert payload["coefficients"] == {"(6, 12]": {"up": 0.04, "down": 0.86}}
    assert payload["history"][-1]["model"] == "asymmetric"
    assert load_record(record_path) == {"(6, 12]": BandDamping(0.04, 0.86)}


def test_a_record_from_before_the_asymmetric_model_reads_as_symmetric(record_path):
    # docs/promoted.json as it stood on 27 October: one number per band, no model.
    record_path.write_text(
        json.dumps({"coefficients": {"(12, 24]": 0.332}, "history": []}), encoding="utf-8"
    )
    assert record_model(record_path) == "symmetric"
    assert load_record(record_path) == {"(12, 24]": BandDamping(0.332, 0.332)}


def test_there_is_no_model_without_a_record(record_path):
    assert record_model(record_path) is None


def test_a_change_of_model_is_not_a_regression():
    # The previous record's promotions were claims about another correction.
    # Replacing it is a commit, not a claim ceasing to hold, but the bands it
    # drops must still be named so the replacement is visible.
    verdicts = [
        verdict("(6, 12]", promoted=True),
        verdict("(12, 24]", promoted=False, reasons=["coefficient unstable"]),
    ]
    passes, message = gate_passes(
        verdicts, previous={"(12, 24]": sym(0.33)}, previous_model="symmetric"
    )

    assert passes
    assert "symmetric model" in message
    assert "not carried over: (12, 24]" in message
    assert "(6, 12]" in message


def test_the_same_model_still_fails_on_regression():
    verdicts = [verdict("(6, 12]", promoted=False, reasons=["interval includes zero"])]
    passes, _ = gate_passes(verdicts, previous={"(6, 12]": sym(0.46)}, previous_model="asymmetric")
    assert not passes


def test_coefficients_from_another_model_are_not_drifted_from(overshooting_store, record_path):
    # A symmetric 0.95 against an asymmetric fit near 0.5 would read as a move
    # of 0.45 in each direction, which is a change of model and not drift.
    record_path.write_text(
        json.dumps({"coefficients": {"(0, 3]": 0.95, "(3, 6]": 0.95}}), encoding="utf-8"
    )
    verdicts, note = evaluate_gate(overshooting_store, record=record_path)

    assert note["previous_model"] == "symmetric"
    assert verdicts
    for item in verdicts:
        assert item.previous is None
        assert not any("moved" in reason for reason in item.reasons)


def test_the_history_can_be_restricted_to_one_model(record_path):
    from gridcast.gate import coefficient_history

    write_record(
        [verdict("(6, 12]", True)],
        path=record_path,
        generated="2026-10-20T00:00:00Z",
        model="symmetric",
    )
    write_record([verdict("(6, 12]", True)], path=record_path, generated="2026-10-27T00:00:00Z")

    assert coefficient_history(record_path).shape[1] == 2
    assert coefficient_history(record_path, model="asymmetric").shape[1] == 1
    assert list(coefficient_history(record_path, model="symmetric").columns) == [
        "2026-10-20T00:00:00Z"
    ]


def test_the_gate_on_an_overshooting_store_promotes_and_records_the_model(
    overshooting_store, record_path, capsys
):
    from gridcast.cli import main

    assert (
        main(["--root", str(overshooting_store), "gate", "--record", str(record_path), "--update"])
        == 0
    )
    out = capsys.readouterr().out
    assert "asymmetric model" in out
    assert "helped in" in out
    assert record_model(record_path) == "asymmetric"
    assert load_record(record_path)
