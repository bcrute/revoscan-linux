"""The depth-completeness harness, proved load-bearing rather than merely runnable.

``tests/test_depth_completeness.py`` owns the contract and the measurements.
This file owns the machinery around them: that the shipped corpus produces
exactly the outcomes it declares, that corrupting a fixture, a scene, an
expectation or a manifest makes the harness FAIL rather than quietly measure
something else, and that the artifact refuses to be a report a reader could
misread.

The controls are the point. A harness whose only evidence was "it ran and said
pass" would keep saying pass through any regression in itself, which is the
same failure mode the corpus exists to catch in a depth completion.

Nothing here touches a scanner, a capture under ``/home/ben/scans``, a display
server, a GPU, the network or the clock.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import pytest

from scanner3d.errors import ScannerError
from scanner3d.v4l2_capture import FRAME_WIDTH
from tests import build_depth_completeness_fixtures as builder
from tests import depth_completeness as contract
from tests import depth_completeness_harness as harness
from tests.quality_harness import FixtureUnusable, HarnessRefused, ReportUnusable, to_json

CASES = harness.CORPUS

#: The cheapest case to corrupt: one frame, no masks, no fusion, no tracking.
CHEAP = "plane-fronto-complete"


@pytest.fixture(scope="module")
def shipped() -> Any:
    """One run of the shipped corpus, shared by every test that reads it."""
    return harness.run()


def _corpus_of(tmp_path: Path, *names: str) -> Path:
    """A writable corpus holding only the named cases, so a control can corrupt it."""
    root = tmp_path / "cases"
    root.mkdir()
    manifest = json.loads((CASES / harness.MANIFEST_NAME).read_text())
    manifest["files"] = [entry for entry in manifest["files"] if entry["case_id"] in names]
    assert len(manifest["files"]) == len(names)
    for entry in manifest["files"]:
        shutil.copy(CASES / entry["name"], root / entry["name"])
    (root / harness.MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return root


def _rewrite(root: Path, case_id: str, edit) -> None:
    """Edit one case in a copied corpus and keep its manifest entry honest."""
    manifest_path = root / harness.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    entry = next(item for item in manifest["files"] if item["case_id"] == case_id)
    path = root / entry["name"]
    document = json.loads(path.read_text())
    edit(document)
    text = builder.serialise(document)
    path.write_text(text)
    entry["bytes"] = len(text.encode("utf-8"))
    entry["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _case(document: Mapping[str, object], case_id: str) -> Any:
    cases = document["cases"]
    assert isinstance(cases, list)
    for entry in cases:
        assert isinstance(entry, Mapping)
        if entry["case_id"] == case_id:
            return entry
    raise AssertionError(f"{case_id} is not in the report")


def _candidate(case: Mapping[str, object], name: str) -> Any:
    candidates = case["candidates"]
    assert isinstance(candidates, list)
    for entry in candidates:
        assert isinstance(entry, Mapping)
        if entry["candidate"] == name:
            return entry
    raise AssertionError(f"{name} did not run in {case['case_id']}")


# --- the shipped corpus -----------------------------------------------------------------


def test_the_shipped_corpus_produces_exactly_what_it_declares(shipped):
    assert shipped["decision"] == "pass"
    totals = shipped["totals"]
    assert totals["cases"] == totals["passed"] == len(list(CASES.glob("*.json"))) - 1
    assert totals["failed"] == totals["malformed"] == 0
    assert totals["candidates"] == totals["candidates_passed"]
    assert totals["candidates"] > 40


def test_every_declared_geometry_is_in_the_corpus(shipped):
    cases = {str(entry["case_id"]) for entry in shipped["cases"]}
    assert cases == {
        "plane-fronto-complete",
        "plane-fronto",
        "plane-oblique",
        "curved-surface",
        "depth-step",
        "narrow-slot",
        "aperture-ambiguity",
        "multiview-object",
        "temporal-motion",
        "temporal-confirmation",
    }


def test_the_baseline_invents_nothing_and_is_never_scored_as_a_success(shipped):
    for case_id in ("plane-fronto", "plane-oblique", "curved-surface", "narrow-slot"):
        baseline = _candidate(_case(shipped, case_id), "baseline")
        assert baseline["observed"]["outcome"] == "fail"
        assert "coverage.supported_fraction" in baseline["observed"]["failed_gates"]
        coverage = next(
            measurement
            for measurement in baseline["quality"]["measurements"]
            if measurement["kind"] == "depth_completeness_coverage"
        )
        assert coverage["reconstructed"] == 0
        assert coverage["false_fill"] == 0
        assert coverage["missing"] > 0


def test_a_dense_fill_is_admitted_where_it_is_right_and_refused_everywhere_else(shipped):
    assert _candidate(_case(shipped, "plane-fronto"), "dense_nearest")["observed"]["outcome"] == (
        "pass"
    )
    for case_id in ("plane-oblique", "curved-surface", "depth-step", "narrow-slot"):
        dense = _candidate(_case(shipped, case_id), "dense_nearest")
        assert dense["observed"]["outcome"] == "fail", case_id


def test_the_exact_pose_control_is_accurate_and_still_short_of_the_target(shipped):
    posed = _candidate(_case(shipped, "multiview-object"), "posed_temporal_carry")
    naive = _candidate(_case(shipped, "multiview-object"), "naive_temporal_carry")
    assert posed["observed"]["failed_gates"] == ["coverage.supported_fraction"]
    assert "reconstructed_depth.p95" in naive["observed"]["failed_gates"]


#: The refusals the two completion-specific controls must earn, beside the nine
#: hand-written corruptions. They are separate because they corrupt a REAL
#: result rather than a hand-written fill: one restates what the run cost, and
#: one cites a reconstruction as evidence for another reconstruction.
COMPLETION_REFUSALS = {"work_counts_inconsistent", "support_not_measured"}

#: The same for the pose-compensated stage's four controls, which also corrupt
#: a real product result: one restates what the run refused, two cite evidence
#: that is not another frame's original measurement, and one calls an estimate
#: nothing independent ever reached confirmed.
#: What a candidate raises when the scene it is offered gives its particular
#: control nothing to corrupt - no other frame, no refused hole, no spatial
#: estimate. Named once because it is caught in exactly one place.
UNAVAILABLE_HERE = (contract.ScoringUnusable, ScannerError)

TEMPORAL_REFUSALS = {
    "work_counts_inconsistent",
    "temporal_support_not_another_frame",
    "support_not_measured",
    "confirmation_inconsistent",
}


def test_every_corruption_control_is_refused_and_never_scored(shipped):
    refused = [
        candidate
        for case in shipped["cases"]
        for candidate in case["candidates"]
        if candidate["observed"]["outcome"] == "refused"
    ]
    assert len(refused) == 44
    seen = {str(candidate["observed"]["refusal"]) for candidate in refused}
    assert (
        seen
        == {name for _candidate_name, _what, name in builder.CORRUPTIONS}
        | COMPLETION_REFUSALS
        | TEMPORAL_REFUSALS
    )
    assert {
        name for _candidate_name, _what, name in builder.TEMPORAL_CORRUPTIONS
    } <= TEMPORAL_REFUSALS
    for candidate in refused:
        assert candidate["quality"] is None
        assert candidate["gate_bindings"] == []
        assert candidate["status"] == "pass"


def test_two_runs_are_byte_identical(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    assert harness.main(["--quiet", "--out", str(first)]) == 0
    assert harness.main(["--quiet", "--out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_a_published_report_reads_back_through_the_strict_reader(tmp_path):
    out = tmp_path / "report.json"
    assert harness.main(["--quiet", "--out", str(out)]) == 0
    document: Any = harness.read_report(out)
    assert document["decision"] == "pass"
    assert list(document["provenance_states"]) == list(contract.STATES)
    assert list(document["refusals"]) == list(contract.REFUSALS)


def test_the_rendered_log_names_every_case_and_candidate(shipped):
    text = harness.render(shipped)
    for case in shipped["cases"]:
        assert str(case["case_id"]) in text
        for candidate in case["candidates"]:
            assert str(candidate["candidate"]) in text
    assert "limitation:" in text


# --- corrupting the corpus ---------------------------------------------------------------


def test_a_fixture_edited_behind_its_manifest_is_malformed(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)
    path = root / f"{CHEAP}.json"
    path.write_text(path.read_text().replace("plane", "plane "))
    document = harness.run(root)
    assert document["decision"] == "fail"
    entry = _case(document, CHEAP)
    assert entry["status"] == "malformed"
    assert "sha256" in str(entry["error"])
    assert entry["candidates"] == []


def test_a_scene_that_no_longer_renders_its_own_frames_is_malformed(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)

    def move_the_plane(document):
        document["scene"]["shapes"][0]["offset"] = 101.0

    _rewrite(root, CHEAP, move_the_plane)
    entry = _case(harness.run(root), CHEAP)
    assert entry["status"] == "malformed"
    assert "renders" in str(entry["error"])


def test_a_mask_that_no_longer_produces_its_own_input_frame_is_malformed(tmp_path):
    root = _corpus_of(tmp_path, "plane-fronto")

    def widen_the_strip(document):
        for mask in document["scene"]["frames"][0]["masks"]:
            if mask["kind"] == "strip":
                mask["last_column"] = 34

    _rewrite(root, "plane-fronto", widen_the_strip)
    entry = _case(harness.run(root), "plane-fronto")
    assert entry["status"] == "malformed"
    assert "input_sha256" in str(entry["error"])


def test_an_expectation_that_does_not_happen_fails_the_case(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)

    def declare_the_discard_acceptable(document):
        for candidate in document["candidates"]:
            if candidate["name"] == "discarded_measurements":
                candidate["expectation"]["outcome"] = "pass"
                candidate["expectation"]["failed_gates"] = []

    _rewrite(root, CHEAP, declare_the_discard_acceptable)
    document = harness.run(root)
    assert document["decision"] == "fail"
    candidate = _candidate(_case(document, CHEAP), "discarded_measurements")
    assert candidate["status"] == "fail"
    assert "expected the case to pass and it did fail" in candidate["mismatches"]


def test_something_failed_is_not_the_expected_failure(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)

    def name_a_different_gate(document):
        for candidate in document["candidates"]:
            if candidate["name"] == "discarded_measurements":
                candidate["expectation"]["failed_gates"] = ["plane.residual.p95"]

    _rewrite(root, CHEAP, name_a_different_gate)
    candidate = _candidate(_case(harness.run(root), CHEAP), "discarded_measurements")
    assert candidate["status"] == "fail"
    assert any("was expected to fail and passed" in line for line in candidate["mismatches"])
    assert any("failed and was not expected to" in line for line in candidate["mismatches"])


def test_a_corruption_control_that_stopped_being_refused_fails_the_case(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)

    def expect_a_different_refusal(document):
        for candidate in document["candidates"]:
            if candidate["name"] == "mutated_raw_frame":
                candidate["expectation"]["refusal"] = "unknown_state"

    _rewrite(root, CHEAP, expect_a_different_refusal)
    candidate = _candidate(_case(harness.run(root), CHEAP), "mutated_raw_frame")
    assert candidate["status"] == "fail"
    assert any("expected refusal" in line for line in candidate["mismatches"])


def test_a_gate_that_did_not_apply_must_be_declared(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)

    def forget_the_inapplicable_gates(document):
        for candidate in document["candidates"]:
            if candidate["name"] == "baseline":
                candidate["expectation"]["inapplicable_gates"] = []

    _rewrite(root, CHEAP, forget_the_inapplicable_gates)
    candidate = _candidate(_case(harness.run(root), CHEAP), "baseline")
    assert candidate["status"] == "fail"
    assert any("did not apply" in line for line in candidate["mismatches"])


def test_a_gate_whose_path_does_not_resolve_is_malformed_and_not_a_crash(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)

    def walk_somewhere_that_is_not_there(document):
        document["gates"][0]["path"] = ["supported_fraction", "somewhere_else"]

    _rewrite(root, CHEAP, walk_somewhere_that_is_not_there)
    entry = _case(harness.run(root), CHEAP)
    assert entry["status"] == "malformed"
    assert "somewhere_else" in str(entry["error"])


def test_a_case_naming_a_candidate_nobody_wrote_is_malformed(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)

    def invent_a_candidate(document):
        document["candidates"][0]["name"] = "clever_new_filler"

    _rewrite(root, CHEAP, invent_a_candidate)
    assert _case(harness.run(root), CHEAP)["status"] == "malformed"


def test_a_case_may_not_expect_a_gate_it_does_not_state(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)

    def name_a_gate_that_does_not_exist(document):
        document["candidates"][0]["expectation"]["outcome"] = "fail"
        document["candidates"][0]["expectation"]["failed_gates"] = ["no.such.gate"]

    _rewrite(root, CHEAP, name_a_gate_that_does_not_exist)
    entry = _case(harness.run(root), CHEAP)
    assert entry["status"] == "malformed"
    assert "no such gate" in str(entry["error"])


def test_independent_cases_all_run_even_when_one_is_broken(tmp_path):
    root = _corpus_of(tmp_path, CHEAP, "plane-fronto")
    path = root / f"{CHEAP}.json"
    path.write_text("{not json at all")
    document = harness.run(root)
    assert _case(document, CHEAP)["status"] == "malformed"
    assert _case(document, "plane-fronto")["status"] == "pass"


# --- corrupting the corpus's shape ---------------------------------------------------------


def test_a_manifest_naming_a_file_that_is_not_there_is_refused(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)
    (root / f"{CHEAP}.json").unlink()
    with pytest.raises(HarnessRefused):
        harness.run(root)


def test_a_case_file_the_manifest_does_not_name_is_refused(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)
    (root / "extra.json").write_text("{}")
    with pytest.raises(HarnessRefused, match="nobody listed"):
        harness.run(root)


def test_a_symbolic_link_is_never_read_as_a_fixture(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)
    real = root / f"{CHEAP}.json"
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(real.read_text())
    real.unlink()
    real.symlink_to(elsewhere)
    with pytest.raises(HarnessRefused, match="symbolic link"):
        harness.run(root)


def test_a_manifest_of_another_schema_version_is_refused(tmp_path):
    root = _corpus_of(tmp_path, CHEAP)
    manifest_path = root / harness.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = harness.FIXTURE_SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(HarnessRefused, match="schema version"):
        harness.run(root)


def test_two_cases_with_one_identity_are_refused(tmp_path):
    root = _corpus_of(tmp_path, CHEAP, "plane-fronto")
    manifest_path = root / harness.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][1]["case_id"] = manifest["files"][0]["case_id"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(HarnessRefused, match="more than one file"):
        harness.run(root)


# --- the command ---------------------------------------------------------------------------


def test_the_command_exits_two_when_it_cannot_run_and_leaves_no_report(tmp_path):
    out = tmp_path / "report.json"
    out.write_text("an earlier run's answer")
    assert harness.main(["--quiet", "--corpus", str(tmp_path / "absent"), "--out", str(out)]) == 2
    assert not out.exists()


def test_the_command_exits_one_when_a_case_does_not_match_its_fixture(tmp_path, capsys):
    root = _corpus_of(tmp_path, CHEAP)

    def declare_the_discard_acceptable(document):
        for candidate in document["candidates"]:
            if candidate["name"] == "discarded_measurements":
                candidate["expectation"]["outcome"] = "pass"
                candidate["expectation"]["failed_gates"] = []

    _rewrite(root, CHEAP, declare_the_discard_acceptable)
    out = tmp_path / "report.json"
    assert harness.main(["--corpus", str(root), "--out", str(out)]) == 1
    # The failing run is the one whose report matters, so it is still published.
    assert harness.read_report(out)["decision"] == "fail"
    assert "[FAIL]" in capsys.readouterr().out


def test_the_report_path_is_cleared_before_the_run(tmp_path):
    out = tmp_path / "report.json"
    out.write_text("an earlier run's answer")
    directory = tmp_path / "a-directory.json"
    directory.mkdir()
    assert harness.main(["--quiet", "--out", str(directory)]) == 2
    assert harness.main(["--quiet", "--out", str(out)]) == 0
    assert harness.read_report(out)["decision"] == "pass"


# --- reading a forged artifact back ----------------------------------------------------------


@pytest.fixture(scope="module")
def published(shipped: Any) -> str:
    return to_json(shipped)


def _forge(text: str, edit) -> str:
    document: Any = json.loads(text)
    edit(document)
    return json.dumps(document)


def _refused(text: str, edit) -> str:
    with pytest.raises(ReportUnusable) as raised:
        harness.loads_report(_forge(text, edit))
    return str(raised.value)


def test_a_decision_that_is_not_its_own_cases_is_refused(published):
    def fail_one_case(document):
        document["cases"][0]["status"] = "fail"

    assert "beside candidates that are" in _refused(published, fail_one_case)


def test_a_total_that_is_not_its_own_cases_is_refused(published):
    def inflate(document):
        document["totals"]["candidates_passed"] += 1

    assert "candidates_passed" in _refused(published, inflate)


def test_a_gate_value_changed_while_its_measurement_was_left_alone_is_refused(published):
    def move_a_gate(document):
        # A gate that already passed, moved to a different passing value, so the
        # report stays internally consistent about WHICH gates failed and only
        # the binding to its measurement can notice.
        for case in document["cases"]:
            for candidate in case["candidates"]:
                if candidate["quality"] is None:
                    continue
                for gate in candidate["quality"]["gates"]:
                    if gate["passed"] and gate["comparison"] == "<=":
                        gate["measured"] = float(gate["limit"]) / 2.0 - 1.0
                        return

    assert "measurement says" in _refused(published, move_a_gate)


def test_a_measurement_changed_while_its_gate_was_left_alone_is_refused(published):
    def move_a_measurement(document):
        for case in document["cases"]:
            for candidate in case["candidates"]:
                if candidate["quality"] is None:
                    continue
                for measurement in candidate["quality"]["measurements"]:
                    if measurement["kind"] == "depth_completeness_coverage":
                        measurement["supported_fraction"] = 1.0
                        return

    assert "supported_fraction" in _refused(published, move_a_measurement)


def test_a_coverage_block_that_does_not_account_for_its_window_is_refused(published):
    def lose_a_sample(document):
        for case in document["cases"]:
            for candidate in case["candidates"]:
                if candidate["quality"] is None:
                    continue
                for measurement in candidate["quality"]["measurements"]:
                    if measurement["kind"] == "depth_completeness_coverage":
                        measurement["missing"] += 1
                        return

    assert "accounts for" in _refused(published, lose_a_sample)


def test_a_pose_sequence_missing_a_frame_is_refused(published):
    def drop_the_worst_frame(document):
        for case in document["cases"]:
            for candidate in case["candidates"]:
                if candidate["quality"] is None:
                    continue
                for measurement in candidate["quality"]["measurements"]:
                    if measurement["kind"] == "pose_sequence_error":
                        measurement["frames"] = measurement["frames"][:-1]
                        return

    assert "gauge frame is excluded" in _refused(published, drop_the_worst_frame)


def test_a_measurement_kind_nobody_has_a_schema_for_is_refused(published):
    def rename_a_kind(document):
        for case in document["cases"]:
            for candidate in case["candidates"]:
                if candidate["quality"] is None:
                    continue
                candidate["quality"]["measurements"][0]["kind"] = "depth_completeness_vibes"
                return

    assert "no schema for" in _refused(published, rename_a_kind)


def test_a_refused_candidate_that_carries_measurements_is_refused(published):
    def score_a_refusal(document):
        for case in document["cases"]:
            for candidate in case["candidates"]:
                if candidate["observed"]["refusal"] is not None:
                    scored = next(
                        other for other in case["candidates"] if other["quality"] is not None
                    )
                    candidate["quality"] = scored["quality"]
                    return

    assert "never scored" in _refused(published, score_a_refusal)


def test_erased_mismatches_are_refused(published):
    def declare_a_pass(document):
        case = document["cases"][0]
        candidate = case["candidates"][0]
        candidate["expected"]["outcome"] = "fail"
        candidate["expected"]["failed_gates"] = ["coverage.measured_fraction"]

    assert "mismatches" in _refused(published, declare_a_pass)


def test_a_dropped_limitation_is_refused(published):
    def drop_one(document):
        document["limitations"] = document["limitations"][:-1]

    assert "limitations" in _refused(published, drop_one)


def test_a_report_of_another_schema_version_is_refused(published):
    def bump(document):
        document["schema_version"] = harness.REPORT_SCHEMA_VERSION + 1

    assert "schema version" in _refused(published, bump)


def test_duplicate_keys_and_non_json_numbers_are_refused(published):
    with pytest.raises(ReportUnusable, match=r"twice|duplicate"):
        harness.loads_report(published.replace('"decision"', '"decision": "fail", "decision"', 1))
    with pytest.raises(ReportUnusable):
        harness.loads_report(published.replace('"decision": "pass"', '"decision": NaN', 1))


# --- the scene, the declarations and the counts a report states twice -------------------------
#
# The mutations below are the ones a reader that only checked the shape of the
# top level would accept: the report keeps every count it needs to refute them,
# and each one is a way of publishing evidence that was never produced.


#: The one case the general-purpose corruption helpers below step over.
#:
#: `aperture-ambiguity` is a deliberately minimal pair: one held-out dropout and
#: one true aperture, two holes in the whole window, both of them accepted - so
#: there is no rejection to decrement - no dimensions region, and truth that is
#: empty at exactly one sample. It exists to state one epistemic boundary, not
#: to be a donor for edits that need a case with a refusal, a dimensions
#: measurement or a spare reconstruction in it. Sorting first alphabetically is
#: the only reason these helpers would ever reach it, which is not a reason.
#: The case carries its own controls in the corpus and in
#: tests/test_depth_completeness.py.
_MINIMAL_PAIR_CASE: Final = "aperture-ambiguity"


def _general_cases(document: Any) -> Any:
    """Every case a general-purpose corruption control may be built from."""
    return [case for case in document["cases"] if case["case_id"] != _MINIMAL_PAIR_CASE]


def _scored(document: Any):
    """The first candidate in the report that was actually scored."""
    for case in _general_cases(document):
        for candidate in case["candidates"]:
            if candidate["quality"] is not None:
                return case, candidate
    raise AssertionError("no candidate in the report was scored")


def _measurement(candidate: Any, name: str) -> Any:
    for measurement in candidate["quality"]["measurements"]:
        if measurement["name"] == name:
            return measurement
    raise AssertionError(f"{name} is not among this candidate's measurements")


def test_a_gate_limit_that_is_not_a_limit_is_refused(published):
    def unreadable_limit(document):
        document["cases"][0]["gates"][0]["limit"] = "not-a-limit"

    assert "not a number" in _refused(published, unreadable_limit)


def test_a_gate_declared_with_the_wrong_kind_of_limit_is_refused(published):
    def boolean_under_an_ordering(document):
        for gate in document["cases"][0]["gates"]:
            if gate["comparison"] in ("<=", ">="):
                gate["limit"] = True
                return
        raise AssertionError("no ordered gate to corrupt")

    assert "not a number" in _refused(published, boolean_under_an_ordering)


def test_a_boolean_gate_declared_in_another_unit_is_refused(published):
    def relabel_the_unit(document):
        for case in document["cases"]:
            for gate in case["gates"]:
                if gate["comparison"] == "is":
                    gate["unit"] = "millimetres"
                    return
        raise AssertionError("no boolean gate to corrupt")

    assert "a boolean is in" in _refused(published, relabel_the_unit)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("comparison", ">="),
        ("unit", "furlongs"),
        ("context", "a bar chosen after the fact"),
    ],
)
def test_a_declaration_edited_after_the_run_is_refused(published, key, value):
    """The declared gate and the evaluated gate are both in the report.

    A limit loosened, a comparison flipped or a unit rewritten AFTER a
    candidate was scored leaves the evaluated gate intact and only the
    declaration changed, which is exactly the edit a reader that trusted the
    declaration would not see.
    """

    def edit_the_declaration(document):
        case, candidate = _scored(document)
        evaluated = {gate["name"] for gate in candidate["quality"]["gates"]}
        for gate in case["gates"]:
            if gate["name"] in evaluated and gate[key] != value:
                gate[key] = value
                return
        raise AssertionError(f"no evaluated gate to edit {key} on")

    assert "declares" in _refused(published, edit_the_declaration)


def test_a_declared_limit_loosened_after_the_run_is_refused(published):
    def loosen(document):
        case, candidate = _scored(document)
        evaluated = {gate["name"] for gate in candidate["quality"]["gates"]}
        for gate in case["gates"]:
            if gate["name"] in evaluated and not isinstance(gate["limit"], bool):
                gate["limit"] = float(gate["limit"]) + 1.0
                return
        raise AssertionError("no ordered evaluated gate to loosen")

    assert "against limit" in _refused(published, loosen)


def test_a_frame_pose_that_is_not_a_pose_is_refused(published):
    def unreadable_pose(document):
        document["cases"][0]["scene"]["frames"][0]["pose"] = "not-a-pose"

    assert "not a JSON object" in _refused(published, unreadable_pose)


@pytest.mark.parametrize("key", ["rotation", "translation", "schema_version"])
def test_a_pose_missing_part_of_itself_is_refused(published, key):
    def drop(document):
        del document["cases"][0]["scene"]["frames"][0]["pose"][key]

    assert "'pose'" in _refused(published, drop)


def test_a_pose_carrying_a_key_nobody_wrote_is_refused(published):
    def add(document):
        document["cases"][0]["scene"]["frames"][0]["pose"]["scale"] = 2.0

    assert "'pose'" in _refused(published, add)


def test_a_pose_of_another_geometry_schema_version_is_refused(published):
    def bump(document):
        document["cases"][0]["scene"]["frames"][0]["pose"]["schema_version"] = 99

    assert "geometry schema version" in _refused(published, bump)


def test_a_pose_that_is_not_a_rigid_transform_is_refused(published):
    def reflect(document):
        pose = document["cases"][0]["scene"]["frames"][0]["pose"]
        pose["rotation"][0] = [-value for value in pose["rotation"][0]]

    assert "not state a rigid transform" in _refused(published, reflect)


def test_a_pose_whose_rotation_is_not_three_rows_is_refused(published):
    def truncate(document):
        pose = document["cases"][0]["scene"]["frames"][0]["pose"]
        pose["rotation"] = pose["rotation"][:2]

    assert "not three rows" in _refused(published, truncate)


def test_a_pose_row_that_is_not_numbers_is_refused(published):
    def unreadable_row(document):
        pose = document["cases"][0]["scene"]["frames"][0]["pose"]
        pose["rotation"][1][2] = True

    assert "not a number" in _refused(published, unreadable_row)


@pytest.mark.parametrize("key", ["rotation_angle_radians", "translation_magnitude"])
def test_a_pose_magnitude_that_is_not_its_own_matrix_is_refused(published, key):
    def forge_a_magnitude(document):
        for case in document["cases"]:
            for frame in case["scene"]["frames"]:
                frame["pose"][key] = float(frame["pose"][key]) + 1.0
            return

    assert key in _refused(published, forge_a_magnitude)


@pytest.mark.parametrize("value", ["not-a-number", True, None])
def test_an_intrinsic_that_is_not_a_number_is_refused(published, value):
    def unreadable_intrinsic(document):
        document["cases"][0]["scene"]["intrinsics"]["fx"] = value

    assert "not a number" in _refused(published, unreadable_intrinsic)


def test_a_focal_length_that_is_not_positive_is_refused(published):
    def flatten(document):
        document["cases"][0]["scene"]["intrinsics"]["fy"] = 0.0

    assert "deprojects nothing" in _refused(published, flatten)


def test_a_non_finite_scene_number_is_refused(shipped):
    """Not through the text: ``Infinity`` never survives being serialised.

    A mapping handed straight to the validator is the third way in, and it is
    how the harness checks its own document before publishing it.
    """
    document = json.loads(to_json(shipped))
    document["cases"][0]["scene"]["depth_scale"] = float("inf")
    with pytest.raises(ReportUnusable, match="not finite"):
        harness.validate_report(document)


@pytest.mark.parametrize("flag", ["plane", "dimensions", "landmarks", "tracking"])
def test_a_region_flag_that_is_not_a_boolean_is_refused(published, flag):
    def unreadable_flag(document):
        document["cases"][0]["scene"]["regions"][flag] = "not-a-boolean"

    assert "not a boolean" in _refused(published, unreadable_flag)


def test_a_region_flag_flipped_leaves_its_measurement_unaccounted_for(published):
    """The regions decide what is measured, so flipping one has to show.

    Turning tracking on in a case that never replayed the tracker would
    otherwise read as a case that measured tracking and passed.
    """

    def claim_tracking(document):
        for case in document["cases"]:
            if not case["scene"]["regions"]["tracking"]:
                case["scene"]["regions"]["tracking"] = True
                return
        raise AssertionError("every case already tracks")

    assert "'tracking'" in _refused(published, claim_tracking)


def test_a_measurement_the_scene_does_not_name_is_refused(published):
    def add_a_measurement(document):
        _, candidate = _scored(document)
        coverage = _measurement(candidate, "coverage")
        extra = dict(coverage)
        extra["name"] = "coverage_but_flattering"
        candidate["quality"]["measurements"].append(extra)

    assert "do not name" in _refused(published, add_a_measurement)


def test_a_required_measurement_dropped_is_refused(published):
    def drop_a_measurement(document):
        _, candidate = _scored(document)
        measurements = candidate["quality"]["measurements"]
        for index, measurement in enumerate(measurements):
            if measurement["name"] == "dimensions":
                del measurements[index]
                return
        raise AssertionError("no dimensions measurement to drop")

    assert "makes none of the measurement(s)" in _refused(published, drop_a_measurement)


@pytest.mark.parametrize("columns", [[True], [-1], [10_000], [3, 3]])
def test_slot_columns_that_are_not_columns_are_refused(published, columns):
    def rewrite_the_slot(document):
        for case in document["cases"]:
            if case["scene"]["regions"]["slot_columns"]:
                case["scene"]["regions"]["slot_columns"] = columns
                return
        raise AssertionError("no case names a slot")

    assert "column" in _refused(published, rewrite_the_slot)


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda volume: volume.update(maximum=volume["minimum"]), "encloses nothing"),
        (lambda volume: volume.update(voxel_size=0.0), "not positive"),
        (lambda volume: volume.update(max_depth=volume["min_depth"]), "no measurement falls"),
        (lambda volume: volume.update(truncation="wide"), "not a number"),
        (lambda volume: volume.pop("min_weight"), "min_weight"),
        (lambda volume: volume.update(colour="red"), "colour"),
    ],
)
def test_a_fusion_volume_that_is_not_a_volume_is_refused(published, edit, expected):
    def corrupt_the_volume(document):
        for case in document["cases"]:
            volume = case["scene"]["regions"]["fusion"]
            if volume is not None:
                edit(volume)
                return
        raise AssertionError("no case fuses")

    assert expected in _refused(published, corrupt_the_volume)


def test_a_window_that_leaves_its_frame_is_refused(published):
    def slide_the_window(document):
        window = document["cases"][0]["scene"]["window"]
        window["u0"] = FRAME_WIDTH - window["width"] + 1

    assert "leaves the" in _refused(published, slide_the_window)


def test_a_window_the_coverage_domain_is_not_scored_over_is_refused(published):
    def shrink_the_window(document):
        window = _general_cases(document)[0]["scene"]["window"]
        window["width"] -= 1
        window["sample_count"] = window["width"] * window["height"]

    assert "the window times the frames" in _refused(published, shrink_the_window)


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("input_returned", 0, "input_measured"),
        ("truth_returned", None, "truth_surface"),
    ],
)
def test_a_frame_count_the_coverage_contradicts_is_refused(published, key, value, expected):
    """The scene states what the frames returned; coverage states it again.

    Round one accepted a scene edited to return nothing beside coverage blocks
    that still counted the original returns. The counts are the same fact, and
    a report that carries both has to carry them agreeing.
    """

    def rewrite_a_frame(document):
        scene = document["cases"][0]["scene"]
        # Every frame count still has to be a count of a window that holds it,
        # so the forgery is the largest one the frame's own checks allow.
        scene["frames"][0][key] = scene["window"]["sample_count"] if value is None else value

    assert expected in _refused(published, rewrite_a_frame)


def test_an_error_denominator_that_is_not_the_dropout_is_refused(published):
    def widen_the_denominator(document):
        for case in document["cases"]:
            for candidate in case["candidates"]:
                if candidate["quality"] is None:
                    continue
                for measurement in candidate["quality"]["measurements"]:
                    if measurement["kind"] != "depth_completeness_reconstructed_error":
                        continue
                    if measurement["dropout_samples"] < 2:
                        continue
                    measurement["dropout_samples"] *= 2
                    measurement["attempted_fraction"] = (
                        measurement["entry_count"] / measurement["dropout_samples"]
                    )
                    return
        raise AssertionError("no reconstructed-error block with a dropout to widen")

    assert "not an improvement" in _refused(published, widen_the_denominator)


def test_a_coverage_relabel_its_own_error_population_contradicts_is_refused(published):
    """One reconstruction moved from surface into empty space, consistently.

    Coverage on its own cannot refute this: every count and fraction inside it
    still adds up. The confidence distribution beside it is moved with it, so
    that check cannot be what refuses the report either. The reconstructed-error
    block summarises exactly the reconstructions with truth behind them, and it
    still says the old number.
    """

    def relabel_one(document):
        for case in document["cases"]:
            for candidate in case["candidates"]:
                if candidate["quality"] is None:
                    continue
                names = {item["name"] for item in candidate["quality"]["measurements"]}
                if "reconstructed_depth" not in names:
                    continue
                coverage = _measurement(candidate, "coverage")
                if coverage["reconstructed_on_surface"] < 1 or coverage["truth_empty"] < 1:
                    continue
                # And with room left in empty space, so that the relabel this
                # control is about is what refuses the report rather than the
                # false-fill population bound beside it.
                if coverage["false_fill"] >= coverage["truth_empty"]:
                    continue
                confidence = _measurement(candidate, "confidence")
                coverage["reconstructed_on_surface"] -= 1
                coverage["false_fill"] += 1
                coverage["supported"] -= 1
                domain = coverage["domain_samples"]
                coverage["supported_fraction"] = coverage["supported"] / domain
                coverage["false_fill_fraction"] = coverage["false_fill"] / coverage["truth_empty"]
                coverage["dropout_recovered_fraction"] = (
                    0.0
                    if coverage["input_dropout"] == 0
                    else coverage["reconstructed_on_surface"] / coverage["input_dropout"]
                )
                _relabel_one_confident_sample(confidence)
                return
        raise AssertionError("no candidate reconstructs onto surface beside empty space")

    assert "not an improvement" in _refused(published, relabel_one)


def _relabel_one_confident_sample(confidence: Any) -> None:
    """Move one reconstruction from surface to empty space inside a confidence block.

    Every count that partitions the same population moves with it - the two
    totals, the confident subset when it would otherwise outgrow the population
    it is drawn from, the worst confident error when that subset empties, and
    the bin the sample sat in. A forger who left any of them stale would be
    refused by this block's own arithmetic rather than by the check the control
    is aimed at.
    """
    confidence["on_surface"] -= 1
    confidence["false_fill"] += 1
    if confidence["high_confidence_on_surface"] > confidence["on_surface"]:
        confidence["high_confidence_on_surface"] -= 1
        confidence["high_confidence_false_fill"] += 1
        if confidence["high_confidence_on_surface"] == 0:
            confidence["high_confidence_worst"] = 0.0
    for entry in confidence["bins"]:
        if entry["on_surface"] > 0:
            entry["on_surface"] -= 1
            entry["false_fill"] += 1
            return
    raise AssertionError("no confidence bin holds a reconstruction on surface")


@pytest.mark.parametrize(
    ("where", "key"),
    [
        (("scene", "window"), "u0"),
        (("scene", "intrinsics"), "cy"),
        (("scene", "regions"), "slot_columns"),
        (("scene",), "depth_scale"),
    ],
)
def test_a_nested_scene_key_that_is_missing_is_refused(published, where, key):
    def drop(document):
        record = document["cases"][0]
        for step in where:
            record = record[step]
        del record[key]

    assert key in _refused(published, drop)


@pytest.mark.parametrize(
    "where",
    [("scene", "window"), ("scene", "intrinsics"), ("scene", "regions"), ("scene",)],
)
def test_a_nested_scene_key_nobody_wrote_is_refused(published, where):
    def add(document):
        record = document["cases"][0]
        for step in where:
            record = record[step]
        record["helpfulness"] = 1.0

    assert "helpfulness" in _refused(published, add)


# --- the optional measurements, and the populations coverage partitions -------------------
#
# Every forgery below recomputes what a careful forger would: the gate values
# resolved from the measurement they name, the verdicts, the counts, the
# outcome lists and the fractions. That is the point of them. A mutation that
# left stale bookkeeping behind would be refused by a check written two rounds
# ago and would prove nothing about the check it is aimed at.


def _rebuild(case: Any, candidate: Any) -> None:
    """Every count, list and outcome one scored candidate states about itself."""
    report = candidate["quality"]
    report["gate_count"] = len(report["gates"])
    report["failures"] = [gate for gate in report["gates"] if not gate["passed"]]
    failing = sorted(gate["name"] for gate in report["failures"])
    report["passed"] = not failing
    evaluated = {gate["name"] for gate in report["gates"]}
    inapplicable = sorted({gate["name"] for gate in case["gates"]} - evaluated)
    for block in (candidate["expected"], candidate["observed"]):
        block["outcome"] = "fail" if failing else "pass"
        block["failed_gates"] = list(failing)
        block["inapplicable_gates"] = list(inapplicable)
    candidate["mismatches"] = []
    candidate["status"] = "pass"


def _resync(case: Any, candidate: Any) -> None:
    """Re-resolve every evaluated gate from its measurement, then rebuild the bookkeeping."""
    report = candidate["quality"]
    by_name = {item["name"]: item for item in report["measurements"]}
    gates = {gate["name"]: gate for gate in report["gates"]}
    for binding in candidate["gate_bindings"]:
        value: Any = by_name[binding["measurement"]]
        for step in binding["path"]:
            value = value[step]
        gate = gates[binding["gate"]]
        gate["measured"] = value
        if gate["comparison"] == "is":
            gate["passed"] = value is gate["limit"]
        elif gate["comparison"] == "<=":
            gate["passed"] = value <= gate["limit"]
        else:
            gate["passed"] = value >= gate["limit"]
    _rebuild(case, candidate)


def _refractionate(coverage: Any) -> None:
    """The six coverage fractions, recomputed from whatever the counts now say."""
    domain = coverage["domain_samples"]
    for key, numerator, denominator in (
        ("measured_fraction", coverage["measured"], domain),
        ("reconstructed_fraction", coverage["reconstructed"], domain),
        ("missing_fraction", coverage["missing"], domain),
        ("supported_fraction", coverage["supported"], domain),
        ("false_fill_fraction", coverage["false_fill"], coverage["truth_empty"]),
        (
            "dropout_recovered_fraction",
            coverage["reconstructed_on_surface"],
            coverage["input_dropout"],
        ),
    ):
        coverage[key] = 0.0 if denominator == 0 else numerator / denominator


def _erase_measurement(candidate: Any, name: str) -> None:
    """Delete one measurement no gate reads, and nothing else.

    The counterfactual tracking records carry no gate on purpose, so there are
    no bindings to take with them: deleting one leaves a report that is
    internally consistent everywhere except in what it no longer discloses.
    """
    report = candidate["quality"]
    assert any(item["name"] == name for item in report["measurements"]), name
    assert not any(binding["measurement"] == name for binding in candidate["gate_bindings"]), name
    report["measurements"] = [item for item in report["measurements"] if item["name"] != name]


def _drop_measurement(case: Any, candidate: Any, name: str) -> None:
    """Delete one measurement, the gates read from it and their bindings, consistently."""
    report = candidate["quality"]
    assert any(item["name"] == name for item in report["measurements"]), name
    report["measurements"] = [item for item in report["measurements"] if item["name"] != name]
    dropped = {
        binding["gate"] for binding in candidate["gate_bindings"] if binding["measurement"] == name
    }
    assert dropped, f"no gate reads {name}"
    candidate["gate_bindings"] = [
        binding for binding in candidate["gate_bindings"] if binding["gate"] not in dropped
    ]
    report["gates"] = [gate for gate in report["gates"] if gate["name"] not in dropped]
    _rebuild(case, candidate)


def test_deleting_the_whole_reconstructed_error_population_is_refused(published):
    """The whole-population form of the hard-sample omission attack.

    Not a few entries trimmed out of the error distribution - the distribution,
    its gates and its bindings removed together, with the gate names moved into
    the inapplicable sets and every count recomputed around the smaller report.
    Coverage next to it still says 139 reconstructions landed on real surface,
    and every one of those has held-out truth behind it to be scored against.
    """

    def delete_the_block(document):
        case = _case(document, "curved-surface")
        _drop_measurement(case, _candidate(case, "dense_nearest"), "reconstructed_depth")

    message = _refused(published, delete_the_block)
    assert "'reconstructed_depth'" in message
    assert "not an improvement" in message


def test_a_reconstructed_error_block_beside_no_reconstruction_is_refused(published):
    """The same rule from the other side: a block whose population does not exist."""

    def borrow_a_block(document):
        case = _case(document, "curved-surface")
        block = _measurement(_candidate(case, "dense_nearest"), "reconstructed_depth")
        baseline = _candidate(case, "baseline")
        assert _measurement(baseline, "coverage")["reconstructed_on_surface"] == 0
        baseline["quality"]["measurements"].append(json.loads(json.dumps(block)))

    assert "nothing for it to measure" in _refused(published, borrow_a_block)


def test_deleting_the_completion_work_block_from_spatial_completion_is_refused(published):
    """The whole work block, erased from a strict report - and now caught.

    ``spatial_completion`` is the one candidate in this corpus that is product
    code rather than a control, and it is the closed candidate identity - not
    a gate a case happened to declare - that says it owes a
    ``completion_work`` measurement. This mirrors the attack a review of an
    earlier round found undetected: select the ``plane-fronto`` case's
    ``spatial_completion`` candidate, remove ``completion_work``, its one gate
    (``completion_work.within_budget``) and its one binding, move that gate
    into both the expected and observed ``inapplicable_gates``, and recompute
    every other count so the report is otherwise internally consistent. A
    strict reader that only cross-checked ``completion_work`` when a report
    chose to carry it would accept this; deriving whether the block must
    exist from the candidate's own name does not.
    """

    def delete_the_block(document):
        case = _case(document, "plane-fronto")
        _drop_measurement(case, _candidate(case, "spatial_completion"), "completion_work")

    message = _refused(published, delete_the_block)
    assert "'spatial_completion'" in message
    assert "'completion_work'" in message


def test_a_completion_work_block_beside_a_candidate_that_never_ran_the_primitive_is_refused(
    published,
):
    """The same rule from the other side: a candidate the identity forbids it for.

    ``dense_nearest`` is scored in ``plane-fronto`` beside ``spatial_completion``
    but never calls :mod:`scanner3d.depth_completion`, so it has no run for a
    ``completion_work`` block to be an account of. Handing it one anyway -
    borrowed whole from the candidate that actually produced it - is refused
    on the candidate's own identity, independent of whatever the borrowed
    block's numbers say.
    """

    def borrow_the_block(document):
        case = _case(document, "plane-fronto")
        block = _measurement(_candidate(case, "spatial_completion"), "completion_work")
        dense_nearest = _candidate(case, "dense_nearest")
        assert not any(
            item["name"] == "completion_work" for item in dense_nearest["quality"]["measurements"]
        )
        dense_nearest["quality"]["measurements"].append(json.loads(json.dumps(block)))

    message = _refused(published, borrow_the_block)
    assert "'dense_nearest'" in message
    assert "did not come from a run that produced one" in message


def test_deleting_the_fusion_surface_agreement_is_refused(published):
    def delete_the_agreement(document):
        case = _case(document, "multiview-object")
        _drop_measurement(case, _candidate(case, "baseline"), "fusion_surface")

    message = _refused(published, delete_the_agreement)
    assert "'fusion_surface'" in message
    assert "not an improvement" in message


def test_a_fusion_surface_beside_a_volume_that_extracted_nothing_is_refused(published):
    def extract_nothing(document):
        case = _case(document, "multiview-object")
        candidate = _candidate(case, "baseline")
        fusion = _measurement(candidate, "fusion")
        fusion["surface_extracted"] = False
        fusion["surface_vertices"] = 0
        fusion["surface_faces"] = 0
        _resync(case, candidate)

    assert "nothing for it to measure" in _refused(published, extract_nothing)


def test_deleting_the_tracking_pose_comparison_is_refused(published):
    def delete_the_poses(document):
        case = _case(document, "multiview-object")
        _drop_measurement(case, _candidate(case, "baseline"), "tracking_pose")

    message = _refused(published, delete_the_poses)
    assert "'tracking_pose'" in message
    assert "not an improvement" in message


def test_deleting_the_counterfactual_completed_cloud_replay_is_refused(published):
    """The unfavourable record cannot be dropped while the favourable one stays."""

    def delete_the_counterfactual(document):
        case = _case(document, "multiview-object")
        # Not _drop_measurement: the counterfactual carries no gate, which is
        # exactly why its presence has to be required structurally.
        _erase_measurement(_candidate(case, "temporal_completion"), "tracking_completed_cloud")

    message = _refused(published, delete_the_counterfactual)
    assert "'tracking_completed_cloud'" in message


def test_deleting_the_counterfactual_pose_comparison_is_refused(published):
    def delete_the_poses(document):
        case = _case(document, "multiview-object")
        _erase_measurement(_candidate(case, "temporal_completion"), "tracking_completed_cloud_pose")

    message = _refused(published, delete_the_poses)
    assert "'tracking_completed_cloud_pose'" in message
    assert "not an improvement" in message


def test_a_measured_only_tracking_record_that_differs_between_candidates_is_refused(published):
    """The gated route is the case's fact, not the candidate's.

    The tracker is offered measured geometry only and no completion enters it,
    so a candidate publishing a different measured-only record either changed a
    measurement it was told to preserve or did not run the measured-only route.
    """

    def move_one_candidates_placement(document):
        case = _case(document, "multiview-object")
        candidate = _candidate(case, "temporal_completion")
        # Publish the counterfactual replay's own internally consistent poses
        # under the gated name. Nothing inside the record is wrong; what is
        # wrong is that it is not the measured-only route's answer.
        forged = json.loads(json.dumps(_measurement(candidate, "tracking_completed_cloud_pose")))
        forged["name"] = "tracking_pose"
        report = candidate["quality"]
        report["measurements"] = [
            forged if item["name"] == "tracking_pose" else item for item in report["measurements"]
        ]
        _resync(case, candidate)

    message = _refused(published, move_one_candidates_placement)
    assert "the case's fact" in message


def test_a_measured_only_tracking_count_that_differs_between_candidates_is_refused(published):
    def move_one_candidates_acceptance(document):
        case = _case(document, "multiview-object")
        candidate = _candidate(case, "temporal_completion")
        tracking = _measurement(candidate, "tracking")
        tracking["origins"][-1] = "anchor"
        _resync(case, candidate)

    assert "the case's fact" in _refused(published, move_one_candidates_acceptance)


def test_a_pose_comparison_beside_a_frame_the_tracker_refused_is_refused(published):
    def refuse_the_last_frame(document):
        case = _case(document, "multiview-object")
        candidate = _candidate(case, "baseline")
        tracking = _measurement(candidate, "tracking")
        tracking["frames_accepted"] -= 1
        tracking["accepted_all"] = False
        tracking["origins"][-1] = "refused"
        tracking["failure_types"] = ["no_overlap"]
        _resync(case, candidate)

    assert "nothing for it to measure" in _refused(published, refuse_the_last_frame)


def test_fusion_offered_frames_that_are_not_the_scenes_is_refused(published):
    def offer_one_more(document):
        case = _case(document, "multiview-object")
        candidate = _candidate(case, "baseline")
        _measurement(candidate, "fusion")["frames_offered"] += 1
        _resync(case, candidate)

    assert "beside a scene of 3" in _refused(published, offer_one_more)


def test_tracking_offered_frames_that_are_not_the_scenes_is_refused(published):
    def offer_one_more(document):
        case = _case(document, "multiview-object")
        candidate = _candidate(case, "baseline")
        tracking = _measurement(candidate, "tracking")
        tracking["frames_offered"] += 1
        tracking["frames_accepted"] += 1
        tracking["origins"].append("adjacent")
        _resync(case, candidate)

    assert "beside a scene of 3" in _refused(published, offer_one_more)


def test_a_pose_sequence_trimmed_to_drop_a_frame_is_refused(published):
    """The sequence trimmed to two poses, internally consistent all the way down.

    Its frame list, its pose count and both distribution counts agree with each
    other; only the scene it was placed over says three frames were offered and
    all three accepted, which is the only reason a pose comparison exists.
    """

    def trim(document):
        case = _case(document, "multiview-object")
        candidate = _candidate(case, "baseline")
        poses = _measurement(candidate, "tracking_pose")
        poses["pose_count"] -= 1
        poses["frames"] = poses["frames"][:-1]
        remaining = poses["frames"][0]
        for key, value in (
            ("rotation", remaining["rotation_degrees"]),
            ("translation", remaining["translation_error"]),
        ):
            poses[key].update(
                {"count": 1, "median": value, "rms": value, "p95": value, "worst": value}
            )
        _resync(case, candidate)

    assert "beside a scene of 3 frame(s)" in _refused(published, trim)


def test_more_measurements_than_the_input_returned_is_refused(published):
    """One sample moved from missing into measured, with every fraction recomputed.

    Nothing inside coverage stops adding up and the scene beside it is
    untouched, because the scene publishes what the input returned and this
    edit does not change that. It claims 731 measurements of 730 raw samples,
    and a measurement is a raw sample that came back.
    """

    def invent_a_measurement(document):
        case = _case(document, "plane-fronto")
        candidate = _candidate(case, "baseline")
        coverage = _measurement(candidate, "coverage")
        assert coverage["measured"] == coverage["input_measured"] == 730
        coverage["measured"] += 1
        coverage["supported"] += 1
        coverage["missing"] -= 1
        _refractionate(coverage)
        _resync(case, candidate)

    assert "cannot report more of them than it was given" in _refused(
        published, invent_a_measurement
    )


def test_a_truth_surface_that_is_not_its_own_input_and_dropout_is_refused(published):
    def grow_the_surface(document):
        case = _case(document, "curved-surface")
        candidate = _candidate(case, "baseline")
        coverage = _measurement(candidate, "coverage")
        coverage["truth_surface"] += 1
        coverage["truth_empty"] -= 1
        _refractionate(coverage)
        _resync(case, candidate)

    assert "survived its dropout mask or did not" in _refused(published, grow_the_surface)


def test_supporting_more_samples_than_the_truth_surface_is_refused(published):
    """One reconstruction relabelled out of empty space and onto surface.

    The counts still add up - reconstructions are still on surface or in empty
    space, supported is still measured plus on-surface - but the case's 658
    measurements and 140 on-surface reconstructions would need 798 real surface
    samples, and the scene has 797.
    """

    def support_one_too_many(document):
        case = _case(document, "curved-surface")
        candidate = _candidate(case, "dense_nearest")
        coverage = _measurement(candidate, "coverage")
        coverage["reconstructed_on_surface"] += 1
        coverage["false_fill"] -= 1
        coverage["supported"] += 1
        _refractionate(coverage)
        _resync(case, candidate)

    assert "against a truth_surface of 797" in _refused(published, support_one_too_many)


def test_filling_more_empty_space_than_the_scene_holds_is_refused(published):
    def fill_one_too_many(document):
        case = _case(document, "curved-surface")
        candidate = _candidate(case, "dense_nearest")
        coverage = _measurement(candidate, "coverage")
        coverage["reconstructed_on_surface"] -= 1
        coverage["false_fill"] += 1
        coverage["supported"] -= 1
        _refractionate(coverage)
        _resync(case, candidate)

    assert "against a truth_empty of 643" in _refused(published, fill_one_too_many)


def test_reconstructing_more_surface_than_the_truth_holds_is_refused(published):
    """Reconstructions claimed onto more surface than the scene's truth holds.

    Eighteen samples are moved out of measured and into reconstruction, and all
    but two reconstructions are then declared to have landed on real surface.
    Every sum inside coverage still holds and the scene is untouched: 798
    on-surface reconstructions against a truth surface of 797 is a population
    larger than the set it is drawn from.
    """

    def reconstruct_more_surface_than_exists(document):
        case = _case(document, "curved-surface")
        candidate = _candidate(case, "dense_nearest")
        coverage = _measurement(candidate, "coverage")
        coverage["measured"] -= 18
        coverage["spatially_reconstructed"] += 18
        coverage["reconstructed"] += 18
        coverage["reconstructed_on_surface"] = 798
        coverage["false_fill"] = coverage["reconstructed"] - 798
        coverage["supported"] = coverage["measured"] + 798
        _refractionate(coverage)
        # The error block beside it moved too, so the population bound is what
        # refuses this and not the error-population binding written beside it.
        error = _measurement(candidate, "reconstructed_depth")
        error["entry_count"] = error["reconstructed_on_surface"] = 798
        error["attempted_fraction"] = 798 / error["dropout_samples"]
        for key in ("absolute", "relative"):
            if error[key] is not None:
                error[key]["count"] = 798
        _resync(case, candidate)

    assert "states reconstructed_on_surface 798 against a truth_surface of 797" in _refused(
        published, reconstruct_more_surface_than_exists
    )


# --- the masks -------------------------------------------------------------------------------


def test_the_pseudo_random_stream_is_fixed_by_its_name():
    first = harness.pseudo_random("a-named-seed", 40)
    assert first == harness.pseudo_random("a-named-seed", 40)
    assert first != harness.pseudo_random("a-different-seed", 40)
    assert all(0.0 <= value < 1.0 for value in first)


def test_every_mask_kind_removes_the_shape_it_names():
    window = contract.Window(300, 188, 10, 4)
    truth = [12800] * 20 + [17920] * 20
    common: dict[str, Any] = {
        "window": window,
        "truth": truth,
        "seed": "mask-test",
        "what": "a mask",
    }
    speckle = harness.mask_indices({"kind": "speckle", "fraction": 0.3}, **common)
    blob = harness.mask_indices({"kind": "blob", "column": 5, "row": 1, "radius": 1.0}, **common)
    strip = harness.mask_indices({"kind": "strip", "first_column": 2, "last_column": 3}, **common)
    complementary = {
        phase: harness.mask_indices(
            {"kind": "complementary", "period": 3, "phase": phase}, **common
        )
        for phase in range(3)
    }
    assert 0 < len(speckle) < window.sample_count
    assert blob == {5, 14, 15, 16, 25}
    assert len(strip) == 2 * window.height
    assert set().union(*complementary.values()) == set(range(window.sample_count))
    assert not complementary[0] & complementary[1]


def test_an_unknown_mask_kind_is_refused():
    with pytest.raises(FixtureUnusable, match="mask kind"):
        harness.mask_indices(
            {"kind": "whatever"},
            window=contract.Window(300, 188, 4, 2),
            truth=[1] * 8,
            seed="s",
            what="a mask",
        )


def test_the_harnesss_closed_candidate_sets_match_what_the_contract_produces():
    """The harness's three closed sets, against the vocabulary they claim to mirror.

    ``harness.CANDIDATES_WITH_COMPLETION_WORK``,
    ``harness.CANDIDATES_WITH_TEMPORAL_WORK`` and
    ``harness.CANDIDATES_WITH_CONFIRMATIONS`` are what
    ``_bind_optional_measurements`` uses to require or forbid the three
    identity-decided blocks, named once each so the requirement cannot be
    re-derived two different ways in two places and drift apart. This is the
    check that would catch that drift: it runs every candidate this benchmark
    knows about and asserts each set is exactly the candidates that come back
    carrying that evidence - not the set the harness remembers, but the set
    :mod:`tests.depth_completeness` actually produces.

    Two scenes, because no single one runs every candidate: a one-frame case
    has no other frame for the temporal stage to read, and a case whose
    completion refuses nothing gives the forged-work control nothing to move.
    A candidate that ran on neither would slip through, so that is asserted
    too.
    """
    scenes = [
        builder._materialise(builder._fronto_masked()),
        builder._materialise(builder._temporal_confirmation()),
    ]
    completion: set[str] = set()
    temporal: set[str] = set()
    confirming: set[str] = set()
    ran: set[str] = set()
    for scene in scenes:
        for name, candidate in contract.CANDIDATES.items():
            try:
                result = candidate(scene)
            except UNAVAILABLE_HERE:
                continue
            ran.add(name)
            if result.work is not None:
                completion.add(name)
            if result.temporal is not None:
                temporal.add(name)
            if any(frame.confirmations for frame in result.frames):
                confirming.add(name)
    assert ran == set(contract.CANDIDATES)
    assert completion == harness.CANDIDATES_WITH_COMPLETION_WORK
    assert temporal == harness.CANDIDATES_WITH_TEMPORAL_WORK
    assert confirming == harness.CANDIDATES_WITH_CONFIRMATIONS


# --- the corpus is what the generator writes ----------------------------------------------------


def test_regenerating_the_corpus_reproduces_it_byte_for_byte(tmp_path):
    root = tmp_path / "cases"
    builder.write(root)
    for path in sorted(CASES.iterdir()):
        assert (root / path.name).read_bytes() == path.read_bytes(), path.name
    assert sorted(item.name for item in root.iterdir()) == sorted(
        item.name for item in CASES.iterdir()
    )


# --- the confidence and work evidence, which are new claims and so new attacks -------------------
#
# The two blocks task 029 added are the ones a completion has the most freedom
# in: one is its own opinion of its output, and the other is its own account of
# what it refused. Neither is recomputable from the artifact alone, so every
# forgery below is aimed at the arithmetic that DOES bind them - to coverage,
# to the policy published beside them, and to their own populations.


def _with_block(document: Any, name: str) -> Any:
    """The first scored candidate carrying the named measurement, and its case."""
    for case in _general_cases(document):
        for candidate in case["candidates"]:
            if candidate["quality"] is None:
                continue
            if any(item["name"] == name for item in candidate["quality"]["measurements"]):
                return case, candidate
    raise AssertionError(f"no candidate in the report carries a {name} measurement")


def _with_recovery(document: Any) -> Any:
    """The first candidate that reports a work record AND recovered something.

    Several controls below need a run with acceptances in it: a completion that
    refused every hole has no acceptance to restate and no support to strip.
    """
    for case in _general_cases(document):
        for candidate in case["candidates"]:
            if candidate["quality"] is None:
                continue
            if not any(
                item["name"] == "completion_work" for item in candidate["quality"]["measurements"]
            ):
                continue
            if _measurement(candidate, "completion_work")["accepted"] > 0:
                return case, candidate
    raise AssertionError("no candidate in the report recovered a hole and said what it cost")


def _without_reconstruction(document: Any) -> Any:
    for case in document["cases"]:
        for candidate in case["candidates"]:
            if candidate["quality"] is None:
                continue
            coverage = _measurement(candidate, "coverage")
            if coverage["reconstructed"] == 0:
                return case, candidate
    raise AssertionError("no scored candidate in the report reconstructed nothing")


def test_deleting_the_whole_confidence_distribution_is_refused(published):
    """The population-omission attack, aimed at the block that describes it.

    Removed together with the gate that reads it and that gate's binding, with
    the gate name moved into the inapplicable sets and every count recomputed
    around the smaller report - exactly what a careful forger would do. Coverage
    beside it still says how many reconstructions there were, and every one of
    them was published at some confidence.
    """

    def delete_the_block(document):
        case, candidate = _with_block(document, "confidence")
        _drop_measurement(case, candidate, "confidence")

    assert "is not optional" in _refused(published, delete_the_block)


def test_a_confidence_distribution_beside_no_reconstruction_is_refused(published):
    """The opposite boundary: a calibrated-looking block with nothing behind it."""

    def add_a_block(document):
        _case_with, source = _with_block(document, "confidence")
        case, candidate = _without_reconstruction(document)
        candidate["quality"]["measurements"].append(
            json.loads(json.dumps(_measurement(source, "confidence")))
        )
        _rebuild(case, candidate)

    assert "nothing for it to measure" in _refused(published, add_a_block)


def test_a_confidence_block_that_summarises_fewer_reconstructions_is_refused(published):
    """The block moved away from coverage on its own, with its bins kept tidy."""

    def summarise_fewer(document):
        case, candidate = _with_block(document, "confidence")
        confidence = _measurement(candidate, "confidence")
        confidence["reconstructed"] -= 1
        for entry in confidence["bins"]:
            if entry["count"] > 0 and entry["on_surface"] > 0:
                entry["count"] -= 1
                entry["on_surface"] -= 1
                break
        confidence["on_surface"] -= 1
        _resync(case, candidate)

    assert "covers every reconstruction" in _refused(published, summarise_fewer)


def test_a_confidence_histogram_missing_its_worst_bin_is_refused(published):
    def drop_the_bin(document):
        case, candidate = _with_block(document, "confidence")
        confidence = _measurement(candidate, "confidence")
        for entry in confidence["bins"]:
            if entry["count"] > 0:
                confidence["bins"].remove(entry)
                break
        else:
            raise AssertionError("every confidence bin is empty")
        _resync(case, candidate)

    assert "the histogram's edges are fixed" in _refused(published, drop_the_bin)


def test_confidence_bins_that_do_not_add_up_to_their_population_are_refused(published):
    def unbalance(document):
        case, candidate = _with_block(document, "confidence")
        confidence = _measurement(candidate, "confidence")
        for entry in confidence["bins"]:
            if entry["count"] > 0:
                entry["count"] -= 1
                entry["false_fill"] = entry["count"] - entry["on_surface"]
                break
        _resync(case, candidate)

    assert "the bins partition the whole population" in _refused(published, unbalance)


def test_a_confident_subset_larger_than_its_population_is_refused(published):
    def inflate(document):
        case, candidate = _with_block(document, "confidence")
        confidence = _measurement(candidate, "confidence")
        confidence["high_confidence"] = confidence["reconstructed"] + 1
        confidence["high_confidence_on_surface"] = confidence["reconstructed"] + 1
        confidence["high_confidence_false_fill"] = 0
        _resync(case, candidate)

    assert "confident" in _refused(published, inflate)


def test_a_surface_advantage_that_is_not_its_own_means_is_refused(published):
    def flatter_itself(document):
        case, candidate = _with_block(document, "confidence")
        confidence = _measurement(candidate, "confidence")
        confidence["surface_advantage"] = 0.25
        _resync(case, candidate)

    assert "the two means beside it" in _refused(published, flatter_itself)


def test_a_worst_confident_error_with_no_confident_sample_behind_it_is_refused(published):
    def claim_an_error(document):
        case, candidate = _with_block(document, "confidence")
        confidence = _measurement(candidate, "confidence")
        confidence["high_confidence"] = 0
        confidence["high_confidence_on_surface"] = 0
        confidence["high_confidence_false_fill"] = 0
        confidence["high_confidence_worst"] = 0.5
        _resync(case, candidate)

    assert "no confident reconstruction on surface" in _refused(published, claim_an_error)


def test_a_work_record_that_faced_fewer_holes_than_its_input_had_is_refused(published):
    """Conservatism measured against a denominator the candidate shrank."""

    def fewer_holes(document):
        case, candidate = _with_block(document, "completion_work")
        work = _measurement(candidate, "completion_work")
        work["holes"] -= 1
        work["attempted"] -= 1
        work["rejected"] -= 1
        for reason, count in work["rejections"].items():
            if count > 0:
                work["rejections"][reason] -= 1
                break
        work["work_budget"] = work["holes"] * (2 * work["policy"]["radius"] + 1) ** 2
        work["accepted_fraction"] = work["accepted"] / work["holes"]
        _resync(case, candidate)

    assert "the input's fact and not the candidate's" in _refused(published, fewer_holes)


def test_a_work_record_claiming_more_refusals_than_its_output_shows_is_refused(published):
    def refuse_more(document):
        case, candidate = _with_recovery(document)
        work = _measurement(candidate, "completion_work")
        work["accepted"] -= 1
        work["rejected"] += 1
        work["rejections"]["insufficient_support"] += 1
        work["accepted_fraction"] = work["accepted"] / work["holes"]
        # And the support total moved with it, because a forger who left that
        # behind would be refused by the policy bound rather than by the
        # binding to coverage this control is aimed at.
        work["support_samples"] = min(
            work["support_samples"], work["accepted"] * work["policy"]["max_support"]
        )
        _resync(case, candidate)

    assert "a conservatism it did not practise" in _refused(published, refuse_more)


def test_work_rejection_reasons_that_do_not_add_up_are_refused(published):
    def unbalance(document):
        case, candidate = _with_block(document, "completion_work")
        work = _measurement(candidate, "completion_work")
        for reason, count in work["rejections"].items():
            if count > 0:
                work["rejections"][reason] -= 1
                break
        _resync(case, candidate)

    assert "gives reasons for" in _refused(published, unbalance)


def test_a_rejection_reason_nobody_declared_is_refused(published):
    def invent_a_reason(document):
        case, candidate = _with_block(document, "completion_work")
        work = _measurement(candidate, "completion_work")
        work["rejections"]["felt_ambiguous"] = 0
        _resync(case, candidate)

    assert "the vocabulary is fixed" in _refused(published, invent_a_reason)


def test_work_spent_past_its_own_budget_is_refused(published):
    def overspend(document):
        case, candidate = _with_block(document, "completion_work")
        work = _measurement(candidate, "completion_work")
        work["work_units"] = work["work_budget"] + 1
        _resync(case, candidate)

    assert "not advisory" in _refused(published, overspend)


def test_a_budget_that_is_not_one_scan_per_hole_is_refused(published):
    """The budget is derived from the input and the policy, not stated by the run."""

    def widen_the_budget(document):
        case, candidate = _with_block(document, "completion_work")
        work = _measurement(candidate, "completion_work")
        work["work_budget"] *= 2
        _resync(case, candidate)

    assert "is not the candidate's to state" in _refused(published, widen_the_budget)


def test_support_a_published_policy_could_not_have_used_is_refused(published):
    def too_little_support(document):
        case, candidate = _with_recovery(document)
        work = _measurement(candidate, "completion_work")
        work["support_samples"] = work["accepted"]
        _resync(case, candidate)

    assert "the policy it publishes allows" in _refused(published, too_little_support)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_support", 0),
        ("max_component_samples", 0),
        ("max_residual_codes", 0.0),
        ("min_confidence", 2.0),
        ("passes", 2),
        ("max_support", 1),
    ],
)
def test_a_published_policy_the_product_would_not_accept_is_refused(published, field, value):
    """The policy is what makes an accepted sample reproducible. An impossible one is not.

    Rebuilt through scanner3d.depth_completion's own type rather than
    re-checked here, so there is one authority for what a policy is.
    """

    def break_the_policy(document):
        case, candidate = _with_recovery(document)
        _measurement(candidate, "completion_work")["policy"][field] = value
        _resync(case, candidate)

    assert "would accept" in _refused(published, break_the_policy)


def test_a_measurement_kind_this_reader_has_no_schema_for_is_refused(published):
    def rename_a_kind(document):
        _case, candidate = _with_block(document, "completion_work")
        _measurement(candidate, "completion_work")["kind"] = "depth_completeness_vibes"

    assert "refuses one it has no schema for" in _refused(published, rename_a_kind)


def test_moving_the_confidence_threshold_is_refused(published):
    """The line a completion's accuracy is judged at is the oracle's, not its own."""

    def raise_the_line(document):
        case, candidate = _with_block(document, "confidence")
        confidence = _measurement(candidate, "confidence")
        confidence["threshold"] = 0.99
        confidence["high_confidence"] = 0
        confidence["high_confidence_on_surface"] = 0
        confidence["high_confidence_false_fill"] = 0
        confidence["high_confidence_worst"] = 0.0
        _resync(case, candidate)

    assert "no candidate can move the line" in _refused(published, raise_the_line)


def test_shrinking_the_confident_population_past_its_own_bins_is_refused(published):
    """The one count in the block a gate reads, bounded by the bins beside it.

    The individual confidences are not published - a thousand of them per
    candidate is a report nobody reads - so the confident subset is not
    recomputable. It is bounded all the same: a reconstruction in a bin wholly
    at or above the threshold is confident whatever the summary says.
    """

    def hide_the_confident(document):
        for case in document["cases"]:
            for candidate in case["candidates"]:
                if candidate["quality"] is None:
                    continue
                if not any(
                    item["name"] == "confidence" for item in candidate["quality"]["measurements"]
                ):
                    continue
                confidence = _measurement(candidate, "confidence")
                edges = contract.CONFIDENCE_BIN_EDGES
                above = sum(
                    entry["count"]
                    for index, entry in enumerate(confidence["bins"])
                    if edges[index] >= confidence["threshold"]
                )
                if above == 0:
                    continue
                confidence["high_confidence"] = 0
                confidence["high_confidence_on_surface"] = 0
                confidence["high_confidence_false_fill"] = 0
                confidence["high_confidence_worst"] = 0.0
                _resync(case, candidate)
                return
        raise AssertionError("no candidate fills a bin wholly above the threshold")

    assert "beside bins that put it between" in _refused(published, hide_the_confident)


# --- the pose-compensated temporal stage's own account of itself, and the ------------------
# confirmations beside it - the two blocks a later task added, so new attacks -------------
#
# ``temporal_work`` is a run's own account of what it refused and what it spent, exactly
# like ``completion_work`` beside it, and ``confirmation`` is what independent posed
# evidence made of a spatial estimate it was offered. Both are bound to
# ``coverage.temporally_reconstructed_on_surface`` - the count coverage carries so the
# pose-compensated recovery cannot be relabelled as bookkeeping without a matching lie
# about a population the scene already fixes - so that count is attacked first, on its
# own, before the two blocks that lean on it.


def _without_block(document: Any, name: str) -> Any:
    """The first scored candidate that carries no measurement of this name."""
    for case in _general_cases(document):
        for candidate in case["candidates"]:
            if candidate["quality"] is None:
                continue
            if not any(item["name"] == name for item in candidate["quality"]["measurements"]):
                return case, candidate
    raise AssertionError(f"every scored candidate in the report carries a {name!r} measurement")


def test_deleting_temporally_reconstructed_on_surface_from_coverage_is_refused(published):
    def drop_the_key(document):
        _case, candidate = _scored(document)
        del _measurement(candidate, "coverage")["temporally_reconstructed_on_surface"]

    assert "'temporally_reconstructed_on_surface'" in _refused(published, drop_the_key)


def test_temporally_reconstructed_on_surface_inflated_above_temporally_reconstructed_is_refused(
    published,
):
    """The temporal share of the recovery moved past the temporal population itself."""

    def inflate(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "spatial_then_temporal_completion")
        coverage = _measurement(candidate, "coverage")
        assert coverage["temporally_reconstructed"] == 284
        assert coverage["reconstructed_on_surface"] == 368
        coverage["temporally_reconstructed_on_surface"] = 285
        _resync(case, candidate)

    assert "against a temporally_reconstructed of 284" in _refused(published, inflate)


def test_temporally_reconstructed_on_surface_inflated_above_reconstructed_on_surface_is_refused(
    published,
):
    """The same bound from the other side: the on-surface population itself."""

    def inflate(document):
        case = _case(document, "temporal-confirmation")
        candidate = _candidate(case, "naive_temporal_carry")
        coverage = _measurement(candidate, "coverage")
        assert coverage["temporally_reconstructed"] == 5
        assert coverage["reconstructed_on_surface"] == 2
        coverage["temporally_reconstructed_on_surface"] = 3
        _resync(case, candidate)

    assert "against a reconstructed_on_surface of 2" in _refused(published, inflate)


def test_temporally_reconstructed_on_surface_shrunk_below_its_floor_is_refused(published):
    """Recovery relabelled as bookkeeping: the floor this count cannot dodge.

    ``temporal_completion`` makes no spatial reconstruction at all in this case, so
    every one of its on-surface reconstructions has to be a temporal one. Shrinking
    the temporal share of them below that floor is exactly the attack of dodging a
    recovery gate by calling the recovery something else.
    """

    def shrink(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        coverage = _measurement(candidate, "coverage")
        assert coverage["spatially_reconstructed"] == 0
        assert coverage["reconstructed_on_surface"] == 284
        coverage["temporally_reconstructed_on_surface"] = 283
        _resync(case, candidate)

    assert "every on-surface sample the spatial ones cannot account for is a temporal one" in (
        _refused(published, shrink)
    )


# --- the temporal_work block ----------------------------------------------------------------


def test_deleting_temporal_work_from_a_candidate_that_runs_the_temporal_stage_is_refused(
    published,
):
    def delete_the_block(document):
        case, candidate = _with_block(document, "temporal_work")
        _drop_measurement(case, candidate, "temporal_work")

    message = _refused(published, delete_the_block)
    assert "'temporal_work'" in message
    assert "always produces a 'temporal_work' measurement" in message


def test_a_temporal_work_block_beside_a_candidate_that_never_ran_the_temporal_stage_is_refused(
    published,
):
    def borrow_the_block(document):
        _source_case, source = _with_block(document, "temporal_work")
        block = _measurement(source, "temporal_work")
        _case_without, candidate = _without_block(document, "temporal_work")
        candidate["quality"]["measurements"].append(json.loads(json.dumps(block)))

    message = _refused(published, borrow_the_block)
    assert "does not run the pose-compensated temporal stage" in message


def test_a_temporal_work_record_whose_attempts_do_not_add_up_is_refused(published):
    def unbalance(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        work["accepted"] += 1
        _resync(case, candidate)

    assert "accounts for" in _refused(published, unbalance)


def test_a_temporal_work_rejection_reason_nobody_declared_is_refused(published):
    def invent_a_reason(document):
        case, candidate = _with_block(document, "temporal_work")
        _measurement(candidate, "temporal_work")["rejections"]["felt_ambiguous"] = 0
        _resync(case, candidate)

    assert "the vocabulary is fixed" in _refused(published, invent_a_reason)


def test_temporal_work_rejection_reasons_that_do_not_add_up_are_refused(published):
    def unbalance(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        work["rejections"]["insufficient_support_frames"] -= 1
        _resync(case, candidate)

    assert "gives reasons for" in _refused(published, unbalance)


def test_confirmed_and_contradicted_past_accepted_in_temporal_work_is_refused(published):
    """Only reachable where independent evidence settled the sample - so bounded by it."""

    def inflate(document):
        case = _case(document, "temporal-confirmation")
        candidate = _candidate(case, "spatial_then_temporal_completion")
        work = _measurement(candidate, "temporal_work")
        assert work["confirmed"] == 1 and work["contradicted"] == 1 and work["accepted"] == 2
        work["confirmed"] = 2
        _resync(case, candidate)

    assert "either verdict is only reached where independent evidence settled the sample" in (
        _refused(published, inflate)
    )


def test_temporal_work_candidates_past_the_footprint_of_its_observations_is_refused(published):
    """Bounded by the footprint, not by the observation count.

    One projection is deposited into every target sample of its footprint it is
    close enough to - a point on an exact half-pixel boundary supports both
    sides of it - so more candidates than observations is legitimate and only
    ``observations * footprint_cells`` bounds them.
    """

    def inflate(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        assert work["observations"] == 4570
        assert work["policy"]["footprint_radius"] == 1  # a 3x3 stamp: nine cells
        work["candidates"] = 4570 * 9 + 1
        _resync(case, candidate)

    assert "one projection may reach every sample of its footprint" in _refused(published, inflate)


def test_temporal_work_candidates_within_the_footprint_of_its_observations_is_read(published):
    """The relation the strict reader must NOT refuse, bound from the other side."""

    def inflate(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        work["candidates"] = 4570 * 9
        _resync(case, candidate)

    # It may still be refused for arithmetic it breaks elsewhere, but never for
    # having deposited more candidates than it projected observations.
    try:
        harness.loads_report(_forge(published, inflate))
    except ReportUnusable as raised:
        assert "footprint" not in str(raised)


def test_temporal_work_observations_past_its_own_eligible_samples_is_refused(published):
    """One projection is one original raw measurement; a run cannot invent evidence."""

    def inflate(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        assert work["eligible_samples"] == 4570
        work["observations"] = 4571
        _resync(case, candidate)

    assert "a run cannot have more evidence than the frames returned" in _refused(
        published, inflate
    )


def test_temporal_work_spent_past_its_own_budget_is_refused(published):
    def overspend(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        work["work_units"] = work["work_budget"] + 1
        _resync(case, candidate)

    assert "not advisory" in _refused(published, overspend)


def test_temporal_within_budget_stated_inconsistently_is_refused(published):
    def flip(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        assert work["work_units"] <= work["work_budget"]
        work["within_budget"] = not work["within_budget"]
        _resync(case, candidate)

    assert "states within_budget beside" in _refused(published, flip)


def test_a_temporal_work_budget_that_is_not_what_its_policy_derives_is_refused(published):
    """The budget is the input and the policy, not a number a run gets to choose."""

    def widen_the_budget(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        work["work_budget"] += 1
        _resync(case, candidate)

    assert "which its own policy derives" in _refused(published, widen_the_budget)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_support_frames", 0),
        ("max_support_frames", 1),
        ("footprint_radius", -1),
        ("max_reprojection_distance", 0.0),
        ("confirmation_tolerance_codes", 0.0),
        ("min_confidence", 2.0),
    ],
)
def test_a_published_temporal_policy_the_product_would_not_accept_is_refused(
    published, field, value
):
    """Rebuilt through scanner3d.depth_temporal's own type, as the completion policy is."""

    def break_the_policy(document):
        case, candidate = _with_block(document, "temporal_work")
        _measurement(candidate, "temporal_work")["policy"][field] = value
        _resync(case, candidate)

    assert "would accept" in _refused(published, break_the_policy)


def test_temporal_work_accepted_that_disagrees_with_coverages_temporal_count_is_refused(
    published,
):
    def restate_accepted(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        assert work["confirmed"] == 0 and work["contradicted"] == 0
        work["accepted"] -= 1
        work["rejected"] += 1
        work["rejections"]["no_posed_evidence"] += 1
        work["accepted_fraction"] = work["accepted"] / work["holes"]
        _resync(case, candidate)

    assert "accepted a different number of hole(s) than coverage carries temporal" in _refused(
        published, restate_accepted
    )


def test_temporal_work_source_frames_that_is_not_every_other_frame_is_refused(published):
    def widen(document):
        case = _case(document, "temporal-motion")
        candidate = _candidate(case, "temporal_completion")
        work = _measurement(candidate, "temporal_work")
        assert work["source_frames"] == 6
        work["source_frames"] = 7
        _resync(case, candidate)

    assert "no frame is ever its own evidence" in _refused(published, widen)


# --- the confirmation block -----------------------------------------------------------------


def test_deleting_confirmation_from_spatial_then_temporal_completion_is_refused(published):
    def delete_the_block(document):
        case = _case(document, "temporal-confirmation")
        _drop_measurement(
            case, _candidate(case, "spatial_then_temporal_completion"), "confirmation"
        )

    message = _refused(published, delete_the_block)
    assert "'confirmation'" in message
    assert "always produces a 'confirmation' measurement" in message


def test_a_confirmation_block_beside_a_candidate_never_offered_a_spatial_estimate_is_refused(
    published,
):
    def borrow_the_block(document):
        case = _case(document, "temporal-confirmation")
        block = _measurement(_candidate(case, "spatial_then_temporal_completion"), "confirmation")
        temporal_only = _candidate(case, "temporal_completion")
        assert not any(
            item["name"] == "confirmation" for item in temporal_only["quality"]["measurements"]
        )
        temporal_only["quality"]["measurements"].append(json.loads(json.dumps(block)))

    message = _refused(published, borrow_the_block)
    assert "'temporal_completion'" in message
    assert "does not run that stage with an accepted spatial estimate to check" in message


def test_a_confirmation_record_whose_verdicts_do_not_add_up_is_refused(published):
    def lose_a_verdict(document):
        case = _case(document, "temporal-confirmation")
        candidate = _candidate(case, "spatial_then_temporal_completion")
        confirmation = _measurement(candidate, "confirmation")
        assert confirmation["spatial_estimates"] == 5
        confirmation["not_observed"] -= 1
        _resync(case, candidate)

    assert "every estimate is confirmed, contradicted or never observed" in _refused(
        published, lose_a_verdict
    )


def test_confirmed_false_is_undeletable_because_the_two_halves_have_to_add_up_to_it(published):
    """The one field this schema exists to keep honest, per its own docstring.

    Growing ``confirmed_false`` without growing ``confirmed`` to match is the same
    shape of attack as shrinking it to hide a bad confirmation: either way the two
    halves stop adding up to the whole, and that is what is checked here.
    """

    def unbalance(document):
        case = _case(document, "temporal-confirmation")
        candidate = _candidate(case, "spatial_then_temporal_completion")
        confirmation = _measurement(candidate, "confirmation")
        assert confirmation["confirmed"] == 1 and confirmation["confirmed_false"] == 0
        confirmation["confirmed_false"] = 1
        _resync(case, candidate)

    assert "every confirmation is one or the other" in _refused(published, unbalance)


def test_confirmed_and_contradicted_past_coverages_temporal_count_is_refused(published):
    def move_a_verdict(document):
        case = _case(document, "temporal-confirmation")
        candidate = _candidate(case, "spatial_then_temporal_completion")
        confirmation = _measurement(candidate, "confirmation")
        coverage = _measurement(candidate, "coverage")
        assert coverage["temporally_reconstructed"] == 2
        assert confirmation["confirmed"] == 1 and confirmation["contradicted"] == 1
        # One estimate moved from never-observed into contradicted, so the
        # totals inside the confirmation record still add up to its own
        # spatial_estimates and only the bound to coverage is what is left
        # to refuse this.
        confirmation["contradicted"] = 2
        confirmation["not_observed"] -= 1
        confirmation["not_observed_false"] = min(
            confirmation["not_observed_false"], confirmation["not_observed"]
        )
        _resync(case, candidate)

    assert "a settled estimate carries the provenance of the evidence that settled it" in (
        _refused(published, move_a_verdict)
    )


def test_settled_estimates_placed_on_more_surface_than_coverage_allows_is_refused(published):
    """The on-surface halves of the settled verdicts, bounded by coverage's own count.

    Nothing inside the confirmation record moves: both settled verdicts already sit
    at coverage's temporal recovery, so the only room left for this attack is
    coverage's own on-surface count, shrunk out from under them.
    """

    def shrink_the_surface(document):
        case = _case(document, "temporal-confirmation")
        candidate = _candidate(case, "spatial_then_temporal_completion")
        confirmation = _measurement(candidate, "confirmation")
        coverage = _measurement(candidate, "coverage")
        assert confirmation["confirmed_on_surface"] == 1
        assert confirmation["contradicted_on_surface"] == 1
        assert coverage["temporally_reconstructed_on_surface"] == 2
        assert coverage["reconstructed_on_surface"] == 2
        assert coverage["spatially_reconstructed"] == 3
        coverage["temporally_reconstructed_on_surface"] = 1
        _resync(case, candidate)

    assert "beside coverage that puts fewer temporal reconstructions there" in _refused(
        published, shrink_the_surface
    )


# --- the provenance-weighted fusion record ---------------------------------------------------
#
# The newest block, and the one with the most arithmetic a report could restate freely:
# every number in it is about a volume the reader cannot rebuild. So each relation the
# record can be forged through is recomputed from the record's OWN other fields, and each
# of those recomputations is attacked here. The two cases carrying a weighted volume carry
# it for opposite reasons - ``multiview-object`` has reconstructions to weight and
# ``aperture-ambiguity`` has to stay byte-identical to its measurements - so the attacks are
# split across both. Neither gates a gain: the multi-view result is that there is none.


def _weighted(document: Any, case_id: str, candidate_name: str) -> Any:
    case = _case(document, case_id)
    candidate = _candidate(case, candidate_name)
    return case, candidate, _measurement(candidate, "weighted_fusion")


def test_deleting_the_weighted_fusion_block_from_a_candidate_that_earned_one_is_refused(
    published,
):
    def delete_the_block(document):
        case, candidate, _record = _weighted(document, "multiview-object", "temporal_completion")
        _drop_measurement(case, candidate, "weighted_fusion_surface")
        _drop_measurement(case, candidate, "weighted_fusion")

    message = _refused(published, delete_the_block)
    assert "always produces a 'weighted_fusion' measurement" in message


def test_a_weighted_fusion_block_beside_a_candidate_with_no_accepted_result_is_refused(published):
    """The route a self-declared provenance does not have, defended in the reader too."""

    def borrow_the_block(document):
        case = _case(document, "aperture-ambiguity")
        source = _candidate(case, "spatial_completion")
        block = _measurement(source, "weighted_fusion")
        names = {
            binding["gate"]
            for binding in source["gate_bindings"]
            if binding["measurement"] == "weighted_fusion"
        }
        # The whole block, with its gates and its bindings, so the report stays
        # internally consistent and only the population check is left to refuse it.
        candidate = _candidate(case, "dense_nearest")
        surface = _measurement(source, "weighted_fusion_surface")
        candidate["quality"]["measurements"].append(json.loads(json.dumps(block)))
        candidate["quality"]["measurements"].append(json.loads(json.dumps(surface)))
        candidate["quality"]["gates"].extend(
            json.loads(
                json.dumps([gate for gate in source["quality"]["gates"] if gate["name"] in names])
            )
        )
        candidate["gate_bindings"].extend(
            json.loads(
                json.dumps(
                    [binding for binding in source["gate_bindings"] if binding["gate"] in names]
                )
            )
        )
        _resync(case, candidate)

    message = _refused(published, borrow_the_block)
    assert "does not run a run that carries an accepted product result" in message


def test_a_weighted_fusion_record_missing_a_field_is_refused(published):
    def drop_a_key(document):
        record = _weighted(document, "multiview-object", "temporal_completion")[2]
        del record["zero_weight_observations"]

    assert "zero_weight_observations" in _refused(published, drop_a_key)


def test_an_adapter_no_accepted_module_implements_is_refused(published):
    def rename_the_producer(document):
        record = _weighted(document, "multiview-object", "temporal_completion")[2]
        record["adapter"] = "learned_prior"

    assert "a producer nobody implemented is a provenance nobody checked" in _refused(
        published, rename_the_producer
    )


def test_provenance_counts_that_are_not_the_whole_frame_are_refused(published):
    """The denominator attack, on the block whose denominator is fixed by the frame."""

    def lose_the_holes(document):
        record = _weighted(document, "multiview-object", "temporal_completion")[2]
        assert record["missing_samples"] == 767497
        record["missing_samples"] -= 1

    assert "the four provenances partition the whole frame" in _refused(published, lose_the_holes)


def test_offered_weight_below_the_measurements_it_carries_is_refused(published):
    """A measurement weighs one here, and a report may not discount it."""

    def discount_a_measurement(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        assert record["measured_samples"] == 487
        record["offered_weight"] = 486.5
        _resync(case, candidate)

    assert "A measurement cannot be discounted here" in _refused(published, discount_a_measurement)


def test_a_reconstruction_that_offers_no_more_weight_than_the_measurements_is_refused(published):
    def zero_the_reconstructions(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        assert record["temporally_reconstructed_samples"] == 16
        record["offered_weight"] = float(record["measured_samples"])
        _resync(case, candidate)

    assert "an accepted reconstruction weighs strictly more than nothing" in _refused(
        published, zero_the_reconstructions
    )


def test_a_reconstruction_promoted_to_the_weight_of_a_measurement_is_refused(published):
    """The weighting's whole point, attacked from the top rather than the bottom."""

    def promote_the_reconstructions(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        record["offered_weight"] = float(
            record["measured_samples"] + record["temporally_reconstructed_samples"]
        )
        _resync(case, candidate)

    assert "a reconstruction weighs strictly less than the measurement beside it" in _refused(
        published, promote_the_reconstructions
    )


def test_a_contribution_worth_more_than_its_own_voxel_observations_is_refused(published):
    def inflate_the_contribution(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        assert record["voxel_updates"] == 5953
        record["contribution"] = 5953.5
        _resync(case, candidate)

    assert "none of them can be worth more than one" in _refused(
        published, inflate_the_contribution
    )


def test_an_unconfirmed_single_frame_estimate_given_any_contribution_at_all_is_refused(published):
    """Not a low weight - zero. The absolute claim, defended as an absolute."""

    def fuse_it_at_a_discount(document):
        case, candidate, record = _weighted(document, "aperture-ambiguity", "spatial_completion")
        assert record["spatially_excluded_samples"] == 2
        record["spatially_excluded_contribution"] = 0.25
        _resync(case, candidate)

    assert "is not fused at a discount" in _refused(published, fuse_it_at_a_discount)


def test_a_reconstruction_at_weight_that_left_the_field_untouched_is_refused(published):
    def claim_nothing_changed(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        assert record["matches_measured_only"] is False
        record["matches_measured_only"] = True
        _resync(case, candidate)

    assert "weight that changes nothing was not applied" in _refused(
        published, claim_nothing_changed
    )


def test_an_excluded_only_run_that_says_it_moved_the_field_is_refused(published):
    """Zero weight is absence. A run with nothing weighted cannot differ from measurement."""

    def claim_a_change(document):
        case, candidate, record = _weighted(document, "aperture-ambiguity", "spatial_completion")
        assert record["temporally_reconstructed_samples"] == 0
        record["matches_measured_only"] = False
        _resync(case, candidate)

    assert "Zero weight is absence, so it cannot" in _refused(published, claim_a_change)


def test_a_counterfactual_that_observes_less_than_the_weighted_field_is_refused(published):
    """The comparison that makes the exclusion load-bearing, bound to its own direction."""

    def shrink_the_counterfactual(document):
        case, candidate, record = _weighted(document, "aperture-ambiguity", "spatial_completion")
        assert record["unweighted_observed_voxels"] == 8457
        assert record["observed_voxels"] == 8445
        record["unweighted_observed_voxels"] = 8444
        _resync(case, candidate)

    assert "the counterfactual is the same evidence plus those estimates" in _refused(
        published, shrink_the_counterfactual
    )


def test_a_surface_standing_on_no_observed_voxel_at_all_is_refused(published):
    def empty_the_field(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        assert record["surface_vertices"] == 333
        record["observed_voxels"] = 0
        _resync(case, candidate)

    assert "unobserved space does not become surface" in _refused(published, empty_the_field)


# --- the output-facing comparison, and the ways it could be flattered ------------------------
#
# This is where a geometric claim would live, so every route to a false one is attacked:
# withholding the support frames from the baseline, stating a gain the record's own two
# measurements do not produce, and claiming one where zero weight was applied at all.


def test_a_baseline_with_the_supporting_views_withheld_is_refused(published):
    """The one attack that would have produced a gain on this corpus.

    Fuse the target frame's weighted input against a baseline that was not given
    the frames its reconstructions came out of, and the observed-voxel count
    rises. The record has to say how many frames its baseline was given, and it
    has to be all of them.
    """

    def withhold_the_support(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        assert record["frames_offered"] == 3
        assert record["measured_only_frames"] == 3
        record["measured_only_frames"] = 1
        record["measured_only_observed_voxels"] = record["withheld_support_observed_voxels"]
        record["observed_voxel_gain"] = (
            record["observed_voxels"] - record["measured_only_observed_voxels"]
        )
        _resync(case, candidate)

    assert "a baseline with the supporting views withheld" in _refused(
        published, withhold_the_support
    )


def test_a_gain_the_records_own_measurements_do_not_produce_is_refused(published):
    def state_a_gain(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        assert record["observed_voxel_gain"] == 0
        record["observed_voxel_gain"] = 225
        _resync(case, candidate)

    assert "A gain is the subtraction, never a number beside it" in _refused(
        published, state_a_gain
    )


def test_a_surface_coverage_gain_that_is_not_the_difference_is_refused(published):
    def improve_the_coverage(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        assert record["surface_coverage_gain"] == 0.0
        record["surface_coverage_gain"] = 0.05
        _resync(case, candidate)

    assert "whose difference is" in _refused(published, improve_the_coverage)


def test_a_gain_claimed_where_no_weight_was_applied_at_all_is_refused(published):
    """Zero weight is absence, so an excluded-only run has nothing to gain with."""

    def claim_recovered_space(document):
        case, candidate, record = _weighted(document, "aperture-ambiguity", "spatial_completion")
        assert record["temporally_reconstructed_samples"] == 0
        record["measured_only_observed_voxels"] = record["observed_voxels"] - 1
        record["observed_voxel_gain"] = 1
        _resync(case, candidate)

    assert "there is nothing there to gain with" in _refused(published, claim_recovered_space)


def test_a_withheld_support_subset_that_observes_more_than_the_whole_scene_is_refused(published):
    """The withheld comparison is a subset, and is bound to read like one."""

    def inflate_the_subset(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        assert record["withheld_support_frames"] == 1
        assert record["withheld_support_observed_voxels"] == 1949
        record["withheld_support_observed_voxels"] = record["measured_only_observed_voxels"] + 1
        _resync(case, candidate)

    assert "Fusing fewer frames cannot observe more of the volume" in _refused(
        published, inflate_the_subset
    )


def test_a_surface_coverage_reported_where_no_surface_was_extracted_is_refused(published):
    def cover_nothing(document):
        case, candidate, record = _weighted(document, "multiview-object", "temporal_completion")
        record["surface_extracted"] = False
        record["surface_vertices"] = 0
        record["surface_faces"] = 0
        record["surface_vertex_gain"] = -record["measured_only_surface_vertices"]
        _resync(case, candidate)

    assert "reports covering" in _refused(published, cover_nothing)
