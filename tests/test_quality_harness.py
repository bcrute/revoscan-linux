"""The fixture harness, proved load-bearing rather than merely runnable.

``tests/test_quality.py`` owns the measurements. This file owns the machinery
around them: that the shipped corpus produces exactly the outcomes it declares,
that corrupting a fixture or an expectation makes the harness FAIL, that a
deliberately bad case cannot pass merely because something failed, that every
independent case is still reported after one of them breaks, and that the
artifact refuses to be a report a reader could misread.

The controls are the point. A harness whose only evidence was "it ran and said
pass" would keep saying pass through any regression in itself, which is the
same failure mode the corpus exists to catch in the reconstruction.

Nothing here touches a scanner, a capture under ``/home/ben/scans``, a display
server, a GPU, the network or the clock.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

import scanner3d.quality as quality
from scanner3d.reconstruction_readiness import (
    COVERAGE_UNSUPPORTED,
    POSE_EXCURSION_INSUFFICIENT,
    POSE_MOTION_BELOW_NOISE,
    POSE_VIEWPOINTS_INSUFFICIENT,
    REASONS,
    SURFACE_UNSUPPORTED,
    ReadinessRecordUnusable,
    ReconstructionReadiness,
    evaluate_readiness,
)
from tests import build_quality_fixtures as builder
from tests import quality_harness as harness

CASES = harness.CORPUS


def _cases_by_id(document: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    entries = document["cases"]
    assert isinstance(entries, list)
    out: dict[str, Mapping[str, object]] = {}
    for entry in entries:
        assert isinstance(entry, Mapping)
        out[str(entry["case_id"])] = entry
    return out


def _copy_corpus(tmp_path: Path) -> Path:
    """A writable copy of the shipped corpus, so a control can corrupt it.

    The corpus itself is never modified by anything in this file - it is
    evidence, and a test that edited it in place would leave the repository
    holding a fixture nobody reviewed.
    """
    root = tmp_path / "cases"
    root.mkdir()
    for path in sorted(CASES.iterdir()):
        if path.is_file():
            (root / path.name).write_bytes(path.read_bytes())
    return root


def _reseal(root: Path, name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Edit one case and update the manifest to match, so the DIGEST passes.

    Deliberate: resealing is what makes the following controls tests of the
    measurement rather than tests of sha256. The unsealed case - an edit the
    manifest does not know about - has its own control below.
    """
    path = root / name
    case = json.loads(path.read_text(encoding="utf-8"))
    mutate(case)
    text = builder.serialise(case)
    path.write_text(text, encoding="utf-8")
    manifest_path = root / harness.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["name"] == name:
            entry["bytes"] = len(text.encode("utf-8"))
            entry["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest_path.write_text(builder.serialise(manifest), encoding="utf-8")


# --- the shipped corpus -------------------------------------------------------------


def test_the_shipped_corpus_runs_and_every_case_matches_its_declared_outcome():
    """The whole point, in one assertion.

    If this fails, either a reconstruction primitive changed what it measures
    or a fixture stopped describing what it measures. The artifact names which.
    """
    document = harness.run()

    assert document["decision"] == "pass"
    totals = document["totals"]
    assert isinstance(totals, Mapping)
    assert totals["failed"] == 0
    assert totals["malformed"] == 0
    assert totals["passed"] == totals["cases"]
    assert int(str(totals["cases"])) >= 8


def test_the_corpus_holds_a_clean_case_and_each_named_failure_mode_separately():
    """A missing surface, invented surface and a lumpy plane fail DIFFERENT gates.

    This is what a single quality score cannot do. If two of these ever name
    the same failing set, the corpus has stopped distinguishing the failures it
    was built to distinguish.
    """
    cases = _cases_by_id(harness.run())

    expectations = {
        name: (case["expected"], case["observed"])
        for name, case in cases.items()
        if isinstance(case["expected"], Mapping)
    }
    for name, (expected, observed) in expectations.items():
        assert isinstance(expected, Mapping) and isinstance(observed, Mapping)
        assert expected["failed_gates"] == observed["failed_gates"], name

    missing = cases["missing-surface"]["observed"]
    spurious = cases["spurious-geometry"]["observed"]
    lumpy = cases["lumpy-panel"]["observed"]
    assert isinstance(missing, Mapping) and isinstance(spurious, Mapping)
    assert isinstance(lumpy, Mapping)
    assert missing["failed_gates"] == [
        "cloud.reference_to_observed.coverage",
        "cloud.reference_to_observed.worst",
    ]
    assert spurious["failed_gates"] == [
        "cloud.observed_to_reference.coverage",
        "cloud.observed_to_reference.p95",
    ]
    assert lumpy["failed_gates"] == ["panel.residual.p95", "panel.residual.worst"]
    assert cases["clean-surface"]["observed"] == {"outcome": "pass", "failed_gates": []}
    assert cases["clean-panel"]["observed"] == {"outcome": "pass", "failed_gates": []}


def test_the_corpus_preserves_the_byte_bound_false_bridge_fixture_beside_it():
    """The 2026-08-28 record stays exactly where and what it was.

    This corpus is a sibling directory, not a rewrite of that evidence. If the
    bridge fixture or its manifest moved into the new schema, the record that
    the relocalization policy exists because of would no longer be the bytes
    the capture wrote.
    """
    bridge = CASES.parent / "bridge-20260828-062457-segment-0001.json"
    assert bridge.is_file()
    manifest = json.loads((CASES.parent / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert [entry["name"] for entry in manifest["files"]] == [bridge.name]


def test_the_harness_applies_no_threshold_the_corpus_does_not_state():
    """Every gate in the artifact came from a fixture, with its own unit.

    A limit that appeared here without appearing in a case file would be a
    hidden standard nobody voted on - which is exactly what scanner3d.quality
    refuses to own, and this harness must not quietly acquire.
    """
    shipped: set[tuple[str, str, float | bool]] = set()
    for path in sorted(CASES.glob("*.json")):
        if path.name == harness.MANIFEST_NAME:
            continue
        case = json.loads(path.read_text(encoding="utf-8"))
        for gate in case["gates"]:
            shipped.add((gate["name"], gate["comparison"], gate["limit"]))

    for entry in _cases_by_id(harness.run()).values():
        report = entry["quality"]
        assert isinstance(report, Mapping)
        gates = report["gates"]
        assert isinstance(gates, list)
        for gate in gates:
            assert isinstance(gate, Mapping)
            key = (str(gate["name"]), str(gate["comparison"]), gate["limit"])
            assert key in shipped, key
            assert str(gate["unit"]).strip()
            assert str(gate["context"]).strip()


# --- the corruption controls --------------------------------------------------------


def test_altered_geometry_fails_the_harness_and_later_cases_are_still_reported(tmp_path):
    """Move the clean surface and the harness must stop calling it clean.

    The second half is as important as the first: the corpus is evaluated in
    name order, and every case after the broken one has to still be in the
    artifact. A harness that stopped at the first failure would hide every
    other regression in the same change.
    """
    root = _copy_corpus(tmp_path)

    def displace(case: dict[str, Any]) -> None:
        cloud = case["measurements"][0]
        cloud["observed"] = [[x + 5.0, y, z] for x, y, z in cloud["observed"]]

    _reseal(root, "clean-surface.json", displace)
    document = harness.run(root)

    assert document["decision"] == "fail"
    cases = _cases_by_id(document)
    assert cases["clean-surface"]["status"] == "fail"
    assert cases["clean-surface"]["mismatches"]
    # Everything that comes after it alphabetically is still here and still
    # passing, so the failure is one case's and not the run's.
    for name in ("lumpy-panel", "missing-surface", "spurious-geometry", "uniform-scale"):
        assert cases[name]["status"] == "pass", name
    totals = document["totals"]
    assert isinstance(totals, Mapping)
    assert totals["failed"] == 1


def test_an_altered_expected_outcome_fails_the_harness(tmp_path):
    """The expectation is load-bearing in both directions.

    Corrupting what a case CLAIMS, without touching the geometry, has to fail
    just as loudly as corrupting the geometry - otherwise a fixture could be
    quietly weakened into one that asserts nothing.
    """
    root = _copy_corpus(tmp_path)

    def weaken(case: dict[str, Any]) -> None:
        case["expectation"] = {
            "outcome": "fail",
            "failed_gates": ["cloud.observed_to_reference.p95"],
        }

    _reseal(root, "clean-surface.json", weaken)
    cases = _cases_by_id(harness.run(root))

    assert cases["clean-surface"]["status"] == "fail"
    assert cases["clean-surface"]["mismatches"] == [
        "expected the case to fail and it did pass",
        "gate 'cloud.observed_to_reference.p95' was expected to fail and passed",
    ]


def test_a_bad_case_that_stops_failing_is_an_unexpected_pass(tmp_path):
    """Loosen a known-bad case's gates and the harness refuses the silence.

    This is the regression that matters most: a change that made a gate too
    permissive would otherwise turn the whole known-bad corpus green.
    """
    root = _copy_corpus(tmp_path)

    def loosen(case: dict[str, Any]) -> None:
        for gate in case["gates"]:
            if gate["comparison"] == "<=":
                gate["limit"] = 1000.0
            else:
                gate["limit"] = 0.0

    _reseal(root, "missing-surface.json", loosen)
    cases = _cases_by_id(harness.run(root))

    assert cases["missing-surface"]["status"] == "fail"
    mismatches = cases["missing-surface"]["mismatches"]
    assert isinstance(mismatches, list)
    assert "expected the case to fail and it did pass" in mismatches


def test_a_bad_case_cannot_pass_by_failing_a_gate_it_was_not_written_for(tmp_path):
    """ "Something failed" is not the expected failure.

    The missing-surface case is declared to fail its two REVERSE gates and to
    pass both forward ones. Break the forward direction as well and the case
    still "fails" - and the harness has to call that a mismatch, because a
    reconstruction that broke in a new way is not the regression this fixture
    was proving stayed caught.
    """
    root = _copy_corpus(tmp_path)

    def also_displace(case: dict[str, Any]) -> None:
        cloud = case["measurements"][0]
        cloud["observed"] = [[x + 9.0, y, z] for x, y, z in cloud["observed"]]

    _reseal(root, "missing-surface.json", also_displace)
    cases = _cases_by_id(harness.run(root))

    entry = cases["missing-surface"]
    assert entry["status"] == "fail"
    observed = entry["observed"]
    assert isinstance(observed, Mapping)
    assert observed["outcome"] == "fail"
    assert entry["mismatches"] == [
        "gate 'cloud.observed_to_reference.coverage' failed and was not expected to",
        "gate 'cloud.observed_to_reference.p95' failed and was not expected to",
    ]


def test_a_fixture_edited_without_the_manifest_is_refused_as_malformed(tmp_path):
    """An unsealed edit is caught before its own claims are even read.

    A fixture is evidence. One whose bytes changed is no longer the thing that
    was reviewed, and the harness says so rather than measuring it.
    """
    root = _copy_corpus(tmp_path)
    path = root / "lumpy-panel.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    case["measurements"][0]["points"][0][0] += 1.0
    path.write_text(builder.serialise(case), encoding="utf-8")

    document = harness.run(root)
    cases = _cases_by_id(document)

    assert cases["lumpy-panel"]["status"] == "malformed"
    assert "sha256" in str(cases["lumpy-panel"]["error"])
    assert cases["lumpy-panel"]["quality"] is None
    assert cases["missing-surface"]["status"] == "pass"
    totals = document["totals"]
    assert isinstance(totals, Mapping)
    assert totals["malformed"] == 1
    assert document["decision"] == "fail"


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(lambda case: case.pop("why"), "missing", id="missing key"),
        pytest.param(lambda case: case.update(extra=1), "unknown", id="unknown key"),
        pytest.param(lambda case: case.update(schema_version=2), "schema version", id="version"),
        pytest.param(
            lambda case: case.update(kind="something-else"),
            "is not a scanner3d.quality-harness-case",
            id="kind",
        ),
        pytest.param(
            lambda case: case["measurements"][0].update(coverage_tolerance=True),
            "not a number",
            id="boolean for a number",
        ),
        pytest.param(
            lambda case: case["gates"][0].update(comparison="~="),
            "refuses one it cannot evaluate",
            id="unknown comparison",
        ),
        pytest.param(
            lambda case: case["gates"].append(dict(case["gates"][0])),
            "two gates called",
            id="duplicate gate name",
        ),
        pytest.param(
            lambda case: case["gates"][0].update(path=["nowhere"]),
            "no 'nowhere'",
            id="path to nothing",
        ),
        pytest.param(
            lambda case: case["expectation"].update(outcome="fail", failed_gates=["no.such.gate"]),
            "states no such gate",
            id="expects a gate that does not exist",
        ),
    ],
)
def test_a_malformed_case_is_refused_and_the_rest_of_the_corpus_still_reports(
    tmp_path, mutate, expected
):
    """Nine separate ways a fixture can be wrong, each named in the artifact.

    Every one of them is a HARNESS FAILURE rather than a skipped case: a
    fixture that cannot be read asserts nothing, and a run that quietly
    dropped it would report a smaller corpus as a passing one.
    """
    root = _copy_corpus(tmp_path)
    _reseal(root, "clean-surface.json", mutate)

    document = harness.run(root)
    cases = _cases_by_id(document)

    assert document["decision"] == "fail"
    assert cases["clean-surface"]["status"] == "malformed"
    assert expected in str(cases["clean-surface"]["error"])
    assert cases["uniform-scale"]["status"] == "pass"


def test_a_non_finite_value_in_a_fixture_is_refused(tmp_path):
    """``NaN`` is not JSON, and Python's decoder accepts it anyway.

    Written as raw text because ``json.dumps`` will not produce it without
    being asked - which is the whole hazard: a fixture carrying one would be
    read by this repository and rejected by anything else.
    """
    root = _copy_corpus(tmp_path)
    path = root / "clean-panel.json"
    text = path.read_text(encoding="utf-8").replace('"limit": 0.001', '"limit": NaN', 1)
    path.write_text(text, encoding="utf-8")
    manifest_path = root / harness.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["name"] == path.name:
            entry["bytes"] = len(text.encode("utf-8"))
            entry["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest_path.write_text(builder.serialise(manifest), encoding="utf-8")

    cases = _cases_by_id(harness.run(root))

    assert cases["clean-panel"]["status"] == "malformed"
    assert "NaN" in str(cases["clean-panel"]["error"])


def test_a_duplicate_key_inside_one_fixture_object_is_refused(tmp_path):
    """Python keeps the last one silently; a fixture must not be able to.

    A duplicated key is a file whose reviewed text and effective content
    differ, which is the one thing a piece of evidence may never be.
    """
    root = _copy_corpus(tmp_path)
    path = root / "clean-panel.json"
    text = path.read_text(encoding="utf-8").replace(
        '"case_id": "clean-panel",', '"case_id": "clean-panel",\n  "case_id": "something-else",', 1
    )
    path.write_text(text, encoding="utf-8")
    manifest_path = root / harness.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["name"] == path.name:
            entry["bytes"] = len(text.encode("utf-8"))
            entry["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    manifest_path.write_text(builder.serialise(manifest), encoding="utf-8")

    cases = _cases_by_id(harness.run(root))

    assert cases["clean-panel"]["status"] == "malformed"
    assert "twice" in str(cases["clean-panel"]["error"])


# --- corpus-level refusals ----------------------------------------------------------


def test_a_missing_corpus_is_refused_rather_than_reported_as_an_empty_pass(tmp_path):
    """No corpus is not a passing run. Exit 2 exists for exactly this."""
    with pytest.raises(harness.HarnessRefused):
        harness.run(tmp_path / "nowhere")
    assert harness.main(["--corpus", str(tmp_path / "nowhere")]) == 2


def test_a_case_file_the_manifest_does_not_name_is_refused(tmp_path):
    """A case nobody listed is a case nobody reviewed."""
    root = _copy_corpus(tmp_path)
    (root / "extra-case.json").write_text("{}", encoding="utf-8")

    with pytest.raises(harness.HarnessRefused, match="does not name"):
        harness.run(root)


def test_a_manifest_entry_with_no_file_behind_it_is_refused(tmp_path):
    """The other direction: a case that silently stopped running."""
    root = _copy_corpus(tmp_path)
    (root / "uniform-scale.json").unlink()

    with pytest.raises(harness.HarnessRefused, match="not a file"):
        harness.run(root)


def test_two_cases_with_one_id_are_refused(tmp_path):
    """Two cases with one name could not both be reported, so neither is."""
    root = _copy_corpus(tmp_path)
    manifest_path = root / harness.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["name"] == "clean-panel.json":
            entry["case_id"] = "clean-surface"
    manifest_path.write_text(builder.serialise(manifest), encoding="utf-8")

    with pytest.raises(harness.HarnessRefused, match="more than one file"):
        harness.run(root)


@pytest.mark.parametrize(
    "name",
    ["../outside.json", "nested/case.json", "/etc/passwd", "Clean-Panel.json", "-dash.json"],
    ids=["parent", "nested", "absolute", "upper case", "leading dash"],
)
def test_a_manifest_naming_a_path_instead_of_a_fixture_is_refused(tmp_path, name):
    """A fixture name is a name, not a path, and never a route out of the corpus."""
    root = _copy_corpus(tmp_path)
    manifest_path = root / harness.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["name"] = name
    manifest_path.write_text(builder.serialise(manifest), encoding="utf-8")

    with pytest.raises(harness.HarnessRefused):
        harness.run(root)


def test_a_symlinked_fixture_is_refused_even_when_it_points_somewhere_valid(tmp_path):
    """A link passes every name check and can still point anywhere on the machine.

    It is refused even when the target is a perfectly good case, because the
    property being protected is that a fixture IS the bytes in the corpus.
    """
    root = _copy_corpus(tmp_path)
    real = root / "clean-panel.json"
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_bytes(real.read_bytes())
    real.unlink()
    real.symlink_to(elsewhere)

    with pytest.raises(harness.HarnessRefused, match="symbolic link"):
        harness.run(root)


# --- the artifact itself ------------------------------------------------------------


def test_two_runs_over_an_unchanged_corpus_are_byte_identical(tmp_path):
    """Determinism, asserted on the published bytes and not on the object.

    A rerun that differed would make "the report changed" useless as a signal,
    which is the only reason to keep an artifact at all.
    """
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    assert harness.main(["--out", str(first), "--quiet"]) == 0
    assert harness.main(["--out", str(second), "--quiet"]) == 0

    assert first.read_bytes() == second.read_bytes()
    text = first.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    # Everything except the limitations, which say IN PROSE that no duration
    # appears - the words are in the artifact precisely because the numbers
    # are not.
    document = json.loads(text)
    document.pop("limitations")
    measured = json.dumps(document)
    for forbidden in ("timing", "duration", "elapsed", "hostname", "seconds", str(tmp_path)):
        assert forbidden not in measured, forbidden


def test_the_published_artifact_is_accepted_by_the_reader_that_wrote_it(tmp_path):
    """Round trip through the strict reader, from the bytes on disk."""
    out = tmp_path / "report.json"
    harness.main(["--out", str(out), "--quiet"])

    document = json.loads(out.read_text(encoding="utf-8"))
    assert harness.validate_report(document) is not None
    assert document["kind"] == harness.REPORT_KIND
    assert document["limitations"] == list(harness.LIMITATIONS)


def test_a_report_that_cannot_be_serialised_leaves_no_artifact_at_all(tmp_path):
    """Serialise first, rename last: a reader never sees half a report.

    The non-finite value is refused by ``allow_nan=False`` before any file
    exists, so neither the destination nor a partial file beside it is left
    behind - and a stale artifact from an earlier run is not silently
    presented as this run's.
    """
    out = tmp_path / "report.json"
    document = dict(harness.run())
    document["decision"] = math.inf

    with pytest.raises(ValueError):
        harness.publish(document, out)

    assert not out.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == []


def test_publishing_into_a_directory_that_does_not_exist_is_refused(tmp_path):
    """Exit 2, not a traceback and not a silently unwritten report."""
    document = harness.run()
    with pytest.raises(harness.HarnessRefused):
        harness.publish(document, tmp_path / "nowhere" / "report.json")
    assert harness.main(["--out", str(tmp_path / "nowhere" / "report.json"), "--quiet"]) == 2


def test_the_artifact_replaces_an_earlier_one_atomically(tmp_path):
    """A rerun leaves exactly one file, and it is the new one."""
    out = tmp_path / "report.json"
    out.write_text("stale", encoding="utf-8")
    harness.main(["--out", str(out), "--quiet"])

    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.json"]
    assert json.loads(out.read_text(encoding="utf-8"))["kind"] == harness.REPORT_KIND


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(lambda doc: doc.pop("totals"), "missing", id="missing key"),
        pytest.param(lambda doc: doc.update(extra=1), "unknown", id="unknown key"),
        pytest.param(lambda doc: doc.update(schema_version=99), "schema version", id="version"),
        pytest.param(lambda doc: doc.update(kind="other"), "kind", id="kind"),
        pytest.param(lambda doc: doc.update(limitations=[]), "limitations", id="limitations"),
        pytest.param(
            lambda doc: doc["cases"].append(dict(doc["cases"][0])),
            "twice",
            id="duplicate case id",
        ),
        pytest.param(
            lambda doc: doc["totals"].update(passed=0),
            "beside cases that are",
            id="totals that do not add up",
        ),
        pytest.param(
            lambda doc: doc.update(decision="fail"),
            "contradicts its own totals",
            id="decision against its own totals",
        ),
        pytest.param(
            lambda doc: doc["totals"].update(cases=True),
            "not a count",
            id="boolean for a count",
        ),
        pytest.param(
            lambda doc: doc["cases"][0].update(mismatches=["invented"]),
            "expected versus observed gives",
            id="a passing case that names a mismatch",
        ),
        pytest.param(
            lambda doc: doc["cases"][0].update(error="invented"),
            "beside error",
            id="a passing case that names an error",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["quality"]["gates"][0].update(measured=True),
            "not the same kind of thing",
            id="a boolean measured against a number",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["quality"]["gates"].append(
                dict(doc["cases"][0]["quality"]["gates"][0])
            ),
            "twice",
            id="duplicate gate name",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["observed"].update(failed_gates=["invented"]),
            "beside 1 failing gate(s)",
            id="a failed gate that is not in the case",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["observed"].update(
                outcome="fail", failed_gates=["invented"]
            ),
            "beside a quality report that passed",
            id="an observed failure the gates do not show",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["fixture"].update(sha256="not a digest"),
            "not a sha256",
            id="fixture digest that is not one",
        ),
    ],
)
def test_a_report_that_could_be_misread_is_refused(mutate, expected):
    """Fifteen shapes a corrupted or hand-edited artifact could otherwise take.

    The reader is the same one ``run()`` validates its own output with, so an
    artifact this repository publishes is one it would also accept - and
    everything above is a report a consumer must never be handed as evidence.
    """
    document = json.loads(harness.to_json(harness.run()))
    mutate(document)

    with pytest.raises(harness.ReportUnusable) as raised:
        harness.validate_report(document)
    assert expected in str(raised.value)


# --- forged reports, and the reader that must refuse them ---------------------------


def _failing_case(document: Mapping[str, Any]) -> dict[str, Any]:
    """One case the shipped corpus reports as an expected failure."""
    return _cases_by_id(document)["missing-surface"]  # type: ignore[return-value]


def _forge_mismatch_prose(document: dict[str, Any]) -> None:
    """Invented findings, with the totals and decision adjusted to match.

    The interesting forgery: every number in the report agrees with every
    other number, and the only thing wrong with it is that the prose describes
    a disagreement between expected and observed that the same entry shows did
    not happen.
    """
    case = document["cases"][0]
    case["status"] = "fail"
    case["mismatches"] = ["gate 'invented' failed and was not expected to"]
    document["totals"]["passed"] -= 1
    document["totals"]["failed"] += 1
    document["decision"] = "fail"


@pytest.mark.parametrize(
    ("forge", "expected"),
    [
        pytest.param(
            lambda doc: doc["corpus"].update(kind="scanner3d.something-else"),
            "states kind",
            id="corpus kind",
        ),
        pytest.param(
            lambda doc: doc["corpus"].update(schema_version=2),
            "fixture schema version",
            id="corpus schema version",
        ),
        pytest.param(
            lambda doc: doc["corpus"].update(manifest_sha256="deadbeef"),
            "not a sha256",
            id="corpus manifest digest",
        ),
        pytest.param(
            lambda doc: doc["corpus"].update(case_count=doc["corpus"]["case_count"] + 1),
            "fewer cases than the corpus",
            id="corpus case count",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["fixture"].update(name="../escape.json"),
            "not a fixture file name",
            id="fixture name that escapes the corpus",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["fixture"].update(bytes=True),
            "not a positive integer",
            id="boolean byte count",
        ),
        pytest.param(
            lambda doc: doc["cases"][0].update(expected=None),
            "not a JSON object",
            id="no expected outcome at all",
        ),
        pytest.param(
            lambda doc: _failing_case(doc)["observed"].update(outcome="pass"),
            "failing result names them",
            id="observed outcome contradicting its own failures",
        ),
        pytest.param(
            lambda doc: _failing_case(doc)["observed"].__setitem__(
                "failed_gates", sorted(_failing_case(doc)["observed"]["failed_gates"] * 2)
            ),
            "more than once",
            id="duplicate observed failures",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["quality"].update(passed=False),
            "failing gate(s)",
            id="forged quality verdict on a passing case",
        ),
        pytest.param(
            lambda doc: _failing_case(doc)["quality"].update(passed=True),
            "failing gate(s)",
            id="forged quality verdict on a failing case",
        ),
        pytest.param(
            lambda doc: _failing_case(doc)["quality"]["gates"][-1].update(passed=True),
            "does not follow from the values",
            id="forged gate verdict",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["quality"].update(
                gate_count=doc["cases"][0]["quality"]["gate_count"] + 1
            ),
            "the count is what a reader uses",
            id="forged gate count",
        ),
        pytest.param(
            lambda doc: _failing_case(doc)["quality"].update(failures=[]),
            "not exactly the gates it reports as failing",
            id="failures the gates contradict",
        ),
        pytest.param(
            lambda doc: doc["cases"][0]["quality"].update(case="somebody-else"),
            "calls itself",
            id="a quality report belonging to another case",
        ),
        pytest.param(_forge_mismatch_prose, "expected versus observed gives", id="invented prose"),
    ],
)
def test_a_forged_report_is_refused_however_consistent_it_looks(forge, expected):
    """Review's own forgeries, each one refused by the reader.

    Every document below is a REAL report from the shipped corpus with one
    nested claim changed, and every one of them is internally plausible: the
    totals add up, the decision follows the totals, and nothing is missing.
    They are refused because the reader recomputes what the report carries the
    evidence for rather than believing what the report says about itself.
    """
    document = json.loads(harness.to_json(harness.run()))
    forge(document)

    with pytest.raises(harness.ReportUnusable) as raised:
        harness.validate_report(document)
    assert expected in str(raised.value)


def test_a_malformed_entry_carrying_evidence_is_refused(tmp_path):
    """A case that could not be read produced no measurements to carry."""
    root = _copy_corpus(tmp_path)
    _reseal(root, "clean-panel.json", lambda case: case.update(kind="not-a-case"))
    document = json.loads(harness.to_json(harness.run(root)))
    entry = next(case for case in document["cases"] if case["case_id"] == "clean-panel")
    assert entry["status"] == "malformed"
    forged = json.loads(harness.to_json(harness.run()))
    entry["quality"] = _cases_by_id(forged)["clean-panel"]["quality"]

    with pytest.raises(harness.ReportUnusable) as raised:
        harness.validate_report(document)
    assert "still carries" in str(raised.value)


# --- provenance: measurement schemas and gate bindings -------------------------------


def _report() -> dict[str, Any]:
    """A real report from the shipped corpus, as plain JSON to forge on."""
    return json.loads(harness.to_json(harness.run()))


def _case(document: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    """One case of a decoded report, as the mutable JSON it is."""
    entry = _cases_by_id(document)[case_id]
    assert isinstance(entry, dict)
    return entry


def _measurements(document: Mapping[str, Any], case_id: str) -> list[dict[str, Any]]:
    measurements = _case(document, case_id)["quality"]["measurements"]
    assert isinstance(measurements, list)
    return measurements


def _set_measured(document: Mapping[str, Any], case_id: str, gate: str, value: object) -> None:
    """Set one gate's ``measured`` everywhere the case repeats it.

    ``failures`` carries its own copy of each failing gate, so a forgery that
    changed only ``gates`` would be caught by the round-2 failures check
    instead of the binding this control is about.
    """
    report = _case(document, case_id)["quality"]
    assert isinstance(report, Mapping)
    for group in ("gates", "failures"):
        for entry in report[group]:
            if entry["name"] == gate:
                entry["measured"] = value


def test_every_gate_names_the_measurement_and_path_its_value_came_from():
    """The provenance a ``scanner3d.quality.Gate`` does not itself record.

    One binding per gate, in the gates' own order, and the value at the named
    path IS the value the gate compared. Checked here against the live report
    rather than only through the reader, so the harness cannot satisfy its own
    validator by writing a binding that resolves to whatever it wrote.
    """
    document = _report()

    for case in document["cases"]:
        assert case["status"] != "malformed"
        bindings, gates = case["gate_bindings"], case["quality"]["gates"]
        assert [binding["gate"] for binding in bindings] == [gate["name"] for gate in gates]
        by_name = {entry["name"]: entry for entry in case["quality"]["measurements"]}
        for binding, gate in zip(bindings, gates, strict=True):
            resolved: Any = by_name[binding["measurement"]]
            for step in binding["path"]:
                resolved = resolved[step]
            assert resolved == gate["measured"]
            assert isinstance(resolved, bool) == isinstance(gate["measured"], bool)


def test_the_shipped_corpus_exercises_every_measurement_kind_the_reader_knows():
    """A schema nothing emits is a schema nothing proves.

    The reverse direction matters more: a kind the corpus emits and this
    reader has no schema for would be a measurement no control ever checked.
    """
    document = _report()
    kinds = {
        measurement["kind"]
        for case in document["cases"]
        for measurement in case["quality"]["measurements"]
    }

    assert kinds == set(harness.MEASUREMENT_KINDS)


def _bend_coverage_fraction(document: dict[str, Any]) -> None:
    """A coverage fraction moved to a value no count over its queries gives.

    The gate beside it is moved with it, so the binding agrees and the gate's
    own verdict is unchanged. What refuses this is the measurement's own
    arithmetic: coverage is a whole number of query points over the query
    count, and 0.7000001 of them is not a whole number of anything.
    """
    _measurements(document, "missing-surface")[0]["reference_to_observed"]["coverage_fraction"] = (
        0.7000001
    )
    _set_measured(document, "missing-surface", "cloud.reference_to_observed.coverage", 0.7000001)


@pytest.mark.parametrize(
    ("forge", "expected"),
    [
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0].update(invented="accepted?"),
            "unknown ['invented']",
            id="an unknown field added to a measurement",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0].pop("region"),
            "missing ['region']",
            id="a required measurement field removed",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0].pop("limitation"),
            "missing ['limitation']",
            id="the plane limitation removed",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0].update(kind="invented_kind"),
            "refuses one it has no schema for",
            id="a measurement kind with no schema",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0].update(schema_version=99),
            "states schema version",
            id="a measurement from another quality schema",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0]["residuals"].update(count=2),
            "beside the 144 the measurement around it reports having",
            id="a distribution summarising a different population",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0]["residuals"].update(median=1.0),
            "which no set of magnitudes produces",
            id="a median above the p95 beside it",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0]["residuals"].update(rms=1.0),
            "cannot exceed the largest magnitude",
            id="a root mean square above the worst value",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0]["residuals"].update(worst=-1.0),
            "not a magnitude",
            id="a negative magnitude",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0]["residuals"].update(unit="mm"),
            "and its measurement is in",
            id="a distribution in another unit than its measurement",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0].update(normal=[1.0, 1.0, 1.0]),
            "is a unit vector",
            id="a plane normal that is not one",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0].update(signed_max=99.0),
            "the residuals are the magnitudes of the signed values",
            id="signed extremes the residuals contradict",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "clean-panel")[0].update(limitation="none, honestly"),
            "own plane limitation",
            id="a roughness measurement without the authority's limitation",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "missing-surface")[0]["observed_to_reference"].update(
                query_count=3
            ),
            "beside the 3 the measurement around it reports having",
            id="a query count the distances contradict",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "missing-surface")[0]["reference_to_observed"].update(
                target_count=7
            ),
            "one pair of clouds has one size",
            id="two directions disagreeing on the cloud sizes",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "missing-surface")[0]["reference_to_observed"].update(
                coverage_tolerance=0.5
            ),
            "one measurement is made at one tolerance",
            id="two coverage tolerances in one measurement",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "missing-surface")[0]["observed_to_reference"].update(
                coverage_tolerance=0.0
            ),
            "the authority refuses a non-positive one",
            id="a coverage tolerance the authority would refuse",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "missing-surface")[0].update(search_gate=0.001),
            "outside the 0.001 search gate",
            id="a search gate the distances fall outside",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "missing-surface")[0]["observed_to_reference"].update(
                direction="reference_to_observed"
            ),
            "and is carried as",
            id="a direction carried under the other name",
        ),
        pytest.param(_bend_coverage_fraction, "not a whole number of them", id="coverage fraction"),
        pytest.param(
            lambda doc: _measurements(doc, "uniform-scale")[0]["entries"][0].update(truth=1.0),
            "states error",
            id="a dimension error its measured and truth contradict",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "uniform-scale")[0]["entries"][0].update(
                absolute_error=0.0
            ),
            "states absolute error",
            id="an absolute error that is not the magnitude of its error",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "uniform-scale")[0]["entries"][0].update(
                relative_error=0.5
            ),
            "states relative error",
            id="a relative error its own error and truth contradict",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "uniform-scale")[0]["entries"].append(
                dict(_measurements(doc, "uniform-scale")[0]["entries"][0])
            ),
            "names an entry more than once",
            id="a duplicated named dimension",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "uniform-scale")[0].update(relative=None),
            "summarises none of them",
            id="every relative error present and none summarised",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "uniform-scale")[1]["entries"][0].update(offset=0.25),
            "which are",
            id="a landmark offset its two points contradict",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "uniform-scale")[1]["entries"][0].update(
                measured=[0.0, 0.0]
            ),
            "not three coordinates",
            id="a landmark that is not a point",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "readiness-valid-orbit")[0].update(
                summary="readiness: ready, honestly"
            ),
            "summarises itself as",
            id="a readiness summary its own verdict contradicts",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "readiness-valid-orbit")[0].update(read_by="me"),
            "says it was read by",
            id="a readiness record no authority read",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "readiness-valid-orbit")[0]["reason_present"].update(
                tracking_lost=True
            ),
            "'reason_present' says",
            id="a reason flag the reasons contradict",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "readiness-surface-displaced")[0].update(reason_count=9),
            "reason(s) and carries",
            id="a reason count the reasons contradict",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "readiness-surface-displaced")[0].update(
                reasons=["invented_reason"]
            ),
            "which scanner3d.reconstruction_readiness does not define",
            id="a reason code the authority does not define",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "readiness-surface-displaced")[0].update(ready=True),
            "readiness is exactly 'no gate failed'",
            id="a readiness verdict its own reasons contradict",
        ),
        pytest.param(
            lambda doc: _measurements(doc, "readiness-valid-orbit")[0].update(surface_measured=1),
            "neither a boolean nor null",
            id="a boolean-as-number in a readiness record",
        ),
    ],
)
def test_a_forged_measurement_is_refused(forge, expected):
    """Every field of every kind the harness emits, held to its own schema.

    Round 2 checked a measurement's ``kind`` and ``name`` and nothing else, so
    an unknown field, a missing one and a changed number inside one were all
    accepted. Each forgery below is a real report with one such change.
    """
    document = _report()
    forge(document)

    with pytest.raises(harness.ReportUnusable) as raised:
        harness.validate_report(document)
    assert expected in str(raised.value)


def _bindings(document: Mapping[str, Any], case_id: str = "clean-panel") -> list[dict[str, Any]]:
    bindings = _case(document, case_id)["gate_bindings"]
    assert isinstance(bindings, list)
    return bindings


@pytest.mark.parametrize(
    ("forge", "expected"),
    [
        pytest.param(
            # Only the p95, and to a value that keeps every invariant inside
            # the measurement: still between the median and the worst, still
            # under the rms bound. The gate beside it is untouched, so the
            # binding is the only thing left that can see it.
            lambda doc: _measurements(doc, "clean-panel")[0]["residuals"].update(p95=6.0e-07),
            "A measurement changed beside the gate that read it",
            id="a measurement changed beside an unchanged gate",
        ),
        pytest.param(
            lambda doc: _set_measured(doc, "clean-panel", "panel.residual.p95", 0.0),
            "A measurement changed beside the gate that read it",
            id="a gate value its measurement does not report",
        ),
        pytest.param(
            lambda doc: _bindings(doc, "readiness-valid-orbit")[0].update(
                path=["reason_present", "tracking_lost"]
            ),
            "and measurement 'readiness' reports",
            id="a boolean gate bound to a boolean that is not its own",
        ),
        pytest.param(
            lambda doc: _case(doc, "clean-panel").update(gate_bindings=None),
            "not a list of gate bindings",
            id="a case carrying no bindings at all",
        ),
        pytest.param(
            lambda doc: _bindings(doc).pop(),
            "unbound",
            id="a gate with no binding",
        ),
        pytest.param(
            lambda doc: _bindings(doc).append(
                {"gate": "invented", "measurement": "panel", "path": ["residuals", "worst"]}
            ),
            "which this case does not report",
            id="a binding for a gate that does not exist",
        ),
        pytest.param(
            lambda doc: _bindings(doc).append(dict(_bindings(doc)[0])),
            "binds a gate more than once",
            id="one gate bound twice",
        ),
        pytest.param(
            lambda doc: _bindings(doc).reverse(),
            "in the gates' own order",
            id="bindings out of the gates' order",
        ),
        pytest.param(
            lambda doc: _bindings(doc)[0].update(measurement="invented"),
            "which this case does not report",
            id="a binding to a measurement that is not there",
        ),
        pytest.param(
            lambda doc: _bindings(doc)[0].update(path=[]),
            "a binding to nothing proves nothing",
            id="an empty path",
        ),
        pytest.param(
            lambda doc: _bindings(doc)[0].update(path=["..", "worst"]),
            "not a plain measurement field name",
            id="a path that is not a field name",
        ),
        pytest.param(
            lambda doc: _bindings(doc)[0].update(path=["residuals", "invented"]),
            "has no 'invented' there",
            id="a path into a field the measurement does not have",
        ),
        pytest.param(
            lambda doc: _bindings(doc)[0].update(path=["residuals"]),
            "which is not a number",
            id="a path resolving to a block rather than a value",
        ),
        pytest.param(
            lambda doc: _bindings(doc, "readiness-valid-orbit")[0].update(path=["summary"]),
            "a boolean and a number are not the same kind of thing",
            id="a boolean gate bound to a string",
        ),
        pytest.param(
            lambda doc: _bindings(doc)[0].pop("path"),
            "missing ['path']",
            id="a binding with no path key",
        ),
        pytest.param(
            lambda doc: _bindings(doc)[0].update(invented="accepted?"),
            "unknown ['invented']",
            id="an unknown field on a binding",
        ),
    ],
)
def test_a_forged_gate_binding_is_refused(forge, expected):
    """The gate's duplicated value, held to the measurement it came from.

    ``scanner3d.quality.Gate`` records a measured value and nothing about
    where it came from, so before these bindings existed a measurement could
    be changed while its gate was left alone and the report still validated.
    """
    document = _report()
    forge(document)

    with pytest.raises(harness.ReportUnusable) as raised:
        harness.validate_report(document)
    assert expected in str(raised.value)


def test_a_malformed_case_carries_no_gate_bindings_either(tmp_path):
    """A fixture that could not be read bound nothing, and may claim nothing."""
    root = _copy_corpus(tmp_path)
    _reseal(root, "clean-panel.json", lambda case: case.update(kind="not-a-case"))
    document = json.loads(harness.to_json(harness.run(root)))
    entry = _case(document, "clean-panel")
    assert entry["status"] == "malformed"
    assert entry["gate_bindings"] is None
    entry["gate_bindings"] = [
        {"gate": "invented", "measurement": "panel", "path": ["residuals", "worst"]}
    ]

    with pytest.raises(harness.ReportUnusable) as raised:
        harness.validate_report(document)
    assert "still carries" in str(raised.value)


def test_the_report_schema_version_moved_with_this_shape_change():
    """Version 2 is the shape with bindings and exact measurement schemas.

    A version-1 reader accepts a measurement this one refuses, so a consumer
    that guessed at a neighbouring version would be reading a weaker promise
    than the number in front of it.
    """
    assert harness.REPORT_SCHEMA_VERSION == 2

    document = _report()
    document["schema_version"] = 1

    with pytest.raises(harness.ReportUnusable) as raised:
        harness.validate_report(document)
    assert "states schema version 1" in str(raised.value)


def test_a_report_is_read_back_from_its_own_bytes_through_the_strict_reader(tmp_path):
    """The round trip a consumer actually performs: path in, checked report out."""
    out = tmp_path / "report.json"
    assert harness.main(["--out", str(out), "--quiet"]) == 0

    document = harness.read_report(out)

    assert document["kind"] == harness.REPORT_KIND
    assert document["decision"] == "pass"
    assert harness.loads_report(out.read_text(encoding="utf-8")) == document


def test_the_text_reader_refuses_a_duplicate_key_no_mapping_could_show(tmp_path):
    """The reason there is a text entry point at all.

    ``json.loads`` keeps the LAST of two identical keys and discards the first,
    so by the time a mapping validator sees the document, the report that said
    both ``"decision": "fail"`` and ``"decision": "pass"`` is indistinguishable
    from one that only ever said ``"pass"``. Only a reader that sees the text
    can refuse it.
    """
    out = tmp_path / "report.json"
    harness.main(["--out", str(out), "--quiet"])
    text = out.read_text(encoding="utf-8")
    forged = text.replace('"decision": "pass"', '"decision": "fail",\n  "decision": "pass"', 1)

    assert json.loads(forged)["decision"] == "pass"
    with pytest.raises(harness.ReportUnusable) as raised:
        harness.loads_report(forged)
    assert "twice in one object" in str(raised.value)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_the_text_reader_refuses_a_non_finite_constant(tmp_path, token):
    """``NaN`` is not JSON, and Python's decoder accepts it anyway."""
    out = tmp_path / "report.json"
    harness.main(["--out", str(out), "--quiet"])
    text = out.read_text(encoding="utf-8")
    forged = text.replace('"case_count": 10', f'"case_count": {token}', 1)
    assert forged != text

    with pytest.raises(harness.ReportUnusable) as raised:
        harness.loads_report(forged)
    assert "not JSON" in str(raised.value)


def test_the_file_reader_refuses_bytes_that_are_not_a_report(tmp_path):
    """A path that holds something else is a refusal, not a traceback."""
    out = tmp_path / "report.json"
    out.write_bytes(b"\xff\xfe not utf-8")
    with pytest.raises(harness.ReportUnusable):
        harness.read_report(out)

    out.write_text("[]", encoding="utf-8")
    with pytest.raises(harness.ReportUnusable):
        harness.read_report(out)

    with pytest.raises(harness.ReportUnusable):
        harness.read_report(tmp_path / "absent.json")


# --- the requested output path never keeps an older answer ---------------------------


def _stale(path: Path) -> None:
    """A previous run's REAL report, left at the path the next run will ask for."""
    assert harness.main(["--out", str(path), "--quiet"]) == 0
    assert harness.read_report(path)["decision"] == "pass"


def test_a_corpus_that_cannot_be_run_leaves_no_earlier_report_behind(tmp_path):
    """Exit 2 with a plausible pass still sitting at --out is the worst outcome.

    Nothing in that file says it is not this invocation's answer: it carries a
    real manifest digest, real cases and a real decision.
    """
    out = tmp_path / "report.json"
    _stale(out)

    assert harness.main(["--corpus", str(tmp_path / "absent"), "--out", str(out), "--quiet"]) == 2

    assert not out.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == []


def test_a_report_that_cannot_be_serialised_leaves_no_earlier_report_behind(tmp_path):
    """The destination is cleared before the document is serialised, not after."""
    out = tmp_path / "report.json"
    _stale(out)
    document = dict(harness.run())
    document["decision"] = math.inf

    with pytest.raises(ValueError):
        harness.publish(document, out)

    assert not out.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == []


def test_a_report_that_cannot_be_written_leaves_no_earlier_report_behind(tmp_path, monkeypatch):
    """A failed rename is the last way a stale report could survive."""
    out = tmp_path / "report.json"
    _stale(out)

    def _refuse(self: Path, target: object) -> None:
        raise OSError("the rename failed")

    monkeypatch.setattr(Path, "replace", _refuse)

    with pytest.raises(harness.HarnessRefused):
        harness.publish(harness.run(), out)

    assert not out.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == []


def test_an_unsafe_output_path_is_refused_rather_than_removed(tmp_path):
    """Refused, not deleted: neither of these is this harness's artifact."""
    directory = tmp_path / "report.json"
    directory.mkdir()
    assert harness.main(["--out", str(directory), "--quiet"]) == 2
    assert directory.is_dir()

    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text("keep me", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(elsewhere)
    assert harness.main(["--out", str(link), "--quiet"]) == 2
    assert link.is_symlink()
    assert elsewhere.read_text(encoding="utf-8") == "keep me"


def test_a_successful_run_replaces_an_earlier_report_and_leaves_no_staging_file(tmp_path):
    """The ordinary path, asserted beside the failing ones."""
    out = tmp_path / "report.json"
    _stale(out)

    assert harness.main(["--out", str(out), "--quiet"]) == 0

    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.json"]
    assert harness.read_report(out)["kind"] == harness.REPORT_KIND
    assert not harness.staging_path(out).exists()


# --- an authority that raises ---------------------------------------------------------


def test_a_measurement_that_raises_is_one_malformed_case_and_the_run_continues(
    tmp_path, monkeypatch
):
    """The promise the command makes about independent cases, under an exception.

    ``plane_roughness`` is made to raise an ordinary ``Exception`` - not a
    ``BaseException``, which would be the operator interrupting the run rather
    than a case failing. The two cases that measure a plane are recorded as
    malformed, every later case still runs and is still in the artifact, the
    strict report is published, and the command exits 1.
    """

    def _raise(*args: object, **kwargs: object) -> None:
        raise RuntimeError("the measurement authority fell over")

    monkeypatch.setattr(quality, "plane_roughness", _raise)
    out = tmp_path / "report.json"

    assert harness.main(["--corpus", str(CASES), "--out", str(out), "--quiet"]) == 1

    document = harness.read_report(out)
    cases = _cases_by_id(document)
    for broken in ("clean-panel", "lumpy-panel"):
        assert cases[broken]["status"] == "malformed"
        assert "RuntimeError: the measurement authority fell over" in str(cases[broken]["error"])
        assert cases[broken]["quality"] is None
    # Every later case - the corpus runs in sorted file order, and these all
    # come after clean-panel.json - is still measured and still reported.
    for later in ("missing-surface", "readiness-valid-orbit", "spurious-geometry", "uniform-scale"):
        assert cases[later]["status"] == "pass"
    totals = document["totals"]
    assert isinstance(totals, Mapping)
    assert totals["malformed"] == 2
    assert totals["cases"] == len(cases)
    assert document["decision"] == "fail"


def test_an_interrupt_while_measuring_is_not_recorded_as_a_malformed_fixture(monkeypatch):
    """A ``KeyboardInterrupt`` ends the run; it does not become evidence."""

    def _interrupt(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(quality, "plane_roughness", _interrupt)

    with pytest.raises(KeyboardInterrupt):
        harness.run()


def test_the_rendered_log_shows_every_gate_and_not_only_the_failures():
    """A log of failures alone makes one failed gate look like one gate.

    The passing gates are what say which failure mode was ruled out - the
    missing-surface case's forward direction being perfect is the whole
    finding.
    """
    text = harness.render(harness.run())

    assert "PASS cloud.observed_to_reference.p95" in text
    assert "FAIL cloud.reference_to_observed.worst" in text
    assert "MISMATCH" not in text
    for limitation in harness.LIMITATIONS:
        assert limitation in text


def test_the_command_exits_one_when_a_case_does_not_match_and_still_writes_its_report(tmp_path):
    """The failing run is the one whose artifact matters.

    Exit 1 with a complete report is what lets CI upload the evidence on the
    run that failed - see the ``harness`` job's ``if: always()``.
    """
    root = _copy_corpus(tmp_path)
    _reseal(
        root,
        "clean-surface.json",
        lambda case: case["expectation"].update(
            outcome="fail", failed_gates=["cloud.observed_to_reference.p95"]
        ),
    )
    out = tmp_path / "report.json"

    assert harness.main(["--corpus", str(root), "--out", str(out), "--quiet"]) == 1

    document = json.loads(out.read_text(encoding="utf-8"))
    assert document["decision"] == "fail"
    assert harness.validate_report(document) is not None
    assert len(document["cases"]) == document["totals"]["cases"]


# --- the readiness boundary ---------------------------------------------------------


def test_the_readiness_cases_are_read_by_the_real_strict_reader():
    """Not a re-implementation of readiness, and not a hand-written record.

    Each shipped record is handed to
    ``ReconstructionReadiness.from_dict`` here, directly, and has to produce
    the same verdict the harness reports for it. If the two ever disagreed,
    the harness would be scoring readiness by a rule of its own.
    """
    expected = {
        "readiness-valid-orbit": (True, ()),
        "readiness-poses-stationary": (
            False,
            (
                POSE_VIEWPOINTS_INSUFFICIENT,
                POSE_EXCURSION_INSUFFICIENT,
                POSE_MOTION_BELOW_NOISE,
                COVERAGE_UNSUPPORTED,
            ),
        ),
        "readiness-coverage-one-sided": (False, (COVERAGE_UNSUPPORTED,)),
        "readiness-surface-displaced": (False, (SURFACE_UNSUPPORTED,)),
    }
    reported = _cases_by_id(harness.run())

    for case_id, (ready, reasons) in expected.items():
        case = json.loads((CASES / f"{case_id}.json").read_text(encoding="utf-8"))
        readiness = ReconstructionReadiness.from_dict(case["measurements"][0]["record"])
        assert readiness.ready is ready, case_id
        assert readiness.reasons == reasons, case_id
        assert reported[case_id]["status"] == "pass", case_id
        report = reported[case_id]["quality"]
        assert isinstance(report, Mapping)
        measurements = report["measurements"]
        assert isinstance(measurements, list)
        assert measurements[0]["ready"] is ready
        assert measurements[0]["reasons"] == list(reasons)


def test_the_frozen_records_are_what_todays_evaluator_still_produces(tmp_path):
    """The link between the shipped evidence and the live product code.

    The records are frozen output of the real
    ``evaluate_readiness``/``surface_evidence``. Re-running the evaluator over
    the same synthetic sweep here is what stops them becoming a museum of a
    decision the product no longer makes - a frozen record that today's
    evaluator would not produce is a fixture that has quietly stopped testing
    anything.

    This is the smallest real boundary that preserves the contract. A full
    export fixture would add megabytes of raw frames and seconds of fusion to
    every CI run to re-prove what ``tests/test_reconstruction_export_readiness.py``
    already proves about the export path; what is measured here is that the
    gates and their reason codes still fire on the same evidence.
    """
    clouds, poses = builder.readiness_sweep(builder._orbit_angles())
    live = evaluate_readiness(tmp_path, clouds, poses)
    frozen = ReconstructionReadiness.from_dict(
        json.loads((CASES / "readiness-valid-orbit.json").read_text(encoding="utf-8"))[
            "measurements"
        ][0]["record"]
    )

    assert live.ready is True
    assert live.reasons == ()
    assert live.poses.as_dict() == frozen.poses.as_dict()
    assert live.coverage.as_dict() == frozen.coverage.as_dict()
    assert live.policy.as_dict() == frozen.policy.as_dict()

    still_clouds, still_poses = builder.readiness_sweep([0.0] * builder.READINESS_FRAMES)
    stationary = evaluate_readiness(tmp_path, still_clouds, still_poses)
    stored = ReconstructionReadiness.from_dict(
        json.loads((CASES / "readiness-poses-stationary.json").read_text(encoding="utf-8"))[
            "measurements"
        ][0]["record"]
    )
    assert stationary.reasons == stored.reasons


def test_a_readiness_record_whose_verdict_contradicts_its_evidence_is_refused(tmp_path):
    """The one shape the readiness module exists to refuse, reaching the harness.

    A record claiming it is ready beside evidence that failed is not a case
    that passes and not a case that fails - it is a fixture that cannot be
    read, and the harness reports it as malformed with the reader's own words.
    """
    root = _copy_corpus(tmp_path)

    def claim_ready(case: dict[str, Any]) -> None:
        case["measurements"][0]["record"]["ready"] = True
        case["measurements"][0]["record"]["failures"] = []

    _reseal(root, "readiness-coverage-one-sided.json", claim_ready)
    cases = _cases_by_id(harness.run(root))

    entry = cases["readiness-coverage-one-sided"]
    assert entry["status"] == "malformed"
    assert "readiness reader refused this record" in str(entry["error"])

    # And the same record, refused by that reader directly, so the message in
    # the artifact is the authority's and not this harness's paraphrase.
    case = json.loads((root / "readiness-coverage-one-sided.json").read_text(encoding="utf-8"))
    with pytest.raises(ReadinessRecordUnusable):
        ReconstructionReadiness.from_dict(case["measurements"][0]["record"])


def test_a_readiness_case_that_stops_being_refused_fails_the_harness(tmp_path):
    """Reason codes are named exactly, so the wrong refusal is not a pass.

    Swapping which code a record is expected to carry has to fail even though
    the record is still refused - "refused for something" is not the finding
    any of these fixtures were written for.
    """
    root = _copy_corpus(tmp_path)

    def rename_reason(case: dict[str, Any]) -> None:
        for gate in case["gates"]:
            if gate["path"] == ["reason_present", COVERAGE_UNSUPPORTED]:
                gate["path"] = ["reason_present", POSE_VIEWPOINTS_INSUFFICIENT]
                gate["name"] = f"readiness.reason.{POSE_VIEWPOINTS_INSUFFICIENT}"
            elif gate["path"] == ["reason_present", POSE_VIEWPOINTS_INSUFFICIENT]:
                gate["path"] = ["reason_present", COVERAGE_UNSUPPORTED]
                gate["name"] = f"readiness.not_reason.{COVERAGE_UNSUPPORTED}"

    _reseal(root, "readiness-coverage-one-sided.json", rename_reason)
    cases = _cases_by_id(harness.run(root))

    entry = cases["readiness-coverage-one-sided"]
    assert entry["status"] == "fail"
    assert entry["mismatches"]


def test_the_corpus_gates_only_reason_codes_the_authority_defines():
    """Every readiness gate names a code from the authority's own list.

    A fixture gating on a code the module does not define would be asserting
    against a name that can never be true, which is a gate that cannot fail.
    """
    for path in sorted(CASES.glob("readiness-*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        for gate in case["gates"]:
            if gate["path"][0] == "reason_present":
                assert gate["path"][1] in REASONS, gate["name"]


def test_every_defined_reason_code_is_gated_by_at_least_one_case():
    """Nothing in the authority's list is silently ungoverned here.

    Not every code can be produced by a synthetic fixture - the tracking and
    not-measurable codes need a capture or a degenerate input this corpus does
    not carry - so the assertion is that every code is at least NAMED, present
    or absent, in every readiness case.
    """
    for path in sorted(CASES.glob("readiness-*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        named = {gate["path"][1] for gate in case["gates"] if gate["path"][0] == "reason_present"}
        assert named == set(REASONS), path.name


# --- the generator ------------------------------------------------------------------


def test_the_generator_reproduces_the_shipped_corpus_exactly(tmp_path):
    """The committed data is what the documented generator produces.

    Regenerating into a temporary directory rather than over the corpus: this
    is a check that the evidence and its recipe agree, not a step that quietly
    rewrites the evidence to make itself pass.
    """
    root = tmp_path / "cases"
    builder.write(root)

    produced = {path.name: path.read_bytes() for path in sorted(root.iterdir())}
    shipped = {path.name: path.read_bytes() for path in sorted(CASES.iterdir()) if path.is_file()}

    assert produced == shipped


def test_the_generator_is_not_collected_as_a_test_and_writes_nothing_on_import():
    """It writes files, so it must never run as part of the suite.

    ``build`` is pure and ``write`` is the only thing that touches the disk.
    pytest collects ``test_*.py``; the name is what keeps this out, and the
    separation is what makes that safe.
    """
    assert not Path(builder.__file__).name.startswith("test_")
    with tempfile.TemporaryDirectory() as directory:
        builder.build()
        assert not sorted(Path(directory).iterdir())
