"""What the surface-oracle corpus reader refuses, and why each refusal is the intended one.

The harness has two readers and both are attacked here. :func:`load_corpus`
and :func:`load_case` read what is on disk, and everything they accept becomes
evidence; :func:`validate_report` reads a published artifact back, and
everything it accepts becomes a claim somebody may quote. An oracle whose
reader can be talked into accepting a swapped baseline, an omitted view, a
forged label, an altered truth, a cherry-picked grid cell, an impossible
improvement, a missing bad control or a candidate-authored threshold is not an
oracle, so each of those is written out here as a corruption and each has to
fail for its own stated reason.

The corpus is run ONCE for this module, because the artifact every reader test
mutates has to be a real one - a hand-written document would only prove the
reader rejects hand-written documents.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests import surface_oracle as contract
from tests import surface_oracle_harness as harness
from tests.quality_harness import FixtureUnusable, HarnessRefused, ReportUnusable

pytestmark = pytest.mark.filterwarnings("error")


@pytest.fixture(scope="module")
def corpus() -> harness.Corpus:
    return harness.load_corpus()


@pytest.fixture(scope="module")
def cases(corpus: harness.Corpus) -> list[harness.CaseSpec]:
    return [
        harness.load_case(harness.CORPUS / entry.name, entry=entry, corpus=corpus)
        for entry in corpus.files
    ]


@pytest.fixture(scope="module")
def report() -> Any:
    """One real run of the committed corpus. Every reader test mutates a copy of it.

    Typed ``Any`` on purpose: this is decoded JSON, and every test below walks
    into it by key. Spelling each step's narrowing out would say nothing about
    the harness and a great deal about mypy.
    """
    return harness.run()


@pytest.fixture
def document(report: Any) -> Any:
    return copy.deepcopy(dict(report))


def _refused(document: object) -> str:
    with pytest.raises(ReportUnusable) as caught:
        harness.validate_report(document)
    return str(caught.value)


def _copy_corpus(tmp_path: Path) -> Path:
    """The committed corpus, byte for byte, in a directory a test may edit.

    The fixture builder owns ``tests/fixtures/surface_oracle/``; nothing here
    writes there. Corruptions are made on this copy.
    """
    root = tmp_path / "cases"
    root.mkdir()
    for path in sorted(harness.CORPUS.glob("*.json")):
        (root / path.name).write_bytes(path.read_bytes())
    return root


def _rewrite(root: Path, name: str, mutate: object) -> None:
    """Edit one case file on the COPY and re-hash it into the copy's manifest."""
    path = root / name
    document = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutate)
    mutate(document)
    text = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    manifest_path = root / harness.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        if entry["name"] == name:
            entry["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
            entry["bytes"] = len(text.encode("utf-8"))
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


# --- the run itself ---------------------------------------------------------------------


def test_the_committed_corpus_runs_and_every_arm_does_what_its_fixture_declares(
    report: Any,
) -> None:
    totals = report["totals"]
    assert isinstance(totals, Mapping)
    assert totals["cases"] == totals["passed"]
    assert totals["candidates"] == totals["candidates_passed"]
    assert report["decision"] == "pass"


def test_the_run_is_deterministic(report: Any) -> None:
    again = harness.run()
    assert json.dumps(again, sort_keys=True) == json.dumps(dict(report), sort_keys=True)


def test_the_corpus_carries_a_case_for_every_declared_perturbation(
    report: Any,
) -> None:
    cases = report["cases"]
    assert isinstance(cases, list)
    labels = {case["scene"]["perturbation"]["label"] for case in cases}
    assert labels == set(contract.PERTURBATIONS)


def test_the_exact_control_proves_the_dots_stay_good(report: Any) -> None:
    """Requirement: one known-good case stays good, measured rather than asserted."""
    cases = report["cases"]
    assert isinstance(cases, list)
    exact = next(case for case in cases if case["case_id"] == "exact-control")
    dots = next(item for item in exact["candidates"] if item["candidate"] == "raw_points")
    assert dots["status"] == "pass"
    assert dots["expected"]["failed_gates"] == []
    assert dots["observed"]["failed_gates"] == []


def test_every_case_reports_a_recommended_cell_and_it_is_a_declared_one(
    report: Any,
) -> None:
    grid = {cell["cell_id"] for cell in report["corpus"]["grid"]}
    for case in report["cases"]:
        assert case["recommended_cell"] in grid


def test_the_artifact_publishes_its_own_limitations(report: Any) -> None:
    limitations = report["limitations"]
    assert isinstance(limitations, list)
    assert any("Synthetic truth" in str(item) for item in limitations)
    assert any("physical length" in str(item) for item in limitations)
    # Round three's two, stated rather than left for a reader to infer.
    assert any("not interchangeable" in str(item) for item in limitations)
    assert any("barycentric lattice" in str(item) for item in limitations)


# --- the resolution an owner actually gets -------------------------------------------------


def test_every_case_publishes_the_resolution_the_shipped_policy_derived_for_it(
    report: Any,
) -> None:
    """The review's scope correction: the grid is an envelope, this is the appliance."""
    finest = min(float(cell["voxel_size"]) for cell in report["corpus"]["grid"])
    for case in report["cases"]:
        appliance = case["scene"]["appliance"]
        derived = float(appliance["derived"]["voxel_size"])
        assert derived < finest
        assert appliance["cell"]["voxel_size"] == derived
        assert appliance["voxel_budget"] == report["corpus"]["appliance"]["voxel_budget"]
        names = {item["candidate"] for item in case["candidates"]}
        assert harness.APPLIANCE_ARM in names


def test_an_artifact_without_the_appliance_arm_is_refused(document: Any) -> None:
    case = document["cases"][0]
    case["candidates"] = [
        item for item in case["candidates"] if item["candidate"] != harness.APPLIANCE_ARM
    ]
    assert "published no 'fused_at_appliance_resolution'" in _refused(document)


def test_a_manifest_with_no_appliance_policy_is_refused(tmp_path: Path) -> None:
    root = _copy_corpus(tmp_path)
    path = root / harness.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["appliance"]
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(HarnessRefused, match="appliance"):
        harness.load_corpus(root)


def test_an_appliance_policy_that_drifted_from_the_product_is_refused(tmp_path: Path) -> None:
    """A declared constant that no longer matches the shipped one is a proxy wearing a name."""
    root = _copy_corpus(tmp_path)
    path = root / harness.MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["appliance"]["min_weight"] = 7.0
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(HarnessRefused, match="drifted"):
        harness.load_corpus(root)


def test_a_case_that_does_not_run_the_appliance_arm_is_refused(tmp_path: Path) -> None:
    root = _copy_corpus(tmp_path)

    def drop(document: dict[str, Any]) -> None:
        document["candidates"] = [
            item for item in document["candidates"] if item["name"] != harness.APPLIANCE_ARM
        ]

    _rewrite(root, "exact-control.json", drop)
    corpus = harness.load_corpus(root)
    entry = next(item for item in corpus.files if item.name == "exact-control.json")
    with pytest.raises(FixtureUnusable, match="fused_at_appliance_resolution"):
        harness.load_case(root / entry.name, entry=entry, corpus=corpus)


# --- the surface measured as a surface -----------------------------------------------------


def _measurement(document: Any, kind: str) -> Any:
    for case in document["cases"]:
        for candidate in case["candidates"]:
            report = candidate.get("quality")
            if not report:
                continue
            for measurement in report["measurements"]:
                if measurement["kind"] == kind:
                    return measurement
    pytest.fail(f"no {kind} measurement was published to corrupt")


def test_a_triangle_sampling_that_skipped_part_of_the_mesh_is_refused(document: Any) -> None:
    """Every emitted triangle is sampled, or the measurement chose what to look at."""
    record = _measurement(document, "surface_oracle_surface_samples")
    record["faces"] = int(record["faces"]) + 1
    assert "Every emitted triangle is sampled" in _refused(document)


def test_a_triangle_sampling_past_its_own_ceiling_is_refused(document: Any) -> None:
    record = _measurement(document, "surface_oracle_surface_samples")
    record["sample_ceiling"] = 1
    assert "against its own ceiling" in _refused(document)


def test_more_hidden_bridging_faces_than_bridging_faces_is_refused(document: Any) -> None:
    record = _measurement(document, "surface_oracle_surface_samples")
    record["bridging_faces_with_every_vertex_outside"] = int(record["bridging_faces"]) + 1
    assert "subset of the bridging faces" in _refused(document)


def test_a_topology_closure_flag_that_contradicts_its_own_counts_is_refused(
    document: Any,
) -> None:
    record = _measurement(document, "surface_oracle_topology")
    record["closed"] = not record["closed"]
    assert "not opinions about the mesh" in _refused(document)


def test_more_unsupported_shells_than_closed_shells_is_refused(document: Any) -> None:
    record = _measurement(document, "surface_oracle_topology")
    record["unsupported_closed_shells"] = int(record["closed_shells"]) + 1
    assert "subset of the one before it" in _refused(document)


def test_an_edge_census_that_does_not_add_up_is_refused(document: Any) -> None:
    record = _measurement(document, "surface_oracle_topology")
    record["edge_count"] = int(record["edge_count"]) + 1
    assert "used once, twice, or more than twice" in _refused(document)


def _closed_topology(document: Any) -> Any:
    """The first topology record that closed at least one shell, to split by hand."""
    for case in document["cases"]:
        for candidate in case["candidates"]:
            report = candidate.get("quality")
            if not report:
                continue
            for measurement in report["measurements"]:
                kind = measurement["kind"]
                if kind == "surface_oracle_topology" and measurement["closed_shells"]:
                    return measurement
    pytest.fail("no closed shell was published to split")


def _split(record: Any, *, shells: int, faces: int, false_shells: int, worst: float) -> None:
    """Overwrite one record's whole shell split, so the corruption is the only variable."""
    record["unsupported_closed_shells"] = shells
    record["unsupported_closed_shell_faces"] = faces
    record["unsupported_inaccurate_closed_shells"] = false_shells
    record["unsupported_inaccurate_closed_shell_faces"] = false_shells
    record["unsupported_inaccurate_worst_distance"] = worst if false_shells else 0.0
    record["unsupported_accurate_closed_shells"] = shells - false_shells
    record["unsupported_accurate_closed_shell_faces"] = shells - false_shells
    record["unsupported_accurate_worst_distance"] = worst if shells - false_shells else 0.0


def test_a_shell_split_that_loses_an_unsupported_shell_is_refused(document: Any) -> None:
    """The two groups are disjoint and exhaustive, or one of them is a selection."""
    record = _closed_topology(document)
    _split(record, shells=2, faces=2, false_shells=1, worst=0.4)
    record["unsupported_accurate_closed_shells"] = 0
    record["unsupported_accurate_closed_shell_faces"] = 0
    record["unsupported_accurate_worst_distance"] = 0.0
    assert "disjoint and exhaustive" in _refused(document)


def test_a_shell_split_whose_faces_do_not_add_up_is_refused(document: Any) -> None:
    record = _closed_topology(document)
    _split(record, shells=1, faces=2, false_shells=1, worst=0.4)
    assert "belongs to whichever group its shell does" in _refused(document)


def test_an_unsupported_shell_that_is_on_the_object_may_not_be_called_false(
    document: Any,
) -> None:
    """The blocking counterexample, written as a corruption the reader has to catch.

    Six unsupported closed shells in this corpus are analytically CORRECT. A
    report that filed one of them under the gated inaccurate group - and so
    presented correct geometry as surface to delete - is refused by its own
    numbers rather than believed.
    """
    record = _closed_topology(document)
    tolerance = float(record["accuracy_tolerance"])
    _split(record, shells=1, faces=1, false_shells=1, worst=tolerance / 2.0)
    assert "the case this split exists to protect" in _refused(document)


def test_a_shell_off_the_object_that_is_called_accurate_is_refused(document: Any) -> None:
    record = _closed_topology(document)
    tolerance = float(record["accuracy_tolerance"])
    _split(record, shells=1, faces=1, false_shells=0, worst=tolerance * 2.0)
    assert "outside its own" in _refused(document)


def test_a_shell_split_made_against_the_support_reference_is_refused(document: Any) -> None:
    """Splitting support by support is not a split, and the artifact has to say which."""
    record = _measurement(document, "surface_oracle_topology")
    record["shell_accuracy_source"] = contract.SUPPORT_SOURCE
    assert "would not be a split at all" in _refused(document)


def test_an_analytic_reference_described_as_something_else_is_refused(document: Any) -> None:
    """The whole value of the closed-form reference is that it is the closed-form one."""
    record = _measurement(document, "surface_oracle_analytic")
    record["truth_source"] = "the exact truth frames, deprojected"
    assert "not the closed-form one" in _refused(document)


def test_an_analytic_coverage_over_its_own_reference_is_refused(document: Any) -> None:
    record = _measurement(document, "surface_oracle_analytic")
    record["reference_covered_points"] = int(record["reference_points"]) + 1
    assert "reference point(s)" in _refused(document)


def test_an_ablation_delta_that_is_not_its_own_subtraction_is_refused(document: Any) -> None:
    record = _measurement(document, "surface_oracle_ablation")
    record["extraction_p95_delta"] = float(record["extraction_p95_delta"]) - 1.0
    assert "The ablation is subtraction" in _refused(document)


def test_the_ablation_scores_both_stages_against_the_same_truth(report: Any) -> None:
    """Fusion and extraction differ only in what they are; the reference is one."""
    seen = 0
    for case in report["cases"]:
        for candidate in case["candidates"]:
            quality = candidate.get("quality")
            if not quality:
                continue
            for measurement in quality["measurements"]:
                if measurement["kind"] != "surface_oracle_ablation":
                    continue
                seen += 1
                assert measurement["measured_against"] == contract.ANALYTIC_SOURCE
    assert seen > 0


# --- reading a published artifact back ----------------------------------------------------


def test_a_duplicated_key_does_not_survive_the_text_reader(
    report: Any,
) -> None:
    text = json.dumps(dict(report))
    doubled = text.replace('"decision":', '"decision": "pass", "decision":', 1)
    with pytest.raises(ReportUnusable, match=r"twice|duplicate"):
        harness.loads_report(doubled)


def test_a_report_of_another_kind_is_refused(document: Any) -> None:
    document["kind"] = "scanner3d.quality-report"
    assert "is not a" in _refused(document)


def test_a_widened_vocabulary_is_refused(document: Any) -> None:
    refusals = list(document["refusals"])
    document["refusals"] = [*refusals, "close_enough"]
    assert "vocabulary" in _refused(document)


def test_a_selection_rule_this_module_does_not_apply_is_refused(
    document: Any,
) -> None:
    corpus = document["corpus"]
    assert isinstance(corpus, dict)
    corpus["selection_rule"] = "the cell that looked best"
    assert "selection rule" in _refused(document)


def test_a_cherry_picked_grid_cell_is_refused(document: Any) -> None:
    """A case that published only the cell that suited it is a search, not an envelope."""
    case = document["cases"][0]
    case["candidates"] = [
        item for item in case["candidates"] if item["candidate"] != "fused:v25-t3-w1-uniform"
    ]
    assert "publishes no result for" in _refused(document)


def test_recommending_a_cell_the_corpus_never_declared_is_refused(
    document: Any,
) -> None:
    document["cases"][0]["recommended_cell"] = "v01-t1-w1-magic"
    assert "not a declared cell" in _refused(document)


def test_a_case_without_its_raw_points_arm_is_refused(document: Any) -> None:
    case = document["cases"][0]
    case["candidates"] = [item for item in case["candidates"] if item["candidate"] != "raw_points"]
    assert "no raw-points arm" in _refused(document)


def test_a_candidate_reported_twice_is_refused(document: Any) -> None:
    case = document["cases"][0]
    case["candidates"] = [*case["candidates"], copy.deepcopy(case["candidates"][0])]
    assert "twice" in _refused(document)


def test_a_case_reported_twice_is_refused(document: Any) -> None:
    cases = document["cases"]
    assert isinstance(cases, list)
    document["cases"] = [*cases, copy.deepcopy(cases[0])]
    assert "twice" in _refused(document)


def test_a_corpus_missing_its_bad_controls_is_refused(document: Any) -> None:
    """Delete every perturbed case and the artifact reports only good news."""
    cases = document["cases"]
    assert isinstance(cases, list)
    document["cases"] = [
        case for case in cases if case["scene"]["perturbation"]["label"] == contract.EXACT
    ]
    document["attribution"] = []
    assert "carries no case for perturbation" in _refused(document)


def test_a_perturbed_case_whose_attribution_went_missing_is_refused(
    document: Any,
) -> None:
    attribution = document["attribution"]
    assert isinstance(attribution, list)
    document["attribution"] = attribution[1:]
    assert "no attribution for it" in _refused(document)


def test_a_perturbed_case_used_as_a_control_is_refused(document: Any) -> None:
    entry = document["attribution"][0]
    entry["control_case"] = "depth-noise"
    assert "perturbed case as a control" in _refused(document)


def test_a_case_named_as_its_own_control_is_refused(document: Any) -> None:
    entry = document["attribution"][0]
    entry["control_case"] = entry["case_id"]
    assert "its own control" in _refused(document)


def test_an_attributed_delta_that_is_not_the_difference_is_refused(
    document: Any,
) -> None:
    metric = document["attribution"][0]["arms"][0]["metrics"][0]
    metric["delta"] = float(metric["delta"]) - 1.0
    assert "states delta" in _refused(document)


def test_an_attributed_number_that_is_not_the_one_the_case_published_is_refused(
    document: Any,
) -> None:
    metric = document["attribution"][0]["arms"][0]["metrics"][0]
    metric["perturbed"] = float(metric["perturbed"]) + 1.0
    metric["delta"] = float(metric["perturbed"]) - float(metric["control"])
    assert "not the ones the two cases published" in _refused(document)


def test_a_degraded_flag_pointing_the_wrong_way_is_refused(document: Any) -> None:
    metric = document["attribution"][0]["arms"][0]["metrics"][0]
    metric["degraded"] = not metric["degraded"]
    assert "degraded=" in _refused(document)


def test_an_attribution_on_a_metric_with_no_declared_direction_is_refused(
    document: Any,
) -> None:
    metric = document["attribution"][0]["arms"][0]["metrics"][0]
    metric["metric"] = "field.observed_voxels"
    assert "declares no direction" in _refused(document)


def test_a_decision_that_is_not_arithmetic_is_refused(document: Any) -> None:
    document["cases"][0]["status"] = "fail"
    document["totals"]["passed"] = int(document["totals"]["passed"]) - 1
    document["totals"]["failed"] = int(document["totals"]["failed"]) + 1
    assert "the decision is arithmetic" in _refused(document)


def test_totals_that_do_not_count_their_own_cases_are_refused(
    document: Any,
) -> None:
    document["totals"]["cases"] = int(document["totals"]["cases"]) + 1
    assert "does not count its own cases" in _refused(document)


def test_an_artifact_with_no_limitations_is_refused(document: Any) -> None:
    document["limitations"] = []
    assert "states no limitations" in _refused(document)


def test_a_non_finite_number_anywhere_is_refused(document: Any) -> None:
    document["cases"][0]["candidates"][0]["quality"]["gates"][0]["measured"] = float("inf")
    assert "finite" in _refused(document)


def test_an_impossible_coverage_fraction_is_refused(document: Any) -> None:
    """A coverage is a fraction, and an improvement past 1.0 is not an improvement."""
    for case in document["cases"]:
        for candidate in case["candidates"]:
            for measurement in candidate["quality"]["measurements"]:
                if measurement["kind"] == "cloud_agreement":
                    measurement["observed_to_reference"]["coverage_fraction"] = 1.4
                    assert "fraction" in _refused(document)
                    return
    pytest.fail("no agreement measurement was published to corrupt")


def test_a_malformed_case_that_still_reports_results_is_refused(
    document: Any,
) -> None:
    case = document["cases"][0]
    case["status"] = "malformed"
    assert "malformed and still reports" in _refused(document)


def test_a_case_that_applies_no_gates_at_all_is_refused(document: Any) -> None:
    document["cases"][0]["gates"] = []
    assert "applies no gates" in _refused(document)


def test_a_report_read_from_a_file_is_validated_on_the_way_in(report: Any, tmp_path: Path) -> None:
    path = tmp_path / "surface-oracle-report.json"
    path.write_text(json.dumps(dict(report)), encoding="utf-8")
    assert harness.read_report(path)["decision"] == "pass"
    path.write_text(json.dumps({"kind": "nope"}), encoding="utf-8")
    with pytest.raises(ReportUnusable):
        harness.read_report(path)


# --- reading the corpus off disk ------------------------------------------------------------


def test_the_committed_manifest_names_every_case_file_and_its_digest(
    corpus: harness.Corpus,
) -> None:
    named = {entry.name for entry in corpus.files}
    on_disk = {path.name for path in harness.CORPUS.glob("*.json")} - {harness.MANIFEST_NAME}
    assert named == on_disk
    for entry in corpus.files:
        assert (
            hashlib.sha256((harness.CORPUS / entry.name).read_bytes()).hexdigest() == entry.sha256
        )


def test_an_altered_case_file_is_refused_before_anything_is_measured(tmp_path: Path) -> None:
    root = _copy_corpus(tmp_path)
    path = root / "depth-noise.json"
    path.write_text(path.read_text(encoding="utf-8").replace("  ", " ", 1), encoding="utf-8")
    corpus = harness.load_corpus(root)
    entry = next(item for item in corpus.files if item.case_id == "depth-noise")
    with pytest.raises(FixtureUnusable, match="no longer the evidence"):
        harness.load_case(path, entry=entry, corpus=corpus)


def test_a_case_file_the_manifest_does_not_name_is_refused(tmp_path: Path) -> None:
    root = _copy_corpus(tmp_path)
    (root / "extra-case.json").write_text("{}", encoding="utf-8")
    with pytest.raises(HarnessRefused, match="does not name"):
        harness.load_corpus(root)


def test_a_corpus_with_no_manifest_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(HarnessRefused, match="not a regular file"):
        harness.load_corpus(root)


def test_altered_truth_is_refused_by_the_committed_digests(tmp_path: Path) -> None:
    """Move one shape and the rendered truth no longer hashes to what is committed."""
    root = _copy_corpus(tmp_path)

    def move_the_sphere(document: Any) -> None:
        for shape in document["scene"]["shapes"]:
            if shape["kind"] == "sphere":
                shape["radius"] = float(shape["radius"]) + 0.05

    _rewrite(root, "exact-control.json", move_the_sphere)
    corpus = harness.load_corpus(root)
    entry = next(item for item in corpus.files if item.case_id == "exact-control")
    with pytest.raises(FixtureUnusable, match=r"truth|digest|sha256"):
        harness.load_case(root / entry.name, entry=entry, corpus=corpus)


def test_a_forged_perturbation_label_is_refused(tmp_path: Path) -> None:
    """File the depth-noise case under 'exact' and the label check catches it."""
    root = _copy_corpus(tmp_path)

    def relabel(document: Any) -> None:
        document["perturbation"]["label"] = contract.EXACT
        document["control_case"] = None

    _rewrite(root, "depth-noise.json", relabel)
    corpus = harness.load_corpus(root)
    entry = next(item for item in corpus.files if item.case_id == "depth-noise")
    with pytest.raises((FixtureUnusable, contract.ScoringUnusable)):
        harness.load_case(root / entry.name, entry=entry, corpus=corpus)


# --- the controls a corpus has to keep ---------------------------------------------------------


def test_the_committed_corpus_passes_its_own_control_check(
    cases: list[harness.CaseSpec], corpus: harness.Corpus
) -> None:
    harness._check_controls(cases, corpus)


def _with(cases: list[harness.CaseSpec], replacement: harness.CaseSpec) -> list[harness.CaseSpec]:
    """The whole corpus with one case swapped, so only the corruption differs."""
    return [replacement if case.case_id == replacement.case_id else case for case in cases]


def test_a_swapped_baseline_is_refused(
    cases: list[harness.CaseSpec], corpus: harness.Corpus
) -> None:
    """Point a perturbed case at another PERTURBED case as its baseline."""
    perturbed = next(case for case in cases if case.case_id == "depth-noise")
    swapped = replace(perturbed, scene=replace(perturbed.scene, control_case="pose-rotation"))
    with pytest.raises(HarnessRefused, match="that case is perturbed"):
        harness._check_controls(_with(cases, swapped), corpus)


def test_a_control_the_corpus_does_not_contain_is_refused(
    cases: list[harness.CaseSpec], corpus: harness.Corpus
) -> None:
    perturbed = next(case for case in cases if case.case_id == "depth-noise")
    orphaned = replace(
        perturbed, scene=replace(perturbed.scene, control_case="a-case-nobody-wrote")
    )
    with pytest.raises(HarnessRefused, match="does not contain"):
        harness._check_controls(_with(cases, orphaned), corpus)


def test_a_control_rendering_a_different_scene_is_refused(
    cases: list[harness.CaseSpec], corpus: harness.Corpus
) -> None:
    """The control's rendered truth must be the perturbed case's truth, view for view."""
    control = next(case for case in cases if case.case_id == "exact-control")
    other = next(case for case in cases if case.case_id == "pose-rotation").scene.views
    swapped = replace(
        control,
        scene=replace(
            control.scene,
            views=tuple(
                replace(view, truth=elsewhere.raw + b"\x00")
                for view, elsewhere in zip(control.scene.views, other, strict=True)
            ),
        ),
    )
    with pytest.raises(HarnessRefused, match="do not render the same"):
        harness._check_controls(_with(cases, swapped), corpus)


def test_a_control_fusing_into_a_different_volume_is_refused(
    cases: list[harness.CaseSpec], corpus: harness.Corpus
) -> None:
    control = next(case for case in cases if case.case_id == "exact-control")
    moved = replace(
        control,
        scene=replace(
            control.scene,
            volume=replace(control.scene.volume, max_depth=control.scene.volume.max_depth + 1.0),
        ),
    )
    with pytest.raises(HarnessRefused, match="observed space has to be the same box"):
        harness._check_controls(_with(cases, moved), corpus)


def test_a_control_naming_different_regions_is_refused(
    cases: list[harness.CaseSpec], corpus: harness.Corpus
) -> None:
    control = next(case for case in cases if case.case_id == "exact-control")
    moved = replace(
        control,
        scene=replace(
            control.scene,
            regions=replace(
                control.scene.regions,
                coverage_tolerance=control.scene.regions.coverage_tolerance * 2.0,
            ),
        ),
    )
    with pytest.raises(HarnessRefused, match="different regions"):
        harness._check_controls(_with(cases, moved), corpus)


def test_a_corpus_that_stopped_expecting_a_corruption_control_is_refused(
    cases: list[harness.CaseSpec], corpus: harness.Corpus
) -> None:
    """The missing-bad-control attack, aimed at the reader rather than at a case."""
    stripped = [
        replace(
            case,
            candidates=tuple(
                candidate
                for candidate in case.candidates
                if candidate.expectation.refusal != "surface_not_derived"
            ),
        )
        for case in cases
    ]
    with pytest.raises(HarnessRefused, match="surface_not_derived"):
        harness._check_controls(stripped, corpus)


def test_every_control_refusal_the_harness_owns_is_earned_somewhere(
    cases: list[harness.CaseSpec],
) -> None:
    earned = {
        candidate.expectation.refusal
        for case in cases
        for candidate in case.candidates
        if candidate.expectation.refusal is not None
    }
    assert earned >= harness.CONTROL_REFUSALS


def test_a_control_is_only_declared_where_its_corruption_is_detectable(
    cases: list[harness.CaseSpec],
) -> None:
    """A refusal a case can never earn would be a permanently failing expectation."""
    for case in cases:
        names = {candidate.name for candidate in case.candidates}
        touches_depth = case.scene.perturbation.touches_depth
        touches_pose = case.scene.perturbation.touches_pose
        truth_control = any(name.startswith("fused_from_truth_frames") for name in names)
        pose_control = any(name.startswith("fused_with_true_poses") for name in names)
        assert truth_control == touches_depth, case.case_id
        assert pose_control == touches_pose, case.case_id


# --- the command --------------------------------------------------------------------------------


def test_the_command_publishes_a_readable_artifact_and_exits_zero(tmp_path: Path) -> None:
    out = tmp_path / "surface-oracle-report.json"
    assert harness.main(["--quiet", "--out", str(out)]) == 0
    assert harness.read_report(out)["decision"] == "pass"


def test_the_command_refuses_an_output_path_it_may_not_write(tmp_path: Path) -> None:
    assert harness.main(["--quiet", "--out", str(tmp_path)]) == 2
