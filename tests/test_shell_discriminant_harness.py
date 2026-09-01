"""The corpus this experiment ships, and the strict reader that refuses a flattering report.

Two halves, and neither of them runs the whole experiment.

The first half is the committed corpus. :func:`tests.shell_discriminant_harness.load_corpus`
reads the manifest and the population without fusing anything, so the sealed
member list, the frozen candidate digest, the seeds, the split and every
corruption control cost milliseconds and are checked here against the files
that are actually committed.

The second half is the reader. :func:`tests.shell_discriminant_harness.validate_report`
recomputes an artifact from the artifact, and the only way to prove it refuses
a forgery is to hand it forgeries. It is handed them over a small VALID report
built here from the module's own typed values - three members at two declared
resolutions, four labelled shells, all 195 declared candidates, a complete
twenty-eight-cell cross-tab per candidate, one counterfactual and one truth-fed
upper bound. Building it costs no fusion at all, which is the point: the
producer's own full run is the harness command's job and takes minutes, and a
reader control that only fails after those minutes is a control nobody runs
while they are working.

The third half, at the bottom, is the committed corpus run once through
:func:`tests.shell_discriminant_harness.run`. It is slow - minutes, dominated
by fifty-four fusions - and it is here on purpose: these tests run in the
ORDINARY pytest job on the existing public GitHub-hosted workflow, so the facts
that make a sealed number mean anything (both classes in the sealed appliance
holdout, one realization per reconstruction input, one bounded decision-matrix
pass under the declared ceiling, a recommendation that follows from its own
gates) are asserted by CI rather than only by whoever last ran the command by
hand.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import shutil
from typing import Any, Final

import pytest

from tests import shell_discriminant as experiment
from tests import shell_discriminant_harness as harness
from tests import surface_oracle as contract
from tests.quality_harness import HarnessRefused, ReportUnusable, publish

#: The eight sealed holdout members, in the exact order the round-two planner
#: packet declared them before any of them was executed.
SEALED = (
    "sealed-outlier-125-p41",
    "sealed-outlier-185-p43",
    "sealed-outlier-155-p47",
    "sealed-outlier-105-p53",
    "sealed-combined-a",
    "sealed-combined-b",
    "sealed-combined-c",
    "sealed-combined-outlier-d",
)

#: The two resolutions the built report declares. The first is the one the
#: product question is asked at; the second is diagnostic.
RESOLUTIONS = (experiment.SELECTION_RESOLUTION, "v10-t4-w1-uniform")

#: Three members: the exact control, one training member and one sealed
#: holdout member. Enough to exercise every rule the reader has and small
#: enough that a validation costs no measurable time.
MEMBERS = (
    ("exact-control", contract.EXACT, experiment.TRAIN, None),
    ("outliers-a", contract.DEPTH_OUTLIERS, experiment.TRAIN, "exact-control"),
    ("sealed-a", contract.COMBINED, experiment.HOLDOUT, "exact-control"),
)

#: The (member, resolution) pairs that carry shells. The others are real,
#: published, empty populations - which is what makes the zero cells of the
#: cross-tab real rather than hypothetical.
POPULATED = (("outliers-a", experiment.SELECTION_RESOLUTION), ("sealed-a", "appliance"))

#: A second sealed holdout member whose appliance population is real closed
#: shells the evaluator labelled NONE of - supported geometry the scene's ideal
#: returns reach. The completeness controls run over this: round two derived a
#: population's split from whether a labelled record happened to exist in it, so
#: this population escaped its counterfactual even when the rule fired on it.
SUPPORTED_ONLY_MEMBERS = (
    *MEMBERS,
    ("sealed-b", contract.COMBINED, experiment.HOLDOUT, "exact-control"),
)
SUPPORTED_ONLY_SEALED = ("sealed-a", "sealed-b")
SUPPORTED_ONLY = (("sealed-b", experiment.SELECTION_RESOLUTION),)


def digest_of(*parts):
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def shell_of(digest, index, *, faces, first_face, contradicted):
    """One observable record with the ray and support profile of a false shell, or not."""
    return experiment.ObservableShell(
        shell_id=f"{digest[:16]}|{index}",
        voxel_size=0.02,
        shell_index=index,
        face_indices=tuple(range(first_face, first_face + faces)),
        faces=faces,
        vertices=6,
        area=0.001 if contradicted else 0.06,
        extent=(0.05, 0.05, 0.05) if contradicted else (0.4, 0.4, 0.4),
        diagonal=0.086 if contradicted else 0.69,
        centroid=(0.0, 0.0, 0.0),
        views_total=5,
        views_containing=0 if contradicted else 5,
        views_touching=1 if contradicted else 5,
        ray_probes=30,
        confirming_probes=3 if contradicted else 24,
        contradicting_probes=24 if contradicted else 3,
        occluded_probes=2,
        no_return_probes=1,
        outside_window_probes=0,
        ray_tolerance=0.04,
        measured_returns=1000,
        nearest_measured_distance=0.4 if contradicted else 0.01,
        farthest_measured_distance=0.6 if contradicted else 0.05,
        nearest_measured_capped=False,
        weight_min=0.2 if contradicted else 4.0,
        weight_mean=0.4 if contradicted else 6.0,
        weight_max=0.6 if contradicted else 8.0,
        min_weight=1.0,
        unavailable=(),
    )


def record_of(shell, *, false_surface, member_id, family, cell_id, split, digest):
    return experiment.ShellRecord(
        observable=shell,
        truth=experiment.ShellTruth(
            shell_id=shell.shell_id,
            false_surface=false_surface,
            worst_analytic_distance=0.4 if false_surface else 0.02,
            accuracy_tolerance=0.15,
            faces=shell.faces,
        ),
        member_id=member_id,
        family=family,
        cell_id=cell_id,
        split=split,
        reconstruction_digest=digest,
        face_indices=shell.face_indices,
    )


def build_populations(members=MEMBERS, populated=POPULATED, unlabelled=()):
    """Every declared member at every declared resolution, typed rather than spelled.

    ``unlabelled`` names populations whose closed shells are real and none of
    which the evaluator labelled: geometry the scene's ideal returns DO reach.
    A sealed holdout population like that still owes a counterfactual when the
    selected rule fires on it, and round two's reader let it escape one.
    """
    out = []
    for member_id, family, split, _control in members:
        for cell_id in RESOLUTIONS:
            digest = digest_of(member_id, cell_id)
            shells = []
            records = []
            if (member_id, cell_id) in unlabelled:
                shells = [shell_of(digest, 0, faces=8, first_face=0, contradicted=True)]
            elif (member_id, cell_id) in populated:
                false_shell = shell_of(digest, 0, faces=8, first_face=0, contradicted=True)
                correct_shell = shell_of(digest, 1, faces=12, first_face=8, contradicted=False)
                shells = [false_shell, correct_shell]
                records = [
                    record_of(
                        false_shell,
                        false_surface=True,
                        member_id=member_id,
                        family=family,
                        cell_id=cell_id,
                        split=split,
                        digest=digest,
                    ),
                    record_of(
                        correct_shell,
                        false_surface=False,
                        member_id=member_id,
                        family=family,
                        cell_id=cell_id,
                        split=split,
                        digest=digest,
                    ),
                ]
            out.append(
                experiment.Population(
                    member_id=member_id,
                    family=family,
                    split=split,
                    cell_id=cell_id,
                    voxel_size=0.02,
                    reconstruction_digest=digest,
                    faces=200,
                    shells=5,
                    closed_shells=len(shells),
                    observables=tuple(shells),
                    records=tuple(records),
                    unavailable=(),
                    work_units=1000,
                    triangle_samples=400,
                )
            )
    return out


def account_of(**overrides):
    base: dict[str, Any] = {
        "faces": 200,
        "vertices": 120,
        "analytic_within_tolerance_fraction": 0.90,
        "reference_coverage_fraction": 0.80,
        "triangle_within_tolerance_fraction": 0.85,
        "protected_area": 0.5,
        "bridging_faces": 2,
        # Exactly 1.0, because ``True == 1.0``: this is the real-valued half of
        # the boolean-confusion control, and a canonical value a boolean could
        # not compare equal to would not mount the attack.
        "surface_area": 1.0,
        "edge_coverage": 0.7,
        "thin_feature_coverage": 0.6,
        "worst_dimension_error": 0.01,
        "unsupported_false_shells": 1,
        "unsupported_false_shell_faces": 8,
        "unsupported_correct_shells": 1,
        "unsupported_correct_shell_faces": 12,
        "nonmanifold_edge_count": 0,
        "boundary_edge_count": 0,
        "shell_count": 5,
    }
    base.update(overrides)
    return experiment.SurfaceAccount(**base)


def gates_of(before, after, *, supported):
    """The twelve preservation gates, from two measured surfaces and the damage."""
    return {
        "counterfactual_false_faces_fell": (
            before.unsupported_false_shell_faces > after.unsupported_false_shell_faces
        ),
        "counterfactual_correct_faces_held": (
            before.unsupported_correct_shell_faces == after.unsupported_correct_shell_faces
        ),
        "counterfactual_removes_no_supported_faces": not supported,
        "counterfactual_analytic_accuracy_held": (
            after.analytic_within_tolerance_fraction >= before.analytic_within_tolerance_fraction
        ),
        "counterfactual_triangle_accuracy_held": (
            after.triangle_within_tolerance_fraction >= before.triangle_within_tolerance_fraction
        ),
        "counterfactual_reference_coverage_held": (
            after.reference_coverage_fraction >= before.reference_coverage_fraction
        ),
        "counterfactual_protected_volume_held": after.protected_area <= before.protected_area,
        "counterfactual_bridging_held": after.bridging_faces <= before.bridging_faces,
        "counterfactual_edge_coverage_held": after.edge_coverage >= before.edge_coverage,
        "counterfactual_thin_feature_held": (
            after.thin_feature_coverage >= before.thin_feature_coverage
        ),
        "counterfactual_dimensions_held": (
            after.worst_dimension_error <= before.worst_dimension_error
        ),
        "counterfactual_topology_held": (
            after.nonmanifold_edge_count <= before.nonmanifold_edge_count
            and after.boundary_edge_count <= before.boundary_edge_count
        ),
    }


def synthetic_perturbation(family):
    """The whole perturbation mapping a committed member declares, at its defaults.

    Not ``{"label": family}``. The reader types every field of a published
    perturbation before comparing it, so a synthetic corpus that declared one
    key would be a corpus no report could describe, and the derived-logic
    controls would then be exercising the key-set refusal rather than the logic
    they are about. The accepted oracle's own value produces the exact twelve
    fields the committed population file carries.
    """
    declared = contract.Perturbation(label=family).as_dict()
    return {key: declared[key] for key in harness.PERTURBATION_KEYS}


def synthetic_corpus(members=MEMBERS, sealed=("sealed-a",)):
    """A small corpus value, built rather than read.

    The DERIVED-logic controls need a population a rule can qualify on, and the
    committed corpus produces none - no candidate cleared the training bar, so
    its report selects nothing, owes no counterfactual and measures no ceiling.
    Those branches are still protocol, so they are exercised over a corpus this
    module builds. It is never handed to the public reader: that boundary takes
    no evidence root at all and accepts the committed corpus alone.
    """
    members = tuple(
        harness.Member(
            member_id=member_id,
            family=family,
            split=split,
            seed=f"shell-discriminant:{member_id}",
            control_case=control,
            perturbation=synthetic_perturbation(family),
            views=(
                {
                    "index": 0,
                    "truth_sha256": digest_of("truth", member_id),
                    "raw_sha256": digest_of("raw", member_id),
                    "truth_returned": 1,
                    "raw_returned": 1,
                },
            ),
        )
        for member_id, family, split, control in members
    )
    return harness.Corpus(
        root=harness.CORPUS,
        manifest_sha256=digest_of("manifest"),
        population_sha256=digest_of("population"),
        base_case="exact-control.json",
        base_case_sha256=digest_of("base-case"),
        base_document={},
        resolutions=RESOLUTIONS,
        selection_resolution=experiment.SELECTION_RESOLUTION,
        members=members,
        sealed_holdout=tuple(sealed),
        candidate_sha256=experiment.FROZEN_CANDIDATE_DIGEST,
        protocol_sha256=experiment.grid_digest(),
        selection_protocol=experiment.SELECTION_PROTOCOL,
        material_fraction=experiment.MATERIAL_FRACTION,
        work_ceiling=experiment.WORK_CEILING,
        upper_bound_member="sealed-a",
        upper_bound_cell="appliance",
    )


def synthetic_evidence(
    populations, *, upper_bound, measured, members=MEMBERS, sealed=("sealed-a",)
):
    """The canonical bundle those populations would be rebuilt into.

    ``measured`` stands in for the trusted rebuild's re-measurement of a pruned
    mesh, keyed by (member, resolution). The committed corpus re-fuses and
    re-measures for real; there is no mesh here to prune, so the derived-logic
    controls declare what that measurement is and then attack everything the
    reader derives from it.
    """
    corpus = synthetic_corpus(members, sealed)
    built = tuple(
        harness.Built(
            member=corpus.member(population.member_id),
            cell_id=population.cell_id,
            population=population,
        )
        for population in populations
    )
    return harness._Evidence(
        corpus=corpus,
        built=built,
        failures=(),
        cells=tuple((item.member.member_id, item.cell_id) for item in built),
        populations={(item.member.member_id, item.cell_id): item.population for item in built},
        records=tuple(record for item in built for record in item.population.records),
        corpus_block=harness._json_ready(harness._corpus_block(corpus)),
        realization_block=tuple(harness._json_ready(harness._realization_block(built))),
        population_block=tuple(harness._json_ready([item.population.as_dict() for item in built])),
        upper_bound_block=harness._json_ready(upper_bound),
        remeasure=lambda population, selected: harness._json_ready(
            measured[(population.member_id, population.cell_id)]
        ),
    )


def build_case(*, members=MEMBERS, sealed=("sealed-a",), populated=POPULATED, unlabelled=()):
    """One canonical rebuild, and the one document that matches it.

    Every derived number is produced by the same code the producer uses - the
    decision matrix, the cross-tab, the aggregates, the selection, the totals -
    and every published block is produced by the harness's own block builders,
    so the reader accepting it is evidence about the reader rather than about a
    hand-copied table. The two measured surfaces are declared here because
    there is no mesh in a built case to prune: the committed corpus re-fuses
    the member and re-measures the pruned surface for real, and
    :func:`synthetic_evidence` stands in for that re-measurement so the
    controls below can attack everything the reader derives from it - every
    delta, gate and passed status - and the exact binding against it.
    """
    populations = build_populations(members, populated, unlabelled)
    records = [record for population in populations for record in population.records]
    experiment.reset_rule_calls()
    matrix = experiment.build_decision_matrix(records)
    selection = experiment.choose(matrix, records)
    assert selection.rule is not None, "the built population must qualify a candidate"
    rule = selection.rule

    aggregates = experiment.aggregate_slices(records)
    candidates = [
        {
            "rule": candidate.as_dict(),
            "cross_tab": {
                cell.slice_name: cell.as_dict()
                for cell in experiment.cross_tab(
                    matrix, candidate.rule_id, records, resolutions=RESOLUTIONS
                )
            },
            "aggregates": {
                name: experiment.confusion(
                    matrix, candidate.rule_id, subset, slice_name=name
                ).as_dict()
                for name, subset in aggregates.items()
            },
        }
        for candidate in experiment.RULE_GRID
    ]

    training = experiment.training_records(records)
    holdout = experiment.holdout_records(records)
    train_counts = experiment.confusion(
        matrix,
        rule.rule_id,
        training,
        slice_name=f"split:{experiment.TRAIN}|cell:{experiment.SELECTION_RESOLUTION}",
    )
    held = experiment.confusion(
        matrix,
        rule.rule_id,
        holdout,
        slice_name=f"split:{experiment.HOLDOUT}|cell:{experiment.SELECTION_RESOLUTION}",
    )

    # Every sealed holdout appliance population the rule fires on, in the
    # corpus's declared order and with no prefilter: exactly the list the
    # reader derives for itself and requires this document to publish.
    counterfactuals = []
    combined_gates: dict[str, bool] = {}
    counterfactual_calls = 0
    upper_bound: dict[str, Any] = {}
    for population in populations:
        if population.cell_id != experiment.SELECTION_RESOLUTION:
            continue
        if population.split != experiment.HOLDOUT or population.member_id not in sealed:
            continue
        counterfactual_calls += len(population.observables)
        fired = experiment.select(rule, population.observables)
        if not fired:
            continue
        labelled = {record.shell_id: record for record in population.records}
        removed = [item for item in population.observables if item.shell_id in fired]
        false_removed = [
            item
            for item in removed
            if item.shell_id in labelled and labelled[item.shell_id].truth.false_surface
        ]
        correct_removed = [
            item
            for item in removed
            if item.shell_id in labelled and not labelled[item.shell_id].truth.false_surface
        ]
        supported_removed = [item for item in removed if item.shell_id not in labelled]
        removed_faces = sum(item.faces for item in removed)
        false_faces = sum(item.faces for item in false_removed)
        before = account_of()
        after = account_of(
            faces=200 - removed_faces,
            vertices=120 - removed_faces,
            analytic_within_tolerance_fraction=0.95,
            unsupported_false_shells=1 - len(false_removed),
            unsupported_false_shell_faces=8 - false_faces,
            shell_count=5 - len(removed),
        )
        evidence = experiment.CounterfactualEvidence(
            member_id=population.member_id,
            cell_id=population.cell_id,
            split=population.split,
            reconstruction_digest=population.reconstruction_digest,
            closed_shells=len(population.observables),
            removed_shell_ids=tuple(sorted(set(fired))),
            removed_faces=removed_faces,
            removed_false_shells=len(false_removed),
            removed_false_shell_faces=false_faces,
            removed_correct_shells=len(correct_removed),
            removed_correct_shell_faces=sum(item.faces for item in correct_removed),
            removed_supported_shells=len(supported_removed),
            removed_supported_shell_faces=sum(item.faces for item in supported_removed),
            before=before,
            after=after,
            gates=gates_of(before, after, supported=supported_removed),
        )
        counterfactuals.append(evidence.as_dict())
        for name, passed in evidence.gates.items():
            combined_gates[name] = combined_gates.get(name, True) and passed
        if (population.member_id, population.cell_id) == ("sealed-a", "appliance"):
            upper_bound = dict(evidence.as_dict())
            upper_bound["selector"] = harness.UPPER_BOUND_SELECTOR
            upper_bound["measured"] = True
    assert upper_bound, "the predeclared upper-bound population must have been measured"
    bundle = synthetic_evidence(
        populations,
        upper_bound=upper_bound,
        measured={(item["member_id"], item["cell_id"]): item for item in counterfactuals},
        members=members,
        sealed=sealed,
    )

    gate_results = {
        "train_removes_no_correct_faces": train_counts.removes_no_correct_faces,
        "train_removes_material_false_faces": train_counts.removes_material_false_faces,
        "holdout_removes_no_correct_faces": held.removes_no_correct_faces,
        "holdout_removes_material_false_faces": held.removes_material_false_faces,
        **combined_gates,
    }
    recommendation = (
        harness.PRUNE
        if all(gate_results.get(gate, False) for gate in experiment.GATES)
        else harness.NO_PRUNING
    )
    evaluations = len(experiment.RULE_GRID) * len(records)
    document = {
        "schema_version": harness.REPORT_SCHEMA_VERSION,
        "kind": harness.REPORT_KIND,
        "corpus": harness._corpus_block(bundle.corpus),
        "namespaces": {
            "observable_source": experiment.OBSERVABLE_SOURCE,
            "label_source": experiment.LABEL_SOURCE,
            "observables": sorted(experiment.FEATURES),
            "label_only_fields": list(experiment.LABEL_ONLY_FIELDS),
        },
        "protocol": experiment.SELECTION_PROTOCOL,
        "gates": list(experiment.GATES),
        "realizations": harness._realization_block(bundle.built),
        "populations": [population.as_dict() for population in populations],
        "candidates": candidates,
        "selection": selection.as_dict(),
        "holdout_confusion": held.as_dict(),
        "gate_results": {name: bool(value) for name, value in sorted(gate_results.items())},
        "counterfactuals": counterfactuals,
        "upper_bound": upper_bound,
        "recommendation": recommendation,
        "failures": [],
        "totals": harness._expected_totals(bundle),
        "cost": {
            "fusions": len(populations),
            "voxel_visits": sum(p.work_units for p in populations),
            "triangle_samples": sum(p.triangle_samples for p in populations),
            "producer_rule_calls": evaluations,
            "counterfactual_rule_calls": counterfactual_calls,
            "reader_audit_rule_calls": 0,
            "unaccounted_rule_calls": 0,
            "total_rule_calls": evaluations + counterfactual_calls,
            "decision_matrix_cells": evaluations,
            "work_ceiling": experiment.WORK_CEILING,
            "counterfactual_remeasurements": 2 * (len(counterfactuals) + 1),
            "note": harness.COST_NOTE,
        },
        "decision": "pass",
        "limitations": list(harness.LIMITATIONS),
    }
    return bundle, json.loads(json.dumps(document, sort_keys=True, allow_nan=False))


#: What the derived-logic controls call their document, in refusal messages.
DERIVED: Final = "a derived-logic shell-discriminant report"


@pytest.fixture(scope="module")
def derived_case():
    """One canonical rebuild and the one document that matches it, built once."""
    bundle, document = build_case()
    harness._validate_document(document, bundle, what=DERIVED)
    return bundle, document


@pytest.fixture
def derived_report(derived_case):
    """A private copy, so a control that deforms it cannot reach another test."""
    return copy.deepcopy(derived_case[1])


@pytest.fixture
def audit(derived_case):
    """Read a document against the synthetic rebuild, not against the committed one."""
    bundle = derived_case[0]

    def _audit(document):
        return harness._validate_document(document, bundle, what=DERIVED)

    return _audit


@pytest.fixture
def corpus_root(tmp_path):
    """A writable copy of the committed corpus, for the corruption controls."""
    root = tmp_path / "shell_discriminant"
    shutil.copytree(harness.CORPUS, root)
    return root


def rewrite(root, *, population=None, manifest=None):
    """Rewrite a copied corpus, keeping the manifest's own digest bookkeeping honest.

    A control that changed the population and forgot the manifest would be
    testing the digest check instead of the thing it meant to test, so the
    population digest is refreshed here unless the control is deliberately
    breaking it.
    """
    if population is not None:
        payload = json.dumps(population, indent=2, sort_keys=True, allow_nan=False) + "\n"
        (root / harness.POPULATION_NAME).write_text(payload, encoding="utf-8")
        document = json.loads((root / harness.MANIFEST_NAME).read_text(encoding="utf-8"))
        document["population"]["sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        document["population"]["bytes"] = len(payload.encode("utf-8"))
        (root / harness.MANIFEST_NAME).write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if manifest is not None:
        (root / harness.MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return root


def read_population(root):
    return json.loads((root / harness.POPULATION_NAME).read_text(encoding="utf-8"))


def read_manifest(root):
    return json.loads((root / harness.MANIFEST_NAME).read_text(encoding="utf-8"))


# --- the committed corpus is the one that was sealed --------------------------------------


def test_the_committed_corpus_binds_the_frozen_candidate_bodies():
    """The candidate digest is the one the packet froze before the holdout ran."""
    corpus = harness.load_corpus()
    assert corpus.candidate_sha256 == experiment.FROZEN_CANDIDATE_DIGEST
    committed = read_manifest(harness.CORPUS)
    assert committed["candidate_sha256"] == experiment.FROZEN_CANDIDATE_DIGEST
    assert corpus.protocol_sha256 == experiment.grid_digest()
    assert corpus.selection_resolution == experiment.SELECTION_RESOLUTION
    assert corpus.material_fraction == experiment.MATERIAL_FRACTION
    assert corpus.work_ceiling == experiment.WORK_CEILING


def test_the_sealed_holdout_is_exactly_the_eight_predeclared_members():
    corpus = harness.load_corpus()
    assert corpus.sealed_holdout == SEALED
    holdout = tuple(
        member.member_id for member in corpus.members if member.split == experiment.HOLDOUT
    )
    assert holdout == SEALED


def test_every_other_member_is_training_data_this_round():
    """Round one inspected its holdout and then widened the grid; all 19 are TRAIN now."""
    corpus = harness.load_corpus()
    training = [member for member in corpus.members if member.split == experiment.TRAIN]
    assert len(training) == 19
    assert len(corpus.members) == 27
    assert not {member.member_id for member in training} & set(SEALED)


def test_every_member_carries_its_own_declared_seed():
    for member in harness.load_corpus().members:
        assert member.seed == f"shell-discriminant:{member.member_id}"


def test_the_corpus_declares_both_resolutions_and_selects_at_the_appliance():
    corpus = harness.load_corpus()
    assert corpus.resolutions == RESOLUTIONS
    assert corpus.selection_resolution in corpus.resolutions
    assert corpus.upper_bound_member in {member.member_id for member in corpus.members}
    assert corpus.upper_bound_cell in corpus.resolutions


def test_no_realization_is_on_both_sides_of_the_split():
    """Checked from the committed frame digests, not from the builder's intentions."""
    corpus = harness.load_corpus()
    by_split: dict[str, set[str]] = {split: set() for split in experiment.SPLITS}
    for member in corpus.members:
        by_split[member.split].add(member.frame_digest)
    assert not by_split[experiment.TRAIN] & by_split[experiment.HOLDOUT]


def test_a_duplicated_realization_across_the_split_is_refused(corpus_root):
    """The scene-leakage control: the same frames under two names, two splits."""
    population = read_population(corpus_root)
    original = next(
        member
        for member in population["members"]
        if member["split"] == experiment.TRAIN and member["family"] != contract.EXACT
    )
    clone = copy.deepcopy(original)
    clone["member_id"] = f"{original['member_id']}-clone"
    clone["split"] = experiment.HOLDOUT
    clone["seed"] = f"shell-discriminant:{clone['member_id']}"
    population["members"].append(clone)
    rewrite(corpus_root, population=population)
    manifest = read_manifest(corpus_root)
    manifest["sealed_holdout"] = [*manifest["sealed_holdout"], clone["member_id"]]
    rewrite(corpus_root, manifest=manifest)
    with pytest.raises(HarnessRefused, match="identical frames"):
        harness.load_corpus(corpus_root)


def test_a_corpus_without_its_exact_control_is_refused(corpus_root):
    population = read_population(corpus_root)
    population["members"] = [
        member for member in population["members"] if member["family"] != contract.EXACT
    ]
    rewrite(corpus_root, population=population)
    with pytest.raises(HarnessRefused, match="exact control"):
        harness.load_corpus(corpus_root)


def test_a_moved_candidate_digest_is_refused(corpus_root):
    """The post-result-body control, at the corpus boundary."""
    manifest = read_manifest(corpus_root)
    manifest["candidate_sha256"] = "0" * 64
    rewrite(corpus_root, manifest=manifest)
    with pytest.raises(HarnessRefused, match="frozen at"):
        harness.load_corpus(corpus_root)


def test_a_moved_protocol_digest_is_refused(corpus_root):
    manifest = read_manifest(corpus_root)
    manifest["protocol_sha256"] = "0" * 64
    rewrite(corpus_root, manifest=manifest)
    with pytest.raises(HarnessRefused, match="protocol digest"):
        harness.load_corpus(corpus_root)


def test_a_rewritten_protocol_is_refused(corpus_root):
    manifest = read_manifest(corpus_root)
    manifest["selection_protocol"] = manifest["selection_protocol"].replace("TRAIN", "HOLDOUT")
    rewrite(corpus_root, manifest=manifest)
    with pytest.raises(HarnessRefused, match="selection protocol"):
        harness.load_corpus(corpus_root)


def test_a_sealed_list_that_is_not_the_holdout_is_refused(corpus_root):
    manifest = read_manifest(corpus_root)
    manifest["sealed_holdout"] = list(SEALED[:-1])
    rewrite(corpus_root, manifest=manifest)
    with pytest.raises(HarnessRefused, match="sealed"):
        harness.load_corpus(corpus_root)


def test_an_edited_population_is_refused_by_its_digest(corpus_root):
    population = read_population(corpus_root)
    population["members"][1]["seed"] = "somebody-elses-seed"
    (corpus_root / harness.POPULATION_NAME).write_text(
        json.dumps(population, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    with pytest.raises(HarnessRefused, match="which is not what the manifest records"):
        harness.load_corpus(corpus_root)


def test_one_resolution_is_refused(corpus_root):
    """A confusion cannot be reported BY resolution from a single resolution."""
    manifest = read_manifest(corpus_root)
    manifest["resolutions"] = manifest["resolutions"][:1]
    rewrite(corpus_root, manifest=manifest)
    with pytest.raises(HarnessRefused, match="separate its confusion counts BY resolution"):
        harness.load_corpus(corpus_root)


def test_an_edited_frame_digest_is_refused_when_the_member_is_built(corpus_root):
    """The altered-truth control: a member whose declared frames are not its frames."""
    population = read_population(corpus_root)
    member = population["members"][1]
    member["views"][0]["raw_sha256"] = "1" * 64
    rewrite(corpus_root, population=population)
    corpus = harness.load_corpus(corpus_root)
    with pytest.raises(Exception, match="raw_sha256"):
        harness.build_member(corpus.member(member["member_id"]), corpus)


def test_a_member_with_a_view_omitted_is_refused(corpus_root):
    """The omitted-view control."""
    population = read_population(corpus_root)
    member = population["members"][1]
    member["views"] = member["views"][:-1]
    for index, view in enumerate(member["views"]):
        view["index"] = index
    rewrite(corpus_root, population=population)
    corpus = harness.load_corpus(corpus_root)
    with pytest.raises(Exception, match="view"):
        harness.build_member(corpus.member(member["member_id"]), corpus)


# --- the built artifact is one the reader accepts -----------------------------------------


def test_the_built_report_is_accepted_and_survives_its_own_text(derived_case, audit):
    """Encoding and decoding the document changes nothing the reader depends on."""
    document = derived_case[1]
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert audit(json.loads(text))["kind"] == harness.REPORT_KIND


def test_the_built_report_publishes_the_complete_cross_product(derived_case):
    """Twenty-eight cells per candidate, zero-shell cells included."""
    valid_report = derived_case[1]
    expected = [experiment.cross_tab_name(*cell) for cell in experiment.cross_cells(RESOLUTIONS)]
    assert len(expected) == 28
    for entry in valid_report["candidates"]:
        assert sorted(entry["cross_tab"]) == sorted(expected)
    first = valid_report["candidates"][0]["cross_tab"]
    assert any(cell["shells"] == 0 for cell in first.values())


def test_the_built_report_carries_no_wall_clock(derived_case):
    """A report that has to be byte-identical between runs cannot carry a timer."""
    text = json.dumps(derived_case[1], sort_keys=True)
    assert "seconds" not in text
    assert "elapsed" not in text


def test_validating_the_report_twice_serialises_identically(derived_case, audit):
    first = json.dumps(derived_case[1], indent=2, sort_keys=True, allow_nan=False)
    second = json.dumps(
        audit(copy.deepcopy(derived_case[1])),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    assert first == second


def test_the_render_names_the_recommendation(derived_case):
    valid_report = derived_case[1]
    rendered = harness.render(valid_report)
    assert str(valid_report["recommendation"]) in rendered
    assert "unsupported closed shell" in rendered


def test_the_reader_accounts_for_its_own_audit_matrix_separately(derived_case, audit):
    """The reader rebuilds a matrix. It is the READER's work and is never hidden."""
    experiment.reset_rule_calls()
    audit(copy.deepcopy(derived_case[1]))
    ledger = experiment.rule_calls()
    assert ledger.get(experiment.PRODUCER_PHASE, 0) == 0
    assert ledger.get("unaccounted", 0) == 0
    assert ledger[experiment.READER_PHASE] > 0
    experiment.reset_rule_calls()


def test_the_public_reader_takes_no_evidence_root(derived_case):
    """The boundary a caller reaches cannot be pointed at other evidence.

    Round two's reader checked a report against its own claims, so what it was
    checked against was whatever the report said. The public entry points now
    rebuild the committed corpus themselves, and they take no parameter that
    could redirect that: a corpus argument here would put the trusted evidence
    back under the caller's control, which is the whole defect.
    """
    for entry in (harness.validate_report, harness.loads_report, harness.read_report):
        names = set(inspect.signature(entry).parameters)
        assert not names & {"corpus", "root", "evidence", "against"}, (entry.__name__, names)
    with pytest.raises(ReportUnusable, match="committed corpus"):
        harness.validate_report(copy.deepcopy(derived_case[1]))


def test_a_backend_only_fusion_policy_change_changes_the_canonical_evidence_key(monkeypatch):
    """The cache key alone, before and after a policy change. Nothing is fused here.

    The trusted rebuild is memoised once per process, so its key has to be a
    complete identity for the code that produces those populations. Moving the
    diagnostic arm off the reference backend and onto the appliance one
    produces a different population from byte-identical committed inputs, and
    round three's key would have handed back the previous backend's bundle: it
    named every committed digest and every protocol constant, and no part of
    the fusion policy at all.
    """
    corpus = harness.load_corpus()
    reference = harness._evidence_key(corpus)
    assert harness._fusion_backend(harness.APPLIANCE) == contract.APPLIANCE_BACKEND
    assert harness._fusion_backend(RESOLUTIONS[1]) == contract.BACKEND
    monkeypatch.setattr(harness, "_fusion_backend", lambda cell_id: contract.APPLIANCE_BACKEND)
    assert harness._evidence_key(corpus) != reference
    assert harness._evidence_key(corpus) == harness._evidence_key(corpus)


# --- the reader refuses a report that flatters itself ---------------------------------------


def test_a_rule_body_edited_after_a_result_is_refused(derived_report, audit):
    """The same accepted id with a moved threshold."""
    body = derived_report["candidates"][0]["rule"]
    body["tests"][0]["threshold"] = body["tests"][0]["threshold"] + 1.0
    with pytest.raises(ReportUnusable, match="publishes rule body"):
        audit(derived_report)


def test_a_report_missing_a_candidate_is_refused(derived_report, audit):
    derived_report["candidates"] = derived_report["candidates"][:-1]
    with pytest.raises(ReportUnusable, match=r"candidate\(s\) against"):
        audit(derived_report)


def test_a_report_with_one_shell_record_omitted_is_refused(derived_report, audit):
    population = next(item for item in derived_report["populations"] if item["records"])
    population["records"] = population["records"][:-1]
    with pytest.raises(ReportUnusable, match="its own records give"):
        audit(derived_report)


def test_a_report_with_a_swapped_label_boolean_is_refused(derived_report, audit):
    population = next(item for item in derived_report["populations"] if item["records"])
    label = population["records"][0]["label"]
    label["false_surface"] = not label["false_surface"]
    with pytest.raises(ReportUnusable, match="swapped label"):
        audit(derived_report)


def test_a_report_with_a_forged_ray_partition_is_refused(derived_report, audit):
    population = next(item for item in derived_report["populations"] if item["records"])
    population["records"][0]["observable"]["confirming_probes"] += 1
    with pytest.raises(ReportUnusable, match="classified ray probe"):
        audit(derived_report)


def test_a_report_with_a_forged_weight_observable_is_refused(derived_report, audit):
    population = next(item for item in derived_report["populations"] if item["records"])
    population["records"][0]["observable"]["weight_mean"] = 99.0
    with pytest.raises(ReportUnusable, match="weight_mean_in_thresholds"):
        audit(derived_report)


def test_a_report_publishing_a_label_field_in_an_observable_is_refused(derived_report, audit):
    population = next(item for item in derived_report["populations"] if item["records"])
    population["records"][0]["observable"]["false_surface"] = True
    with pytest.raises(ReportUnusable, match=r"unexpected \['false_surface'\]"):
        audit(derived_report)


def test_a_report_with_the_wrong_face_indices_is_refused(derived_report, audit):
    population = next(item for item in derived_report["populations"] if item["records"])
    indices = population["records"][0]["face_indices"]
    population["records"][0]["face_indices"] = [index + 100 for index in indices]
    with pytest.raises(ReportUnusable, match="not the observable record's own"):
        audit(derived_report)


def test_a_report_whose_confusion_does_not_reconcile_is_refused(derived_report, audit):
    cross = derived_report["candidates"][0]["cross_tab"]
    name = next(iter(sorted(cross)))
    cross[name]["false_shells"] += 1
    with pytest.raises(ReportUnusable, match="do not add up"):
        audit(derived_report)


def test_a_report_that_removes_more_faces_than_exist_is_refused(derived_report, audit):
    aggregates = derived_report["candidates"][0]["aggregates"]
    entry = aggregates["all"]
    entry["selected_false_faces"] = entry["false_faces"] + 1
    with pytest.raises(ReportUnusable, match="more false faces"):
        audit(derived_report)


def test_a_report_missing_a_cross_tab_cell_is_refused(derived_report, audit):
    cross = derived_report["candidates"][0]["cross_tab"]
    del cross[next(iter(sorted(cross)))]
    with pytest.raises(ReportUnusable, match="cross-tab cell"):
        audit(derived_report)


def test_a_report_with_an_extra_cross_tab_cell_is_refused(derived_report, audit):
    cross = derived_report["candidates"][0]["cross_tab"]
    name = next(iter(sorted(cross)))
    cross["split:train|family:invented|cell:appliance"] = copy.deepcopy(cross[name])
    with pytest.raises(ReportUnusable, match="cross-tab cell"):
        audit(derived_report)


def test_a_duplicated_cross_tab_cell_is_refused(derived_case):
    """One object with two answers in it, which no decoded mapping could show."""
    text = json.dumps(derived_case[1], sort_keys=True)
    name = experiment.cross_tab_name(*experiment.cross_cells(RESOLUTIONS)[0])
    doubled = text.replace('"cross_tab": {', '"cross_tab": {"' + name + '": null, ', 1)
    assert doubled != text
    with pytest.raises(ReportUnusable, match="twice in one object"):
        harness.loads_report(doubled)


def test_a_duplicated_key_in_a_published_report_is_refused(derived_case):
    text = json.dumps(derived_case[1], sort_keys=True)
    doubled = "{" + '"kind": "x", ' + text[1:]
    with pytest.raises(ReportUnusable, match="twice in one object"):
        harness.loads_report(doubled)


def test_a_report_carrying_a_json_constant_is_refused(derived_case):
    text = json.dumps(derived_case[1], sort_keys=True)
    poisoned = text.replace('"material_fraction": 0.5', '"material_fraction": NaN', 1)
    assert poisoned != text
    with pytest.raises(ReportUnusable, match="JSON constant"):
        harness.loads_report(poisoned)
    poisoned = text.replace('"surface_area": 1.0', '"surface_area": Infinity', 1)
    assert poisoned != text
    with pytest.raises(ReportUnusable, match="JSON constant"):
        harness.loads_report(poisoned)


def test_a_report_with_a_forged_selection_is_refused(derived_report, audit):
    derived_report["selection"]["selected"]["rule_id"] = "single:faces<=4"
    with pytest.raises(ReportUnusable, match="does not reproduce"):
        audit(derived_report)


def test_a_report_with_a_forged_reason_is_refused(derived_report, audit):
    derived_report["selection"]["reason"] = "because it looked good"
    with pytest.raises(ReportUnusable, match="does not reproduce"):
        audit(derived_report)


def test_a_report_with_a_forged_holdout_confusion_is_refused(derived_report, audit):
    derived_report["holdout_confusion"]["selected_false_faces"] = 0
    derived_report["holdout_confusion"]["selected_false_shells"] = 0
    derived_report["holdout_confusion"]["selected_shells"] = 0
    derived_report["holdout_confusion"]["false_face_recall"] = 0.0
    derived_report["holdout_confusion"]["removes_material_false_faces"] = False
    with pytest.raises(ReportUnusable, match="does not reproduce"):
        audit(derived_report)


def test_a_report_with_a_forged_upper_bound_face_gain_is_refused(derived_report, audit):
    derived_report["upper_bound"]["false_faces_removed"] += 1
    with pytest.raises(ReportUnusable, match="false_faces_removed"):
        audit(derived_report)


def test_a_report_with_a_forged_upper_bound_accuracy_is_refused(derived_report, audit):
    derived_report["upper_bound"]["after"]["analytic_within_tolerance_fraction"] = 0.1
    with pytest.raises(ReportUnusable, match="counterfactual_analytic_accuracy_held"):
        audit(derived_report)


def test_a_counterfactual_removing_a_shell_the_rule_did_not_select_is_refused(
    derived_report, audit
):
    derived_report["counterfactuals"][0]["removed_shell_ids"] = ["somewhere|else|0"]
    with pytest.raises(ReportUnusable, match="says it removed"):
        audit(derived_report)


def test_a_claimed_gain_the_recomputed_mesh_does_not_show_is_refused(derived_report, audit):
    entry = derived_report["counterfactuals"][0]
    entry["after"]["faces"] = entry["before"]["faces"]
    with pytest.raises(ReportUnusable, match="do not reconcile"):
        audit(derived_report)


def test_a_counterfactual_hiding_the_supported_surface_it_removed_is_refused(derived_report, audit):
    derived_report["counterfactuals"][0]["removed_supported_shells"] = 1
    with pytest.raises(ReportUnusable, match="removed_supported_shells"):
        audit(derived_report)


def test_a_boolean_where_a_counterfactual_count_belongs_is_refused(derived_report, audit):
    """A measured count, published as ``true``. ``True == 1``, and the key set holds.

    Round three key-checked both accounts and then compared the whole entry
    against the re-measured mesh with ``==``, so a boolean standing in for the
    one false shell the evidence measured survived both. It is refused for
    being a boolean now, before any delta, gate or remeasurement binding reads
    it.
    """
    before = derived_report["counterfactuals"][0]["before"]
    assert before["unsupported_false_shells"] == 1
    before["unsupported_false_shells"] = True
    with pytest.raises(ReportUnusable, match="'unsupported_false_shells' is True, not an integer"):
        audit(derived_report)


def test_a_boolean_where_a_counterfactual_measurement_belongs_is_refused(derived_report, audit):
    """The real-valued half, on the other side: ``true`` for 1.0 of surface area.

    ``surface_area`` is in no delta and no gate, so nothing else in the entry
    contradicts it, and ``True == 1.0`` carried it through the exact comparison
    as well.
    """
    after = derived_report["counterfactuals"][0]["after"]
    assert after["surface_area"] == 1.0
    after["surface_area"] = True
    with pytest.raises(ReportUnusable, match="'surface_area' is True, not a number"):
        audit(derived_report)


def test_a_boolean_in_a_measured_upper_bound_account_is_refused(derived_report, audit):
    """The ceiling is measured on the same path, and typed by the same code."""
    before = derived_report["upper_bound"]["before"]
    assert before["unsupported_correct_shells"] == 1
    before["unsupported_correct_shells"] = True
    with pytest.raises(
        ReportUnusable, match="'unsupported_correct_shells' is True, not an integer"
    ):
        audit(derived_report)


def test_a_report_with_forged_work_counts_is_refused(derived_report, audit):
    derived_report["cost"]["producer_rule_calls"] += 1
    derived_report["cost"]["total_rule_calls"] += 1
    with pytest.raises(ReportUnusable, match="producer_rule_calls"):
        audit(derived_report)


def test_a_report_hiding_reader_audit_work_in_the_producer_account_is_refused(
    derived_report, audit
):
    derived_report["cost"]["reader_audit_rule_calls"] = 780
    derived_report["cost"]["total_rule_calls"] += 780
    with pytest.raises(ReportUnusable, match="reader_audit_rule_calls"):
        audit(derived_report)


def test_a_report_naming_an_undeclared_gate_is_refused(derived_report, audit):
    derived_report["gate_results"]["a_gate_the_candidate_wrote"] = True
    with pytest.raises(ReportUnusable, match="does not name"):
        audit(derived_report)


def test_a_report_with_a_forged_gate_result_is_refused(derived_report, audit):
    derived_report["gate_results"]["counterfactual_correct_faces_held"] = False
    with pytest.raises(ReportUnusable, match="gate results the reader does not reproduce"):
        audit(derived_report)


def test_a_recommendation_its_own_gates_do_not_support_is_refused(derived_report, audit):
    assert derived_report["recommendation"] == harness.PRUNE
    derived_report["recommendation"] = harness.NO_PRUNING
    with pytest.raises(ReportUnusable, match="its own selection, counterfactuals and gates"):
        audit(derived_report)


def test_a_report_with_a_candidate_digest_that_is_not_the_frozen_one_is_refused(
    derived_report, audit
):
    derived_report["corpus"]["candidate_sha256"] = "0" * 64
    with pytest.raises(ReportUnusable, match="frozen at"):
        audit(derived_report)


def test_a_report_whose_sealed_list_is_not_its_holdout_is_refused(derived_report, audit):
    derived_report["corpus"]["sealed_holdout"] = ["outliers-a"]
    with pytest.raises(ReportUnusable, match="seals"):
        audit(derived_report)


def test_a_report_with_a_member_seed_that_is_not_its_own_is_refused(derived_report, audit):
    derived_report["corpus"]["members"][1]["seed"] = "shell-discriminant:somebody-else"
    with pytest.raises(ReportUnusable, match="seed"):
        audit(derived_report)


def test_a_report_omitting_a_population_is_refused(derived_report, audit):
    derived_report["populations"] = [
        item for item in derived_report["populations"] if item["cell_id"] != RESOLUTIONS[1]
    ]
    with pytest.raises(ReportUnusable, match="the committed corpus builds"):
        audit(derived_report)


def test_a_report_publishing_one_realization_twice_is_refused(derived_report, audit):
    first = copy.deepcopy(derived_report["realizations"][0])
    second = copy.deepcopy(derived_report["realizations"][1])
    second["reconstruction_digest"] = first["reconstruction_digest"]
    derived_report["realizations"][1] = second
    with pytest.raises(ReportUnusable, match="is not the realization"):
        audit(derived_report)


def test_a_report_whose_totals_do_not_reconcile_is_refused(derived_report, audit):
    derived_report["totals"]["false_faces"] += 1
    with pytest.raises(ReportUnusable, match=r"totals\['false_faces'\]"):
        audit(derived_report)


# --- the schemas, the identities and the work account ---------------------------------------
#
# A schema field is a claim about which validator a value belongs to and a unit
# is a claim about what its numbers mean, so both are established rather than
# read past. The typed accessors matter for the same reason: in Python
# ``True == 1``, so an untyped comparison against an integer schema version
# accepts a boolean, and ``str(item)`` turns any object at all into a string
# that might compare equal to the right one.


def populated_of(document):
    """The first published population that carries labelled records."""
    return next(item for item in document["populations"] if item["records"])


def test_a_boolean_where_a_top_level_integer_belongs_is_refused(derived_report, audit):
    """``True == 1``, so an untyped comparison accepts a boolean schema version."""
    derived_report["schema_version"] = True
    with pytest.raises(ReportUnusable, match="not an integer"):
        audit(derived_report)


def test_a_boolean_where_a_nested_integer_belongs_is_refused(derived_report, audit):
    populated_of(derived_report)["records"][0]["observable"]["schema_version"] = True
    with pytest.raises(ReportUnusable, match="not an integer"):
        audit(derived_report)


def test_a_list_item_that_is_not_text_is_refused(derived_report, audit):
    """``[str(item) for item in ...]`` would have repaired this into a string."""
    derived_report["corpus"]["resolutions"] = [experiment.SELECTION_RESOLUTION, 7]
    with pytest.raises(ReportUnusable, match="not text"):
        audit(derived_report)


def test_a_population_kind_that_is_not_the_declared_one_is_refused(derived_report, audit):
    derived_report["populations"][0]["kind"] = harness.POPULATION_KIND
    with pytest.raises(ReportUnusable, match="not a shell_discriminant_population"):
        audit(derived_report)


def test_an_observable_kind_that_is_not_the_declared_one_is_refused(derived_report, audit):
    populated_of(derived_report)["closed_shells_observable"][0]["kind"] = harness.LABEL_KIND
    with pytest.raises(ReportUnusable, match="not a shell_discriminant_observable"):
        audit(derived_report)


def test_a_label_kind_that_is_not_the_declared_one_is_refused(derived_report, audit):
    populated_of(derived_report)["records"][0]["label"]["kind"] = harness.OBSERVABLE_KIND
    with pytest.raises(ReportUnusable, match="not a shell_discriminant_label"):
        audit(derived_report)


def test_a_record_kind_that_is_not_the_declared_one_is_refused(derived_report, audit):
    populated_of(derived_report)["records"][0]["kind"] = harness.POPULATION_RECORD_KIND
    with pytest.raises(ReportUnusable, match="not a shell_discriminant_record"):
        audit(derived_report)


def test_a_confusion_kind_that_is_not_the_declared_one_is_refused(derived_report, audit):
    cross = derived_report["candidates"][0]["cross_tab"]
    cross[next(iter(sorted(cross)))]["kind"] = harness.RECORD_KIND
    with pytest.raises(ReportUnusable, match="not a shell_discriminant_confusion"):
        audit(derived_report)


def test_a_selection_kind_that_is_not_the_declared_one_is_refused(derived_report, audit):
    derived_report["selection"]["kind"] = harness.RECORD_KIND
    with pytest.raises(ReportUnusable, match="'selection' is kind"):
        audit(derived_report)


def test_a_counterfactual_kind_that_is_not_the_declared_one_is_refused(derived_report, audit):
    derived_report["counterfactuals"][0]["kind"] = harness.RECORD_KIND
    with pytest.raises(ReportUnusable, match="not a shell_discriminant_counterfactual"):
        audit(derived_report)


def test_an_observable_unit_that_is_not_the_declared_one_is_refused(derived_report, audit):
    populated_of(derived_report)["closed_shells_observable"][0]["unit"] = experiment.COUNT_UNIT
    with pytest.raises(ReportUnusable, match="unit that is not the module's"):
        audit(derived_report)


def test_a_confusion_unit_that_is_not_the_declared_one_is_refused(derived_report, audit):
    cross = derived_report["candidates"][0]["cross_tab"]
    cross[next(iter(sorted(cross)))]["unit"] = experiment.COUNT_UNIT
    with pytest.raises(ReportUnusable, match="unit that is not the module's"):
        audit(derived_report)


def test_a_nested_schema_version_that_is_not_the_declared_one_is_refused(derived_report, audit):
    populated_of(derived_report)["records"][0]["label"]["schema_version"] = 2
    with pytest.raises(ReportUnusable, match="label schema"):
        audit(derived_report)


def test_a_population_publishing_one_closed_shell_identity_twice_is_refused(derived_report, audit):
    shells = populated_of(derived_report)["closed_shells_observable"]
    shells[1]["shell_id"] = shells[0]["shell_id"]
    with pytest.raises(ReportUnusable, match="one closed shell identity twice"):
        audit(derived_report)


def test_every_cost_field_is_recomputed_from_the_evidence(derived_report, audit):
    """Not the producer search alone. Every line of the account, one at a time.

    Round two recomputed ``producer_rule_calls`` and took the voxel visits, the
    triangle samples, the counterfactual calls, the totals and the remeasurement
    count on trust, so raising a pair of them together stayed coherent and
    passed.
    """
    for field in harness.COST_KEYS:
        if field == "note":
            continue
        forged = copy.deepcopy(derived_report)
        forged["cost"][field] = forged["cost"][field] + 1
        with pytest.raises(ReportUnusable, match=field):
            audit(forged)


def test_a_rewritten_cost_note_is_refused(derived_report, audit):
    """A note rewritten to fit its own numbers explains nothing."""
    derived_report["cost"]["note"] = "a deterministic work account, not a wall clock"
    with pytest.raises(ReportUnusable, match="note that is not the module's"):
        audit(derived_report)


# --- the counterfactual list is an exact ordered set, not a sample ---------------------------


@pytest.fixture(scope="module")
def supported_only_case():
    """A rebuild whose second sealed holdout population carries no labelled shell."""
    bundle, document = build_case(
        members=SUPPORTED_ONLY_MEMBERS,
        sealed=SUPPORTED_ONLY_SEALED,
        unlabelled=SUPPORTED_ONLY,
    )
    harness._validate_document(document, bundle, what=DERIVED)
    return bundle, document


@pytest.fixture
def supported_only_report(supported_only_case):
    return copy.deepcopy(supported_only_case[1])


@pytest.fixture
def supported_only_audit(supported_only_case):
    bundle = supported_only_case[0]

    def _audit(document):
        return harness._validate_document(document, bundle, what=DERIVED)

    return _audit


def test_a_holdout_population_of_only_supported_shells_still_owes_a_counterfactual(
    supported_only_case, supported_only_report, supported_only_audit
):
    """Round two's exact escape hatch, closed.

    ``sealed-b`` is a sealed holdout member whose appliance population is one
    real closed shell that the evaluator never labelled. The selected rule fires
    on it, so removing it is damage an owner would see, and the recommendation
    is not allowed to be made without measuring it. Round two decided the
    population's split from ``here[0].split if here else None`` - the split of
    its first LABELLED record - so a population with none of them had no split,
    failed the holdout test and was silently excused.
    """
    document = supported_only_case[1]
    supported = next(
        item
        for item in document["populations"]
        if (item["member_id"], item["cell_id"]) == ("sealed-b", experiment.SELECTION_RESOLUTION)
    )
    assert supported["closed_shells"] == 1
    assert supported["unsupported_closed_shells"] == 0
    assert [(item["member_id"], item["cell_id"]) for item in document["counterfactuals"]] == [
        ("sealed-a", experiment.SELECTION_RESOLUTION),
        ("sealed-b", experiment.SELECTION_RESOLUTION),
    ]
    published = supported_only_report["counterfactuals"]
    supported_only_report["counterfactuals"] = [
        item for item in published if item["member_id"] != "sealed-b"
    ]
    with pytest.raises(ReportUnusable, match="sealed-b/appliance"):
        supported_only_audit(supported_only_report)


def test_one_closed_shell_identity_in_two_populations_is_refused(
    supported_only_report, supported_only_audit
):
    """Uniqueness over EVERY closed shell, not only the labelled ones."""
    published = supported_only_report["populations"]
    populations = {(item["member_id"], item["cell_id"]): item for item in published}
    borrowed = populations[("outliers-a", experiment.SELECTION_RESOLUTION)]
    here = populations[("sealed-b", experiment.SELECTION_RESOLUTION)]
    here["closed_shells_observable"][0]["shell_id"] = borrowed["closed_shells_observable"][0][
        "shell_id"
    ]
    with pytest.raises(ReportUnusable, match="whether or not it is labelled"):
        supported_only_audit(supported_only_report)


def test_a_duplicated_counterfactual_is_refused(derived_report, audit):
    derived_report["counterfactuals"].append(copy.deepcopy(derived_report["counterfactuals"][0]))
    with pytest.raises(ReportUnusable, match="counting it twice counts its gain twice"):
        audit(derived_report)


def test_a_counterfactual_on_a_training_population_is_refused(derived_report, audit):
    extra = copy.deepcopy(derived_report["counterfactuals"][0])
    extra["member_id"] = "outliers-a"
    derived_report["counterfactuals"].append(extra)
    with pytest.raises(ReportUnusable, match="'train' split"):
        audit(derived_report)


def test_a_counterfactual_at_the_diagnostic_resolution_is_refused(derived_report, audit):
    extra = copy.deepcopy(derived_report["counterfactuals"][0])
    extra["cell_id"] = RESOLUTIONS[1]
    derived_report["counterfactuals"].append(extra)
    with pytest.raises(ReportUnusable, match="diagnostic resolution"):
        audit(derived_report)


def test_an_invented_counterfactual_population_is_refused(derived_report, audit):
    extra = copy.deepcopy(derived_report["counterfactuals"][0])
    extra["member_id"] = "somewhere-else"
    derived_report["counterfactuals"].append(extra)
    with pytest.raises(ReportUnusable, match="does not build"):
        audit(derived_report)


def test_an_invented_counterfactual_measurement_is_refused(derived_report, audit):
    """A before/after pair that agrees with itself perfectly, and with no mesh.

    ``surface_area`` is in neither a delta nor a gate, so changing it on both
    sides leaves the entry entirely self-consistent: the removed shells, the
    face counts, the deltas, the twelve gates and ``passed`` all still hold.
    The reader refuses it anyway, because it re-measures the pruned mesh and
    that is not what the mesh says.
    """
    entry = derived_report["counterfactuals"][0]
    entry["before"]["surface_area"] = 99.0
    entry["after"]["surface_area"] = 99.0
    with pytest.raises(ReportUnusable, match="is not the counterfactual"):
        audit(derived_report)


def test_an_integer_too_large_to_be_a_measurement_is_refused(derived_report, audit):
    """A JSON integer has no width limit, and ``float()`` on one raises.

    A reader refuses; it does not crash with an ``OverflowError`` a caller
    catching report errors would never see.
    """
    derived_report["corpus"]["material_fraction"] = 10**400
    with pytest.raises(ReportUnusable, match="too large to be a measurement"):
        audit(derived_report)


def test_an_integer_too_large_to_be_a_coordinate_is_refused(derived_report, audit):
    populated_of(derived_report)["closed_shells_observable"][0]["extent"] = [10**400, 0.0, 0.0]
    with pytest.raises(ReportUnusable, match="too large to be a coordinate"):
        audit(derived_report)


def test_a_boolean_where_the_weight_ratio_belongs_is_refused(derived_report, audit):
    """``True == 1.0``, so this field needs a typed accessor like every other."""
    populated_of(derived_report)["closed_shells_observable"][0]["weight_mean_in_thresholds"] = True
    with pytest.raises(ReportUnusable, match="not a number"):
        audit(derived_report)


def test_an_unavailable_item_that_is_not_text_is_refused(derived_report, audit):
    """Refused where it is read, rather than reshaped into ``'1'`` and refused later."""
    populated_of(derived_report)["closed_shells_observable"][0]["unavailable"] = [1]
    with pytest.raises(ReportUnusable, match="not text"):
        audit(derived_report)


def test_a_boolean_threshold_in_a_published_rule_is_refused(derived_report, audit):
    """Two declared thresholds are exactly 0.0 and 1.0, which a boolean equals."""
    derived_report["candidates"][0]["rule"]["tests"][0]["threshold"] = True
    with pytest.raises(ReportUnusable, match="not a number"):
        audit(derived_report)


def test_a_boolean_in_the_selection_block_is_refused(derived_report, audit):
    derived_report["selection"]["training_shells"] = True
    with pytest.raises(ReportUnusable, match="not an integer"):
        audit(derived_report)


def test_a_gate_result_that_is_not_a_boolean_is_refused(derived_report, audit):
    derived_report["gate_results"]["counterfactual_false_faces_fell"] = 1
    with pytest.raises(ReportUnusable, match="not a boolean"):
        audit(derived_report)


# --- the committed corpus, run once ------------------------------------------------------
#
# The full producer over the shipped fixtures. It is slow - minutes, dominated
# by fifty-four fusions - and it runs in the ORDINARY pytest job on the existing
# public GitHub-hosted workflow, beside everything above. It is here because the
# facts it establishes are facts about the committed corpus rather than about
# the reader: a fixture edit that quietly emptied one class out of the sealed
# holdout, or pushed the bounded pass over the declared ceiling, would otherwise
# only show up when somebody happened to run the command by hand.


@pytest.fixture(scope="module")
def sealed_run():
    return harness.run()


def test_the_committed_corpus_produces_the_declared_sealed_population(sealed_run):
    totals = sealed_run["totals"]
    assert totals["members"] == 27
    assert totals["resolutions"] == 2
    assert totals["populations"] == 54
    assert totals["failures"] == 0
    assert sealed_run["decision"] == "pass"
    corpus = sealed_run["corpus"]
    assert corpus["candidate_sha256"] == experiment.FROZEN_CANDIDATE_DIGEST
    assert corpus["selection_resolution"] == experiment.SELECTION_RESOLUTION
    assert len(corpus["sealed_holdout"]) == 8


def test_the_sealed_appliance_holdout_carries_both_classes(sealed_run):
    """The refusal this run would raise if it did not, made visible as an assertion.

    ``run`` refuses a sealed population that cannot grade both the gain and the
    damage. Asserting it here says out loud what that refusal protects, and it
    is the one fact a holdout has to have before any number taken on it means
    anything.
    """
    holdout = [
        population
        for population in sealed_run["populations"]
        if population["split"] == experiment.HOLDOUT
        and population["cell_id"] == experiment.SELECTION_RESOLUTION
    ]
    assert sum(population["false_faces"] for population in holdout) > 0
    assert sum(population["correct_faces"] for population in holdout) > 0


def test_the_exact_control_produces_no_unsupported_closed_shell(sealed_run):
    control = [
        population
        for population in sealed_run["populations"]
        if population["member_id"] == "exact-control"
    ]
    assert control
    assert all(population["unsupported_closed_shells"] == 0 for population in control)


def test_the_committed_run_pays_for_exactly_one_bounded_pass(sealed_run):
    cost = sealed_run["cost"]
    expected = len(experiment.RULE_GRID) * sealed_run["totals"]["shells"]
    assert cost["producer_rule_calls"] == expected
    assert cost["decision_matrix_cells"] == expected
    assert expected <= experiment.WORK_CEILING
    assert cost["unaccounted_rule_calls"] == 0
    assert cost["reader_audit_rule_calls"] == 0
    assert cost["total_rule_calls"] == (
        cost["producer_rule_calls"] + cost["counterfactual_rule_calls"]
    )


def test_no_reconstruction_input_appears_twice_in_the_committed_run(sealed_run):
    digests = [item["reconstruction_digest"] for item in sealed_run["realizations"]]
    assert len(set(digests)) == len(digests)
    assert len(digests) == sealed_run["totals"]["populations"]


def test_the_committed_run_recommendation_follows_from_its_own_gates(sealed_run):
    selected = sealed_run["selection"]["selected"]
    if selected is None:
        assert sealed_run["recommendation"] == harness.NO_PRUNING
        assert sealed_run["counterfactuals"] == []
        assert sealed_run["holdout_confusion"] is None
    else:
        passed = all(sealed_run["gate_results"].get(gate) for gate in experiment.GATES)
        expected = harness.PRUNE if passed and sealed_run["counterfactuals"] else harness.NO_PRUNING
        assert sealed_run["recommendation"] == expected


def test_the_committed_artifact_survives_the_strict_reader_from_its_own_text(sealed_run, tmp_path):
    path = tmp_path / "shell-discriminant-report.json"
    publish(sealed_run, path)
    assert harness.read_report(path)["kind"] == harness.REPORT_KIND


# --- fifteen coherent forgeries of the COMMITTED artifact ------------------------------------
#
# Round two's reviewer mutated the published nine-megabyte artifact fifteen
# ways, kept each altered claim internally coherent where one was needed, and
# handed the result to the public strict reader. All fifteen were accepted,
# because that reader checked a report against its own claims rather than
# against the corpus. Each one is a test now, over the real artifact and through
# the real public boundary, and each is refused for its own reason.


@pytest.fixture
def sealed_report(sealed_run):
    """A private copy of the committed artifact, so a forgery reaches no other test."""
    return copy.deepcopy(sealed_run)


def sealed_population(document):
    """The first committed population that carries labelled records."""
    return next(item for item in document["populations"] if item["records"])


def published_copies(population, shell_id):
    """BOTH published copies of one closed shell: the population's, and the record's.

    A forgery that changed only one of them would be refused for disagreeing
    with itself, which is a weaker control than the one being tested.
    """
    shells = population["closed_shells_observable"]
    out = [item for item in shells if item["shell_id"] == shell_id]
    out += [
        record["observable"]
        for record in population["records"]
        if record["observable"]["shell_id"] == shell_id
    ]
    assert len(out) == 2, out
    return out


def test_a_forged_manifest_digest_is_refused(sealed_report):
    """1. Another syntactically valid digest, of nothing this corpus contains."""
    sealed_report["corpus"]["manifest_sha256"] = digest_of("another manifest")
    with pytest.raises(ReportUnusable, match="manifest_sha256"):
        harness.validate_report(sealed_report)


def test_a_forged_population_digest_is_refused(sealed_report):
    """2. The same attack on the population recipe."""
    sealed_report["corpus"]["population_sha256"] = digest_of("another population")
    with pytest.raises(ReportUnusable, match="population_sha256"):
        harness.validate_report(sealed_report)


def test_a_forged_base_case_name_is_refused(sealed_report):
    """3. The scene geometry every member is built from, renamed."""
    sealed_report["corpus"]["base_case"] = "somewhere-else.json"
    with pytest.raises(ReportUnusable, match="base_case"):
        harness.validate_report(sealed_report)


def test_a_forged_member_frame_digest_is_refused(sealed_report):
    """4. A member's frames, replaced by another valid digest."""
    sealed_report["corpus"]["members"][-1]["frame_digest"] = digest_of("another frame set")
    with pytest.raises(ReportUnusable, match="frame digest"):
        harness.validate_report(sealed_report)


def test_a_forged_member_perturbation_is_refused(sealed_report):
    """5. A sealed magnitude, moved after the fact."""
    for member in sealed_report["corpus"]["members"]:
        for key, value in sorted(member["perturbation"].items()):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            member["perturbation"][key] = value + 1
            with pytest.raises(ReportUnusable, match="perturbation parameters"):
                harness.validate_report(sealed_report)
            return
    pytest.fail("no committed member declares a numeric perturbation parameter")


def test_a_forged_population_kind_is_refused(sealed_report):
    """6. A population wearing another validator's name."""
    sealed_report["populations"][0]["kind"] = harness.POPULATION_KIND
    with pytest.raises(ReportUnusable, match="not a shell_discriminant_population"):
        harness.validate_report(sealed_report)


def test_coherently_forged_voxel_work_is_refused(sealed_report):
    """7. A population's voxel work and the aggregate, raised together."""
    sealed_report["populations"][0]["work_units"] += 1000
    sealed_report["cost"]["voxel_visits"] += 1000
    with pytest.raises(ReportUnusable, match="work_units"):
        harness.validate_report(sealed_report)


def test_coherently_forged_triangle_work_is_refused(sealed_report):
    """8. The same attack on the triangles every mesh was scored at."""
    sealed_report["populations"][0]["triangle_samples"] += 500
    sealed_report["cost"]["triangle_samples"] += 500
    with pytest.raises(ReportUnusable, match="triangle_samples"):
        harness.validate_report(sealed_report)


def test_coherently_forged_counterfactual_work_is_refused(sealed_report):
    """9. Rule calls this run never made, with the total moved to match."""
    sealed_report["cost"]["counterfactual_rule_calls"] += 776
    sealed_report["cost"]["total_rule_calls"] += 776
    with pytest.raises(ReportUnusable, match="counterfactual_rule_calls"):
        harness.validate_report(sealed_report)


def test_a_fabricated_gain_on_an_unmeasured_upper_bound_is_refused(sealed_report):
    """10. The unmeasured ceiling is six keys, and it is not an open namespace."""
    assert sealed_report["upper_bound"]["measured"] is False
    sealed_report["upper_bound"]["false_faces_removed"] = 999
    with pytest.raises(ReportUnusable, match=r"unexpected \['false_faces_removed'\]"):
        harness.validate_report(sealed_report)


def test_coherently_forged_shell_weights_are_refused(sealed_report):
    """11. Every weight and the extraction threshold doubled: the ratio survives."""
    population = sealed_population(sealed_report)
    shells = population["closed_shells_observable"]
    shell = next(item for item in shells if item["weight_mean"] is not None)
    for entry in published_copies(population, shell["shell_id"]):
        for key in ("weight_min", "weight_mean", "weight_max", "min_weight"):
            entry[key] = entry[key] * 2.0
    with pytest.raises(ReportUnusable, match="does not publish the closed shells"):
        harness.validate_report(sealed_report)


def test_coherently_forged_ray_counts_are_refused(sealed_report):
    """12. Vertices and every ray count doubled: the partition and the fractions survive."""
    population = sealed_population(sealed_report)
    shell = population["closed_shells_observable"][0]
    for entry in published_copies(population, shell["shell_id"]):
        entry["vertices"] *= 2
        for key in (
            "ray_probes",
            "confirming_probes",
            "contradicting_probes",
            "occluded_probes",
            "no_return_probes",
            "outside_window_probes",
        ):
            entry[key] *= 2
    with pytest.raises(ReportUnusable, match="does not publish the closed shells"):
        harness.validate_report(sealed_report)


def test_a_fabricated_reconstruction_identity_is_refused(sealed_report):
    """13. A whole realization renamed, coherently, in all three places."""
    forged = digest_of("another reconstruction")
    population = sealed_population(sealed_report)
    where = (population["member_id"], population["cell_id"])
    population["reconstruction_digest"] = forged
    for record in population["records"]:
        record["reconstruction_digest"] = forged
    for realization in sealed_report["realizations"]:
        if (realization["member_id"], realization["cell_id"]) == where:
            realization["reconstruction_digest"] = forged
    with pytest.raises(ReportUnusable, match="reconstruction_digest"):
        harness.validate_report(sealed_report)


def test_an_invented_extra_realization_is_refused(sealed_report):
    """14. One more realization than anything was reconstructed for."""
    extra = copy.deepcopy(sealed_report["realizations"][0])
    extra["member_id"] = "invented-member"
    sealed_report["realizations"].append(extra)
    with pytest.raises(ReportUnusable, match="extra realization"):
        harness.validate_report(sealed_report)


def test_a_reordered_member_list_is_refused(sealed_report):
    """15. The declared order is part of the seal, not a presentation choice."""
    members = sealed_report["corpus"]["members"]
    members[0], members[1] = members[1], members[0]
    with pytest.raises(ReportUnusable, match="member ORDER is part of the seal"):
        harness.validate_report(sealed_report)


def test_a_swapped_predeclared_upper_bound_member_is_refused(sealed_report):
    """The predeclaration binds whether or not it flatters the result."""
    other = next(
        member["member_id"]
        for member in sealed_report["corpus"]["members"]
        if member["member_id"] != sealed_report["upper_bound"]["member_id"]
    )
    sealed_report["corpus"]["upper_bound"]["member_id"] = other
    sealed_report["upper_bound"]["member_id"] = other
    with pytest.raises(ReportUnusable, match="is not a predeclaration"):
        harness.validate_report(sealed_report)


def test_an_unmeasured_upper_bound_claiming_to_be_measured_is_refused(sealed_report):
    """The form follows the predeclared population, never a boolean in the report."""
    assert sealed_report["upper_bound"]["measured"] is False
    sealed_report["upper_bound"]["measured"] = True
    with pytest.raises(ReportUnusable, match="not measurable"):
        harness.validate_report(sealed_report)


def test_a_rewritten_unmeasured_upper_bound_reason_is_refused(sealed_report):
    """Why a ceiling could not be measured is a fact about the member, not prose."""
    sealed_report["upper_bound"]["why"] = "there was nothing worth measuring"
    with pytest.raises(ReportUnusable, match="unmeasured ceiling"):
        harness.validate_report(sealed_report)


def test_a_forged_committed_failure_list_is_refused(sealed_report):
    """A failure list is evidence about the run, and a clean one is a claim."""
    sealed_report["failures"] = ["outliers-140-p17: something went wrong"]
    sealed_report["decision"] = "fail"
    with pytest.raises(ReportUnusable, match="publishes failures"):
        harness.validate_report(sealed_report)


# --- the type confusions an exact comparison inherits ----------------------------------------
#
# Binding a published value to the rebuilt one closes a coherent forgery only
# as far as ``==`` distinguishes the two, and in Python it does not distinguish
# ``False`` from ``0`` or ``True`` from ``1`` and ``1.0``. The corpus block was
# bound exactly and its perturbations were still not established, because a
# boolean published where a sealed magnitude belongs compares equal to that
# magnitude. These run over the real committed artifact through the real
# public boundary, like the fifteen above.


def exact_control_perturbation(document):
    """The exact control's declared perturbation: every sealed magnitude is zero.

    Which makes it the member these are cheapest to mount on. ``False == 0``
    and ``False == 0.0``, so each mutation below compares equal to what the
    committed corpus declares and survives an exact comparison against it.
    """
    member = next(
        item for item in document["corpus"]["members"] if item["member_id"] == "exact-control"
    )
    return member["perturbation"]


def test_a_boolean_where_a_committed_perturbation_code_belongs_is_refused(sealed_report):
    """``false`` where the committed corpus declares the integer ``0``.

    The attack the round-three reader accepted: it compared the perturbation
    mapping exactly and did nothing else with it, and
    ``{"depth_noise_codes": False} == {"depth_noise_codes": 0}``. A member is
    its declaration, and a boolean is not the depth noise this corpus sealed.
    """
    perturbation = exact_control_perturbation(sealed_report)
    assert perturbation["depth_noise_codes"] == 0
    perturbation["depth_noise_codes"] = False
    with pytest.raises(ReportUnusable, match="'depth_noise_codes' is False, not an integer"):
        harness.validate_report(sealed_report)


def test_a_boolean_where_a_committed_pose_magnitude_belongs_is_refused(sealed_report):
    """The same confusion against a real-valued field, where ``False == 0.0``."""
    perturbation = exact_control_perturbation(sealed_report)
    assert perturbation["rotation_bias_degrees"] == 0.0
    perturbation["rotation_bias_degrees"] = False
    with pytest.raises(ReportUnusable, match="'rotation_bias_degrees' is False, not a number"):
        harness.validate_report(sealed_report)


def test_a_boolean_inside_a_committed_perturbation_vector_is_refused(sealed_report):
    """And inside a three-vector: ``[0.0, True, 0.0]`` is the declared axis under ``==``."""
    perturbation = exact_control_perturbation(sealed_report)
    assert perturbation["rotation_axis"] == [0.0, 1.0, 0.0]
    perturbation["rotation_axis"] = [0.0, True, 0.0]
    with pytest.raises(ReportUnusable, match=r"'rotation_axis'\[1\] is True"):
        harness.validate_report(sealed_report)


def test_a_committed_perturbation_missing_a_declared_field_is_refused(sealed_report):
    """The perturbation key set is exact, and it is established before any value is."""
    perturbation = exact_control_perturbation(sealed_report)
    del perturbation["band_shift"]
    with pytest.raises(ReportUnusable, match=r"Missing \['band_shift'\]"):
        harness.validate_report(sealed_report)
