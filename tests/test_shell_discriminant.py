"""The wall between the two namespaces, and every way somebody could get around it.

These are the contract tests for :mod:`tests.shell_discriminant`: they run in
seconds, they need no corpus artifact, and they exist to attack the experiment
rather than to demonstrate it. The corpus-level evidence -
:mod:`tests.test_shell_discriminant_harness` - is a separate module, because a
control that only fails when a long run finishes is a control nobody runs while
they are working.

Round two moved the truth wall one level down, and these tests moved with it.
The derivation now takes exactly one :class:`~tests.shell_discriminant.ObservableInputs`
and the evaluator's half is a separate function, so the leak controls here no
longer ask whether a function happened not to read the scene: they make every
truth and true-pose accessor RAISE and prove the derivation still produces
byte-identical observables, and they hand the inputs value an analytic object
under every field name it has.

Every adverse control the round-two packet named has a test here or in the
harness module, and each one asserts the SPECIFIC refusal rather than merely
that something raised: a truth-fed selector, a rule subclass, an overridden
``holds``, an adaptive threshold, a swapped label, a forged face set, a
reconstruction identity that ignores half its inputs, a second decision-matrix
pass hiding inside a summary, a diagnostic-resolution record qualifying a rule,
a missing cross-tab cell, and a counterfactual that removes surface nothing in
the labelled population accounts for.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import pytest

from scanner3d.geometry import Rigid
from scanner3d.mesh import edge_census
from tests import shell_discriminant as experiment
from tests import surface_oracle as contract
from tests import surface_oracle_harness as oracle

#: The coarsest cell the accepted corpus declares. Every test here that needs a
#: real fused mesh uses it rather than the appliance arm: the appliance cell is
#: two orders of magnitude finer and nothing in this module is asking a question
#: about resolution.
CELL = "v25-t3-w1-uniform"

#: The second declared resolution of the committed corpus. Diagnostic, and the
#: selection tests below prove it can never qualify a candidate.
DIAGNOSTIC = "v10-t4-w1-uniform"

RESOLUTIONS = (experiment.SELECTION_RESOLUTION, DIAGNOSTIC)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def observable(**overrides):
    """A syntactically valid observable record, for the controls to deform."""
    base = experiment.ObservableShell(
        shell_id="0123456789abcdef|0",
        voxel_size=0.1,
        shell_index=0,
        face_indices=tuple(range(8)),
        faces=8,
        vertices=6,
        area=0.01,
        extent=(0.1, 0.1, 0.1),
        diagonal=0.17,
        centroid=(0.0, 0.0, 0.0),
        views_total=5,
        views_containing=5,
        views_touching=5,
        ray_probes=30,
        confirming_probes=6,
        contradicting_probes=12,
        occluded_probes=9,
        no_return_probes=3,
        outside_window_probes=0,
        ray_tolerance=0.2,
        measured_returns=1000,
        nearest_measured_distance=0.05,
        farthest_measured_distance=0.2,
        nearest_measured_capped=False,
        weight_min=1.0,
        weight_mean=2.0,
        weight_max=3.0,
        min_weight=1.0,
        unavailable=(),
    )
    return replace(base, **overrides) if overrides else base


def truth_for(shell, *, false_surface):
    return experiment.ShellTruth(
        shell_id=shell.shell_id,
        false_surface=false_surface,
        worst_analytic_distance=0.4 if false_surface else 0.02,
        accuracy_tolerance=0.15,
        faces=shell.faces,
    )


def record(
    *,
    false_surface,
    split=experiment.TRAIN,
    member_id="train-a",
    family=contract.DEPTH_OUTLIERS,
    cell_id=experiment.SELECTION_RESOLUTION,
    reconstruction_digest=DIGEST_A,
    shell=None,
    **overrides,
):
    """One evaluator record over a hand-built observable."""
    shell = observable(**overrides) if shell is None else shell
    return experiment.ShellRecord(
        observable=shell,
        truth=truth_for(shell, false_surface=false_surface),
        member_id=member_id,
        family=family,
        cell_id=cell_id,
        split=split,
        reconstruction_digest=reconstruction_digest,
        face_indices=shell.face_indices,
    )


def contradicted(shell_id, *, fraction_high):
    """A shell the raw rays mostly see through, or mostly confirm."""
    return observable(
        shell_id=shell_id,
        confirming_probes=3 if fraction_high else 24,
        contradicting_probes=24 if fraction_high else 3,
        occluded_probes=2,
        no_return_probes=1,
        outside_window_probes=0,
    )


def _with(value, **overrides):
    """``dataclasses.replace`` behind an untyped door, so a control may lie."""
    return replace(value, **overrides)


def selects(rule, shells):
    """``select`` behind an untyped door, so a control may hand it anything at all."""
    return experiment.select(rule, shells)


def boom(*_args, **_kwargs):
    raise AssertionError("the observable derivation reached the evaluator's own truth")


# --- one real fused mesh, shared by everything that needs geometry ---------------------


@pytest.fixture(scope="module")
def fused_case():
    """One real fused mesh from the accepted corpus, at its cheapest declared cell.

    The accepted ``combined-realistic`` case rather than a mesh built here: the
    counterfactual and identity tests are about geometry the accepted oracle
    produced - it carries twelve components and two CLOSED shells at this cell,
    which is what makes "that face set is another component" a real control -
    and a hand-built triangle soup would prove nothing about either.
    """
    corpus = oracle.load_corpus()
    entry = next(item for item in corpus.files if item.case_id == "combined-realistic")
    case = oracle.load_case(oracle.CORPUS / entry.name, entry=entry, corpus=corpus)
    scene = case.scene
    cell = scene.cell(CELL)
    volume, _, _ = contract.fuse_keeping_volume(scene, cell)
    return scene, contract.fused(scene, CELL), cell, volume


@pytest.fixture(scope="module")
def derived(fused_case):
    """The observable inputs of that mesh, and every closed shell derived from them."""
    scene, result, cell, volume = fused_case
    inputs = experiment.observable_inputs(
        scene, result, cell=cell, volume=volume, backend=volume.backend
    )
    return inputs, experiment.derive_observables(inputs)


# --- 1. the derivation is truth-free, structurally rather than by convention ------------


def test_the_observable_derivation_survives_every_truth_accessor_raising(
    fused_case, derived, monkeypatch
):
    """The leak control, and it is not a reading of the source.

    Every analytic accessor the evaluator uses is replaced with something that
    raises - the scene's shapes, both truth frames, the true pose, the ideal
    support, the analytic distance, the surface scoring and the coverage mask -
    and the derivation is asked to run again over the same inputs. It produces
    byte-identical observables, because the inputs value it was given has
    nowhere to reach any of them from.
    """
    inputs, before = derived
    scene = fused_case[0]
    monkeypatch.setattr(contract.Scene, "shapes", property(boom))
    monkeypatch.setattr(contract.Scene, "truth_codes", boom)
    monkeypatch.setattr(contract.ViewEvidence, "truth", property(boom))
    monkeypatch.setattr(contract.ViewEvidence, "true_pose", property(boom))
    for name in (
        "truth_points",
        "truth_surface",
        "analytic_distance",
        "score_surface",
        "_covered_mask",
    ):
        monkeypatch.setattr(contract, name, boom)
    # The patches really do bite, or the assertion below would prove nothing.
    with pytest.raises(AssertionError):
        _ = scene.shapes
    with pytest.raises(AssertionError):
        _ = scene.views[0].true_pose
    with pytest.raises(AssertionError):
        contract.truth_points(scene)

    after = experiment.derive_observables(inputs)
    assert [shell.as_dict() for shell in after] == [shell.as_dict() for shell in before]
    assert after


def test_the_observable_inputs_hold_no_analytic_object(derived):
    """Neither in their declared types nor in the values one actually carries."""
    inputs, _ = derived
    named = {field.name: str(field.type) for field in fields(experiment.ObservableInputs)}
    leaked = [
        name
        for name, declared in named.items()
        if any(
            word in declared
            for word in ("Scene", "ViewEvidence", "SurfaceScoring", "Perturbation", "Regions")
        )
    ]
    assert not leaked, f"the appliance's inputs declare {leaked}"
    forbidden = (contract.Scene, contract.ViewEvidence, contract.SurfaceScoring)
    for name in experiment.ObservableInputs.__slots__:
        value = getattr(inputs, name)
        items = value if isinstance(value, tuple) else (value,)
        for item in items:
            assert not isinstance(item, forbidden), f"{name} reaches a {type(item).__name__}"
            assert not isinstance(item, contract.Perturbation), name


def test_the_derivation_refuses_an_inputs_value_that_reaches_the_evaluator(fused_case, derived):
    """A Scene, a ViewEvidence, a scoring or a perturbation under ANY field name."""
    inputs, _ = derived
    scene, result, _, _ = fused_case
    with pytest.raises(experiment.TruthLeak, match="Scene"):
        _with(inputs, intrinsics=scene)
    with pytest.raises(experiment.TruthLeak, match="ViewEvidence"):
        _with(inputs, fused_poses=tuple(scene.views[0] for _ in inputs.fused_poses))
    with pytest.raises(experiment.TruthLeak, match="Perturbation"):
        _with(inputs, intrinsics=scene.perturbation)
    with pytest.raises(experiment.TruthLeak, match="SurfaceScoring"):
        _with(inputs, intrinsics=contract.score_surface(scene, result))


def test_the_derivation_takes_observable_inputs_and_nothing_else(fused_case):
    scene = fused_case[0]
    with pytest.raises(experiment.TruthLeak, match="ObservableInputs and nothing else"):
        experiment.check_inputs(scene)


def test_the_derivation_returns_every_closed_shell_rather_than_a_labelled_subset(derived):
    """A deployable rule sees the whole mesh, so the derivation produces the whole mesh."""
    inputs, shells = derived
    closed = [
        index
        for index, members in enumerate(contract._shells(inputs.faces))
        if edge_census([inputs.faces[i] for i in members]).closed
    ]
    assert [shell.shell_index for shell in shells] == closed
    for shell in shells:
        assert shell.shell_id.startswith(experiment.reconstruction_digest(inputs)[:16] + "|")


# --- 2. the selector wall ---------------------------------------------------------------


def test_the_observable_record_carries_no_field_of_the_label():
    """The structural claim, asserted against the classes rather than by eye."""
    shared = set(experiment.ObservableShell.__slots__) & set(experiment.LABEL_ONLY_FIELDS)
    assert not shared, f"the observable record carries the label's {sorted(shared)}"


def test_the_observable_record_carries_no_evaluator_metadata():
    """Round one put the member, the family and the resolution on this record."""
    for name in ("member_id", "family", "cell_id", "split"):
        assert name not in experiment.ObservableShell.__slots__


def test_no_declared_feature_reads_anything_but_the_observable():
    shell = observable()
    for name, accessor in experiment.FEATURES.items():
        value = accessor(shell)
        assert value is None or isinstance(value, float), name


def test_a_rule_cannot_name_a_label_field():
    for name in experiment.LABEL_ONLY_FIELDS:
        with pytest.raises(experiment.RuleRefused, match="not an observable"):
            experiment.FeatureTest(name, "<=", 1.0)


def test_the_selector_refuses_the_evaluator_s_own_records():
    """The truth-fed selector control. Handing select() the labels must not work."""
    with pytest.raises(experiment.TruthLeak, match="ShellRecord"):
        experiment.select(experiment.RULE_GRID[0], [record(false_surface=True)])


def test_the_selector_refuses_an_observable_subclass_that_smuggles_the_label():
    """isinstance would accept this. Exact type identity is why the wall holds."""

    @dataclass(frozen=True, slots=True)
    class Leaky(experiment.ObservableShell):
        false_surface: bool = True

    shell = observable()
    leaked = Leaky(**{name: getattr(shell, name) for name in experiment.ObservableShell.__slots__})
    with pytest.raises(experiment.TruthLeak, match="Leaky"):
        experiment.select(experiment.RULE_GRID[0], [leaked])


def test_the_selector_refuses_a_mapping_dressed_as_an_observable():
    with pytest.raises(experiment.TruthLeak, match="dict"):
        experiment.select(experiment.RULE_GRID[0], [observable().as_dict()])


def test_the_selector_refuses_a_forged_observable_before_a_rule_reads_it():
    """select runs the full consistency validator, not a type check."""
    forged = observable(confirming_probes=7)
    with pytest.raises(experiment.DiscriminantUnusable, match="classified ray probe"):
        experiment.select(experiment.RULE_GRID[0], [forged])


def test_the_selector_refuses_a_rule_subclass():
    """A three-line subclass with an overridden holds was round one's whole leak."""

    @dataclass(frozen=True, slots=True)
    class Sharper(experiment.Rule):
        pass

    subclass = Sharper(
        rule_id="single:faces<=4", tests=(experiment.FeatureTest("faces", "<=", 4.0),)
    )
    with pytest.raises(experiment.RuleRefused, match="Sharper"):
        experiment.select(subclass, [observable()])


def test_the_selector_refuses_a_feature_test_subclass():
    @dataclass(frozen=True, slots=True)
    class Sneaky(experiment.FeatureTest):
        pass

    rule = experiment.Rule(rule_id="probe", tests=(Sneaky("faces", "<=", 4.0),))
    with pytest.raises(experiment.RuleRefused, match="Sneaky"):
        experiment.select(rule, [observable()])


def test_the_selector_refuses_an_object_with_its_own_holds():
    class Impostor:
        rule_id = "impostor"
        tests = ()

        def holds(self, _shell):
            return True

    with pytest.raises(experiment.RuleRefused, match="not a declared candidate rule"):
        selects(Impostor(), [observable()])


def test_the_selector_refuses_an_adaptive_threshold():
    """A callable where a number belongs is a rule that can look at the shell."""
    test = experiment.FeatureTest("faces", "<=", 4.0)
    object.__setattr__(test, "threshold", lambda shell: float(shell.faces))
    rule = experiment.Rule(rule_id="probe", tests=(test,))
    with pytest.raises(experiment.RuleRefused, match="callable"):
        experiment.select(rule, [observable()])


def test_a_non_finite_threshold_is_refused_at_construction():
    with pytest.raises(experiment.RuleRefused, match="threshold"):
        experiment.FeatureTest("faces", "<=", float("inf"))


def test_an_unavailable_observable_never_fires_a_rule():
    """Unavailable is not zero, and a rule that read it as zero would prune on nothing."""
    shell = observable(
        weight_min=None,
        weight_mean=None,
        weight_max=None,
        unavailable=(experiment.FUSED_WEIGHT,),
    )
    rule = experiment.Rule(
        rule_id="probe", tests=(experiment.FeatureTest("weight_mean_in_thresholds", "<=", 8.0),)
    )
    assert experiment.select(rule, [shell]) == ()


def test_a_rule_body_edited_after_a_result_cannot_reuse_an_accepted_id():
    """declared() compares the complete structural rule, not the name on it."""
    accepted = experiment.RULE_GRID[0]
    assert experiment.declared(accepted)
    forged = experiment.Rule(
        rule_id=accepted.rule_id,
        tests=(experiment.FeatureTest("faces", "<=", 5.0),),
    )
    assert forged.rule_id == accepted.rule_id
    assert not experiment.declared(forged)


def test_a_post_result_threshold_is_not_a_declared_candidate():
    invented = experiment.Rule(
        rule_id="single:contradicting_fraction>=0.37",
        tests=(experiment.FeatureTest("contradicting_fraction", ">=", 0.37),),
    )
    assert not experiment.declared(invented)


# --- forged observables ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"confirming_probes": 7}, "classified ray probe"),
        ({"views_containing": 6, "views_touching": 5}, "contain it"),
        ({"faces": 0}, "not a sealed shell"),
        ({"area": 0.0}, "claims area"),
        ({"nearest_measured_distance": 0.9}, "worst vertex"),
        ({"unavailable": ("something_else",)}, "as unavailable"),
        ({"vertices": 7}, "every vertex is probed in every view"),
    ],
)
def test_a_forged_observable_is_refused_for_its_own_reason(overrides, message):
    """Each forgery is refused by the field it contradicts, not by a blanket check."""
    with pytest.raises(experiment.DiscriminantUnusable, match=message):
        experiment.check_observable_is_consistent(observable(**overrides))


def test_an_unavailable_item_may_not_also_carry_a_value():
    with pytest.raises(experiment.DiscriminantUnusable, match="unavailable and carries one"):
        experiment.check_observable_is_consistent(
            observable(unavailable=(experiment.FUSED_WEIGHT,))
        )


# --- 3. record and realization identity ---------------------------------------------------


def test_a_label_whose_boolean_is_not_its_own_comparison_is_refused():
    """The swapped-label control at its source: the boolean IS the comparison."""
    with pytest.raises(experiment.DiscriminantUnusable, match="swapped label"):
        experiment.ShellTruth(
            shell_id="probe|0",
            false_surface=False,
            worst_analytic_distance=0.4,
            accuracy_tolerance=0.15,
            faces=8,
        )
    with pytest.raises(experiment.DiscriminantUnusable, match="swapped label"):
        experiment.ShellTruth(
            shell_id="probe|0",
            false_surface=True,
            worst_analytic_distance=0.02,
            accuracy_tolerance=0.15,
            faces=8,
        )


def test_a_label_with_a_non_finite_measurement_is_refused():
    with pytest.raises(experiment.DiscriminantUnusable, match="worst_analytic_distance"):
        experiment.ShellTruth(
            shell_id="probe|0",
            false_surface=True,
            worst_analytic_distance=float("nan"),
            accuracy_tolerance=0.15,
            faces=8,
        )


def test_a_record_whose_halves_disagree_about_the_face_count_is_refused():
    shell = observable()
    with pytest.raises(experiment.DiscriminantUnusable, match="labelled with"):
        experiment.ShellRecord(
            observable=shell,
            truth=experiment.ShellTruth(
                shell_id=shell.shell_id,
                false_surface=True,
                worst_analytic_distance=0.4,
                accuracy_tolerance=0.15,
                faces=9,
            ),
            member_id="train-a",
            family=contract.DEPTH_OUTLIERS,
            cell_id=experiment.SELECTION_RESOLUTION,
            split=experiment.TRAIN,
            reconstruction_digest=DIGEST_A,
            face_indices=shell.face_indices,
        )


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        ((0, 0, 1, 2, 3, 4, 5, 6), "same face index twice"),
        ((-1, 1, 2, 3, 4, 5, 6, 7), "negative face index"),
        ((1, 2, 3, 4, 5, 6, 7, 8), "not the observable"),
        ((0, 1, 2), "carries 3"),
    ],
)
def test_a_record_with_an_unusable_face_set_is_refused(indices, message):
    shell = observable()
    with pytest.raises(experiment.DiscriminantUnusable, match=message):
        experiment.ShellRecord(
            observable=shell,
            truth=truth_for(shell, false_surface=True),
            member_id="train-a",
            family=contract.DEPTH_OUTLIERS,
            cell_id=experiment.SELECTION_RESOLUTION,
            split=experiment.TRAIN,
            reconstruction_digest=DIGEST_A,
            face_indices=indices,
        )


def test_a_record_over_something_that_is_not_an_observable_is_refused():
    shell = observable()
    with pytest.raises(experiment.TruthLeak, match="not exactly"):
        experiment.ShellRecord(
            observable=shell.as_dict(),
            truth=truth_for(shell, false_surface=True),
            member_id="train-a",
            family=contract.DEPTH_OUTLIERS,
            cell_id=experiment.SELECTION_RESOLUTION,
            split=experiment.TRAIN,
            reconstruction_digest=DIGEST_A,
            face_indices=shell.face_indices,
        )


def test_a_record_with_a_reconstruction_identity_that_is_not_a_digest_is_refused():
    with pytest.raises(experiment.DiscriminantUnusable, match="not a SHA-256"):
        record(false_surface=True, reconstruction_digest="not-a-digest")


def test_the_reconstruction_identity_notices_a_raw_frame_change(derived):
    """Raw-only: the same poses, one different measured frame."""
    inputs, _ = derived
    before = experiment.reconstruction_digest(inputs)
    frames = list(inputs.raw_frames)
    frames[0] = bytes([(frames[0][0] + 1) % 256]) + frames[0][1:]
    after = experiment.reconstruction_digest(_with(inputs, raw_frames=tuple(frames)))
    assert after != before


def test_the_reconstruction_identity_notices_a_fused_pose_change(derived):
    """Pose-only: the same frames, one different pose the reconstruction used.

    Round one hashed the frame digests alone, so two reconstructions that
    differed only in the poses they were told looked like one realization.
    """
    inputs, _ = derived
    before = experiment.reconstruction_digest(inputs)
    poses = list(inputs.fused_poses)
    moved = poses[0]
    poses[0] = Rigid.of(
        moved.rotation,
        (moved.translation[0] + 0.013, moved.translation[1], moved.translation[2]),
    )
    after = experiment.reconstruction_digest(_with(inputs, fused_poses=tuple(poses)))
    assert after != before


def test_the_reconstruction_identity_of_an_exact_clone_is_the_same(derived):
    inputs, _ = derived
    clone = experiment.ObservableInputs(
        **{name: getattr(inputs, name) for name in experiment.ObservableInputs.__slots__}
    )
    assert clone is not inputs
    assert experiment.reconstruction_digest(clone) == experiment.reconstruction_digest(inputs)


def test_the_reconstruction_identity_notices_a_fusion_backend_change(derived):
    """Backend-only: the same frames, poses, cell and volume box, one different backend.

    The reconstruction identity binds the backend for the same reason it binds the poses: the
    backend is part of what a reconstruction is a function of, and two runs that are byte-
    identical in every other observable input but ran on different backends are two different
    reconstructions, not one reconstruction measured twice.
    """
    inputs, _ = derived
    before = experiment.reconstruction_digest(inputs)
    other = next(name for name in experiment.FUSION_BACKENDS if name != inputs.fusion_backend)
    after = experiment.reconstruction_digest(_with(inputs, fusion_backend=other))
    assert after != before


def test_an_undeclared_fusion_backend_is_refused(derived):
    """A backend this oracle does not run is refused, not silently accepted.

    This experiment's declared arms fuse on exactly the two backends in
    :data:`~tests.shell_discriminant.FUSION_BACKENDS`. A candidate ``fusion_backend`` naming a
    third is not a value this corpus can have produced, so it is refused here rather than
    becoming part of an identity nothing actually measured.
    """
    inputs, _ = derived
    with pytest.raises(experiment.DiscriminantUnusable, match="this oracle runs"):
        _with(inputs, fusion_backend="cuda")


def test_a_fusion_backend_that_is_not_text_is_refused(derived):
    """``None`` and an int are not ``str``, and the check requires exactly that type.

    The same guard the other scalar fields get: a ``fusion_backend`` that is not text cannot be
    one of :data:`~tests.shell_discriminant.FUSION_BACKENDS` either, but the type is checked on
    its own so the refusal names the real defect instead of reporting a bare membership
    failure.
    """
    inputs, _ = derived
    with pytest.raises(experiment.DiscriminantUnusable, match="this oracle runs"):
        _with(inputs, fusion_backend=None)
    with pytest.raises(experiment.DiscriminantUnusable, match="this oracle runs"):
        _with(inputs, fusion_backend=7)


def test_the_declared_fusion_backends_are_exactly_the_oracles_two():
    """The whole vocabulary: the pure-Python reference and the appliance backend, no other."""
    assert experiment.FUSION_BACKENDS == (contract.BACKEND, contract.APPLIANCE_BACKEND)


# --- 4. the fusion the observables were read out of ---------------------------------------


def test_cold_and_warm_fusions_agree_and_a_moved_pose_is_a_different_field(fused_case):
    """The memo is a claim that two calls MUST produce the same field.

    Cold, warm and volume-keeping fusions of the same inputs produce identical
    vertices, faces and FieldRecord; a scene that differs only in the poses the
    reconstruction was told does not reach the same cache entry, because the
    fused poses are inside the memo key.
    """
    scene, _, cell, _ = fused_case
    contract._FUSION_CACHE.clear()
    cold_vertices, cold_record = contract.fuse(scene, cell)
    assert len(contract._FUSION_CACHE) == 1
    warm_vertices, warm_record = contract.fuse(scene, cell)
    assert warm_vertices == cold_vertices
    assert warm_record == cold_record
    _, kept_vertices, kept_record = contract.fuse_keeping_volume(scene, cell)
    assert kept_vertices == cold_vertices
    assert kept_record.faces == cold_record.faces
    assert kept_record == cold_record

    moved = _with(
        scene,
        views=tuple(
            _with(
                view,
                fused_pose=Rigid.of(
                    view.fused_pose.rotation,
                    (
                        view.fused_pose.translation[0] + 0.13,
                        view.fused_pose.translation[1],
                        view.fused_pose.translation[2],
                    ),
                ),
            )
            for view in scene.views
        ),
    )
    assert contract._fusion_key(moved, cell, range(len(moved.views)), contract.BACKEND) != (
        contract._fusion_key(scene, cell, range(len(scene.views)), contract.BACKEND)
    )
    moved_vertices, _ = contract.fuse(moved, cell)
    assert len(contract._FUSION_CACHE) == 2
    assert moved_vertices != cold_vertices


def test_the_shell_weights_come_from_the_field_that_made_the_mesh(fused_case, derived):
    """The kept volume is the one the published surface was extracted from."""
    scene, result, cell, volume = fused_case
    _, vertices, field_record = contract.fuse_keeping_volume(scene, cell)
    assert tuple(result.points) == tuple(vertices)
    assert tuple(result.faces) == tuple(field_record.faces)
    inputs, shells = derived
    assert inputs.field is volume
    assert any(shell.weight_mean is not None for shell in shells)


# --- 5. one bounded decision matrix -------------------------------------------------------


def population():
    """Four synthetic records: two appliance training, one holdout, one diagnostic."""
    return [
        record(
            false_surface=True,
            shell=contradicted("aaaaaaaaaaaaaaaa|0", fraction_high=True),
        ),
        record(
            false_surface=False,
            shell=contradicted("aaaaaaaaaaaaaaaa|1", fraction_high=False),
        ),
        record(
            false_surface=True,
            split=experiment.HOLDOUT,
            member_id="sealed-a",
            family=contract.COMBINED,
            reconstruction_digest=DIGEST_B,
            shell=contradicted("bbbbbbbbbbbbbbbb|0", fraction_high=True),
        ),
        record(
            false_surface=True,
            cell_id=DIAGNOSTIC,
            reconstruction_digest=DIGEST_C,
            shell=contradicted("cccccccccccccccc|0", fraction_high=True),
        ),
    ]


def test_the_producer_pays_for_exactly_one_rule_per_candidate_per_shell():
    """Every summary afterwards is a lookup, and the ledger proves it."""
    records = population()
    experiment.reset_rule_calls()
    matrix = experiment.build_decision_matrix(records)
    spent = experiment.rule_calls()[experiment.PRODUCER_PHASE]
    assert spent == len(experiment.RULE_GRID) * len(records)
    assert matrix.calls == spent

    for rule in experiment.RULE_GRID:
        experiment.cross_tab(matrix, rule.rule_id, records, resolutions=RESOLUTIONS)
        for name, subset in experiment.aggregate_slices(records).items():
            experiment.confusion(matrix, rule.rule_id, subset, slice_name=name)
    experiment.choose(matrix, records)
    assert experiment.rule_calls() == {experiment.PRODUCER_PHASE: spent}


def test_a_decision_matrix_answers_only_about_the_population_it_was_built_over():
    records = population()
    experiment.reset_rule_calls()
    matrix = experiment.build_decision_matrix(records)
    rule_id = experiment.RULE_GRID[0].rule_id
    assert matrix.holds(rule_id, records[0].shell_id) in (True, False)
    with pytest.raises(experiment.DiscriminantUnusable, match="not a column"):
        matrix.holds(rule_id, "somewhere|else|0")
    with pytest.raises(experiment.DiscriminantUnusable, match="not a row"):
        matrix.selected("single:invented<=1")


def test_a_population_past_the_work_ceiling_is_refused_rather_than_tuned(monkeypatch):
    records = population()
    assert experiment.preflight(records) == len(experiment.RULE_GRID) * len(records)
    monkeypatch.setattr(experiment, "WORK_CEILING", 1)
    with pytest.raises(experiment.DiscriminantUnusable, match="past the declared ceiling"):
        experiment.preflight(records)
    with pytest.raises(experiment.DiscriminantUnusable, match="past the declared ceiling"):
        experiment.build_decision_matrix(records)


def test_the_declared_ceiling_is_the_predeclared_one():
    assert experiment.WORK_CEILING == 40000
    assert experiment.MATERIAL_FRACTION == 0.5
    assert experiment.SELECTION_RESOLUTION == "appliance"


# --- 6. selection reads the product's own resolution, and the training half ---------------


def test_a_diagnostic_resolution_record_cannot_qualify_a_candidate():
    """The whole point of a predeclared selection resolution.

    The appliance training half here holds one CORRECT shell and no false one,
    so no rule can clear the materiality gate on it. The diagnostic half holds
    a false shell that half the grid selects. If the coarse cell were allowed
    to count, a candidate would qualify; it must not.
    """
    diagnostic = record(
        false_surface=True,
        cell_id=DIAGNOSTIC,
        reconstruction_digest=DIGEST_C,
        shell=contradicted("cccccccccccccccc|0", fraction_high=True),
    )
    appliance_correct = record(
        false_surface=False,
        shell=contradicted("aaaaaaaaaaaaaaaa|1", fraction_high=False),
    )
    records = [appliance_correct, diagnostic]
    experiment.reset_rule_calls()
    selection = experiment.choose(experiment.build_decision_matrix(records), records)
    assert selection.rule is None
    assert selection.selection_resolution == experiment.SELECTION_RESOLUTION
    assert selection.training_shells == 1

    # The same shell at the appliance resolution DOES qualify a candidate, so
    # the refusal above is about the resolution and not about the shell.
    promoted = record(
        false_surface=True,
        reconstruction_digest=DIGEST_C,
        shell=contradicted("cccccccccccccccc|0", fraction_high=True),
    )
    both = [appliance_correct, promoted]
    experiment.reset_rule_calls()
    assert experiment.choose(experiment.build_decision_matrix(both), both).rule is not None


def test_selection_is_unchanged_when_the_holdout_labels_are_swapped():
    records = population()
    experiment.reset_rule_calls()
    matrix = experiment.build_decision_matrix(records)
    honest = experiment.choose(matrix, records)
    assert honest.rule is not None
    swapped = [
        item
        if item.split != experiment.HOLDOUT
        else experiment.ShellRecord(
            observable=item.observable,
            truth=truth_for(item.observable, false_surface=not item.truth.false_surface),
            member_id=item.member_id,
            family=item.family,
            cell_id=item.cell_id,
            split=item.split,
            reconstruction_digest=item.reconstruction_digest,
            face_indices=item.face_indices,
        )
        for item in records
    ]
    assert [r.truth.false_surface for r in swapped] != [r.truth.false_surface for r in records]
    assert experiment.choose(matrix, swapped).as_dict() == honest.as_dict()


def test_selection_is_unchanged_when_the_holdout_records_are_withheld():
    records = population()
    experiment.reset_rule_calls()
    honest = experiment.choose(experiment.build_decision_matrix(records), records)
    training = list(experiment.training_records(records))
    experiment.reset_rule_calls()
    withheld = experiment.choose(experiment.build_decision_matrix(training), training)
    for name in ("eligible_rule_ids", "reason", "selection_resolution", "training_shells"):
        assert getattr(withheld, name) == getattr(honest, name)
    assert withheld.rule is not None
    assert honest.rule is not None
    assert withheld.rule.as_dict() == honest.rule.as_dict()


def test_the_split_helpers_read_the_declared_halves_at_the_declared_resolution():
    records = population()
    assert [r.shell_id for r in experiment.training_records(records)] == [
        records[0].shell_id,
        records[1].shell_id,
    ]
    assert [r.shell_id for r in experiment.holdout_records(records)] == [records[2].shell_id]


def test_a_selected_rule_is_always_one_of_the_declared_candidates():
    records = population()
    experiment.reset_rule_calls()
    selection = experiment.choose(experiment.build_decision_matrix(records), records)
    assert selection.rule is not None
    assert experiment.declared(selection.rule)
    assert selection.eligible_rule_ids
    assert selection.rule.rule_id in selection.eligible_rule_ids


# --- 7. the complete cross-product, zero cells included -----------------------------------


def test_the_cross_product_is_every_split_family_and_resolution():
    cells = experiment.cross_cells(RESOLUTIONS)
    assert len(cells) == len(experiment.SPLITS) * len(contract.PERTURBATIONS) * len(RESOLUTIONS)
    assert len(cells) == 28
    assert len(set(cells)) == len(cells)
    assert cells[0] == (experiment.TRAIN, contract.PERTURBATIONS[0], RESOLUTIONS[0])


def test_the_cross_tab_publishes_one_confusion_per_declared_cell_in_that_order():
    """Including the cells no shell landed in, which is how a one-family rule hides."""
    records = population()
    experiment.reset_rule_calls()
    matrix = experiment.build_decision_matrix(records)
    rule_id = experiment.RULE_GRID[0].rule_id
    cells = experiment.cross_tab(matrix, rule_id, records, resolutions=RESOLUTIONS)
    assert len(cells) == 28
    assert [cell.slice_name for cell in cells] == [
        experiment.cross_tab_name(*key) for key in experiment.cross_cells(RESOLUTIONS)
    ]
    empty = [cell for cell in cells if cell.shells == 0]
    assert len(empty) == 28 - 3
    assert sum(cell.shells for cell in cells) == len(records)


def test_a_record_outside_the_declared_cross_product_is_refused():
    records = population()
    experiment.reset_rule_calls()
    matrix = experiment.build_decision_matrix(records)
    with pytest.raises(experiment.DiscriminantUnusable, match="not one of the declared"):
        experiment.cross_tab(
            matrix,
            experiment.RULE_GRID[0].rule_id,
            records,
            resolutions=(experiment.SELECTION_RESOLUTION,),
        )


def test_a_confusion_reconciles_or_is_refused():
    with pytest.raises(experiment.DiscriminantUnusable, match="do not add up"):
        experiment.Confusion(
            slice_name="probe",
            rule_id="probe",
            shells=3,
            false_shells=1,
            correct_shells=1,
            false_faces=8,
            correct_faces=8,
            selected_shells=0,
            selected_false_shells=0,
            selected_correct_shells=0,
            selected_false_faces=0,
            selected_correct_faces=0,
        )


def test_swapping_the_labels_destroys_the_result():
    """The swapped-label control: the same rule must not pass on inverted truth."""
    records = population()[:2]
    experiment.reset_rule_calls()
    matrix = experiment.build_decision_matrix(records)
    rule = experiment.Rule(
        rule_id="single:contradicting_fraction>=0.5",
        tests=(experiment.FeatureTest("contradicting_fraction", ">=", 0.5),),
    )
    honest = experiment.confusion(matrix, rule.rule_id, records, slice_name="probe")
    assert honest.removes_no_correct_faces and honest.removes_material_false_faces
    swapped = [
        experiment.ShellRecord(
            observable=item.observable,
            truth=truth_for(item.observable, false_surface=not item.truth.false_surface),
            member_id=item.member_id,
            family=item.family,
            cell_id=item.cell_id,
            split=item.split,
            reconstruction_digest=item.reconstruction_digest,
            face_indices=item.face_indices,
        )
        for item in records
    ]
    inverted = experiment.confusion(matrix, rule.rule_id, swapped, slice_name="probe")
    assert not inverted.removes_no_correct_faces


# --- the predeclared matrix ---------------------------------------------------------------


def test_the_grid_is_complete_and_uniquely_named():
    ids = [rule.rule_id for rule in experiment.RULE_GRID]
    assert len(set(ids)) == len(ids)
    expected = sum(len(values) for _, values in experiment.GRID.values())
    expected += sum(
        len(experiment.GRID[first][1]) * len(experiment.GRID[second][1])
        for first, second in experiment.PAIRS
    )
    assert len(experiment.RULE_GRID) == expected
    assert len(experiment.RULE_GRID) == 195


def test_every_declared_rule_reads_only_declared_features():
    for rule in experiment.RULE_GRID:
        for test in rule.tests:
            assert test.feature in experiment.FEATURES
            assert test.threshold in experiment.GRID[test.feature][1]
            assert test.comparison == experiment.GRID[test.feature][0]


def test_the_candidate_bodies_are_the_ones_that_were_frozen():
    assert experiment.candidate_digest() == experiment.FROZEN_CANDIDATE_DIGEST
    assert experiment.check_candidates_are_frozen() == experiment.FROZEN_CANDIDATE_DIGEST


def test_the_candidate_digest_ignores_the_protocol_and_the_grid_digest_does_not(monkeypatch):
    """The protocol was legitimately rewritten this round; the bodies were not."""
    candidates = experiment.candidate_digest()
    protocol = experiment.grid_digest()
    monkeypatch.setattr(experiment, "SELECTION_PROTOCOL", "something else entirely")
    assert experiment.candidate_digest() == candidates
    assert experiment.grid_digest() != protocol


def test_a_moved_candidate_body_is_refused(monkeypatch):
    monkeypatch.setattr(
        experiment,
        "RULE_GRID",
        experiment.RULE_GRID[:-1],
    )
    with pytest.raises(experiment.RuleRefused, match="frozen at"):
        experiment.check_candidates_are_frozen()


# --- 8. the counterfactual, which is the only thing that settles a recommendation ---------


def real_record(shell, *, false_surface, member_id="combined-realistic", cell_id=CELL):
    return experiment.ShellRecord(
        observable=shell,
        truth=truth_for(shell, false_surface=false_surface),
        member_id=member_id,
        family=contract.COMBINED,
        cell_id=cell_id,
        split=experiment.TRAIN,
        reconstruction_digest=DIGEST_A,
        face_indices=shell.face_indices,
    )


def test_pruning_removes_exactly_those_faces_and_reindexes(fused_case):
    _, result, _, _ = fused_case
    removed = [0, 1, 2]
    pruned = experiment.prune(result, removed)
    assert len(pruned.faces) == len(result.faces) - len(removed)
    assert max(index for face in pruned.faces for index in face) < len(pruned.points)
    assert pruned.field_evidence is None
    kept = {tuple(sorted(result.points[index] for index in face)) for face in result.faces[3:]}
    now = {tuple(sorted(pruned.points[index] for index in face)) for face in pruned.faces}
    assert now == kept


def test_pruning_a_face_that_is_not_there_is_refused(fused_case):
    _, result, _, _ = fused_case
    with pytest.raises(experiment.DiscriminantUnusable, match="asked to remove face"):
        experiment.prune(result, [len(result.faces)])


def test_shell_identity_refuses_a_record_pointing_at_another_component(fused_case, derived):
    """Round one trusted the face indices a record carried.

    The components are recomputed here from the mesh, so a record whose face
    set is a different component is refused BEFORE a triangle is removed rather
    than quietly measuring the wrong removal.
    """
    _, result, _, _ = fused_case
    _, shells = derived
    assert len(shells) >= 2
    forged = _with(
        shells[0],
        face_indices=shells[1].face_indices,
        faces=shells[1].faces,
    )
    with pytest.raises(experiment.DiscriminantUnusable, match="not component"):
        experiment.check_shell_identity(
            result,
            [forged],
            [],
            member_id="combined-realistic",
            cell_id=CELL,
        )
    honest = experiment.check_shell_identity(
        result,
        shells,
        [real_record(shells[0], false_surface=True)],
        member_id="combined-realistic",
        cell_id=CELL,
    )
    assert len(honest) == len(contract._shells(result.faces))


def test_shell_identity_refuses_a_record_from_another_member(fused_case, derived):
    _, result, _, _ = fused_case
    _, shells = derived
    with pytest.raises(experiment.DiscriminantUnusable, match="offered to the counterfactual"):
        experiment.check_shell_identity(
            result,
            shells,
            [real_record(shells[0], false_surface=True, member_id="somebody-else")],
            member_id="combined-realistic",
            cell_id=CELL,
        )


def test_a_counterfactual_that_names_a_shell_of_another_mesh_is_refused(fused_case, derived):
    scene, result, _, _ = fused_case
    _, shells = derived
    with pytest.raises(experiment.DiscriminantUnusable, match="not closed shells of this mesh"):
        experiment.counterfactual(
            scene,
            result,
            shells,
            [],
            ["somewhere|else|0"],
            member_id="combined-realistic",
            cell_id=CELL,
            split=experiment.TRAIN,
            reconstruction_digest=DIGEST_A,
        )


def test_removing_nothing_changes_nothing_and_earns_no_gain(fused_case, derived):
    """The floor of the counterfactual: an empty removal must not report a win."""
    scene, result, _, _ = fused_case
    _, shells = derived
    evidence = experiment.counterfactual(
        scene,
        result,
        shells,
        [],
        [],
        member_id="combined-realistic",
        cell_id=CELL,
        split=experiment.TRAIN,
        reconstruction_digest=DIGEST_A,
    )
    assert evidence.removed_faces == 0
    assert evidence.before.as_dict() == evidence.after.as_dict()
    assert evidence.gates["counterfactual_false_faces_fell"] is False
    assert evidence.false_faces_removed == 0
    assert evidence.passed is False


def test_removing_a_shell_nothing_labelled_is_damage_and_fails_its_gate(fused_case, derived):
    """The prefilter an appliance does not have, made visible.

    The rule is applied to every closed shell, so a shell the evaluator's
    unsupported population never contained can be removed - and that is surface
    the scene's own ideal returns DO reach. It is published as
    ``removed_supported_*`` damage and it fails its own gate.
    """
    scene, result, _, _ = fused_case
    _, shells = derived
    labelled = [real_record(shells[0], false_surface=True)]
    evidence = experiment.counterfactual(
        scene,
        result,
        shells,
        labelled,
        [shells[1].shell_id],
        member_id="combined-realistic",
        cell_id=CELL,
        split=experiment.TRAIN,
        reconstruction_digest=DIGEST_A,
    )
    assert evidence.removed_shell_ids == (shells[1].shell_id,)
    assert evidence.removed_supported_shells == 1
    assert evidence.removed_supported_shell_faces == shells[1].faces
    assert evidence.removed_false_shells == 0
    assert evidence.removed_correct_shells == 0
    assert evidence.gates["counterfactual_removes_no_supported_faces"] is False
    assert evidence.passed is False
    assert evidence.after.faces == evidence.before.faces - shells[1].faces


def test_the_counterfactual_gate_names_are_the_declared_ones(fused_case, derived):
    scene, result, _, _ = fused_case
    _, shells = derived
    evidence = experiment.counterfactual(
        scene,
        result,
        shells,
        [],
        [],
        member_id="combined-realistic",
        cell_id=CELL,
        split=experiment.TRAIN,
        reconstruction_digest=DIGEST_A,
    )
    declared = {gate for gate in experiment.GATES if gate.startswith("counterfactual_")}
    assert set(evidence.gates) == declared
    assert "counterfactual_removes_no_supported_faces" in declared
    assert "counterfactual_triangle_accuracy_held" in declared
