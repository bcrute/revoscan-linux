"""Regenerate the observable-shell discriminator population. The recipe behind the fixtures.

Run it with::

    .venv/bin/python -m tests.build_shell_discriminant_fixtures

This is a **command, not a test**: pytest never runs it, and a run that changes
a fixture is changing evidence. It is the only thing in this repository allowed
to write inside ``tests/fixtures/shell_discriminant/``, it writes nowhere else,
and it rewrites that directory whole, so a member deleted from the recipe below
cannot survive on disk as a file the manifest no longer names. It does not
touch ``tests/fixtures/surface_oracle/``, which
:mod:`tests.build_surface_oracle_fixtures` owns and this experiment only reads.

What is in here, and what is deliberately not
---------------------------------------------
In here: the population design. Which perturbations, at which magnitudes, with
which seeds, and which half of the train/holdout split each of them belongs to.
All of it is written before a run, and each member is a WHOLE synthetic
realization: its shells go entirely to one side of the split, so no shell of
one realization can be learned from and graded on at the same time.

Not in here: any threshold, any rule, any gate and any measurement. The
candidate matrix lives in :mod:`tests.shell_discriminant` as reviewed code and
is bound into the manifest only by its digest, so a fixture cannot move a
threshold and a threshold cannot be moved without every fixture noticing.

The scene is not this module's either
-------------------------------------
Every member reuses the accepted surface-oracle corpus's ``exact-control``
scene - the same object, the same five views, the same intrinsics, window,
volume and named regions - byte for byte, and the manifest records that case's
SHA-256. What a member contributes is one perturbation and one seed. That keeps
the geometry an already-reviewed artifact and makes the population a study of
the PERTURBATION rather than of a second object nobody has looked at.

Why these members
-----------------
The accepted oracle found its seven motivating shells in two places: localized
depth outliers, which throw sealed blobs off the object, and a strongly
perturbed combined case, whose noise and pose error seal small correct blobs
the ideal returns do not reach. So the population varies the OUTLIER magnitude
and period - the period moves which samples are hit, which moves where the
blobs are, and the sign already alternates by construction - and it varies the
combined case's noise magnitude, band geometry and pose error. The families
that produced no unsupported shell at all in the accepted corpus are kept
anyway, at their own magnitudes, because a population that contained only the
families known to produce shells would be a population selected on the outcome.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from scanner3d.geometry import Point
from tests import depth_completeness as completeness
from tests import shell_discriminant as experiment
from tests import shell_discriminant_harness as harness
from tests import surface_oracle as contract
from tests import surface_oracle_harness as oracle

DESTINATION: Final = Path(__file__).parent / "fixtures" / "shell_discriminant"
BASE_CASE: Final = "exact-control.json"
GENERATED_BY: Final = "tests/build_shell_discriminant_fixtures.py"

CORPUS_WHAT: Final = (
    "One whole synthetic realization per member, over the accepted surface-oracle exact-control "
    "scene, declared before any result and split into training and holdout halves by whole "
    "realization."
)
CORPUS_WHY: Final = (
    "The accepted surface oracle proved that a fused surface contains sealed shells nothing "
    "measured, and that some are false surface while others are correct geometry. This "
    "population exists to answer whether a bounded rule over evidence the appliance actually has "
    "can tell those two apart on data it was not chosen on."
)
CORPUS_UNITS: Final = (
    "Lengths are synthetic depth units: raw code times depth scale, exact by construction. No "
    "physical length is implied anywhere in this population."
)

#: The two resolutions every member is fused at. The appliance arm is where the
#: motivating shells live; the fine grid cell is the accepted corpus's own
#: finest declared cell, and it is here so the confusion can be reported BY
#: resolution rather than at one operating point that might be special.
RESOLUTIONS: Final[tuple[str, ...]] = (harness.APPLIANCE, "v10-t4-w1-uniform")

#: The member the truth-fed upper bound is measured on, named before any
#: result. Round two's planner packet predeclared it as
#: ``sealed-outlier-185-p43`` at appliance resolution, and it was written here
#: before any builder, probe, scene construction, fusion or harness run touched
#: a sealed id.
UPPER_BOUND_MEMBER: Final = "sealed-outlier-185-p43"
UPPER_BOUND_CELL: Final = harness.APPLIANCE

#: There is deliberately no counterfactual member ceiling any more. Round one
#: capped it at four (member, resolution) pairs, and a recommendation resting on
#: a partial sample is not a recommendation: a rule that qualifies is now
#: re-measured on EVERY sealed holdout appliance population it fires on, over
#: all closed shells rather than the evaluator's unsupported subset.

_STRONG_COMBINED: Final[dict[str, object]] = {
    "depth_noise_codes": 20,
    "band_period": 7,
    "band_width": 2,
    "band_shift": 3,
    "translation_bias": [0.05, 0.0, 0.0],
    "translation_jitter": 0.05,
    "rotation_bias_degrees": 0.3,
    "rotation_jitter_degrees": 0.3,
}

#: The population, in manifest order. ``(member_id, family, split, overrides)``.
MEMBERS: Final[tuple[tuple[str, str, str, dict[str, object]], ...]] = (
    ("exact-control", contract.EXACT, experiment.TRAIN, {}),
    (
        "outliers-140-p17",
        contract.DEPTH_OUTLIERS,
        experiment.TRAIN,
        {"outlier_codes": 140, "outlier_period": 17},
    ),
    (
        "outliers-140-p37",
        contract.DEPTH_OUTLIERS,
        experiment.TRAIN,
        {"outlier_codes": 140, "outlier_period": 37},
    ),
    (
        "outliers-200-p29",
        contract.DEPTH_OUTLIERS,
        experiment.TRAIN,
        {"outlier_codes": 200, "outlier_period": 29},
    ),
    ("noise-20", contract.DEPTH_NOISE, experiment.TRAIN, {"depth_noise_codes": 20}),
    (
        "bands-2-7",
        contract.NO_RETURN_BANDS,
        experiment.TRAIN,
        {"band_width": 2, "band_period": 7, "band_shift": 3},
    ),
    (
        "pose-translation-05",
        contract.POSE_TRANSLATION,
        experiment.TRAIN,
        {"translation_bias": [0.05, 0.0, 0.0], "translation_jitter": 0.05},
    ),
    (
        "pose-rotation-03",
        contract.POSE_ROTATION,
        experiment.TRAIN,
        {"rotation_bias_degrees": 0.3, "rotation_jitter_degrees": 0.3},
    ),
    ("combined-strong", contract.COMBINED, experiment.TRAIN, dict(_STRONG_COMBINED)),
    (
        "combined-outliers",
        contract.COMBINED,
        experiment.TRAIN,
        {**_STRONG_COMBINED, "outlier_codes": 140, "outlier_period": 37},
    ),
    (
        "outliers-110-p17",
        contract.DEPTH_OUTLIERS,
        experiment.TRAIN,
        {"outlier_codes": 110, "outlier_period": 17},
    ),
    (
        "outliers-140-p19",
        contract.DEPTH_OUTLIERS,
        experiment.TRAIN,
        {"outlier_codes": 140, "outlier_period": 19},
    ),
    (
        "outliers-140-p23",
        contract.DEPTH_OUTLIERS,
        experiment.TRAIN,
        {"outlier_codes": 140, "outlier_period": 23},
    ),
    (
        "outliers-140-p31",
        contract.DEPTH_OUTLIERS,
        experiment.TRAIN,
        {"outlier_codes": 140, "outlier_period": 31},
    ),
    (
        "outliers-170-p37",
        contract.DEPTH_OUTLIERS,
        experiment.TRAIN,
        {"outlier_codes": 170, "outlier_period": 37},
    ),
    ("noise-24", contract.DEPTH_NOISE, experiment.TRAIN, {"depth_noise_codes": 24}),
    (
        "bands-3-11",
        contract.NO_RETURN_BANDS,
        experiment.TRAIN,
        {"band_width": 3, "band_period": 11, "band_shift": 2},
    ),
    (
        "combined-strong-b",
        contract.COMBINED,
        experiment.TRAIN,
        {
            **_STRONG_COMBINED,
            "depth_noise_codes": 22,
            "translation_bias": [0.06, 0.0, 0.01],
            "rotation_bias_degrees": 0.35,
        },
    ),
    (
        "combined-outliers-b",
        contract.COMBINED,
        experiment.TRAIN,
        {**_STRONG_COMBINED, "outlier_codes": 150, "outlier_period": 23},
    ),
    # -- the sealed holdout ------------------------------------------------
    #
    # Round one inspected every member of its own holdout during development
    # and then widened the candidate grid, so all nineteen members above are
    # TRAIN and none of them may support a held-out claim. The eight members
    # below are the round-two planner packet's SEALED holdout, transcribed
    # verbatim from .coordination/planner/current-task.md and written into this
    # recipe before any builder, probe, scene construction, fusion or harness
    # run touched a sealed id. Their seed is exactly
    # ``shell-discriminant:<member-id>`` through SEED_PREFIX below.
    #
    # Nothing here may be tuned after an output exists. If the sealed appliance
    # population turns out to lack either class, or if 195 candidates times the
    # declared shell count exceeds WORK_CEILING, the population is REFUSED and
    # returned for planner review rather than adjusted until it passes.
    (
        "sealed-outlier-125-p41",
        contract.DEPTH_OUTLIERS,
        experiment.HOLDOUT,
        {"outlier_codes": 125, "outlier_period": 41},
    ),
    (
        "sealed-outlier-185-p43",
        contract.DEPTH_OUTLIERS,
        experiment.HOLDOUT,
        {"outlier_codes": 185, "outlier_period": 43},
    ),
    (
        "sealed-outlier-155-p47",
        contract.DEPTH_OUTLIERS,
        experiment.HOLDOUT,
        {"outlier_codes": 155, "outlier_period": 47},
    ),
    (
        "sealed-outlier-105-p53",
        contract.DEPTH_OUTLIERS,
        experiment.HOLDOUT,
        {"outlier_codes": 105, "outlier_period": 53},
    ),
    (
        "sealed-combined-a",
        contract.COMBINED,
        experiment.HOLDOUT,
        {
            "depth_noise_codes": 19,
            "band_width": 2,
            "band_period": 7,
            "band_shift": 4,
            "translation_bias": [0.052, 0.0, -0.006],
            "translation_jitter": 0.047,
            "rotation_bias_degrees": 0.31,
            "rotation_jitter_degrees": 0.27,
        },
    ),
    (
        "sealed-combined-b",
        contract.COMBINED,
        experiment.HOLDOUT,
        {
            "depth_noise_codes": 23,
            "band_width": 3,
            "band_period": 19,
            "band_shift": 4,
            "translation_bias": [0.064, 0.0, 0.012],
            "translation_jitter": 0.056,
            "rotation_bias_degrees": 0.38,
            "rotation_jitter_degrees": 0.34,
        },
    ),
    (
        "sealed-combined-c",
        contract.COMBINED,
        experiment.HOLDOUT,
        {
            "depth_noise_codes": 17,
            "band_width": 4,
            "band_period": 19,
            "band_shift": 5,
            "translation_bias": [0.046, 0.0, -0.013],
            "translation_jitter": 0.041,
            "rotation_bias_degrees": 0.26,
            "rotation_jitter_degrees": 0.33,
        },
    ),
    (
        "sealed-combined-outlier-d",
        contract.COMBINED,
        experiment.HOLDOUT,
        {
            "depth_noise_codes": 21,
            "outlier_codes": 135,
            "outlier_period": 43,
            "band_width": 2,
            "band_period": 11,
            "band_shift": 4,
            "translation_bias": [0.057, 0.0, 0.008],
            "translation_jitter": 0.049,
            "rotation_bias_degrees": 0.34,
            "rotation_jitter_degrees": 0.28,
        },
    ),
)

#: The exact sealed holdout member ids, in the order the planner declared them.
#: The builder refuses to write a population whose HOLDOUT half is not exactly
#: this list, so a sealed member cannot be quietly dropped, renamed or joined by
#: a tenth one after a result exists.
SEALED_HOLDOUT: Final[tuple[str, ...]] = (
    "sealed-outlier-125-p41",
    "sealed-outlier-185-p43",
    "sealed-outlier-155-p47",
    "sealed-outlier-105-p53",
    "sealed-combined-a",
    "sealed-combined-b",
    "sealed-combined-c",
    "sealed-combined-outlier-d",
)

#: The seed prefix every member's draws come from. The member id is the rest of
#: it, so two members can never share a draw sequence by accident.
SEED_PREFIX: Final = "shell-discriminant"


def _count(overrides: Mapping[str, object], key: str) -> int:
    """One integer field of a member's recipe, or zero. Refuses anything else.

    The recipe above is a mapping of plain JSON values so it reads as data, and
    reading it back through named accessors is what keeps that readable
    without the builder quietly coercing a typo into a number.
    """
    value = overrides.get(key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"the recipe gives {key} as {value!r}, which is not a whole number")
    return value


def _amount(overrides: Mapping[str, object], key: str) -> float:
    value = overrides.get(key, 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"the recipe gives {key} as {value!r}, which is not a number")
    return float(value)


def _direction(overrides: Mapping[str, object], key: str, fallback: Point) -> Point:
    value = overrides.get(key, fallback)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SystemExit(f"the recipe gives {key} as {value!r}, which is not a three-vector")
    return (float(value[0]), float(value[1]), float(value[2]))


def _perturbation(family: str, overrides: dict[str, object]) -> dict[str, object]:
    """One member's perturbation, as the accepted contract's own full record.

    Built through :class:`tests.surface_oracle.Perturbation` rather than
    written out by hand, so ``check_perturbation_label`` refuses a member whose
    declared family is not what its fields actually do - here, at build time,
    rather than as a mislabelled case somebody reads later.
    """
    perturbation = contract.Perturbation(
        label=family,
        depth_noise_codes=_count(overrides, "depth_noise_codes"),
        outlier_period=_count(overrides, "outlier_period"),
        outlier_codes=_count(overrides, "outlier_codes"),
        band_period=_count(overrides, "band_period"),
        band_width=_count(overrides, "band_width"),
        band_shift=_count(overrides, "band_shift"),
        translation_bias=_direction(overrides, "translation_bias", (0.0, 0.0, 0.0)),
        translation_jitter=_amount(overrides, "translation_jitter"),
        rotation_bias_degrees=_amount(overrides, "rotation_bias_degrees"),
        rotation_jitter_degrees=_amount(overrides, "rotation_jitter_degrees"),
        rotation_axis=_direction(overrides, "rotation_axis", (0.0, 1.0, 0.0)),
    )
    contract.check_perturbation_label(perturbation)
    record = perturbation.as_dict()
    # touches_depth/touches_pose are derived properties the accepted case
    # schema does not carry, and a fixture that stated them would be stating a
    # conclusion the contract recomputes anyway.
    record.pop("touches_depth", None)
    record.pop("touches_pose", None)
    return record


def _views(base: dict[str, object], perturbation: dict[str, object], seed: str) -> list[dict]:
    """Render each view once and record what the member's frames actually hash to.

    Exactly the accepted corpus's own rendering path: the same shapes, the same
    intrinsics, the same window and the same ``perturb_codes``. The digests are
    a fact about what this recipe produces, and
    :func:`tests.surface_oracle_harness.build_scene` re-derives and re-checks
    them every time the corpus is loaded.
    """
    scene_spec = base["scene"]
    assert isinstance(scene_spec, dict)
    intrinsics = oracle._intrinsics(scene_spec["intrinsics"], "the base intrinsics")
    window = oracle._window(scene_spec["window"], "the base window")
    depth_scale = float(scene_spec["depth_scale"])
    shapes = [
        oracle._shape(spec, f"the base shape {index}")
        for index, spec in enumerate(scene_spec["shapes"])
    ]
    declared = oracle._perturbation(perturbation, "the member perturbation")
    out: list[dict] = []
    for index, view in enumerate(scene_spec["views"]):
        pose = oracle._render_pose(view["pose"], f"the base view {index} pose")
        rendered = oracle._render(shapes, intrinsics=intrinsics, scale=depth_scale, pose=pose)
        truth_codes = completeness.codes_of(rendered, window)
        truth_frame = completeness.frame_of_codes(truth_codes, window)
        raw_codes = contract.perturb_codes(
            truth_codes, declared, window=window, view=index, seed=f"{seed}:view{index}"
        )
        raw_frame = completeness.frame_of_codes(raw_codes, window)
        out.append(
            {
                "index": index,
                "truth_sha256": hashlib.sha256(truth_frame).hexdigest(),
                "raw_sha256": hashlib.sha256(raw_frame).hexdigest(),
                "truth_returned": sum(1 for code in truth_codes if code != contract.NO_RETURN),
                "raw_returned": sum(1 for code in raw_codes if code != contract.NO_RETURN),
            }
        )
    return out


def build() -> dict[str, str]:
    """Write the population and the manifest. Returns each file's sha256."""
    experiment.check_candidates_are_frozen()
    base_path = oracle.CORPUS / BASE_CASE
    base_bytes = base_path.read_bytes()
    base_digest = hashlib.sha256(base_bytes).hexdigest()
    base = json.loads(base_bytes.decode("utf-8"))

    exact = [member for member in MEMBERS if member[1] == contract.EXACT]
    if len(exact) != 1:
        raise SystemExit(
            f"the recipe declares {len(exact)} exact control(s); exactly one is the control "
            "every perturbed member names"
        )
    control_id = exact[0][0]
    declared_holdout = tuple(
        member_id for member_id, _, split, _ in MEMBERS if split == experiment.HOLDOUT
    )
    if declared_holdout != SEALED_HOLDOUT:
        raise SystemExit(
            f"the recipe's holdout is {list(declared_holdout)} and the seal is "
            f"{list(SEALED_HOLDOUT)}. The sealed members were declared before any of them was "
            "executed and cannot gain, lose or rename one"
        )
    if UPPER_BOUND_MEMBER not in SEALED_HOLDOUT:
        raise SystemExit(
            f"the predeclared upper-bound member {UPPER_BOUND_MEMBER!r} is not a sealed member"
        )

    members: list[dict[str, object]] = []
    for member_id, family, split, overrides in MEMBERS:
        perturbation = _perturbation(family, overrides)
        seed = f"{SEED_PREFIX}:{member_id}"
        members.append(
            {
                "member_id": member_id,
                "family": family,
                "split": split,
                "seed": seed,
                "control_case": None if family == contract.EXACT else control_id,
                "perturbation": perturbation,
                "views": _views(base, perturbation, seed),
            }
        )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    for stale in sorted(DESTINATION.iterdir()):
        if stale.is_file():
            stale.unlink()

    population = {
        "kind": harness.POPULATION_KIND,
        "schema_version": harness.FIXTURE_SCHEMA_VERSION,
        "base_case": BASE_CASE,
        "base_case_sha256": base_digest,
        "members": members,
    }
    text = json.dumps(population, indent=2, sort_keys=True, allow_nan=False) + "\n"
    payload = text.encode("utf-8")
    (DESTINATION / harness.POPULATION_NAME).write_bytes(payload)
    digests = {harness.POPULATION_NAME: hashlib.sha256(payload).hexdigest()}

    manifest = {
        "kind": harness.CORPUS_KIND,
        "schema_version": harness.FIXTURE_SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "what": CORPUS_WHAT,
        "why": CORPUS_WHY,
        "units": CORPUS_UNITS,
        "population": {
            "name": harness.POPULATION_NAME,
            "sha256": digests[harness.POPULATION_NAME],
            "bytes": len(payload),
        },
        "resolutions": list(RESOLUTIONS),
        "selection_resolution": experiment.SELECTION_RESOLUTION,
        "candidate_sha256": experiment.check_candidates_are_frozen(),
        "protocol_sha256": experiment.grid_digest(),
        "selection_protocol": experiment.SELECTION_PROTOCOL,
        "material_fraction": experiment.MATERIAL_FRACTION,
        "work_ceiling": experiment.WORK_CEILING,
        "sealed_holdout": list(SEALED_HOLDOUT),
        "upper_bound": {"member_id": UPPER_BOUND_MEMBER, "cell_id": UPPER_BOUND_CELL},
    }
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    payload = text.encode("utf-8")
    (DESTINATION / harness.MANIFEST_NAME).write_bytes(payload)
    digests[harness.MANIFEST_NAME] = hashlib.sha256(payload).hexdigest()
    return digests


def main() -> int:
    written = build()
    print(f"wrote {len(written)} file(s) to {DESTINATION}")
    for name in sorted(written):
        print(f"  {written[name]}  {name}")
    return 0


if __name__ == "__main__":  # pragma: no cover - the command's own entry point
    raise SystemExit(main())
