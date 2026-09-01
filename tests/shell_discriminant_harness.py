"""Run the observable-shell discriminator experiment, and publish what it found.

One command::

    .venv/bin/python -m tests.shell_discriminant_harness --out shell-discriminant-report.json

Exit codes, and nothing wraps them:

* **0** - every declared member produced its population, every declared rule
  was evaluated on every declared shell, the counts reconcile and the
  recommendation follows from the gates. A run that recommends NO pruning
  exits 0: a negative result is this experiment's result, not its failure;
* **1** - the experiment ran and something did not hold. A member that could
  not be built or scored, a confusion whose classes do not reconcile, an
  unevaluated grid cell, a work ceiling that was exceeded, a counterfactual
  that removed a different shell than the rule selected, or a recommendation
  that its own gates do not support;
* **2** - the experiment could not be run at all: no corpus, a manifest that is
  not a manifest, a population file that is not the one the manifest names, a
  candidate matrix whose digest is not the one the corpus was reviewed with, or
  an output path a report may not be written to.

Why a fourth command
--------------------
:mod:`tests.surface_oracle_harness` establishes THAT the fused surface contains
unsupported closed shells and that some are false surface while others are
correct geometry nothing measured. It deliberately stops there, because the
next question is not a measurement of a mesh: it is whether a bounded rule over
evidence the appliance actually has can tell those two apart. That needs a
train/holdout population, a predeclared candidate matrix, a confusion over
every candidate, and an offline counterfactual on the recommended one - none of
which is a shape the oracle's own case/candidate/gate schema can carry.

So this is a fourth corpus with its own manifest, artifact and strict reader,
built out of the accepted ones rather than beside a copy of them. The scenes,
the perturbations, the frames, the fusion, the extraction and every measurement
are :mod:`tests.surface_oracle`'s; the population design, the observable
records, the rule grid and the counterfactual are
:mod:`tests.shell_discriminant`'s; the path safety, the strict JSON and the
publication path are :mod:`tests.quality_harness`'s.

The population is fixture-owned
-------------------------------
Every member is a whole synthetic realization: one perturbation, one seed, one
set of frames whose digests the fixture records. A member belongs to exactly
one split, so no shell of one realization can appear on both sides of the
train/holdout boundary, and :func:`load_corpus` refuses a corpus in which two
members of different splits render the same frames.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from scanner3d.errors import ScannerError
from scanner3d.tsdf import TsdfVolume
from tests import shell_discriminant as experiment
from tests import surface_oracle as contract
from tests import surface_oracle_harness as oracle
from tests.quality_harness import (
    FixtureUnusable,
    HarnessRefused,
    ReportUnusable,
    clear_destination,
    publish,
)
from tests.surface_oracle_harness import (
    _digest_text as digest_text,
)
from tests.surface_oracle_harness import (
    _exact_keys as exact_keys,
)
from tests.surface_oracle_harness import (
    _integer as integer,
)
from tests.surface_oracle_harness import (
    _loads as loads,
)
from tests.surface_oracle_harness import (
    _mapping as mapping,
)
from tests.surface_oracle_harness import (
    _number as number,
)
from tests.surface_oracle_harness import (
    _text as text,
)

CORPUS: Final = Path(__file__).parent / "fixtures" / "shell_discriminant"
MANIFEST_NAME: Final = "manifest.json"
POPULATION_NAME: Final = "population.json"

FIXTURE_SCHEMA_VERSION: Final = 1
CORPUS_KIND: Final = "scanner3d.shell-discriminant-corpus"
POPULATION_KIND: Final = "scanner3d.shell-discriminant-population"

REPORT_SCHEMA_VERSION: Final = 1
REPORT_KIND: Final = "scanner3d.shell-discriminant-report"

#: The arm every member is fused through beside the declared grid cells. The
#: accepted oracle's name for the shipped-policy resolution, reused rather than
#: renamed, because the shells this experiment exists to separate live there.
APPLIANCE: Final = contract.APPLIANCE_CELL_ID

#: The two recommendations this experiment can reach. There is no third, and in
#: particular there is no "promising": a rule either cleared every declared
#: gate on held-out data and its own counterfactual, or the answer is that dots
#: stay primary and nothing is pruned.
PRUNE: Final = "prune-the-selected-shells"
NO_PRUNING: Final = "no-pruning-keep-dots-primary"

#: What this corpus cannot say, published in the artifact so a reader never has
#: to reconstruct it from the prose.
LIMITATIONS: Final[tuple[str, ...]] = (
    "Synthetic. scanner3d.simulate is not a sensor model, so a separation demonstrated here is a "
    "separation of the arithmetic on exact truth. It is necessary evidence for a product change "
    "and never sufficient.",
    "The POPULATION is the unsupported closed shells, and 'unsupported' is defined against the "
    "scene's ideal returns, which is evaluator knowledge. An appliance cannot enumerate this "
    "population; it can only apply a rule to every closed shell. A deployable rule would have to "
    "be evaluated over all closed shells, and this experiment does not do that.",
    "The observable support frontier (nearest_measured_distance) is measured against the RAW "
    "returns an appliance really has, and is deliberately not the same reference the population "
    "and the labels use.",
    "No physical shell population exists. The recorded captures refuse before fusion at the "
    "shipped readiness policy, so tests.shell_discriminant_replay reports that refusal and makes "
    "no physical discrimination claim.",
    "Nothing here is wired into scanner3d. There is no pruning code path in the product, this "
    "experiment does not add one, and a recommendation from it would still be an owner decision "
    "with its own task.",
)


#: The cost block's explanatory note, verbatim. A module constant rather than a
#: string literal inside the producer, because the strict reader requires the
#: EXACT text: a note is where an account explains itself, and a reader that
#: accepted arbitrary prose there would accept a report whose own explanation
#: of its numbers had been rewritten to fit them.
COST_NOTE: Final = (
    "a deterministic work account, not a wall clock. producer_rule_calls is the ONE "
    "bounded decision-matrix pass - every declared candidate against every declared "
    "unsupported observable, exactly once - and every confusion, cross-tab cell, "
    "aggregate, eligibility test, ordering, gate and recommendation in this report is "
    "a lookup into that matrix rather than a second application of a rule. "
    "counterfactual_rule_calls is the separate application of the selected rule to "
    "ALL closed shells of each sealed holdout appliance population. "
    "reader_audit_rule_calls is a standalone strict reader's own independent audit "
    "matrix and is zero in a producer run. Voxel visits are the accepted oracle's "
    "own work_units - one voxel per integrated view - and triangle samples are the "
    "interior lattice points every mesh was scored at. Wall-clock cost is measured "
    "and reported in docs/shell-discriminant.md instead, because an artifact that "
    "has to be byte-identical between two runs cannot carry a number that differs "
    "between them."
)

#: How the truth-fed ceiling names itself. Never a candidate rule, and the
#: reader refuses an upper bound that does not say so in exactly these words.
UPPER_BOUND_SELECTOR: Final = "analytic truth - EVALUATOR ONLY, never a candidate rule"

#: The two reasons a predeclared upper bound goes unmeasured. Both are facts
#: about the sealed member rather than choices, and both are required verbatim,
#: so "unmeasured" cannot become a place to put a sentence that flatters the
#: result.
UPPER_BOUND_UNBUILT: Final = "the declared upper-bound population could not be built"
UPPER_BOUND_NO_FALSE_SHELL: Final = (
    "the declared upper-bound member produced no false unsupported shell at this resolution, so "
    "there is nothing a perfect prune could have removed"
)


# --- the corpus -------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Member:
    """One whole synthetic realization: a perturbation, a seed and its frames.

    The frames are not committed - a family of 512000-byte frames is not
    reviewable JSON - so what is committed is the perturbation, the seed and
    the SHA-256 of every truth and measured frame they render.
    :func:`tests.surface_oracle_harness.build_scene` refuses the member if what
    it renders does not hash to what the fixture says, which is the same
    altered-truth control the accepted corpus has.
    """

    member_id: str
    family: str
    split: str
    seed: str
    control_case: str | None
    perturbation: Mapping[str, object]
    views: tuple[Mapping[str, object], ...]

    @property
    def frame_digest(self) -> str:
        """A digest over every frame this member declares.

        The split-leakage control is built on this: two members of different
        splits that render the same frames are the same realization under two
        names, and :func:`load_corpus` refuses that outright.
        """
        payload = json.dumps(
            [[view.get("truth_sha256"), view.get("raw_sha256")] for view in self.views],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Corpus:
    """A manifest and a population file that have been checked against each other."""

    root: Path
    manifest_sha256: str
    population_sha256: str
    base_case: str
    base_case_sha256: str
    base_document: Mapping[str, object]
    resolutions: tuple[str, ...]
    #: The one resolution eligibility, ordering and the holdout gates read.
    selection_resolution: str
    members: tuple[Member, ...]
    #: The exact holdout member ids the manifest seals, in declared order.
    sealed_holdout: tuple[str, ...]
    #: The candidate bodies alone, frozen before the sealed holdout ran.
    candidate_sha256: str
    #: The candidates plus the selection protocol and the gate names. Allowed
    #: to move when the protocol is legitimately rewritten; the candidate
    #: digest above is not.
    protocol_sha256: str
    selection_protocol: str
    material_fraction: float
    work_ceiling: int
    upper_bound_member: str
    upper_bound_cell: str

    def member(self, member_id: str) -> Member:
        for candidate in self.members:
            if candidate.member_id == member_id:
                return candidate
        raise HarnessRefused(f"{member_id!r} is not a member of this corpus")


def _regular_file(path: Path, what: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise HarnessRefused(
            f"{what} at {path} is not a regular file. A fixture is evidence and must be the "
            "bytes in the corpus, not a pointer to bytes somewhere else"
        )
    return path.read_bytes()


def _view_record(spec: object, index: int, what: str) -> Mapping[str, object]:
    record = mapping(spec, f"{what} view {index}")
    exact_keys(
        record,
        ("index", "truth_sha256", "raw_sha256", "truth_returned", "raw_returned"),
        f"{what} view {index}",
    )
    if integer(record, "index", f"{what} view {index}") != index:
        raise FixtureUnusable(f"{what} view {index} states an index that is not its position")
    digest_text(record, "truth_sha256", f"{what} view {index}")
    digest_text(record, "raw_sha256", f"{what} view {index}")
    return record


def _member(spec: object, what: str) -> Member:
    record = mapping(spec, what)
    exact_keys(
        record,
        ("member_id", "family", "split", "seed", "control_case", "perturbation", "views"),
        what,
    )
    member_id = text(record, "member_id", what)
    if not oracle.CELL_ID.match(member_id):
        raise FixtureUnusable(f"{what} member id {member_id!r} is not a plain lower-case name")
    family = text(record, "family", what)
    if family not in contract.PERTURBATIONS:
        raise FixtureUnusable(
            f"{what} declares family {family!r}; the accepted oracle defines "
            f"{list(contract.PERTURBATIONS)} and this experiment reports by that vocabulary"
        )
    split = text(record, "split", what)
    if split not in experiment.SPLITS:
        raise FixtureUnusable(f"{what} declares split {split!r}, not one of {experiment.SPLITS}")
    control = record.get("control_case")
    if control is not None and not isinstance(control, str):
        raise FixtureUnusable(f"{what} 'control_case' is {control!r}, not text or null")
    views = record.get("views")
    if not isinstance(views, list) or not views:
        raise FixtureUnusable(f"{what} names no views")
    return Member(
        member_id=member_id,
        family=family,
        split=split,
        seed=text(record, "seed", what),
        control_case=control,
        perturbation=mapping(record.get("perturbation"), f"{what} 'perturbation'"),
        views=tuple(_view_record(view, index, what) for index, view in enumerate(views)),
    )


def load_corpus(root: Path = CORPUS) -> Corpus:
    """Read the manifest, the population and the base case, and check all three.

    Seven things beyond reading the files, and each of them is a control:

    * the CANDIDATE-only digest must be exactly
      :data:`tests.shell_discriminant.FROZEN_CANDIDATE_DIGEST`. The candidate
      vocabulary, directions, thresholds, pairs, order and 195 rule bodies are
      inside it and the selection protocol and gate names are deliberately not,
      so a body that moved after the sealed holdout was declared is a refusal
      here while a protocol sentence that was legitimately rewritten this round
      is not;
    * the PROTOCOL digest must be exactly
      :func:`tests.shell_discriminant.grid_digest`, so the protocol and gates
      the corpus was reviewed with are the ones in the module;
    * the declared selection resolution must be the module's, and it must be
      one of the declared resolutions;
    * the population file must hash to what the manifest records, and the base
      surface-oracle case must hash to what the population records. The scene
      geometry is the accepted corpus's, byte for byte;
    * no two members of DIFFERENT splits may render the same frames. That is
      the fixture-level scene-leakage control; the run adds a stronger one over
      the whole reconstruction input;
    * exactly one member may be the exact control, every other member must name
      it, both splits must be non-empty, and the HOLDOUT half must be exactly
      the sealed member list the manifest records;
    * the declared resolutions must be the appliance arm plus grid cells the
      base case actually declares, so a resolution nobody could run is refused
      before anything is fused.
    """
    experiment.check_candidates_are_frozen()
    if not root.is_dir():
        raise HarnessRefused(f"{root} is not a directory; there is no corpus to run")
    manifest_bytes = _regular_file(root / MANIFEST_NAME, "the manifest")
    document = mapping(
        loads(manifest_bytes.decode("utf-8"), str(root / MANIFEST_NAME)),
        str(root / MANIFEST_NAME),
    )
    what = str(root / MANIFEST_NAME)
    exact_keys(
        document,
        (
            "kind",
            "schema_version",
            "generated_by",
            "what",
            "why",
            "units",
            "population",
            "resolutions",
            "selection_resolution",
            "candidate_sha256",
            "protocol_sha256",
            "selection_protocol",
            "material_fraction",
            "work_ceiling",
            "sealed_holdout",
            "upper_bound",
        ),
        what,
    )
    if text(document, "kind", what) != CORPUS_KIND:
        raise HarnessRefused(f"{what} is not a {CORPUS_KIND}")
    if integer(document, "schema_version", what) != FIXTURE_SCHEMA_VERSION:
        raise HarnessRefused(f"{what} is schema {document['schema_version']!r}")
    recorded_candidates = digest_text(document, "candidate_sha256", what)
    if recorded_candidates != experiment.FROZEN_CANDIDATE_DIGEST:
        raise HarnessRefused(
            f"{what} records candidate digest {recorded_candidates}; the bodies were frozen at "
            f"{experiment.FROZEN_CANDIDATE_DIGEST} before the sealed holdout was declared"
        )
    recorded_protocol = digest_text(document, "protocol_sha256", what)
    if recorded_protocol != experiment.grid_digest():
        raise HarnessRefused(
            f"{what} was reviewed against protocol digest {recorded_protocol}; "
            f"tests.shell_discriminant now hashes to {experiment.grid_digest()}. The grid, its "
            "pairs, the selection protocol and the gate names are all inside that digest, so "
            "this is either a protocol that moved after a result or a fixture that was never "
            "updated - and neither one may run"
        )
    if text(document, "selection_protocol", what) != experiment.SELECTION_PROTOCOL:
        raise HarnessRefused(
            f"{what} states a selection protocol that is not the reviewed one, verbatim"
        )
    selection_resolution = text(document, "selection_resolution", what)
    if selection_resolution != experiment.SELECTION_RESOLUTION:
        raise HarnessRefused(
            f"{what} declares selection resolution {selection_resolution!r}; the module asks the "
            f"product question at {experiment.SELECTION_RESOLUTION!r} and nowhere else"
        )
    if number(document, "material_fraction", what) != experiment.MATERIAL_FRACTION:
        raise HarnessRefused(f"{what} states a material fraction the module does not")
    if integer(document, "work_ceiling", what) != experiment.WORK_CEILING:
        raise HarnessRefused(f"{what} states a work ceiling the module does not")

    sealed_spec = document.get("sealed_holdout")
    if not isinstance(sealed_spec, list) or not sealed_spec:
        raise HarnessRefused(f"{what} records no sealed holdout")
    sealed: list[str] = []
    for item in sealed_spec:
        if not isinstance(item, str) or not item:
            raise FixtureUnusable(f"{what} lists sealed member {item!r}, which is not a name")
        sealed.append(item)
    if len(set(sealed)) != len(sealed):
        raise HarnessRefused(f"{what} lists a sealed member twice")

    population_spec = mapping(document.get("population"), f"{what} 'population'")
    exact_keys(population_spec, ("name", "sha256", "bytes"), f"{what} 'population'")
    population_name = text(population_spec, "name", f"{what} 'population'")
    if population_name != POPULATION_NAME:
        raise HarnessRefused(f"{what} names population file {population_name!r}")
    population_bytes = _regular_file(root / POPULATION_NAME, "the population")
    actual = hashlib.sha256(population_bytes).hexdigest()
    if actual != digest_text(population_spec, "sha256", f"{what} 'population'"):
        raise HarnessRefused(
            f"{root / POPULATION_NAME} hashes to {actual}, which is not what the manifest records"
        )
    if len(population_bytes) != integer(population_spec, "bytes", f"{what} 'population'"):
        raise HarnessRefused(f"{root / POPULATION_NAME} is not the length the manifest records")

    population = mapping(
        loads(population_bytes.decode("utf-8"), str(root / POPULATION_NAME)),
        str(root / POPULATION_NAME),
    )
    where = str(root / POPULATION_NAME)
    exact_keys(
        population,
        ("kind", "schema_version", "base_case", "base_case_sha256", "members"),
        where,
    )
    if text(population, "kind", where) != POPULATION_KIND:
        raise HarnessRefused(f"{where} is not a {POPULATION_KIND}")
    base_case = text(population, "base_case", where)
    base_path = oracle.CORPUS / base_case
    if not oracle.FIXTURE_NAME.match(base_case) or base_path.resolve().parent != (
        oracle.CORPUS.resolve()
    ):
        raise HarnessRefused(
            f"{where} names base case {base_case!r}, which is not a case of the accepted "
            "surface-oracle corpus"
        )
    base_bytes = _regular_file(base_path, "the base case")
    base_digest = hashlib.sha256(base_bytes).hexdigest()
    if base_digest != digest_text(population, "base_case_sha256", where):
        raise HarnessRefused(
            f"{base_path} hashes to {base_digest}; the population was built against "
            f"{population['base_case_sha256']!r}. This experiment reuses the accepted corpus's "
            "scene geometry byte for byte and will not run against a different one"
        )
    base_document = mapping(loads(base_bytes.decode("utf-8"), str(base_path)), str(base_path))

    members_spec = population.get("members")
    if not isinstance(members_spec, list) or not members_spec:
        raise HarnessRefused(f"{where} declares no members")
    members = tuple(
        _member(spec, f"{where} member {index}") for index, spec in enumerate(members_spec)
    )
    seen: set[str] = set()
    for member in members:
        if member.member_id in seen:
            raise HarnessRefused(f"{where} declares member {member.member_id!r} twice")
        seen.add(member.member_id)
    exact = [member for member in members if member.family == contract.EXACT]
    if len(exact) != 1:
        raise HarnessRefused(
            f"{where} declares {len(exact)} exact control(s). Exactly one is the control every "
            "perturbed member is compared with, and a population without one cannot show that "
            "an unperturbed scene produces no unsupported shell at all"
        )
    for member in members:
        if member.family == contract.EXACT:
            if member.control_case is not None:
                raise HarnessRefused(f"{where}: the exact control names a control of its own")
            continue
        if member.control_case != exact[0].member_id:
            raise HarnessRefused(
                f"{where}: {member.member_id} names control {member.control_case!r}; the corpus's "
                f"exact control is {exact[0].member_id!r}"
            )
    for split in experiment.SPLITS:
        if not [member for member in members if member.split == split]:
            raise HarnessRefused(f"{where} puts no member in the {split!r} split")
    declared_holdout = [
        member.member_id for member in members if member.split == experiment.HOLDOUT
    ]
    if declared_holdout != sealed:
        raise HarnessRefused(
            f"{where} puts {declared_holdout} in the holdout while the manifest seals {sealed}. "
            "The sealed members were declared before any of them was executed, and a holdout "
            "that has gained, lost or renamed one is not the sealed holdout"
        )
    by_frames: dict[str, Member] = {}
    for member in members:
        other = by_frames.get(member.frame_digest)
        if other is not None and other.split != member.split:
            raise HarnessRefused(
                f"{where}: {member.member_id} and {other.member_id} render identical frames and "
                f"are in different splits. That is one synthetic realization on both sides of the "
                "train/holdout boundary, which is the leak this split exists to prevent"
            )
        by_frames.setdefault(member.frame_digest, member)

    resolutions_spec = document.get("resolutions")
    if not isinstance(resolutions_spec, list) or not resolutions_spec:
        raise HarnessRefused(f"{what} declares no resolutions")
    declared_cells = set(_base_grid_ids())
    resolutions: list[str] = []
    for item in resolutions_spec:
        if not isinstance(item, str):
            raise FixtureUnusable(f"{what} lists resolution {item!r}, which is not text")
        if item != APPLIANCE and item not in declared_cells:
            raise HarnessRefused(
                f"{what} declares resolution {item!r}, which is neither the appliance arm nor a "
                f"cell the base case runs ({sorted(declared_cells)})"
            )
        resolutions.append(item)
    if len(set(resolutions)) != len(resolutions):
        raise HarnessRefused(f"{what} declares a resolution twice")
    if len(resolutions) < 2:
        raise HarnessRefused(
            f"{what} declares {len(resolutions)} resolution(s). The report is required to "
            "separate its confusion counts BY resolution, which one resolution cannot do"
        )
    if selection_resolution not in resolutions:
        raise HarnessRefused(
            f"{what} selects at {selection_resolution!r}, which it does not declare as a resolution"
        )

    bound = mapping(document.get("upper_bound"), f"{what} 'upper_bound'")
    exact_keys(bound, ("member_id", "cell_id"), f"{what} 'upper_bound'")
    upper_member = text(bound, "member_id", f"{what} 'upper_bound'")
    upper_cell = text(bound, "cell_id", f"{what} 'upper_bound'")
    if upper_member not in seen:
        raise HarnessRefused(f"{what} names upper-bound member {upper_member!r}")
    if upper_cell not in resolutions:
        raise HarnessRefused(f"{what} names upper-bound resolution {upper_cell!r}")

    return Corpus(
        root=root,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        population_sha256=actual,
        base_case=base_case,
        base_case_sha256=base_digest,
        base_document=base_document,
        resolutions=tuple(resolutions),
        selection_resolution=selection_resolution,
        members=members,
        sealed_holdout=tuple(sealed),
        candidate_sha256=recorded_candidates,
        protocol_sha256=recorded_protocol,
        selection_protocol=text(document, "selection_protocol", what),
        material_fraction=number(document, "material_fraction", what),
        work_ceiling=integer(document, "work_ceiling", what),
        upper_bound_member=upper_member,
        upper_bound_cell=upper_cell,
    )


def _base_grid_ids() -> tuple[str, ...]:
    """The grid cell ids the accepted corpus declares, read from its own manifest."""
    manifest = mapping(
        loads(
            (oracle.CORPUS / oracle.MANIFEST_NAME).read_text(encoding="utf-8"),
            str(oracle.CORPUS / oracle.MANIFEST_NAME),
        ),
        str(oracle.CORPUS / oracle.MANIFEST_NAME),
    )
    grid = manifest.get("grid")
    if not isinstance(grid, list):
        raise HarnessRefused("the accepted surface-oracle manifest declares no grid")
    return tuple(
        str(mapping(cell, "a surface-oracle grid cell")["cell_id"])
        for cell in grid
        if isinstance(cell, Mapping)
    )


# --- building one member's scene from the accepted corpus's geometry --------------------


def build_member(member: Member, corpus: Corpus) -> contract.Scene:
    """Assemble one member's case document and hand it to the accepted builder.

    The scene - intrinsics, window, depth scale, shapes, volume, regions and
    the five view POSES - is the accepted base case's, unchanged. What a member
    contributes is its perturbation, its seed and the digests of the frames
    those two render. :func:`tests.surface_oracle_harness.build_scene` does the
    rendering and the digest check, so a member whose declared frames are not
    the frames its own perturbation produces is refused there rather than
    quietly benchmarked.
    """
    document = copy.deepcopy(dict(corpus.base_document))
    document["case_id"] = member.member_id
    document["control_case"] = member.control_case
    document["perturbation"] = dict(member.perturbation)
    scene_spec = document.get("scene")
    if not isinstance(scene_spec, dict):
        raise FixtureUnusable(f"{corpus.base_case} carries no scene to build {member.member_id}")
    scene_spec["seed"] = member.seed
    views = scene_spec.get("views")
    if not isinstance(views, list) or len(views) != len(member.views):
        raise FixtureUnusable(
            f"{member.member_id} declares {len(member.views)} view(s) and the base case has "
            f"{len(views) if isinstance(views, list) else 'none'}"
        )
    for index, (base_view, declared) in enumerate(zip(views, member.views, strict=True)):
        if not isinstance(base_view, dict):
            raise FixtureUnusable(f"{corpus.base_case} view {index} is not an object")
        base_view.update(
            {
                "truth_sha256": declared["truth_sha256"],
                "raw_sha256": declared["raw_sha256"],
                "truth_returned": declared["truth_returned"],
                "raw_returned": declared["raw_returned"],
            }
        )
    return oracle.build_scene(
        document,
        grid=_grid_cells(),
        work_budget=_work_budget(),
        appliance=_appliance_policy(),
        what=f"member {member.member_id}",
    )


def _accepted_manifest() -> Mapping[str, object]:
    path = oracle.CORPUS / oracle.MANIFEST_NAME
    return mapping(loads(path.read_text(encoding="utf-8"), str(path)), str(path))


def _grid_cells() -> tuple[contract.GridCell, ...]:
    manifest = _accepted_manifest()
    grid = manifest.get("grid")
    if not isinstance(grid, list):
        raise HarnessRefused("the accepted surface-oracle manifest declares no grid")
    return tuple(oracle._grid_cell(cell, "a surface-oracle grid cell") for cell in grid)


def _work_budget() -> int:
    return integer(_accepted_manifest(), "work_budget", "the surface-oracle manifest")


def _appliance_policy() -> contract.AppliancePolicy:
    return oracle._appliance_policy(
        _accepted_manifest().get("appliance"), "the surface-oracle appliance policy"
    )


def _fusion_backend(cell_id: str) -> str:
    """Which backend one arm fuses on. The POLICY, in one place.

    The appliance arm is the product-facing one and runs the appliance
    backend; every diagnostic cell runs the reference backend. Nothing in the
    corpus says so - this is a policy of the harness - and it decides which
    population comes out, so :func:`_evidence_key` evaluates it over every
    declared resolution instead of naming a constant beside it. Moving the
    diagnostic arm onto the appliance backend is then a different cache entry
    even though every committed byte is unchanged.
    """
    return contract.APPLIANCE_BACKEND if cell_id == APPLIANCE else contract.BACKEND


def _arm(
    scene: contract.Scene, cell_id: str
) -> tuple[contract.ArmResult, TsdfVolume, contract.GridCell]:
    """One member at one resolution, the FIELD its surface came from, and the cell.

    The field is kept because the observable weight summary has to be read out
    of the volume the surface was extracted from. Reading it out of a second
    fusion would be evidence about a different field - and since round two the
    fusion memo's own key binds the fused poses, the intrinsics, the depth
    scale, the volume box and every cell parameter, so "the same fusion" is a
    checkable claim rather than a naming convention.
    """
    cell = contract.appliance_cell(scene) if cell_id == APPLIANCE else scene.cell(cell_id)
    volume, vertices, record = contract.fuse_keeping_volume(
        scene, cell, backend=_fusion_backend(cell_id)
    )
    result = (
        contract.fused_at_appliance_resolution(scene)
        if cell_id == APPLIANCE
        else contract.fused(scene, cell_id)
    )
    if tuple(result.points) != tuple(vertices) or tuple(result.faces) != tuple(record.faces):
        raise HarnessRefused(
            f"the kept-volume fusion of {scene.case_id}/{cell_id} does not produce the surface "
            "the arm published. The weights under every shell would then be read out of a "
            "different field than the mesh came from, which is the whole point of keeping it"
        )
    return result, volume, cell


def _result_only(scene: contract.Scene, cell_id: str) -> contract.ArmResult:
    """The same arm's surface without paying for the volume again."""
    return (
        contract.fused_at_appliance_resolution(scene)
        if cell_id == APPLIANCE
        else contract.fused(scene, cell_id)
    )


# --- running the experiment ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Built:
    """One member at one resolution, kept together for the counterfactual pass."""

    member: Member
    cell_id: str
    population: experiment.Population


def _populations(corpus: Corpus) -> tuple[list[Built], list[str]]:
    """Every member at every declared resolution. Members are independent.

    A member that cannot be built or scored is recorded as a failure and the
    next one still runs, so a broken fixture costs one member's evidence rather
    than the whole experiment's.

    The two halves are visibly ordered here: ``observable_inputs`` projects the
    scene down to what an appliance has, ``derive_observables`` produces every
    closed shell from that projection alone, and only then does ``evaluate``
    bring the ideal support and the analytic scoring in.
    """
    built: list[Built] = []
    failures: list[str] = []
    for member in corpus.members:
        try:
            scene = build_member(member, corpus)
        except Exception as error:  # recorded against this member, not swallowed
            failures.append(f"{member.member_id}: {type(error).__name__}: {error}")
            continue
        support = contract.truth_points(scene)
        for cell_id in corpus.resolutions:
            try:
                result, volume, cell = _arm(scene, cell_id)
                # The backend the field was ACTUALLY produced on, read off the
                # volume rather than re-derived, so the reconstruction identity
                # binds what ran instead of what the policy would have picked.
                inputs = experiment.observable_inputs(
                    scene, result, cell=cell, volume=volume, backend=volume.backend
                )
                observables = experiment.derive_observables(inputs)
                built.append(
                    Built(
                        member=member,
                        cell_id=cell_id,
                        population=experiment.evaluate(
                            scene,
                            result,
                            observables,
                            inputs,
                            support=support,
                            member_id=member.member_id,
                            family=member.family,
                            split=member.split,
                        ),
                    )
                )
            except Exception as error:  # recorded against this member, not swallowed
                failures.append(f"{member.member_id}/{cell_id}: {type(error).__name__}: {error}")
    return built, failures


def _check_realizations(built: Sequence[Built]) -> None:
    """No reconstruction input may appear twice, and never across the split.

    Round one hashed frame digests alone, which left the fused poses - the
    other half of what a reconstruction is a function of - outside the identity.
    This binds the ordered raw bytes, the poses actually used, the intrinsics,
    the window, the depth scale and bounds and the cell, so two populations
    that came from the same reconstruction problem are one realization whatever
    their members are called.
    """
    seen: dict[str, Built] = {}
    for item in built:
        other = seen.get(item.population.reconstruction_digest)
        if other is not None:
            raise HarnessRefused(
                f"{item.member.member_id}/{item.cell_id} and "
                f"{other.member.member_id}/{other.cell_id} have the same reconstruction-input "
                f"identity {item.population.reconstruction_digest}. They are one realization "
                "under two names"
                + (
                    ", and they are on opposite sides of the train/holdout boundary"
                    if other.population.split != item.population.split
                    else ""
                )
            )
        seen[item.population.reconstruction_digest] = item


def _check_sealed_population(records: Sequence[experiment.ShellRecord]) -> None:
    """The sealed appliance holdout must contain BOTH classes, or it is refused.

    Not tuned into existence. The eight sealed members, their seeds and their
    magnitudes were declared by the planner before any of them was executed; if
    the population they produce cannot grade a rule on both false and correct
    unsupported surface, the honest move is to hand it back for planner review
    rather than to widen a magnitude until it does.
    """
    holdout = experiment.holdout_records(records)
    false_faces = sum(r.truth.faces for r in holdout if r.truth.false_surface)
    correct_faces = sum(r.truth.faces for r in holdout if not r.truth.false_surface)
    if not false_faces or not correct_faces:
        raise HarnessRefused(
            f"the sealed {experiment.SELECTION_RESOLUTION} holdout carries {false_faces} false "
            f"and {correct_faces} correct unsupported shell face(s). A holdout that cannot "
            "measure both the gain and the damage grades nothing. The declared members, seeds, "
            "magnitudes, candidate bodies, gates and work ceiling are frozen and are not the "
            "numbers to move: this population is refused and returned for planner review"
        )


# --- one canonical rebuild of the committed experiment ------------------------------------


@dataclass(frozen=True, slots=True)
class _Evidence:
    """The committed experiment, rebuilt from the corpus and from nothing else.

    This is what the standalone strict reader binds a report to. Round two's
    reader rebuilt typed values out of the REPORT and then checked that the
    report agreed with itself, so a forgery that stayed internally coherent -
    another valid-looking manifest digest, a doubled ray count with its
    fractions preserved, a fabricated reconstruction identity - was reproduced
    rather than refused. Everything here is derived from :func:`load_corpus`
    over committed bytes and from the module's own protocol, and a published
    document is compared against it.

    It is expensive: every declared member is fused at every declared
    resolution. :func:`_evidence` therefore memoises one bundle per process
    under a key made only of committed digests and code-level protocol
    identity, so a producer run and a strict read in the same process pay for
    it once and a second corpus can never reach the first one's entry.
    """

    corpus: Corpus
    built: tuple[Built, ...]
    failures: tuple[str, ...]
    #: The (member, resolution) pairs, in the order the corpus declares them.
    cells: tuple[tuple[str, str], ...]
    populations: Mapping[tuple[str, str], experiment.Population]
    records: tuple[experiment.ShellRecord, ...]
    #: The canonical blocks, already in the shape a published artifact carries
    #: them, so the reader compares like with like rather than re-deriving the
    #: producer's serialisation at every comparison.
    corpus_block: Mapping[str, object]
    realization_block: tuple[Mapping[str, object], ...]
    population_block: tuple[Mapping[str, object], ...]
    upper_bound_block: Mapping[str, object] | None
    #: How a counterfactual on one canonical population is RE-MEASURED. Not a
    #: block, because which populations owe one depends on the rule the reader
    #: independently selects. The trusted rebuild re-fuses the member, prunes
    #: exactly those shells and re-measures the mesh with the accepted oracle's
    #: own code, so a published before/after is bound to a mesh rather than
    #: only to itself. It is called for no population at all when no rule
    #: qualifies, which is what this corpus produces.
    remeasure: Callable[[experiment.Population, Sequence[str]], Mapping[str, object]]


def _json_ready(value: object) -> Any:
    """The same value as a published artifact carries it.

    A block assembled in this process holds Python tuples where JSON holds
    arrays, and a reader comparing a decoded report against it would refuse an
    honest report over a difference the wire cannot express. One round trip
    through the encoder the artifact is published with removes exactly that
    difference: ``allow_nan=False`` keeps it from inventing a value the
    artifact could not carry.
    """
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _accepted_manifest_digest() -> str:
    """The accepted surface-oracle manifest's own bytes.

    Every member is built through that manifest's grid cells, work budget and
    appliance policy, so it is a committed input of this rebuild even though it
    lives in the other corpus's directory.
    """
    path = oracle.CORPUS / oracle.MANIFEST_NAME
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corpus_block(corpus: Corpus) -> dict[str, object]:
    """The corpus, exactly as a report publishes it. One writer, one shape."""
    return {
        "kind": CORPUS_KIND,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "manifest_sha256": corpus.manifest_sha256,
        "population_sha256": corpus.population_sha256,
        "base_case": corpus.base_case,
        "base_case_sha256": corpus.base_case_sha256,
        "candidate_sha256": corpus.candidate_sha256,
        "protocol_sha256": corpus.protocol_sha256,
        "resolutions": list(corpus.resolutions),
        "selection_resolution": corpus.selection_resolution,
        "material_fraction": corpus.material_fraction,
        "work_ceiling": corpus.work_ceiling,
        "sealed_holdout": list(corpus.sealed_holdout),
        "upper_bound": {
            "member_id": corpus.upper_bound_member,
            "cell_id": corpus.upper_bound_cell,
        },
        "members": [
            {
                "member_id": member.member_id,
                "family": member.family,
                "split": member.split,
                "seed": member.seed,
                "control_case": member.control_case,
                "frame_digest": member.frame_digest,
                "perturbation": dict(member.perturbation),
            }
            for member in corpus.members
        ],
    }


def _realization_block(built: Sequence[Built]) -> list[dict[str, object]]:
    """One realization identity per built population, in the declared order."""
    return [
        {
            "member_id": item.member.member_id,
            "cell_id": item.cell_id,
            "split": item.population.split,
            "family": item.population.family,
            "reconstruction_digest": item.population.reconstruction_digest,
        }
        for item in built
    ]


_EVIDENCE: dict[tuple[object, ...], _Evidence] = {}


def _evidence_key(corpus: Corpus) -> tuple[object, ...]:
    """What a canonical rebuild is a function of: committed bytes, and code.

    Committed inputs only - the manifest, the population recipe and the base
    surface-oracle case by digest, the frozen candidate bodies, the protocol
    digest and every declaration the manifest seals - together with the
    module-level identity of the protocol itself, including the fusion policy
    the populations are actually produced on: the backend this harness runs
    each declared resolution's arm at, evaluated rather than named, so moving
    the diagnostic arm from the reference backend to the appliance one is a
    different key even though no committed byte moved. There is no filesystem
    path, no clock and nothing a report could supply anywhere in it, so a
    cached bundle cannot be mistaken for a bundle of a different corpus and no
    caller can reach an entry by describing one.
    """
    return (
        FIXTURE_SCHEMA_VERSION,
        REPORT_SCHEMA_VERSION,
        CORPUS_KIND,
        POPULATION_KIND,
        REPORT_KIND,
        experiment.SCHEMA_VERSION,
        corpus.manifest_sha256,
        corpus.population_sha256,
        corpus.base_case,
        corpus.base_case_sha256,
        corpus.candidate_sha256,
        corpus.protocol_sha256,
        _accepted_manifest_digest(),
        corpus.resolutions,
        corpus.selection_resolution,
        corpus.sealed_holdout,
        corpus.material_fraction,
        corpus.work_ceiling,
        corpus.upper_bound_member,
        corpus.upper_bound_cell,
        experiment.FROZEN_CANDIDATE_DIGEST,
        experiment.grid_digest(),
        experiment.SELECTION_PROTOCOL,
        experiment.GATES,
        experiment.SELECTION_RESOLUTION,
        experiment.MATERIAL_FRACTION,
        experiment.WORK_CEILING,
        experiment.OBSERVABLE_SOURCE,
        experiment.LABEL_SOURCE,
        experiment.FUSION_BACKENDS,
        tuple((cell_id, _fusion_backend(cell_id)) for cell_id in corpus.resolutions),
        LIMITATIONS,
        COST_NOTE,
        UPPER_BOUND_SELECTOR,
        UPPER_BOUND_UNBUILT,
        UPPER_BOUND_NO_FALSE_SHELL,
    )


def _remeasure_counterfactual(
    corpus: Corpus, item: Built, selected: Sequence[str]
) -> Mapping[str, object]:
    """Prune exactly those shells from that member's real mesh, and re-measure it.

    The accepted surface oracle's own measurement code over the same scene, so
    a published ``before`` / ``after`` is bound to a mesh somebody can rebuild
    rather than to the rest of the entry agreeing with it. Nothing here reads
    the report.
    """
    scene = build_member(item.member, corpus)
    result = _result_only(scene, item.cell_id)
    measured = experiment.counterfactual(
        scene,
        result,
        item.population.observables,
        item.population.records,
        selected,
        member_id=item.member.member_id,
        cell_id=item.cell_id,
        split=item.population.split,
        reconstruction_digest=item.population.reconstruction_digest,
    )
    return _json_ready(measured.as_dict())


def _build_evidence(corpus: Corpus) -> _Evidence:
    """Fuse every declared member at every declared resolution, and label them.

    The producer's build and the reader's trusted rebuild are the same
    derivation because they are the same code, and neither of them reads a
    document: what comes out is a function of the corpus alone.
    """
    built, failures = _populations(corpus)
    _check_realizations(built)
    records = [record for item in built for record in item.population.records]
    try:
        experiment.preflight(records)
    except experiment.DiscriminantUnusable as error:
        raise HarnessRefused(str(error)) from error
    _check_sealed_population(records)
    by_cell = {(item.member.member_id, item.cell_id): item for item in built}

    def remeasure(
        population: experiment.Population, selected: Sequence[str]
    ) -> Mapping[str, object]:
        return _remeasure_counterfactual(
            corpus, by_cell[(population.member_id, population.cell_id)], selected
        )

    return _Evidence(
        corpus=corpus,
        built=tuple(built),
        failures=tuple(failures),
        cells=tuple((item.member.member_id, item.cell_id) for item in built),
        populations={(item.member.member_id, item.cell_id): item.population for item in built},
        records=tuple(records),
        corpus_block=_json_ready(_corpus_block(corpus)),
        realization_block=tuple(_json_ready(_realization_block(built))),
        population_block=tuple(_json_ready([item.population.as_dict() for item in built])),
        upper_bound_block=_json_ready(_upper_bound(corpus, built)),
        remeasure=remeasure,
    )


def _evidence(corpus: Corpus) -> _Evidence:
    """One canonical rebuild per set of committed inputs, built at most once."""
    key = _evidence_key(corpus)
    bundle = _EVIDENCE.get(key)
    if bundle is None:
        bundle = _build_evidence(corpus)
        _EVIDENCE[key] = bundle
    return bundle


def _trusted_evidence() -> _Evidence:
    """The canonical rebuild of the COMMITTED corpus, and of nothing else.

    Deliberately without a parameter. The public strict reader is a claim about
    the experiment this repository committed, so the evidence root it checks a
    report against is not something a caller may supply: a report for a
    corrupted copy under a temporary directory is not a report this reader
    accepts, however faithfully it describes that copy. :func:`load_corpus`
    remains the named API for checking a copied or corrupted fixture root.
    """
    return _evidence(load_corpus())


def run(root: Path = CORPUS) -> dict[str, object]:
    """Build the population, evaluate every candidate ONCE, and publish the artifact.

    Nothing in the returned document depends on the machine that produced it.
    There is no wall clock anywhere in it: the cost account is a deterministic
    work account, so two runs of this command produce byte-identical artifacts
    and a byte comparison is a real determinism check rather than a check that
    two clocks agreed.

    One bounded pass. Every candidate is applied to every declared unsupported
    observable exactly once, into :class:`~tests.shell_discriminant.DecisionMatrix`,
    and the twenty-eight cross-tab cells, the aggregates, the eligibility test,
    the ordering, the holdout gates and the recommendation are all lookups into
    it. The only other rule invocations in this command are the counterfactual's
    application of the SELECTED rule to every closed shell, and they are
    counted and published separately.
    """
    experiment.reset_rule_calls()
    corpus = load_corpus(root)
    # The producer assembles its document from ONE canonical rebuild, and the
    # strict read at the bottom binds that document to the same rebuild rather
    # than to the document's own claims. There is no producer/reader recursion
    # in that: the bundle is a function of the corpus, the assembly reads the
    # bundle, and the reader reads the bundle and the finished document.
    bundle = _evidence(corpus)
    built = list(bundle.built)
    failures = list(bundle.failures)
    populations = [item.population for item in built]
    records = list(bundle.records)

    matrix = experiment.build_decision_matrix(records)
    aggregates = experiment.aggregate_slices(records)
    candidates: list[dict[str, object]] = []
    for rule in experiment.RULE_GRID:
        cells = experiment.cross_tab(matrix, rule.rule_id, records, resolutions=corpus.resolutions)
        candidates.append(
            {
                "rule": rule.as_dict(),
                "cross_tab": {cell.slice_name: cell.as_dict() for cell in cells},
                "aggregates": {
                    name: experiment.confusion(
                        matrix, rule.rule_id, subset, slice_name=name
                    ).as_dict()
                    for name, subset in aggregates.items()
                },
            }
        )

    selection = experiment.choose(matrix, records)
    holdout = experiment.holdout_records(records)
    training = experiment.training_records(records)
    holdout_confusion = (
        experiment.confusion(
            matrix,
            selection.rule.rule_id,
            holdout,
            slice_name=f"split:{experiment.HOLDOUT}|cell:{corpus.selection_resolution}",
        ).as_dict()
        if selection.rule is not None
        else None
    )

    counterfactuals: list[dict[str, object]] = []
    gate_results: dict[str, bool] = {}
    if selection.rule is not None:
        train_confusion = experiment.confusion(
            matrix,
            selection.rule.rule_id,
            training,
            slice_name=f"split:{experiment.TRAIN}|cell:{corpus.selection_resolution}",
        )
        held = experiment.confusion(
            matrix,
            selection.rule.rule_id,
            holdout,
            slice_name=f"split:{experiment.HOLDOUT}|cell:{corpus.selection_resolution}",
        )
        gate_results["train_removes_no_correct_faces"] = train_confusion.removes_no_correct_faces
        gate_results["train_removes_material_false_faces"] = (
            train_confusion.removes_material_false_faces
        )
        gate_results["holdout_removes_no_correct_faces"] = held.removes_no_correct_faces
        gate_results["holdout_removes_material_false_faces"] = held.removes_material_false_faces
        # Every sealed holdout appliance population the rule fires on. No
        # ceiling and no prefilter: the rule is applied to ALL closed shells,
        # because an appliance cannot enumerate the unsupported ones.
        for item in built:
            if (
                item.population.split != experiment.HOLDOUT
                or item.cell_id != corpus.selection_resolution
            ):
                continue
            with experiment.accounting(experiment.COUNTERFACTUAL_PHASE):
                fired = experiment.select(selection.rule, item.population.observables)
            if not fired:
                continue
            scene = build_member(item.member, corpus)
            result = _result_only(scene, item.cell_id)
            evidence = experiment.counterfactual(
                scene,
                result,
                item.population.observables,
                item.population.records,
                fired,
                member_id=item.member.member_id,
                cell_id=item.cell_id,
                split=item.population.split,
                reconstruction_digest=item.population.reconstruction_digest,
            )
            counterfactuals.append(evidence.as_dict())
            for gate, passed in evidence.gates.items():
                gate_results[gate] = gate_results.get(gate, True) and passed
        if not counterfactuals:
            for gate in experiment.GATES:
                if gate.startswith("counterfactual_"):
                    gate_results[gate] = False

    recommendation = (
        PRUNE
        if selection.rule is not None
        and counterfactuals
        and all(gate_results.get(gate, False) for gate in experiment.GATES)
        else NO_PRUNING
    )

    # Copied out of the cached bundle: the document this returns is the
    # caller's, and a caller that edited it must not be able to edit the
    # evidence every later strict read is checked against.
    upper_bound = copy.deepcopy(bundle.upper_bound_block)
    ledger = experiment.rule_calls()

    totals = {
        "members": len(corpus.members),
        "resolutions": len(corpus.resolutions),
        "populations": len(populations),
        "closed_shells": sum(len(p.observables) for p in populations),
        "shells": len(records),
        "supported_closed_shells": sum(len(p.observables) - len(p.records) for p in populations),
        "train_shells": sum(1 for record in records if record.split == experiment.TRAIN),
        "holdout_shells": sum(1 for record in records if record.split == experiment.HOLDOUT),
        "selection_train_shells": len(training),
        "selection_holdout_shells": len(holdout),
        "false_shells": sum(1 for record in records if record.truth.false_surface),
        "correct_shells": sum(1 for record in records if not record.truth.false_surface),
        "false_faces": sum(record.truth.faces for record in records if record.truth.false_surface),
        "correct_faces": sum(
            record.truth.faces for record in records if not record.truth.false_surface
        ),
        "candidates": len(candidates),
        "cross_tab_cells": len(experiment.cross_cells(corpus.resolutions)),
        "evaluations": matrix.calls,
        "failures": len(failures),
    }
    document: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "corpus": _corpus_block(corpus),
        "namespaces": {
            "observable_source": experiment.OBSERVABLE_SOURCE,
            "label_source": experiment.LABEL_SOURCE,
            "observables": sorted(experiment.FEATURES),
            "label_only_fields": list(experiment.LABEL_ONLY_FIELDS),
        },
        "protocol": experiment.SELECTION_PROTOCOL,
        "gates": list(experiment.GATES),
        "realizations": _realization_block(built),
        "populations": [population.as_dict() for population in populations],
        "candidates": candidates,
        "selection": selection.as_dict(),
        "holdout_confusion": holdout_confusion,
        "gate_results": {name: bool(value) for name, value in sorted(gate_results.items())},
        "counterfactuals": counterfactuals,
        "upper_bound": upper_bound,
        "recommendation": recommendation,
        "failures": failures,
        "totals": totals,
        "cost": {
            "fusions": len(populations),
            "voxel_visits": sum(population.work_units for population in populations),
            "triangle_samples": sum(population.triangle_samples for population in populations),
            "producer_rule_calls": ledger.get(experiment.PRODUCER_PHASE, 0),
            "counterfactual_rule_calls": ledger.get(experiment.COUNTERFACTUAL_PHASE, 0),
            "reader_audit_rule_calls": ledger.get(experiment.READER_PHASE, 0),
            "unaccounted_rule_calls": ledger.get("unaccounted", 0),
            "total_rule_calls": sum(ledger.values()),
            "decision_matrix_cells": matrix.calls,
            "work_ceiling": experiment.WORK_CEILING,
            "counterfactual_remeasurements": 2
            * (len(counterfactuals) + (1 if _measured(upper_bound) else 0)),
            "note": COST_NOTE,
        },
        "decision": "pass" if not failures else "fail",
        "limitations": list(LIMITATIONS),
    }
    _validate_document(document, bundle, what="shell-discriminant report")
    return document


def _measured(bound: object) -> bool:
    return isinstance(bound, Mapping) and bool(bound.get("measured"))


def _upper_bound(corpus: Corpus, built: Sequence[Built]) -> dict[str, object] | None:
    """What a PERFECT, truth-fed prune would gain on one declared member.

    Not a candidate and not a recommendation. It is the ceiling on the whole
    idea: it removes exactly the shells the analytic labels call false, on the
    member and resolution the MANIFEST names - predeclared by the planner before
    any sealed member was executed - and re-measures the mesh. If even this
    gains almost nothing output-facing, then no observable rule can gain more
    than almost nothing either, and that is worth knowing whichever way the
    candidate matrix comes out.

    It never touches :func:`tests.shell_discriminant.select`, so it costs no
    rule invocation and cannot become a truth-fed selector by accident.
    """
    here = [
        item
        for item in built
        if item.member.member_id == corpus.upper_bound_member
        and item.cell_id == corpus.upper_bound_cell
    ]
    if not here:
        return {
            "member_id": corpus.upper_bound_member,
            "cell_id": corpus.upper_bound_cell,
            "selector": UPPER_BOUND_SELECTOR,
            "removed_shell_ids": [],
            "measured": False,
            "why": UPPER_BOUND_UNBUILT,
        }
    item = here[0]
    false_shells = [
        record.shell_id for record in item.population.records if record.truth.false_surface
    ]
    if not false_shells:
        return {
            "member_id": corpus.upper_bound_member,
            "cell_id": corpus.upper_bound_cell,
            "selector": UPPER_BOUND_SELECTOR,
            "removed_shell_ids": [],
            "measured": False,
            "why": UPPER_BOUND_NO_FALSE_SHELL,
        }
    scene = build_member(item.member, corpus)
    result = _result_only(scene, corpus.upper_bound_cell)
    evidence = experiment.counterfactual(
        scene,
        result,
        item.population.observables,
        item.population.records,
        false_shells,
        member_id=corpus.upper_bound_member,
        cell_id=corpus.upper_bound_cell,
        split=item.population.split,
        reconstruction_digest=item.population.reconstruction_digest,
    )
    published = dict(evidence.as_dict())
    published["selector"] = UPPER_BOUND_SELECTOR
    published["measured"] = True
    return published


def _report_mapping(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReportUnusable(f"{what} is {type(value).__name__}, not an object")
    return value


def _report_list(record: Mapping[str, object], key: str, what: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise ReportUnusable(f"{what} {key!r} is {type(value).__name__}, not a list")
    return value


def _report_text(record: Mapping[str, object], key: str, what: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ReportUnusable(f"{what} {key!r} is {value!r}, not text")
    return value


def _report_int(record: Mapping[str, object], key: str, what: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportUnusable(f"{what} {key!r} is {value!r}, not an integer")
    return value


def _report_number(record: Mapping[str, object], key: str, what: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportUnusable(f"{what} {key!r} is {value!r}, not a number")
    # A JSON integer has no width limit, and float() on a four-hundred-digit one
    # raises OverflowError. A reader refuses; it does not crash.
    try:
        number = float(value)
    except OverflowError as error:
        raise ReportUnusable(
            f"{what} {key!r} is an integer too large to be a measurement"
        ) from error
    if not math.isfinite(number):
        raise ReportUnusable(f"{what} {key!r} is {value!r}")
    return number


def _report_bool(record: Mapping[str, object], key: str, what: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise ReportUnusable(f"{what} {key!r} is {value!r}, not a boolean")
    return value


def _report_optional_number(record: Mapping[str, object], key: str, what: str) -> float | None:
    if record.get(key) is None:
        return None
    return _report_number(record, key, what)


def _report_keys(record: Mapping[str, object], expected: Sequence[str], what: str) -> None:
    missing = sorted(set(expected) - set(record))
    extra = sorted(set(record) - set(expected))
    if missing or extra:
        raise ReportUnusable(
            f"{what} does not carry exactly the declared keys. Missing {missing}; unexpected "
            f"{extra}"
        )


def _report_point(record: Mapping[str, object], key: str, what: str) -> tuple[float, float, float]:
    value = record.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise ReportUnusable(f"{what} {key!r} is {value!r}, not a three-vector")
    out = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ReportUnusable(f"{what} {key!r}[{index}] is {item!r}")
        try:
            component = float(item)
        except OverflowError as error:
            raise ReportUnusable(
                f"{what} {key!r}[{index}] is an integer too large to be a coordinate"
            ) from error
        if not math.isfinite(component):
            raise ReportUnusable(f"{what} {key!r}[{index}] is {item!r}")
        out.append(component)
    return (out[0], out[1], out[2])


def _report_text_list(record: Mapping[str, object], key: str, what: str) -> list[str]:
    """A list of TEXT, rather than a list coerced to text.

    ``[str(item) for item in ...]`` turns ``1``, ``None`` and ``{}`` into
    ``"1"``, ``"None"`` and ``"{}"``, so a declared list whose items are not
    strings used to be silently repaired into strings and then compared. An
    item that is not text is a report the producer did not write, and it is
    refused here rather than reshaped.
    """
    out: list[str] = []
    for index, item in enumerate(_report_list(record, key, what)):
        if not isinstance(item, str):
            raise ReportUnusable(f"{what} {key!r}[{index}] is {item!r}, not text")
        out.append(item)
    return out


def _report_optional_text(record: Mapping[str, object], key: str, what: str) -> str | None:
    """Text, or an explicit null. A number standing in for either is refused."""
    value = record.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ReportUnusable(f"{what} {key!r} is {value!r}, not text or null")
    return value


def _first_difference(published: object, canonical: object, path: str) -> str:
    """Where a published value first stops being the evidence's value.

    A canonical block is megabytes. Printing both sides of a mismatch would
    make a refusal unreadable and therefore unactionable, so the walk stops at
    the first field that differs and names it.
    """
    if isinstance(canonical, Mapping):
        if not isinstance(published, Mapping):
            return f"{path} is {type(published).__name__}, not an object"
        extra = sorted(set(published) - set(canonical))
        missing = sorted(set(canonical) - set(published))
        if extra or missing:
            return f"{path} carries unexpected {extra} and omits {missing}"
        for key in canonical:
            if published[key] != canonical[key]:
                return _first_difference(published[key], canonical[key], f"{path}[{key!r}]")
        return f"{path} differs"
    if isinstance(canonical, list):
        if not isinstance(published, list):
            return f"{path} is {type(published).__name__}, not a list"
        if len(published) != len(canonical):
            return f"{path} carries {len(published)} item(s); the evidence has {len(canonical)}"
        for index, (mine, theirs) in enumerate(zip(published, canonical, strict=True)):
            if mine != theirs:
                return _first_difference(mine, theirs, f"{path}[{index}]")
        return f"{path} differs"
    return f"{path} is {published!r}; the evidence gives {canonical!r}"


def _bind_exact(published: object, canonical: object, what: str, *, why: str) -> None:
    """Refuse a published value that is not the one rebuilt from the corpus.

    The canonical side is never another part of the report. It is a rebuild of
    the committed experiment, so a difference here is not a formatting
    difference and cannot be repaired: it is a claim the evidence does not
    make.
    """
    if published == canonical:
        return
    raise ReportUnusable(f"{what} {why}. {_first_difference(published, canonical, what)}")


#: The exact keys a published report carries at the top level.
REPORT_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "kind",
    "corpus",
    "namespaces",
    "protocol",
    "gates",
    "realizations",
    "populations",
    "candidates",
    "selection",
    "holdout_confusion",
    "gate_results",
    "counterfactuals",
    "upper_bound",
    "recommendation",
    "failures",
    "totals",
    "cost",
    "decision",
    "limitations",
)

NAMESPACE_KEYS: Final[tuple[str, ...]] = (
    "observable_source",
    "label_source",
    "observables",
    "label_only_fields",
)

CORPUS_KEYS: Final[tuple[str, ...]] = (
    "kind",
    "schema_version",
    "manifest_sha256",
    "population_sha256",
    "base_case",
    "base_case_sha256",
    "candidate_sha256",
    "protocol_sha256",
    "resolutions",
    "selection_resolution",
    "material_fraction",
    "work_ceiling",
    "sealed_holdout",
    "upper_bound",
    "members",
)

MEMBER_KEYS: Final[tuple[str, ...]] = (
    "member_id",
    "family",
    "split",
    "seed",
    "control_case",
    "frame_digest",
    "perturbation",
)

#: The exact perturbation one committed member declares, grouped by the type
#: each field has to have. A perturbation IS the member - what was done to the
#: exact scene and by how much - and equality alone does not establish it,
#: because ``False == 0`` and ``True == 1`` in Python. A report that publishes
#: JSON ``false`` where the corpus declares the integer ``0`` compares equal to
#: the evidence and is still not the declaration, so every field is typed
#: before it is compared.
PERTURBATION_LABELS: Final[tuple[str, ...]] = ("label",)
PERTURBATION_CODES: Final[tuple[str, ...]] = (
    "depth_noise_codes",
    "outlier_period",
    "outlier_codes",
    "band_period",
    "band_width",
    "band_shift",
)
PERTURBATION_MAGNITUDES: Final[tuple[str, ...]] = (
    "translation_jitter",
    "rotation_bias_degrees",
    "rotation_jitter_degrees",
)
PERTURBATION_VECTORS: Final[tuple[str, ...]] = ("translation_bias", "rotation_axis")
PERTURBATION_KEYS: Final[tuple[str, ...]] = (
    *PERTURBATION_LABELS,
    *PERTURBATION_CODES,
    *PERTURBATION_MAGNITUDES,
    *PERTURBATION_VECTORS,
)

REALIZATION_KEYS: Final[tuple[str, ...]] = (
    "member_id",
    "cell_id",
    "split",
    "family",
    "reconstruction_digest",
)

POPULATION_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "kind",
    "member_id",
    "family",
    "split",
    "cell_id",
    "voxel_size",
    "reconstruction_digest",
    "faces",
    "shells",
    "closed_shells",
    "unsupported_closed_shells",
    "supported_closed_shells",
    "false_shells",
    "correct_shells",
    "false_faces",
    "correct_faces",
    "unavailable",
    "work_units",
    "triangle_samples",
    "closed_shells_observable",
    "records",
)

RECORD_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "kind",
    "member_id",
    "family",
    "cell_id",
    "split",
    "reconstruction_digest",
    "face_indices",
    "observable",
    "label",
)

SELECTION_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "kind",
    "selected",
    "eligible_rule_ids",
    "considered_rules",
    "evaluations",
    "work_ceiling",
    "selection_resolution",
    "training_shells",
    "protocol",
    "reason",
)

#: The exact keys one published candidate rule and one feature test carry.
RULE_KEYS: Final[tuple[str, ...]] = ("rule_id", "family", "text", "tests")
FEATURE_TEST_KEYS: Final[tuple[str, ...]] = ("feature", "comparison", "threshold")

#: The exact keys one re-measured surface carries on either side of a
#: counterfactual. Bound so a fabricated measurement cannot be smuggled in
#: beside the real ones.
ACCOUNT_KEYS: Final[tuple[str, ...]] = (
    "faces",
    "vertices",
    "analytic_within_tolerance_fraction",
    "reference_coverage_fraction",
    "triangle_within_tolerance_fraction",
    "protected_area",
    "bridging_faces",
    "surface_area",
    "edge_coverage",
    "thin_feature_coverage",
    "worst_dimension_error",
    "unsupported_false_shells",
    "unsupported_false_shell_faces",
    "unsupported_correct_shells",
    "unsupported_correct_shell_faces",
    "nonmanifold_edge_count",
    "boundary_edge_count",
    "shell_count",
)

#: Which of those are COUNTS, and therefore integers. The rest are measured
#: amounts and must be finite numbers. The split matters for the same reason
#: the perturbation's does: a published ``true`` where the evidence measured
#: one false shell, and a published ``true`` where it measured a unit of
#: surface area, both compare equal to the evidence under ``==``. Neither is a
#: measurement, so an account is typed before anything is recomputed from it
#: and before it is compared against the re-measured mesh.
ACCOUNT_COUNTS: Final[tuple[str, ...]] = (
    "faces",
    "vertices",
    "bridging_faces",
    "unsupported_false_shells",
    "unsupported_false_shell_faces",
    "unsupported_correct_shells",
    "unsupported_correct_shell_faces",
    "nonmanifold_edge_count",
    "boundary_edge_count",
    "shell_count",
)
ACCOUNT_AMOUNTS: Final[tuple[str, ...]] = tuple(
    key for key in ACCOUNT_KEYS if key not in ACCOUNT_COUNTS
)

COUNTERFACTUAL_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "kind",
    "member_id",
    "cell_id",
    "split",
    "reconstruction_digest",
    "closed_shells",
    "removed_shell_ids",
    "removed_faces",
    "removed_false_shells",
    "removed_false_shell_faces",
    "removed_correct_shells",
    "removed_correct_shell_faces",
    "removed_supported_shells",
    "removed_supported_shell_faces",
    "false_faces_removed",
    "correct_faces_removed",
    "before",
    "after",
    "gates",
    "passed",
)

#: A MEASURED upper bound is one counterfactual plus the two fields that say
#: whose selector it was and that it ran. An UNMEASURED one is six keys and no
#: measurement at all: no gain, no before, no after, no gates.
MEASURED_BOUND_KEYS: Final[tuple[str, ...]] = (*COUNTERFACTUAL_KEYS, "selector", "measured")
UNMEASURED_BOUND_KEYS: Final[tuple[str, ...]] = (
    "member_id",
    "cell_id",
    "selector",
    "removed_shell_ids",
    "measured",
    "why",
)

COST_KEYS: Final[tuple[str, ...]] = (
    "fusions",
    "voxel_visits",
    "triangle_samples",
    "producer_rule_calls",
    "counterfactual_rule_calls",
    "reader_audit_rule_calls",
    "unaccounted_rule_calls",
    "total_rule_calls",
    "decision_matrix_cells",
    "work_ceiling",
    "counterfactual_remeasurements",
    "note",
)

#: The exact ``kind`` every nested record declares. A kind is a claim about
#: which validator a value belongs to, so a record wearing another one is
#: refused rather than validated as whatever it says it is.
OBSERVABLE_KIND: Final = "shell_discriminant_observable"
LABEL_KIND: Final = "shell_discriminant_label"
CONFUSION_KIND: Final = "shell_discriminant_confusion"
RECORD_KIND: Final = "shell_discriminant_record"
SELECTION_KIND: Final = "shell_discriminant_selection"
COUNTERFACTUAL_KIND: Final = "shell_discriminant_counterfactual"
#: The kind of one POPULATION inside a report. Deliberately not
#: :data:`POPULATION_KIND`, which is the fixture file's own kind.
POPULATION_RECORD_KIND: Final = "shell_discriminant_population"


#: The exact keys one published observable record carries.
OBSERVABLE_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "kind",
    "shell_id",
    "voxel_size",
    "shell_index",
    "face_indices",
    "faces",
    "vertices",
    "area",
    "extent",
    "diagonal",
    "centroid",
    "views_total",
    "views_containing",
    "views_touching",
    "ray_probes",
    "confirming_probes",
    "contradicting_probes",
    "occluded_probes",
    "no_return_probes",
    "outside_window_probes",
    "confirming_fraction",
    "contradicting_fraction",
    "occluded_fraction",
    "no_return_fraction",
    "ray_tolerance",
    "measured_returns",
    "nearest_measured_distance",
    "farthest_measured_distance",
    "nearest_measured_capped",
    "support_search_ceiling",
    "weight_min",
    "weight_mean",
    "weight_max",
    "weight_mean_in_thresholds",
    "min_weight",
    "unavailable",
    "observable_source",
    "unit",
)

#: The exact keys one published label record carries.
LABEL_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "kind",
    "shell_id",
    "label",
    "false_surface",
    "worst_analytic_distance",
    "accuracy_tolerance",
    "faces",
    "label_source",
)

#: The exact keys one published confusion carries.
CONFUSION_KEYS: Final[tuple[str, ...]] = (
    "schema_version",
    "kind",
    "slice",
    "rule_id",
    "shells",
    "false_shells",
    "correct_shells",
    "false_faces",
    "correct_faces",
    "selected_shells",
    "selected_false_shells",
    "selected_correct_shells",
    "selected_false_faces",
    "selected_correct_faces",
    "false_face_recall",
    "correct_face_loss",
    "removes_no_correct_faces",
    "removes_material_false_faces",
    "material_fraction",
    "unit",
)


def _report_perturbation(record: Mapping[str, object], what: str) -> None:
    """Type a published perturbation completely, before it is compared.

    The comparison that follows is exact, and it is still not enough on its
    own: ``{"depth_noise_codes": False} == {"depth_noise_codes": 0}`` is true
    in Python, so a report that turned a sealed magnitude into a boolean bound
    cleanly against the committed corpus. Every field is established here as
    the kind of thing it is - the family label as text, the depth, outlier and
    band settings as strict integers, the pose magnitudes as finite numbers and
    the translation bias and rotation axis as finite three-vectors - so a value
    of another type is refused for being another type rather than accepted for
    comparing equal.
    """
    _report_keys(record, PERTURBATION_KEYS, what)
    for key in PERTURBATION_LABELS:
        _report_text(record, key, what)
    for key in PERTURBATION_CODES:
        _report_int(record, key, what)
    for key in PERTURBATION_MAGNITUDES:
        _report_number(record, key, what)
    for key in PERTURBATION_VECTORS:
        _report_point(record, key, what)


def _report_account(record: Mapping[str, object], what: str) -> None:
    """Type one whole measured surface, before anything is derived from it.

    The same attack as the perturbation's, on the numbers a recommendation
    would actually rest on. Key sets do not close it: an account is compared
    against a re-measured mesh with ``==``, and a published ``true`` equals
    both the ``1`` false shell and the ``1.0`` of surface area the evidence
    measures. Both sides of every counterfactual go through here, and so does
    a measured upper bound, before a delta, a gate, a passed status or an
    exact remeasurement binding reads any of it.
    """
    _report_keys(record, ACCOUNT_KEYS, what)
    for key in ACCOUNT_COUNTS:
        _report_int(record, key, what)
    for key in ACCOUNT_AMOUNTS:
        _report_number(record, key, what)


def _rebuild_observable(record: object, what: str) -> experiment.ObservableShell:
    """Rebuild a typed observable record and run every invariant on it.

    Not a shape check. The value that comes out is the same class the producer
    made and it has been through :func:`check_observable` and the full
    consistency validator, so a forged ray partition, a containing count above
    the touching count, an unavailable item carrying a value or a fabricated
    weight is refused by exactly the code that refused it upstream.
    """
    entry = _report_mapping(record, what)
    _report_keys(entry, OBSERVABLE_KEYS, what)
    leaked = sorted(set(entry) & set(experiment.LABEL_ONLY_FIELDS))
    if leaked:
        raise ReportUnusable(
            f"{what} publishes an observable record carrying {leaked}, which is the label's "
            "namespace"
        )
    if _report_int(entry, "schema_version", what) != experiment.SCHEMA_VERSION:
        raise ReportUnusable(f"{what} is observable schema {entry['schema_version']!r}")
    if _report_text(entry, "kind", what) != OBSERVABLE_KIND:
        raise ReportUnusable(f"{what} is kind {entry['kind']!r}, not a {OBSERVABLE_KIND}")
    if _report_text(entry, "unit", what) != experiment.DEPTH_UNIT:
        raise ReportUnusable(f"{what} states a unit that is not the module's")
    if _report_text(entry, "observable_source", what) != experiment.OBSERVABLE_SOURCE:
        raise ReportUnusable(f"{what} states an observable source that is not the module's")
    indices = _report_list(entry, "face_indices", what)
    faces: list[int] = []
    for index, item in enumerate(indices):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ReportUnusable(f"{what} face index {index} is {item!r}")
        faces.append(item)
    unavailable = _report_text_list(entry, "unavailable", what)
    try:
        shell = experiment.ObservableShell(
            shell_id=_report_text(entry, "shell_id", what),
            voxel_size=_report_number(entry, "voxel_size", what),
            shell_index=_report_int(entry, "shell_index", what),
            face_indices=tuple(faces),
            faces=_report_int(entry, "faces", what),
            vertices=_report_int(entry, "vertices", what),
            area=_report_number(entry, "area", what),
            extent=_report_point(entry, "extent", what),
            diagonal=_report_number(entry, "diagonal", what),
            centroid=_report_point(entry, "centroid", what),
            views_total=_report_int(entry, "views_total", what),
            views_containing=_report_int(entry, "views_containing", what),
            views_touching=_report_int(entry, "views_touching", what),
            ray_probes=_report_int(entry, "ray_probes", what),
            confirming_probes=_report_int(entry, "confirming_probes", what),
            contradicting_probes=_report_int(entry, "contradicting_probes", what),
            occluded_probes=_report_int(entry, "occluded_probes", what),
            no_return_probes=_report_int(entry, "no_return_probes", what),
            outside_window_probes=_report_int(entry, "outside_window_probes", what),
            ray_tolerance=_report_number(entry, "ray_tolerance", what),
            measured_returns=_report_int(entry, "measured_returns", what),
            nearest_measured_distance=_report_number(entry, "nearest_measured_distance", what),
            farthest_measured_distance=_report_number(entry, "farthest_measured_distance", what),
            nearest_measured_capped=_report_bool(entry, "nearest_measured_capped", what),
            weight_min=_report_optional_number(entry, "weight_min", what),
            weight_mean=_report_optional_number(entry, "weight_mean", what),
            weight_max=_report_optional_number(entry, "weight_max", what),
            min_weight=_report_number(entry, "min_weight", what),
            unavailable=tuple(unavailable),
        )
        experiment.check_observable(shell)
        experiment.check_observable_is_consistent(shell)
    except ScannerError as error:
        raise ReportUnusable(f"{what}: {error}") from error
    if len(faces) != shell.faces:
        raise ReportUnusable(
            f"{what} claims {shell.faces} face(s) and lists {len(faces)} face index/indices"
        )
    if len(set(faces)) != len(faces):
        raise ReportUnusable(f"{what} lists the same face index twice")
    for key, value in (
        ("confirming_fraction", shell.confirming_fraction),
        ("contradicting_fraction", shell.contradicting_fraction),
        ("occluded_fraction", shell.occluded_fraction),
        ("no_return_fraction", shell.no_return_fraction),
    ):
        if _report_number(entry, key, what) != value:
            raise ReportUnusable(f"{what} publishes {key} that its own probe counts do not give")
    # Through a typed accessor, not a raw comparison: ``True == 1.0`` in Python,
    # so a JSON boolean would stand in for a ratio of exactly one.
    published = _report_optional_number(entry, "weight_mean_in_thresholds", what)
    if published != shell.weight_mean_in_thresholds:
        raise ReportUnusable(
            f"{what} publishes weight_mean_in_thresholds {published!r}, which is not its own "
            "mean weight over its own extraction threshold"
        )
    if _report_number(entry, "support_search_ceiling", what) != experiment.SUPPORT_SEARCH_CEILING:
        raise ReportUnusable(f"{what} states a support search ceiling that is not the module's")
    return shell


def _rebuild_truth(record: object, what: str) -> experiment.ShellTruth:
    entry = _report_mapping(record, what)
    _report_keys(entry, LABEL_KEYS, what)
    if _report_int(entry, "schema_version", what) != experiment.SCHEMA_VERSION:
        raise ReportUnusable(f"{what} is label schema {entry['schema_version']!r}")
    if _report_text(entry, "kind", what) != LABEL_KIND:
        raise ReportUnusable(f"{what} is kind {entry['kind']!r}, not a {LABEL_KIND}")
    if _report_text(entry, "label_source", what) != experiment.LABEL_SOURCE:
        raise ReportUnusable(f"{what} states a label source that is not the module's")
    try:
        truth = experiment.ShellTruth(
            shell_id=_report_text(entry, "shell_id", what),
            false_surface=_report_bool(entry, "false_surface", what),
            worst_analytic_distance=_report_number(entry, "worst_analytic_distance", what),
            accuracy_tolerance=_report_number(entry, "accuracy_tolerance", what),
            faces=_report_int(entry, "faces", what),
        )
    except ScannerError as error:
        raise ReportUnusable(f"{what}: {error}") from error
    if _report_text(entry, "label", what) != truth.label:
        raise ReportUnusable(
            f"{what} publishes label {entry['label']!r} beside false_surface "
            f"{truth.false_surface!r}. A swapped label is refused here, not read"
        )
    return truth


def _rebuild_confusion(record: object, what: str) -> experiment.Confusion:
    entry = _report_mapping(record, what)
    _report_keys(entry, CONFUSION_KEYS, what)
    if _report_int(entry, "schema_version", what) != experiment.SCHEMA_VERSION:
        raise ReportUnusable(f"{what} is confusion schema {entry['schema_version']!r}")
    if _report_text(entry, "kind", what) != CONFUSION_KIND:
        raise ReportUnusable(f"{what} is kind {entry['kind']!r}, not a {CONFUSION_KIND}")
    if _report_text(entry, "unit", what) != experiment.FACE_UNIT:
        raise ReportUnusable(f"{what} states a unit that is not the module's")
    if _report_number(entry, "material_fraction", what) != experiment.MATERIAL_FRACTION:
        raise ReportUnusable(f"{what} states a material fraction that is not the module's")
    try:
        counts = experiment.Confusion(
            slice_name=_report_text(entry, "slice", what),
            rule_id=_report_text(entry, "rule_id", what),
            shells=_report_int(entry, "shells", what),
            false_shells=_report_int(entry, "false_shells", what),
            correct_shells=_report_int(entry, "correct_shells", what),
            false_faces=_report_int(entry, "false_faces", what),
            correct_faces=_report_int(entry, "correct_faces", what),
            selected_shells=_report_int(entry, "selected_shells", what),
            selected_false_shells=_report_int(entry, "selected_false_shells", what),
            selected_correct_shells=_report_int(entry, "selected_correct_shells", what),
            selected_false_faces=_report_int(entry, "selected_false_faces", what),
            selected_correct_faces=_report_int(entry, "selected_correct_faces", what),
        )
    except ScannerError as error:
        raise ReportUnusable(f"{what}: {error}") from error
    if _report_number(entry, "false_face_recall", what) != counts.false_face_recall:
        raise ReportUnusable(f"{what} publishes a recall its own counts do not give")
    if _report_number(entry, "correct_face_loss", what) != counts.correct_face_loss:
        raise ReportUnusable(f"{what} publishes a loss its own counts do not give")
    if _report_bool(entry, "removes_no_correct_faces", what) != counts.removes_no_correct_faces:
        raise ReportUnusable(f"{what} publishes a zero-loss claim its own counts do not give")
    if (
        _report_bool(entry, "removes_material_false_faces", what)
        != counts.removes_material_false_faces
    ):
        raise ReportUnusable(f"{what} publishes a materiality claim its own counts do not give")
    return counts


def _same_confusion(
    published: experiment.Confusion, recomputed: experiment.Confusion, what: str
) -> None:
    if published.as_dict() != recomputed.as_dict():
        raise ReportUnusable(
            f"{what} publishes a confusion the reader's own audit matrix does not reproduce. "
            f"Published {published.as_dict()}; recomputed {recomputed.as_dict()}"
        )


def _typed_rule(body: object, what: str) -> Mapping[str, object]:
    """Read a published rule body through typed accessors, then hand it back.

    A rule body is compared against the declared candidate with ``==``, and
    ``True == 1.0`` in Python. Two of the declared thresholds are exactly
    ``0.0`` and ``1.0``, so a threshold published as a JSON boolean would
    compare equal to a declared number. Every field is read for its type first.
    """
    entry = _report_mapping(body, what)
    _report_keys(entry, RULE_KEYS, what)
    _report_text(entry, "rule_id", what)
    _report_text(entry, "family", what)
    _report_text(entry, "text", what)
    tests = _report_list(entry, "tests", what)
    if not 1 <= len(tests) <= 2:
        raise ReportUnusable(f"{what} carries {len(tests)} feature test(s), not one or two")
    for index, item in enumerate(tests):
        test = _report_mapping(item, f"{what} test {index}")
        _report_keys(test, FEATURE_TEST_KEYS, f"{what} test {index}")
        _report_text(test, "feature", f"{what} test {index}")
        _report_text(test, "comparison", f"{what} test {index}")
        _report_number(test, "threshold", f"{what} test {index}")
    return entry


def _bind_realizations(report: Mapping[str, object], evidence: _Evidence, what: str) -> None:
    """Bind the realization list to the populations the committed corpus builds.

    Exact, ordered and complete. An invented extra realization, a repeated one,
    an omitted one and a re-ordered one are each refused for their own reason,
    and every identity is the one the rebuilt reconstruction actually has.
    """
    published = _report_list(report, "realizations", what)
    canonical = evidence.realization_block
    seen: set[tuple[str, str]] = set()
    for position, item in enumerate(published):
        entry = _report_mapping(item, f"{what} realization {position}")
        _report_keys(entry, REALIZATION_KEYS, f"{what} realization {position}")
        member_id = _report_text(entry, "member_id", f"{what} realization {position}")
        cell_id = _report_text(entry, "cell_id", f"{what} realization {position}")
        if (member_id, cell_id) in seen:
            raise ReportUnusable(f"{what} publishes the realization {member_id}/{cell_id} twice")
        seen.add((member_id, cell_id))
        if position >= len(canonical):
            raise ReportUnusable(
                f"{what} publishes an extra realization {member_id}/{cell_id} at position "
                f"{position}; the committed corpus builds {len(canonical)} population(s) and no "
                "more. A realization nothing was reconstructed for is an invented one"
            )
        declared = canonical[position]
        if (member_id, cell_id) != (declared["member_id"], declared["cell_id"]):
            raise ReportUnusable(
                f"{what} publishes realization {member_id}/{cell_id} at position {position}, "
                f"where the committed corpus builds "
                f"{declared['member_id']}/{declared['cell_id']}"
            )
        for key in ("split", "family", "reconstruction_digest"):
            _report_text(entry, key, f"{what} realization {member_id}/{cell_id}")
        _bind_exact(
            dict(entry),
            declared,
            f"{what} realization {member_id}/{cell_id}",
            why="is not the realization the committed corpus reconstructs",
        )
    if len(published) < len(canonical):
        omitted = canonical[len(published) :]
        missing = [f"{item['member_id']}/{item['cell_id']}" for item in omitted]
        raise ReportUnusable(
            f"{what} omits realization(s) {missing}. Every declared member at every declared "
            "resolution, or the population is a subset somebody chose"
        )


def _bind_populations(
    report: Mapping[str, object], evidence: _Evidence, what: str
) -> list[experiment.ShellRecord]:
    """Bind every population to the one the committed corpus rebuilds, then retype it.

    Two passes over the same data, and they close different attacks. The first
    compares the published population against the canonical rebuild exactly -
    schema, kind, metadata, identity, counts, work, every closed shell and every
    labelled record - so a coherent forgery whose derived fractions still agree
    with its forged counts is refused because the evidence does not contain it.
    The second rebuilds the typed values and runs the module's own invariants on
    them, so a malformed record is refused by exactly the code that refused it
    upstream rather than by a comparison that never got that far.
    """
    published = _report_list(report, "populations", what)
    canonical = evidence.population_block
    if len(published) != len(canonical):
        raise ReportUnusable(
            f"{what} publishes {len(published)} population(s); the committed corpus builds "
            f"{len(canonical)} - every declared member at every declared resolution"
        )
    seen: set[tuple[str, str]] = set()
    records: list[experiment.ShellRecord] = []
    identities: set[str] = set()
    shell_identities: set[str] = set()
    for position, (item, declared) in enumerate(zip(published, canonical, strict=True)):
        entry = _report_mapping(item, f"{what} population {position}")
        _report_keys(entry, POPULATION_KEYS, f"{what} population {position}")
        member_id = _report_text(entry, "member_id", f"{what} population {position}")
        cell_id = _report_text(entry, "cell_id", f"{what} population {position}")
        where = f"{what} population {member_id}/{cell_id}"
        if (member_id, cell_id) in seen:
            raise ReportUnusable(f"{what} publishes {member_id}/{cell_id} twice")
        seen.add((member_id, cell_id))
        if (member_id, cell_id) != (declared["member_id"], declared["cell_id"]):
            raise ReportUnusable(
                f"{what} publishes population {member_id}/{cell_id} at position {position}, "
                f"where the committed corpus builds "
                f"{declared['member_id']}/{declared['cell_id']}"
            )
        if _report_int(entry, "schema_version", where) != experiment.SCHEMA_VERSION:
            raise ReportUnusable(f"{where} is population schema {entry['schema_version']!r}")
        if _report_text(entry, "kind", where) != POPULATION_RECORD_KIND:
            raise ReportUnusable(
                f"{where} is kind {entry['kind']!r}, not a {POPULATION_RECORD_KIND}"
            )
        family = _report_text(entry, "family", where)
        split = _report_text(entry, "split", where)
        digest = _report_text(entry, "reconstruction_digest", where)
        for key, subject in (
            ("family", "perturbation family"),
            ("split", "split"),
            ("reconstruction_digest", "reconstruction-input identity"),
        ):
            if entry[key] != declared[key]:
                raise ReportUnusable(
                    f"{where} publishes {subject} {entry[key]!r}; the population rebuilt from the "
                    f"committed corpus has {declared[key]!r}"
                )
        if _report_number(entry, "voxel_size", where) != declared["voxel_size"]:
            raise ReportUnusable(
                f"{where} publishes voxel_size {entry['voxel_size']!r}; the cell the committed "
                f"corpus fused it at is {declared['voxel_size']!r}"
            )
        for key in (
            "faces",
            "shells",
            "closed_shells",
            "unsupported_closed_shells",
            "supported_closed_shells",
            "false_shells",
            "correct_shells",
            "false_faces",
            "correct_faces",
            "work_units",
            "triangle_samples",
        ):
            if _report_int(entry, key, where) != declared[key]:
                raise ReportUnusable(
                    f"{where} publishes {key}={entry[key]!r}; the population rebuilt from the "
                    f"committed corpus gives {declared[key]!r}"
                )
        if _report_text_list(entry, "unavailable", where) != declared["unavailable"]:
            raise ReportUnusable(
                f"{where} publishes unavailable {entry['unavailable']!r}; the rebuilt population "
                f"could not carry {declared['unavailable']!r}"
            )
        shells = [
            _rebuild_observable(value, f"{where} closed shell {index}")
            for index, value in enumerate(_report_list(entry, "closed_shells_observable", where))
        ]
        by_id = {shell.shell_id: shell for shell in shells}
        if len(by_id) != len(shells):
            raise ReportUnusable(f"{where} publishes one closed shell identity twice")
        for shell in shells:
            if shell.shell_id in shell_identities:
                raise ReportUnusable(
                    f"{what} publishes closed shell identity {shell.shell_id} twice. Every closed "
                    "shell of every population is one shell, whether or not it is labelled"
                )
            shell_identities.add(shell.shell_id)
        here: list[experiment.ShellRecord] = []
        for index, value in enumerate(_report_list(entry, "records", where)):
            record_entry = _report_mapping(value, f"{where} record {index}")
            _report_keys(record_entry, RECORD_KEYS, f"{where} record {index}")
            if (
                _report_int(record_entry, "schema_version", f"{where} record {index}")
                != experiment.SCHEMA_VERSION
            ):
                raise ReportUnusable(
                    f"{where} record {index} is record schema {record_entry['schema_version']!r}"
                )
            if _report_text(record_entry, "kind", f"{where} record {index}") != RECORD_KIND:
                raise ReportUnusable(
                    f"{where} record {index} is kind {record_entry['kind']!r}, not a {RECORD_KIND}"
                )
            observable = _rebuild_observable(
                record_entry["observable"], f"{where} record {index} observable"
            )
            declared_shell = by_id.get(observable.shell_id)
            if declared_shell is None:
                raise ReportUnusable(
                    f"{where} record {index} labels {observable.shell_id}, which is not one of "
                    "this population's own closed shells"
                )
            if declared_shell.as_dict() != observable.as_dict():
                raise ReportUnusable(
                    f"{where} record {index} publishes an observable that differs from the same "
                    "shell in this population's closed-shell list"
                )
            truth = _rebuild_truth(record_entry["label"], f"{where} record {index} label")
            # Every scalar in this reader goes through a typed accessor. A face
            # index published as "3" or 3.9 is a report the producer did not
            # write, and coercing it here would let a reader accept a topology
            # binding nothing generated.
            face_indices: list[int] = []
            for position, value in enumerate(
                _report_list(record_entry, "face_indices", f"{where} record {index}")
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ReportUnusable(
                        f"{where} record {index} face index {position} is {value!r}, not a "
                        "non-negative integer"
                    )
                face_indices.append(value)
            try:
                rebuilt = experiment.ShellRecord(
                    observable=observable,
                    truth=truth,
                    member_id=_report_text(record_entry, "member_id", f"{where} record {index}"),
                    family=_report_text(record_entry, "family", f"{where} record {index}"),
                    cell_id=_report_text(record_entry, "cell_id", f"{where} record {index}"),
                    split=_report_text(record_entry, "split", f"{where} record {index}"),
                    reconstruction_digest=_report_text(
                        record_entry, "reconstruction_digest", f"{where} record {index}"
                    ),
                    face_indices=tuple(face_indices),
                )
            except ScannerError as error:
                raise ReportUnusable(f"{where} record {index}: {error}") from error
            if (rebuilt.member_id, rebuilt.cell_id, rebuilt.split, rebuilt.family) != (
                member_id,
                cell_id,
                split,
                family,
            ):
                raise ReportUnusable(
                    f"{where} record {index} names a member, resolution, split or family that is "
                    "not its own population's"
                )
            if rebuilt.reconstruction_digest != digest:
                raise ReportUnusable(
                    f"{where} record {index} names another realization's reconstruction identity"
                )
            if rebuilt.shell_id in identities:
                raise ReportUnusable(f"{what} publishes shell identity {rebuilt.shell_id} twice")
            identities.add(rebuilt.shell_id)
            here.append(rebuilt)
        records.extend(here)

        false_records = [record for record in here if record.truth.false_surface]
        correct_records = [record for record in here if not record.truth.false_surface]
        for key, value in (
            ("closed_shells", len(shells)),
            ("unsupported_closed_shells", len(here)),
            ("supported_closed_shells", len(shells) - len(here)),
            ("false_shells", len(false_records)),
            ("correct_shells", len(correct_records)),
            ("false_faces", sum(record.truth.faces for record in false_records)),
            ("correct_faces", sum(record.truth.faces for record in correct_records)),
        ):
            if _report_int(entry, key, where) != value:
                raise ReportUnusable(
                    f"{where} publishes {key}={entry[key]!r}; its own records give {value}"
                )
        # Last, and deliberately so. Everything above refuses a malformed value
        # with the module's own reason for refusing it; this refuses a value
        # that is perfectly well formed, agrees with itself and is still not
        # the one the committed corpus derives.
        _bind_exact(
            _report_list(entry, "closed_shells_observable", where),
            declared["closed_shells_observable"],
            f"{where} 'closed_shells_observable'",
            why="does not publish the closed shells the committed corpus derives",
        )
        _bind_exact(
            _report_list(entry, "records", where),
            declared["records"],
            f"{where} 'records'",
            why="does not publish the labelled population the committed corpus derives",
        )
    return records


def _bind_corpus(
    report: Mapping[str, object], evidence: _Evidence, what: str
) -> Mapping[str, object]:
    """Bind the published corpus block to the corpus this repository committed.

    Round two checked digest SYNTAX and a handful of module constants, so a
    report could name another syntactically valid manifest digest, another base
    case, another member frame digest or another perturbation and still be
    accepted. Every field here is compared against ``load_corpus()`` over the
    committed bytes, in the declared order, and each mismatch is refused for
    its own reason.
    """
    where = f"{what} 'corpus'"
    corpus = _report_mapping(report["corpus"], where)
    _report_keys(corpus, CORPUS_KEYS, where)
    canonical = evidence.corpus_block
    if _report_text(corpus, "kind", where) != CORPUS_KIND:
        raise ReportUnusable(f"{where} is not a {CORPUS_KIND}")
    if _report_int(corpus, "schema_version", where) != FIXTURE_SCHEMA_VERSION:
        raise ReportUnusable(f"{where} is fixture schema {corpus['schema_version']!r}")
    for key in ("manifest_sha256", "population_sha256", "base_case_sha256"):
        value = _report_text(corpus, key, where)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ReportUnusable(f"{where} {key!r} is {value!r}, not a SHA-256")
    if _report_text(corpus, "candidate_sha256", where) != experiment.FROZEN_CANDIDATE_DIGEST:
        raise ReportUnusable(
            f"{where} publishes candidate digest {corpus['candidate_sha256']!r}; the 195 bodies "
            f"were frozen at {experiment.FROZEN_CANDIDATE_DIGEST} before the sealed holdout ran"
        )
    if _report_text(corpus, "protocol_sha256", where) != experiment.grid_digest():
        raise ReportUnusable(f"{where} publishes a protocol digest that is not the module's")
    # Every committed input, bound to the bytes on disk rather than to its own
    # shape. A syntactically valid digest of nothing in particular is exactly
    # the forgery this closes.
    for key, subject in (
        ("manifest_sha256", "the committed manifest hashes to"),
        ("population_sha256", "the committed population recipe hashes to"),
        ("base_case", "the committed manifest names as its base surface-oracle case"),
        ("base_case_sha256", "the committed base surface-oracle case hashes to"),
        ("candidate_sha256", "the committed manifest seals the candidate bodies at"),
        ("protocol_sha256", "the committed manifest seals the protocol at"),
        ("selection_resolution", "the committed manifest declares selection at"),
    ):
        published = _report_text(corpus, key, where)
        if published != canonical[key]:
            raise ReportUnusable(
                f"{where} publishes {key} {published!r}; what {subject} is {canonical[key]!r}. "
                "This reader accepts a report for the committed corpus and for no other"
            )
    if _report_number(corpus, "material_fraction", where) != experiment.MATERIAL_FRACTION:
        raise ReportUnusable(f"{where} publishes a material fraction that is not the module's")
    if _report_int(corpus, "work_ceiling", where) != experiment.WORK_CEILING:
        raise ReportUnusable(f"{where} publishes a work ceiling that is not the module's")
    if _report_text(corpus, "selection_resolution", where) != experiment.SELECTION_RESOLUTION:
        raise ReportUnusable(f"{where} publishes a selection resolution that is not the module's")
    resolutions = _report_text_list(corpus, "resolutions", where)
    if len(set(resolutions)) != len(resolutions) or len(resolutions) < 2:
        raise ReportUnusable(f"{where} publishes resolutions {resolutions}")
    if experiment.SELECTION_RESOLUTION not in resolutions:
        raise ReportUnusable(f"{where} does not declare the resolution it selects at")
    if resolutions != canonical["resolutions"]:
        raise ReportUnusable(
            f"{where} publishes resolutions {resolutions}; the committed corpus declares "
            f"{canonical['resolutions']!r}, in that order"
        )

    members = [
        _report_mapping(item, f"{where} member") for item in _report_list(corpus, "members", where)
    ]
    if not members:
        raise ReportUnusable(f"{where} publishes no members")
    canonical_members = canonical["members"]
    assert isinstance(canonical_members, list)
    if len(members) != len(canonical_members):
        raise ReportUnusable(
            f"{where} publishes {len(members)} member(s); the committed corpus declares "
            f"{len(canonical_members)}"
        )
    ids: list[str] = []
    for position, (member, declared_member) in enumerate(
        zip(members, canonical_members, strict=True)
    ):
        assert isinstance(declared_member, Mapping)
        _report_keys(member, MEMBER_KEYS, f"{where} member")
        member_id = _report_text(member, "member_id", f"{where} member")
        ids.append(member_id)
        if member_id != declared_member["member_id"]:
            raise ReportUnusable(
                f"{where} publishes member {member_id!r} at position {position}, where the "
                f"committed corpus declares {declared_member['member_id']!r}. The declared "
                "member ORDER is part of the seal, not a presentation choice"
            )
        if _report_text(member, "family", f"{where} member") not in contract.PERTURBATIONS:
            raise ReportUnusable(f"{where} member {member_id} declares an unknown family")
        if _report_text(member, "split", f"{where} member") not in experiment.SPLITS:
            raise ReportUnusable(f"{where} member {member_id} declares an unknown split")
        if _report_text(member, "seed", f"{where} member") != f"shell-discriminant:{member_id}":
            raise ReportUnusable(
                f"{where} member {member_id} publishes seed {member['seed']!r}; every member's "
                "seed is exactly shell-discriminant:<member-id>"
            )
        _report_optional_text(member, "control_case", f"{where} member {member_id}")
        for key, subject in (
            ("family", "perturbation family"),
            ("split", "split"),
            ("seed", "seed"),
            ("control_case", "control case"),
            ("frame_digest", "frame digest"),
        ):
            if member[key] != declared_member[key]:
                raise ReportUnusable(
                    f"{where} member {member_id} publishes {subject} {member[key]!r}; the "
                    f"committed corpus declares {declared_member[key]!r}. A member is its "
                    "declaration, and a report that renames one describes a different corpus"
                )
        perturbation = _report_mapping(
            member.get("perturbation"), f"{where} member {member_id} perturbation"
        )
        _report_perturbation(perturbation, f"{where} member {member_id} 'perturbation'")
        _bind_exact(
            dict(perturbation),
            declared_member["perturbation"],
            f"{where} member {member_id} 'perturbation'",
            why="publishes perturbation parameters the committed corpus does not declare",
        )
    if len(set(ids)) != len(ids):
        raise ReportUnusable(f"{where} publishes one member twice")
    sealed = _report_text_list(corpus, "sealed_holdout", where)
    holdout = [
        _report_text(member, "member_id", f"{where} member")
        for member in members
        if _report_text(member, "split", f"{where} member") == experiment.HOLDOUT
    ]
    if sealed != holdout:
        raise ReportUnusable(
            f"{where} seals {sealed} and puts {holdout} in the holdout. The sealed members were "
            "declared before any of them was executed"
        )
    if sealed != canonical["sealed_holdout"]:
        raise ReportUnusable(
            f"{where} seals {sealed}; the committed manifest seals "
            f"{canonical['sealed_holdout']!r}, in that order"
        )
    for split in experiment.SPLITS:
        if not [
            member
            for member in members
            if _report_text(member, "split", f"{where} member") == split
        ]:
            raise ReportUnusable(f"{where} puts no member in the {split!r} split")
    bound = _report_mapping(corpus.get("upper_bound"), f"{where} 'upper_bound'")
    _report_keys(bound, ("member_id", "cell_id"), f"{where} 'upper_bound'")
    if _report_text(bound, "member_id", where) not in ids:
        raise ReportUnusable(f"{where} names an upper-bound member it does not declare")
    if _report_text(bound, "cell_id", where) not in resolutions:
        raise ReportUnusable(f"{where} names an upper-bound resolution it does not declare")
    _bind_exact(
        dict(bound),
        canonical["upper_bound"],
        f"{where} 'upper_bound'",
        why=(
            "names an upper bound the committed manifest did not predeclare. A predeclaration "
            "that only binds when it flatters the result is not a predeclaration"
        ),
    )
    # The catch-all. Everything above refuses for its own reason; this refuses
    # anything left over, so no field of the corpus block is unbound.
    _bind_exact(
        dict(corpus),
        canonical,
        where,
        why="is not the corpus this repository committed",
    )
    return corpus


def _check_counterfactual(
    entry: Mapping[str, object],
    what: str,
    *,
    population: experiment.Population,
    selected: Sequence[str],
) -> None:
    """Bind one counterfactual to its canonical population, then recompute it.

    The identity half is bound to ``population`` - the one the committed corpus
    rebuilt - so a counterfactual cannot claim another member, another
    resolution, another split or another reconstruction. The account half is
    recomputed from the two measurements the entry publishes: which shells the
    rule takes out, what that removes, what it damages, every gate and the
    passed status.
    """
    closed = population.observables
    records = population.records
    _report_keys(entry, COUNTERFACTUAL_KEYS, what)
    if _report_int(entry, "schema_version", what) != experiment.SCHEMA_VERSION:
        raise ReportUnusable(f"{what} is counterfactual schema {entry['schema_version']!r}")
    if _report_text(entry, "kind", what) != COUNTERFACTUAL_KIND:
        raise ReportUnusable(f"{what} is kind {entry['kind']!r}, not a {COUNTERFACTUAL_KIND}")
    for key, declared in (
        ("member_id", population.member_id),
        ("cell_id", population.cell_id),
        ("split", population.split),
        ("reconstruction_digest", population.reconstruction_digest),
    ):
        if _report_text(entry, key, what) != declared:
            raise ReportUnusable(
                f"{what} publishes {key} {entry[key]!r}; the population it was measured on is "
                f"{declared!r}"
            )
    before = _report_mapping(entry["before"], f"{what} 'before'")
    after = _report_mapping(entry["after"], f"{what} 'after'")
    _report_account(before, f"{what} 'before'")
    _report_account(after, f"{what} 'after'")
    if _report_int(entry, "closed_shells", what) != len(closed):
        raise ReportUnusable(
            f"{what} says it was applied to {entry['closed_shells']!r} closed shell(s); its "
            f"population publishes {len(closed)}"
        )
    published_ids = _report_text_list(entry, "removed_shell_ids", what)
    if published_ids != sorted(set(selected)):
        raise ReportUnusable(
            f"{what} says it removed {published_ids}; applying the published selected rule to "
            f"every closed shell of this population selects {sorted(set(selected))}"
        )
    by_id = {shell.shell_id: shell for shell in closed}
    labelled = {record.shell_id: record for record in records}
    removed = [by_id[shell_id] for shell_id in published_ids]
    if _report_int(entry, "removed_faces", what) != sum(shell.faces for shell in removed):
        raise ReportUnusable(f"{what} publishes a removed-face count its own shells do not give")
    if _report_int(before, "faces", what) - _report_int(after, "faces", what) != _report_int(
        entry, "removed_faces", what
    ):
        raise ReportUnusable(
            f"{what} publishes a counterfactual whose face counts do not reconcile with the "
            f"{entry['removed_faces']!r} face(s) it says it removed. A claimed gain the "
            "recomputed mesh does not show is not a gain"
        )
    false_shells = [
        s for s in removed if s.shell_id in labelled and labelled[s.shell_id].truth.false_surface
    ]
    correct_shells = [
        s
        for s in removed
        if s.shell_id in labelled and not labelled[s.shell_id].truth.false_surface
    ]
    supported = [s for s in removed if s.shell_id not in labelled]
    for key, value in (
        ("removed_false_shells", len(false_shells)),
        ("removed_false_shell_faces", sum(s.faces for s in false_shells)),
        ("removed_correct_shells", len(correct_shells)),
        ("removed_correct_shell_faces", sum(s.faces for s in correct_shells)),
        ("removed_supported_shells", len(supported)),
        ("removed_supported_shell_faces", sum(s.faces for s in supported)),
    ):
        if _report_int(entry, key, what) != value:
            raise ReportUnusable(
                f"{what} publishes {key}={entry[key]!r}; its own removed shells give {value}"
            )
    deltas = {
        "false_faces_removed": _report_int(before, "unsupported_false_shell_faces", what)
        - _report_int(after, "unsupported_false_shell_faces", what),
        "correct_faces_removed": _report_int(before, "unsupported_correct_shell_faces", what)
        - _report_int(after, "unsupported_correct_shell_faces", what),
    }
    for key, value in deltas.items():
        if _report_int(entry, key, what) != value:
            raise ReportUnusable(
                f"{what} publishes {key}={entry[key]!r}; its own before/after measurements give "
                f"{value}"
            )
    gates = {
        "counterfactual_false_faces_fell": deltas["false_faces_removed"] > 0,
        "counterfactual_correct_faces_held": deltas["correct_faces_removed"] == 0,
        "counterfactual_removes_no_supported_faces": not supported,
        "counterfactual_analytic_accuracy_held": _report_number(
            after, "analytic_within_tolerance_fraction", what
        )
        >= _report_number(before, "analytic_within_tolerance_fraction", what),
        "counterfactual_triangle_accuracy_held": _report_number(
            after, "triangle_within_tolerance_fraction", what
        )
        >= _report_number(before, "triangle_within_tolerance_fraction", what),
        "counterfactual_reference_coverage_held": _report_number(
            after, "reference_coverage_fraction", what
        )
        >= _report_number(before, "reference_coverage_fraction", what),
        "counterfactual_protected_volume_held": _report_number(after, "protected_area", what)
        <= _report_number(before, "protected_area", what),
        "counterfactual_bridging_held": _report_int(after, "bridging_faces", what)
        <= _report_int(before, "bridging_faces", what),
        "counterfactual_edge_coverage_held": _report_number(after, "edge_coverage", what)
        >= _report_number(before, "edge_coverage", what),
        "counterfactual_thin_feature_held": _report_number(after, "thin_feature_coverage", what)
        >= _report_number(before, "thin_feature_coverage", what),
        "counterfactual_dimensions_held": _report_number(after, "worst_dimension_error", what)
        <= _report_number(before, "worst_dimension_error", what),
        "counterfactual_topology_held": (
            _report_int(after, "nonmanifold_edge_count", what)
            <= _report_int(before, "nonmanifold_edge_count", what)
            and _report_int(after, "boundary_edge_count", what)
            <= _report_int(before, "boundary_edge_count", what)
        ),
    }
    published_gates = _report_mapping(entry["gates"], f"{what} 'gates'")
    _report_keys(published_gates, tuple(gates), f"{what} 'gates'")
    for name, value in sorted(gates.items()):
        if _report_bool(published_gates, name, what) != value:
            raise ReportUnusable(
                f"{what} publishes {name}={published_gates[name]!r}; its own before/after "
                f"measurements give {value}"
            )
    if _report_bool(entry, "passed", what) != all(gates.values()):
        raise ReportUnusable(f"{what} publishes a passed status its own gates do not give")


def _bind_counterfactuals(
    counterfactuals: Sequence[object],
    evidence: _Evidence,
    rule: experiment.Rule,
    what: str,
) -> dict[str, bool]:
    """The counterfactual list, as an exact ordered set the evidence determines.

    A recommendation rests on what the selected rule does to real meshes, so
    the list of populations it was measured on is not a sample anybody chooses.
    It is exactly the sealed HOLDOUT populations at the shipped resolution on
    which applying the rule to EVERY closed shell selects at least one - derived
    here from the canonical rebuild, in the corpus's declared order.

    Round two derived a population's split from whether an unsupported labelled
    record happened to exist in it, so a sealed holdout population made only of
    supported closed shells escaped its counterfactual even when the rule fired
    on it. The split comes from the corpus now.
    """
    corpus = evidence.corpus
    expected: list[tuple[str, str]] = []
    for member_id, cell_id in evidence.cells:
        population = evidence.populations[(member_id, cell_id)]
        if cell_id != experiment.SELECTION_RESOLUTION:
            continue
        if population.split != experiment.HOLDOUT or member_id not in corpus.sealed_holdout:
            continue
        with experiment.accounting(experiment.READER_PHASE):
            fired = experiment.select(rule, population.observables)
        if fired:
            expected.append((member_id, cell_id))

    seen: set[tuple[str, str]] = set()
    published: list[tuple[str, str]] = []
    gates: dict[str, bool] = {}
    for position, item in enumerate(counterfactuals):
        entry = _report_mapping(item, f"{what} counterfactual {position}")
        member_id = _report_text(entry, "member_id", f"{what} counterfactual {position}")
        cell_id = _report_text(entry, "cell_id", f"{what} counterfactual {position}")
        where = f"{what} counterfactual {member_id}/{cell_id}"
        if (member_id, cell_id) in seen:
            raise ReportUnusable(
                f"{what} publishes a counterfactual for {member_id}/{cell_id} twice. One "
                "population is one measurement, and counting it twice counts its gain twice"
            )
        seen.add((member_id, cell_id))
        here = evidence.populations.get((member_id, cell_id))
        if here is None:
            raise ReportUnusable(
                f"{what} publishes a counterfactual for {member_id}/{cell_id}, which the "
                "committed corpus does not build. An invented population is not evidence"
            )
        if cell_id != experiment.SELECTION_RESOLUTION:
            raise ReportUnusable(
                f"{what} publishes a counterfactual at the diagnostic resolution {cell_id!r}. "
                f"The recommendation is made at {experiment.SELECTION_RESOLUTION!r}, and a "
                "measurement at another resolution cannot support it"
            )
        if here.split != experiment.HOLDOUT or member_id not in corpus.sealed_holdout:
            raise ReportUnusable(
                f"{what} publishes a counterfactual on {member_id}/{cell_id}, which is in the "
                f"{here.split!r} split. A counterfactual is held-out evidence, and a rule "
                "re-measured on the data it was chosen on measures nothing"
            )
        if (member_id, cell_id) not in expected:
            raise ReportUnusable(
                f"{what} publishes a counterfactual on {member_id}/{cell_id}, where the selected "
                "rule fires on no closed shell. There was nothing there to remove"
            )
        with experiment.accounting(experiment.READER_PHASE):
            fired = experiment.select(rule, here.observables)
        _check_counterfactual(entry, where, population=here, selected=fired)
        # And then against a mesh. Everything above refuses an entry that does
        # not agree with itself; this refuses one that agrees with itself
        # perfectly and is still not what pruning those shells from that member
        # actually does.
        _bind_exact(
            dict(entry),
            evidence.remeasure(here, fired),
            where,
            why="is not the counterfactual the committed corpus measures on that population",
        )
        for name, value in _report_mapping(entry["gates"], where).items():
            gates[name] = gates.get(name, True) and bool(value)
        published.append((member_id, cell_id))

    if published != expected:
        missing = [f"{member}/{cell}" for member, cell in expected if (member, cell) not in seen]
        if missing:
            raise ReportUnusable(
                f"{what} recommends on a rule that fires on sealed holdout population(s) "
                f"{missing} and publishes no counterfactual for them. A partial sample can never "
                "support a recommendation"
            )
        raise ReportUnusable(
            f"{what} publishes counterfactuals for {published}; the sealed holdout populations "
            f"the selected rule fires on, in the corpus's declared order, are {expected}"
        )
    return gates


def _bind_upper_bound(report: Mapping[str, object], evidence: _Evidence, what: str) -> None:
    """The predeclared ceiling: its FORM from the evidence, then its content.

    Round two checked the bound only when it said ``measured: true``, so its
    unmeasured form was an open namespace a fabricated gain could be added to,
    and the form itself was a boolean the report chose. Both are decided here
    by the predeclared population rebuilt from the committed corpus: if it
    carries false unsupported shells the bound is measured and every number in
    it is the recomputed one, and if it does not the bound is exactly six keys,
    an empty removal list and the canonical reason.
    """
    where = f"{what} 'upper_bound'"
    published = report["upper_bound"]
    canonical = evidence.upper_bound_block
    if canonical is None:
        if published is not None:
            raise ReportUnusable(
                f"{what} publishes an upper bound where the committed corpus declares none"
            )
        return
    entry = _report_mapping(published, where)
    member_id = _report_text(entry, "member_id", where)
    cell_id = _report_text(entry, "cell_id", where)
    if (member_id, cell_id) != (
        evidence.corpus.upper_bound_member,
        evidence.corpus.upper_bound_cell,
    ):
        raise ReportUnusable(
            f"{what} measures its upper bound on {member_id}/{cell_id}; the committed manifest "
            f"predeclared {evidence.corpus.upper_bound_member}/"
            f"{evidence.corpus.upper_bound_cell}. A predeclaration that only binds when it "
            "flatters the result is not a predeclaration"
        )
    if _report_text(entry, "selector", where) != UPPER_BOUND_SELECTOR:
        raise ReportUnusable(f"{what} publishes an upper bound that is not labelled truth-fed")
    measured = _report_bool(entry, "measured", where)
    canonical_measured = _measured(canonical)
    if measured != canonical_measured:
        state = "measurable" if canonical_measured else "not measurable"
        raise ReportUnusable(
            f"{where} publishes measured={measured!r}; the predeclared population rebuilt from "
            f"the committed corpus is {state}. The form follows the population, never a boolean "
            "in the report"
        )
    if not measured:
        _report_keys(entry, UNMEASURED_BOUND_KEYS, where)
        if _report_text_list(entry, "removed_shell_ids", where):
            raise ReportUnusable(f"{where} lists removed shells and says it was never measured")
        _report_text(entry, "why", where)
        _bind_exact(
            dict(entry),
            canonical,
            where,
            why="is not the unmeasured ceiling the committed corpus produces",
        )
        return
    _report_keys(entry, MEASURED_BOUND_KEYS, where)
    population = evidence.populations.get((member_id, cell_id))
    if population is None:
        raise ReportUnusable(
            f"{where} measures a ceiling on a population the committed corpus does not build"
        )
    _check_counterfactual(
        {key: value for key, value in entry.items() if key not in ("selector", "measured")},
        where,
        population=population,
        selected=[record.shell_id for record in population.records if record.truth.false_surface],
    )
    _bind_exact(
        dict(entry),
        canonical,
        where,
        why="is not the ceiling the committed corpus measures on the predeclared member",
    )


def _expected_totals(evidence: _Evidence) -> dict[str, int]:
    """Every total, recomputed from the canonical rebuild rather than the report."""
    populations = [item.population for item in evidence.built]
    records = list(evidence.records)
    return {
        "members": len(evidence.corpus.members),
        "resolutions": len(evidence.corpus.resolutions),
        "populations": len(populations),
        "closed_shells": sum(len(item.observables) for item in populations),
        "shells": len(records),
        "supported_closed_shells": sum(
            len(item.observables) - len(item.records) for item in populations
        ),
        "train_shells": sum(1 for record in records if record.split == experiment.TRAIN),
        "holdout_shells": sum(1 for record in records if record.split == experiment.HOLDOUT),
        "selection_train_shells": len(experiment.training_records(records)),
        "selection_holdout_shells": len(experiment.holdout_records(records)),
        "false_shells": sum(1 for record in records if record.truth.false_surface),
        "correct_shells": sum(1 for record in records if not record.truth.false_surface),
        "false_faces": sum(record.truth.faces for record in records if record.truth.false_surface),
        "correct_faces": sum(
            record.truth.faces for record in records if not record.truth.false_surface
        ),
        "candidates": len(experiment.RULE_GRID),
        "cross_tab_cells": len(experiment.cross_cells(evidence.corpus.resolutions)),
        "evaluations": len(experiment.RULE_GRID) * len(records),
        "failures": len(evidence.failures),
    }


def _bind_cost(
    report: Mapping[str, object],
    evidence: _Evidence,
    what: str,
    *,
    rule: experiment.Rule | None,
    counterfactuals: int,
) -> None:
    """Every line of the work account, recomputed from the evidence and the protocol.

    Round two recomputed the producer's search and took the rest on trust, so a
    voxel count, a triangle count and a counterfactual rule-call total could all
    be raised together and stay coherent. Each of them is a function of the
    canonical rebuild and of exactly how this protocol spends a rule: one
    producer call per candidate per labelled shell, one counterfactual call per
    closed shell of every sealed holdout appliance population when a rule is
    selected and none when one is not, and two remeasurements for each emitted
    counterfactual plus two for a measured ceiling.
    """
    where = f"{what} 'cost'"
    cost = _report_mapping(report["cost"], where)
    _report_keys(cost, COST_KEYS, where)
    populations = [item.population for item in evidence.built]
    evaluations = len(experiment.RULE_GRID) * len(evidence.records)
    if evaluations > experiment.WORK_CEILING:
        raise ReportUnusable(
            f"{what} publishes {evaluations} evaluation(s) past the declared ceiling of "
            f"{experiment.WORK_CEILING}"
        )
    counterfactual_calls = (
        sum(
            len(item.observables)
            for item in populations
            if item.split == experiment.HOLDOUT
            and item.cell_id == experiment.SELECTION_RESOLUTION
            and item.member_id in evidence.corpus.sealed_holdout
        )
        if rule is not None
        else 0
    )
    expected = {
        "fusions": len(populations),
        "voxel_visits": sum(item.work_units for item in populations),
        "triangle_samples": sum(item.triangle_samples for item in populations),
        "producer_rule_calls": evaluations,
        "counterfactual_rule_calls": counterfactual_calls,
        # A producer publishes a producer's account. A standalone reader's own
        # audit matrix is the reader's work and is never inside this number.
        "reader_audit_rule_calls": 0,
        "unaccounted_rule_calls": 0,
        "total_rule_calls": evaluations + counterfactual_calls,
        "decision_matrix_cells": evaluations,
        "work_ceiling": experiment.WORK_CEILING,
        "counterfactual_remeasurements": 2
        * (counterfactuals + (1 if _measured(evidence.upper_bound_block) else 0)),
    }
    for key, value in expected.items():
        if _report_int(cost, key, where) != value:
            raise ReportUnusable(
                f"{where} publishes {key}={cost[key]!r}; the committed corpus and this protocol "
                f"give {value}"
            )
    if _report_text(cost, "note", where) != COST_NOTE:
        raise ReportUnusable(
            f"{where} publishes a note that is not the module's. A note is where an account "
            "explains itself, and one rewritten to fit its numbers explains nothing"
        )


def _validate_document(document: object, evidence: _Evidence, *, what: str) -> Mapping[str, object]:
    """Establish this artifact against ``evidence``, and recompute what it derives.

    Two things, and neither of them is a shape check. The first is a binding:
    the corpus block, every realization, every population, every closed shell,
    every labelled record, the work account and the predeclared upper bound are
    compared against a rebuild of the committed experiment, so a coherent
    forgery is refused because the evidence does not contain it. The second is a
    recomputation: every typed value is rebuilt and run through the module's own
    invariants, ONE independent audit decision matrix is built over the rebuilt
    records and accounted separately, and the twenty-eight-cell cross-tab, every
    aggregate, the eligible ordering, the selected rule and its stated reason,
    the holdout confusion, every gate, every counterfactual's deltas and gate
    booleans, the recommendation, every total and the decision are recomputed
    from it.

    Private, and takes its evidence rather than finding it. The public entry
    points below always pass the committed corpus's rebuild; the parameter
    exists so the derived-logic half can be exercised over small populations
    without minutes of fusion, and never so a caller can choose what a published
    report is checked against.
    """
    report = _report_mapping(document, what)
    _report_keys(report, REPORT_KEYS, what)
    if _report_int(report, "schema_version", what) != REPORT_SCHEMA_VERSION:
        raise ReportUnusable(f"{what} is schema {report['schema_version']!r}")
    if _report_text(report, "kind", what) != REPORT_KIND:
        raise ReportUnusable(f"{what} is not a {REPORT_KIND}")
    if _report_text(report, "protocol", what) != experiment.SELECTION_PROTOCOL:
        raise ReportUnusable(f"{what} publishes a selection protocol that is not the module's")
    if _report_text_list(report, "gates", what) != list(experiment.GATES):
        raise ReportUnusable(f"{what} publishes a gate list that is not the module's")
    if _report_text_list(report, "limitations", what) != list(LIMITATIONS):
        raise ReportUnusable(f"{what} publishes limitations that are not the module's")

    namespaces = _report_mapping(report["namespaces"], f"{what} 'namespaces'")
    _report_keys(namespaces, NAMESPACE_KEYS, f"{what} 'namespaces'")
    if _report_text(namespaces, "observable_source", what) != experiment.OBSERVABLE_SOURCE:
        raise ReportUnusable(f"{what} publishes an observable source that is not the module's")
    if _report_text(namespaces, "label_source", what) != experiment.LABEL_SOURCE:
        raise ReportUnusable(f"{what} publishes a label source that is not the module's")
    if _report_text_list(namespaces, "observables", what) != sorted(experiment.FEATURES):
        raise ReportUnusable(f"{what} publishes an observable list that is not the module's")
    if _report_text_list(namespaces, "label_only_fields", what) != list(
        experiment.LABEL_ONLY_FIELDS
    ):
        raise ReportUnusable(f"{what} publishes a label namespace that is not the module's")

    corpus = _bind_corpus(report, evidence, what)
    resolutions = _report_text_list(corpus, "resolutions", what)
    _bind_realizations(report, evidence, what)
    records = _bind_populations(report, evidence, what)
    _bind_upper_bound(report, evidence, what)

    # One independent audit matrix, accounted as the reader's own work and
    # never as producer search.
    audit = experiment.build_decision_matrix(records, phase=experiment.READER_PHASE)

    candidates = _report_list(report, "candidates", what)
    declared = list(experiment.RULE_GRID)
    if len(candidates) != len(declared):
        raise ReportUnusable(
            f"{what} publishes {len(candidates)} candidate(s) against the {len(declared)} the "
            "module declares"
        )
    aggregates = experiment.aggregate_slices(records)
    cross_names = [experiment.cross_tab_name(*cell) for cell in experiment.cross_cells(resolutions)]
    for index, (item, rule) in enumerate(zip(candidates, declared, strict=True)):
        entry = _report_mapping(item, f"{what} candidate {index}")
        _report_keys(entry, ("rule", "cross_tab", "aggregates"), f"{what} candidate {index}")
        body = _typed_rule(entry["rule"], f"{what} candidate {index} 'rule'")
        if dict(body) != rule.as_dict():
            raise ReportUnusable(
                f"{what} candidate {index} publishes rule body {dict(body)!r} where the declared "
                f"candidate at that position is {rule.as_dict()!r}. A body edited after a result "
                "cannot reuse an accepted id here"
            )
        cross = _report_mapping(entry["cross_tab"], f"{what} {rule.rule_id} cross_tab")
        if sorted(cross) != sorted(cross_names):
            missing = sorted(set(cross_names) - set(cross))
            extra = sorted(set(cross) - set(cross_names))
            raise ReportUnusable(
                f"{what} {rule.rule_id} publishes {len(cross)} cross-tab cell(s) against the "
                f"{len(cross_names)} declared. Missing {missing}; unexpected {extra}. Every "
                "split x family x resolution cell, including the ones no shell landed in"
            )
        recomputed_cells = {
            cell.slice_name: cell
            for cell in experiment.cross_tab(audit, rule.rule_id, records, resolutions=resolutions)
        }
        for name in cross_names:
            published = _rebuild_confusion(cross[name], f"{what} {rule.rule_id}/{name}")
            _same_confusion(published, recomputed_cells[name], f"{what} {rule.rule_id}/{name}")
        published_aggregates = _report_mapping(
            entry["aggregates"], f"{what} {rule.rule_id} aggregates"
        )
        if sorted(published_aggregates) != sorted(aggregates):
            raise ReportUnusable(
                f"{what} {rule.rule_id} publishes aggregate slices that are not the declared ones"
            )
        for name, subset in aggregates.items():
            published = _rebuild_confusion(
                published_aggregates[name], f"{what} {rule.rule_id}/{name}"
            )
            _same_confusion(
                published,
                experiment.confusion(audit, rule.rule_id, subset, slice_name=name),
                f"{what} {rule.rule_id}/{name}",
            )

    selection = _report_mapping(report["selection"], f"{what} 'selection'")
    _report_keys(selection, SELECTION_KEYS, f"{what} 'selection'")
    if _report_text(selection, "kind", f"{what} 'selection'") != SELECTION_KIND:
        raise ReportUnusable(f"{what} 'selection' is kind {selection['kind']!r}")
    # Typed before the comparison, for the same reason a rule body is: the
    # blanket ``!=`` below inherits Python's ``True == 1``, so every scalar in
    # the block is read for its type first.
    if _report_int(selection, "schema_version", f"{what} 'selection'") != experiment.SCHEMA_VERSION:
        raise ReportUnusable(f"{what} 'selection' is schema {selection['schema_version']!r}")
    for key in ("considered_rules", "evaluations", "work_ceiling", "training_shells"):
        _report_int(selection, key, f"{what} 'selection'")
    for key in ("selection_resolution", "protocol", "reason"):
        _report_text(selection, key, f"{what} 'selection'")
    _report_text_list(selection, "eligible_rule_ids", f"{what} 'selection'")
    if selection["selected"] is not None:
        _typed_rule(selection["selected"], f"{what} 'selection' 'selected'")
    recomputed = experiment.choose(audit, records)
    if dict(_report_mapping(selection, f"{what} 'selection'")) != recomputed.as_dict():
        raise ReportUnusable(
            f"{what} publishes a selection the reader's own audit matrix does not reproduce. "
            f"Published {dict(selection)!r}; recomputed {recomputed.as_dict()!r}"
        )

    holdout = experiment.holdout_records(records)
    training = experiment.training_records(records)
    published_holdout = report["holdout_confusion"]
    if recomputed.rule is None:
        if published_holdout is not None:
            raise ReportUnusable(f"{what} publishes a holdout confusion for no selected rule")
    else:
        name = f"split:{experiment.HOLDOUT}|cell:{experiment.SELECTION_RESOLUTION}"
        _same_confusion(
            _rebuild_confusion(published_holdout, f"{what} 'holdout_confusion'"),
            experiment.confusion(audit, recomputed.rule.rule_id, holdout, slice_name=name),
            f"{what} 'holdout_confusion'",
        )

    gate_results = _report_mapping(report["gate_results"], f"{what} 'gate_results'")
    unknown = sorted(set(gate_results) - set(experiment.GATES))
    if unknown:
        raise ReportUnusable(f"{what} reports gates {unknown} that this experiment does not name")

    counterfactuals = _report_list(report, "counterfactuals", what)
    expected_gates: dict[str, bool] = {}
    if recomputed.rule is not None:
        train_counts = experiment.confusion(
            audit,
            recomputed.rule.rule_id,
            training,
            slice_name=f"split:{experiment.TRAIN}|cell:{experiment.SELECTION_RESOLUTION}",
        )
        held = experiment.confusion(
            audit,
            recomputed.rule.rule_id,
            holdout,
            slice_name=f"split:{experiment.HOLDOUT}|cell:{experiment.SELECTION_RESOLUTION}",
        )
        expected_gates["train_removes_no_correct_faces"] = train_counts.removes_no_correct_faces
        expected_gates["train_removes_material_false_faces"] = (
            train_counts.removes_material_false_faces
        )
        expected_gates["holdout_removes_no_correct_faces"] = held.removes_no_correct_faces
        expected_gates["holdout_removes_material_false_faces"] = held.removes_material_false_faces
        for name, passed in _bind_counterfactuals(
            counterfactuals, evidence, recomputed.rule, what
        ).items():
            expected_gates[name] = passed
        if not counterfactuals:
            for gate in experiment.GATES:
                if gate.startswith("counterfactual_"):
                    expected_gates[gate] = False
    elif counterfactuals:
        raise ReportUnusable(f"{what} publishes a counterfactual with no selected rule")

    published_gates = {
        name: _report_bool(gate_results, name, f"{what} 'gate_results'") for name in gate_results
    }
    if published_gates != expected_gates:
        raise ReportUnusable(
            f"{what} publishes gate results the reader does not reproduce. Published "
            f"{dict(gate_results)!r}; recomputed {expected_gates!r}"
        )

    recommendation = _report_text(report, "recommendation", what)
    if recommendation not in (PRUNE, NO_PRUNING):
        raise ReportUnusable(f"{what} recommends {recommendation!r}")
    expected_recommendation = (
        PRUNE
        if recomputed.rule is not None
        and counterfactuals
        and all(expected_gates.get(gate, False) for gate in experiment.GATES)
        else NO_PRUNING
    )
    if recommendation != expected_recommendation:
        raise ReportUnusable(
            f"{what} recommends {recommendation!r}; its own selection, counterfactuals and gates "
            f"give {expected_recommendation!r}"
        )

    failures = _report_text_list(report, "failures", what)
    if failures != list(evidence.failures):
        raise ReportUnusable(
            f"{what} publishes failures {failures}; building the committed corpus produced "
            f"{list(evidence.failures)}. A failure list is evidence about the run, not a summary"
        )
    totals = _report_mapping(report["totals"], f"{what} 'totals'")
    expected_totals = _expected_totals(evidence)
    _report_keys(totals, tuple(expected_totals), f"{what} 'totals'")
    for key, value in expected_totals.items():
        if _report_int(totals, key, f"{what} 'totals'") != value:
            raise ReportUnusable(
                f"{what} publishes totals[{key!r}]={totals[key]!r}; the committed corpus gives "
                f"{value}"
            )
    _bind_cost(report, evidence, what, rule=recomputed.rule, counterfactuals=len(counterfactuals))

    decision = _report_text(report, "decision", what)
    if decision not in ("pass", "fail"):
        raise ReportUnusable(f"{what} decides {decision!r}")
    if (decision == "fail") != bool(failures):
        raise ReportUnusable(f"{what} records {len(failures)} failure(s) and decides {decision!r}")
    return report


def validate_report(
    document: object, *, what: str = "shell-discriminant report"
) -> Mapping[str, object]:
    """Establish an artifact against the experiment THIS repository committed.

    The public boundary, and it takes no evidence root. A report is a claim
    about the sealed corpus in ``tests/fixtures/shell_discriminant``, so the
    reader rebuilds that corpus itself - every member, at every declared
    resolution, from the committed bytes - and binds the document to it before
    auditing anything the document derives. A report for a copied or corrupted
    fixture root is refused however faithfully it describes that root;
    :func:`load_corpus` remains the named API for checking one of those.

    The rebuild is minutes of fusion the first time in a process and free
    afterwards. That is the price of a reader that establishes its evidence
    rather than checking a document against itself.
    """
    return _validate_document(document, _trusted_evidence(), what=what)


def loads_report(text_: str, *, what: str = "shell-discriminant report") -> Mapping[str, object]:
    """Read an artifact from its TEXT, strictly, and validate it.

    From the text rather than from an already-decoded mapping: a duplicated key
    is one mapping by the time an ordinary decoder is done with it, and no
    validator over that mapping could tell the file had two answers in it. The
    same goes for ``NaN``, ``Infinity`` and ``-Infinity``: CPython's decoder
    accepts all three by default, and a measurement that decodes to a NaN
    compares false against every bar it is gated on, so they are refused at the
    boundary rather than read.
    """

    def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        seen: dict[str, object] = {}
        for key, value in pairs:
            if key in seen:
                raise ReportUnusable(f"{what} states {key!r} twice in one object")
            seen[key] = value
        return seen

    def _no_constants(name: str) -> object:
        raise ReportUnusable(
            f"{what} carries the JSON constant {name!r}. A report is a set of finite "
            "measurements; a NaN or an infinity is a number that passes every comparison it is "
            "not supposed to"
        )

    try:
        document = json.loads(text_, object_pairs_hook=_no_duplicates, parse_constant=_no_constants)
    except json.JSONDecodeError as exc:
        raise ReportUnusable(f"{what} is not JSON: {exc}") from exc
    return validate_report(document, what=what)


def read_report(path: Path, *, what: str | None = None) -> Mapping[str, object]:
    return loads_report(path.read_text(encoding="utf-8"), what=what or str(path))


# --- rendering and the command ---------------------------------------------------------


def render(document: Mapping[str, object]) -> str:
    """A short human summary. Every number in it is already in the artifact."""
    totals = _report_mapping(document["totals"], "totals")
    selection = _report_mapping(document["selection"], "selection")
    lines = [
        "Observable-shell discriminator - can an appliance tell a false sealed shell from a "
        "correct one?",
        "",
        f"members {totals['members']} at {totals['resolutions']} resolution(s); "
        f"{totals['shells']} unsupported closed shell(s) "
        f"({totals['train_shells']} train, {totals['holdout_shells']} holdout)",
        f"population: {totals['false_shells']} false shell(s) / {totals['false_faces']} face(s), "
        f"{totals['correct_shells']} correct shell(s) / {totals['correct_faces']} face(s)",
        f"candidates: {totals['candidates']} declared rule(s), "
        f"{totals['evaluations']} evaluation(s)",
        "",
    ]
    chosen = selection.get("selected")
    if isinstance(chosen, Mapping):
        lines.append(f"selected: {chosen.get('rule_id')}  ({chosen.get('text')})")
    else:
        lines.append("selected: none")
    lines.append(f"reason: {selection.get('reason')}")
    gate_results = _report_mapping(document["gate_results"], "gate_results")
    if gate_results:
        lines.append("gates:")
        lines.extend(
            f"  {'pass' if value else 'FAIL'}  {name}"
            for name, value in sorted(gate_results.items())
        )
    bound = document.get("upper_bound")
    if isinstance(bound, Mapping) and bound.get("measured"):
        lines.append(
            f"truth-fed upper bound on {bound.get('member_id')}/{bound.get('cell_id')}: "
            f"{bound.get('false_faces_removed')} false face(s) removable at best"
        )
    lines.append("")
    lines.append(f"recommendation: {document['recommendation']}")
    failures = document.get("failures")
    if isinstance(failures, list) and failures:
        lines.append("")
        lines.append("failures:")
        lines.extend(f"  {item}" for item in failures)
    lines.append("")
    lines.append(f"decision: {document['decision']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.shell_discriminant_harness",
        description=(
            "Ask whether a bounded rule over observable evidence can separate false sealed "
            "shells from correct ones, and publish the artifact."
        ),
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS, help="the corpus directory")
    parser.add_argument("--out", type=Path, default=None, help="where to write the artifact")
    parser.add_argument(
        "--quiet", action="store_true", help="publish the artifact without rendering it"
    )
    args = parser.parse_args(argv)

    if args.out is not None:
        # Cleared before the corpus is loaded, so a requested destination is
        # never left holding an EARLIER report after a run that could not
        # produce one.
        try:
            clear_destination(args.out)
        except ScannerError as error:
            print(f"cannot publish to {args.out}: {error}", file=sys.stderr)
            return 2
    try:
        document = run(args.corpus)
    except (HarnessRefused, FixtureUnusable, ReportUnusable) as error:
        print(f"the experiment could not be run: {error}", file=sys.stderr)
        return 2
    if args.out is not None:
        try:
            publish(document, args.out)
        except ScannerError as error:
            print(f"the artifact could not be published: {error}", file=sys.stderr)
            return 2
    if not args.quiet:
        print(render(document))
    return 0 if document["decision"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover - the command's own entry point
    raise SystemExit(main())
