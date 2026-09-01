"""Can an appliance tell a false sealed shell from a correct one, without truth?

The accepted surface oracle (:mod:`tests.surface_oracle`) established the fact
this module starts from. A fused surface contains **unsupported closed
shells** - sealed, watertight blobs standing where none of the scene's views
measured anything - and those shells are not one thing:

* some are **false surface**: nothing measured near them AND they are not on
  the object. Removing them is pure gain;
* some are **correct surface**: geometry that really is on the object and that
  nothing happened to measure. Removing them is pure loss.

The accepted artifact publishes those two groups separately and refuses to
collapse them, which is why the obvious "delete what nothing supports" rule has
never been allowed anywhere near the product. This module asks the next
question, and it is an empirical one rather than an opinion:

    Using only evidence the appliance actually has, can a bounded rule
    separate the false shells from the correct ones - removing a material
    amount of false surface while removing none of the correct surface?

Two namespaces, and one of them is a wall
-----------------------------------------
Everything here is built around the fact that the answer is worthless if the
candidate rule can see the answer. So a shell is described twice, by two
disjoint record types:

* :class:`ObservableShell` - what an appliance could compute at export time:
  the shell's own size and extent, which camera frusta contain it, what the
  RAW frames say along the rays that cross it, how far it is from the nearest
  measured return, and what the fused field's own weights are underneath it.
  Every field is derived from raw frames, the accepted poses, the intrinsics,
  the fused TSDF field and the extracted topology, and from nothing else;
* :class:`ShellTruth` - the analytic label: is this shell within the fixture's
  tolerance of the exact object, and how far off is it at its worst. This is
  computed by :func:`tests.surface_oracle.analytic_distance` from the scene's
  declared shapes, and it exists for the EVALUATOR only.

:func:`select` - the only way a rule is ever applied - accepts a sequence of
:class:`ObservableShell` and nothing else. Not a subclass, not a
:class:`ShellRecord`, not a mapping: ``type(item) is ObservableShell`` or it is
refused as :class:`TruthLeak`. A rule names its features out of
:data:`FEATURES`, a closed table of observable accessors, so a rule that tried
to name ``false_surface`` is refused at construction. Truth leakage is not
discouraged here, it is unrepresentable, and :mod:`tests.test_shell_discriminant`
attacks it directly rather than taking that on trust.

An item that cannot be derived is unavailable, never substituted
---------------------------------------------------------------
If a bounded API cannot produce one of the observable items for a shell - the
fused weights when no field was kept, for instance - the item is ``None`` and
its name is listed in :attr:`ObservableShell.unavailable`. It is never filled
in from analytic truth and never quietly defaulted to a number a rule could
threshold. :func:`Rule.holds` refuses to fire on a shell whose feature is
unavailable, so an unavailable item can never be read as a small one.

Predeclared, and in that order
------------------------------
:data:`RULE_GRID` is the complete candidate matrix - every rule that will ever
be evaluated - and it is code, reviewed before any result exists, with its
digest bound into the fixture manifest so a post-result edit to either side
shows up as a refusal. :data:`SELECTION_PROTOCOL` states, in one paragraph, the
order in which candidates are considered, which split may inform the choice
(training truth only) and which may not (holdout truth is evaluator-only), and
the gates a rule must clear. Nothing here searches adaptively, and no threshold
is authored after a result.

Counting is not enough
----------------------
A rule that "classifies well" has proved nothing about the mesh an owner gets.
So a rule that survives its gates is applied for real: the selected whole
shells are removed from the extracted surface in
:func:`prune`, and the pruned mesh is re-measured over the same inputs by the
accepted oracle's own measurement code - analytic accuracy and coverage,
triangle interiors, the protected volume and bridging, named dimensions, the
edge and thin-feature bands, and topology. :class:`CounterfactualEvidence`
publishes both sides and a set of preservation gates, and the recommendation
requires an output-facing gain with no preservation regression. A confusion
matrix alone never recommends anything.

Nothing in this module routes anything into the product. It reads
:mod:`tests.surface_oracle`, :mod:`scanner3d.tsdf` and
:mod:`scanner3d.tsdf_surface` and changes none of them; there is no pruning
code path in :mod:`scanner3d` and this module does not add one.

What this cannot say
--------------------
It is synthetic, and it inherits every limitation the surface oracle states
about itself. It is also bounded by a fact about the appliance rather than
about the arithmetic: the POPULATION here is "unsupported closed shells", and
unsupported is defined against the scene's ideal returns, which is evaluator
knowledge. An appliance cannot enumerate this population; it can only apply a
rule to every closed shell. The observable analogue -
:attr:`ObservableShell.nearest_measured_distance`, which is measured against
the RAW returns an appliance really has - is published for exactly that reason,
and the documentation says plainly that a deployable rule would have to be
evaluated over all closed shells rather than over this one.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import tests.surface_oracle as contract
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point, Rigid
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.mesh import edge_census
from scanner3d.tsdf import TsdfVolume
from tests.depth_completeness import Window, cloud_of, codes_of

SCHEMA_VERSION: Final = 1

#: What every observable item is derived from, in one sentence a reader can
#: check against the code below.
OBSERVABLE_SOURCE: Final = (
    "the case's raw measured frames, the poses the reconstruction was given, the camera "
    "intrinsics and window, the fused TSDF field and its accumulated weights, and the extracted "
    "topology. No analytic shape, no exact truth frame and no exact pose is read anywhere in "
    "tests.shell_discriminant.observe"
)

#: What the label is derived from, and who is allowed to read it.
LABEL_SOURCE: Final = (
    "tests.surface_oracle.analytic_distance over the scene's declared shapes, at the shell's own "
    "vertices and inside its own triangles, against the fixture's coverage tolerance. Evaluator "
    "only: tests.shell_discriminant.select accepts ObservableShell and nothing else"
)

#: The unit prose on every length here. The surface oracle's, unchanged.
DEPTH_UNIT: Final = contract.DEPTH_UNIT
COUNT_UNIT: Final = "shells"
FACE_UNIT: Final = "triangles"

#: The two halves of the declared population. A whole synthetic realization -
#: one member, one seed, all of its shells - belongs to exactly one of them.
TRAIN: Final = "train"
HOLDOUT: Final = "holdout"
SPLITS: Final[tuple[str, ...]] = (TRAIN, HOLDOUT)

#: The two analytic classes, named so a reader never has to infer polarity.
FALSE_SURFACE: Final = "false"
CORRECT_SURFACE: Final = "correct"
CLASSES: Final[tuple[str, ...]] = (FALSE_SURFACE, CORRECT_SURFACE)

#: The fields that belong to the LABEL and to nothing else. An observable
#: record carrying one of these, or a rule naming one, is a leak - and
#: :func:`check_observable` refuses it rather than trusting a reviewer to
#: notice. ``shell_id`` and ``faces`` deliberately appear on both records and
#: are not here: an identity and a triangle count are things an appliance can
#: see, and the reconciliation between the two namespaces is built on them.
LABEL_ONLY_FIELDS: Final[tuple[str, ...]] = (
    "false_surface",
    "label",
    "worst_analytic_distance",
    "accuracy_tolerance",
)


class DiscriminantUnusable(ScannerError):
    """The experiment cannot be run or scored as declared."""


class TruthLeak(ScannerError):
    """Something tried to give a candidate rule access to the answer."""


class RuleRefused(ScannerError):
    """A candidate rule is not one this module will ever evaluate."""


# --- what an appliance can see about one sealed shell ----------------------------------


#: How far past a raw return's own range a shell may sit before the ray is read
#: as CONTRADICTING it rather than confirming it, and how far in front before
#: it is read as OCCLUDED. Declared here, published in every record, and never
#: chosen per shell: a tolerance a case could pick after seeing its shells
#: would be a threshold wearing a tolerance's name.
RAY_TOLERANCE_VOXELS: Final = 2.0

#: The ceiling on the search for the nearest raw measured return. A shell
#: further than this from every measured point reports exactly this number and
#: says so through :attr:`ObservableShell.nearest_measured_capped`, rather than
#: reporting an unbounded search nobody paid for.
SUPPORT_SEARCH_CEILING: Final = 3.0

#: The names an observable item is listed under when it could not be derived.
FUSED_WEIGHT: Final = "fused_weight"
UNAVAILABLE_ITEMS: Final[tuple[str, ...]] = (FUSED_WEIGHT,)

#: The nearest-return lattice cell, in voxels of the arm's own cell. Declared
#: here rather than taken from the fixture's coverage tolerance, which is what
#: round one did: the tolerance is the EVALUATOR's bar, and a pure observable
#: derivation that read it would be reading the label's own parameter to lay
#: out its search. The lattice changes nothing a rule can see - the search is
#: exact within :data:`SUPPORT_SEARCH_CEILING` either way - and it keeps the
#: derivation a function of observable inputs alone.
SUPPORT_LATTICE_VOXELS: Final = 8.0

#: The only two backends this corpus's arms ever fuse on: the pure-Python
#: reference every declared grid cell runs, and the appliance backend the
#: derived shipped-policy cell runs instead. Closed to exactly these two
#: because a reconstruction is a function of the backend that fused it as much
#: as of its frames and poses - the same frames and poses fused again on the
#: other backend are a different reconstruction, not the same one measured
#: twice.
FUSION_BACKENDS: Final[tuple[str, ...]] = (contract.BACKEND, contract.APPLIANCE_BACKEND)

#: Attribute names that only an analytic object carries. :func:`check_inputs`
#: refuses an :class:`ObservableInputs` field that exposes one of them, so
#: handing the derivation a Scene, a ViewEvidence, a SurfaceScoring or a
#: Perturbation under any field name is a refusal rather than a leak nobody
#: happened to read. Plain values - text, bytes, numbers, booleans and the
#: tuples of them this experiment's geometry is made of - are not scanned,
#: because ``"appliance".split`` is a string method rather than a train/holdout
#: boundary and a guard that cannot tell the two apart guards nothing.
ANALYTIC_ATTRIBUTES: Final[tuple[str, ...]] = (
    "shapes",
    "truth",
    "truth_codes",
    "true_pose",
    "perturbation",
    "regions",
    "control_case",
    "case_id",
    "vertex_distances",
    "sample_distances",
    "samples",
    "label",
    "false_surface",
    "split",
    "family",
)


@dataclass(frozen=True, slots=True)
class ObservableInputs:
    """Everything the appliance holds when it exports, and NOTHING else.

    This type is the truth wall, moved one level below the rule. Round one put
    the wall at :func:`select` and then handed the derivation a whole
    :class:`~tests.surface_oracle.Scene` plus the ideal support and the
    analytic scoring in the same call, so "the observables are truth-free" was
    a property of how the function happened to be written. Here it is a
    property of what the function can reach: the derivation takes one of these
    and nothing else, and a Scene, the declared shapes, the ideal-return
    support, a truth frame, a true pose, a
    :class:`~tests.surface_oracle.SurfaceScoring`, the perturbation family, the
    split and every analytic label are simply not in it.

    Every field is something a shipped reconstruction demonstrably has: the raw
    frames it was given in order, the fused poses it actually used, its
    intrinsics, its scoring window and depth bounds, its depth scale, the
    surface it extracted, the parameters of the cell it fused into, the backend
    it fused on, and the field the surface came out of.
    """

    #: The raw measured frames, in the order they were integrated.
    raw_frames: tuple[bytes, ...]
    #: The poses the reconstruction ACTUALLY used, one per frame above. Not the
    #: exact poses, which exist only for the evaluator and are not here.
    fused_poses: tuple[Rigid, ...]
    intrinsics: IntrinsicsRecord
    window: Window
    depth_scale: float
    min_depth: float
    max_depth: float

    #: The extracted surface, exactly as the arm emitted it.
    points: tuple[Point, ...]
    faces: tuple[tuple[int, int, int], ...]

    #: The cell the surface was fused and extracted at.
    voxel_size: float
    truncation: float
    min_weight: float
    weight_policy: str
    volume_minimum: Point
    volume_maximum: Point

    #: The backend the arm actually fused on, one of :data:`FUSION_BACKENDS`.
    #: Published because the backend is part of what a reconstruction is a
    #: function of: two otherwise identical reconstructions run on different
    #: backends are two different reconstructions, not one reconstruction
    #: measured twice.
    fusion_backend: str

    #: The field that GENERATED this mesh, kept so the accumulated weight under
    #: a shell is read out of the volume the surface came from rather than out
    #: of a second fusion of the same frames. ``None`` when no field was kept,
    #: in which case every shell's weight is unavailable and no rule fires on
    #: it.
    field: TsdfVolume | None

    def __post_init__(self) -> None:
        check_inputs(self)


def check_inputs(inputs: ObservableInputs) -> ObservableInputs:
    """Refuse an inputs value that is not internally sound, or that carries truth.

    Two jobs. The first is ordinary soundness: one pose per frame, finite
    scalars, a window and a positive cell. The second is the wall: every field
    is checked for the attributes only an analytic object has, so a caller that
    passed a :class:`~tests.surface_oracle.Scene` as ``intrinsics``, or a
    :class:`~tests.surface_oracle.ViewEvidence` in ``fused_poses``, is refused
    here instead of quietly making every number downstream meaningless.
    """
    if type(inputs) is not ObservableInputs:
        raise TruthLeak(
            f"the observable derivation was handed a {type(inputs).__name__}. It takes "
            "ObservableInputs and nothing else, because that type is the list of things an "
            "appliance actually has"
        )
    if not inputs.raw_frames:
        raise DiscriminantUnusable("an observable derivation was given no raw frames")
    if len(inputs.raw_frames) != len(inputs.fused_poses):
        raise DiscriminantUnusable(
            f"{len(inputs.raw_frames)} raw frame(s) against {len(inputs.fused_poses)} fused "
            "pose(s); the poses a reconstruction used are one per frame it was given"
        )
    for name in ("depth_scale", "min_depth", "max_depth", "voxel_size", "truncation"):
        value = getattr(inputs, name)
        if not math.isfinite(value) or value <= 0.0:
            raise DiscriminantUnusable(f"an observable derivation was given {name}={value!r}")
    if inputs.min_depth >= inputs.max_depth:
        raise DiscriminantUnusable(
            f"a depth window of [{inputs.min_depth}, {inputs.max_depth}] accepts nothing"
        )
    if not math.isfinite(inputs.min_weight) or inputs.min_weight <= 0.0:
        raise DiscriminantUnusable(f"an extraction threshold of {inputs.min_weight!r}")
    if type(inputs.fusion_backend) is not str or inputs.fusion_backend not in FUSION_BACKENDS:
        raise DiscriminantUnusable(
            f"an observable derivation was given fusion_backend={inputs.fusion_backend!r}; "
            f"this oracle runs {list(FUSION_BACKENDS)}"
        )
    for axis in range(3):
        if inputs.volume_minimum[axis] >= inputs.volume_maximum[axis]:
            raise DiscriminantUnusable("an observable derivation was given an empty volume box")
    for face in inputs.faces:
        if len(face) != 3 or any(not 0 <= index < len(inputs.points) for index in face):
            raise DiscriminantUnusable(f"a face {face!r} does not index this mesh")
    for field_name in ObservableInputs.__slots__:
        for value in _scanned(getattr(inputs, field_name)):
            leaked = sorted(name for name in ANALYTIC_ATTRIBUTES if hasattr(value, name))
            if leaked:
                raise TruthLeak(
                    f"the observable input {field_name!r} reaches a {type(value).__name__} "
                    f"exposing {leaked}. Those belong to the evaluator's own objects, and an "
                    "inputs value that can reach one is not the appliance's evidence however it "
                    "is spelled"
                )
    return inputs


def _scanned(value: object) -> Iterator[object]:
    """The non-primitive objects reachable from one inputs field, tuples included."""
    if isinstance(value, (str, bytes, bool, int, float, type(None))):
        return
    if isinstance(value, tuple):
        for item in value:
            yield from _scanned(item)
        return
    yield value


def reconstruction_digest(inputs: ObservableInputs) -> str:
    """The identity of one whole reconstruction INPUT, and of nothing else.

    Round one identified a realization by its frame digests alone, which left
    the fused poses - the other half of what a reconstruction is a function of
    - outside the identity, so two members that shared frames and differed only
    in pose error looked like one realization and two that differed only in
    frames looked like two. This binds the ordered raw bytes, the poses
    actually used, the intrinsics, the window, the depth scale and bounds, and
    the volume and cell parameters the fusion ran under.

    The backend the fusion actually ran on is bound here too, for the same
    reason the poses are: the same frames and poses fused on a different
    backend are a different reconstruction, not the same one measured twice,
    and an identity that could not tell them apart would be quietly treating
    two different reconstruction problems as one.

    Deliberately inputs only. The extracted mesh is an OUTPUT of exactly these
    inputs, so hashing it too would make the digest unable to answer the
    question it exists for: whether two populations came from the same
    reconstruction problem.
    """
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "raw_frames": [
                [hashlib.sha256(frame).hexdigest(), len(frame)] for frame in inputs.raw_frames
            ],
            "fused_poses": [
                [[list(row) for row in pose.rotation], list(pose.translation)]
                for pose in inputs.fused_poses
            ],
            "intrinsics": sorted(
                (name, float(value)) for name, value in inputs.intrinsics.values.items()
            ),
            "window": [
                inputs.window.u0,
                inputs.window.v0,
                inputs.window.width,
                inputs.window.height,
            ],
            "depth_scale": inputs.depth_scale,
            "fusion_backend": inputs.fusion_backend,
            "depth_bounds": [inputs.min_depth, inputs.max_depth],
            "volume": [list(inputs.volume_minimum), list(inputs.volume_maximum)],
            "cell": [
                inputs.voxel_size,
                inputs.truncation,
                inputs.min_weight,
                inputs.weight_policy,
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ObservableShell:
    """One sealed shell, as the appliance could describe it. No truth anywhere.

    Every field is derived in :func:`observe` from :data:`OBSERVABLE_SOURCE`.
    The type is the wall: :func:`select` accepts exactly this class, so a
    candidate rule cannot be handed a record that also carries the answer, and
    the closed :data:`FEATURES` table means a rule cannot name a field that is
    not here either.
    """

    #: Stable across runs and unique in the population:
    #: ``<first 16 hex of the reconstruction-input digest>|<shell index>``. The
    #: shell index is the accepted oracle's own deterministic shell ordering.
    #:
    #: Deliberately NOT the member id, the perturbation family or the
    #: resolution name. Round one carried all three on this record, so an
    #: observable a rule was shown also told it which fixture family it came
    #: from; member, family, split and resolution are evaluator and report
    #: metadata now and live on :class:`ShellRecord` alone. What survives here
    #: is an identity derived from the reconstruction INPUTS, which an
    #: appliance has, plus the two cell parameters it also has.
    shell_id: str
    voxel_size: float
    shell_index: int
    #: The face indices this shell occupies in the extracted mesh. Pure
    #: topology of a mesh the appliance is holding, so it is observable - and
    #: it is what makes an independent recomputation of the selected component
    #: checkable rather than trusted.
    face_indices: tuple[int, ...]

    # -- size and extent -------------------------------------------------
    faces: int
    vertices: int
    area: float
    extent: Point
    diagonal: float
    centroid: Point

    # -- view geometry ---------------------------------------------------
    #: Views the case has at all.
    views_total: int
    #: Views whose frustum contains EVERY vertex of the shell, within the
    #: case's own depth range and scoring window.
    views_containing: int
    #: Views whose frustum contains at least one vertex.
    views_touching: int

    # -- what the raw rays say -------------------------------------------
    #: One probe per (shell vertex, view). The four counts below partition it.
    ray_probes: int
    #: A raw return within :attr:`ray_tolerance` of the shell along that ray.
    confirming_probes: int
    #: A raw return BEYOND the shell along that ray: the sensor saw through
    #: where this surface claims to be. This is the free-space contradiction.
    contradicting_probes: int
    #: A raw return in FRONT of the shell: something else was measured first,
    #: so this ray says nothing about the shell either way.
    occluded_probes: int
    #: The ray landed in the window and returned nothing at all.
    no_return_probes: int
    #: The ray missed the scoring window or fell outside the depth range.
    outside_window_probes: int
    ray_tolerance: float

    # -- the observed-support frontier -----------------------------------
    #: Raw measured returns the case produced, placed by the poses the
    #: reconstruction was given. The denominator of the frontier below.
    measured_returns: int
    #: Distance from the shell to the nearest RAW measured return, over its
    #: vertices. Not the oracle's support reference, which is the scene's
    #: ideal returns and is evaluator knowledge; this is the frontier an
    #: appliance can actually measure.
    nearest_measured_distance: float
    #: The same, for the shell's worst vertex.
    farthest_measured_distance: float
    #: True when the search hit :data:`SUPPORT_SEARCH_CEILING` and the two
    #: distances above are that ceiling rather than a located neighbour.
    nearest_measured_capped: bool

    # -- what the fused field says under the shell -----------------------
    #: Accumulated TSDF weight at the voxel each vertex falls in. ``None``
    #: when no field was kept for this arm, in which case ``fused_weight`` is
    #: listed in :attr:`unavailable`.
    weight_min: float | None
    weight_mean: float | None
    weight_max: float | None
    #: The extraction threshold the cell declares, for scale.
    min_weight: float

    #: Observable items this shell could not carry. Never substituted.
    unavailable: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.shell_id

    @property
    def kind(self) -> str:
        return "shell_discriminant_observable"

    @property
    def confirming_fraction(self) -> float:
        return self.confirming_probes / self.ray_probes if self.ray_probes else 0.0

    @property
    def contradicting_fraction(self) -> float:
        return self.contradicting_probes / self.ray_probes if self.ray_probes else 0.0

    @property
    def occluded_fraction(self) -> float:
        return self.occluded_probes / self.ray_probes if self.ray_probes else 0.0

    @property
    def no_return_fraction(self) -> float:
        return self.no_return_probes / self.ray_probes if self.ray_probes else 0.0

    @property
    def weight_mean_in_thresholds(self) -> float | None:
        """The mean weight in units of the cell's own extraction threshold."""
        if self.weight_mean is None or self.min_weight <= 0.0:
            return None
        return self.weight_mean / self.min_weight

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "shell_id": self.shell_id,
            "voxel_size": self.voxel_size,
            "shell_index": self.shell_index,
            "face_indices": list(self.face_indices),
            "faces": self.faces,
            "vertices": self.vertices,
            "area": self.area,
            "extent": list(self.extent),
            "diagonal": self.diagonal,
            "centroid": list(self.centroid),
            "views_total": self.views_total,
            "views_containing": self.views_containing,
            "views_touching": self.views_touching,
            "ray_probes": self.ray_probes,
            "confirming_probes": self.confirming_probes,
            "contradicting_probes": self.contradicting_probes,
            "occluded_probes": self.occluded_probes,
            "no_return_probes": self.no_return_probes,
            "outside_window_probes": self.outside_window_probes,
            "confirming_fraction": self.confirming_fraction,
            "contradicting_fraction": self.contradicting_fraction,
            "occluded_fraction": self.occluded_fraction,
            "no_return_fraction": self.no_return_fraction,
            "ray_tolerance": self.ray_tolerance,
            "measured_returns": self.measured_returns,
            "nearest_measured_distance": self.nearest_measured_distance,
            "farthest_measured_distance": self.farthest_measured_distance,
            "nearest_measured_capped": self.nearest_measured_capped,
            "support_search_ceiling": SUPPORT_SEARCH_CEILING,
            "weight_min": self.weight_min,
            "weight_mean": self.weight_mean,
            "weight_max": self.weight_max,
            "weight_mean_in_thresholds": self.weight_mean_in_thresholds,
            "min_weight": self.min_weight,
            "unavailable": list(self.unavailable),
            "observable_source": OBSERVABLE_SOURCE,
            "unit": DEPTH_UNIT,
        }


@dataclass(frozen=True, slots=True)
class ShellTruth:
    """The analytic label of one shell. Read by the evaluator and by nobody else.

    ``false_surface`` is exactly the accepted oracle's own split: the worst
    analytic distance the shell reaches - over its vertices AND inside its
    triangles - against the fixture's coverage tolerance. It is not recomputed
    with a second rule here, because a corpus that graded itself on a slightly
    different bar than the one it published would be measuring two things.
    """

    shell_id: str
    #: True when this shell is off the object: false surface, and the whole of
    #: what a pruning rule could honestly claim to remove.
    false_surface: bool
    worst_analytic_distance: float
    accuracy_tolerance: float
    faces: int

    def __post_init__(self) -> None:
        for name in ("worst_analytic_distance", "accuracy_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise DiscriminantUnusable(f"{self.shell_id} labels {name}={value!r}")
        if self.accuracy_tolerance <= 0.0:
            raise DiscriminantUnusable(
                f"{self.shell_id} was graded against a tolerance of {self.accuracy_tolerance!r}"
            )
        if self.worst_analytic_distance < 0.0:
            raise DiscriminantUnusable(f"{self.shell_id} claims a negative worst analytic distance")
        if self.faces <= 0:
            raise DiscriminantUnusable(f"{self.shell_id} labels {self.faces} face(s)")
        if self.false_surface != (self.worst_analytic_distance > self.accuracy_tolerance):
            raise DiscriminantUnusable(
                f"{self.shell_id} is labelled {self.label!r} with a worst analytic distance of "
                f"{self.worst_analytic_distance!r} against a tolerance of "
                f"{self.accuracy_tolerance!r}. The boolean IS that comparison, and a record in "
                "which the two disagree is a swapped label rather than a measurement"
            )

    @property
    def label(self) -> str:
        return FALSE_SURFACE if self.false_surface else CORRECT_SURFACE

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "shell_discriminant_label",
            "shell_id": self.shell_id,
            "label": self.label,
            "false_surface": self.false_surface,
            "worst_analytic_distance": self.worst_analytic_distance,
            "accuracy_tolerance": self.accuracy_tolerance,
            "faces": self.faces,
            "label_source": LABEL_SOURCE,
        }


@dataclass(frozen=True, slots=True)
class ShellRecord:
    """One shell's two namespaces, kept together for the evaluator only.

    Deliberately NOT what :func:`select` accepts. The evaluator holds these,
    hands ``record.observable`` to a rule, and reads ``record.truth`` after the
    rule has spoken.
    """

    observable: ObservableShell
    truth: ShellTruth
    #: Evaluator and report metadata. None of it is on the observable record
    #: and none of it is reachable from a rule.
    member_id: str
    family: str
    cell_id: str
    #: Which half of the declared population this shell's member belongs to.
    split: str
    #: The identity of the whole reconstruction INPUT this shell came out of.
    #: Two records that share it came from the same reconstruction problem,
    #: whatever their members are called.
    reconstruction_digest: str
    #: The face indices this shell occupies in its arm's mesh, so a
    #: counterfactual removes exactly the shell the rule selected. Bound to the
    #: observable record's own indices below.
    face_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.observable) is not ObservableShell:
            raise TruthLeak(
                f"a record's observable half is a {type(self.observable).__name__}, not exactly "
                "an ObservableShell"
            )
        if type(self.truth) is not ShellTruth:
            raise DiscriminantUnusable(
                f"a record's label half is a {type(self.truth).__name__}, not exactly a ShellTruth"
            )
        if self.observable.shell_id != self.truth.shell_id:
            raise DiscriminantUnusable(
                f"observable {self.observable.shell_id!r} was paired with label "
                f"{self.truth.shell_id!r}; a record whose two halves describe different shells "
                "cannot score either of them"
            )
        if self.split not in SPLITS:
            raise DiscriminantUnusable(f"{self.observable.shell_id} is in split {self.split!r}")
        if self.family not in contract.PERTURBATIONS:
            raise DiscriminantUnusable(
                f"{self.observable.shell_id} declares family {self.family!r}; this experiment "
                f"reports by {list(contract.PERTURBATIONS)}"
            )
        if not self.member_id or not self.cell_id:
            raise DiscriminantUnusable(
                f"{self.observable.shell_id} carries member {self.member_id!r} and resolution "
                f"{self.cell_id!r}"
            )
        if len(self.reconstruction_digest) != 64 or any(
            character not in "0123456789abcdef" for character in self.reconstruction_digest
        ):
            raise DiscriminantUnusable(
                f"{self.observable.shell_id} carries reconstruction identity "
                f"{self.reconstruction_digest!r}, which is not a SHA-256"
            )
        if self.observable.faces != self.truth.faces:
            raise DiscriminantUnusable(
                f"{self.observable.shell_id} is observed with {self.observable.faces} face(s) "
                f"and labelled with {self.truth.faces}. The label grades the shell the rule was "
                "shown, or it grades something else"
            )
        if len(self.face_indices) != self.observable.faces:
            raise DiscriminantUnusable(
                f"{self.observable.shell_id} says it has {self.observable.faces} face(s) and "
                f"carries {len(self.face_indices)}"
            )
        if len(set(self.face_indices)) != len(self.face_indices):
            raise DiscriminantUnusable(
                f"{self.observable.shell_id} names the same face index twice, so its faces are "
                "not a set of triangles a removal could take out once"
            )
        if any(index < 0 for index in self.face_indices):
            raise DiscriminantUnusable(f"{self.observable.shell_id} carries a negative face index")
        if tuple(self.face_indices) != tuple(self.observable.face_indices):
            raise DiscriminantUnusable(
                f"{self.observable.shell_id}: the record's face set is not the observable "
                "record's own. The two halves must point at exactly one component of one mesh"
            )

    @property
    def shell_id(self) -> str:
        return self.observable.shell_id

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "shell_discriminant_record",
            "member_id": self.member_id,
            "family": self.family,
            "cell_id": self.cell_id,
            "split": self.split,
            "reconstruction_digest": self.reconstruction_digest,
            "face_indices": list(self.face_indices),
            "observable": self.observable.as_dict(),
            "label": self.truth.as_dict(),
        }


# --- deriving the observables, from raw evidence and nothing else ----------------------


def _shell_members(faces: Sequence[tuple[int, int, int]]) -> list[list[int]]:
    """The accepted oracle's own shell grouping, reused rather than re-derived."""
    return contract._shells(faces)


def _extent_of(points: Sequence[Point]) -> tuple[Point, Point, Point]:
    """``(minimum, maximum, extent)`` of a non-empty point set."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    minimum = (min(xs), min(ys), min(zs))
    maximum = (max(xs), max(ys), max(zs))
    return (
        minimum,
        maximum,
        (
            maximum[0] - minimum[0],
            maximum[1] - minimum[1],
            maximum[2] - minimum[2],
        ),
    )


class _NearestSearch:
    """Nearest-neighbour distance to a fixed point set, bounded and exact.

    A uniform lattice at :data:`SUPPORT_SEARCH_CEILING` divided into rings, so
    a query stops as soon as the ring it is expanding into cannot hold anything
    closer than what it already found. Exact within the ceiling and capped at
    it, rather than an approximation nobody can state the error of.
    """

    def __init__(self, points: Sequence[Point], *, cell: float) -> None:
        if cell <= 0.0:
            raise DiscriminantUnusable(f"a nearest-neighbour lattice of {cell!r} has no cells")
        self._cell = cell
        self._points = points
        self._grid: dict[tuple[int, int, int], list[int]] = {}
        for index, point in enumerate(points):
            self._grid.setdefault(self._key(point), []).append(index)
        self._max_ring = max(1, math.ceil(SUPPORT_SEARCH_CEILING / cell))

    def _key(self, point: Point) -> tuple[int, int, int]:
        return (
            math.floor(point[0] / self._cell),
            math.floor(point[1] / self._cell),
            math.floor(point[2] / self._cell),
        )

    def distance(self, query: Point) -> tuple[float, bool]:
        """``(distance, capped)`` to the nearest stored point."""
        if not self._points:
            return SUPPORT_SEARCH_CEILING, True
        base = self._key(query)
        best = SUPPORT_SEARCH_CEILING
        found = False
        for ring in range(self._max_ring + 1):
            if found and (ring - 1) * self._cell > best:
                break
            for di in range(-ring, ring + 1):
                for dj in range(-ring, ring + 1):
                    for dk in range(-ring, ring + 1):
                        if max(abs(di), abs(dj), abs(dk)) != ring:
                            continue
                        for index in self._grid.get((base[0] + di, base[1] + dj, base[2] + dk), ()):
                            target = self._points[index]
                            dx = query[0] - target[0]
                            dy = query[1] - target[1]
                            dz = query[2] - target[2]
                            candidate = math.sqrt(dx * dx + dy * dy + dz * dz)
                            if candidate < best:
                                best = candidate
                                found = True
        return (best, not found)


def _weight_at(volume: TsdfVolume, point: Point) -> float | None:
    """The accumulated weight of the voxel ``point`` falls in, or ``None`` outside it."""
    config = volume.config
    bounds = config.bounds
    size = config.voxel_size
    nx, ny, nz = config.shape
    i = math.floor((point[0] - bounds.min_x) / size)
    j = math.floor((point[1] - bounds.min_y) / size)
    k = math.floor((point[2] - bounds.min_z) / size)
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        return None
    return float(volume.weights[(i * ny + j) * nz + k])


def _ray_evidence(
    inputs: ObservableInputs,
    codes_by_view: Sequence[Sequence[int]],
    vertices: Sequence[Point],
    *,
    tolerance: float,
) -> dict[str, int]:
    """What the RAW frames say along each view's ray through each shell vertex.

    The raw frames and the FUSED poses, which is exactly what the
    reconstruction was given - and here that is not a discipline, it is all
    there is: :class:`ObservableInputs` carries no truth frame and no true
    pose, so this function could not read one if it wanted to.
    """
    values = inputs.intrinsics.values
    fx = float(values["fx"])
    fy = float(values["fy"])
    cx = float(values["cx"])
    cy = float(values["cy"])
    window = inputs.window
    counts = {
        "probes": 0,
        "confirming": 0,
        "contradicting": 0,
        "occluded": 0,
        "no_return": 0,
        "outside": 0,
        "containing": 0,
        "touching": 0,
    }
    for pose, codes in zip(inputs.fused_poses, codes_by_view, strict=True):
        inverse = pose.inverse()
        inside = 0
        for vertex in vertices:
            counts["probes"] += 1
            x, y, z = inverse.apply(vertex)
            if z < inputs.min_depth or z > inputs.max_depth or z <= 0.0:
                counts["outside"] += 1
                continue
            u = math.floor(x * fx / z + cx + 0.5)
            v = math.floor(y * fy / z + cy + 0.5)
            index = window.index_of(u, v)
            if index is None:
                counts["outside"] += 1
                continue
            inside += 1
            code = codes[index]
            if code == contract.NO_RETURN:
                counts["no_return"] += 1
                continue
            measured = code * inputs.depth_scale
            if abs(measured - z) <= tolerance:
                counts["confirming"] += 1
            elif measured > z:
                counts["contradicting"] += 1
            else:
                counts["occluded"] += 1
        if inside:
            counts["touching"] += 1
            if inside == len(vertices):
                counts["containing"] += 1
    return counts


def measured_returns(inputs: ObservableInputs) -> tuple[Point, ...]:
    """Every raw return the case produced, placed by the poses actually used.

    The appliance's own dots. Clipped to the fused volume box and to nothing
    else, exactly as :func:`tests.surface_oracle.measured_points` does it, but
    reached from the observable inputs rather than from a Scene.
    """
    minimum = inputs.volume_minimum
    maximum = inputs.volume_maximum
    out: list[Point] = []
    for frame, pose in zip(inputs.raw_frames, inputs.fused_poses, strict=True):
        for point in cloud_of(
            codes_of(frame, inputs.window),
            inputs.window,
            intrinsics=inputs.intrinsics,
            depth_scale=inputs.depth_scale,
            pose=pose,
        ):
            if all(minimum[axis] <= point[axis] <= maximum[axis] for axis in range(3)):
                out.append(point)
    return tuple(out)


def derive_observables(inputs: ObservableInputs) -> tuple[ObservableShell, ...]:
    """Every CLOSED shell of the extracted mesh, described observably. Pure.

    The whole truth-free half of the experiment, and the only thing that
    produces an :class:`ObservableShell`. It takes exactly one
    :class:`ObservableInputs` and returns one record per closed shell - every
    closed shell, not the evaluator's unsupported subset, because a deployable
    rule sees the whole mesh and would have to be applied to all of it.

    Nothing here filters by support, reads an analytic distance or knows what a
    perturbation family is. The evaluator does that separately in
    :func:`evaluate`, against the same shells, by shell index.
    """
    check_inputs(inputs)
    digest = reconstruction_digest(inputs)
    faces = inputs.faces
    points = inputs.points
    codes_by_view = [codes_of(frame, inputs.window) for frame in inputs.raw_frames]
    measured = measured_returns(inputs)
    lattice = max(inputs.voxel_size * SUPPORT_LATTICE_VOXELS, SUPPORT_SEARCH_CEILING / 256.0)
    search = _NearestSearch(measured, cell=lattice)
    ray_tolerance = RAY_TOLERANCE_VOXELS * inputs.voxel_size
    volume = inputs.field

    out: list[ObservableShell] = []
    for shell_index, members in enumerate(_shell_members(faces)):
        shell_faces = [faces[index] for index in members]
        if not edge_census(shell_faces).closed:
            continue
        corners = sorted({index for face in shell_faces for index in face})
        vertices = [points[index] for index in corners]
        _, _, extent = _extent_of(vertices)
        area = math.fsum(
            contract._triangle_area(points[a], points[b], points[c]) for a, b, c in shell_faces
        )
        rays = _ray_evidence(inputs, codes_by_view, vertices, tolerance=ray_tolerance)
        distances = [search.distance(vertex) for vertex in vertices]
        nearest = min(value for value, _ in distances)
        farthest = max(value for value, _ in distances)
        capped = all(flag for _, flag in distances)
        weights = [] if volume is None else [_weight_at(volume, vertex) for vertex in vertices]
        present = [value for value in weights if value is not None]
        shell = ObservableShell(
            shell_id=f"{digest[:16]}|{shell_index}",
            voxel_size=inputs.voxel_size,
            shell_index=shell_index,
            face_indices=tuple(members),
            faces=len(members),
            vertices=len(corners),
            area=area,
            extent=extent,
            diagonal=math.sqrt(extent[0] ** 2 + extent[1] ** 2 + extent[2] ** 2),
            centroid=(
                math.fsum(vertex[0] for vertex in vertices) / len(vertices),
                math.fsum(vertex[1] for vertex in vertices) / len(vertices),
                math.fsum(vertex[2] for vertex in vertices) / len(vertices),
            ),
            views_total=len(inputs.raw_frames),
            views_containing=rays["containing"],
            views_touching=rays["touching"],
            ray_probes=rays["probes"],
            confirming_probes=rays["confirming"],
            contradicting_probes=rays["contradicting"],
            occluded_probes=rays["occluded"],
            no_return_probes=rays["no_return"],
            outside_window_probes=rays["outside"],
            ray_tolerance=ray_tolerance,
            measured_returns=len(measured),
            nearest_measured_distance=nearest,
            farthest_measured_distance=farthest,
            nearest_measured_capped=capped,
            weight_min=min(present) if present else None,
            weight_mean=(math.fsum(present) / len(present)) if present else None,
            weight_max=max(present) if present else None,
            min_weight=inputs.min_weight,
            # Per shell, not per population: a field can exist and still hold
            # no voxel this shell's vertices land in, and that shell's weight
            # is unavailable however available the field is.
            unavailable=() if present else (FUSED_WEIGHT,),
        )
        check_observable(shell)
        check_observable_is_consistent(shell)
        out.append(shell)
    return tuple(out)


def observable_inputs(
    scene: contract.Scene,
    result: contract.ArmResult,
    *,
    cell: contract.GridCell,
    volume: TsdfVolume | None,
    backend: str,
) -> ObservableInputs:
    """Take exactly the appliance's half of a scene and leave the rest behind.

    The one place the evaluator's world and the observable world touch, and it
    is a projection rather than a handover: what comes out carries the raw
    frames, the fused poses, the intrinsics, the window, the depth scale and
    bounds, the extracted mesh, the cell and the backend it was fused on, and
    it structurally cannot carry the shapes, the truth frames, the true poses,
    the perturbation or the regions, because :class:`ObservableInputs` has
    nowhere to put them.

    ``backend`` is required rather than defaulted, because a caller that kept
    no field (``volume=None``) still fused on a specific backend, and the
    reconstruction identity needs to know which one.
    """
    return ObservableInputs(
        raw_frames=tuple(view.raw for view in scene.views),
        fused_poses=tuple(view.fused_pose for view in scene.views),
        intrinsics=scene.intrinsics,
        window=scene.window,
        depth_scale=scene.depth_scale,
        min_depth=scene.volume.min_depth,
        max_depth=scene.volume.max_depth,
        points=tuple(result.points),
        faces=tuple(result.faces),
        voxel_size=cell.voxel_size,
        truncation=cell.truncation,
        min_weight=cell.min_weight,
        weight_policy=cell.weight_policy,
        volume_minimum=scene.volume.minimum,
        volume_maximum=scene.volume.maximum,
        fusion_backend=backend,
        field=volume,
    )


@dataclass(frozen=True, slots=True)
class Population:
    """One (member, resolution): every closed shell, and the labelled subset.

    :attr:`observables` is what a deployable rule would be applied to - every
    closed shell of the mesh. :attr:`records` is the evaluator's population:
    the closed shells the scene's ideal returns do not reach, each carrying the
    analytic label. The two are kept side by side on purpose, because the
    difference between them is precisely the prefilter an appliance cannot
    perform.
    """

    member_id: str
    family: str
    split: str
    cell_id: str
    voxel_size: float
    reconstruction_digest: str
    faces: int
    shells: int
    closed_shells: int
    #: Every closed shell, observable only. The counterfactual's population.
    observables: tuple[ObservableShell, ...]
    #: The unsupported closed shells, labelled. The decision matrix's population.
    records: tuple[ShellRecord, ...]
    #: Observable items no shell of this population could carry.
    unavailable: tuple[str, ...]
    #: What deriving this population cost, in units that are the same on every
    #: machine. Wall-clock is deliberately absent from the artifact: a report
    #: that has to be byte-identical between two runs cannot carry a number
    #: that differs between them.
    work_units: int
    triangle_samples: int

    @property
    def observable_only(self) -> tuple[ObservableShell, ...]:
        return tuple(record.observable for record in self.records)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "shell_discriminant_population",
            "member_id": self.member_id,
            "family": self.family,
            "split": self.split,
            "cell_id": self.cell_id,
            "voxel_size": self.voxel_size,
            "reconstruction_digest": self.reconstruction_digest,
            "faces": self.faces,
            "shells": self.shells,
            "closed_shells": self.closed_shells,
            "unsupported_closed_shells": len(self.records),
            "supported_closed_shells": len(self.observables) - len(self.records),
            "false_shells": sum(1 for r in self.records if r.truth.false_surface),
            "correct_shells": sum(1 for r in self.records if not r.truth.false_surface),
            "false_faces": sum(r.truth.faces for r in self.records if r.truth.false_surface),
            "correct_faces": sum(r.truth.faces for r in self.records if not r.truth.false_surface),
            "unavailable": list(self.unavailable),
            "work_units": self.work_units,
            "triangle_samples": self.triangle_samples,
            # Every closed shell, observably described. The strict reader needs
            # them: a counterfactual applies the selected rule to ALL closed
            # shells rather than to the evaluator's unsupported subset, so a
            # report that published only the labelled ones could not be audited
            # on the thing the recommendation actually rests on.
            "closed_shells_observable": [shell.as_dict() for shell in self.observables],
            "records": [record.as_dict() for record in self.records],
        }


def evaluate(
    scene: contract.Scene,
    result: contract.ArmResult,
    observables: Sequence[ObservableShell],
    inputs: ObservableInputs,
    *,
    support: Sequence[Point],
    member_id: str,
    family: str,
    split: str,
    scoring: contract.SurfaceScoring | None = None,
) -> Population:
    """Attach the analytic label to the observable shells. EVALUATOR ONLY.

    Strictly the second half. :func:`derive_observables` has already produced
    every closed shell from :class:`ObservableInputs` alone; this function uses
    the scene's ideal-return ``support`` to filter that list down to the
    unsupported ones and :func:`tests.surface_oracle.score_surface` to grade
    each of them. It never edits an observable record and never hands one of
    its own inputs back into the derivation.
    """
    if split not in SPLITS:
        raise DiscriminantUnusable(f"{member_id} declares split {split!r}, not one of {SPLITS}")
    if family not in contract.PERTURBATIONS:
        raise DiscriminantUnusable(f"{member_id} declares family {family!r}")
    faces = result.faces
    points = result.points
    tolerance = scene.regions.coverage_tolerance
    scoring = contract.score_surface(scene, result) if scoring is None else scoring
    supported = contract._covered_mask(points, support, tolerance)
    face_distance = [0.0] * len(faces)
    for (face_index, _), distance in zip(scoring.samples, scoring.sample_distances, strict=True):
        if distance > face_distance[face_index]:
            face_distance[face_index] = distance

    digest = reconstruction_digest(inputs)
    cell_id = str(result.cell_id)
    all_shells = _shell_members(faces)
    records: list[ShellRecord] = []
    for shell in observables:
        members = all_shells[shell.shell_index]
        if tuple(members) != tuple(shell.face_indices):
            raise DiscriminantUnusable(
                f"{shell.shell_id} carries a face set that is not component "
                f"{shell.shell_index} of this mesh"
            )
        corners = sorted({index for face_index in members for index in faces[face_index]})
        if any(supported[index] for index in corners):
            continue
        worst = max(
            max(scoring.vertex_distances[index] for index in corners),
            max(face_distance[index] for index in members),
        )
        records.append(
            ShellRecord(
                observable=shell,
                truth=ShellTruth(
                    shell_id=shell.shell_id,
                    false_surface=worst > tolerance,
                    worst_analytic_distance=worst,
                    accuracy_tolerance=tolerance,
                    faces=len(members),
                ),
                member_id=member_id,
                family=family,
                cell_id=cell_id,
                split=split,
                reconstruction_digest=digest,
                face_indices=tuple(members),
            )
        )
    return Population(
        member_id=member_id,
        family=family,
        split=split,
        cell_id=cell_id,
        voxel_size=inputs.voxel_size,
        reconstruction_digest=digest,
        faces=len(faces),
        shells=len(all_shells),
        closed_shells=len(observables),
        observables=tuple(observables),
        records=tuple(records),
        unavailable=() if inputs.field is not None else (FUSED_WEIGHT,),
        work_units=(result.field_evidence.work_units if result.field_evidence is not None else 0),
        triangle_samples=len(scoring.samples),
    )


def observe(
    scene: contract.Scene,
    result: contract.ArmResult,
    *,
    support: Sequence[Point],
    member_id: str,
    family: str,
    split: str,
    volume: TsdfVolume | None,
    backend: str | None = None,
    cell: contract.GridCell | None = None,
    scoring: contract.SurfaceScoring | None = None,
) -> Population:
    """Derive, then label. The two halves in the only order they may run in.

    A convenience over :func:`observable_inputs`, :func:`derive_observables`
    and :func:`evaluate` that keeps the wall visible: the observables are
    produced from the projection FIRST, with no access to ``support`` or
    ``scoring``, and only then does the evaluator grade them.

    ``backend`` defaults to ``None``, in which case it is resolved the same way
    the corpus itself resolves it - :func:`~tests.surface_oracle.backend_for`
    on the result's own cell id - rather than assumed to be the pure-Python
    reference, which is wrong for the one cell that is not.
    """
    resolved_backend = (
        contract.backend_for(scene, str(result.cell_id)) if backend is None else backend
    )
    inputs = observable_inputs(
        scene,
        result,
        cell=scene.cell(str(result.cell_id)) if cell is None else cell,
        volume=volume,
        backend=resolved_backend,
    )
    observables = derive_observables(inputs)
    return evaluate(
        scene,
        result,
        observables,
        inputs,
        support=support,
        member_id=member_id,
        family=family,
        split=split,
        scoring=scoring,
    )


# --- the wall: what a rule may read, and how it is stopped reading anything else -------


#: Every observable a candidate rule may threshold, and the accessor that
#: produces it. Closed and ordered: a rule naming anything not in this table is
#: refused at construction, which is why "the rule read the answer" is not a
#: thing a reviewer has to check by eye. Each accessor returns ``None`` when
#: the item is unavailable for that shell, and :meth:`Rule.holds` treats
#: ``None`` as "this rule does not fire" rather than as a small number.
FEATURES: Final[Mapping[str, Callable[[ObservableShell], float | None]]] = {
    "faces": lambda shell: float(shell.faces),
    "area": lambda shell: shell.area,
    "diagonal": lambda shell: shell.diagonal,
    "views_containing": lambda shell: float(shell.views_containing),
    "views_touching": lambda shell: float(shell.views_touching),
    "confirming_fraction": lambda shell: shell.confirming_fraction,
    "contradicting_fraction": lambda shell: shell.contradicting_fraction,
    "occluded_fraction": lambda shell: shell.occluded_fraction,
    "no_return_fraction": lambda shell: shell.no_return_fraction,
    "nearest_measured_distance": lambda shell: shell.nearest_measured_distance,
    "weight_mean_in_thresholds": lambda shell: shell.weight_mean_in_thresholds,
}

#: The two comparisons a test may make. No equality, because a float threshold
#: nobody can hit exactly would be a test that never fires.
COMPARISONS: Final[tuple[str, ...]] = ("<=", ">=")

#: The candidate families. ``single`` is one thresholded observable; ``pair``
#: is the conjunction of two, which is the smallest family that can express
#: "small AND contradicted" without becoming an unbounded search.
SINGLE: Final = "single"
PAIR: Final = "pair"
FAMILIES: Final[tuple[str, ...]] = (SINGLE, PAIR)


@dataclass(frozen=True, slots=True)
class FeatureTest:
    """One thresholded observable. The whole vocabulary of a candidate rule."""

    feature: str
    comparison: str
    threshold: float

    def __post_init__(self) -> None:
        if self.feature not in FEATURES:
            raise RuleRefused(
                f"{self.feature!r} is not an observable this experiment publishes. A rule may "
                f"read {sorted(FEATURES)} and nothing else, which is what makes a truth-fed "
                "rule unrepresentable rather than merely discouraged"
            )
        if self.comparison not in COMPARISONS:
            raise RuleRefused(f"{self.comparison!r} is not one of {list(COMPARISONS)}")
        if not math.isfinite(self.threshold):
            raise RuleRefused(f"{self.feature} was given a threshold of {self.threshold!r}")

    def holds(self, shell: ObservableShell) -> bool:
        value = FEATURES[self.feature](shell)
        if value is None:
            return False
        return value <= self.threshold if self.comparison == "<=" else value >= self.threshold

    @property
    def text(self) -> str:
        return f"{self.feature} {self.comparison} {self.threshold:g}"

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "comparison": self.comparison,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class Rule:
    """One candidate: a conjunction of one or two feature tests, and its identity."""

    rule_id: str
    tests: tuple[FeatureTest, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.tests) <= 2:
            raise RuleRefused(
                f"{self.rule_id} carries {len(self.tests)} test(s); this experiment declared "
                "single and pair families and evaluates no other shape"
            )
        if len({test.feature for test in self.tests}) != len(self.tests):
            raise RuleRefused(
                f"{self.rule_id} tests the same observable twice, which is an interval "
                "expressed as a conjunction and is not one of the declared families"
            )

    @property
    def family(self) -> str:
        return SINGLE if len(self.tests) == 1 else PAIR

    @property
    def text(self) -> str:
        return " and ".join(test.text for test in self.tests)

    def holds(self, shell: ObservableShell) -> bool:
        return all(test.holds(shell) for test in self.tests)

    def as_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "family": self.family,
            "text": self.text,
            "tests": [test.as_dict() for test in self.tests],
        }


def check_observable(shell: object) -> ObservableShell:
    """Refuse anything that is not exactly an observable record.

    Exact type identity rather than :func:`isinstance`, and that is the point.
    A subclass carrying a truth field would satisfy ``isinstance`` and would be
    exactly the leak this experiment exists to make impossible, so a subclass
    is refused too. The check also refuses a record whose field names have
    drifted into the label's namespace, so a future edit that added
    ``false_surface`` to :class:`ObservableShell` fails here rather than
    quietly making every result meaningless.
    """
    if type(shell) is not ObservableShell:
        raise TruthLeak(
            f"a candidate rule was offered a {type(shell).__name__}. It is shown "
            "ObservableShell and nothing else - not a subclass, not a ShellRecord, not a "
            "mapping - because the whole result depends on the rule never seeing the label"
        )
    leaked = sorted(set(ObservableShell.__slots__) & set(LABEL_ONLY_FIELDS))
    if leaked:
        raise TruthLeak(
            f"the observable record now carries {leaked}, which is the label's own namespace. "
            "The two namespaces are disjoint by design and a shared field is a leak whatever it "
            "happens to be called"
        )
    return shell


def check_observable_is_consistent(shell: ObservableShell) -> ObservableShell:
    """Refuse an observable record whose own numbers cannot have come from a mesh.

    The forged-feature control, and it is a real one rather than a formality:
    the whole experiment rests on the observables being derived from raw
    evidence, and a record whose ray probes do not partition, whose containing
    views outnumber its touching views, or whose shell has no area at all is a
    record somebody wrote rather than measured. Every field checked here is
    checked against another field of the same record, so the check needs no
    access to the scene and cannot itself become a way to smuggle one in.
    """
    parts = (
        shell.confirming_probes
        + shell.contradicting_probes
        + shell.occluded_probes
        + shell.no_return_probes
        + shell.outside_window_probes
    )
    if parts != shell.ray_probes:
        raise DiscriminantUnusable(
            f"{shell.shell_id}: {parts} classified ray probe(s) against {shell.ray_probes} "
            "taken. The four outcomes and the misses partition the probes, or the fractions a "
            "rule reads are fractions of nothing"
        )
    if shell.ray_probes != shell.vertices * shell.views_total:
        raise DiscriminantUnusable(
            f"{shell.shell_id} probes {shell.ray_probes} ray(s) for {shell.vertices} vertices "
            f"across {shell.views_total} view(s); every vertex is probed in every view"
        )
    if not 0 <= shell.views_containing <= shell.views_touching <= shell.views_total:
        raise DiscriminantUnusable(
            f"{shell.shell_id} says {shell.views_containing} view(s) contain it, "
            f"{shell.views_touching} touch it and it has {shell.views_total}"
        )
    if shell.faces <= 0 or shell.vertices < 3:
        raise DiscriminantUnusable(
            f"{shell.shell_id} claims {shell.faces} face(s) on {shell.vertices} vertices, which "
            "is not a sealed shell"
        )
    if shell.area <= 0.0 or shell.diagonal < 0.0:
        raise DiscriminantUnusable(
            f"{shell.shell_id} claims area {shell.area!r} and diagonal {shell.diagonal!r}"
        )
    if shell.nearest_measured_distance > shell.farthest_measured_distance:
        raise DiscriminantUnusable(
            f"{shell.shell_id} is nearer to the measured returns at its worst vertex than at its "
            "best one"
        )
    if FUSED_WEIGHT in shell.unavailable and shell.weight_mean is not None:
        raise DiscriminantUnusable(
            f"{shell.shell_id} lists the fused weight as unavailable and carries one anyway"
        )
    if shell.weight_mean is None and FUSED_WEIGHT not in shell.unavailable:
        raise DiscriminantUnusable(
            f"{shell.shell_id} carries no fused weight and does not list it as unavailable. "
            "A missing item is unavailable, which no rule fires on; an unlisted missing item is "
            "a hole a reader would have to guess about"
        )
    for name in shell.unavailable:
        if name not in UNAVAILABLE_ITEMS:
            raise DiscriminantUnusable(
                f"{shell.shell_id} lists {name!r} as unavailable; this experiment names "
                f"{list(UNAVAILABLE_ITEMS)}"
            )
    return shell


#: Rule invocations, counted per named phase. The producer is required to make
#: exactly ONE bounded pass - every declared candidate against every declared
#: unsupported observable, once - and round one instead applied all 195 rules to
#: eleven overlapping confusion slices and then again inside the selector, for
#: 89,505 calls published as 16,380. Counting cannot be forgotten here because
#: :func:`select` is the only way a rule is ever applied and it is what
#: increments the ledger, so every summary, gate, renderer or reader that
#: reached for a rule again would show up as a call in some phase.
_RULE_CALLS: dict[str, int] = {}
_PHASE: list[str] = ["unaccounted"]

#: The producer's one permitted pass. Its count must be exactly
#: ``len(RULE_GRID) * len(records)``.
PRODUCER_PHASE: Final = "producer_decision_matrix"
#: The counterfactual's application of the SELECTED rule to every closed shell.
#: Separate because it is deliberately a different population - all closed
#: shells rather than the evaluator's unsupported subset - and hiding it inside
#: the producer's number would understate one and overstate the other.
COUNTERFACTUAL_PHASE: Final = "counterfactual"
#: A standalone strict reader rebuilding its own audit matrix from a published
#: report. Accounted and published truthfully, never as producer search.
READER_PHASE: Final = "reader_audit"


@contextmanager
def accounting(phase: str) -> Iterator[None]:
    """Count every rule invocation made inside this block against ``phase``."""
    _PHASE.append(phase)
    try:
        yield
    finally:
        _PHASE.pop()


def rule_calls() -> Mapping[str, int]:
    """The rule invocations made so far, by phase. A fact, not an estimate."""
    return dict(_RULE_CALLS)


def reset_rule_calls() -> None:
    """Start a fresh account. The harness does this once, before it builds anything."""
    _RULE_CALLS.clear()


def check_rule(rule: object) -> Rule:
    """Refuse anything that is not EXACTLY a declared-shape candidate rule.

    Round one accepted any :class:`Rule` subclass and called its overridable
    ``holds``, so a three-line subclass returning ``shell.family ==
    'depth-outliers'`` selected shells perfectly and the whole truth wall was
    decoration. Exact type identity on the rule AND on every one of its feature
    tests closes that: a subclass, an instance with an adaptive callable in
    place of a threshold, or a test naming something outside :data:`FEATURES` is
    refused before a single shell is looked at.
    """
    if type(rule) is not Rule:
        raise RuleRefused(
            f"a {type(rule).__name__} is not a declared candidate rule. This experiment applies "
            "exactly tests.shell_discriminant.Rule - not a subclass, not a duck-typed object "
            "with a holds method - because an overridable holds is an unlimited channel to the "
            "answer"
        )
    if type(rule).holds is not Rule.holds or type(rule).as_dict is not Rule.as_dict:
        raise RuleRefused(f"{rule.rule_id} does not use the declared rule behaviour")
    if not isinstance(rule.rule_id, str) or not rule.rule_id:
        raise RuleRefused(f"a candidate rule identifies itself as {rule.rule_id!r}")
    if not isinstance(rule.tests, tuple) or not 1 <= len(rule.tests) <= 2:
        raise RuleRefused(f"{rule.rule_id} does not carry one or two declared feature tests")
    for test in rule.tests:
        if type(test) is not FeatureTest:
            raise RuleRefused(
                f"{rule.rule_id} carries a {type(test).__name__} where a FeatureTest belongs. A "
                "test whose comparison is code rather than data is not a rule this experiment "
                "declared, whatever it is called"
            )
        if type(test).holds is not FeatureTest.holds:
            raise RuleRefused(f"{rule.rule_id} carries a feature test with overridden behaviour")
        if test.feature not in FEATURES:
            raise RuleRefused(f"{rule.rule_id} reads {test.feature!r}, which is not an observable")
        if test.comparison not in COMPARISONS:
            raise RuleRefused(f"{rule.rule_id} compares with {test.comparison!r}")
        if isinstance(test.threshold, bool) or not isinstance(test.threshold, (int, float)):
            raise RuleRefused(
                f"{rule.rule_id} thresholds {test.feature} at {test.threshold!r}. A threshold is "
                "a number declared before any result, not a callable that can look at the shell"
            )
        if not math.isfinite(float(test.threshold)):
            raise RuleRefused(f"{rule.rule_id} thresholds {test.feature} at {test.threshold!r}")
    return rule


def select(rule: Rule, shells: Sequence[object]) -> tuple[str, ...]:
    """The shell ids ``rule`` selects. The ONLY way a rule is ever applied.

    Every element is passed through :func:`check_observable` and the full
    :func:`check_observable_is_consistent` validator first, so the signature is
    not a suggestion: handing this function the evaluator's own
    :class:`ShellRecord` list, a subclass of the observable record, a mapping or
    a forged observable raises rather than working. Every invocation is counted
    against the current phase of the rule-call ledger.
    """
    check_rule(rule)
    phase = _PHASE[-1]
    chosen: list[str] = []
    for item in shells:
        shell = check_observable_is_consistent(check_observable(item))
        _RULE_CALLS[phase] = _RULE_CALLS.get(phase, 0) + 1
        if rule.holds(shell):
            chosen.append(shell.shell_id)
    return tuple(chosen)


def declared(rule: object) -> bool:
    """Whether ``rule`` is one of the predeclared candidates, BODY and all.

    The post-result-threshold control, and structural rather than nominal:
    round one compared ``rule_id`` alone, so a rule body edited after a result
    could reuse an accepted id and be published as a declared candidate. This
    compares the complete serialised rule - id, family, text and every feature,
    comparison and threshold - against the reviewed matrix.
    """
    try:
        candidate = check_rule(rule)
    except RuleRefused:
        return False
    body = candidate.as_dict()
    return any(body == other.as_dict() for other in RULE_GRID)


# --- the predeclared candidate matrix -------------------------------------------------


#: For each observable, the direction that would mean "suspect this shell" and
#: every threshold that will ever be tried.
#:
#: The thresholds are a SWEEP of each observable's own natural range rather
#: than a set of interesting-looking values: the four fractions get the same
#: uniform 0.1-to-0.9 sweep because that is the whole range a fraction has, the
#: counts get a doubling sweep, and the two lengths get a geometric sweep from
#: a fraction of a voxel to several. That is what makes it a grid rather than a
#: search - nobody had to know where the answer was to write it - and the whole
#: matrix is hashed into the fixture manifest by :func:`grid_digest`, so a
#: threshold moved or added after a confusion count exists is a refusal rather
#: than a footnote.
GRID: Final[Mapping[str, tuple[str, tuple[float, ...]]]] = {
    "faces": ("<=", (4.0, 8.0, 16.0, 32.0, 64.0, 128.0)),
    "area": ("<=", (0.001, 0.005, 0.02, 0.05, 0.2)),
    "diagonal": ("<=", (0.05, 0.1, 0.2, 0.4, 0.8)),
    "views_containing": ("<=", (0.0, 1.0, 2.0, 3.0, 4.0)),
    "views_touching": ("<=", (1.0, 2.0, 3.0, 4.0)),
    "confirming_fraction": ("<=", (0.1, 0.3, 0.5, 0.7, 0.9)),
    "contradicting_fraction": (">=", (0.1, 0.3, 0.5, 0.7, 0.9)),
    "occluded_fraction": (">=", (0.1, 0.3, 0.5, 0.7, 0.9)),
    "no_return_fraction": (">=", (0.1, 0.3, 0.5, 0.7, 0.9)),
    "nearest_measured_distance": (">=", (0.02, 0.05, 0.1, 0.2, 0.4)),
    "weight_mean_in_thresholds": ("<=", (0.5, 1.0, 2.0, 4.0, 8.0)),
}

#: The conjunctions that will be tried, as ordered feature pairs. A fixed list
#: rather than every pair of the eleven observables: the full product would be
#: 55 pairs and over a thousand cells, which is a search wearing a grid's
#: name. These five are the hypotheses the surface-oracle findings actually
#: motivate - a small blob far from any return, a small blob the rays see
#: through, a distant blob the rays see through, a small blob the field barely
#: weighted, and a small blob nothing confirms.
PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("nearest_measured_distance", "faces"),
    ("contradicting_fraction", "faces"),
    ("nearest_measured_distance", "contradicting_fraction"),
    ("weight_mean_in_thresholds", "faces"),
    ("confirming_fraction", "diagonal"),
)


def rule_grid() -> tuple[Rule, ...]:
    """Every candidate that will ever be evaluated, in the order it is considered.

    Deterministic and total: singles in :data:`GRID` order then pairs in
    :data:`PAIRS` order, every threshold combination of each. Nothing is
    skipped after a result, and :mod:`tests.test_shell_discriminant` refuses a
    report whose evaluated cells are not exactly this list.
    """
    rules: list[Rule] = []
    for feature, (comparison, thresholds) in GRID.items():
        for threshold in thresholds:
            rules.append(
                Rule(
                    rule_id=f"{SINGLE}:{feature}{comparison}{threshold:g}",
                    tests=(FeatureTest(feature, comparison, threshold),),
                )
            )
    for first, second in PAIRS:
        first_comparison, first_thresholds = GRID[first]
        second_comparison, second_thresholds = GRID[second]
        for left in first_thresholds:
            for right in second_thresholds:
                rules.append(
                    Rule(
                        rule_id=(
                            f"{PAIR}:{first}{first_comparison}{left:g}"
                            f"&{second}{second_comparison}{right:g}"
                        ),
                        tests=(
                            FeatureTest(first, first_comparison, left),
                            FeatureTest(second, second_comparison, right),
                        ),
                    )
                )
    return tuple(rules)


RULE_GRID: Final[tuple[Rule, ...]] = rule_grid()


#: The resolution the product question is actually asked at. Predeclared by the
#: round-two planner packet: training eligibility, the ordering and every
#: holdout gate read APPLIANCE records and nothing else. The second declared
#: resolution stays in the corpus and in every cross-tab as DIAGNOSTIC evidence
#: and can never help a candidate qualify or pass.
SELECTION_RESOLUTION: Final = "appliance"

#: A material fraction of the false surface. Predeclared: a rule that removed a
#: shell or two out of many would be a curiosity, not a change worth making to
#: an export path, and a bar chosen after the counts existed would be no bar.
MATERIAL_FRACTION: Final = 0.5

#: The most candidate evaluations this experiment will ever pay for: every
#: declared rule against every shell of the declared population, once. Stated
#: before the population existed so "bounded" is a fact rather than a
#: measurement, and published beside the spend.
WORK_CEILING: Final = 40000

#: The order candidates are considered in and who may inform what, in one
#: paragraph. Hashed into :func:`grid_digest`, so the protocol a report claims
#: to have followed is the protocol that was reviewed. Deliberately NOT hashed
#: into :func:`candidate_digest`: this protocol was rewritten for round two,
#: while the candidate bodies were frozen before the sealed holdout ran.
SELECTION_PROTOCOL: Final = (
    "Every rule in tests.shell_discriminant.RULE_GRID is evaluated exactly once against every "
    "declared unsupported observable shell, into one immutable cached decision matrix, and every "
    "confusion, cross-tab, eligibility test, ordering, gate, selection and recommendation is "
    "derived from that matrix rather than from a second application of any rule. Selection is "
    "asked at the product's own resolution: training eligibility and ordering read TRAIN records "
    "at SELECTION_RESOLUTION and nothing else, and the second declared resolution is diagnostic "
    "and can never help a candidate qualify or pass. A rule is eligible when, over the training "
    "appliance records, it removes zero analytically acceptable shell faces and at least "
    "MATERIAL_FRACTION of the false shell faces. Eligible rules are ordered by the most false "
    "training faces removed, then by the fewest shells selected, then by rule id, and the first "
    "is the selected rule. The sealed HOLDOUT split is measured once afterwards, at the same "
    "appliance resolution, as evaluation only: the candidate bodies, the gates, the ceiling and "
    "the selection are frozen before any holdout number exists, and swapping or withholding the "
    "holdout labels cannot change the selection. A selected rule is recommended only if it also "
    "clears the holdout gates AND its counterfactual - the actual removal of exactly the whole "
    "shells the rule selects over ALL closed shells, with no analytic unsupported prefilter, on "
    "every sealed holdout appliance population where it fires, re-measured by the accepted "
    "surface oracle's own code - shows an output-facing false-surface gain with every "
    "preservation gate holding across the complete set. Any analytically acceptable supported or "
    "unsupported surface the rule removes is damage and is published as such. If no rule is "
    "eligible, or the selected rule fails holdout or any counterfactual, the experiment selects "
    "none and recommends no pruning. The truth-fed upper bound is evaluator-only and is never a "
    "candidate."
)

#: The gates, named once so the report, the reader and the documentation cannot
#: drift apart.
GATES: Final[tuple[str, ...]] = (
    "train_removes_no_correct_faces",
    "train_removes_material_false_faces",
    "holdout_removes_no_correct_faces",
    "holdout_removes_material_false_faces",
    "counterfactual_false_faces_fell",
    "counterfactual_correct_faces_held",
    "counterfactual_removes_no_supported_faces",
    "counterfactual_analytic_accuracy_held",
    "counterfactual_triangle_accuracy_held",
    "counterfactual_reference_coverage_held",
    "counterfactual_protected_volume_held",
    "counterfactual_bridging_held",
    "counterfactual_edge_coverage_held",
    "counterfactual_thin_feature_held",
    "counterfactual_dimensions_held",
    "counterfactual_topology_held",
)


#: The candidate vocabulary, directions, thresholds, pairs, order and 195 Rule
#: bodies, hashed on their own. Frozen by the round-two planner packet BEFORE
#: any sealed member was executed, and separate from :func:`grid_digest` on
#: purpose: the selection protocol and the gate names were rewritten this round
#: and their digest legitimately moved, while a candidate body that moved after
#: a result would be the thing this experiment exists to make impossible.
FROZEN_CANDIDATE_DIGEST: Final = "4fe6627ea7618b372cf833eae805547420fa91c412808e1e12701b1191178ca2"


def candidate_digest() -> str:
    """A SHA-256 over the CANDIDATES alone - no protocol, no gates.

    Canonical sorted compact JSON over the candidate schema version,
    :data:`GRID`, :data:`PAIRS`, every :meth:`Rule.as_dict` value and the sorted
    :data:`FEATURES` names. The manifest binds it, the report publishes it and
    :func:`check_candidates_are_frozen` refuses any drift from
    :data:`FROZEN_CANDIDATE_DIGEST`.
    """
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "grid": {name: [direction, list(values)] for name, (direction, values) in GRID.items()},
            "pairs": [list(pair) for pair in PAIRS],
            "rules": [rule.as_dict() for rule in RULE_GRID],
            "features": sorted(FEATURES),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_candidates_are_frozen() -> str:
    """Refuse to run at all if a candidate body moved after the round-one result.

    Called by the harness before anything is fused. The round-two contract is
    that the 195 bodies are exactly the ones round one published, so this is
    not a formality: a widened threshold, a reordered pair or a twelfth feature
    all change this digest, and all of them would make a sealed holdout number
    meaningless.
    """
    actual = candidate_digest()
    if actual != FROZEN_CANDIDATE_DIGEST:
        raise RuleRefused(
            f"the candidate matrix hashes to {actual}; it was frozen at "
            f"{FROZEN_CANDIDATE_DIGEST} before the sealed holdout was declared. A candidate "
            "vocabulary, direction, threshold, pair, order or rule body that moved after a "
            "result is not something this experiment tunes - it is a refusal"
        )
    return actual


def grid_digest() -> str:
    """A SHA-256 over the whole candidate matrix, for the manifest to bind.

    The matrix is code and the population is data, and neither is allowed to
    move without the other noticing. The fixture manifest records this digest;
    :func:`tests.shell_discriminant_harness.load_corpus` refuses a corpus whose
    recorded digest is not this one.
    """
    payload = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "grid": {name: [direction, list(values)] for name, (direction, values) in GRID.items()},
            "pairs": [list(pair) for pair in PAIRS],
            "rules": [rule.as_dict() for rule in RULE_GRID],
            "features": sorted(FEATURES),
            "protocol": SELECTION_PROTOCOL,
            "gates": GATES,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- what a candidate did, counted ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Confusion:
    """One rule's complete confusion over one slice of the population.

    Complete rather than summarised: both classes, both as shells and as
    FACES, selected and left. Faces because the thing a pruning change would
    remove is triangles, and a rule that removed one enormous correct shell
    and twenty tiny false ones would look excellent counted as shells.
    """

    slice_name: str
    rule_id: str
    shells: int
    false_shells: int
    correct_shells: int
    false_faces: int
    correct_faces: int
    selected_shells: int
    selected_false_shells: int
    selected_correct_shells: int
    selected_false_faces: int
    selected_correct_faces: int

    def __post_init__(self) -> None:
        if self.false_shells + self.correct_shells != self.shells:
            raise DiscriminantUnusable(
                f"{self.slice_name}/{self.rule_id}: {self.false_shells} false and "
                f"{self.correct_shells} correct shells do not add up to {self.shells}. A "
                "confusion whose classes do not reconcile is not a confusion"
            )
        if self.selected_false_shells + self.selected_correct_shells != self.selected_shells:
            raise DiscriminantUnusable(
                f"{self.slice_name}/{self.rule_id}: the selected shells do not reconcile"
            )
        if self.selected_false_faces > self.false_faces:
            raise DiscriminantUnusable(
                f"{self.slice_name}/{self.rule_id} removes more false faces than exist"
            )
        if self.selected_correct_faces > self.correct_faces:
            raise DiscriminantUnusable(
                f"{self.slice_name}/{self.rule_id} removes more correct faces than exist"
            )

    @property
    def false_face_recall(self) -> float:
        return self.selected_false_faces / self.false_faces if self.false_faces else 0.0

    @property
    def correct_face_loss(self) -> float:
        return self.selected_correct_faces / self.correct_faces if self.correct_faces else 0.0

    @property
    def removes_no_correct_faces(self) -> bool:
        return self.selected_correct_faces == 0

    @property
    def removes_material_false_faces(self) -> bool:
        return self.false_faces > 0 and self.false_face_recall >= MATERIAL_FRACTION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "shell_discriminant_confusion",
            "slice": self.slice_name,
            "rule_id": self.rule_id,
            "shells": self.shells,
            "false_shells": self.false_shells,
            "correct_shells": self.correct_shells,
            "false_faces": self.false_faces,
            "correct_faces": self.correct_faces,
            "selected_shells": self.selected_shells,
            "selected_false_shells": self.selected_false_shells,
            "selected_correct_shells": self.selected_correct_shells,
            "selected_false_faces": self.selected_false_faces,
            "selected_correct_faces": self.selected_correct_faces,
            "false_face_recall": self.false_face_recall,
            "correct_face_loss": self.correct_face_loss,
            "removes_no_correct_faces": self.removes_no_correct_faces,
            "removes_material_false_faces": self.removes_material_false_faces,
            "material_fraction": MATERIAL_FRACTION,
            "unit": FACE_UNIT,
        }


@dataclass(frozen=True, slots=True)
class DecisionMatrix:
    """Every candidate's decision on every declared shell, computed once.

    The producer's ONE bounded pass, and the only place a rule is applied
    during it. Round one applied all 195 rules to eleven overlapping confusion
    slices and then again inside the selector - 89,505 invocations published as
    16,380, against a declared ceiling of 40,000. Here every confusion, every
    cross-tab cell, every eligibility test, the ordering, the gates and the
    recommendation are LOOKUPS into this object, so the overlapping summaries
    cost nothing and the published spend is the real one.
    """

    rule_ids: tuple[str, ...]
    shell_ids: tuple[str, ...]
    #: rule id -> the shell ids that rule selects. Frozen sets, so a consumer
    #: cannot edit a decision after the fact.
    decisions: Mapping[str, frozenset[str]]
    #: The rule invocations this matrix cost. Exactly rules times shells.
    calls: int

    def __post_init__(self) -> None:
        if len(set(self.rule_ids)) != len(self.rule_ids):
            raise DiscriminantUnusable("a decision matrix names the same rule twice")
        if len(set(self.shell_ids)) != len(self.shell_ids):
            raise DiscriminantUnusable("a decision matrix names the same shell twice")
        if set(self.decisions) != set(self.rule_ids):
            raise DiscriminantUnusable("a decision matrix has no row for every declared rule")
        known = set(self.shell_ids)
        for rule_id, chosen in self.decisions.items():
            unknown = sorted(chosen - known)
            if unknown:
                raise DiscriminantUnusable(
                    f"{rule_id} decided about {unknown}, which are not in the declared population"
                )
        if self.calls != len(self.rule_ids) * len(self.shell_ids):
            raise DiscriminantUnusable(
                f"a decision matrix of {len(self.rule_ids)} rule(s) over "
                f"{len(self.shell_ids)} shell(s) cost {self.calls} invocation(s); one bounded "
                "pass is exactly one decision per cell"
            )

    def selected(self, rule_id: str) -> frozenset[str]:
        try:
            return self.decisions[rule_id]
        except KeyError:
            raise DiscriminantUnusable(f"{rule_id} is not a row of this decision matrix") from None

    def holds(self, rule_id: str, shell_id: str) -> bool:
        if shell_id not in set(self.shell_ids):
            raise DiscriminantUnusable(f"{shell_id} is not a column of this decision matrix")
        return shell_id in self.selected(rule_id)


def preflight(records: Sequence[ShellRecord]) -> int:
    """The declared spend, checked against the unchanged ceiling BEFORE any pass.

    ``len(RULE_GRID)`` times the complete unsupported-shell record count. The
    ceiling is the one that was declared before this population existed and is
    not the number to move: a population that does not fit is returned for
    planner review.
    """
    evaluations = len(RULE_GRID) * len(records)
    if evaluations > WORK_CEILING:
        raise DiscriminantUnusable(
            f"{len(RULE_GRID)} rule(s) over {len(records)} shell(s) is {evaluations} "
            f"evaluation(s), past the declared ceiling of {WORK_CEILING}. The ceiling was stated "
            "before the population existed and is not the number to move; the population is "
            "refused and returned for planner review"
        )
    return evaluations


def build_decision_matrix(
    records: Sequence[ShellRecord], *, phase: str = PRODUCER_PHASE
) -> DecisionMatrix:
    """Evaluate every declared candidate on every declared shell, exactly once.

    ``phase`` names whose work this is. It defaults to the producer's one
    permitted pass; a standalone strict reader building its own independent
    audit matrix passes :data:`READER_PHASE` so its invocations are accounted
    truthfully and are never hidden inside the producer's number.
    """
    evaluations = preflight(records)
    shells = [record.observable for record in records]
    shell_ids = tuple(record.shell_id for record in records)
    if len(set(shell_ids)) != len(shell_ids):
        raise DiscriminantUnusable(
            "the declared population names one shell identity twice, so a decision about it "
            "would be a decision about two shells"
        )
    before = rule_calls().get(phase, 0)
    decisions: dict[str, frozenset[str]] = {}
    with accounting(phase):
        for rule in RULE_GRID:
            decisions[rule.rule_id] = frozenset(select(rule, shells))
    spent = rule_calls().get(phase, 0) - before
    if spent != evaluations:
        raise DiscriminantUnusable(
            f"building the decision matrix cost {spent} rule invocation(s) where the declared "
            f"one bounded pass is {evaluations}"
        )
    return DecisionMatrix(
        rule_ids=tuple(rule.rule_id for rule in RULE_GRID),
        shell_ids=shell_ids,
        decisions=MappingProxyType(decisions),
        calls=spent,
    )


def confusion(
    matrix: DecisionMatrix,
    rule_id: str,
    records: Sequence[ShellRecord],
    *,
    slice_name: str,
) -> Confusion:
    """Count what ``rule_id`` did over ``records``, entirely out of the matrix.

    No rule is applied here. That is the point: this function is called for
    every candidate on every one of the twenty-eight cross-tab cells plus the
    aggregates, and if it applied a rule the published work account would be a
    multiple of the real matrix again.
    """
    selected = matrix.selected(rule_id)
    known = {record.shell_id for record in records}
    false_records = [record for record in records if record.truth.false_surface]
    correct_records = [record for record in records if not record.truth.false_surface]
    here = selected & known
    return Confusion(
        slice_name=slice_name,
        rule_id=rule_id,
        shells=len(records),
        false_shells=len(false_records),
        correct_shells=len(correct_records),
        false_faces=sum(record.truth.faces for record in false_records),
        correct_faces=sum(record.truth.faces for record in correct_records),
        selected_shells=len(here),
        selected_false_shells=sum(1 for record in false_records if record.shell_id in here),
        selected_correct_shells=sum(1 for record in correct_records if record.shell_id in here),
        selected_false_faces=sum(
            record.truth.faces for record in false_records if record.shell_id in here
        ),
        selected_correct_faces=sum(
            record.truth.faces for record in correct_records if record.shell_id in here
        ),
    )


def cross_cells(resolutions: Sequence[str]) -> tuple[tuple[str, str, str], ...]:
    """The exact split x family x resolution cross-product, in declared order.

    Every cell, including the ones no shell landed in. Round one published four
    separate one-dimensional summaries and silently omitted the families that
    produced no shell, which is exactly how a rule that only works on one
    family looks like a rule that works.
    """
    return tuple(
        (split, family, cell)
        for split in SPLITS
        for family in contract.PERTURBATIONS
        for cell in resolutions
    )


def cross_tab_name(split: str, family: str, cell: str) -> str:
    return f"split:{split}|family:{family}|cell:{cell}"


def cross_tab(
    matrix: DecisionMatrix,
    rule_id: str,
    records: Sequence[ShellRecord],
    *,
    resolutions: Sequence[str],
) -> tuple[Confusion, ...]:
    """One candidate's complete cross-product of confusions, zero cells included."""
    grouped: dict[tuple[str, str, str], list[ShellRecord]] = {
        cell: [] for cell in cross_cells(resolutions)
    }
    for record in records:
        key = (record.split, record.family, record.cell_id)
        if key not in grouped:
            raise DiscriminantUnusable(
                f"{record.shell_id} is in split {record.split!r}, family {record.family!r} and "
                f"resolution {record.cell_id!r}, which is not one of the declared cross-tab cells"
            )
        grouped[key].append(record)
    return tuple(
        confusion(matrix, rule_id, grouped[key], slice_name=cross_tab_name(*key))
        for key in cross_cells(resolutions)
    )


def aggregate_slices(records: Sequence[ShellRecord]) -> dict[str, tuple[ShellRecord, ...]]:
    """The readable roll-ups, retained beside the exact cross-product.

    Aggregates only. Every one of them is derived from the same records and the
    same decision matrix as the cross-tab, so they cost no rule invocation and
    the strict reader recomputes each of them rather than believing it.
    """
    out: dict[str, list[ShellRecord]] = {"all": []}
    for split in SPLITS:
        out[f"split:{split}"] = []
    for record in records:
        out["all"].append(record)
        out[f"split:{record.split}"].append(record)
        out.setdefault(f"family:{record.family}", []).append(record)
        out.setdefault(f"cell:{record.cell_id}", []).append(record)
        out.setdefault(f"split:{record.split}|cell:{record.cell_id}", []).append(record)
    return {name: tuple(out[name]) for name in sorted(out)}


# --- selection, and then the only thing that actually settles it ------------------------


@dataclass(frozen=True, slots=True)
class Selection:
    """What the declared protocol chose, and why - including choosing nothing."""

    rule: Rule | None
    eligible_rule_ids: tuple[str, ...]
    considered: int
    evaluations: int
    work_ceiling: int
    reason: str
    #: The resolution eligibility and ordering were decided at. Predeclared,
    #: published, and the only one a candidate can qualify on.
    selection_resolution: str
    #: How many records the choice actually read. Published so "trained on the
    #: appliance arm only" is a number a reader can check rather than a claim.
    training_shells: int

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "shell_discriminant_selection",
            "selected": self.rule.as_dict() if self.rule is not None else None,
            "eligible_rule_ids": list(self.eligible_rule_ids),
            "considered_rules": self.considered,
            "evaluations": self.evaluations,
            "work_ceiling": self.work_ceiling,
            "selection_resolution": self.selection_resolution,
            "training_shells": self.training_shells,
            "protocol": SELECTION_PROTOCOL,
            "reason": self.reason,
        }


def training_records(records: Sequence[ShellRecord]) -> tuple[ShellRecord, ...]:
    """The records selection is allowed to read: TRAIN, at the shipped resolution."""
    return tuple(
        record
        for record in records
        if record.split == TRAIN and record.cell_id == SELECTION_RESOLUTION
    )


def holdout_records(records: Sequence[ShellRecord]) -> tuple[ShellRecord, ...]:
    """The records the holdout gates read: sealed HOLDOUT, at the same resolution."""
    return tuple(
        record
        for record in records
        if record.split == HOLDOUT and record.cell_id == SELECTION_RESOLUTION
    )


def choose(matrix: DecisionMatrix, records: Sequence[ShellRecord]) -> Selection:
    """Apply :data:`SELECTION_PROTOCOL` to the training appliance records only.

    Two things this does not do, both of which round one did. It does not
    aggregate the coarse diagnostic cell with the appliance cell, because the
    product question is asked at the resolution an owner actually gets and a
    rule that only works two orders of magnitude coarser has not answered it.
    And it does not apply a single rule: every eligibility test is a lookup
    into ``matrix``, so the choice costs no invocation and cannot smuggle a
    second pass past the ceiling.

    The holdout labels are never read here. Swapping or withholding them
    changes nothing about what this returns, which is what
    :mod:`tests.test_shell_discriminant` asserts directly.
    """
    training = training_records(records)
    evaluations = len(RULE_GRID) * len(records)
    if not training:
        return Selection(
            rule=None,
            eligible_rule_ids=(),
            considered=len(RULE_GRID),
            evaluations=evaluations,
            work_ceiling=WORK_CEILING,
            reason=(
                "the training split contains no unsupported closed shell at "
                f"{SELECTION_RESOLUTION} resolution to learn from"
            ),
            selection_resolution=SELECTION_RESOLUTION,
            training_shells=0,
        )
    scored: list[tuple[int, int, str, Rule]] = []
    for rule in RULE_GRID:
        counts = confusion(
            matrix, rule.rule_id, training, slice_name=f"split:{TRAIN}|cell:{SELECTION_RESOLUTION}"
        )
        if counts.removes_no_correct_faces and counts.removes_material_false_faces:
            scored.append(
                (-counts.selected_false_faces, counts.selected_shells, rule.rule_id, rule)
            )
    if not scored:
        return Selection(
            rule=None,
            eligible_rule_ids=(),
            considered=len(RULE_GRID),
            evaluations=evaluations,
            work_ceiling=WORK_CEILING,
            reason=(
                "no declared rule removed at least "
                f"{MATERIAL_FRACTION:.0%} of the training split's false shell faces at "
                f"{SELECTION_RESOLUTION} resolution while removing none of its correct ones"
            ),
            selection_resolution=SELECTION_RESOLUTION,
            training_shells=len(training),
        )
    scored.sort(key=lambda item: (item[0], item[1], item[2]))
    return Selection(
        rule=scored[0][3],
        eligible_rule_ids=tuple(
            item[2] for item in sorted(scored, key=lambda i: (i[0], i[1], i[2]))
        ),
        considered=len(RULE_GRID),
        evaluations=evaluations,
        work_ceiling=WORK_CEILING,
        reason=(
            f"{len(scored)} rule(s) cleared both training gates at {SELECTION_RESOLUTION} "
            f"resolution; the protocol's order selected {scored[0][2]}"
        ),
        selection_resolution=SELECTION_RESOLUTION,
        training_shells=len(training),
    )


def prune(result: contract.ArmResult, remove: Sequence[int]) -> contract.ArmResult:
    """The same surface with exactly those FACE indices removed, and nothing else.

    Whole shells only, by construction: the caller passes the face indices a
    :class:`ShellRecord` carries, which are the accepted oracle's own shell
    grouping. Vertices left unreferenced by the removal are dropped and the
    remaining faces reindexed, because a mesh that shipped orphan vertices
    would score its accuracy on points no triangle uses - which is the exact
    mistake the surface oracle was built to stop making.

    :attr:`~tests.surface_oracle.ArmResult.field_evidence` is dropped on the
    result: the pruned mesh is not what the extractor emitted, and carrying the
    field's bookkeeping onto it would let a reader believe otherwise.
    """
    removing = set(remove)
    unknown = sorted(index for index in removing if not 0 <= index < len(result.faces))
    if unknown:
        raise DiscriminantUnusable(
            f"a counterfactual was asked to remove face(s) {unknown} from a mesh with "
            f"{len(result.faces)} face(s)"
        )
    kept = [face for index, face in enumerate(result.faces) if index not in removing]
    used = sorted({index for face in kept for index in face})
    remap = {old: new for new, old in enumerate(used)}
    return contract.ArmResult(
        name=f"{result.name}+pruned",
        arm=result.arm,
        cell_id=result.cell_id,
        views_used=result.views_used,
        poses_used=result.poses_used,
        input_sha256=result.input_sha256,
        points=tuple(result.points[index] for index in used),
        work=result.work,
        faces=tuple((remap[a], remap[b], remap[c]) for a, b, c in kept),
        field_evidence=None,
    )


@dataclass(frozen=True, slots=True)
class SurfaceAccount:
    """The output-facing numbers one mesh scores, before or after a removal.

    Every one of them is produced by the accepted surface oracle's own
    measurement code over the same scene and the same references, so the two
    sides of a counterfactual cannot disagree about what is being measured.
    """

    faces: int
    vertices: int
    analytic_within_tolerance_fraction: float
    reference_coverage_fraction: float
    triangle_within_tolerance_fraction: float
    protected_area: float
    bridging_faces: int
    surface_area: float
    edge_coverage: float
    thin_feature_coverage: float
    worst_dimension_error: float
    unsupported_false_shells: int
    unsupported_false_shell_faces: int
    unsupported_correct_shells: int
    unsupported_correct_shell_faces: int
    nonmanifold_edge_count: int
    boundary_edge_count: int
    shell_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "faces": self.faces,
            "vertices": self.vertices,
            "analytic_within_tolerance_fraction": self.analytic_within_tolerance_fraction,
            "reference_coverage_fraction": self.reference_coverage_fraction,
            "triangle_within_tolerance_fraction": self.triangle_within_tolerance_fraction,
            "protected_area": self.protected_area,
            "bridging_faces": self.bridging_faces,
            "surface_area": self.surface_area,
            "edge_coverage": self.edge_coverage,
            "thin_feature_coverage": self.thin_feature_coverage,
            "worst_dimension_error": self.worst_dimension_error,
            "unsupported_false_shells": self.unsupported_false_shells,
            "unsupported_false_shell_faces": self.unsupported_false_shell_faces,
            "unsupported_correct_shells": self.unsupported_correct_shells,
            "unsupported_correct_shell_faces": self.unsupported_correct_shell_faces,
            "nonmanifold_edge_count": self.nonmanifold_edge_count,
            "boundary_edge_count": self.boundary_edge_count,
            "shell_count": self.shell_count,
        }


def _amount(record: Mapping[str, object], key: str) -> float:
    """One number out of a published measurement, or a refusal.

    The accepted oracle's measurements serialise to JSON-shaped mappings, so a
    counterfactual reading one back is reading ``object``. Coercing that
    silently is how a missing key becomes a zero and a zero becomes a gain, so
    an absent or non-numeric field is a refusal rather than a default.
    """
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DiscriminantUnusable(
            f"a re-measured surface reports {key}={value!r}, which is not a number this "
            "counterfactual can compare"
        )
    return float(value)


def _tally(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiscriminantUnusable(
            f"a re-measured surface reports {key}={value!r}, which is not a count"
        )
    return value


def account(scene: contract.Scene, result: contract.ArmResult) -> SurfaceAccount:
    """Re-measure one mesh with :func:`tests.surface_oracle.measure` and read it back.

    Deliberately the accepted entry point rather than a shortcut through the
    individual measurements: a counterfactual scored by code written beside it
    would be a second opinion about its own change.
    """
    measurements = {item.name: item.as_dict() for item in contract.measure(scene, result)}
    analytic = measurements.get("analytic", {})
    samples = measurements.get("surface_samples", {})
    topology = measurements.get("topology", {})
    edge = measurements.get("edge", {})
    dimensions = measurements.get("dimensions", {})
    errors = dimensions.get("errors", {})
    worst = 0.0
    if isinstance(errors, Mapping):
        values = [abs(_amount(errors, name)) for name in errors]
        worst = max(values) if values else 0.0
    return SurfaceAccount(
        faces=len(result.faces),
        vertices=len(result.points),
        analytic_within_tolerance_fraction=_amount(analytic, "within_tolerance_fraction"),
        reference_coverage_fraction=_amount(analytic, "reference_coverage_fraction"),
        triangle_within_tolerance_fraction=_amount(samples, "within_tolerance_fraction"),
        protected_area=_amount(samples, "protected_area"),
        bridging_faces=_tally(samples, "bridging_faces"),
        surface_area=_amount(samples, "surface_area"),
        edge_coverage=_amount(edge, "edge_coverage"),
        thin_feature_coverage=_amount(edge, "thin_feature_coverage"),
        worst_dimension_error=worst,
        unsupported_false_shells=_tally(topology, "unsupported_inaccurate_closed_shells"),
        unsupported_false_shell_faces=_tally(topology, "unsupported_inaccurate_closed_shell_faces"),
        unsupported_correct_shells=_tally(topology, "unsupported_accurate_closed_shells"),
        unsupported_correct_shell_faces=_tally(topology, "unsupported_accurate_closed_shell_faces"),
        nonmanifold_edge_count=_tally(topology, "nonmanifold_edge_count"),
        boundary_edge_count=_tally(topology, "boundary_edge_count"),
        shell_count=_tally(topology, "shell_count"),
    )


@dataclass(frozen=True, slots=True)
class CounterfactualEvidence:
    """What removing exactly those shells did to the mesh an owner would get.

    Both accounts, the removal itself, the DAMAGE it caused, and every
    preservation gate named and evaluated. A gain here is an output-facing
    measurement of the recomputed surface; a confusion count is not evidence
    that the mesh improved and this class is what the recommendation is
    actually allowed to rest on.

    The removal is over ALL closed shells, without the evaluator's unsupported
    prefilter, because that prefilter is knowledge a deployable rule does not
    have. Every analytically acceptable face the rule takes out - unsupported
    but correct, or supported and never in the labelled population at all - is
    counted here and gated on.
    """

    member_id: str
    cell_id: str
    split: str
    reconstruction_digest: str
    #: Every closed shell the rule was applied to, and how many it removed.
    closed_shells: int
    removed_shell_ids: tuple[str, ...]
    removed_faces: int
    #: The removal, broken down by what the evaluator knows each shell was.
    removed_false_shells: int
    removed_false_shell_faces: int
    removed_correct_shells: int
    removed_correct_shell_faces: int
    #: Shells the rule removed that are not in the unsupported population at
    #: all: geometry the scene's own ideal returns DO reach. Pure damage, and
    #: invisible to any evidence built on the prefiltered population.
    removed_supported_shells: int
    removed_supported_shell_faces: int
    before: SurfaceAccount
    after: SurfaceAccount
    gates: Mapping[str, bool]

    @property
    def false_faces_removed(self) -> int:
        return self.before.unsupported_false_shell_faces - self.after.unsupported_false_shell_faces

    @property
    def correct_faces_removed(self) -> int:
        return (
            self.before.unsupported_correct_shell_faces - self.after.unsupported_correct_shell_faces
        )

    @property
    def passed(self) -> bool:
        return all(self.gates.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "shell_discriminant_counterfactual",
            "member_id": self.member_id,
            "cell_id": self.cell_id,
            "split": self.split,
            "reconstruction_digest": self.reconstruction_digest,
            "closed_shells": self.closed_shells,
            "removed_shell_ids": list(self.removed_shell_ids),
            "removed_faces": self.removed_faces,
            "removed_false_shells": self.removed_false_shells,
            "removed_false_shell_faces": self.removed_false_shell_faces,
            "removed_correct_shells": self.removed_correct_shells,
            "removed_correct_shell_faces": self.removed_correct_shell_faces,
            "removed_supported_shells": self.removed_supported_shells,
            "removed_supported_shell_faces": self.removed_supported_shell_faces,
            "false_faces_removed": self.false_faces_removed,
            "correct_faces_removed": self.correct_faces_removed,
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
            "gates": {name: bool(value) for name, value in sorted(self.gates.items())},
            "passed": self.passed,
        }


def check_shell_identity(
    result: contract.ArmResult,
    observables: Sequence[ObservableShell],
    records: Sequence[ShellRecord],
    *,
    member_id: str,
    cell_id: str,
) -> tuple[tuple[int, ...], ...]:
    """Recompute the mesh's components and bind every published shell to one.

    Independent of everything the population says about itself: the connected
    components are recomputed here from ``result.faces``, and each observable's
    and each record's declared shell index, face set, member and resolution are
    checked against them BEFORE a triangle is removed. Round one trusted the
    face indices the record carried, so its advertised wrong-shell control
    happily measured the wrong triangles instead of refusing them.
    """
    components = tuple(tuple(members) for members in _shell_members(result.faces))
    for shell in observables:
        if not 0 <= shell.shell_index < len(components):
            raise DiscriminantUnusable(
                f"{shell.shell_id} claims component {shell.shell_index} of a mesh with "
                f"{len(components)} component(s)"
            )
        if tuple(shell.face_indices) != components[shell.shell_index]:
            raise DiscriminantUnusable(
                f"{shell.shell_id} carries a face set that is not component "
                f"{shell.shell_index} of this mesh. A record pointing at another component would "
                "have a counterfactual remove triangles the rule never selected"
            )
    known = {shell.shell_id: shell for shell in observables}
    for record in records:
        if record.member_id != member_id or record.cell_id != cell_id:
            raise DiscriminantUnusable(
                f"{record.shell_id} belongs to {record.member_id}/{record.cell_id} and was "
                f"offered to the counterfactual for {member_id}/{cell_id}"
            )
        matched = known.get(record.shell_id)
        if matched is None:
            raise DiscriminantUnusable(
                f"{record.shell_id} is labelled here but is not one of this mesh's closed shells"
            )
        if record.observable.shell_index != matched.shell_index:
            raise DiscriminantUnusable(f"{record.shell_id} names two different components")
        if tuple(record.face_indices) != components[record.observable.shell_index]:
            raise DiscriminantUnusable(
                f"{record.shell_id} carries a face set that is not its own recomputed component"
            )
    return components


def counterfactual(
    scene: contract.Scene,
    result: contract.ArmResult,
    observables: Sequence[ObservableShell],
    records: Sequence[ShellRecord],
    selected: Sequence[str],
    *,
    member_id: str,
    cell_id: str,
    split: str,
    reconstruction_digest: str,
) -> CounterfactualEvidence:
    """Remove exactly the selected whole shells and re-measure everything.

    ``observables`` is EVERY closed shell of this mesh and ``selected`` is what
    the rule chose over all of them; ``records`` is the labelled unsupported
    subset and is used only to say what each removed shell turned out to be.
    The identity of every shell is recomputed and checked by
    :func:`check_shell_identity` before anything is pruned.
    """
    components = check_shell_identity(
        result, observables, records, member_id=member_id, cell_id=cell_id
    )
    by_id = {shell.shell_id: shell for shell in observables}
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise DiscriminantUnusable(
            f"the counterfactual for {member_id}/{cell_id} was asked to remove {unknown}, which "
            "are not closed shells of this mesh"
        )
    chosen = [by_id[shell_id] for shell_id in sorted(set(selected))]
    faces: list[int] = []
    for shell in chosen:
        faces.extend(components[shell.shell_index])
    if len(set(faces)) != len(faces):
        raise DiscriminantUnusable(
            f"the shells selected on {member_id}/{cell_id} share a face, so they are not "
            "disjoint shells"
        )
    pruned = prune(result, faces)
    if len(pruned.faces) != len(result.faces) - len(faces):
        raise DiscriminantUnusable(
            f"removing {len(faces)} face(s) from {member_id} left {len(pruned.faces)} of "
            f"{len(result.faces)}"
        )
    labelled = {record.shell_id: record for record in records}
    removed_false = [
        s for s in chosen if s.shell_id in labelled and labelled[s.shell_id].truth.false_surface
    ]
    removed_correct = [
        s for s in chosen if s.shell_id in labelled and not labelled[s.shell_id].truth.false_surface
    ]
    removed_supported = [s for s in chosen if s.shell_id not in labelled]
    before = account(scene, result)
    after = account(scene, pruned)
    gates = {
        "counterfactual_false_faces_fell": (
            after.unsupported_false_shell_faces < before.unsupported_false_shell_faces
        ),
        "counterfactual_correct_faces_held": (
            after.unsupported_correct_shell_faces == before.unsupported_correct_shell_faces
        ),
        "counterfactual_removes_no_supported_faces": not removed_supported,
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
    return CounterfactualEvidence(
        member_id=member_id,
        cell_id=cell_id,
        split=split,
        reconstruction_digest=reconstruction_digest,
        closed_shells=len(observables),
        removed_shell_ids=tuple(shell.shell_id for shell in chosen),
        removed_faces=len(faces),
        removed_false_shells=len(removed_false),
        removed_false_shell_faces=sum(shell.faces for shell in removed_false),
        removed_correct_shells=len(removed_correct),
        removed_correct_shell_faces=sum(shell.faces for shell in removed_correct),
        removed_supported_shells=len(removed_supported),
        removed_supported_shell_faces=sum(shell.faces for shell in removed_supported),
        before=before,
        after=after,
        gates=gates,
    )
