"""Regenerate the surface-oracle corpus. The reviewable recipe behind every committed number.

Run it with::

    .venv/bin/python -m tests.build_surface_oracle_fixtures

This is a **command, not a test**: pytest never runs it, and a run that changes
a fixture is changing evidence. It is the only thing in this repository allowed
to write inside ``tests/fixtures/surface_oracle/cases/``, it writes nowhere
else, and it rewrites that directory whole - every case file and the manifest -
so a case that was deleted from the recipe cannot survive on disk as a file the
manifest no longer names.

What is in here, and what is deliberately not
---------------------------------------------
In here: the object, the views, the perturbations, the parameter grid, the work
ceiling, the named regions, the gates and their limits, and the expectation
declared for every arm of every case. All of it is data written before a run,
which is what makes the corpus an oracle rather than a record of what happened.

Not in here: any measurement. This module renders frames and hashes them; it
never scores an arm, never reads a report and never adjusts a limit to what a
result turned out to be. When reality disagrees with :data:`EXPECTATIONS` the
harness exits 1 and says exactly which arm and which gate - and the answer is
to change the scene, the parameter grid or the declared expectation
deliberately, in this file, with the reason written down.

The object, and why it has four features
-----------------------------------------
One asymmetric object seen from five exact views, carrying **a plane** (the
backing surface), **a curved surface** (a sphere), **a hard depth step** (a box
standing in front of the plane) and **a thin feature** (a pillar about two
pixels wide at the object's own depth). They are together on purpose: every
parameter that smooths one of them damages another, so no cell of the grid can
pass by erasing the geometry it could not represent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Final

from tests import depth_completeness as completeness
from tests import surface_oracle as contract
from tests import surface_oracle_harness as harness

DESTINATION: Final = harness.CORPUS

#: The camera. A short focal length so the whole object fits a small window:
#: every denominator in this corpus is that window, and a larger one would cost
#: run time without adding a failure mode.
INTRINSICS: Final[Mapping[str, object]] = {
    "fx": 120.0,
    "fy": 120.0,
    "cx": 320.0,
    "cy": 200.0,
    "width": 640,
    "height": 400,
}
DEPTH_SCALE: Final = 0.005
WINDOW: Final[Mapping[str, object]] = {
    "u0": 288,
    "v0": 180,
    "width": 64,
    "height": 40,
    "sample_count": 64 * 40,
}

#: The object. Four features, one scene, and nothing decorative.
SHAPES: Final[tuple[Mapping[str, object], ...]] = (
    # The backing plane at z = 12: the flat surface every roughness number is
    # measured on, and the thing a bridged depth step is bridged TO.
    {"kind": "plane", "normal": [0.0, 0.0, -1.0], "offset": -12.0},
    # The curved surface. A staircase fill or a coarse voxel roughens a sphere
    # in a way it cannot roughen a plane.
    {"kind": "sphere", "centre": [-1.2, 0.0, 9.2], "radius": 0.9},
    # The hard depth step: a box four units in front of the plane, so the
    # silhouette is a real discontinuity rather than a gradient.
    {"kind": "box", "half_extents": [0.8, 1.0, 1.0], "translation": [1.3, 0.0, 9.0]},
    # The thin feature: a pillar 0.2 units across, about 2.8 pixels wide at its
    # own depth. A voxel of 0.25 cannot hold it, which is the point.
    {"kind": "box", "half_extents": [0.1, 1.0, 0.1], "translation": [0.0, 0.0, 8.6]},
)

#: Five views, rotating the scene about a vertical axis through the object.
#: Five rather than three because a pose error has to be visible as a
#: DISAGREEMENT between views, and three views give any error two chances to
#: look like a small consistent offset.
VIEW_DEGREES: Final[tuple[float, ...]] = (-12.0, -6.0, 0.0, 6.0, 12.0)
VIEW_AXIS: Final[tuple[float, float, float]] = (0.0, 1.0, 0.0)
VIEW_THROUGH: Final[tuple[float, float, float]] = (0.0, 0.0, 9.3)

#: The fixed volume both arms are clipped to and every cell fuses into. It is
#: the observed space of the whole corpus, stated once.
VOLUME: Final[Mapping[str, object]] = {
    "minimum": [-2.6, -1.7, 7.6],
    "maximum": [2.6, 1.7, 12.5],
    "min_depth": 4.0,
    "max_depth": 16.0,
}

#: Every named region, in world coordinates, chosen from the object's own
#: construction and never from a result.
REGIONS: Final[Mapping[str, object]] = {
    # The backing plane, inset from the volume so the fit is not dominated by
    # the volume's own clipped border.
    "plane": {"minimum": [-2.4, -1.5, 11.6], "maximum": [2.4, 1.5, 12.4]},
    # Free space beside the box's right silhouette. The box stops at x = 2.1
    # and the plane is four units behind it, so nothing the scene contains is
    # in here - a point here is a surface bridging the step.
    "protected": {"minimum": [2.25, -0.9, 8.3], "maximum": [2.5, 0.9, 11.5]},
    # The step itself: the right edge of the box front face and its side.
    "edge_band": {"minimum": [1.7, -1.0, 7.8], "maximum": [2.15, 1.0, 10.2]},
    # The pillar's front face.
    "thin_feature": {"minimum": [-0.3, -1.1, 8.35], "maximum": [0.3, 1.1, 8.75]},
    # The object without its backing plane, for named extents.
    "object_region": {"minimum": [-2.6, -1.7, 7.6], "maximum": [2.6, 1.7, 11.0]},
    "coverage_tolerance": 0.15,
    "truth_pitch": 0.12,
    # Sub-rays per pixel per axis for the DENSE ANALYTIC reference. Two gives
    # four exact intersections per pixel over the 64x40 window and five views -
    # 51200 closed-form reference points - which is dense against a 0.15
    # coverage tolerance without making the reference cost more than the
    # fusions it judges.
    "analytic_supersample": 2,
    # Sub-triangles per edge for the interior lattice. Three gives nine
    # strictly interior samples per emitted triangle - one at the centroid of
    # each of the nine EQUAL-AREA sub-triangles, upright and inverted - so a
    # triangle spanning the 0.25 wide protected volume is sampled several times
    # across it rather than only at its centroid, and the fraction of a face's
    # samples that land in the gap is an unbiased estimate of the fraction of
    # its area that does.
    "triangle_sample_order": 3,
    # The ceiling on that sampling, owned here and stated before any result.
    # The shipped-policy arm emits up to about seventy-five thousand triangles
    # on this scene, so nine samples each is about seven hundred thousand; the
    # ceiling is that with room and is a refusal rather than a signal to sample
    # less. It moved from 600 000 with the lattice itself: the equal-area
    # lattice of order n has n**2 points where the upright-only one had
    # n(n+1)/2, which is half as many again at order three. No result was
    # consulted to set it.
    "max_triangle_samples": 1_000_000,
    "linkage_radius": 0.3,
    "isolated_max_points": 8,
    "bounds_margin": 0.05,
}

#: The SHIPPED resolution policy, which is not a grid cell and never becomes
#: one. Its voxel size is absent on purpose: ``choose_tsdf_resolution`` derives
#: it per scene from that scene's own measured points, and
#: :func:`tests.surface_oracle.check_appliance_policy` refuses this record if
#: any constant here has drifted from the one
#: :mod:`scanner3d.reconstruction_export` exports today.
#:
#: The budget is the desktop Reconstruct action's own
#: ``scanner3d.gui.RECONSTRUCT_VOXEL_BUDGET``, written out as ``128 ** 3``
#: here so the fixture a reviewer reads carries a number rather than an import.
#: It is not taken on trust:
#: :func:`tests.surface_oracle.check_appliance_policy` imports that constant
#: and refuses this record if the two disagree, so a budget that drifted from
#: the window's would stop the corpus instead of quietly renaming a fourth
#: voxel size "the appliance". On this scene it derives a voxel of about
#: 0.0352, which is roughly a third of the finest cell the grid declares - the
#: reason this arm exists at all.
APPLIANCE: Final[Mapping[str, object]] = {
    "policy": (
        "scanner3d.reconstruction_export.choose_tsdf_resolution over the case's own measured "
        "points at the desktop Reconstruct action's voxel budget (scanner3d.gui."
        "RECONSTRUCT_VOXEL_BUDGET, 128 ** 3), fused and extracted with the constants "
        "scanner3d.reconstruction_export ships"
    ),
    "voxel_budget": 128**3,
    "truncation_voxels": 3.0,
    "min_weight": 1.0,
    "weight_policy": "uniform",
    "backend": "numpy",
    # About 2.01 million voxels times five views on this scene. The ceiling is
    # that with room, and it is separate from the corpus ceiling because the
    # appliance grid is two orders of magnitude finer than the declared one.
    "work_budget": 14_000_000,
}

#: The predeclared parameter grid, owned by the corpus manifest. Six cells that
#: vary one thing at a time from the first: the extraction minimum weight, the
#: truncation ratio, the weight policy, a coarser voxel and a finer one.
GRID: Final[tuple[Mapping[str, object], ...]] = (
    {
        "cell_id": "v15-t3-w1-uniform",
        "voxel_size": 0.15,
        "truncation_voxels": 3.0,
        "min_weight": 1.0,
        "weight_policy": "uniform",
    },
    {
        "cell_id": "v15-t3-w2-uniform",
        "voxel_size": 0.15,
        "truncation_voxels": 3.0,
        "min_weight": 2.0,
        "weight_policy": "uniform",
    },
    {
        "cell_id": "v15-t2-w1-uniform",
        "voxel_size": 0.15,
        "truncation_voxels": 2.0,
        "min_weight": 1.0,
        "weight_policy": "uniform",
    },
    {
        "cell_id": "v15-t3-w1-ramped",
        "voxel_size": 0.15,
        "truncation_voxels": 3.0,
        "min_weight": 1.0,
        "weight_policy": "ramped_behind",
    },
    {
        "cell_id": "v25-t3-w1-uniform",
        "voxel_size": 0.25,
        "truncation_voxels": 3.0,
        "min_weight": 1.0,
        "weight_policy": "uniform",
    },
    {
        "cell_id": "v10-t4-w1-uniform",
        "voxel_size": 0.10,
        "truncation_voxels": 4.0,
        "min_weight": 1.0,
        "weight_policy": "uniform",
    },
)

#: The work ceiling, in voxel-view units, owned by the corpus. It is the finest
#: cell's own cost with a margin and nothing more: a ceiling nothing could
#: exceed would not be a ceiling, and one tuned to a result would not be
#: evidence. 86632 voxels x 5 views = 433160 for v10-t4-w1-uniform.
WORK_BUDGET: Final = 500_000

#: The perturbations, one case each, plus the exact control. Magnitudes are in
#: raw codes (1 code = 0.005 scene units) and in scene units, and are chosen to
#: be visible against a 0.15 voxel without swamping the object.
CASES: Final[tuple[Mapping[str, object], ...]] = (
    {
        "case_id": "exact-control",
        "what": "the object from five exact views with no perturbation at all",
        "why": (
            "The control every other case is measured against, and the only case that can show "
            "the oracle's own floor. On exact input the dots and the fused surface disagree "
            "only by what fusion and extraction themselves do, so any difference here is the "
            "method rather than the input."
        ),
        "perturbation": {"label": contract.EXACT},
        "control_case": None,
    },
    {
        "case_id": "depth-noise",
        "what": "bounded zero-mean depth noise of plus or minus 20 codes (0.1 units)",
        "why": (
            "The noise a TSDF is supposed to be good at: zero-mean, bounded, and averaged over "
            "five views. If the fused surface is not smoother than the dots here, averaging is "
            "not what is making the mesh lumpy."
        ),
        "perturbation": {"label": contract.DEPTH_NOISE, "depth_noise_codes": 20},
        "control_case": "exact-control",
    },
    {
        "case_id": "depth-outliers",
        "what": "a localized outlier of plus or minus 60 codes (0.3 units) on every 37th return",
        "why": (
            "The noise a running weighted mean is NOT good at. An outlier is not cancelled by "
            "averaging; it drags the zero crossing of every voxel it touches, and the isolated "
            "lumps that result are what a person calls blobby."
        ),
        "perturbation": {
            "label": contract.DEPTH_OUTLIERS,
            "outlier_period": 37,
            "outlier_codes": 60,
        },
        "control_case": "exact-control",
    },
    {
        "case_id": "no-return-bands",
        "what": "two of every seven columns never return, walking three columns per view",
        "why": (
            "Structured loss rather than speckle, and complementary across views: every column "
            "is measured by some view. It separates 'the surface is missing because nothing saw "
            "it' from 'the surface is missing because fusion would not commit to it'."
        ),
        "perturbation": {
            "label": contract.NO_RETURN_BANDS,
            "band_period": 7,
            "band_width": 2,
            "band_shift": 3,
        },
        "control_case": "exact-control",
    },
    {
        "case_id": "pose-translation",
        "what": "a 0.05-unit lateral pose bias with a 0.05-unit per-view jitter on top",
        "why": (
            "The frames are exact and only the pose the fusion is GIVEN is wrong, which is "
            "precisely a tracking error. A third of a voxel of disagreement between views is "
            "enough to see whether the surface thickens or breaks."
        ),
        "perturbation": {
            "label": contract.POSE_TRANSLATION,
            "translation_bias": [0.05, 0.0, 0.0],
            "translation_jitter": 0.05,
        },
        "control_case": "exact-control",
    },
    {
        "case_id": "pose-rotation",
        "what": "a 0.3-degree rotational pose bias with a 0.3-degree per-view jitter",
        "why": (
            "The same failure as the case above with a lever arm: a third of a degree is nothing "
            "at the origin and about 0.06 units out at the object, so it tests whether the "
            "damage scales with distance the way a rotation error must."
        ),
        "perturbation": {
            "label": contract.POSE_ROTATION,
            "rotation_bias_degrees": 0.3,
            "rotation_jitter_degrees": 0.3,
            "rotation_axis": [0.0, 1.0, 0.0],
        },
        "control_case": "exact-control",
    },
    {
        "case_id": "combined-realistic",
        "what": "depth noise, structured no-return and both pose errors together",
        "why": (
            "The interaction case, and the only one that resembles a real capture. It exists so "
            "the report can say whether the causes add up or whether one of them dominates the "
            "rest; a corpus of single perturbations alone could not tell those apart."
        ),
        "perturbation": {
            "label": contract.COMBINED,
            "depth_noise_codes": 20,
            "band_period": 7,
            "band_width": 2,
            "band_shift": 3,
            "translation_bias": [0.05, 0.0, 0.0],
            "translation_jitter": 0.05,
            "rotation_bias_degrees": 0.3,
            "rotation_jitter_degrees": 0.3,
        },
        "control_case": "exact-control",
    },
)

#: The gates, identical across every case on purpose. A case with its own bars
#: could not be compared with its control, and "the perturbed case passed" would
#: mean "the perturbed case was judged more gently".
GATES: Final[tuple[Mapping[str, object], ...]] = (
    {
        "name": "analytic_accuracy_p95",
        "measurement": "analytic",
        "path": ["distances", "p95"],
        "comparison": "<=",
        "limit": 0.15,
        "unit": "synthetic depth units",
        "context": (
            "Nineteen points in twenty must sit within one coverage tolerance of the OBJECT, in "
            "closed form. This is the accuracy gate, and it is deliberately not measured "
            "against the sensor's own returns: a vertex correctly interpolated between two "
            "camera rays is on the object and scores zero here, which round two of this corpus "
            "reported as an extraction defect because its reference was the rays."
        ),
    },
    {
        "name": "analytic_surface_coverage",
        "measurement": "analytic",
        "path": ["reference_coverage_fraction"],
        "comparison": ">=",
        "limit": 0.8,
        "unit": "fraction of 1",
        "context": (
            "Four fifths of a dense exact sampling of the object's own surface must have "
            "geometry within the coverage tolerance. Coverage bought by inventing surface fails "
            "the bridging, topology and bounds gates beside this one."
        ),
    },
    {
        "name": "support_coverage",
        "measurement": "support",
        "path": ["reference_to_observed", "coverage_fraction"],
        "comparison": ">=",
        "limit": 0.8,
        "unit": "fraction of 1",
        "context": (
            "The same question against MEASUREMENT SUPPORT rather than truth: four fifths of "
            "the ideal returns the five views produced must have geometry near them. It is not "
            "an accuracy gate and is never read as one."
        ),
    },
    {
        "name": "supported_surface",
        "measurement": "support",
        "path": ["observed_to_reference", "coverage_fraction"],
        "comparison": ">=",
        "limit": 0.95,
        "unit": "fraction of 1",
        "context": (
            "The other direction of the same agreement, and the one that catches surface no "
            "measurement supports: at most one point in twenty may sit further than the coverage "
            "tolerance from an ideal return. It says where the sensor SAMPLED, not where the "
            "object is: a point on a face no view could see fails this and passes "
            "analytic_accuracy_p95, and both of those are true at once."
        ),
    },
    {
        "name": "triangle_interior_accuracy_p95",
        "measurement": "surface_samples",
        "path": ["distances", "p95"],
        "comparison": "<=",
        "limit": 0.15,
        "unit": "synthetic depth units",
        "context": (
            "The same closed-form accuracy, taken INSIDE the emitted triangles rather than at "
            "their corners. Marching-cubes vertices sit on the field's own zero crossings, so a "
            "vertex-only accuracy gate cannot see a triangle whose corners are fine and whose "
            "middle is not. Inapplicable to the raw-points arm, which has no faces."
        ),
    },
    {
        "name": "no_bridging_triangles",
        "measurement": "surface_samples",
        "path": ["bridging_faces"],
        "comparison": "<=",
        "limit": 0.0,
        "unit": "points",
        "context": (
            "The protected volume beside the depth step is empty in the scene, and this counts "
            "the emitted TRIANGLES whose interiors enter it. It is strictly stronger than "
            "no_edge_bridging beside it, which counts only vertices there: a triangle can span "
            "the whole gap with all three of its corners outside."
        ),
    },
    {
        "name": "no_false_closed_shells",
        "measurement": "topology",
        "path": ["unsupported_inaccurate_closed_shells"],
        "comparison": "<=",
        "limit": 0.0,
        "unit": "components",
        "context": (
            "A shell every edge of which is used exactly twice - a sealed, watertight blob - "
            "with no ideal return within the coverage tolerance of any of its vertices AND no "
            "part of it - corner or sampled triangle interior - within that tolerance of the "
            "object itself. Both halves are required. Unsupported alone is not a defect: a "
            "sealed shell on a face no view could see is unsupported measurement and correct "
            "geometry at once, and this corpus measures six of exactly those. What this gate "
            "counts is false surface, and neither a distance summary nor a point-linkage "
            "component count can distinguish it from a legitimately separate body."
        ),
    },
    {
        "name": "manifold_surface",
        "measurement": "topology",
        "path": ["nonmanifold_edge_count"],
        "comparison": "<=",
        "limit": 0.0,
        "unit": "components",
        "context": (
            "An edge used by three or more faces. The shipped extractor documents that it "
            "emits shared vertices and paired edges, so this is a check on that promise rather "
            "than a tolerance - and a mesh that failed it would make every closure count below "
            "it meaningless."
        ),
    },
    {
        "name": "plane_roughness_p95",
        "measurement": "plane",
        "path": ["residuals", "p95"],
        "comparison": "<=",
        "limit": 0.06,
        "unit": "synthetic depth units",
        "context": (
            "The backing plane really is flat. A residual above 0.06 units - under half a 0.15 "
            "voxel - is the reconstruction's own lumpiness on a surface with none."
        ),
    },
    {
        "name": "no_edge_bridging",
        "measurement": "edge",
        "path": ["bridged_points"],
        "comparison": "<=",
        "limit": 0.0,
        "unit": "points",
        "context": (
            "The protected volume beside the depth step is empty in the scene. Any geometry "
            "there is a surface that bridged an edge that is really there, and the limit is "
            "absolute rather than a tolerance."
        ),
    },
    {
        "name": "edge_retained",
        "measurement": "edge",
        "path": ["edge_coverage"],
        "comparison": ">=",
        "limit": 0.85,
        "unit": "fraction of 1",
        "context": (
            "The opposite failure to bridging: a truncation or a minimum weight that refuses to "
            "commit at the step erodes it. Both gates are on, so neither can be traded for the "
            "other."
        ),
    },
    {
        "name": "thin_feature_retained",
        "measurement": "edge",
        "path": ["thin_feature_coverage"],
        "comparison": ">=",
        "limit": 0.5,
        "unit": "fraction of 1",
        "context": (
            "Half of the pillar has to survive. This is the gate that stops a coarse voxel "
            "passing by erasing the geometry it could not represent, which would otherwise read "
            "as a smoother, cleaner surface."
        ),
    },
    {
        "name": "dimension_error",
        "measurement": "dimensions",
        "path": ["absolute", "worst"],
        "comparison": "<=",
        "limit": 0.2,
        "unit": "synthetic depth units",
        "context": (
            "The object's three named extents against the exact truth's. An arm that shrank the "
            "object inward while smoothing it moves this and nothing else."
        ),
    },
    {
        "name": "no_isolated_lumps",
        "measurement": "components",
        "path": ["isolated_components"],
        "comparison": "<=",
        "limit": 0.0,
        "unit": "components",
        "context": (
            "A component of at most eight points, standing apart from the body at the linkage "
            "radius, is a blob. This is the measurement the original complaint is actually "
            "about, and every distance summary can stay respectable while it fails."
        ),
    },
    {
        "name": "body_count",
        "measurement": "components",
        "path": ["component_count"],
        "comparison": "<=",
        "limit": 6.0,
        "unit": "components",
        "context": (
            "The scene genuinely contains separate bodies - the backing plane, the sphere, the "
            "box and the pillar - so the bar is not one connected surface but SIX, which is one "
            "more than the exact control's own five. A reconstruction that shattered into more "
            "pieces than the scene has is lumpy however accurate each piece is."
        ),
    },
    {
        "name": "within_budget",
        "measurement": "field",
        "path": ["within_budget"],
        "comparison": "is",
        "limit": True,
        "context": (
            "The voxel-view spend against the ceiling the corpus manifest owns. Inapplicable to "
            "the raw-points arm, which allocates no volume - recorded as inapplicable rather "
            "than as passing."
        ),
    },
    {
        "name": "points_inside_volume",
        "measurement": "geometry",
        "path": ["outside_volume"],
        "comparison": "<=",
        "limit": 0.0,
        "unit": "points",
        "context": (
            "Observed space is the fixed volume both arms share. Geometry outside it was never "
            "measured by anything."
        ),
    },
)

#: The perturbation labels whose RAW FRAMES differ from the truth frames, and
#: the labels whose GIVEN POSES differ from the true ones. Two corruption
#: controls are only detectable on one of these lists: fusing the truth frames
#: is byte-identical to fusing the measured frames when the case has no depth
#: perturbation, and fusing with the true poses is identical to fusing with the
#: given ones when the case has no pose perturbation. Declaring those controls
#: on a case that cannot tell the difference would be declaring a refusal the
#: reader can never earn, which is a failing corpus rather than a strict one.
#: :data:`tests.surface_oracle_harness.CONTROL_REFUSALS` still requires every
#: refusal to be earned SOMEWHERE, so this narrows where a control runs and
#: cannot remove one.
DEPTH_PERTURBED: Final[tuple[str, ...]] = (
    contract.DEPTH_NOISE,
    contract.DEPTH_OUTLIERS,
    contract.NO_RETURN_BANDS,
    contract.COMBINED,
)
POSE_PERTURBED: Final[tuple[str, ...]] = (
    contract.POSE_TRANSLATION,
    contract.POSE_ROTATION,
    contract.COMBINED,
)

#: Every arm each case runs, in fixture order. The six fused cells are required
#: by the harness; the raw arm is the comparison the corpus exists to make; the
#: rest are the corruption controls, each of which must earn one named refusal.
#: The fourth field names the perturbation labels the control is detectable on,
#: or ``None`` for the controls every case can run.
CONTROL_ARMS: Final[tuple[tuple[str, str, str, tuple[str, ...] | None], ...]] = (
    (
        "fused_from_truth_frames:v15-t3-w1-uniform",
        "raw_frame_edited",
        (
            "Fuses the exact truth frames and presents the surface as the measured one. Its "
            "geometry really is better than any honest cell's, which is why an oracle that read "
            "only the numbers would report it as the fix."
        ),
        DEPTH_PERTURBED,
    ),
    (
        "fused_dropping_a_view:v15-t3-w1-uniform",
        "views_omitted",
        (
            "Leaves out the last view. Dropping the frame that disagrees removes the "
            "disagreement, and removes what that frame alone saw with it."
        ),
        None,
    ),
    (
        "fused_with_true_poses:v15-t3-w1-uniform",
        "pose_substituted",
        (
            "Fuses with the poses the scene kept back. No reconstruction has them, so a result "
            "that used them is a counterfactual rather than an arm - the harness publishes that "
            "counterfactual itself, ungated, in the attribution."
        ),
        POSE_PERTURBED,
    ),
    (
        "fused_with_edited_surface:v15-t3-w1-uniform",
        "surface_not_derived",
        (
            "Fuses honestly and then deletes the vertices standing in free space, or - when the "
            "honest surface put none there - the last vertex the extractor emitted. The most "
            "plausible forgery here, because it is exactly what somebody would do by hand after "
            "looking at the mesh."
        ),
        None,
    ),
    (
        "fused_with_bridging_triangle:v15-t3-w1-uniform",
        "surface_not_derived",
        (
            "Keeps every honest vertex exactly where the extractor put it and adds ONE triangle "
            "spanning the protected volume. The point list is byte-identical to the honest "
            "arm's, so every vertex-based measurement in this corpus agrees with it to the last "
            "bit; only the faces differ. It is the reason the surface is measured as a surface."
        ),
        None,
    ),
    (
        "raw_points_dropping_hard_samples",
        "sample_count_mismatch",
        (
            "The raw arm's own way of cheating: drop the measurements at the depth step, where "
            "the dots are noisiest, and report a cleaner cloud over a smaller denominator."
        ),
        None,
    ),
    (
        "fused_overclaiming_work:v15-t3-w1-uniform",
        "claim_not_supported",
        (
            "An honest surface with half its voxel-view spend declared. The accounting is how a "
            "coarse grid would be sold as a cheap one."
        ),
        None,
    ),
    (
        "fused_claiming_the_unmeasurable:v15-t3-w1-uniform",
        "claim_unknown",
        (
            "An honest surface that publishes a 'surface_accuracy_target' of its own. A "
            "candidate that can author a bar is not being measured by this corpus."
        ),
        None,
    ),
)

#: What each scored arm must do, as ``{case: {candidate: [failing gate, ...]}}``.
#: An arm that is absent from a case's mapping must PASS every gate that applies
#: to it. Declared here, and reconciled deliberately: the harness exits 1 and
#: names the arm and the gate whenever reality and this table disagree, which is
#: the only mechanism that makes the table mean anything.
#:
#: Read the table itself as the corpus's finding, because that is what it is.
#: Round three moved the finding, and it moved because the REFERENCE moved:
#:
#: * ``supported_surface`` still fails on every fused arm of every case. It is
#:   proximity to the ideal RETURNS, and a marching-cubes vertex interpolated
#:   between two camera rays is far from all of them while being exactly on the
#:   object. Round two called that number surface accuracy and concluded the
#:   extractor had produced an unsupported shell. It had not;
#: * ``analytic_accuracy_p95`` and ``triangle_interior_accuracy_p95`` are the
#:   closed-form distance to the object itself, at the vertices and inside the
#:   triangles. Both pass on every cell of every case except the coarsest, 0.25
#:   voxel one - and at the shipped resolution the exact control scores a
#:   triangle-interior p95 of 0.016 with every sample inside tolerance;
#: * the ablation says why. ``ablation.extraction_p95_delta`` is between
#:   -0.0265 and +0.0091 on every fused arm in the corpus - a fraction of one
#:   voxel at every cell that produced it - and
#:   ``surface_points_without_field_support_fraction`` is 0.0 on all of them, so
#:   the extractor reproduces the field it is handed. The remaining error is
#:   fusion's - discretisation, pose and depth - not marching cubes'.
#:
#: ``no_false_closed_shells`` is the gate round three added that could not have
#: been written from vertices, and it fails where nothing else does: under
#: depth outliers, at v10 and again at the appliance resolution, a watertight
#: shell stands where no view measured anything AND off the object - worst
#: analytic distance 0.3042 and 0.3030 against a 0.15 tolerance. Every
#: point-based lump gate passes on those same arms.
#:
#: It is deliberately not a gate on ``unsupported_closed_shells``, and the
#: difference is the whole of what round three learned. The combined case's
#: appliance arm seals SIX unsupported shells and every one of them is
#: analytically correct: 100 faces, worst analytic distance 0.1247, inside the
#: same 0.15 tolerance. Nothing measured them because nothing could - they
#: stand where no view had a ray - and a rule that pruned unsupported shells
#: would delete all six pieces of correct surface to remove no false surface at
#: all. Both numbers are published on every arm so the trade is visible rather
#: than assumed.
#:
#: ``points_inside_volume`` fails on three appliance arms for a boundary reason
#: rather than a geometric one, and it is recorded rather than waived:
#: :attr:`scanner3d.tsdf.TsdfConfig.shape` rounds the voxel count UP, so on an
#: axis whose extent is not a whole multiple of the voxel size the outermost
#: voxel centre sits just past the declared box - at most half a voxel, which is
#: 0.018 units here and well inside the fixture's own 0.05 bounds margin. The
#: backing plane reaches that far in y, so a few hundred of twenty thousand
#: vertices land there. It is not invented geometry and the docs say so.
#:
#: NO GATE LIMIT WAS MOVED to produce this table. The bars in :data:`GATES` are
#: the ones this round was authored with, and the two limits that carry over
#: from round two - 0.15 on accuracy, 0.95 on supported surface - are unchanged.
#: What changed is which reference the accuracy gate reads.
EXPECTATIONS: Final[Mapping[str, Mapping[str, tuple[str, ...]]]] = {
    # The floor of the method, and the correction of round two's headline.
    # Every cell fails supported_surface - proximity to the RETURNS - while the
    # closed-form analytic gates pass everywhere but the coarsest cell. The dots
    # pass every gate that applies to them and are absent from this table.
    "exact-control": {
        "fused:v15-t3-w1-uniform": ("dimension_error", "supported_surface"),
        "fused:v15-t3-w2-uniform": ("dimension_error", "supported_surface"),
        "fused:v15-t2-w1-uniform": ("dimension_error", "no_isolated_lumps", "supported_surface"),
        "fused:v15-t3-w1-ramped": ("dimension_error", "supported_surface"),
        "fused:v25-t3-w1-uniform": (
            "analytic_accuracy_p95",
            "dimension_error",
            "supported_surface",
            "triangle_interior_accuracy_p95",
        ),
        "fused:v10-t4-w1-uniform": ("supported_surface",),
        "fused_at_appliance_resolution": ("dimension_error", "points_inside_volume"),
    },
    # Bounded zero-mean noise. Averaging over five views does help: the fused
    # analytic accuracy beats the dots' own at v10 and at the appliance
    # resolution. What it does not fix is the plane, which stays rough on every
    # arm including the dots.
    "depth-noise": {
        "raw_points": ("plane_roughness_p95",),
        "fused:v15-t3-w1-uniform": ("dimension_error", "plane_roughness_p95", "supported_surface"),
        "fused:v15-t3-w2-uniform": ("dimension_error", "plane_roughness_p95", "supported_surface"),
        "fused:v15-t2-w1-uniform": ("dimension_error", "plane_roughness_p95", "supported_surface"),
        "fused:v15-t3-w1-ramped": ("dimension_error", "plane_roughness_p95", "supported_surface"),
        "fused:v25-t3-w1-uniform": (
            "analytic_accuracy_p95",
            "dimension_error",
            "no_bridging_triangles",
            "no_edge_bridging",
            "no_isolated_lumps",
            "plane_roughness_p95",
            "supported_surface",
            "triangle_interior_accuracy_p95",
        ),
        "fused:v10-t4-w1-uniform": ("plane_roughness_p95", "supported_surface"),
        "fused_at_appliance_resolution": ("plane_roughness_p95", "points_inside_volume"),
    },
    # The noise a running weighted mean cannot cancel, and the first case where
    # the DOTS themselves fail. It is also where the new topology gate earns its
    # place, and the only case in the corpus that does: v10 and the appliance
    # arm each seal one shell that nothing measured AND that is not on the
    # object - worst analytic distance 0.3042 and 0.3030 - while passing every
    # point-linkage lump gate beside it.
    "depth-outliers": {
        "raw_points": ("body_count", "dimension_error", "no_isolated_lumps", "supported_surface"),
        "fused:v15-t3-w1-uniform": (
            "dimension_error",
            "no_isolated_lumps",
            "plane_roughness_p95",
            "supported_surface",
        ),
        "fused:v15-t3-w2-uniform": ("dimension_error", "plane_roughness_p95", "supported_surface"),
        "fused:v15-t2-w1-uniform": (
            "dimension_error",
            "no_isolated_lumps",
            "plane_roughness_p95",
            "supported_surface",
        ),
        "fused:v15-t3-w1-ramped": ("dimension_error", "plane_roughness_p95", "supported_surface"),
        "fused:v25-t3-w1-uniform": (
            "analytic_accuracy_p95",
            "dimension_error",
            "supported_surface",
            "triangle_interior_accuracy_p95",
        ),
        "fused:v10-t4-w1-uniform": (
            "no_false_closed_shells",
            "plane_roughness_p95",
            "supported_surface",
        ),
        "fused_at_appliance_resolution": (
            "no_false_closed_shells",
            "plane_roughness_p95",
            "supported_surface",
        ),
    },
    # Complementary structured loss. Every column is measured by some view, and
    # the appliance arm recovers all of it: one dimension error and nothing else.
    "no-return-bands": {
        "fused:v15-t3-w1-uniform": ("dimension_error", "no_isolated_lumps", "supported_surface"),
        "fused:v15-t3-w2-uniform": (
            "dimension_error",
            "no_isolated_lumps",
            "support_coverage",
            "supported_surface",
        ),
        "fused:v15-t2-w1-uniform": (
            "body_count",
            "dimension_error",
            "no_isolated_lumps",
            "supported_surface",
        ),
        "fused:v15-t3-w1-ramped": (
            "body_count",
            "dimension_error",
            "no_isolated_lumps",
            "support_coverage",
            "supported_surface",
        ),
        "fused:v25-t3-w1-uniform": (
            "analytic_accuracy_p95",
            "dimension_error",
            "no_isolated_lumps",
            "support_coverage",
            "supported_surface",
            "triangle_interior_accuracy_p95",
        ),
        "fused:v10-t4-w1-uniform": ("supported_surface",),
        "fused_at_appliance_resolution": ("dimension_error",),
    },
    # Pose error with exact frames. The bridging v10 and v25 show here is
    # geometry pulled across the depth step by disagreeing views, and it is the
    # one place a coarse grid bridges what the appliance resolution does not.
    "pose-translation": {
        "fused:v15-t3-w1-uniform": ("dimension_error", "supported_surface"),
        "fused:v15-t3-w2-uniform": ("dimension_error", "no_isolated_lumps", "supported_surface"),
        "fused:v15-t2-w1-uniform": ("dimension_error", "supported_surface"),
        "fused:v15-t3-w1-ramped": ("dimension_error", "no_isolated_lumps", "supported_surface"),
        "fused:v25-t3-w1-uniform": (
            "analytic_accuracy_p95",
            "dimension_error",
            "no_bridging_triangles",
            "no_edge_bridging",
            "supported_surface",
            "triangle_interior_accuracy_p95",
        ),
        "fused:v10-t4-w1-uniform": (
            "no_bridging_triangles",
            "no_edge_bridging",
            "supported_surface",
        ),
        "fused_at_appliance_resolution": ("dimension_error", "points_inside_volume"),
    },
    # The same failure with a lever arm, and the same shape of result.
    "pose-rotation": {
        "fused:v15-t3-w1-uniform": ("dimension_error", "supported_surface"),
        "fused:v15-t3-w2-uniform": ("supported_surface",),
        "fused:v15-t2-w1-uniform": ("supported_surface",),
        "fused:v15-t3-w1-ramped": ("dimension_error", "supported_surface"),
        "fused:v25-t3-w1-uniform": (
            "analytic_accuracy_p95",
            "dimension_error",
            "no_bridging_triangles",
            "no_edge_bridging",
            "support_coverage",
            "supported_surface",
            "triangle_interior_accuracy_p95",
        ),
        "fused:v10-t4-w1-uniform": (
            "no_bridging_triangles",
            "no_edge_bridging",
            "supported_surface",
        ),
        "fused_at_appliance_resolution": ("dimension_error",),
    },
    # The interaction case, and the only one resembling a real capture. Read the
    # appliance arm rather than the grid: at the resolution an owner actually
    # gets, what remains is plane roughness, dimensional error and unsupported
    # measurement - not bridging, not a coarse voxel, and not the extractor.
    # Six sealed shells stand where nothing was measured and NONE of them is
    # false surface: every one is within 0.1247 of the object against a 0.15
    # tolerance, so the false-shell gate passes here and the six are published
    # as the cost a support-only pruning rule would pay on this case.
    "combined-realistic": {
        "raw_points": ("dimension_error", "plane_roughness_p95"),
        "fused:v15-t3-w1-uniform": (
            "analytic_accuracy_p95",
            "dimension_error",
            "no_bridging_triangles",
            "no_edge_bridging",
            "no_isolated_lumps",
            "no_false_closed_shells",
            "plane_roughness_p95",
            "supported_surface",
            "thin_feature_retained",
            "triangle_interior_accuracy_p95",
        ),
        "fused:v15-t3-w2-uniform": (
            "no_isolated_lumps",
            "plane_roughness_p95",
            "support_coverage",
            "supported_surface",
            "thin_feature_retained",
        ),
        "fused:v15-t2-w1-uniform": (
            "dimension_error",
            "no_isolated_lumps",
            "plane_roughness_p95",
            "supported_surface",
        ),
        "fused:v15-t3-w1-ramped": (
            "plane_roughness_p95",
            "support_coverage",
            "supported_surface",
            "thin_feature_retained",
        ),
        "fused:v25-t3-w1-uniform": (
            "analytic_accuracy_p95",
            "analytic_surface_coverage",
            "dimension_error",
            "no_bridging_triangles",
            "no_edge_bridging",
            "no_isolated_lumps",
            "no_false_closed_shells",
            "plane_roughness_p95",
            "support_coverage",
            "supported_surface",
            "thin_feature_retained",
            "triangle_interior_accuracy_p95",
        ),
        "fused:v10-t4-w1-uniform": (
            "dimension_error",
            "no_bridging_triangles",
            "no_edge_bridging",
            "plane_roughness_p95",
            "supported_surface",
        ),
        "fused_at_appliance_resolution": (
            "dimension_error",
            "plane_roughness_p95",
            "supported_surface",
        ),
    },
}

#: Gates that never apply to the raw-points arm, because it allocates no volume.
#: The gates that do not apply to the measured dots. The raw arm allocates no
#: volume, and it is a cloud rather than a mesh: it has no faces, so nothing
#: samples its triangle interiors and nothing counts its edges. Recorded as
#: inapplicable rather than as passing, because "there was no surface" and "the
#: surface was fine" are different facts.
RAW_INAPPLICABLE: Final[tuple[str, ...]] = (
    "within_budget",
    "triangle_interior_accuracy_p95",
    "no_bridging_triangles",
    "no_false_closed_shells",
    "manifold_surface",
)

CORPUS_WHAT: Final = (
    "Exact-truth multi-view scenes for comparing the measured point cloud with the raw TSDF "
    "surface extracted from the same frames and the same poses, one perturbation at a time."
)
CORPUS_WHY: Final = (
    "The fused surface looks lumpier and less complete than the measured dots. This corpus "
    "separates raw depth noise, localized outliers, structured no-return loss, translational "
    "pose error, rotational pose error and their interaction as causes of that, against exact "
    "truth and a predeclared parameter grid, so a fix is chosen from evidence rather than by "
    "eye."
)
CORPUS_UNITS: Final = (
    "Lengths are synthetic depth units: raw code times depth scale, exact by construction. No "
    "physical length is implied anywhere in this corpus."
)
GENERATED_BY: Final = "tests/build_surface_oracle_fixtures.py"


def _mapping_of(value: object, what: str) -> Mapping[str, object]:
    """Narrow one declared field to a mapping, or say which field was not one."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} is {type(value).__name__}, not a mapping")
    return {str(key): item for key, item in value.items()}


def _perturbation(declared: Mapping[str, object]) -> dict[str, object]:
    """One perturbation record with every field present, defaulted to no change.

    Every field written out, always, rather than only the ones a case uses: the
    fixture is what a reviewer reads, and a case that says nothing about
    rotation should say ``0.0`` rather than leave a reader to remember the
    default.
    """
    record: dict[str, object] = {
        "label": declared["label"],
        "depth_noise_codes": 0,
        "outlier_period": 0,
        "outlier_codes": 0,
        "band_period": 0,
        "band_width": 0,
        "band_shift": 0,
        "translation_bias": [0.0, 0.0, 0.0],
        "translation_jitter": 0.0,
        "rotation_bias_degrees": 0.0,
        "rotation_jitter_degrees": 0.0,
        "rotation_axis": [0.0, 1.0, 0.0],
    }
    for key, value in declared.items():
        if key not in record:
            raise ValueError(f"{declared['label']} declares unknown perturbation field {key!r}")
        record[key] = value
    return record


def _expectation(
    case_id: str, candidate: str, *, refusal: str | None, inapplicable: Sequence[str]
) -> dict[str, object]:
    if refusal is not None:
        return {
            "outcome": "refused",
            "failed_gates": [],
            "inapplicable_gates": sorted(str(gate["name"]) for gate in GATES),
            "refusal": refusal,
        }
    failed = sorted(EXPECTATIONS.get(case_id, {}).get(candidate, ()))
    return {
        "outcome": "fail" if failed else "pass",
        "failed_gates": failed,
        "inapplicable_gates": sorted(inapplicable),
        "refusal": None,
    }


def _candidates(case_id: str, label: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = [
        {
            "name": "raw_points",
            "what": (
                "The measured dots, unchanged and clipped to the shared volume. The arm the "
                "user says already looks right."
            ),
            "expectation": _expectation(
                case_id, "raw_points", refusal=None, inapplicable=RAW_INAPPLICABLE
            ),
        }
    ]
    for cell in GRID:
        name = f"fused:{cell['cell_id']}"
        out.append(
            {
                "name": name,
                "what": (
                    f"The unchanged raw TSDF and the unchanged extractor at {cell['cell_id']}: "
                    f"voxel {cell['voxel_size']}, truncation {cell['truncation_voxels']} voxels, "
                    f"minimum weight {cell['min_weight']}, {cell['weight_policy']} weighting."
                ),
                "expectation": _expectation(case_id, name, refusal=None, inapplicable=()),
            }
        )
    out.append(
        {
            "name": "fused_at_appliance_resolution",
            "what": (
                "The same two modules at the resolution the SHIPPED export policy chooses for "
                "this scene: choose_tsdf_resolution over this case's own measured points at the "
                "desktop Reconstruct action's voxel budget, with the truncation ratio, minimum "
                "weight and weight policy scanner3d.reconstruction_export exports. Several "
                "times finer than the finest declared cell, and the only arm bound to the mesh "
                "an owner actually gets."
            ),
            "expectation": _expectation(
                case_id, "fused_at_appliance_resolution", refusal=None, inapplicable=()
            ),
        }
    )
    for name, refusal, what, applies_to in CONTROL_ARMS:
        if applies_to is not None and label not in applies_to:
            continue
        out.append(
            {
                "name": name,
                "what": what,
                "expectation": _expectation(case_id, name, refusal=refusal, inapplicable=()),
            }
        )
    return out


def _views() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "pose": {
                "axis": list(VIEW_AXIS),
                "degrees": degrees,
                "through": list(VIEW_THROUGH),
            },
        }
        for index, degrees in enumerate(VIEW_DEGREES)
    ]


def _case_document(case: Mapping[str, object]) -> dict[str, object]:
    case_id = str(case["case_id"])
    declared = _mapping_of(case["perturbation"], f"{case_id} perturbation")
    label = str(declared["label"])
    return {
        "schema_version": harness.FIXTURE_SCHEMA_VERSION,
        "kind": harness.CASE_KIND,
        "case_id": case_id,
        "what": case["what"],
        "why": case["why"],
        "notes": [
            "Synthetic. scanner3d.simulate is not a sensor model, and no conclusion about the "
            "device follows from this case.",
            "Both arms are given byte-identical frames and identical poses. The only thing that "
            "differs between them is what is done with those frames.",
            "The perturbation is applied by tests/surface_oracle.py from this fixture's own "
            "declaration, never by a candidate.",
        ],
        "control_case": case["control_case"],
        "perturbation": _perturbation(declared),
        "scene": {
            "intrinsics": dict(INTRINSICS),
            "depth_scale": DEPTH_SCALE,
            "window": dict(WINDOW),
            "shapes": [dict(shape) for shape in SHAPES],
            "seed": f"surface-oracle:{case_id}",
            "volume": dict(VOLUME),
            "views": _views(),
        },
        "regions": dict(REGIONS),
        "gates": [dict(gate) for gate in GATES],
        "candidates": _candidates(case_id, label),
    }


def _fill_digests(document: dict[str, object]) -> dict[str, object]:
    """Render every view of a built case and write its truth and measured digests.

    The digests are the corpus's immutability: the harness renders the same
    scene and refuses the case if what it produces does not hash to what is
    committed here. Editing a truth frame, a shape, a view or a perturbation
    changes them, which is what makes an altered truth a refusal rather than a
    quietly different benchmark.
    """
    scene = document["scene"]
    assert isinstance(scene, dict)
    what = f"{document['case_id']} (being generated)"
    intrinsics = harness._intrinsics(dict(INTRINSICS), what)
    window = harness._window(dict(WINDOW), what)
    shapes = [harness._shape(dict(shape), what) for shape in SHAPES]
    perturbation = harness._perturbation(_mapping_of(document["perturbation"], what), what)
    seed = str(scene["seed"])
    views = scene["views"]
    assert isinstance(views, list)
    for index, view in enumerate(views):
        assert isinstance(view, dict)
        pose = harness._render_pose(dict(view["pose"]), what)
        rendered = harness._render(shapes, intrinsics=intrinsics, scale=DEPTH_SCALE, pose=pose)
        truth_codes = completeness.codes_of(rendered, window)
        truth_frame = completeness.frame_of_codes(truth_codes, window)
        raw_codes = contract.perturb_codes(
            truth_codes, perturbation, window=window, view=index, seed=f"{seed}:view{index}"
        )
        raw_frame = completeness.frame_of_codes(raw_codes, window)
        view["truth_sha256"] = hashlib.sha256(truth_frame).hexdigest()
        view["raw_sha256"] = hashlib.sha256(raw_frame).hexdigest()
        view["truth_returned"] = sum(1 for code in truth_codes if code != contract.NO_RETURN)
        view["raw_returned"] = sum(1 for code in raw_codes if code != contract.NO_RETURN)
    return document


def build() -> dict[str, str]:
    """Write every case file and the manifest. Returns each file's sha256."""
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for stale in sorted(DESTINATION.iterdir()):
        if stale.is_file():
            stale.unlink()

    entries: list[dict[str, object]] = []
    digests: dict[str, str] = {}
    for case in CASES:
        document = _fill_digests(_case_document(case))
        name = f"{document['case_id']}.json"
        text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
        payload = text.encode("utf-8")
        (DESTINATION / name).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        digests[name] = digest
        perturbation = document["perturbation"]
        assert isinstance(perturbation, dict)
        entries.append(
            {
                "name": name,
                "case_id": document["case_id"],
                "perturbation": perturbation["label"],
                "bytes": len(payload),
                "sha256": digest,
            }
        )

    manifest = {
        "schema_version": harness.FIXTURE_SCHEMA_VERSION,
        "kind": harness.CORPUS_KIND,
        "what": CORPUS_WHAT,
        "why": CORPUS_WHY,
        "units": CORPUS_UNITS,
        "generated_by": GENERATED_BY,
        "grid": [dict(cell) for cell in GRID],
        "work_budget": WORK_BUDGET,
        "appliance": dict(APPLIANCE),
        "selection_rule": harness.SELECTION_RULE,
        "files": sorted(entries, key=lambda entry: str(entry["name"])),
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
