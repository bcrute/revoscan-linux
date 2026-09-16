"""Fit spheres in stationary depth frames and compare them with a caliper-measured ball.

Why this exists
---------------
A sphere is the one reference whose size the calipers can give at every
orientation and whose centre a depth camera can locate without touching an
edge pixel. Fitting the visible cap answers two questions the printed target
never could:

* **Scale.** The fitted diameter against the caliper diameter, and the fitted
  centre spacing of two balls against the caliper spacing, in the axis the two
  balls are laid along. Centre spacing cancels edge dropout and any matte
  coating; the diameter does not.
* **Anisotropy.** An axis-aligned ellipsoid fitted to the same cap shows
  whether lateral (x, y) and depth (z) are scaled alike. A wrong focal length
  or a wrong depth scale leaves a sphere looking like an ellipsoid whose
  lateral and depth semi-axes disagree by exactly that factor.

The tool reads raw ``Y16`` 640x400 frames (``frame_*.raw``) from one or more
capture directories, so it works on ``scanner_timing_probe.py --save-raw``
output and on any of the sealed stationary evidence roots. It writes nothing
inside an input directory; the report goes wherever ``--output`` points.

What it does not do: correct anything. Every fit uses the intrinsics and depth
scale it was given (the factory record by default) and reports what came out.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage, optimize

from scanner3d.depth_scale import SDK_MINI_FALLBACK_SCALE
from scanner3d.intrinsics import IntrinsicsRecord
from scanner3d.simulate import FACTORY_INTRINSICS, intrinsics_of
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

SCHEMA_VERSION: Final = 1
FRAME_GLOB: Final = "frame_*.raw"
NO_RETURN: Final = 0
RESERVED: Final = 0xFFFF

#: Default depth band a stationary reference is expected to sit in.
DEFAULT_BAND_MM: Final = (100.0, 300.0)
#: Returns whose surface normal is further than this from the view direction are
#: rim returns: foreshortened, dropout-prone and the first to carry bias.
DEFAULT_RIM_DEGREES: Final = 60.0
#: Fewest returns a blob needs before it is worth fitting a sphere to.
DEFAULT_MIN_PIXELS: Final = 1500
#: Robust trimming: keep returns within this many robust sigmas of the fit.
TRIM_SIGMAS: Final = 3.0
#: Floor on the robust sigma so a near-perfect fit does not trim to nothing.
TRIM_FLOOR_MM: Final = 0.02
GAUSS_NEWTON_ROUNDS: Final = 30
TRIM_ROUNDS: Final = 4

Floats = NDArray[np.float64]
Codes = NDArray[np.uint16]
Mask = NDArray[np.bool_]


class SphereReferenceError(Exception):
    """Refused input or an unusable fit. Never silently degraded."""


@dataclass(frozen=True, slots=True)
class Blob:
    """A connected patch of in-band returns and how disc-like it is."""

    label: int
    area: int
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def fill(self) -> float:
        return self.area / max(1, (self.x1 - self.x0) * (self.y1 - self.y0))

    @property
    def aspect(self) -> float:
        return (self.x1 - self.x0) / max(1, self.y1 - self.y0)

    @property
    def disc_like(self) -> bool:
        return 0.55 <= self.fill <= 0.95 and 0.6 <= self.aspect <= 1.7

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "area": self.area,
            "bounds_x0_y0_x1_y1": [self.x0, self.y0, self.x1, self.y1],
            "fill": self.fill,
            "aspect": self.aspect,
            "disc_like": self.disc_like,
        }


@dataclass(frozen=True, slots=True)
class SphereFit:
    centre: tuple[float, float, float]
    radius: float
    rms_mm: float
    inliers: int
    rejected_by_residual: int
    rejected_by_rim: int
    cap_half_angle_degrees: float
    #: Statistical 1-sigma from the final normal equations. Not accuracy: a
    #: biased surface fits tightly too.
    sigma_centre: tuple[float, float, float]
    sigma_radius: float

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    def as_dict(self) -> dict[str, object]:
        return {
            "centre_mm": list(self.centre),
            "radius_mm": self.radius,
            "diameter_mm": self.diameter,
            "rms_mm": self.rms_mm,
            "inliers": self.inliers,
            "rejected_by_residual": self.rejected_by_residual,
            "rejected_by_rim": self.rejected_by_rim,
            "cap_half_angle_degrees": self.cap_half_angle_degrees,
            "sigma_centre_mm_statistical": list(self.sigma_centre),
            "sigma_radius_mm_statistical": self.sigma_radius,
        }


@dataclass(frozen=True, slots=True)
class EllipsoidFit:
    centre: tuple[float, float, float]
    semi_axes: tuple[float, float, float]
    rms_mm: float
    sigma_semi_axes: tuple[float, float, float]

    @property
    def x_over_z(self) -> float:
        return self.semi_axes[0] / self.semi_axes[2]

    @property
    def y_over_z(self) -> float:
        return self.semi_axes[1] / self.semi_axes[2]

    def as_dict(self) -> dict[str, object]:
        return {
            "centre_mm": list(self.centre),
            "semi_axes_mm_xyz": list(self.semi_axes),
            "diameters_mm_xyz": [2.0 * s for s in self.semi_axes],
            "x_over_z": self.x_over_z,
            "y_over_z": self.y_over_z,
            "rms_mm": self.rms_mm,
            "sigma_semi_axes_mm_statistical": list(self.sigma_semi_axes),
        }


@dataclass(frozen=True, slots=True)
class BallFrame:
    """One ball in one frame."""

    frame: str
    blob: Blob
    sphere: SphereFit
    ellipsoid: EllipsoidFit | None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "blob": self.blob.as_dict(),
            "sphere": self.sphere.as_dict(),
            "ellipsoid": None if self.ellipsoid is None else self.ellipsoid.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class Options:
    intrinsics: IntrinsicsRecord = FACTORY_INTRINSICS
    depth_scale: float = SDK_MINI_FALLBACK_SCALE
    band_mm: tuple[float, float] = DEFAULT_BAND_MM
    rim_degrees: float = DEFAULT_RIM_DEGREES
    min_pixels: int = DEFAULT_MIN_PIXELS
    balls: int = 1
    rois: tuple[tuple[int, int, int, int], ...] = ()
    ellipsoid: bool = True
    #: Frames to drop from the start of each capture: a timing change settles
    #: over the first dozen or so deliveries and the probe keeps them all.
    skip_frames: int = 0
    max_frames: int | None = None
    #: When known, a fitted diameter this far off refuses the frame: a crescent
    #: of a half-hidden ball or a cap contaminated by its neighbour.
    expected_diameter_mm: float | None = None


#: Tolerance of the expected-diameter check, as a fraction.
DIAMETER_SANITY: Final = 0.03
#: With a known ball, a fit must also cover this fraction of the full cap and fit
#: this tightly, or it is a fold of something else that passed through a sphere.
MIN_CAP_FRACTION: Final = 0.3
MAX_CAP_RMS_MM: Final = 0.08


# --- frames ---------------------------------------------------------------------


def load_codes(path: Path) -> Codes:
    raw = path.read_bytes()
    if len(raw) != EXPECTED_FRAME_BYTES:
        raise SphereReferenceError(
            f"{path} holds {len(raw)} bytes, not one {FRAME_WIDTH}x{FRAME_HEIGHT} Y16 frame "
            f"of {EXPECTED_FRAME_BYTES}"
        )
    return np.frombuffer(raw, dtype="<u2").reshape(FRAME_HEIGHT, FRAME_WIDTH)


def depth_mm(codes: Codes, scale: float) -> Floats:
    """Depth in the scale's units (millimetres for the SDK scale); NaN where no return."""
    valid = (codes != NO_RETURN) & (codes != RESERVED)
    return np.where(valid, codes.astype(np.float64) * scale, np.nan)


def deproject(depth: Floats, mask: Mask, intrinsics: IntrinsicsRecord) -> Floats:
    """Camera-frame points, (N, 3), for the masked pixels, pinhole as the SDK does it."""
    fx, fy = intrinsics.values["fx"], intrinsics.values["fy"]
    cx, cy = intrinsics.values["cx"], intrinsics.values["cy"]
    rows, cols = np.nonzero(mask)
    z = depth[rows, cols]
    x = (cols.astype(np.float64) - cx) * z / fx
    y = (rows.astype(np.float64) - cy) * z / fy
    return np.column_stack([x, y, z])


# --- segmentation -----------------------------------------------------------------


#: A depth change per pixel above this is an occluding edge or a grazing rim,
#: not the body of a ball: at 60 degrees incidence a sphere changes depth by
#: about 0.4 mm per pixel at working distance, an occlusion step by many mm.
STEP_MM_PER_PX: Final = 1.5
#: Erosion applied after the edges are cut, to sever whatever thin bridge is left.
SPLIT_EROSION_PX: Final = 3
#: How far from its core a cap pixel may lie and still belong to it.
CORE_REACH_PX: Final = 6.0

Labels = NDArray[np.int32]


def label_balls(depth: Floats, mask: Mask) -> Labels:
    """Label ``mask`` so that touching or overlapping balls come out separately.

    Two balls in contact overlap in the image: the nearer one hides part of the
    farther one, so their union is one blob with a wide junction. What does
    separate them is the depth: along the nearer ball's occluding contour the
    depth steps and both rims graze, so cutting every pixel with a steep
    depth change, then eroding a little, leaves one core per ball. Every masked
    pixel within reach is then handed the label of its nearest core.
    """
    structure = np.ones((3, 3), dtype=bool)
    z = np.where(mask, depth, np.nan)
    edge = np.zeros(mask.shape, dtype=bool)
    across = np.abs(np.diff(z, axis=1))
    steep = np.isfinite(across) & (across > STEP_MM_PER_PX)
    edge[:, 1:] |= steep
    edge[:, :-1] |= steep
    down = np.abs(np.diff(z, axis=0))
    steep = np.isfinite(down) & (down > STEP_MM_PER_PX)
    edge[1:, :] |= steep
    edge[:-1, :] |= steep
    # Fill enclosed holes (dropouts, specular highlights) so the erosion does
    # not fragment a core around every one of them.
    solid = ndimage.binary_fill_holes(mask) & ~edge
    core = ndimage.binary_erosion(solid, structure=structure, iterations=SPLIT_EROSION_PX)
    labels, count = ndimage.label(core, structure=structure)
    if count == 0:
        return np.zeros(mask.shape, dtype=np.int32)
    nearest = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    distances = np.asarray(nearest[0], dtype=np.float64)
    indices = np.asarray(nearest[1])
    filled = labels[indices[0], indices[1]]
    return np.where(mask & (distances <= CORE_REACH_PX), filled, 0).astype(np.int32)


def blobs_from_labels(labels: Labels, min_pixels: int) -> list[Blob]:
    """The labelled components of at least ``min_pixels``, largest first."""
    count = int(labels.max()) if labels.size else 0
    if count == 0:
        return []
    areas = ndimage.sum_labels(labels > 0, labels, index=np.arange(1, count + 1))
    blobs: list[Blob] = []
    for label, (area, bounds) in enumerate(
        zip(areas, ndimage.find_objects(labels), strict=True), 1
    ):
        if bounds is None or int(area) < min_pixels:
            continue
        rows, cols = bounds
        blobs.append(
            Blob(
                label=label,
                area=int(area),
                x0=int(cols.start),
                y0=int(rows.start),
                x1=int(cols.stop),
                y1=int(rows.stop),
            )
        )
    blobs.sort(key=lambda blob: blob.area, reverse=True)
    return blobs


def find_blobs(depth: Floats, mask: Mask, min_pixels: int) -> list[Blob]:
    """Components of ``mask`` with touching balls split apart, largest first."""
    return blobs_from_labels(label_balls(depth, mask), min_pixels)


def select_balls(
    depth: Floats, mask: Mask, options: Options
) -> tuple[list[tuple[Blob, Mask]], list[Blob]]:
    """The ball masks for one frame, left to right, plus every candidate considered.

    With ``rois`` each region is searched for its largest in-band component.
    Without them, a single ball must be the largest disc-like component; for
    several balls the largest components are taken as they are, because a ball
    half hidden behind its neighbour is a crescent, not a disc, and the sphere
    fit plus the expected-diameter check judge it better than its outline.
    """
    labels = label_balls(depth, mask)
    chosen: list[tuple[Blob, Mask]] = []
    if options.rois:
        candidates: list[Blob] = []
        for x0, y0, x1, y1 in options.rois:
            window = np.zeros_like(labels)
            window[y0:y1, x0:x1] = labels[y0:y1, x0:x1]
            found = blobs_from_labels(window, options.min_pixels)
            candidates.extend(found)
            if not found:
                raise SphereReferenceError(
                    f"region {x0},{y0},{x1},{y1} holds no in-band component of at least "
                    f"{options.min_pixels} returns"
                )
            chosen.append((found[0], window == found[0].label))
        return chosen, candidates
    candidates = blobs_from_labels(labels, options.min_pixels)
    if options.balls == 1:
        picks = [blob for blob in candidates if blob.disc_like][:1]
    else:
        # Two spare candidates: a support cloth or table edge may outrank a ball
        # by area, and the sphere fit is what tells them apart.
        picks = candidates[: options.balls + 2]
    if len(picks) < options.balls:
        raise SphereReferenceError(
            f"found {len(picks)} {'disc-like ' if options.balls == 1 else ''}component(s) of "
            f"at least {options.min_pixels} returns in the {options.band_mm[0]:.0f}-"
            f"{options.band_mm[1]:.0f} mm band, needed {options.balls}; candidates: "
            + json.dumps([blob.as_dict() for blob in candidates[:8]])
            + ". Give --roi for each ball instead."
        )
    picks.sort(key=lambda blob: blob.x0)
    for blob in picks:
        chosen.append((blob, labels == blob.label))
    return chosen, candidates


# --- fitting -----------------------------------------------------------------------


def _algebraic_sphere(points: Floats) -> tuple[Floats, float]:
    """Linear least-squares sphere: x²+y²+z² = 2c·p + k, so R² = k + |c|²."""
    design = np.column_stack([2.0 * points, np.ones(len(points))])
    target = np.sum(points * points, axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    centre = solution[:3]
    radius_sq = solution[3] + float(centre @ centre)
    if not radius_sq > 0.0:
        raise SphereReferenceError("algebraic sphere fit produced a non-positive radius")
    return centre, math.sqrt(radius_sq)


def _gauss_newton_sphere(
    points: Floats, centre: Floats, radius: float
) -> tuple[Floats, float, Floats]:
    """Refine on the geometric residual |p - c| - R. Returns centre, radius, covariance."""
    params = np.append(centre, radius)
    jac = np.empty((len(points), 4))
    residual = np.empty(len(points))
    for _ in range(GAUSS_NEWTON_ROUNDS):
        offsets = points - params[:3]
        distances = np.linalg.norm(offsets, axis=1)
        if np.any(distances == 0.0):
            raise SphereReferenceError("a return coincides with the sphere centre")
        residual[:] = distances - params[3]
        jac[:, :3] = -offsets / distances[:, None]
        jac[:, 3] = -1.0
        normal = jac.T @ jac
        try:
            step = np.linalg.solve(normal, -(jac.T @ residual))
        except np.linalg.LinAlgError as error:
            raise SphereReferenceError(
                "the returns do not constrain a sphere (flat or degenerate patch)"
            ) from error
        params = params + step
        if np.max(np.abs(step)) < 1e-9:
            break
    offsets = points - params[:3]
    distances = np.linalg.norm(offsets, axis=1)
    residual = distances - params[3]
    jac[:, :3] = -offsets / distances[:, None]
    dof = max(1, len(points) - 4)
    covariance = np.linalg.inv(jac.T @ jac) * float(residual @ residual) / dof
    return params[:3], float(params[3]), covariance


#: RANSAC seeding when the radius is known: minimal samples tried, and how close
#: a return must sit to a candidate sphere to count for it.
RANSAC_ITERATIONS: Final = 400
RANSAC_INLIER_MM: Final = 0.25
RANSAC_RADIUS_TOLERANCE: Final = 0.05


def _ransac_sphere(points: Floats, expected_radius: float, seed: int = 0) -> Mask:
    """Returns that lie on the best sphere of about ``expected_radius`` found by
    minimal 4-point samples: a ball's cap survives even when a cloth or table
    contributes most of the blob."""
    rng = np.random.default_rng(seed)
    best: Mask | None = None
    best_count = 0
    count = len(points)
    for _ in range(RANSAC_ITERATIONS):
        sample = points[rng.choice(count, 4, replace=False)]
        design = np.column_stack([2.0 * sample, np.ones(4)])
        target = np.sum(sample * sample, axis=1)
        try:
            solution = np.linalg.solve(design, target)
        except np.linalg.LinAlgError:
            continue
        centre = solution[:3]
        radius_sq = solution[3] + float(centre @ centre)
        if radius_sq <= 0.0:
            continue
        radius = math.sqrt(radius_sq)
        if abs(radius - expected_radius) > RANSAC_RADIUS_TOLERANCE * expected_radius:
            continue
        inliers = np.abs(np.linalg.norm(points - centre, axis=1) - radius) <= RANSAC_INLIER_MM
        found = int(inliers.sum())
        if found > best_count:
            best_count = found
            best = inliers
    if best is None or best_count < 50:
        raise SphereReferenceError(
            f"no sphere of about {2 * expected_radius:.1f} mm diameter is supported by the returns"
        )
    return best


def fit_sphere(
    points: Floats, rim_degrees: float, expected_radius: float | None = None
) -> SphereFit:
    """Robust sphere fit: algebraic seed, Gauss-Newton, residual trimming, rim exclusion."""
    if len(points) < 50:
        raise SphereReferenceError(f"{len(points)} returns are too few to fit a sphere")
    if expected_radius is None:
        keep = np.ones(len(points), dtype=bool)
        centre, radius = _algebraic_sphere(points)
    else:
        keep = _ransac_sphere(points, expected_radius)
        centre, radius = _algebraic_sphere(points[keep])
    for _ in range(TRIM_ROUNDS):
        centre, radius, _ = _gauss_newton_sphere(points[keep], centre, radius)
        residual = np.linalg.norm(points - centre, axis=1) - radius
        sigma = max(TRIM_FLOOR_MM, 1.4826 * float(np.median(np.abs(residual[keep]))))
        refreshed = np.abs(residual) <= TRIM_SIGMAS * sigma
        if np.array_equal(refreshed, keep):
            break
        keep = refreshed
        if keep.sum() < 50:
            raise SphereReferenceError("residual trimming left too few returns; not a sphere")
    rejected_by_residual = int((~keep).sum())
    # Rim: incidence angle between the fitted normal and the ray back to the camera.
    normals = (points - centre) / radius
    to_camera = -points / np.linalg.norm(points, axis=1)[:, None]
    incidence = np.degrees(np.arccos(np.clip(np.sum(normals * to_camera, axis=1), -1.0, 1.0)))
    rim = keep & (incidence > rim_degrees)
    keep &= ~rim
    if keep.sum() < 50:
        raise SphereReferenceError("rim exclusion left too few returns; lower --rim-degrees")
    centre, radius, covariance = _gauss_newton_sphere(points[keep], centre, radius)
    residual = np.linalg.norm(points[keep] - centre, axis=1) - radius
    sigmas = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return SphereFit(
        centre=(float(centre[0]), float(centre[1]), float(centre[2])),
        radius=radius,
        rms_mm=float(np.sqrt(np.mean(residual * residual))),
        inliers=int(keep.sum()),
        rejected_by_residual=rejected_by_residual,
        rejected_by_rim=int(rim.sum()),
        cap_half_angle_degrees=float(np.max(incidence[keep])),
        sigma_centre=(float(sigmas[0]), float(sigmas[1]), float(sigmas[2])),
        sigma_radius=float(sigmas[3]),
    )


def inlier_mask(points: Floats, fit: SphereFit, rim_degrees: float) -> Mask:
    """The returns the sphere fit kept, recomputed from its own parameters."""
    centre = np.asarray(fit.centre)
    residual = np.linalg.norm(points - centre, axis=1) - fit.radius
    sigma = max(TRIM_FLOOR_MM, 1.4826 * float(np.median(np.abs(residual))))
    normals = (points - centre) / fit.radius
    to_camera = -points / np.linalg.norm(points, axis=1)[:, None]
    incidence = np.degrees(np.arccos(np.clip(np.sum(normals * to_camera, axis=1), -1.0, 1.0)))
    kept = (np.abs(residual) <= TRIM_SIGMAS * sigma) & (incidence <= rim_degrees)
    return np.asarray(kept, dtype=np.bool_)


def fit_ellipsoid(points: Floats, seed: SphereFit) -> EllipsoidFit:
    """Axis-aligned ellipsoid through the sphere's inliers, seeded from the sphere.

    The residual is the normalised radial excess, ``sqrt(sum(((p-c)/s)²)) - 1``,
    which for a near-sphere is the geometric distance divided by the radius;
    the reported rms is rescaled by the mean semi-axis into millimetres.
    """

    def residual(params: Floats) -> Floats:
        offsets = (points - params[:3]) / params[3:]
        return np.sqrt(np.sum(offsets * offsets, axis=1)) - 1.0

    def jacobian(params: Floats) -> Floats:
        scaled = (points - params[:3]) / params[3:]
        q = np.sqrt(np.sum(scaled * scaled, axis=1))
        jac = np.empty((len(points), 6))
        jac[:, :3] = -scaled / (params[3:] * q[:, None])
        jac[:, 3:] = -(scaled * scaled) / (params[3:] * q[:, None])
        return jac

    start = np.array([*seed.centre, seed.radius, seed.radius, seed.radius])
    result = optimize.least_squares(residual, start, jac=jacobian, method="lm", xtol=1e-12)
    params = np.asarray(result.x, dtype=np.float64)
    if not np.all(params[3:] > 0.0):
        raise SphereReferenceError("ellipsoid fit produced a non-positive semi-axis")
    final = residual(params)
    mean_axis = float(np.mean(params[3:]))
    jac = jacobian(params)
    dof = max(1, len(points) - 6)
    covariance = np.linalg.inv(jac.T @ jac) * float(final @ final) / dof
    sigmas = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    return EllipsoidFit(
        centre=(float(params[0]), float(params[1]), float(params[2])),
        semi_axes=(float(params[3]), float(params[4]), float(params[5])),
        rms_mm=float(np.sqrt(np.mean(final * final))) * mean_axis,
        sigma_semi_axes=(float(sigmas[3]), float(sigmas[4]), float(sigmas[5])),
    )


# --- per capture --------------------------------------------------------------------


def analyse_frame(path: Path, options: Options) -> tuple[list[BallFrame], list[Blob]]:
    depth = depth_mm(load_codes(path), options.depth_scale)
    low, high = options.band_mm
    in_band = np.isfinite(depth) & (depth >= low) & (depth <= high)
    balls, candidates = select_balls(depth, in_band, options)
    fitted: list[BallFrame] = []
    reasons: list[str] = []
    for blob, mask in balls:
        points = deproject(depth, mask, options.intrinsics)
        expected = options.expected_diameter_mm
        try:
            sphere = fit_sphere(
                points, options.rim_degrees, None if expected is None else expected / 2.0
            )
        except SphereReferenceError as error:
            reasons.append(f"component {blob.label}: {error}")
            continue
        if expected is not None and abs(sphere.diameter - expected) > DIAMETER_SANITY * expected:
            reasons.append(
                f"component {blob.label}: fitted diameter {sphere.diameter:.2f} mm is not within "
                f"{100 * DIAMETER_SANITY:.0f}% of the expected {expected:.2f} mm"
            )
            continue
        if expected is not None:
            # A real cap of this ball at this depth covers about this many pixels
            # inside the rim limit; a fold in a cloth that happens to pass through
            # four points of such a sphere covers far fewer and fits far worse.
            fx = options.intrinsics.values["fx"]
            full_cap = (
                math.pi
                * (
                    sphere.radius
                    * math.sin(math.radians(options.rim_degrees))
                    * fx
                    / sphere.centre[2]
                )
                ** 2
            )
            if sphere.inliers < MIN_CAP_FRACTION * full_cap or sphere.rms_mm > MAX_CAP_RMS_MM:
                reasons.append(
                    f"component {blob.label}: {sphere.inliers} inliers of about {full_cap:.0f} "
                    f"for a full cap and rms {sphere.rms_mm:.3f} mm do not make a ball"
                )
                continue
        ellipsoid = None
        if options.ellipsoid:
            ellipsoid = fit_ellipsoid(
                points[inlier_mask(points, sphere, options.rim_degrees)], sphere
            )
        fitted.append(BallFrame(frame=path.name, blob=blob, sphere=sphere, ellipsoid=ellipsoid))
    if len(fitted) < options.balls:
        raise SphereReferenceError(
            f"{path.name}: {len(fitted)} usable ball(s) of {options.balls} wanted; "
            + "; ".join(reasons)
        )
    fitted.sort(key=lambda ball: ball.sphere.inliers, reverse=True)
    fitted = fitted[: options.balls]
    fitted.sort(key=lambda ball: ball.blob.x0)
    return fitted, candidates


def _robust(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    return {
        "median": median,
        "mad_sigma": 1.4826 * float(np.median(np.abs(array - median))),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "n": int(array.size),
    }


@dataclass
class Comparison:
    caliper_mm: float | None = None
    caliper_spread_mm: float | None = None
    spacing_mm: float | None = None
    spacing_spread_mm: float | None = None
    notes: list[str] = field(default_factory=list)


def _compare(
    measured: float, reference: float | None, spread: float | None
) -> dict[str, object] | None:
    if reference is None:
        return None
    error = measured - reference
    half = (spread or 0.0) / 2.0
    return {
        "reference_mm": reference,
        "reference_half_spread_mm": half,
        "error_mm": error,
        "error_percent": 100.0 * error / reference,
        "outside_reference_spread": abs(error) > half,
    }


def summarise_capture(
    directory: Path, results: list[list[BallFrame]], comparison: Comparison
) -> dict[str, object]:
    balls = max(len(frame) for frame in results)
    summary: dict[str, object] = {"directory": str(directory), "frames": len(results), "balls": []}
    ball_summaries: list[dict[str, object]] = []
    for index in range(balls):
        rows = [frame[index] for frame in results if len(frame) > index]
        diameters = [row.sphere.diameter for row in rows]
        entry: dict[str, object] = {
            "index": index,
            "frames_fitted": len(rows),
            "diameter_mm": _robust(diameters),
            "centre_mm": {
                axis: _robust([row.sphere.centre[k] for row in rows])
                for k, axis in enumerate("xyz")
            },
            "rms_mm": _robust([row.sphere.rms_mm for row in rows]),
            "cap_half_angle_degrees": _robust([row.sphere.cap_half_angle_degrees for row in rows]),
            "inliers": _robust([float(row.sphere.inliers) for row in rows]),
            "diameter_vs_caliper": _compare(
                float(np.median(diameters)), comparison.caliper_mm, comparison.caliper_spread_mm
            ),
            "per_frame": [row.as_dict() for row in rows],
        }
        ellipsoids = [row.ellipsoid for row in rows if row.ellipsoid is not None]
        if ellipsoids:
            entry["ellipsoid"] = {
                "diameter_x_mm": _robust([2.0 * e.semi_axes[0] for e in ellipsoids]),
                "diameter_y_mm": _robust([2.0 * e.semi_axes[1] for e in ellipsoids]),
                "diameter_z_mm": _robust([2.0 * e.semi_axes[2] for e in ellipsoids]),
                "x_over_z": _robust([e.x_over_z for e in ellipsoids]),
                "y_over_z": _robust([e.y_over_z for e in ellipsoids]),
                "rms_mm": _robust([e.rms_mm for e in ellipsoids]),
            }
        ball_summaries.append(entry)
    summary["balls"] = ball_summaries
    if balls >= 2:
        pairs = [frame for frame in results if len(frame) >= 2]
        deltas = np.array(
            [np.subtract(frame[1].sphere.centre, frame[0].sphere.centre) for frame in pairs]
        )
        distances = np.linalg.norm(deltas, axis=1)
        spacing: dict[str, object] = {
            "between": [0, 1],
            "distance_mm": _robust(distances.tolist()),
            "delta_mm": {axis: _robust(deltas[:, k].tolist()) for k, axis in enumerate("xyz")},
            "distance_vs_caliper": _compare(
                float(np.median(distances)), comparison.spacing_mm, comparison.spacing_spread_mm
            ),
        }
        if comparison.spacing_mm is not None and len(pairs) >= 2:
            spacing["axis_scales"] = fit_axis_scales(deltas, comparison.spacing_mm)
        summary["spacing"] = spacing
    return summary


#: An axis needs the pair to have pointed at least this far along it (as a
#: fraction of the spacing) before its scale factor is fitted rather than pinned.
MIN_COVERAGE: Final = 0.25


def fit_axis_scales(deltas: Floats, length: float) -> dict[str, object]:
    """Measured-over-true scale along the directions the spacing vectors covered.

    Each measured component is the true component times a factor for its
    direction, so the true length is ``sqrt(sum((m_i / f_i)^2))``. The
    directions are chosen from the data: the plane the spacing vectors lie in
    (a pair on a turntable sweeps the table plane), the in-plane direction
    closest to camera x, and the in-plane direction perpendicular to it. On a
    horizontal table seen from a pitched camera the second direction mixes
    camera y and z in a fixed ratio, and the plane normal is never covered, so
    y and z cannot be told apart from a turn alone; that is reported through the
    coverage figures rather than hidden in a fitted number. The isotropic fit is
    given alongside so the reader can see whether one factor already suffices.
    """
    if len(deltas) < 2:
        raise SphereReferenceError("a scale fit needs at least two spacing vectors")
    scatter = deltas.T @ deltas / len(deltas)
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    x_axis = np.array([1.0, 0.0, 0.0])
    first = x_axis - float(x_axis @ normal) * normal
    if np.linalg.norm(first) < 1e-6:
        first = np.array([0.0, 1.0, 0.0]) - float(normal[1]) * normal
    first = first / np.linalg.norm(first)
    second = np.cross(normal, first)
    axes = np.vstack([first, second, normal])
    for k in range(3):
        if axes[k][int(np.argmax(np.abs(axes[k])))] < 0.0:
            axes[k] = -axes[k]
    projected = deltas @ axes.T
    coverage = np.max(np.abs(projected), axis=0) / length
    fitted = coverage >= MIN_COVERAGE

    def factors_of(params: Floats) -> Floats:
        factors = np.ones(3)
        factors[fitted] = params
        return factors

    def residual(params: Floats) -> Floats:
        return np.sqrt(np.sum((projected / factors_of(params)) ** 2, axis=1)) - length

    def isotropic_residual(factor: Floats) -> Floats:
        return np.sqrt(np.sum((projected / factor[0]) ** 2, axis=1)) - length

    count = int(fitted.sum())
    if count == 0:
        raise SphereReferenceError("the spacing vectors cover no direction well enough to fit")
    fit = optimize.least_squares(residual, np.ones(count), bounds=(0.9, 1.1))
    isotropic = optimize.least_squares(isotropic_residual, np.ones(1), bounds=(0.9, 1.1))
    params = np.asarray(fit.x, dtype=np.float64)
    final = residual(params)
    jac = np.asarray(fit.jac, dtype=np.float64)
    dof = max(1, len(final) - count)
    try:
        covariance = np.linalg.inv(jac.T @ jac) * float(final @ final) / dof
        sigmas = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    except np.linalg.LinAlgError:
        sigmas = np.full(count, math.nan)
    factors = factors_of(params)
    sigma_full = np.full(3, math.nan)
    sigma_full[fitted] = sigmas
    iso = float(isotropic.x[0])
    iso_final = isotropic_residual(np.array([iso]))
    return {
        "frames": len(deltas),
        "directions_camera_xyz": [[float(v) for v in axis] for axis in axes],
        "direction_names": [
            "closest to x in the covered plane",
            "in-plane perpendicular",
            "plane normal",
        ],
        "coverage_max_component_over_length": [float(c) for c in coverage],
        "measured_over_true": [float(factors[k]) if fitted[k] else None for k in range(3)],
        "sigma_statistical": [float(sigma_full[k]) if fitted[k] else None for k in range(3)],
        "rms_mm": float(np.sqrt(np.mean(final * final))),
        "isotropic_measured_over_true": iso,
        "isotropic_rms_mm": float(np.sqrt(np.mean(iso_final * iso_final))),
    }


def _match_previous(
    fitted: list[BallFrame], previous: Sequence[tuple[float, float, float]]
) -> list[BallFrame]:
    """Reorder this frame's balls so each continues the nearest ball of the last frame."""
    best: tuple[float, tuple[int, ...]] | None = None
    for order in itertools.permutations(range(len(fitted))):
        cost = sum(math.dist(fitted[i].sphere.centre, previous[k]) for k, i in enumerate(order))
        if best is None or cost < best[0]:
            best = (cost, order)
    assert best is not None
    return [fitted[i] for i in best[1]]


def analyse_capture(directory: Path, options: Options, comparison: Comparison) -> dict[str, object]:
    paths = sorted(directory.glob(FRAME_GLOB))
    if not paths:
        raise SphereReferenceError(f"{directory} holds no {FRAME_GLOB} frames")
    paths = paths[options.skip_frames :]
    if options.max_frames is not None:
        paths = paths[: options.max_frames]
    if not paths:
        raise SphereReferenceError(
            f"{directory}: no frames left after skipping {options.skip_frames}"
        )
    results: list[list[BallFrame]] = []
    refusals: list[dict[str, str]] = []
    candidates: list[Blob] = []
    previous: list[tuple[float, float, float]] | None = None
    for path in paths:
        try:
            fitted, seen = analyse_frame(path, options)
        except SphereReferenceError as error:
            refusals.append({"frame": path.name, "reason": str(error)})
            continue
        if previous is not None and len(fitted) == len(previous) > 1:
            fitted = _match_previous(fitted, previous)
        previous = [ball.sphere.centre for ball in fitted]
        candidates = seen
        results.append(fitted)
    if not results:
        raise SphereReferenceError(
            f"no frame in {directory} yielded a sphere fit; first refusal: "
            + (refusals[0]["reason"] if refusals else "none recorded")
        )
    summary = summarise_capture(directory, results, comparison)
    summary["refused_frames"] = refusals
    summary["last_frame_candidates"] = [blob.as_dict() for blob in candidates[:8]]
    return summary


# --- reporting ------------------------------------------------------------------------


def _line(entry: dict[str, object], key: str) -> str:
    stats = entry[key]
    assert isinstance(stats, dict)
    return f"{stats['median']:.4f} mm (spread sigma {stats['mad_sigma']:.4f}, n={stats['n']})"


def print_summary(summary: dict[str, object], stream: Any = None) -> None:
    stream = sys.stdout if stream is None else stream
    print(f"{summary['directory']}: {summary['frames']} frames", file=stream)
    balls = summary["balls"]
    assert isinstance(balls, list)
    for entry in balls:
        assert isinstance(entry, dict)
        print(f"  ball {entry['index']}: diameter {_line(entry, 'diameter_mm')}", file=stream)
        print(
            f"    fit rms {_line(entry, 'rms_mm')}; cap half-angle "
            f"{_line(entry, 'cap_half_angle_degrees').replace(' mm', ' deg')}",
            file=stream,
        )
        versus = entry.get("diameter_vs_caliper")
        if isinstance(versus, dict):
            print(
                f"    vs caliper {versus['reference_mm']:.3f} mm: {versus['error_mm']:+.4f} mm "
                f"({versus['error_percent']:+.3f}%), reference half-spread "
                f"{versus['reference_half_spread_mm']:.3f} mm",
                file=stream,
            )
        ellipsoid = entry.get("ellipsoid")
        if isinstance(ellipsoid, dict):
            print(
                f"    ellipsoid diameters x/y/z {ellipsoid['diameter_x_mm']['median']:.4f} / "
                f"{ellipsoid['diameter_y_mm']['median']:.4f} / "
                f"{ellipsoid['diameter_z_mm']['median']:.4f} mm; lateral over depth x "
                f"{ellipsoid['x_over_z']['median']:.5f}, y {ellipsoid['y_over_z']['median']:.5f}",
                file=stream,
            )
    spacing = summary.get("spacing")
    if isinstance(spacing, dict):
        delta = spacing["delta_mm"]
        assert isinstance(delta, dict)
        print(
            f"  centre spacing balls 0-1: {_line(spacing, 'distance_mm')}; components x "
            f"{delta['x']['median']:+.3f} y {delta['y']['median']:+.3f} "
            f"z {delta['z']['median']:+.3f}",
            file=stream,
        )
        versus = spacing.get("distance_vs_caliper")
        if isinstance(versus, dict):
            print(
                f"    vs caliper {versus['reference_mm']:.3f} mm: {versus['error_mm']:+.4f} mm "
                f"({versus['error_percent']:+.3f}%)",
                file=stream,
            )
        scales = spacing.get("axis_scales")
        if isinstance(scales, dict):
            directions = scales["directions_camera_xyz"]
            factors = scales["measured_over_true"]
            sigmas = scales["sigma_statistical"]
            coverage = scales["coverage_max_component_over_length"]
            assert isinstance(directions, list) and isinstance(factors, list)
            assert isinstance(sigmas, list) and isinstance(coverage, list)
            for k in range(3):
                vector = ", ".join(f"{v:+.2f}" for v in directions[k])
                if factors[k] is None:
                    verdict = f"not covered (coverage {coverage[k]:.2f}), left at 1"
                else:
                    verdict = (
                        f"{factors[k]:.5f} (sigma {sigmas[k]:.5f}, coverage {coverage[k]:.2f})"
                    )
                print(f"    measured over true along ({vector}): {verdict}", file=stream)
            print(
                f"    per-direction rms {scales['rms_mm']:.4f} mm over {scales['frames']} "
                f"frames; isotropic alternative {scales['isotropic_measured_over_true']:.5f}, "
                f"rms {scales['isotropic_rms_mm']:.4f} mm",
                file=stream,
            )
    refused = summary.get("refused_frames")
    if isinstance(refused, list) and refused:
        print(f"  refused frames: {len(refused)} (first: {refused[0]['reason']})", file=stream)


LIMITATIONS: Final = (
    "Every number uses the intrinsics and depth scale it was given; nothing is corrected.",
    "Fit rms and statistical sigmas measure self-consistency, not accuracy: a biased cap "
    "fits tightly.",
    "A single cap constrains the depth semi-axis less than the lateral ones; read the "
    "ellipsoid ratios with their frame-to-frame spread, and prefer two-ball spacing "
    "laid along the axis in question.",
    "Diameter comparisons include any coating; centre spacing does not.",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "directories", type=Path, nargs="+", help="capture directories of frame_*.raw"
    )
    parser.add_argument("--output", type=Path, help="JSON report path (refuses to overwrite)")
    parser.add_argument(
        "--balls", type=int, default=1, help="balls per frame to find automatically"
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        action="append",
        metavar=("X0", "Y0", "X1", "Y1"),
        help="pixel region holding one ball (exclusive upper bounds); repeat per ball",
    )
    parser.add_argument(
        "--band", type=float, nargs=2, default=list(DEFAULT_BAND_MM), metavar=("NEAR", "FAR")
    )
    parser.add_argument("--rim-degrees", type=float, default=DEFAULT_RIM_DEGREES)
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--skip-frames", type=int, default=0, help="drop this many leading frames")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-ellipsoid", action="store_true")
    parser.add_argument("--caliper-mm", type=float, help="caliper diameter of the ball")
    parser.add_argument(
        "--caliper-spread-mm", type=float, help="max minus min of the caliper readings"
    )
    parser.add_argument("--spacing-mm", type=float, help="caliper centre spacing of two balls")
    parser.add_argument("--spacing-spread-mm", type=float)
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        help="override the factory record, for what-if deprojection only",
    )
    parser.add_argument("--depth-scale", type=float, default=SDK_MINI_FALLBACK_SCALE)
    return parser


def options_from(args: argparse.Namespace) -> Options:
    intrinsics = FACTORY_INTRINSICS
    if args.intrinsics is not None:
        fx, fy, cx, cy = args.intrinsics
        intrinsics = intrinsics_of(fx=fx, fy=fy, cx=cx, cy=cy)
    rois = tuple(tuple(int(v) for v in roi) for roi in (args.roi or []))
    for roi in rois:
        x0, y0, x1, y1 = roi
        if not (0 <= x0 < x1 <= FRAME_WIDTH and 0 <= y0 < y1 <= FRAME_HEIGHT):
            raise SphereReferenceError(
                f"region {roi} is not inside the {FRAME_WIDTH}x{FRAME_HEIGHT} frame"
            )
    if args.balls < 1 or args.balls > 4:
        raise SphereReferenceError("--balls must be 1 to 4")
    near, far = args.band
    if not (0.0 < near < far):
        raise SphereReferenceError("--band needs 0 < NEAR < FAR")
    return Options(
        intrinsics=intrinsics,
        depth_scale=args.depth_scale,
        band_mm=(near, far),
        rim_degrees=args.rim_degrees,
        min_pixels=args.min_pixels,
        balls=len(rois) if rois else args.balls,
        rois=tuple((r[0], r[1], r[2], r[3]) for r in rois),
        ellipsoid=not args.no_ellipsoid,
        skip_frames=max(0, args.skip_frames),
        max_frames=args.max_frames,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        options = replace(options_from(args), expected_diameter_mm=args.caliper_mm)
        comparison = Comparison(
            caliper_mm=args.caliper_mm,
            caliper_spread_mm=args.caliper_spread_mm,
            spacing_mm=args.spacing_mm,
            spacing_spread_mm=args.spacing_spread_mm,
        )
        if args.output is not None and args.output.exists():
            raise SphereReferenceError(f"{args.output} exists; refusing to overwrite a report")
        captures = [
            analyse_capture(directory, options, comparison) for directory in args.directories
        ]
    except SphereReferenceError as error:
        print(f"sphere-reference: {error}", file=sys.stderr)
        return 2
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "intrinsics": options.intrinsics.as_dict(),
        "depth_scale": options.depth_scale,
        "band_mm": list(options.band_mm),
        "rim_degrees": options.rim_degrees,
        "reference": {
            "caliper_mm": comparison.caliper_mm,
            "caliper_spread_mm": comparison.caliper_spread_mm,
            "spacing_mm": comparison.spacing_mm,
            "spacing_spread_mm": comparison.spacing_spread_mm,
        },
        "captures": captures,
        "limitations": list(LIMITATIONS),
    }
    for capture in captures:
        print_summary(capture)
    if args.output is not None:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=1)
        print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
