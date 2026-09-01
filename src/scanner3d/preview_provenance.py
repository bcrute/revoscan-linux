"""What each preview IS, said on the screen rather than left to be inferred.

The report this module answers
-------------------------------
The appliance draws three pictures of the same scan and they do not have the
same standing:

* the false-coloured depth panel is the sensor's own 16-bit codes, with a
  neutral checker where the sensor returned nothing at all;
* the model panel's dots are those returns placed by the factory calibration
  and by this scan's own pose estimates;
* the opt-in surface is derived geometry - and there are TWO of those, made by
  two different producers, which is the distinction this module was corrected
  to carry.

Only the first is a raw observation, only the first two contain measurements at
all, and none of the three is a checked dimension. Until now the difference
lived in a long tooltip on one checkbox, which means it was only ever read by
somebody who already suspected there was something to read. An operator
comparing clean dots, a lumpy shell and a sparse depth frame should not have to
hover to find out which pixels are sensor returns and which shape was made up.

The two surfaces are not the same claim
---------------------------------------
While a capture runs, the panel's surface is the occupancy grid the accepted
frames are folded into: geometry invented on cell edges, from the poses the scan
accepted AS IT RAN, which the owner directly observed standing outside dots that
looked clean. When Reconstruct finishes, the panel's surface is the one that was
just exported: the zero level of a signed-distance field fused from the captured
full-resolution frames at the reconstruction's CORRECTED poses. Both are derived
rather than measured, and that is all they have in common - so a single sentence
covering both would have to be false about one of them. :class:`SurfaceKind` is
therefore carried from the producer, and every visible consumer of it here is
keyed on it.

What this module is
-------------------
One enum for the picture that is actually mounted, one enum for WHICH surface
produced the mesh, one small record for the numbers a derived surface has to
state about itself, and pure formatters that turn those into the sentences the
panel shows: the provenance line, the coverage summary's surface clause, and the
surface control's own title and tooltip. No Qt, no capture state, no widget: the
window asks its viewer what is on the screen, and this decides what may
truthfully be said about it.

What it refuses to do
---------------------
* It never assembles a claim from a checkbox or a remembered preference. Those
  say what an operator ASKED for; :class:`ModelPicture` says what is drawn, and
  a picture that could not be drawn has no wording here.
* It never guesses which surface it is looking at. :class:`SurfaceKind` is set
  by the producer and validated here; a missing or unrecognised kind is refused,
  because the two available answers are exactly the two sentences that would be
  false about the other one.
* It never promises a physical unit. Raw depth codes are codes, and the cell or
  voxel edge of a derived surface is stated in the same model coordinates the
  panel draws in.
* It is display wording and nothing else. Nothing in this module accepts or
  refuses a frame, moves a gate, changes a threshold or reaches any completion,
  fusion, tracking or export path, and both enums' members carry stable
  identifiers rather than the sentences, so no caller can be tempted to branch on
  the words on the screen.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from scanner3d.errors import ScannerError


class ProvenanceUnusable(ScannerError):
    """A provenance state that does not describe any picture this panel draws.

    Raised rather than papered over. The one failure this whole seam exists to
    prevent is a confident sentence beside a picture it does not describe, and a
    caller that has lost track of which of the two it is showing cannot be given
    a default without inventing exactly that.
    """


class ModelPicture(enum.Enum):
    """Which picture the model panel is ACTUALLY drawing.

    Derived from the mounted viewer, never from the surface checkbox: a ticked
    box over a scan with no drawable surface is the state this distinction was
    written for.

    The members carry short stable identifiers rather than the operator-facing
    sentences, so that the wording can be rewritten without moving any state and
    so that nothing downstream can end up branching on display text.
    """

    #: Nothing has been drawn yet: no measured points and no derived surface.
    NOTHING = "nothing"
    #: The coverage-coloured cloud of measured depth returns.
    POINTS = "points"
    #: A derived surface - which one is said by :class:`SurfaceKind`.
    SURFACE = "surface"


class SurfaceKind(enum.Enum):
    """WHICH producer built the surface the panel is holding.

    The panel has exactly two surface producers and they are not interchangeable:
    one is a live diagnostic built while the scan runs, the other is the file the
    operator is being asked to trust at the end of it. Set explicitly by the
    producer, alongside the mesh it made, and never inferred - not from the
    snapshot's note, not from whether the view re-frames, not from the checkbox,
    not from a remembered preference and not from a default. There is no default:
    the only two candidates are the two sentences that are false about each
    other.

    Identifiers rather than sentences, for :class:`ModelPicture`'s reason.
    """

    #: The live occupancy grid the accepted frames are folded into while the
    #: capture runs, marched into triangles on cell edges. Capture-time poses.
    LIVE_OCCUPANCY = "live_occupancy"
    #: The surface Reconstruct exported: the zero level of a signed-distance
    #: field fused from the captured full-resolution frames at corrected poses.
    RECONSTRUCTED_TSDF = "reconstructed_tsdf"


def _exact_kind(value: object) -> SurfaceKind:
    """A genuine :class:`SurfaceKind`. Everything else is refused, not guessed."""
    if not isinstance(value, SurfaceKind):
        raise ProvenanceUnusable(
            f"surface kind is {value!r}, which is not a SurfaceKind; the producer of a "
            "drawn surface says which surface it is, and neither answer may be assumed"
        )
    return value


def _exact_count(value: object, what: str) -> int:
    """A genuine positive ``int``. ``bool`` is refused, as it is not a count."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProvenanceUnusable(f"{what} is {value!r}, which is not an integer count")
    if value <= 0:
        raise ProvenanceUnusable(f"{what} is {value!r}; a drawn surface has at least one triangle")
    return value


def _positive_size(value: object, what: str) -> float:
    """A finite positive real. ``bool`` is refused for the same reason."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceUnusable(f"{what} is {value!r}, which is not a number")
    size = float(value)
    if not math.isfinite(size):
        raise ProvenanceUnusable(f"{what} is {value!r}, which is not finite")
    if size <= 0.0:
        raise ProvenanceUnusable(f"{what} is {value!r}; a marched grid has a positive cell edge")
    return size


@dataclass(frozen=True, slots=True)
class SurfaceEvidence:
    """What a derived surface has to state about itself before it is described.

    ``kind`` is which of the two producers made it, ``faces`` is how many
    triangles the panel was handed and ``cell`` is the grid edge they were
    marched at, in the model's own coordinates. All three are validated here
    rather than at the format string: a surface that cannot say how coarse it is
    has no business being called a preview of anything - a zero or missing cell
    edge printed as ``0.00`` would read as an infinitely fine one - and a surface
    that cannot say which producer made it is exactly the state in which the live
    preview's sentence gets printed under the exported reconstruction.
    """

    kind: SurfaceKind
    faces: int
    cell: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _exact_kind(self.kind))
        object.__setattr__(self, "faces", _exact_count(self.faces, "surface triangle count"))
        object.__setattr__(self, "cell", _positive_size(self.cell, "surface cell edge"))


#: What the false-coloured depth panel is, said beside the picture and not in a
#: tooltip. Three claims, in the order an operator meets them: these are the
#: sensor's own codes, the checker is an absence rather than a reading, and
#: nothing has been filled in. No physical unit appears, because none is
#: established - see :class:`~scanner3d.viewer.FrameStats`.
RAW_DEPTH_PROVENANCE: Final = (
    "RAW sensor depth codes, false-coloured. The grey checker is NO RETURN - "
    "the sensor reported nothing for those pixels, which is not a near reading "
    "and not a far one. No completion, filling, smoothing or carry-over from "
    "another frame is applied, and these codes are not millimetres or any other "
    "distance."
)

#: The model panel before anything has been drawn into it.
NO_MODEL_PROVENANCE: Final = (
    "Nothing drawn yet: no measured points and no derived surface in this view."
)

#: The dots, in one sentence that has to be true of BOTH clouds this panel can
#: hold: the live preview sampled from the frames accepted so far, and the
#: completed registered reconstruction drawn at the corrected poses. So it says
#: what does not change between them - every dot is a returned sample, its
#: position depends on the calibration and on pose ESTIMATES rather than on
#: surveyed coordinates, and what is on the screen has been sampled and thinned
#: to get there - and it does not call either of them the other one.
POINTS_PROVENANCE: Final = (
    "MEASURED depth returns: every dot is a sample the sensor returned. Where "
    "each one is drawn depends on the factory calibration and on this scan's "
    "pose estimates, which are estimates rather than surveyed coordinates, and "
    "the cloud on the screen is sampled and thinned for display."
)

#: The LIVE surface: the occupancy grid, in its own words, with the two numbers
#: that say how coarse it is - and then the reminder that the coverage figures
#: beside it are still measured on the dots. Its poses are the capture-time ones
#: because that is what the live accumulator has; nothing corrects them.
LIVE_SURFACE_PROVENANCE: Final = (
    "DERIVED occupancy surface - NOT measured: {faces} triangles marched at "
    "cell {cell:.2f} from the grid the poses accepted during capture were folded "
    "into. It is built on grid cells rather than fitted to the samples, so it "
    "can look faceted or lumpy and can stand outside the dots. The coverage "
    "percentages and the red 'still needs scanning' colouring describe the "
    "POINTS, not this surface."
)

#: The FINAL surface: what Reconstruct exported. Its true construction, and
#: deliberately none of the live preview's: no grid of filled cells, no
#: capture-time poses, and the word this panel used to print over it - occupancy
#: - does not appear at all. Still derived, and still unchecked against any
#: physical reference, which is the claim a better surface must not quietly
#: acquire.
RECONSTRUCTED_SURFACE_PROVENANCE: Final = (
    "DERIVED signed-distance (TSDF) surface - NOT measured: {faces} triangles "
    "marched at the zero level of a field of voxel {cell:.2f}, fused from the "
    "captured full-resolution depth frames at this reconstruction's CORRECTED "
    "poses. It is the surface Reconstruct exported, not the live preview, and no "
    "dimension of it has been checked against a physical reference. The coverage "
    "percentages and the red 'still needs scanning' colouring describe the "
    "POINTS, not this surface."
)

#: The one sentence the provenance line may use for each surface. A mapping
#: rather than a chain of ``if``s, because the failure this seam exists to
#: prevent is one branch quietly covering both kinds: a kind with no wording is
#: refused below rather than falling through to whichever came first.
_SURFACE_PROVENANCE: Final[Mapping[SurfaceKind, str]] = {
    SurfaceKind.LIVE_OCCUPANCY: LIVE_SURFACE_PROVENANCE,
    SurfaceKind.RECONSTRUCTED_TSDF: RECONSTRUCTED_SURFACE_PROVENANCE,
}

#: The surface clause of the coverage summary, which sits beside percentages
#: that are measured on the POINTS. Same two kinds, same distinction, because a
#: summary calling the exported surface an experimental preview contradicts the
#: provenance line one widget above it.
LIVE_SURFACE_SUMMARY: Final = (
    "Surface: {faces} triangles at cell {cell:.2f}, marched from the grid the "
    "poses accepted so far were folded into. Experimental occupancy preview, "
    "drawn only while its box is ticked."
)
RECONSTRUCTED_SURFACE_SUMMARY: Final = (
    "Surface: {faces} triangles at voxel {cell:.2f}, the exported signed-distance "
    "(TSDF) surface fused from the full-resolution frames at the corrected poses. "
    "Derived geometry, drawn only while its box is ticked."
)

_SURFACE_SUMMARY: Final[Mapping[SurfaceKind, str]] = {
    SurfaceKind.LIVE_OCCUPANCY: LIVE_SURFACE_SUMMARY,
    SurfaceKind.RECONSTRUCTED_TSDF: RECONSTRUCTED_SURFACE_SUMMARY,
}


@dataclass(frozen=True, slots=True)
class SurfaceControl:
    """What the surface checkbox is CALLED and what it says when hovered.

    Two strings rather than two lookups, so that a caller cannot end up with a
    title for one surface and a tooltip for the other.
    """

    title: str
    tooltip: str


#: The control while the panel holds no surface at all. Not the live wording:
#: this is a disabled box over a panel that has never been given a mesh, and
#: naming one of the two producers here would be the same guess the provenance
#: line refuses. It says instead what each of them would be.
NO_SURFACE_CONTROL_TITLE: Final = "No surface available"
NO_SURFACE_CONTROL_TOOLTIP: Final = (
    "There is no surface to show yet, so this box does nothing.\n\n"
    "A running capture builds an experimental occupancy preview from the poses\n"
    "it accepts as it goes. Reconstruct instead exports a signed-distance\n"
    "surface fused from the full-resolution frames at its corrected poses.\n"
    "They are different surfaces made by different arithmetic, so this control\n"
    "names whichever one the panel is actually holding rather than assuming.\n\n"
    "The points are the live picture and the one to read the scan from.\n"
    "Coverage - the red 'still needs scanning' colouring - is measured on the\n"
    "points, so it is shown there."
)

#: The control while the panel holds the LIVE occupancy preview.
LIVE_SURFACE_CONTROL_TITLE: Final = "Show experimental surface instead of points"
LIVE_SURFACE_CONTROL_TOOLTIP: Final = (
    "EXPERIMENTAL, and off until you ask for it. The points are the live\n"
    "picture and the one to read the scan from.\n\n"
    "This draws the voxel occupancy grid the accepted frames are folded into.\n"
    "A cell joins the surface when its neighbours are occupied too, so what is\n"
    "marched is faceted at the cell size and stands as much as one cell OUTSIDE\n"
    "the dots: it can look lumpy where the dots look clean. It does not smooth\n"
    "or refit the samples, and it turns a small disagreement between two\n"
    "accepted poses into solid shape.\n\n"
    "It is built from the poses that were accepted AS THE SCAN RAN, so it\n"
    "drifts where the scan drifts, and its cell size is never made coarser to\n"
    "keep the picture affordable: an object that outgrows the live budget loses\n"
    "the preview, with the reason beside the coverage numbers.\n\n"
    "The final mesh is separate work, and it is now a DIFFERENT surface.\n"
    "Reconstruct places the frames better - it registers every frame and\n"
    "works from the CORRECTED poses - and it no longer calls this mesher at\n"
    "all: it fuses a signed-distance field from the raw frames and marches\n"
    "its zero level, so that surface passes THROUGH the samples instead of\n"
    "wrapping the cells they filled, and it stops where the frames stopped\n"
    "seeing rather than closing over a side nobody looked at. When the panel\n"
    "is holding that one, this control says so; what is drawn here is still\n"
    "this occupancy grid and still stands outside the dots.\n\n"
    "That is a better surface, and it is still not a measurement: no\n"
    "dimension in it has been checked against a physical reference, and it\n"
    "is only as right as the poses it was given.\n\n"
    "Coverage - the red 'still needs scanning' colouring - is measured on the\n"
    "points, so it is shown there."
)

#: The control while the panel holds the surface Reconstruct exported. The live
#: preview's words are not merely softened here, they are absent: this mesh was
#: not marched from filled cells and was not placed at capture-time poses, so
#: saying either of those would be the exact defect this round corrects.
RECONSTRUCTED_SURFACE_CONTROL_TITLE: Final = "Show reconstructed surface instead of points"
RECONSTRUCTED_SURFACE_CONTROL_TOOLTIP: Final = (
    "This is the surface Reconstruct just exported, and it is the same mesh as\n"
    "the file in the capture folder rather than a second one made for the\n"
    "screen.\n\n"
    "It is the zero level set of a truncated signed-distance field, fused from\n"
    "the captured full-resolution depth frames placed at the CORRECTED poses\n"
    "this reconstruction solved for. It passes through the samples rather than\n"
    "wrapping the cells they filled, and it stops where the frames stopped\n"
    "seeing rather than closing over a side nobody looked at, so an open mesh\n"
    "here is a report about the capture.\n\n"
    "It is still DERIVED geometry and not a measurement: no dimension of it has\n"
    "been checked against a physical reference, and it is only as right as the\n"
    "poses and the frames it was fused from.\n\n"
    "The points are still the picture coverage is measured on, so the red\n"
    "'still needs scanning' colouring is shown there and describes them."
)

_SURFACE_CONTROL: Final[Mapping[SurfaceKind, SurfaceControl]] = {
    SurfaceKind.LIVE_OCCUPANCY: SurfaceControl(
        title=LIVE_SURFACE_CONTROL_TITLE, tooltip=LIVE_SURFACE_CONTROL_TOOLTIP
    ),
    SurfaceKind.RECONSTRUCTED_TSDF: SurfaceControl(
        title=RECONSTRUCTED_SURFACE_CONTROL_TITLE, tooltip=RECONSTRUCTED_SURFACE_CONTROL_TOOLTIP
    ),
}

#: The control with nothing to describe. Held as a record like the other two so
#: that every caller reads title and tooltip the same way.
NO_SURFACE_CONTROL: Final = SurfaceControl(
    title=NO_SURFACE_CONTROL_TITLE, tooltip=NO_SURFACE_CONTROL_TOOLTIP
)


def _wording(surface: SurfaceEvidence, table: Mapping[SurfaceKind, str]) -> str:
    """The one sentence ``surface``'s own kind selects, or a refusal.

    The kind is re-checked here rather than trusted from the record, because a
    record can be assembled by anything - including a caller that bypassed
    ``__init__`` - and the whole point of this seam is that an unrecognised
    surface is refused instead of being described as one of the two known ones.
    """
    kind = _exact_kind(surface.kind)
    said = table.get(kind)
    if said is None:
        raise ProvenanceUnusable(
            f"{kind} has no wording here; a surface kind without its own sentence "
            "must not borrow another kind's"
        )
    return said.format(faces=surface.faces, cell=surface.cell)


def model_provenance(picture: ModelPicture, surface: SurfaceEvidence | None = None) -> str:
    """One sentence describing exactly the picture named by ``picture``.

    ``surface`` is required by :attr:`ModelPicture.SURFACE` and refused by the
    other two. That is the strict half of this boundary: a caller holding a
    surface's triangle count while drawing the dots is a caller about to print
    one picture's numbers beside another picture, which is the defect this seam
    exists to make impossible rather than unlikely.

    Which of the two surface sentences comes back is decided by
    :attr:`SurfaceEvidence.kind` alone. Nothing else here is consulted, so the
    exported reconstruction cannot be described as the live preview by a caller
    that simply forgot to say which it had.
    """
    if not isinstance(picture, ModelPicture):
        raise ProvenanceUnusable(
            f"{picture!r} is not a ModelPicture; there is no picture to describe"
        )
    if picture is ModelPicture.SURFACE:
        if not isinstance(surface, SurfaceEvidence):
            raise ProvenanceUnusable(
                "a derived surface must state which surface it is, its triangle count "
                f"and its cell edge, and {surface!r} is not SurfaceEvidence"
            )
        return _wording(surface, _SURFACE_PROVENANCE)
    if surface is not None:
        raise ProvenanceUnusable(
            f"{picture} is not the surface, so it may not carry {surface!r} with it"
        )
    if picture is ModelPicture.POINTS:
        return POINTS_PROVENANCE
    return NO_MODEL_PROVENANCE


def surface_summary(surface: SurfaceEvidence) -> str:
    """The coverage summary's surface clause, for the surface that is HELD.

    Said whether or not the surface is the picture showing, because the coverage
    percentages either side of it are measured on the points - and driven by the
    same :class:`SurfaceKind` as the provenance line, because two sentences about
    one mesh a few pixels apart must not be able to disagree about what it is.
    """
    if not isinstance(surface, SurfaceEvidence):
        raise ProvenanceUnusable(
            f"{surface!r} is not SurfaceEvidence; there is no surface to summarise"
        )
    return _wording(surface, _SURFACE_SUMMARY)


def surface_control(surface: SurfaceEvidence | None) -> SurfaceControl:
    """What the surface checkbox should say, for the surface the panel HOLDS.

    ``None`` means the panel holds no surface, which is its own wording rather
    than either producer's: a disabled box over a panel that has never been given
    a mesh is not evidence that the mesh it lacks would have been the live one.

    Keyed on :class:`SurfaceKind` for the reason the sentences above are. The
    round-one defect was not only in the provenance line: the same checkbox
    described the occupancy grid while drawing the exported reconstruction, and a
    persistent line cannot be truthful while the control beside it contradicts
    it.
    """
    if surface is None:
        return NO_SURFACE_CONTROL
    if not isinstance(surface, SurfaceEvidence):
        raise ProvenanceUnusable(
            f"{surface!r} is not SurfaceEvidence; there is no surface for this control to name"
        )
    kind = _exact_kind(surface.kind)
    control = _SURFACE_CONTROL.get(kind)
    if control is None:
        raise ProvenanceUnusable(
            f"{kind} has no control wording here; a surface kind without its own "
            "title and tooltip must not borrow another kind's"
        )
    return control
