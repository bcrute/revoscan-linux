"""A window you can point at an object and get a scan out of.

Why this exists
---------------
Everything this project can do was, until now, reachable only by typing a
command with the right arguments in the right order. The person holding the
scanner is standing at a turntable deciding where to point it next, and asking
them to alt-tab to a terminal to find out how the capture is going is asking
them to do the one thing the situation does not allow.

Three views, because there are three different questions
--------------------------------------------------------
COLOUR answers "is the object where I think it is" - the framing question, and
the reason the vendor's own software shows this camera rather than the depth
one while you position. It runs DURING a capture as well as before one, because
the four minutes of a sweep is precisely when the question keeps being asked.
The RGB camera returns nothing unless a depth stream is already running, so it
is started by the first delivered depth frame and stopped with that stream.

DEPTH answers "is the scanner actually seeing it" - an object can be perfectly
framed and still return almost nothing, if it is too close, too far, too shiny
or too dark. It shows every COMPLETE frame the scanner delivers, whatever the
decimation is set to: how often a picture reaches a screen and how often a frame
is written and aligned are two different questions, and answering the first one
with the second made a capture at decimate 3 look like a scanner running at a
third of its real rate. Only the saving, the deprojection and the alignment are
thinned; an incomplete delivery is still shown to nobody.

MODEL answers "what have I actually got" - the accumulated cloud, turnable,
with the under-scanned parts in soft red and the frame that has just arrived
lifted towards white on top of that meaning. This is the one that decides
whether a capture was worth the four minutes, so it gets the space; the other
two are glanced at rather than studied. It re-frames itself only at a boundary -
a new capture, a registered reconstruction - so an update that adds a frame
looks like an addition instead of being normalised back to the same apparent
size with the operator's own pan thrown away.

A third question, "is my rotation speed sensible", is answered by the travel
readout. It reports how far the surface moves between frames rather than an
angle, because a small object turning fast moves its surface less than a large
one turning slowly, and it is the millimetres that alignment cares about.

Reconstruct is where a capture becomes files
--------------------------------------------
Capture writes raw frames and nothing else. Reconstruct reads a folder of them -
the INPUT, which is what its chooser asks for - registers every one, and writes a
fresh folder beside them holding the registered point cloud, the same surface as
a PLY mesh and an OBJ, and a record naming all three with their lengths, their
digests, the axis convention and everything none of it establishes. The model
reaches the screen only once those files exist, because a picture is the wrong
thing to be the only evidence that a scan can be opened somewhere else.

Everything shown and everything written is in one upright right-handed frame -
:data:`~scanner3d.reconstruction_export.UPRIGHT_FROM_OPTICAL`, a half turn of
the depth camera's optical frame about x. Depth coordinates put +y DOWN the
image, which is correct for a camera and upside down in every viewer, this one
included. The rotation is applied at the seam between measuring and showing:
raw frames, deprojection, poses and scoring never see it, and the exported
record states it so a consumer can undo it exactly.

What this does NOT do
---------------------
The live model shown DURING a capture is a preview built from chained
alignment only. It has no loop closure, so it drifts, and on a real sweep that
drift reached 65 degrees over one revolution while every internal metric
looked healthy. Treat it as "have I pointed the scanner at that yet", never as
a measure of quality. The number that can be trusted comes from Reconstruct,
which scores against evidence it was not allowed to fit.

It is also deliberately INCOMPLETE. It is rebuilt from a fixed number of frames
sampled across the whole sweep rather than from every frame aligned so far, so
a rebuild two hundred frames in costs what one forty frames in cost. Every frame
is still on disk at full length, and Reconstruct still reads all of them.

Nothing frequent is queued to the window
----------------------------------------
Depth frames, colour frames, the saved-frame count and the model all cross to
the GUI through a one-slot channel that replaces its contents rather than
queueing them. A window that falls behind therefore skips to the newest of each
instead of working through a backlog of pictures that were true several seconds
ago.

That backlog is a risk this corrects, not a diagnosis it proves. The owner's
2026-08-26 report - the model updating two or three times and then sitting
still - is consistent with it, and offline replay of that capture ruled out the
scanner, the frames on disk and the geometry; but no trace of that run's event
loop was ever taken, so the queue is not established as its sole cause. What
IS established is that an unbounded producer wins against any consumer
eventually. The produced/applied/superseded lines in the log are what a second
physical report can be answered with.

Long work runs on worker threads, and every long job can be stopped. A window
that freezes during a four-minute capture looks crashed; a capture that cannot
be abandoned means noticing a mistake in the first ten seconds and having to
watch the remaining four minutes anyway.

Stop stops, and stays stopped
-----------------------------
Stopping a capture is a request the job honours at the next safe boundary -
after the current buffer has gone back to the driver, because a callback that
raises skips that hand-back and this hardware answers a buffer held at teardown
by re-enumerating. So Stop is never instant, and it is never a lie either: the
capture ends, the frames already written stay written, and the live view is NOT
put back up. A capture that answered Stop by restarting the preview stream is
what the owner saw on 2026-08-26, and from outside the process it is
indistinguishable from the button doing nothing at all.

A capture that reaches its own end still resumes the preview it interrupted,
because that is the view you want next when a scan finishes on its own terms.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Executor, Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from scanner3d import preparation
from scanner3d.cadence import REFUSAL_REASON, CadenceController
from scanner3d.capture_sequence import (
    TRACKING_LOSS_RECORD_NAME,
    TRACKING_LOSS_STREAK,
    CaptureSequenceUnreadable,
    read_capture_sequence,
)
from scanner3d.cloud_viewer import CloudViewer
from scanner3d.config import (
    CAPTURES_DIR_ENV_VAR,
    GUI_CAPTURE_COMMAND,
    CaptureDirUnavailable,
    ConfigFileUnreadable,
    ConfigWriteRefused,
    allocate_capture_dir,
    captures_dir_source,
    configured_captures_dir,
    resolve_serial,
    set_captures_dir,
    validate_capture_name,
)
from scanner3d.coverage import coverage_map
from scanner3d.denoise import estimate_spacing
from scanner3d.discovery import find_scanners
from scanner3d.geometry import Rigid
from scanner3d.logs import configure as configure_logging
from scanner3d.logs import get_logger, timed
from scanner3d.mesh import Mesh, SurfaceAccumulator
from scanner3d.pipeline import travel_verdict
from scanner3d.preparation import (
    FusionFrame,
    FusionMaskSummary,
    PreparationSummary,
    PreparedCloud,
    filter_statement,
    mask_support_band,
)
from scanner3d.preview_provenance import (
    LIVE_SURFACE_CONTROL_TITLE,
    LIVE_SURFACE_CONTROL_TOOLTIP,
    RAW_DEPTH_PROVENANCE,
    ModelPicture,
    ProvenanceUnusable,
    SurfaceEvidence,
    SurfaceKind,
    model_provenance,
    surface_control,
    surface_summary,
)
from scanner3d.reconstruction_export import (
    ReconstructionExport,
    export_reconstruction,
    upright_pose,
)
from scanner3d.reconstruction_readiness import ReconstructionRefused
from scanner3d.relocalize import (
    MAX_ATTEMPTS,
    MAX_TENTATIVE_BYTES,
    MAX_TENTATIVE_FRAMES,
    MAX_TENTATIVE_SECONDS,
    MIN_TENTATIVE_FRAMES,
    LocalEdge,
    Relocalization,
    bounded_reference,
    evidence_summary,
    local_chain_gates,
    relocalize,
)
from scanner3d.return_health import (
    COLLAPSING,
    NO_RETURNS,
    SEVERELY_SPARSE,
    ReturnHealthTracker,
    spatial_summary,
)
from scanner3d.v4l2_stream import (
    CALLER_STOP_REASON,
    MAX_DURATION_S,
    MAX_FRAMES,
    Cadence,
    FrameRecord,
    StreamDeviceNotReady,
    StreamResult,
    wait_for_capture_access,
)
from scanner3d.viewer import FrameStats, render_frame

if TYPE_CHECKING:
    from scanner3d.depth_metric import DepthUnitAssumption
    from scanner3d.pipeline import SweepReconstruction

#: The only frame geometry this scanner's depth mode produces. Named here
#: rather than imported so the window does not depend on another module's
#: private surface for a number the hardware fixes.
FRAME_WIDTH: Final = 640
FRAME_HEIGHT: Final = 400

#: Frames delivered per kept frame. Three gives roughly two degrees between
#: kept frames on a turntable turning once in two minutes, the density measured
#: to register well. Large steps make alignment fail silently.
DEFAULT_DECIMATE: Final = 3

#: Delivered-frame budget for one capture: a little over four minutes at the
#: measured 4.6 fps, enough for two full revolutions.
DEFAULT_FRAMES: Final = 1250

#: SUCCESSFUL ALIGNMENTS between live-coverage refreshes.
#:
#: Counted in alignments rather than in saved frames because the refresh is
#: built out of aligned clouds and nothing else. Keying it to saved frame
#: numbers meant a frame that failed to align at frame 12 or 24 skipped that
#: refresh entirely and the model sat still for the next two dozen frames with
#: no explanation - which is what the owner saw on 2026-08-26.
#:
#: Four, and no longer twelve, because the owner's 15:50:53 run MEASURED what
#: twelve costs: eight live models, produced at 3, 15, 27, 39, 51, 63, 75 and 87
#: aligned frames, every one of them applied and none superseded or dropped.
#: That is a picture that changes once every 8-9 seconds, which over a
#: four-minute sweep is a view the operator cannot tell from a stalled one. The
#: same run also measured what a rebuild costs now that it is bounded - 0.008 s
#: to 0.175 s, against the roughly 0.7 s each aligned frame takes - so the host
#: has room to spare and the limit was never compute. At four the same run would
#: have refreshed roughly every 2-3 seconds.
COVERAGE_EVERY: Final = 4

#: Aligned clouds needed before the model is worth drawing at all. Below this
#: the coverage grid has too few observations for its "needs more scanning"
#: judgement to mean anything, and drawing one or two frames as a model invites
#: exactly the wrong conclusion about a scan that has barely started.
MIN_COVERAGE_CLOUDS: Final = 3

#: How the depth panel's return-health line looks when there is nothing wrong:
#: no text, no colour, no space taken from the picture above it.
QUIET_RETURN_STYLE: Final = "font-size:13pt; font-weight:bold; padding:2px;"

#: ...and when there IS. Amber rather than the clipping banner's red, because
#: these are two different problems and an operator who has learned that red
#: means "the object is running off the edge" must not have to re-read it. Loud
#: enough to be unmissable beside the image, and still just a label: it takes no
#: click, no focus and no key press away from someone holding the scanner.
LOUD_RETURN_STYLE: Final = (
    "font-size:13pt; font-weight:bold; padding:6px; color:#ffffff; background:#8a4b00;"
)

#: How both provenance lines look: the one under the depth image and the one
#: under the model. Quiet - this is not a warning and there is nothing to act on
#: - but always present and never hidden behind a hover, which is the whole
#: point of it. One style for both, because they answer the same question about
#: two pictures and reading them as a pair is how the two are told apart.
PROVENANCE_STYLE: Final = "font-weight:bold; color:#3b4252; padding:2px;"

#: The primary status card has one vocabulary across capture, depth and
#: tracking.  The title carries the colour; the two lines below it stay plain
#: so the operator can read the action without decoding a paragraph.
SCAN_STATUS_STYLES: Final[dict[str, str]] = {
    "idle": (
        "font-size:17pt; font-weight:bold; padding:8px; color:#30363d; "
        "background:#eaeef2; border-radius:4px;"
    ),
    "info": (
        "font-size:17pt; font-weight:bold; padding:8px; color:#ffffff; "
        "background:#0969da; border-radius:4px;"
    ),
    "ok": (
        "font-size:17pt; font-weight:bold; padding:8px; color:#ffffff; "
        "background:#1a7f37; border-radius:4px;"
    ),
    "warning": (
        "font-size:17pt; font-weight:bold; padding:8px; color:#ffffff; "
        "background:#9a6700; border-radius:4px;"
    ),
    "error": (
        "font-size:17pt; font-weight:bold; padding:8px; color:#ffffff; "
        "background:#cf222e; border-radius:4px;"
    ),
}

SCAN_STATUS_CAPTURE_SOURCE: Final = "capture"
SCAN_STATUS_TRACKING_SOURCE: Final = "tracking"
SCAN_STATUS_DEPTH_SOURCE: Final = "depth"
SCAN_STATUS_MOTION_SOURCE: Final = "motion"

IDLE_SCAN_STATUS_TITLE: Final = "READY"
IDLE_SCAN_STATUS_SUMMARY: Final = "No scan is running."
IDLE_SCAN_STATUS_ACTION: Final = "Frame the object in Preview, then start a capture."

#: How many failures of one live-preview stage are reported individually before
#: the rest are summarised. A capture delivers frames at about 4.6 fps, so a
#: stage that fails on every frame would write thousands of identical lines into
#: a log whose whole purpose is being readable after the fact. The first few
#: carry the reason; the summaries prove it never recovered.
FAILURES_REPORTED: Final = 3

#: How often to restate a live-preview failure once it has stopped being news.
FAILURE_SUMMARY_EVERY: Final = 25

#: Points kept per cloud for the live preview. Low on purpose: this estimate is
#: thrown away and recomputed properly by Reconstruct.
PREVIEW_POINTS: Final = 900

#: What the Reconstruct toolbar action is CALLED, and what it says when hovered.
#:
#: Constants rather than literals inside the toolbar builder, because the wording
#: is the part of this action that has to be right: it both READS a folder and
#: CREATES one, and an operator who reads "Reconstruct..." and is then shown a
#: folder chooser has every reason to think they are being asked where to put
#: the output. Naming them here is also what makes the wording assertable
#: without a Qt toolbar, which is the only way it gets checked at all.
RECONSTRUCT_TITLE: Final = "Reconstruct + export..."
RECONSTRUCT_TOOLTIP: Final = (
    "Turn a capture you have already taken into files you can open elsewhere.\n\n"
    "You are asked to pick the folder of CAPTURED FRAMES to read - the input.\n"
    "Every frame in it is registered, and then a NEW folder is created inside\n"
    "it holding the registered point cloud (PLY), the surface as a mesh (PLY\n"
    "and OBJ), and a record naming all three with their checksums and what\n"
    "they do not establish.\n\n"
    "An earlier reconstruction of the same capture is never overwritten: each\n"
    "run makes its own folder. Cancelling the chooser writes nothing."
)

#: The folder chooser's own caption, for the same reason and asserted with it.
RECONSTRUCT_DIALOG_CAPTION: Final = "Choose a capture folder of frames to reconstruct (input)"

#: Points kept per frame when Reconstruct reads a capture back off disk.
#:
#: Re-exported from :mod:`scanner3d.preparation`, which owns it now: the window
#: and ``scanner3d reconstruct`` both thin to it, and the value being one
#: constant instead of two matching literals is what keeps them reconstructing
#: the same capture the same way. The name stays here because this module's
#: readers and tests reach for it here.
#:
#: What it does NOT reach is the exported surface. That mesh is fused by
#: :mod:`scanner3d.tsdf` from the ORIGINAL full-length Y16 frames - the same
#: bytes that were read off disk, at the poses the registration accepted - so
#: the thinning here bounds what registration costs and what the cloud file
#: holds, and no decimated point is what the object's shape was decided from.
#: The live preview's own bounds (:data:`LIVE_MODEL_CLOUDS`,
#: :data:`LIVE_MODEL_POINTS`) are not applied to any of it.
RECONSTRUCT_POINTS: Final = preparation.RECONSTRUCT_POINTS

#: How often the reconstruct job restates what preparation is doing to the
#: frames it is reading. The read loop already reports its progress every 25
#: frames; the filter line rides the same cadence rather than adding one of its
#: own, so a long sweep does not turn the log into a per-frame filter report.
#:
#: What every frame was actually PUT through is no longer a constant here. It is
#: assembled by :func:`scanner3d.preparation.filter_statement` from what the run
#: did, because both filters in that path can withdraw and a fixed sentence
#: saying they ran would be wrong for whichever frames they did not act on.
PREPARATION_PROGRESS_EVERY: Final = 25

#: Voxels the fused field the exported mesh is extracted from may occupy, and so
#: - with the placed extent - the resolution that surface is recovered at.
#:
#: 128^3, the same voxel count the occupancy grid this replaced was allowed, and
#: an eighth of :data:`scanner3d.tsdf.MAX_VOXELS`. Deliberately below that
#: ceiling: the field costs eight bytes a voxel and EVERY frame of a capture is
#: projected into all of them, so the module ceiling would be eight times both
#: numbers below for a resolution finer than the depth image supports at this
#: working distance.
#:
#: Measured on this machine over a synthetic 20-unit sphere at 90, whose placed
#: extent lands the walk on a 126x126x126 field of 2000376 voxels at voxel
#: 0.339: 15.3 MiB of field, 0.096 s to fuse one 512000-byte frame into it - so
#: about 9.6 s over a hundred-frame capture - and 0.428 s to extract the surface
#: from eight fused views. That is against a registration of the same capture
#: that takes minutes. NOT measured on a real capture; the numbers are the
#: arithmetic's cost, not the scanner's.
#:
#: A caller may lower it and nothing may raise it. Named here rather than left
#: to the default so that the number the record reports is a decision this
#: window made.
RECONSTRUCT_VOXEL_BUDGET: Final = 128 * 128 * 128

#: Where the numbers that turned samples into coordinates came from, for the
#: provenance of everything Reconstruct exports. Named here beside the loader
#: that actually applies them, so the claim in the file cannot describe a
#: different record from the one the points came through.
CALIBRATION_SOURCE: Final = (
    "the factory intrinsics record this repository carries as "
    "scanner3d.simulate.FACTORY_INTRINSICS, applied unmodified to every frame of this capture; "
    "no calibration was performed for this scan, and the record's own principal-point "
    "observation stands"
)

#: Aligned frames the live model is built from, however many have been aligned.
#: A rebuild that uses EVERY frame so far costs more the longer the capture runs
#: - measured at 0.008 s over 3 frames and 0.200 s over 51, on a sweep that
#: keeps going for hundreds - and produces a payload that grows with it. Both
#: are wrong for a picture that is thrown away and rebuilt a dozen frames later.
#: Sampling a fixed number of them spread across the whole sweep makes the cost
#: and the size flat, and :func:`_sample_span` is what keeps "spread across the
#: whole sweep" from degenerating into "the beginning" or "the end".
#:
#: This bounds the PREVIEW only. Reconstruct reads every frame off disk and is
#: not affected by anything here.
LIVE_MODEL_CLOUDS: Final = 40

#: The live surface's cell edge, as a multiple of the measured spacing of the
#: points it is first given.
#:
#: NOT a constant edge, because there is no constant that works. A cell only
#: joins a surface when its neighbours are occupied too, so the edge has to be
#: about the distance between samples - and that distance is a property of the
#: object's size, of how far away it is and of the decimation the loader applied,
#: none of which this window knows in advance. Measured instead, once, by
#: :func:`~scanner3d.denoise.estimate_spacing` over the geometry that exists at
#: the first refresh.
#:
#: A little over one spacing, so that neighbouring samples land in neighbouring
#: cells and the occupancy is a connected sheet rather than a scatter of
#: isolated cells - which marches to nothing at all, and would leave the panel
#: empty for the first minute of every capture.
#:
#: A starting guess and not an answer, because the quantity that actually
#: decides whether a sheet is connected is the largest GAP between samples and
#: this measures the typical distance between them. Overlapping frames make the
#: two differ a lot: forty views of one patch put many samples very close
#: together and leave the gaps between patches exactly as wide as they were. So
#: the guess is checked against the outcome - see
#: :data:`LIVE_MESH_COARSEN_ATTEMPTS`.
#:
#: Sized ONCE and never resized. The scan gets denser as it goes and a finer
#: edge would suit it better later, but coarsening is the only direction this
#: grid moves: refining would mean re-reading points that were discarded when
#: they were folded in. So the live surface is a preview at the resolution the
#: scan's first few frames could support, and the exported mesh - fused from
#: every raw frame at :data:`RECONSTRUCT_VOXEL_BUDGET` - is the detailed one.
LIVE_MESH_SPACING_MULTIPLE: Final = 1.5

#: How many times the live grid may double its cell edge because the surface it
#: marched had no faces in it.
#:
#: An empty march is not a failure and not an error: it is what a grid whose
#: cells are isolated produces, correctly, because an isolated cell straddles no
#: node and there is nothing for the level to cut. It is also indistinguishable
#: on the screen from a scan that has not started. So rather than trusting the
#: spacing estimate above, the answer is checked: no faces means the cells are
#: too small for how far apart the samples are, and the edge doubles.
#:
#: Bounded, and cheap to reach: each doubling makes the next march about eight
#: times cheaper, so the whole retry sequence costs less than the first attempt.
#: Past the bound the honest answer is that this scan has no surface yet, and
#: the panel says so instead of coarsening until anything at all appears.
#:
#: A budget for the whole CAPTURE and not for one refresh. Per refresh it would
#: not be a bound at all: a scan that never produces a surface - one of air, or
#: of something too dark to return depth - would spend it again every few
#: frames, and after a couple of minutes the cell edge would be a number with no
#: relationship to anything that was measured.
LIVE_MESH_COARSEN_ATTEMPTS: Final = 4

#: Grid cells the live surface may occupy before the preview is WITHDRAWN.
#: A time budget, since the fill and the march are both linear in cells and both
#: run on the stream thread - but it is a fidelity floor first.
#:
#: The number it replaces was 250000, and what happened at that boundary was
#: worse than a slow refresh. Crossing it called
#: :meth:`~scanner3d.mesh.SurfaceAccumulator.coarsen_to_fit`, which DOUBLED the
#: cell edge of the one accumulator the capture had and carried on as though
#: nothing had changed. The owner's capture 20260827-143636 ran at voxel 2.713
#: to 246796 cells over 35 frames and then showed the same object at 5.425 over
#: 39900; 20260827-143407 did the same thing at frame 123 of 158, going from
#: 2.472 to 4.944. The occupancy mesher's surface already stands as much as a
#: voxel outside the samples, so doubling the edge doubles that error - which is
#: the blob the owner reported beside dots whose fidelity had not changed.
#:
#: Sized from those two captures rather than picked. The log records the
#: COARSENED grids they finished on - 97760 cells at 4.944 and 73892 at 5.425 -
#: and coarsening maps a cell index to ``(i + 1) // 2``, so a fine span is at
#: most twice the coarse span plus one and a fine grid is therefore strictly
#: under eight times the coarse one. The fine grids those two captures would
#: have KEPT are bounded above by 782080 and 591136 cells. A million covers both
#: with better than a quarter to spare, and is under half
#: :data:`~scanner3d.mesh.MAX_GRID_CELLS`, so the module's own ceiling never
#: decides this first.
#:
#: What that costs, measured on this machine over a synthetic sweep shaped like
#: those captures - a 230 mm shell, 784 points a frame at 1.8 mm spacing, cell
#: edge sized from the first three frames the way a capture sizes it:
#:
#:     cells      faces    march
#:      253890    41468    0.32 s
#:      514176    73524    0.61 s
#:      755634   106348    0.90 s
#:     1018961   154376    1.30 s
#:
#: Linear in cells, as the fill and the march both are. 1.30 s at the ceiling is
#: three times the 0.43 s the old budget chose, which is exactly why a cell
#: count is no longer the only bound: :data:`LIVE_MODEL_DUTY` is what keeps a
#: refresh that expensive from taking the stream thread with it.
#:
#: Past this the grid is neither coarsened nor re-attempted. See ``grow_surface``
#: in :meth:`ScannerWindow._start_capture`, which withdraws the surface, says
#: why, and leaves the points as the picture.
LIVE_MESH_CELLS: Final = 1_000_000

#: The share of the stream thread one live model refresh may take.
#:
#: The refresh runs between frames, after the driver has its buffer back, and
#: everything queued behind it waits: the next dequeue, the depth preview, the
#: colour pump's turn, the Stop poll and a wanted Pause. At the old 250000-cell
#: budget the worst refresh was 0.43 s against a frame every ~0.7 s and the
#: question did not really arise. At :data:`LIVE_MESH_CELLS` it can be 1.3 s,
#: and a bound that was only a cell count would spend that every
#: :data:`COVERAGE_EVERY` alignments however long it took.
#:
#: So the cost of the LAST refresh is measured, and the next one is held off
#: until the thread has had at least three times that long to itself. A quarter
#: rather than a half because the rest of that thread is the capture: the write,
#: the deprojection and the alignment that each accepted frame already costs.
#:
#: Nothing is skipped and nothing goes stale. The points and the surface are
#: still built together and published together; a refresh that is held off is
#: simply taken a frame or two later.
LIVE_MODEL_DUTY: Final = 0.25

#: How expensive a refresh has to have been before :data:`LIVE_MODEL_DUTY`
#: applies to it at all.
#:
#: Roughly what one aligned frame already costs this thread - the write, the
#: deprojection, the speckle filter and the ICP fit - so a refresh below it is
#: cheaper than the frames it is waiting for and holding it off would buy the
#: capture nothing while costing the operator feedback. It is also just above
#: the 0.43 s worst case the old 250000-cell budget chose, which means every
#: refresh that would have been affordable before this task is still taken on
#: exactly the old cadence: this gate is only ever reached by the large grids
#: raising :data:`LIVE_MESH_CELLS` made possible, from around 600000 cells up.
LIVE_MODEL_REFRESH_FLOOR: Final = 0.7

#: Ceiling on the points in one live model, applied where the model is BUILT.
#: Below :data:`~scanner3d.cloud_view.MAX_RENDER_POINTS`, so the view never has
#: to throw any of it away: a payload the producer bounds is one that is never
#: assembled, copied across a thread boundary and held in full just to be
#: decimated at the far end.
LIVE_MODEL_POINTS: Final = 45_000

#: Whether the model panel starts with the SURFACE selected once a scan has one.
#:
#: False, which is the owner's decision of 2026-08-27 and reverses what this
#: panel did when the live surface was added. The occupancy preview is not a
#: fit to the samples: a cell joins the surface when its neighbours are occupied
#: too, so what is marched is faceted at the cell edge, stands as much as one
#: cell OUTSIDE the points it was built from, and neither smooths nor refits.
#: Two accepted poses that disagree by a little are drawn as solid geometry
#: rather than as two clouds slightly apart.
#:
#: Not the coarsening the previous task removed, and not a cost problem. The
#: owner's capture 20260827-152842 wrote 116 frames, held cell 1.981 from its
#: first refresh to its last, grew to 774300 cells and 66616 faces and never
#: coarsened - and the owner still directly observed that surface as lumpy
#: beside points that looked clean. So the finest surface this preview can build
#: is still the less trustworthy of the two pictures, and the one the panel
#: opens on is the points.
#:
#: The preview is KEPT, as an opt-in diagnostic: it answers "what shape has this
#: become" faster than a cloud does, and it is the only live view of what the
#: occupancy grid actually holds. What changes is that the operator asks for it,
#: and that asking is remembered - see :meth:`ScannerWindow._show_surface_control`.
#: A pose-corrected final reconstruction is separate work; nothing here is it.
SURFACE_SELECTED_BY_DEFAULT: Final = False

#: What the model panel's surface control is called and what it says when
#: hovered WHILE IT IS HOLDING THE LIVE PREVIEW. Named constants for the same
#: reason Reconstruct's and Pause's are: the wording is the part of this control
#: that has to be right - it offers a picture the operator is being told NOT to
#: trust over the dots - and naming it here is the only way it is asserted
#: without building a widget.
#:
#: The wording itself lives in :mod:`scanner3d.preview_provenance`, keyed by
#: :class:`~scanner3d.preview_provenance.SurfaceKind`, because this control does
#: not always draw the occupancy grid: once Reconstruct has published, the same
#: box draws the exported signed-distance surface, and a control still
#: describing filled cells and capture-time poses would contradict the
#: provenance line one widget above it. These two names are the LIVE kind's
#: wording; :func:`~scanner3d.preview_provenance.surface_control` is what the
#: window applies, and it is given the surface the panel actually holds.
SURFACE_TOGGLE_TITLE: Final = LIVE_SURFACE_CONTROL_TITLE
SURFACE_TOGGLE_TOOLTIP: Final = LIVE_SURFACE_CONTROL_TOOLTIP

#: How long a stopping stream waits for the colour grabber to notice. A grab
#: that has already begun holds the RGB node for up to its own 8 s deadline, and
#: the wait is bounded rather than absent so a normal stop does not race the
#: next run for that node. Nothing depends on it succeeding: a pump that is
#: still running has already been forbidden to touch the window, and its thread
#: is a daemon, so it can neither confuse the next capture nor hold the
#: application open.
COLOUR_STOP_TIMEOUT_S: Final = 10.0

#: The namespace this window claims under the captures folder. Inside it, each
#: capture gets a freshly created directory - named by whoever started the scan,
#: or dated when they left the name blank: this window writes frame_0000.raw
#: upward, so a second capture sharing a directory would overwrite the first
#: frame by frame. Defined in :mod:`scanner3d.config` so that ``scanner3d
#: config`` reports the same name this uses.
CAPTURE_PREFIX: Final = GUI_CAPTURE_COMMAND

#: Aligned frames a capture needs before Pause is offered at all.
#:
#: Pausing is only useful because the frames collected after it can be placed
#: against the model already accepted, and that model is the reference the
#: decision is made from. Below this there is not enough accepted surface for a
#: global search to mean anything, so the control is disabled rather than
#: offered and then refused - a button that is there and never works is worse
#: than one that arrives when it becomes usable.
#:
#: Eight, which at the default decimation is a couple of seconds of scanning and
#: enough of the object for :func:`~scanner3d.relocalize.bounded_reference` to
#: build a surface rather than a scatter.
MIN_PAUSE_REFERENCE_CLOUDS: Final = 8

#: Frames a wanted pause may wait for a boundary it can be taken at.
#:
#: A pause is only safe immediately after a frame that both reached disk AND
#: aligned, because the bridge is measured against that frame's pose and the
#: durable record relates it to the frame immediately before the segment. In
#: normal scanning the very next frame is such a boundary, so this is almost
#: never reached; it is here so that a run whose alignment has stopped working
#: answers a Pause with a refusal the operator can act on rather than with a
#: control that silently never engages.
#:
#: Larger than :data:`~scanner3d.capture_sequence.TRACKING_LOSS_STREAK`, and
#: that ordering decides which answer a waiting pause gets: three consecutive
#: alignment failures end the capture as a tracking loss before eight deferred
#: frames can accumulate, so a pause waiting through failing alignment is now
#: answered by the loss rather than by the refusal below. That is the stronger
#: answer - it names what is actually wrong instead of only what cannot be done
#: about it - and the refusal is still what answers a pause deferred by frames
#: skipped for any other reason.
PAUSE_BOUNDARY_GRACE: Final = 8

#: What the Pause toolbar action is called and what it says when hovered. Named
#: constants for the same reason Reconstruct's are: the wording is the part that
#: has to be right, and this is the only way it gets asserted without a toolbar.
#:
#: Two titles for one checkable action, because one word cannot describe both of
#: its states. A control still saying "Pause" over a paused capture is telling
#: the operator that pressing it pauses - which is what the owner read it as on
#: 2026-08-28 - so the words follow the state the CAPTURE published, and
#: :data:`RESUME_TITLE` appears only while that state is actually ``paused``.
PAUSE_TITLE: Final = "Pause"
RESUME_TITLE: Final = "Resume"
PAUSE_TOOLTIP: Final = (
    "Stop adding frames so you can pick the object up and put it down\n"
    "differently, then carry on into the same scan.\n\n"
    "While paused the camera views keep running so you can frame the new\n"
    "position, and NOTHING is written or added to the model, and this\n"
    "button becomes Resume.\n\n"
    "When you press Resume the scan collects a few frames and waits until\n"
    "it can see enough surface it has already scanned to work out where the\n"
    "object has moved to. Until it can, those frames are held back and none\n"
    "of them reach the scan. Move back over a part you have already scanned,\n"
    "and over a part that does not look like every other part of the object:\n"
    "a shape that would sit several ways is refused rather than guessed.\n\n"
    "Available once enough of the object has been scanned to recognise."
)

#: What the Pause action is called and says once tracking has been lost.
#:
#: The control cannot be left saying "Pause". Pause exists to let the operator
#: move the object and carry on into the SAME scan, and after a loss there is no
#: same scan to carry on into - the model is frozen and nothing this version has
#: can join what follows to it. So the action is disabled AND relabelled: a
#: disabled control still reading "Pause" says "not right now", and what is true
#: is "not at all, and Stop is what you want".
TRACKING_LOST_PAUSE_TITLE: Final = "Pause (tracking lost)"
TRACKING_LOST_PAUSE_TOOLTIP: Final = (
    "Unavailable: the scan has lost track of where the object is.\n\n"
    "Pause and Resume place new frames against the model already accepted,\n"
    "and that is exactly what has stopped working here.\n\n"
    "The scan tries to find the object again by itself, once, within bounds.\n"
    "If it cannot, nothing more is added to the model.\n\n"
    "Press Stop and rescan. Every frame written before the loss is safe on\n"
    "disk either way."
)

#: What the Pause action is called and says while the capture is trying to find
#: the object again by itself. Disabled for the same reason it is disabled after
#: a loss - the machinery a pause needs is the machinery currently in use - but
#: the words are different because the situation is: this one is working on it.
RECOVERING_PAUSE_TITLE: Final = "Pause (recovering tracking)"
RECOVERING_PAUSE_TOOLTIP: Final = (
    "Unavailable: the scan is trying to work out where the object is again.\n\n"
    "It is collecting a small segment of frames and looking for the model it\n"
    "had already accepted. None of those frames are in the scan, and none of\n"
    "them will be unless the match clears every gate.\n\n"
    "Move back over a part you have already scanned. Stop still works, and\n"
    "every frame written before the loss is safe on disk."
)

LOG = get_logger("gui")


@dataclass(frozen=True, slots=True)
class ScanStatus:
    """One short, current answer for the person operating the scanner.

    ``source`` lets independent evidence coexist: a healthy tracking update
    must not erase a depth-collapse warning, and a recovered depth image must
    not erase a terminal tracking loss.  The window shows the highest-priority
    active answer and keeps the full ``detail`` only as a tooltip and in the
    technical log.
    """

    source: str
    title: str
    summary: str
    action: str
    severity: str
    priority: int
    detail: str = ""


IDLE_SCAN_STATUS: Final = ScanStatus(
    source=SCAN_STATUS_CAPTURE_SOURCE,
    title=IDLE_SCAN_STATUS_TITLE,
    summary=IDLE_SCAN_STATUS_SUMMARY,
    action=IDLE_SCAN_STATUS_ACTION,
    severity="idle",
    priority=0,
)


@dataclass(frozen=True, slots=True)
class PauseState:
    """What the capture is doing about a pause, for the panel and the log.

    Carries the WORDS rather than a code the window would have to translate,
    because the interesting cases - "collecting 2 of 3", each bounded refusal
    reason - are sentences the capture thread already has and the GUI thread
    cannot re-derive. ``paused`` is what the toolbar's checked state follows, so
    a pause the capture REFUSED puts the button back up by itself.
    """

    #: ``"running"``, ``"paused"`` or ``"relocalizing"``.
    state: str
    detail: str
    paused: bool


class PauseControl:
    """The operator's pause request, read by the capture at its own boundary.

    Two values rather than one flag, and the second is the whole reason this is
    a class. ``paused`` is the state the operator currently wants. ``epoch``
    counts how many times a pause has been ASKED FOR, and the capture records
    which epoch it has answered - so a pause and a resume that both land between
    two frames still force the relocalization, instead of the scan quietly
    carrying on with adjacent alignment across a reposition nobody told it
    about. A toggle that is too fast for the capture to see as a state is still
    a toggle it must see as an event.

    Every method is safe from either thread and none of them block: the capture
    thread reads this between a driver buffer arriving and being handed back,
    which is the one place in this program that must never wait on the GUI.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._paused = False
        self._epoch = 0

    def pause(self) -> None:
        with self._lock:
            if not self._paused:
                self._paused = True
                self._epoch += 1

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def state(self) -> tuple[bool, int]:
        """``(paused, epoch)``, read together so they cannot disagree."""
        with self._lock:
            return self._paused, self._epoch

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused


class Stopped(Exception):
    """Raised inside a job when the user pressed Stop.

    Carried as an exception rather than a return value so it unwinds through
    :func:`~scanner3d.v4l2_stream.stream_frames`, whose ``finally`` performs
    STREAMOFF, unmap, release and close. Stopping a capture must leave the
    device as tidy as finishing one.
    """


class LatestOnly:
    """One slot holding the newest payload, in place of a queue that can grow.

    A Qt queued connection is a QUEUE. A worker that emits faster than the GUI
    thread can draw does not slow down - it accumulates, and what the operator
    then watches is a backlog of pictures that were true several seconds ago,
    arriving in order, long after the moment they described. Seen from outside
    the process that is one of the things "the model updates two or three times
    and then freezes" can look like - which is a reason to bound the boundary,
    not a finding about any particular run - and no amount of making the drawing
    faster removes it: a producer with no backpressure will always win
    eventually.

    A depth frame, a colour frame, a saved-frame count and an accumulated model
    are all things where an older payload is not partial information, it is
    WRONG information - the newest one supersedes it completely. So this keeps
    exactly one slot, replaces it in place, and wakes the GUI at most once per
    slot. Fifty frames produced while the GUI is busy cost one redraw of the
    fiftieth, not fifty redraws ending at the fiftieth.

    Deliberately shaped like a Qt signal - :meth:`connect` and :meth:`emit` -
    so that a job producing these payloads reads the same either way, and the
    decision to coalesce lives in ONE place instead of at every call site.

    NOT for anything whose older values still matter. ``progress`` stays an
    ordinary signal: it appends lines to a report, and a coalesced log is a log
    with holes in it.

    :meth:`emit` runs on the producing thread and :meth:`deliver` on the GUI
    one; the slot and the counters are guarded by one lock held only for the
    handful of instructions between them.
    """

    def __init__(self, name: str, notify: Callable[[LatestOnly], None]) -> None:
        self.name = name
        self._notify = notify
        self._lock = threading.Lock()
        self._pending: tuple[object, ...] | None = None
        #: Whether the GUI has already been woken for the current slot and has
        #: not drained it yet. This, not the payload, is what makes N emissions
        #: cost one wake-up.
        self._waiting = False
        self._slots: list[Callable[..., object]] = []
        #: Offered by the producer, applied to the window, and replaced while
        #: still waiting to be drawn. Counted rather than silently dropped
        #: because a physical report has to be able to tell "the model was never
        #: produced" from "it was produced and superseded before it was drawn",
        #: and those two have entirely different causes.
        self.offered = 0
        self.applied = 0
        self.superseded = 0
        #: Payloads dropped by :meth:`discard` because whatever produced them
        #: ended before the window drew them. Separate from ``superseded``: one
        #: is "a newer one replaced it", the other is "it stopped being true".
        self.discarded = 0

    def connect(self, slot: Callable[..., object]) -> None:  # Qt's name
        self._slots.append(slot)

    def emit(self, *payload: object) -> None:  # Qt's name
        """Offer the newest payload. PRODUCER THREAD; never waits for the GUI."""
        with self._lock:
            self.offered += 1
            stale = self._pending is not None
            self._pending = payload
            waiting, self._waiting = self._waiting, True
            if stale:
                self.superseded += 1
        if stale:
            # DEBUG because at capture rates this is the common case and the
            # readable form of it is the one-line summary at the end of the run.
            LOG.debug("%s: an update was superseded before it could be drawn", self.name)
        if not waiting:
            self._notify(self)

    def deliver(self) -> None:
        """Apply whatever is in the slot right now. GUI THREAD.

        Empty is normal and is not an error: two wake-ups can be outstanding
        across a producer that emitted either side of a drain.
        """
        with self._lock:
            payload, self._pending = self._pending, None
            self._waiting = False
        if payload is None:
            return
        self.applied += 1
        for slot in self._slots:
            slot(*payload)

    def discard(self) -> bool:
        """Empty the slot so a queued wake-up applies nothing. ANY THREAD.

        Refusing to produce more is not enough to stop a producer that has
        already ended from reaching the window. One payload can be sitting in
        the slot with its wake-up already posted, and that wake-up is answered
        whenever the GUI next gets a turn - which can be after the stream that
        produced it has been torn down, and after the NEXT one has put its own
        picture on the screen. What the operator then sees is the previous
        scan's frame, arriving on top of the current one, with nothing to
        distinguish it.

        So teardown says so explicitly rather than relying on the emission
        guard. The wake-up itself is left outstanding on purpose: an empty
        delivery is already a normal outcome, and cancelling it is not
        something a producer thread can do to a queued Qt event anyway.

        Returns whether anything was actually dropped, and counts it: a payload
        that was produced and never drawn must not be indistinguishable from
        one that was never produced.
        """
        with self._lock:
            payload, self._pending = self._pending, None
            if payload is None:
                return False
            self.discarded += 1
        LOG.debug("%s: an update was dropped because its stream ended first", self.name)
        return True

    def summary(self) -> str:
        """One line naming what this channel produced, drew, and did not draw."""
        return (
            f"{self.name}: {self.offered} produced, {self.applied} applied, "
            f"{self.superseded} superseded before they could be drawn, "
            f"{self.discarded} dropped when the stream ended"
        )


class Worker(QThread):
    """Runs one long job off the UI thread and reports back by signal."""

    progress = Signal(str)
    finished_ok = Signal(object)
    stopped = Signal()
    failed = Signal(str)
    #: Wakes the GUI for ONE :class:`LatestOnly` channel. It carries the channel
    #: and not the payload, which is the whole difference: by the time this
    #: arrives, what should be drawn is whatever is in that channel's slot then,
    #: not whatever was in it when the wake-up was posted.
    _ready = Signal(object)

    def __init__(self, job: Callable[[Worker], object]) -> None:
        super().__init__()
        self._job = job
        self.cancelled = False
        # A QThread OBJECT lives in the thread that created it - the GUI one -
        # so this is a queued connection from run() and _deliver really does
        # happen on the GUI thread.
        self._ready.connect(self._deliver)
        #: Every kept depth frame. Latest-only: a frame the operator never saw
        #: is not worth catching up on.
        self.preview = LatestOnly("depth preview", self._wake)
        self.colour = LatestOnly("colour preview", self._wake)
        self.coverage = LatestOnly("live model", self._wake)
        #: One saved frame. Deliberately the cheapest payload here - an integer,
        #: no geometry - because it is the only thing that fires on EVERY kept
        #: frame and it is what tells the operator the capture is still alive
        #: between the much rarer model refreshes. Coalescing it loses nothing:
        #: the count only ever climbs, so the newest one is the true one.
        self.saved = LatestOnly("saved-frame count", self._wake)
        self.travel = LatestOnly("rotation speed", self._wake)
        #: What the capture is doing about a pause. Latest-only like the rest:
        #: this is a STATE, so an operator who is two updates behind wants the
        #: current one, not the one before it. Every bounded refusal reason also
        #: goes to :attr:`progress` and to the log, which are not coalesced, so
        #: nothing that has to be readable afterwards depends on this one.
        self.pause_state = LatestOnly("pause state", self._wake)
        #: The one operator-facing answer to "what is happening right now?".
        #: Full diagnostics continue through ``progress``; this channel is
        #: deliberately latest-only because a person moving the scanner needs
        #: the current instruction, not a queue of instructions that have
        #: already ceased to be true.
        self.scan_status = LatestOnly("scan status", self._wake)

    def channels(self) -> tuple[LatestOnly, ...]:
        """Every coalescing channel, for reporting what the run cost the GUI."""
        return (
            self.preview,
            self.colour,
            self.coverage,
            self.saved,
            self.travel,
            self.pause_state,
            self.scan_status,
        )

    def _wake(self, channel: LatestOnly) -> None:
        """PRODUCER THREAD. One posted event, whatever is waiting in the slot."""
        self._ready.emit(channel)

    def _deliver(self, channel: object) -> None:
        """GUI THREAD, by queued connection from :meth:`_wake`."""
        if isinstance(channel, LatestOnly):
            channel.deliver()

    def cancel(self) -> None:
        """Ask the job to stop. Checked by the job at its own safe points."""
        self.cancelled = True

    def check(self) -> None:
        if self.cancelled:
            raise Stopped

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._job(self))
        except Stopped:
            self.stopped.emit()
        # A GUI must show the failure, not die: an unhandled worker exception
        # would leave the window alive but permanently busy with no way to
        # find out why.
        except Exception:
            self.failed.emit(traceback.format_exc())


def read_firmware_version(serial: str) -> str:
    """The scanner's own firmware string, or a reason it could not be read.

    This briefly unbinds ``uvcvideo`` from interface 0, performs one read-only
    60-byte control transfer, and hands the interface back. Measured on this
    hardware the whole sequence takes about 16 ms and streaming works normally
    afterwards - verified by capturing frames immediately after.

    It is worth being explicit that this is the same CLASS of operation that
    wedged this scanner earlier in development, when repeated unbounded control
    traffic left the endpoint returning EPROTO. The difference is that this is
    one bounded transfer with an explicit timeout, and the interface is restored
    in every path. It is nonetheless the most invasive thing this window does,
    which is why it reports what happened rather than failing silently.

    Never raises. A version string is a nice thing to know and never a reason to
    stop the application from opening.
    """
    try:
        from scanner3d.ownership import UsbfsOwnershipBackend
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"
    try:
        outcome = UsbfsOwnershipBackend().run(serial)
    except Exception as exc:
        return f"not read ({type(exc).__name__})"
    response = outcome.as_dict().get("response")
    if isinstance(response, dict):
        version = response.get("firmware_version")
        if isinstance(version, str) and version:
            algorithm = response.get("algorithm_version")
            if isinstance(algorithm, str) and algorithm:
                return f"{version}  (algorithm {algorithm})"
            return version
    return "not read"


def _depth_unit() -> DepthUnitAssumption:
    """What ``raw * scale`` is assumed to measure here, and on whose authority.

    Delegated to :func:`scanner3d.preparation.preparation_unit`, which owns it
    for both entry points. This assumption reaches the deprojection inside the
    shared preparation path AND the provenance written into every exported
    file, and the window used to keep its own copy of the three strings - free
    to drift from the CLI's copy of the same claim without anything noticing.

    Imported inside the function for the reason every heavy import in this
    module is: a window that opens has not yet been asked to deproject
    anything.
    """
    from scanner3d.preparation import preparation_unit

    return preparation_unit()


def _prepare_frame(raw: bytes, target_points: int) -> PreparedCloud:
    """One raw depth frame through the shared preparation path, with its evidence.

    This window and ``scanner3d reconstruct`` each used to write the steps out
    for themselves - deproject, clip, thin, denoise - with a comment in each
    asking the reader to check the other still agreed. They now call
    :func:`scanner3d.preparation.prepare_frame`, which owns the order and the
    thresholds, so the two entry points cannot reconstruct the same captured
    frames into two different clouds. :mod:`scanner3d.preparation` states why
    the steps are in the order they are.

    Returns the whole record rather than just the points, because the
    reconstruction path has to report what the filters did - in particular which
    frames kept their support plane and why - and a count that has to be
    recomputed from the cloud is a count that can disagree with it.
    """
    from scanner3d.preparation import prepare_frame

    return prepare_frame(raw, target_points=target_points)


def _cloud_from_bytes(raw: bytes, target_points: int) -> tuple[tuple[float, float, float], ...]:
    """The prepared cloud alone, for the capture paths that need no record.

    Split out from :func:`_load_cloud` for the frames a resumed capture holds
    TENTATIVELY: they have been delivered and selected, and they must not be
    written as ``frame_*.raw`` until the relocalization that would justify them
    has been accepted, so there is no path to read them back from. The
    arithmetic is the same arithmetic either way, which is the point of it being
    one function - and the same arithmetic again as the reconstruction path,
    which differs only in keeping the record :func:`_prepare_frame` returns.
    """
    return _prepare_frame(raw, target_points).points


def _load_cloud(path: Path, target_points: int) -> tuple[tuple[float, float, float], ...]:
    return _cloud_from_bytes(path.read_bytes(), target_points)


def _coloured_cloud(
    clouds: Sequence[Sequence[tuple[float, float, float]]],
    poses: Sequence[Rigid],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Place every frame in world space and flag the points that need more scanning.

    Coverage is decided per CELL by :mod:`scanner3d.coverage`, then applied back
    to the points that fall in each cell. Colouring the points rather than the
    cells is what makes the picture look like the object instead of like a
    voxel grid, while the judgement stays where it was measured.

    The fourth value flags the points contributed by the LAST cloud given, which
    for a live model is the frame that has just arrived. It is built here rather
    than inferred by a caller counting backwards from the end, because the
    per-cloud ordering of the points is this function's own business and a
    caller that recomputed it would silently mislabel the highlight the first
    time that ordering changed. What the flag MEANS is the caller's: a
    reconstruction hands over frames the optimiser placed all at once, and
    nothing in one of those is new.
    """
    report = coverage_map(list(clouds), list(poses), cell=6.0, min_observations=3)
    thin_cells = {cell.cell_index for cell in report.thin_cells()}
    size = report.cell_size
    points: list[tuple[float, float, float]] = []
    thin: list[bool] = []
    newest: list[bool] = []
    last = len(clouds) - 1
    for index, (cloud, pose) in enumerate(zip(clouds, poses, strict=True)):
        for point in pose.apply_all(cloud):
            points.append(point)
            key = (
                math.floor(point[0] / size),
                math.floor(point[1] / size),
                math.floor(point[2] / size),
            )
            thin.append(key in thin_cells)
            newest.append(index == last)
    return (
        np.array(points, dtype=np.float32).reshape(-1, 3),
        np.array(thin, dtype=bool),
        report.well_covered_fraction,
        np.array(newest, dtype=bool),
    )


def _surface_arrays(surface: Mesh | None) -> tuple[np.ndarray, np.ndarray]:
    """One mesh as the two arrays a view draws, or two empty ones for no mesh.

    The conversion happens on the PRODUCING thread, with the rest of the
    snapshot, so nothing but arrays crosses to the GUI: a tuple of a hundred
    thousand Python tuples is exactly the kind of payload this window bounds
    everywhere else, and unpacking one on the thread that has to stay responsive
    would undo that.
    """
    if surface is None or not surface.faces:
        return (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int32))
    return (
        np.array(surface.vertices, dtype=np.float32).reshape(-1, 3),
        np.array(surface.faces, dtype=np.int32).reshape(-1, 3),
    )


@dataclass(frozen=True, slots=True)
class ModelSnapshot:
    """One accumulated model, carrying the counts it was BUILT from.

    The counts travel WITH the picture instead of being read off the window at
    the moment it is drawn, and that is the whole reason this type exists. A
    snapshot can be superseded, and a snapshot can be drawn a moment after
    several more frames have landed on disk; a panel that says "rebuilt after 41
    frames" while showing the model built at frame 27 is describing something
    that was never on the screen, and it is worse than saying nothing because it
    is believable.
    """

    points: np.ndarray
    thin: np.ndarray  #: Parallel to ``points``: which of them came from the newest contributing
    #: frame. All false for a registered reconstruction, where every frame was
    #: placed at once and calling one of them "new" would be a decoration
    #: pretending to be information.
    newest: np.ndarray
    well_covered: float
    #: Aligned clouds and saved frames AT BUILD TIME, never at draw time.
    aligned: int
    saved: int
    #: How many of those aligned clouds actually contributed points, which is
    #: :data:`LIVE_MODEL_CLOUDS` once a sweep is longer than that.
    sampled: int
    note: str
    #: Whether drawing this should re-frame the view. True at a boundary - the
    #: first model of a capture, a registered reconstruction - and false for
    #: every incremental update, which is what lets the operator keep the pan
    #: and zoom they chose and see the model GROW rather than be renormalised.
    reframe: bool = False
    #: The surface the same model implies, as vertices and triangle indices, or
    #: two empty arrays when there is none yet.
    #:
    #: Carried WITH the points rather than sent separately for the reason every
    #: other count on this record is: a panel showing a surface built at frame
    #: 40 beside points built at frame 52 is describing something that never
    #: existed. They are produced together and they are drawn together.
    surface_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    surface_faces: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.int32))
    #: The cell edge that surface was marched at, or 0.0 when there is none.
    #: Sized once from the scan's own point spacing and then held there: a live
    #: grid is never re-cut coarser to fit a budget, so this number says what
    #: fidelity the picture actually has rather than what it started with.
    surface_voxel: float = 0.0
    #: WHICH surface that is, set by the producer that built it and by nothing
    #: else, and ``None`` when the producer carried no surface to describe.
    #:
    #: This window has two surface producers and they make materially different
    #: claims: the live capture's occupancy grid, marched from cells the poses
    #: accepted AS THE SCAN RAN filled, and Reconstruct's exported
    #: signed-distance surface, fused from the full-resolution frames at
    #: CORRECTED poses. A snapshot that carried a mesh without saying which is
    #: the state in which the panel printed the live preview's sentence under
    #: the final reconstruction, so it is carried here beside the geometry and
    #: is never inferred from :attr:`note`, from :attr:`reframe`, from the
    #: checkbox or from a default.
    surface_kind: SurfaceKind | None = None
    #: Why there is no surface, in the operator's own words, or empty. Empty
    #: covers both the ordinary cases - there IS one, or the capture has not
    #: built its first yet - and is only filled in when a surface that could
    #: have been drawn was deliberately withheld. A panel that simply stopped
    #: showing a surface, or one that showed a coarser surface without saying
    #: so, is the silent degradation this replaced.
    surface_note: str = ""

    @property
    def has_surface(self) -> bool:
        """Whether this snapshot carries a surface worth drawing."""
        return bool(len(self.surface_faces))


def _relocalization_executor() -> Executor:
    """The one worker thread a resume's global search runs on.

    A module-level function rather than a constructor call inside the capture
    for two reasons, and neither is decoration. The first is the contract: this
    search must not run on the stream thread, because that thread is between
    ``VIDIOC_DQBUF`` and the next delivery and everything queued behind it - the
    depth preview, the colour pump's turn, the Stop poll - waits for it; and it
    must not run on the GUI thread either. The second is that a test can replace
    it with an executor it drives by hand, which is the only way to hold a
    search IN FLIGHT and then assert what the capture did while it was.

    One worker, deliberately. A second concurrent search would be a second
    answer for the same segment arriving in an order nothing defines, and the
    retry policy already says a new attempt only happens once a new frame has
    added evidence.

    A thread, not a process, and that carries a limitation worth stating: a pure
    Python search still contends for the interpreter lock. It is survivable here
    because the stream thread spends almost all of its time inside ``poll`` and
    ``ioctl``, which release it, and because the search runs through the
    selected :mod:`scanner3d.compute` backend - so the way to make it genuinely
    parallel is to give it a backend that leaves Python, not to move it back
    onto the thread that owns the device.
    """
    return ThreadPoolExecutor(max_workers=1, thread_name_prefix="scanner3d-relocalize")


@dataclass(slots=True)
class TentativeSegment:
    """Frames a resumed capture is holding until it knows where they belong.

    Nothing in here has been written, counted, drawn or promised. That is the
    whole contract: a resume collects a few frames, tries to work out where the
    object went, and until it can the frames exist only in this object - so a
    resume that never succeeds, a second pause, a Stop, a failure or the window
    closing all leave a capture folder holding exactly what it held before.

    Bounded three ways, because "hold frames until it works" is otherwise a
    memory leak with a friendly name: a frame count, a byte count and a clock.
    When the frame or byte bound is reached the OLDEST tentative frame is
    dropped rather than the newest refused, and the local chain is re-anchored
    on what is left - so the segment keeps describing where the scanner is now,
    which is the thing the operator is currently moving.
    """

    raws: list[bytes]
    clouds: list[tuple[tuple[float, float, float], ...]]
    poses: list[Rigid]
    #: Parallel to the frames: how the link from the PREVIOUS held frame to this
    #: one measured, or ``None`` for the anchor, which has no link. Written into
    #: the durable record beside the bridge, so an accepted segment carries the
    #: evidence for its own chain and not only for where that chain was put.
    edges: list[LocalEdge | None]
    started: float = 0.0
    attempts: int = 0
    #: Set once the attempt, time or size bound is reached, so the operator is
    #: told once and no further CPU is spent on a resume that is not working.
    #: Cleared only by :meth:`clear`, which is the operator pausing again.
    exhausted: bool = False
    #: Why, in the operator's words. Empty while ``exhausted`` is false.
    exhausted_reason: str = ""
    #: Tentative frames dropped by the sliding bound, for the log.
    dropped: int = 0

    @classmethod
    def empty(cls) -> TentativeSegment:
        return cls(raws=[], clouds=[], poses=[], edges=[])

    @property
    def frames(self) -> int:
        return len(self.raws)

    @property
    def held_bytes(self) -> int:
        return sum(len(raw) for raw in self.raws)

    def clear(self, *, now: float = 0.0) -> None:
        self.raws.clear()
        self.clouds.clear()
        self.poses.clear()
        self.edges.clear()
        self.started = now
        self.attempts = 0
        self.exhausted = False
        self.exhausted_reason = ""
        self.dropped = 0

    def add(
        self,
        raw: bytes,
        cloud: tuple[tuple[float, float, float], ...],
        pose: Rigid,
        edge: LocalEdge | None,
    ) -> str:
        """Take one chained tentative frame, then apply the bounds.

        Returns "" when the frame is being held, or the reason it can never be:
        the byte ceiling is a real ceiling, so a frame that alone would exceed it
        is refused rather than admitted and then counted as an exception to the
        bound. That was the previous behaviour and it made
        :data:`~scanner3d.relocalize.MAX_TENTATIVE_BYTES` a number that stopped
        applying exactly when it mattered - the minimum three frames were kept
        whatever their size.
        """
        if MIN_TENTATIVE_FRAMES * len(raw) > MAX_TENTATIVE_BYTES:
            # Refused BEFORE it is held, because the alternative is to hold it
            # and then break the ceiling to keep it: the previous version made
            # the minimum three frames an exception to the byte bound, which is
            # the bound not applying in the one case it was written for.
            return (
                f"a frame of this mode is {len(raw)} bytes, and {MIN_TENTATIVE_FRAMES} of them "
                f"do not fit inside the {MAX_TENTATIVE_BYTES}-byte ceiling on frames held "
                "back. A segment this decision could be made from can never be collected."
            )
        self.raws.append(raw)
        self.clouds.append(cloud)
        self.poses.append(pose)
        self.edges.append(edge)
        # ``> 1`` rather than ``> MIN_TENTATIVE_FRAMES``: the bound holds at
        # every size, and a segment squeezed below the minimum by it is a resume
        # that cannot be decided rather than a reason to break the bound.
        while self.frames > MAX_TENTATIVE_FRAMES or (
            self.frames > 1 and self.held_bytes > MAX_TENTATIVE_BYTES
        ):
            self._drop_oldest()
        return ""

    def _drop_oldest(self) -> None:
        """Forget the front of the buffer and re-anchor the chain on the rest.

        The poses are a chain whose first entry is the identity by construction,
        so dropping the frame that identity refers to without re-anchoring would
        leave every remaining pose expressed against a frame that is no longer
        in the segment. It would still align, and the transform it produced
        would be wrong by exactly the step that was dropped - which is the kind
        of error that survives every check except this comment.
        """
        if self.frames <= 1:
            return
        del self.raws[0]
        del self.clouds[0]
        base = self.poses[1].inverse()
        del self.poses[0]
        self.poses[:] = [base.compose(pose) for pose in self.poses]
        # The dropped frame's own link went with it, and the link that described
        # the step INTO the new anchor no longer describes anything in this
        # segment: the anchor is where the chain now starts.
        del self.edges[0]
        if self.edges:
            self.edges[0] = None
        self.dropped += 1

    def chain_record(self) -> dict[str, object]:
        """The local chain as the durable record embeds it: gates and edges."""
        return {
            "gates": local_chain_gates(),
            "edges": [edge.as_dict() for edge in self.edges if edge is not None],
            "anchor_frames": sum(1 for edge in self.edges if edge is None),
        }


@dataclass(frozen=True, slots=True)
class ReconstructionOutcome:
    """What a finished Reconstruct produced: the poses, and the files.

    Both, together, because the report has to name both and neither can be
    re-derived from the other. Keeping them in one value also means the
    completion callback cannot be reached with an export that has not happened:
    this object is built after the writing is done, on the same line the job
    returns.
    """

    registration: SweepReconstruction
    export: ReconstructionExport


def _sample_span(
    clouds: Sequence[Sequence[tuple[float, float, float]]],
    poses: Sequence[Rigid],
    limit: int,
) -> tuple[list[Sequence[tuple[float, float, float]]], list[Rigid]]:
    """At most ``limit`` frames, spread evenly across the WHOLE sweep.

    Not the first ``limit`` and not the last ``limit``, both of which are a
    cheaper way to bound this and both of which answer the model panel's one
    question wrongly. "Have I pointed the scanner at that yet" is about the
    object, so a model that quietly stops including anything after frame 40
    reports an object that stops there, and one that keeps only the last 40
    forgets the half already scanned and sends the operator back over it.

    The first and last frames are always kept: the last is the one that has just
    arrived, and the first is where the sweep started from.
    """
    total = len(clouds)
    if total <= limit:
        return list(clouds), list(poses)
    picks = [round(index * (total - 1) / (limit - 1)) for index in range(limit)]
    return [clouds[index] for index in picks], [poses[index] for index in picks]


def _live_snapshot(
    clouds: Sequence[Sequence[tuple[float, float, float]]],
    poses: Sequence[Rigid],
    *,
    saved: int,
    aligned: int,
    reframe: bool = False,
    surface: Mesh | None = None,
    surface_voxel: float = 0.0,
    surface_note: str = "",
) -> ModelSnapshot:
    """The disposable "scanned so far" model, bounded where it is BUILT.

    Two bounds, both applied here rather than in the view. Which frames
    contribute is bounded by :data:`LIVE_MODEL_CLOUDS`, which is what keeps the
    cost of a rebuild flat as the sweep grows; how many points cross to the GUI
    is bounded by :data:`LIVE_MODEL_POINTS`, which is what keeps a payload from
    being assembled in full, copied across a thread boundary and held, only for
    the view to throw most of it away at the far end.

    Neither bound touches the capture. Every frame is on disk at full length and
    Reconstruct reads all of them; what is sampled here is only the picture.

    Both bounds carry the "this is the newest frame" flags with the points they
    describe. That is what makes the highlight survivable: :func:`_sample_span`
    always ends its span on the frame that has just arrived, so there is always
    something to highlight, and the point limiting below steps all three arrays
    together so the flags never come loose from the geometry.

    ``reframe`` is passed through to the view rather than decided here. The
    producer knows whether this is the first model of a capture; the widget
    knows what a camera is; neither should be guessing at the other's job.

    ``surface`` is the mesh the caller's accumulator currently implies, or
    ``None``. It is NOT built here and it is not built from ``clouds``: it comes
    from every frame the capture has accepted, not the
    :data:`LIVE_MODEL_CLOUDS` this samples, and it is the caller that owns the
    grid those frames were folded into. ``surface_note`` travels with it and is
    the caller's reason for a surface it chose not to build; it is passed
    through untouched, since only the caller knows why.

    The points cross to the view in the UPRIGHT presentation frame, through the
    same :func:`~scanner3d.reconstruction_export.upright_pose` composition
    Reconstruct exports through. One frame for both, because there is only one
    :class:`~scanner3d.cloud_view.CloudView`: it draws +y up and +z towards the
    viewer, and handing it the depth camera's optical frame - +y DOWN the image,
    +z away - would draw the live model upside down and back to front while the
    registered model beside it was neither. The poses themselves are untouched;
    this composes a copy for the picture and nothing else sees it.
    """
    sampled_clouds, sampled_poses = _sample_span(clouds, poses, LIVE_MODEL_CLOUDS)
    points, thin, well_covered, newest = _coloured_cloud(
        sampled_clouds, [upright_pose(pose) for pose in sampled_poses]
    )
    if len(points) > LIVE_MODEL_POINTS:
        # Every mask with the points, on the same step. Decimating the points
        # alone would leave flags indexed against an array that no longer
        # exists, which numpy answers with either an exception or - worse - a
        # picture whose colours have quietly slid off the geometry.
        step = len(points) // LIVE_MODEL_POINTS + 1
        points, thin, newest = points[::step], thin[::step], newest[::step]
    # NOT decimated with the points. The surface is bounded by its own grid -
    # every vertex is on a cell edge, and the accumulator's cell budget is what
    # limits how many there can be - so a stride here would punch holes in a
    # closed surface to save nothing the producer had not already saved.
    surface_vertices, surface_faces = _surface_arrays(surface)
    return ModelSnapshot(
        points=points,
        thin=thin,
        newest=newest,
        well_covered=well_covered,
        aligned=aligned,
        saved=saved,
        sampled=len(sampled_clouds),
        note=f"preview after {aligned} aligned frames of {saved} saved -",
        reframe=reframe,
        surface_vertices=surface_vertices,
        surface_faces=surface_faces,
        surface_voxel=surface_voxel if len(surface_faces) else 0.0,
        # Stated by the producer, beside the mesh it describes: this function IS
        # the live path, and what it carries is the capture-time occupancy grid.
        # Carried only when there is one, exactly as the cell edge is, so that a
        # snapshot with no surface makes no claim about a surface.
        surface_kind=SurfaceKind.LIVE_OCCUPANCY if len(surface_faces) else None,
        # Carried whether or not there is a surface, because it is the sentence
        # that distinguishes "no surface yet" from "a surface this window
        # refused to draw at the fidelity it could have drawn it".
        surface_note=surface_note,
    )


class ColourPump:
    """Pulls colour frames for as long as a depth stream is up, and no longer.

    MEASURED: the RGB camera delivers nothing on its own - a lone capture times
    out after 8 s with no buffer - but returns frames reliably while a depth
    stream is running. The vendor's own depth-first ordering is therefore a
    requirement, not a habit, and it is why :meth:`start` is called from the
    first delivered depth frame rather than before the stream opens.

    It is specifically STREAMING that is required, not merely the depth node
    being open: holding the node open with no STREAMON still times out at 8 s.
    So a colour preview cannot be had with the projector dark. This project has
    separately failed to darken the emitter by any means - registers 0x922,
    0xb04, 0xb01, 0xb00 and the device's own actuator ioctl were all accepted
    while the owner watched it stay lit - so the blue light while colour is
    running is not a bug and is not currently avoidable.

    ONE thread per stream. Not one per frame, and never one that outlives the
    stream that started it: a grabber still running after a capture ends holds
    the RGB node the NEXT preview or capture wants, and keeps pushing pictures
    into a window that has moved on to a different scan. :meth:`stop` therefore
    runs from the stream's ``finally`` - on completion, on Stop and on failure
    alike - and it both ends the loop and closes the door on any picture already
    in flight.
    """

    def __init__(
        self,
        serial: str,
        emit: Callable[[bytes], None],
        *,
        should_stop: Callable[[], bool],
    ) -> None:
        self._serial = serial
        self._emit = emit
        self._should_stop = should_stop
        # Reentrant because :meth:`_publish` holds it ACROSS the emission - that
        # is what makes the "stopped means stopped" guarantee airtight - and a
        # slot on the far end is entitled to answer a picture by stopping the
        # pump that sent it.
        self._lock = threading.RLock()
        self._stopped = False
        self._thread: threading.Thread | None = None
        #: Pictures handed to the window, and grabs that produced none.
        self.frames = 0
        self.failures = 0

    def start(self) -> None:
        """Begin pumping, at most once. Called once DEPTH has delivered."""
        with self._lock:
            if self._stopped or self._thread is not None or self._should_stop():
                return
            self._thread = threading.Thread(target=self._run, name="colour-pump", daemon=True)
            self._thread.start()
        LOG.info("colour pump started, after the first depth frame")

    def stop(self) -> None:
        """End the pump and forbid any further emission. Bounded, idempotent."""
        with self._lock:
            already, self._stopped = self._stopped, True
            thread = self._thread
            # A stop reached from the pump's OWN thread - a colour slot deciding
            # it has seen enough - cannot join itself, and has no need to: the
            # flag is set, so nothing further is published and the loop ends at
            # its next turn.
            joinable = thread is not None and thread is not threading.current_thread()
        if already or thread is None:
            return
        if joinable:
            thread.join(COLOUR_STOP_TIMEOUT_S)
            if thread.is_alive():
                LOG.warning(
                    "the colour grabber has not finished within %.0fs; it can no longer reach "
                    "the window and it is a daemon thread, so it cannot outlast the application",
                    COLOUR_STOP_TIMEOUT_S,
                )
        LOG.info(
            "colour pump ended: %d pictures shown, %d grabs without one",
            self.frames,
            self.failures,
        )

    def _done(self) -> bool:
        with self._lock:
            if self._stopped:
                return True
        return self._should_stop()

    def _publish(self, frame: bytes) -> None:
        """Hand one picture to the window, unless the pump has been stopped.

        Under the same lock :meth:`stop` sets its flag with, so a picture
        grabbed in the instant before Stop cannot arrive in a window that has
        already moved on. Cheap to hold: the far end is a one-slot channel that
        does not wait for the GUI.
        """
        with self._lock:
            if self._stopped:
                return
            self.frames += 1
            self._emit(frame)

    def _run(self) -> None:
        """The pump itself. COLOUR THREAD. Never raises into the depth stream."""
        import tempfile

        from scanner3d.rgb_frame import V4l2RgbFrameBackend

        statuses = 0
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            frame = target / "rgb-frame.jpg"
            backend = V4l2RgbFrameBackend()
            while not self._done():
                try:
                    outcome = backend.run(self._serial, target)
                except Exception as exc:
                    # Colour is a convenience for framing. Losing it must never
                    # take the depth stream down with it - but it must not
                    # vanish silently either, or a blank colour panel is
                    # unexplainable. Reported once and then given up on: a
                    # grabber that raised on the device is not going to succeed
                    # on the next turn of this loop, and retrying it twice a
                    # second for a four-minute capture is a thousand identical
                    # lines in a log whose only purpose is being read afterwards.
                    self.failures += 1
                    LOG.warning("colour grab failed, giving up: %s", exc)
                    return
                status = outcome.as_dict().get("status")
                if status != "ok":
                    # Not a failure of the run - the depth stream is unaffected
                    # and the next grab may well succeed - so it is reported on
                    # the same "first few, then summaries" budget the live model
                    # failures use rather than once per grab.
                    statuses += 1
                    self.failures += 1
                    if statuses <= FAILURES_REPORTED:
                        LOG.info("colour grab status %s (%d so far)", status, statuses)
                    elif statuses % FAILURE_SUMMARY_EVERY == 0:
                        LOG.info("colour grab status %s, %d times now", status, statuses)
                if frame.exists():
                    payload = frame.read_bytes()
                    frame.unlink(missing_ok=True)
                    self._publish(payload)


def stream_with_colour(
    worker: Worker,
    serial: str | None,
    on_frame: Callable[[bytes, FrameRecord], None],
    *,
    frames: int,
    decimate: int | Cadence,
    duration_s: float,
    after_frame: Callable[[bytes, FrameRecord], None] | None = None,
    on_delivery: Callable[[bytes, FrameRecord], None] | None = None,
) -> StreamResult:
    """Run the depth stream with the colour camera pumped alongside it.

    Shared by preview and capture because the ORDER is a device requirement
    rather than a preference. Launching the grabber first meant it sat timing
    out while depth was still starting and the first picture took 22 s to appear
    instead of about one; and starting no grabber at all during a capture - which
    is what happened until now - meant the colour panel froze for the entire
    sweep, at exactly the moment the operator is deciding where to point next.

    ``serial`` may be ``None``. A scanner whose serial could not be read is
    still perfectly capturable, and colour is the part that is given up.

    ``on_delivery`` is offered every COMPLETE delivery rather than every keep,
    and is how the depth view moves at the rate the scanner is delivering while
    the writing and the alignment stay decimated. It is optional because the
    preview path runs at ``decimate=1``, where the two are the same frames.

    ``decimate`` is passed straight through and may be a controller as well as a
    number - see :func:`~scanner3d.v4l2_stream.stream_frames`. Nothing here
    reads it, so a spacing that varies changes neither the colour pump nor the
    order the two streams are started in.

    ``after_frame`` is offered the same frames as ``on_frame``, after the driver
    has its buffer back. Passed straight through; see
    :func:`~scanner3d.v4l2_stream.stream_frames` for why the distinction
    matters.
    """
    from scanner3d.v4l2_stream import stream_frames

    pump = (
        ColourPump(serial, worker.colour.emit, should_stop=lambda: worker.cancelled)
        if serial
        else None
    )

    started = False

    def delivered(buf: bytes, record: FrameRecord) -> None:
        # Depth first, always. The frame is what the run is FOR, and starting
        # the pump is the last thing done with the FIRST one - once, here, and
        # not once per frame. :meth:`ColourPump.start` refuses a second thread
        # by itself as well, but a callback that asks per frame is a callback
        # one edit away from meaning it.
        #
        # The FIRST COMPLETE DELIVERY, which at a decimation above one arrives
        # before the first keep does. What the RGB camera needs is a depth
        # stream that is running, and a delivery is proof of exactly that;
        # waiting for a frame this run happens to want to save would delay the
        # colour view for no reason the device knows about.
        nonlocal started
        if on_delivery is not None:
            on_delivery(buf, record)
        if pump is not None and not started:
            started = True
            pump.start()

    try:
        return stream_frames(
            frames=frames,
            decimate=decimate,
            duration_s=duration_s,
            on_frame=on_frame,
            after_frame=after_frame,
            on_delivery=delivered,
            # Stopping through the flag rather than by raising: the stream
            # re-queues the buffer first, and every frame already written stays
            # written.
            should_stop=lambda: worker.cancelled,
        )
    finally:
        # Every path out: the stream's own end, the operator's Stop, and an
        # exception on the way through. A pump that survives any of them keeps
        # the RGB node away from the next run and keeps drawing into a window
        # that is no longer showing this capture.
        if pump is not None:
            pump.stop()
        # AFTER the stop, and never before it: the stop is what guarantees no
        # further picture can be published, so this is the last one there will
        # ever be. Stopping the pump closes the door on future emissions but
        # says nothing about a picture already in the slot with its wake-up
        # posted - and that one, drawn whenever the GUI next gets a turn, would
        # land in whatever run is on screen by then. Frames already APPLIED are
        # untouched: this empties one slot, it does not undraw anything.
        if worker.colour.discard():
            LOG.info("colour: a picture offered before teardown was dropped rather than shown")


@dataclass(frozen=True, slots=True)
class _FrameCounts:
    """What a finished capture folder holds, split by what it can be used for.

    Two numbers rather than one, because after a recovery they are genuinely
    two different facts and only the first is a scan: ``frames`` are the frames
    the reader hands over as one reconstructable sequence, and ``excluded`` are
    the frames written while tracking was lost, kept exactly where they were as
    evidence and deliberately not part of it. Reporting only the first made a
    recovered capture finish on a SMALLER number than the live saved counter had
    already shown, with nothing said about the difference.
    """

    frames: int
    excluded: int

    @property
    def total(self) -> int:
        """Every raw frame on disk - which is what the live counter counted."""
        return self.frames + self.excluded


def _frame_counts(directory: Path | None) -> _FrameCounts:
    """What a finished capture left, counted authoritatively.

    Through :func:`~scanner3d.capture_sequence.read_capture_sequence` rather
    than a glob, because a capture that was repositioned mid-sweep keeps the
    frames after each boundary in that boundary's own directory - and a glob of
    the root would report the capture as having stopped at the pause. A folder
    the reader refuses is reported as nothing rather than guessed at: these
    numbers go into a sentence telling the operator their frames are safe, and
    that sentence must not be the only place a broken capture folder is
    discovered.
    """
    if directory is None:
        return _FrameCounts(frames=0, excluded=0)
    try:
        sequence = read_capture_sequence(directory)
    except (CaptureSequenceUnreadable, OSError) as exc:
        LOG.warning("cannot count the frames in %s: %s", directory, exc)
        return _FrameCounts(frames=0, excluded=0)
    return _FrameCounts(frames=len(sequence.frames), excluded=len(sequence.excluded))


def _excluded_note(counts: _FrameCounts) -> str:
    """One sentence accounting for frames on disk that are not the scan, or "".

    Said only when there are any, and always beside the reconstructable count
    rather than instead of it. A capture that recovered from a tracking loss
    wrote more files than it hands over, and the live saved counter counted all
    of them - so an ending that printed only the smaller number would look like
    frames had gone missing, and one that printed only the larger would be
    calling failure evidence part of the scan.
    """
    if not counts.excluded:
        return ""
    return (
        f"{counts.excluded} further frame(s) - {counts.total} files in all - were written while "
        "tracking was lost. They are still on disk exactly where they were written, as the "
        f"evidence for that loss, and they are NOT part of the {counts.frames} reconstructable "
        "frames above."
    )


#: The pause-channel state a capture publishes exactly once, when it has
#: decided that tracking is lost. It travels on that channel because it drives
#: the same banner, the same label and the same control the pause states drive,
#: and a second channel for one terminal message would be a second thing to
#: keep in step with them.
TRACKING_LOST_STATE: Final = "tracking_lost"

#: The state a capture publishes INSTEAD of the one above when the loss it has
#: contained could not be made durable. Two states rather than one, because they
#: are two different facts and only one of them may be reported as a loss: the
#: capture stops writing either way, but ``tracking_lost`` says a record of it
#: exists on disk and this says the opposite. A capture that published the first
#: over a directory holding no record would be telling the operator to go and
#: read a file that is not there, and telling a later reader - through the same
#: sentence - that the folder carries its own explanation when it does not.
TRACKING_LOSS_UNRECORDED_STATE: Final = "tracking_loss_unrecorded"

#: The state a capture publishes once its loss is DURABLE and it has begun
#: trying to find the object again. A distinct state rather than a variation on
#: ``tracking_lost``, because the two say different things to the operator and
#: only one of them is over: the loss is recorded either way, but this one is
#: still collecting frames and still looking, and a window showing the terminal
#: banner over a capture that is about to recover would be as wrong as the
#: silent 2026-08-28 capture was in the other direction.
#:
#: It is published only AFTER the record exists, for exactly the reason
#: ``tracking_lost`` is: a recovery is a claim about a loss, and there is nothing
#: to recover from until the loss is on disk.
RECOVERING_TRACKING_STATE: Final = "recovering_tracking"

#: The state a capture publishes when a recovery has been PUBLISHED - the frames
#: and the binding visible under one segment name - and ordinary tracking has
#: resumed. It is the one state that clears the window's lost banner, and it may
#: only be published after that publication returns: a capture that announced a
#: recovery it had not written would be the original failure with a happier
#: sentence on it.
TRACKING_RECOVERED_STATE: Final = "tracking_recovered"


def _tracking_loss_epilogue(directory: Path | None, lead: str) -> str:
    """The last thing an operator is told about a capture that lost tracking.

    Reached only when the loss is still OPEN at the end of the run: a capture
    that recovered clears the window's lost state when it publishes the segment
    that closed it, and ends with the ordinary "press Reconstruct" instead.

    One sentence about what is on disk, one about why Reconstruct will refuse
    it, and no offer to fix it. The frames ARE safe - that is the point of
    containing the loss - but they are not one sweep, and a closing line saying
    "press Reconstruct" over them is exactly the reassurance that made the
    original failure invisible.
    """
    where = f" in {directory}" if directory is not None else ""
    return (
        f"{lead} The scan lost track of where the object was. Every frame written before "
        f"that is still on disk{where}, untouched, and the evidence for the loss is in "
        f"{TRACKING_LOSS_RECORD_NAME} beside them.\n"
        "Reconstruct will REFUSE this folder, on purpose: nothing places the frames taken "
        "after the loss against the ones taken before it. The scan tried to find the object "
        "again and could not, and a refusal is better than a guessed pose - scan the object "
        "again."
    )


def capture_progress_text(
    saved: int, refreshed_at: int | None, refreshes: int, *, running: bool = True
) -> str:
    """What the model panel says about the capture it is showing.

    Two different facts, kept apart on purpose. ``saved`` advances on every
    frame written and is the heartbeat: while that number climbs, the capture is
    alive, whatever the picture below it is doing. ``refreshed_at`` is the saved
    frame count the accumulated model was last rebuilt at, and it moves far less
    often - a rebuild registers every aligned frame so far, so it is deliberately
    occasional.

    Conflating them is what made a working capture look dead: the 3D view
    changes subtly between refreshes on a slowly turning object, and with
    nothing else moving there was no way to tell "still scanning" from "stopped
    responding".

    ``running`` false is the same two counts after the capture has ended, by any
    route. The tense matters: a label still saying "Capturing" over a scan that
    stopped a minute ago is the same lie in the opposite direction.

    A pure function of its arguments so the sentence can be tested without a
    display; the caller owns the label.
    """
    if saved <= 0:
        return "No frames saved yet." if running else "No frames were saved."
    frames = "1 frame" if saved == 1 else f"{saved} frames"
    lead = f"Capturing: {frames} saved." if running else f"Capture ended: {frames} saved."
    if refreshed_at is None:
        tail = (
            "The model below appears once enough of them have aligned."
            if running
            else "Too few of them aligned to build a model here; Reconstruct still can."
        )
    else:
        rebuilds = "1 rebuild" if refreshes == 1 else f"{refreshes} rebuilds"
        tail = f"The model below was last rebuilt after {refreshed_at} of them ({rebuilds})."
    return f"{lead} {tail}"


class ScannerWindow(QMainWindow):
    """Capture and reconstruct, showing both what the scanner sees and what it has."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D Scanner")
        self.resize(1400, 820)
        self._worker: Worker | None = None
        self._firmware_worker: Worker | None = None
        self._preview_worker: Worker | None = None
        self._previous_stream: Worker | None = None
        self._resume_preview_after = False
        self._capture_dir: Path | None = None
        self._firmware = "reading..."
        # The heartbeat's counters. All live on the GUI thread and are written
        # only by its slots; the worker sends numbers, not state.
        self._saved_frames = 0
        self._model_refreshes = 0
        self._model_refreshed_at: int | None = None
        self._capturing = False
        #: The bounded recent-yield window the depth warning is measured
        #: against. One window per RUN: reset where a preview or a capture
        #: BEGINS and where it ENDS by any route, and when the window closes,
        #: so a scan is never judged against the collapse of the scan before
        #: it and no warning outlives the run that raised it.
        self._return_health = ReturnHealthTracker()
        #: Which run the depth panel is currently showing. Bumped by
        #: :meth:`_begin_depth_run` and :meth:`_end_depth_run`; every depth
        #: frame is delivered through a slot that remembers the run it was
        #: connected for (:meth:`_frame_slot`), so a frame emitted by a preview
        #: that has already been stopped is dropped instead of redrawing the
        #: panel or seeding the next run's baseline. A queued signal really can
        #: arrive after the stop - that ordering is what this counter is for,
        #: rather than a hope about timing.
        self._depth_run = 0
        #: The last verdict shown, so the log records TRANSITIONS rather than a
        #: line per frame. Ten frames a second of "still sparse" is a flood
        #: that buries the moment it changed.
        self._return_verdict: str | None = None
        #: The running capture's pause request, or ``None`` between captures.
        #: Created by :meth:`_start_capture` and dropped when the capture ends by
        #: any route, so a Pause pressed against a finished scan cannot reach a
        #: worker that has gone.
        self._pause: PauseControl | None = None
        #: What the capture said when it lost tracking, or ``None`` while it
        #: still knows where the object is. Set once per capture and left set
        #: until the next one starts: the state is terminal, and the sentence
        #: the operator is told at the end of the run has to be the true one.
        self._tracking_loss: str | None = None
        #: What the capture said when it lost tracking AND could not record it.
        #: Kept apart from the field above rather than folded into it, because
        #: the two differ in exactly the thing the operator is told next: that
        #: one has a ``tracking-loss.json`` to point at and this one has none.
        #: Both deny Pause, because both mean the scan no longer knows where the
        #: object is.
        self._tracking_loss_unrecorded: str | None = None
        #: Aligned clouds the last model snapshot was built from. What decides
        #: whether Pause is offered, because it is the accepted geometry a
        #: relocalization would be measured against.
        self._aligned_frames = 0
        #: Independent capture, tracking and depth evidence contributes to one
        #: prominent operator answer.  The highest-priority current answer is
        #: drawn; full evidence remains in the technical log.
        self._scan_statuses: dict[str, ScanStatus] = {IDLE_SCAN_STATUS.source: IDLE_SCAN_STATUS}
        #: Secondary evidence and the event log are collapsed together by
        #: default. Hiding these widgets is presentation only; no evidence is
        #: dropped and no decision changes.
        self._technical_details_visible = False
        self._technical_widgets: list[QWidget | QAction] = []
        #: Compact live model metrics. The verbose model sentence remains
        #: available in technical details.
        self._coverage_well: float | None = None
        self._coverage_needs: float | None = None

        self._build_toolbar()

        # Three views, sized by what each is FOR. Colour and depth are things
        # you glance at to check the scanner is seeing what you think it is;
        # the model is the thing you turn around and interrogate, so it gets
        # the room. Splitters throughout, because which one matters most
        # changes depending on whether you are framing or reviewing.
        monitors = QSplitter(Qt.Orientation.Vertical)
        monitors.addWidget(self._build_colour_panel())
        monitors.addWidget(self._build_depth_panel())
        monitors.setSizes([340, 340])

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(monitors)
        split.addWidget(self._build_model_panel())
        split.setSizes([460, 940])

        centre = QWidget()
        layout = QVBoxLayout(centre)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(split, stretch=3)
        self._technical_toggle = QToolButton()
        self._technical_toggle.setCheckable(True)
        self._technical_toggle.setText("Show technical details")
        self._technical_toggle.setToolTip(
            "Show advanced capture settings, raw frame statistics, provenance, renderer "
            "information and the full event log."
        )
        self._technical_toggle.toggled.connect(self._toggle_technical_details)
        layout.addWidget(self._technical_toggle)
        self._report = QPlainTextEdit()
        self._report.setReadOnly(True)
        self._report.setPlaceholderText("Technical events appear here.")
        self._report.setMaximumHeight(190)
        layout.addWidget(self._report, stretch=1)
        self._register_technical_widget(self._report)
        self.setCentralWidget(centre)

        self.setStatusBar(QStatusBar())
        # After the results panel exists, because a viewer that has ALREADY
        # fallen back - a PySide6 with no OpenGL module at all - reports itself
        # into that panel the moment it is asked.
        self._show_viewer_backend(self._cloud.backend_description)
        self._refresh_device()
        self._start_firmware_read()

    # -- construction ------------------------------------------------------
    def _register_technical_widget(self, widget: QWidget | QAction) -> None:
        """Put one secondary widget or toolbar action behind Details."""
        widgets = getattr(self, "_technical_widgets", None)
        if widgets is None:
            # Small panel-only test fixtures deliberately own no application
            # shell. In that context the panel remains self-describing.
            return
        widgets.append(widget)
        widget.setVisible(self._technical_details_visible)

    def _set_technical_widget_visible(self, widget: QWidget, relevant: bool) -> None:
        """Respect Details while allowing a panel-only fixture to stand alone."""
        shown = getattr(self, "_technical_details_visible", True)
        widget.setVisible(relevant and shown)

    def _toggle_technical_details(self, shown: bool) -> None:
        """Show or collapse every diagnostic surface as one UI choice."""
        self._technical_details_visible = shown
        self._technical_toggle.setText(
            "Hide technical details" if shown else "Show technical details"
        )
        for widget in self._technical_widgets:
            widget.setVisible(shown)

    def _replace_scan_statuses(self, *statuses: ScanStatus) -> None:
        """Begin a new operator context with no stale warning from the last one."""
        self._scan_statuses.clear()
        for status in statuses:
            self._scan_statuses[status.source] = status
        self._render_scan_status()

    def _show_scan_status(self, status: ScanStatus) -> None:
        """Add or replace one source's current answer, then show the strongest."""
        self._scan_statuses[status.source] = status
        self._render_scan_status()

    def _clear_scan_status(self, source: str) -> None:
        """Withdraw one source without disturbing another source's warning."""
        self._scan_statuses.pop(source, None)
        self._render_scan_status()

    def _render_scan_status(self) -> None:
        """Render the highest-priority current status in three short lines."""
        if not hasattr(self, "_scan_status_title"):
            return
        statuses = getattr(self, "_scan_statuses", None)
        status = max(
            statuses.values() if statuses else (),
            key=lambda item: item.priority,
            default=IDLE_SCAN_STATUS,
        )
        style = SCAN_STATUS_STYLES.get(status.severity, SCAN_STATUS_STYLES["idle"])
        detail = status.detail or "\n".join(
            part for part in (status.summary, status.action) if part
        )
        self._scan_status_title.setText(status.title)
        self._scan_status_title.setStyleSheet(style)
        self._scan_status_title.setToolTip(detail)
        self._scan_status_summary.setText(status.summary)
        self._scan_status_summary.setToolTip(detail)
        self._scan_status_action.setText(f"Next: {status.action}" if status.action else "")
        self._scan_status_action.setVisible(bool(status.action))
        self._scan_status_action.setToolTip(detail)

    def _refresh_scan_metrics(self) -> None:
        """Keep the useful live numbers to one glanceable line."""
        saved = getattr(self, "_saved_frames", 0)
        capturing = getattr(self, "_capturing", False)
        coverage_well = getattr(self, "_coverage_well", None)
        coverage_needs = getattr(self, "_coverage_needs", None)
        aligned = getattr(self, "_aligned_frames", 0)
        if saved <= 0:
            text = "Starting scan…" if capturing else "No scan data yet."
        else:
            frame_word = "frame" if saved == 1 else "frames"
            parts = [f"{saved} {frame_word} saved"]
            if coverage_well is not None and coverage_needs is not None:
                parts.extend(
                    (
                        f"{100.0 * coverage_well:.0f}% covered",
                        f"{100.0 * coverage_needs:.0f}% still needs scanning",
                    )
                )
            elif aligned > 0:
                parts.append(f"{aligned} aligned")
            else:
                parts.append("building the first model")
            text = "  •  ".join(parts)
        self._scan_metrics.setText(text)

    def _build_toolbar(self) -> None:
        bar = QToolBar("Controls")
        bar.setMovable(False)
        self.addToolBar(bar)

        self._preview_action = QAction("Preview", self)
        self._preview_action.setCheckable(True)
        self._preview_action.setToolTip(
            "Show what the scanner sees without recording anything.\n"
            "Use it to frame and position the object before committing to a capture."
        )
        self._preview_action.toggled.connect(self._toggle_preview)
        bar.addAction(self._preview_action)

        self._capture_action = QAction("Start capture", self)
        self._capture_action.setToolTip(
            "Record a scan. You are asked for a name first - give it one you will\n"
            "recognise, or leave it blank and today's date and time are used.\n\n"
            "Every scan gets a brand-new folder of its own under the captures\n"
            "folder, so a scan can never be written on top of an earlier one. If\n"
            "the name is already taken, -2, -3 and so on are added."
        )
        self._capture_action.triggered.connect(self._start_capture)
        bar.addAction(self._capture_action)

        # Between Start and Stop, because that is the order they are reached in
        # and because Pause is the one of the three whose availability changes
        # DURING a run: it appears when the scan has enough of the object to
        # recognise itself by, and Stop is never anything but available.
        self._pause_action = QAction(PAUSE_TITLE, self)
        self._pause_action.setCheckable(True)
        self._pause_action.setToolTip(PAUSE_TOOLTIP)
        self._pause_action.setEnabled(False)
        self._pause_action.toggled.connect(self._toggle_pause)
        bar.addAction(self._pause_action)

        self._stop_action = QAction("Stop", self)
        self._stop_action.triggered.connect(self._stop)
        self._stop_action.setEnabled(False)
        bar.addAction(self._stop_action)

        self._recon_action = QAction(RECONSTRUCT_TITLE, self)
        self._recon_action.setToolTip(RECONSTRUCT_TOOLTIP)
        self._recon_action.triggered.connect(self._start_reconstruct)
        bar.addAction(self._recon_action)
        bar.addSeparator()

        self._captures_action = QAction("Captures folder...", self)
        self._captures_action.triggered.connect(self._choose_captures_dir)
        bar.addAction(self._captures_action)
        self._refresh_captures_action()
        ScannerWindow._register_technical_widget(self, bar.addSeparator())

        frames_label = QLabel(" frames ")
        ScannerWindow._register_technical_widget(self, bar.addWidget(frames_label))
        self._frames_spin = QSpinBox()
        self._frames_spin.setRange(30, 5000)
        self._frames_spin.setValue(DEFAULT_FRAMES)
        ScannerWindow._register_technical_widget(self, bar.addWidget(self._frames_spin))

        decimate_label = QLabel(" use 1 frame in every ")
        ScannerWindow._register_technical_widget(self, bar.addWidget(decimate_label))
        self._decimate_spin = QSpinBox()
        self._decimate_spin.setRange(1, 10)
        self._decimate_spin.setValue(DEFAULT_DECIMATE)
        self._decimate_spin.setToolTip(
            "Keeps every Nth frame as it arrives and discards the rest unexamined.\n"
            "It does NOT pick the best frame out of each group - nothing is compared.\n\n"
            "The point is spacing, not quality: frames arrive faster than the object\n"
            "turns, and near-identical frames add capture time without adding\n"
            "information. At the measured cadence, 3 gives roughly two degrees\n"
            "between kept frames on a turntable turning once in two minutes."
        )
        ScannerWindow._register_technical_widget(self, bar.addWidget(self._decimate_spin))

        #: Whether the spacing above is a fixed setting or a starting point.
        #:
        #: Off by default, deliberately. A capture whose frame spacing is a
        #: number the operator typed is a capture they can reason about
        #: afterwards; one that varied is a capture whose record has to be read
        #: to know what it did. That is a fair trade on a hand-carried sweep and
        #: not one to make on somebody's behalf.
        self._hold_cadence = QCheckBox("hold spacing")
        self._hold_cadence.setChecked(False)
        self._hold_cadence.setToolTip(
            "Treats the number to the left as a starting point and then moves it,\n"
            "between 1 and 10, to hold the MEASURED travel between kept frames\n"
            "inside the band that registers well.\n\n"
            "It keeps more frames when you speed up and fewer when you crawl, and\n"
            "any frame the tracker refuses makes it finer at once. It is not a fix\n"
            "for moving too fast: the finest it can go is every frame, which you\n"
            "can also just set here yourself. What it buys is the other end - a\n"
            "sweep so slow that each alignment measures noise rather than motion."
        )
        ScannerWindow._register_technical_widget(self, bar.addWidget(self._hold_cadence))
        ScannerWindow._register_technical_widget(self, bar.addSeparator())

        compute_label = QLabel(" compute ")
        ScannerWindow._register_technical_widget(self, bar.addWidget(compute_label))
        self._backend = QComboBox()
        self._backend.addItems(["auto", "scipy", "numpy", "python"])
        ScannerWindow._register_technical_widget(self, bar.addWidget(self._backend))

        workers_label = QLabel(" workers ")
        ScannerWindow._register_technical_widget(self, bar.addWidget(workers_label))
        self._workers = QSpinBox()
        self._workers.setRange(1, 32)
        self._workers.setValue(1)
        ScannerWindow._register_technical_widget(self, bar.addWidget(self._workers))
        ScannerWindow._register_technical_widget(self, bar.addSeparator())

        refresh = QPushButton("Refresh device")
        refresh.clicked.connect(self._refresh_device)
        bar.addWidget(refresh)

    def _build_colour_panel(self) -> QWidget:
        """The colour camera: for framing, never recorded into the model."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Colour camera - for positioning")
        title.setStyleSheet("font-weight:bold;")
        layout.addWidget(title)
        self._colour_view = QLabel("preview off")
        self._colour_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._colour_view.setMinimumSize(300, 190)
        self._colour_view.setStyleSheet("background:#111; color:#888;")
        layout.addWidget(self._colour_view, stretch=1)
        self._colour_label = QLabel("-")
        self._colour_label.setWordWrap(True)
        layout.addWidget(self._colour_label)
        ScannerWindow._register_technical_widget(self, self._colour_label)
        return panel

    def _build_depth_panel(self) -> QWidget:
        """The depth camera: what actually becomes the model."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Depth camera - what is captured")
        title.setStyleSheet("font-weight:bold;")
        layout.addWidget(title)
        self._view = QLabel("no frames yet")
        self._view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._view.setMinimumSize(300, 190)
        self._view.setStyleSheet("background:#111; color:#888;")
        layout.addWidget(self._view, stretch=1)
        # Directly under the depth image and above the numeric line, because a
        # frame the sensor returned nothing for used to be a black rectangle
        # with a quiet caption under it, which reads as a broken panel rather
        # than as missing data. Prominent, and never modal: it must not take a
        # click, or the focus, from someone whose hands are on the scanner.
        self._return_label = QLabel("")
        self._return_label.setWordWrap(True)
        self._return_label.setStyleSheet(QUIET_RETURN_STYLE)
        layout.addWidget(self._return_label)
        self._return_advice = QLabel("")
        self._return_advice.setWordWrap(True)
        layout.addWidget(self._return_advice)

        self._frame_label = QLabel("-")
        self._frame_label.setWordWrap(True)
        layout.addWidget(self._frame_label)
        ScannerWindow._register_technical_widget(self, self._frame_label)

        # What that picture and those numbers ARE, permanently, without a hover.
        # The panel above shows raw sensor codes and a checker where the sensor
        # returned nothing; neither of those facts was stated anywhere on the
        # screen, so a sparse frame beside a lumpy surface left an operator to
        # infer which of the two contained measurements. Set once at build and
        # never rewritten: it describes the panel, not the frame, so no depth
        # frame, run boundary or return-health verdict may touch it.
        self._depth_provenance = QLabel(RAW_DEPTH_PROVENANCE)
        self._depth_provenance.setWordWrap(True)
        self._depth_provenance.setStyleSheet(PROVENANCE_STYLE)
        layout.addWidget(self._depth_provenance)
        ScannerWindow._register_technical_widget(self, self._depth_provenance)

        # The speed readout sits under DEPTH because it is measured from depth
        # alignment - it describes the data being captured, not the colour view.
        self._speed_label = QLabel("rotation speed: waiting for motion")
        self._speed_label.setStyleSheet("font-size:13pt; font-weight:bold; padding:2px;")
        self._speed_label.setWordWrap(True)
        layout.addWidget(self._speed_label)
        self._speed_advice = QLabel("")
        self._speed_advice.setWordWrap(True)
        layout.addWidget(self._speed_advice)
        return panel

    def _build_model_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("What has been scanned so far")
        title.setStyleSheet("font-weight:bold;")
        layout.addWidget(title)
        self._scan_status_title = QLabel("")
        self._scan_status_title.setWordWrap(True)
        layout.addWidget(self._scan_status_title)
        self._scan_status_summary = QLabel("")
        self._scan_status_summary.setWordWrap(True)
        self._scan_status_summary.setStyleSheet("font-size:13pt; padding:2px;")
        layout.addWidget(self._scan_status_summary)
        self._scan_status_action = QLabel("")
        self._scan_status_action.setWordWrap(True)
        self._scan_status_action.setStyleSheet("font-size:13pt; font-weight:bold; padding:2px;")
        layout.addWidget(self._scan_status_action)
        self._scan_metrics = QLabel("")
        self._scan_metrics.setWordWrap(True)
        self._scan_metrics.setStyleSheet("font-weight:bold; padding:2px;")
        layout.addWidget(self._scan_metrics)
        self._cloud = CloudViewer()
        layout.addWidget(self._cloud, stretch=1)
        #: The triangle count and cell edge of the surface the panel currently
        #: HOLDS, or ``None`` when it holds none. Kept beside the viewer rather
        #: than read back off it, because the numbers belong to the model that
        #: was published and the viewer is entitled to stride a large mesh down
        #: before drawing it. Written in exactly one place - :meth:`_show_coverage`,
        #: at the same moment the surface itself is given to or taken from the
        #: viewer - so the two can never describe different models.
        self._surface_evidence: SurfaceEvidence | None = None
        # Immediately under the picture and above everything about how to move
        # it: what this view is showing, in one line, always present. Which of
        # the two pictures it describes is derived from the mounted viewer in
        # :meth:`_mounted_picture` - not from the checkbox below and not from
        # the remembered preference - so a surface that could not be drawn
        # cannot be described as though it were on the screen.
        self._model_provenance = QLabel(model_provenance(ModelPicture.NOTHING))
        self._model_provenance.setWordWrap(True)
        self._model_provenance.setStyleSheet(PROVENANCE_STYLE)
        layout.addWidget(self._model_provenance)
        ScannerWindow._register_technical_widget(self, self._model_provenance)
        key = QLabel(
            "left-drag to turn (all the way over, in any direction)   "
            "shift+left-drag to roll   middle-drag to move   wheel to zoom        "
            "RED = still needs scanning"
        )
        key.setWordWrap(True)
        layout.addWidget(key)
        ScannerWindow._register_technical_widget(self, key)
        # A toggle and not a replacement, and one the operator has to reach for.
        # A surface is easier to read as a shape, but while a capture runs it is
        # an occupancy preview that stands outside its own samples and was
        # observed lumpy beside clean dots at its finest cell edge - see
        # SURFACE_SELECTED_BY_DEFAULT - while the points are what the coverage
        # measurement is made on and "where do I still need to scan" is the
        # question this panel exists to answer. So the panel opens on the points
        # and offers the other picture.
        # Built with the wording for a panel that holds NO surface, because at
        # this point it holds none: naming one of the two producers here would be
        # the guess :func:`~scanner3d.preview_provenance.surface_control` exists
        # to refuse. :meth:`_refresh_model_provenance` names whichever surface
        # arrives, at the moment it arrives.
        self._surface_toggle = QCheckBox(surface_control(None).title)
        #: What the OPERATOR last asked for, which is not the same thing as what
        #: is on the screen. A surface can go away - a scan that outgrows the
        #: live cell budget has none to draw - and the control has to follow the
        #: screen while remembering the choice, so the choice comes back with the
        #: surface instead of being silently reset by it.
        #:
        #: Starts at :data:`SURFACE_SELECTED_BY_DEFAULT`, so the first surface a
        #: scan produces ENABLES this control without selecting it: a surface
        #: becoming available is not the operator asking for it.
        self._surface_wanted = SURFACE_SELECTED_BY_DEFAULT
        # Unchecked and disabled to begin with, because at this point there is
        # no surface: a ticked box over a panel that is showing points is a
        # control claiming a picture nobody can see.
        self._surface_toggle.setChecked(False)
        self._surface_toggle.setEnabled(False)
        self._cloud.set_show_surface(False)
        self._surface_toggle.setToolTip(surface_control(None).tooltip)
        self._surface_toggle.toggled.connect(self._toggle_surface)
        layout.addWidget(self._surface_toggle)
        # Which renderer is drawing the panel above, said where it is being
        # looked at. "It feels smooth" does not distinguish a graphics card from
        # a CPU that is keeping up with a small cloud, and the difference is the
        # whole reason this machine's card was brought in - so it is stated
        # rather than inferred, and the reason is given when it is a fallback.
        self._viewer_backend = QLabel("")
        self._viewer_backend.setWordWrap(True)
        self._viewer_backend.setStyleSheet("color:#8a8f99;")
        layout.addWidget(self._viewer_backend)
        ScannerWindow._register_technical_widget(self, self._viewer_backend)
        self._cloud.backend_changed.connect(self._show_viewer_backend)
        # Above the coverage summary and separate from it, because they answer
        # different questions at different rates: this one says the capture is
        # still running, the one below says what the model currently contains.
        self._capture_progress = QLabel(capture_progress_text(0, None, 0))
        self._capture_progress.setWordWrap(True)
        self._capture_progress.setStyleSheet("font-weight:bold; padding:2px;")
        layout.addWidget(self._capture_progress)
        ScannerWindow._register_technical_widget(self, self._capture_progress)
        self._coverage_summary = QLabel("No coverage yet.")
        self._coverage_summary.setWordWrap(True)
        layout.addWidget(self._coverage_summary)
        ScannerWindow._register_technical_widget(self, self._coverage_summary)
        # Only present while a pause is in play. A line that is always there
        # saying "not paused" is a line that stops being read, and this one has
        # to be read: while it is showing, frames are being held back from the
        # scan and the operator is the only one who can change that.
        self._pause_label = QLabel("")
        self._pause_label.setWordWrap(True)
        self._pause_label.setStyleSheet("font-weight:bold; color:#9a6700; padding:2px;")
        self._pause_label.setVisible(False)
        layout.addWidget(self._pause_label)
        ScannerWindow._register_technical_widget(self, self._pause_label)
        # Derived once from the viewer that has just been built, rather than
        # left at the constructed text: the label's opening state is subject to
        # exactly the rule every later one is.
        self._refresh_model_provenance()
        ScannerWindow._render_scan_status(self)
        ScannerWindow._refresh_scan_metrics(self)
        return panel

    def _show_viewer_backend(self, description: str) -> None:
        """Say what is drawing the model. GUI THREAD.

        Also into the results panel when it is a fallback, because that is the
        one the operator did not choose and the one they may want to act on; a
        working graphics card is worth stating quietly and no more.

        The provenance line is re-derived here because a fallback rebuilds the
        picture: :meth:`~scanner3d.cloud_viewer.CloudViewer.fall_back` restores
        the cloud, the surface and the operator's choice onto the replacement
        before it says so, and this asks the replacement what it ended up
        drawing. A report from a renderer that has already been swapped out
        never reaches here at all - that viewer drops it - so an obsolete
        failure cannot relabel the picture that replaced it.
        """
        self._viewer_backend.setText(f"3D view: {description}")
        self._refresh_model_provenance()
        if self._cloud.is_software and "OpenGL unavailable" in description:
            self._log(f"3D view fell back to software rendering: {description}")

    # -- device ------------------------------------------------------------
    def _start_firmware_read(self) -> None:
        """Read the firmware version in the background, once, at startup.

        On a worker thread because it touches the device: a read that hangs
        must not stop the window from appearing. It also runs BEFORE any
        capture, which is the safest moment to borrow interface 0 - doing it
        mid-stream would interrupt a sweep the user cannot recapture.
        """
        try:
            devices = find_scanners()
        except Exception:
            self._firmware = "no device"
            return
        if not devices:
            self._firmware = "no device"
            return
        serial = devices[0].serial
        if not serial:
            # The read mutates kernel driver binding, so it must never be aimed
            # at "whatever is plugged in".
            self._firmware = "no serial"
            return

        worker = Worker(lambda _worker: read_firmware_version(serial))
        self._firmware_worker = worker
        worker.finished_ok.connect(self._firmware_read)
        worker.failed.connect(lambda _text: self._firmware_read("not read"))
        worker.start()

    def _firmware_read(self, version: object) -> None:
        self._firmware = str(version)
        self._refresh_device()
        # Only now: the firmware read borrows interface 0, and starting a
        # stream while it holds it would collide. Previewing on open matches
        # what the object is actually for - you look before you scan.
        if self._capture_action.isEnabled():
            self._preview_action.setChecked(True)

    def _refresh_device(self) -> None:
        try:
            devices = find_scanners()
        except Exception as exc:
            self.statusBar().showMessage(f"could not enumerate: {exc}")
            return
        if not devices:
            self.statusBar().showMessage("No scanner found. Check it is plugged in.")
            self._capture_action.setEnabled(False)
            return
        device = devices[0]
        self.statusBar().showMessage(
            f"Serial: {device.serial}    "
            f"USB ID: {device.vendor_id:04x}:{device.product_id:04x}    "
            f"Port: {device.usb_path}    "
            f"Link: {device.usb_speed_label}    "
            f"Firmware: {self._firmware}"
        )
        self._capture_action.setEnabled(True)

    # -- preview -----------------------------------------------------------
    def _toggle_preview(self, on: bool) -> None:
        if on:
            self._start_preview()
        else:
            self._stop_preview()

    def _show_colour(self, jpeg: bytes) -> None:
        """Show the colour camera. Used for framing, never recorded."""
        image = QImage()
        if not image.loadFromData(jpeg):
            return
        self._colour_view.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._colour_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._colour_label.setText(f"{image.width()}x{image.height()}, about 2 frames a second")

    def _scanner_serial(self) -> str | None:
        """The serial the COLOUR camera will be aimed at, or ``None``.

        Resolved by :func:`~scanner3d.config.resolve_serial`, which is the same
        authority :func:`~scanner3d.v4l2_stream.stream_frames` resolves the
        depth node through: configuration first, and the single attached device
        only when nothing is configured. Asking the enumeration directly for
        whichever scanner it happened to list first was wrong on the one bench
        where it matters - two units plugged in and a serial written in the
        config file - because then colour and depth are two different scanners
        and the framing view shows an object the capture is not recording.

        ``None`` means "not resolvable", and that covers both unavailability
        (nothing attached, enumeration failed, the config file unreadable) and
        AMBIGUITY (several attached, none configured). Refusing the ambiguous
        case is the point: this feeds an optional convenience view, and the
        cost of guessing wrong is a silently misleading picture.

        It changes nothing about the depth capture either way. Every caller
        treats ``None`` as a reason to do less - a capture without a resolvable
        serial is still a capture, just without colour - and the depth stream
        resolves its own node, so an ambiguity that stops colour still reaches
        the operator as the stream's own explicit refusal rather than as this
        quietly doing nothing.

        Never raises. GUI thread, before the job starts.
        """
        try:
            return resolve_serial()
        except Exception as exc:
            # Once per run, not per frame. A blank colour panel with nothing in
            # the log is unexplainable, and "which scanner" is exactly the
            # question a two-scanner bench needs answered in words.
            LOG.info("no colour camera this run: %s", exc)
            return None

    def _start_preview(self) -> None:
        """Stream frames to the screen and write nothing.

        Framing an object is the first thing anyone does, and doing it by
        starting a real capture means either committing to four minutes or
        leaving junk on disk. This records nothing at all.
        """
        if self._preview_worker is not None and self._preview_worker.isRunning():
            return
        if hasattr(self, "_scan_statuses"):
            ScannerWindow._replace_scan_statuses(
                self,
                ScanStatus(
                    source=SCAN_STATUS_CAPTURE_SOURCE,
                    title="PREVIEW — NOT RECORDING",
                    summary="Use the colour and depth views to frame the object.",
                    action="Start capture when the depth view is returning the surface clearly.",
                    severity="info",
                    priority=10,
                ),
            )
        # A preview is a run. Whatever the last one saw collapse is not
        # evidence about this one's first frames, and the new identity means a
        # frame still in flight from the last one cannot be drawn as this one's.
        self._begin_depth_run()
        # One identity for the whole run, exactly as in a capture: colour is
        # aimed at it and the readiness wait waits for it, so a configured bench
        # cannot end up watching one scanner become usable and streaming
        # another. Framing that cannot be resolved to one scanner must still not
        # decide whether the DEPTH view runs: None keeps its "any scanner"
        # meaning in the wait, and the stream resolves its own node and says so
        # out loud when it cannot - that message is worth more to the operator
        # than a Preview button that silently does nothing.
        serial = self._scanner_serial()

        def job(worker: Worker) -> object:
            self._await_device(serial)
            # The module's own ceilings rather than a number of my choosing.
            # They exist as runaway guards, and a preview that runs until the
            # user stops it is exactly the shape of thing they guard against -
            # so it ends on its own if the stop signal is ever missed. At the
            # measured rate MAX_FRAMES is about twenty minutes of framing.
            #
            # Colour is pumped by the shared helper, which starts it on the
            # first delivered depth frame and stops it on every way out. It used
            # to be a thread started here that ended only when this worker was
            # CANCELLED - so a preview that hit its own ceiling left a grabber
            # running against a stream that had closed.
            return stream_with_colour(
                worker,
                serial,
                lambda buf, _record: worker.preview.emit(buf),
                frames=MAX_FRAMES,
                decimate=1,
                duration_s=MAX_DURATION_S,
            )

        worker = Worker(job)
        self._preview_worker = worker
        # Connected FOR this run, so a frame this worker emits after the run
        # has ended is dropped rather than drawn - see :meth:`_frame_slot`.
        worker.preview.connect(self._frame_slot(self._depth_run))
        worker.colour.connect(self._show_colour)
        # Bound to this run for the same reason the frame slot is: Qt queues a
        # worker's terminal signal too, so a preview that finished a moment
        # before Stop can have its completion delivered after the capture that
        # replaced it has begun. See :meth:`_preview_end_slot`.
        worker.failed.connect(self._preview_end_slot(self._depth_run, self._preview_failed))
        worker.finished_ok.connect(self._preview_end_slot(self._depth_run, self._preview_ended))
        worker.start()
        self.statusBar().showMessage("previewing - nothing is being recorded", 4000)

    def _stop_preview(self) -> None:
        """Ask the preview to stop. Does NOT wait for it.

        Waiting here would block the GUI thread for as long as it takes the
        stream to notice - and then again for the device to re-enumerate, which
        it does on every stream close. That was measured at up to eleven
        seconds of unresponsive window, which is indistinguishable from a crash
        to the person looking at it.

        The waiting still has to happen; it happens on the WORKER thread that
        needs the device next, via :meth:`_await_device`. Nothing here blocks.
        """
        # The run ends here, whether or not there is a worker to ask: the
        # warning describes a stream that is stopping, and a frame already in
        # flight from it belongs to the run being ended, not to whatever runs
        # next.
        self._end_depth_run()
        if hasattr(self, "_scan_statuses"):
            ScannerWindow._replace_scan_statuses(self, IDLE_SCAN_STATUS)
        worker = self._preview_worker
        if worker is None:
            return
        LOG.info("preview stop requested")
        worker.cancel()
        self._previous_stream = worker
        self._preview_worker = None

    def _await_device(self, serial: str | None) -> None:
        """Wait until the depth node is USABLE. WORKER THREAD ONLY.

        Two openers of one V4L2 node fails in a way that reads as the scanner
        being broken, so the wait for the previous stream genuinely must
        happen - just not where the user can feel it.

        The second half used to be two back-to-back calls to an
        enumeration-only wait, and that was the defect the owner hit on
        2026-08-26: this device re-enumerates on the USB bus every time a stream
        closes, both calls returned in milliseconds, and the open that followed
        failed with ``EACCES`` on a node created in the same second - a node
        that carried a valid ACL for this account when it was inspected
        afterwards. That the ACL had not yet been attached at the instant of
        that open is the inference those observations support; it was not
        separately recorded. Either way the node was registered before it was
        openable, so asking whether it is readable and writable is the question
        that was actually being asked - once, because asking the wrong one
        twice never helped.

        ``serial`` is the identity already resolved by :meth:`_scanner_serial`,
        so a configured run waits for the same scanner colour and depth will
        use. ``None`` carries its meaning through unchanged: wait for any
        scanner, and leave ambiguity to the stream's own resolver.

        Raises :class:`~scanner3d.v4l2_stream.StreamDeviceNotReady` rather than
        returning, so a permission problem that never clears reaches the
        operator as a stated failure instead of a stream opened on a node this
        process cannot use.
        """
        previous = self._previous_stream
        if previous is not None:
            with timed(LOG, "waiting for the previous stream to end", warn_after=2.0):
                previous.wait(8000)
            self._previous_stream = None
        with timed(LOG, "waiting for the depth node to become usable", warn_after=2.0):
            readiness = wait_for_capture_access(serial)
        if not readiness.ready:
            LOG.error("device not ready after %d checks: %s", readiness.polls, readiness.reason)
            raise StreamDeviceNotReady(readiness.failure_text(), status=readiness.status)
        LOG.info(
            "device ready after %.3fs and %d checks: %s",
            readiness.waited_s,
            readiness.polls,
            readiness.reason,
        )

    def _preview_ended(self, _result: object) -> None:
        """The stream hit its own ceiling. Untick, so it does not look frozen."""
        LOG.info("preview stream ended on its own")
        self._end_depth_run()
        self._preview_action.setChecked(False)
        if hasattr(self, "_scan_statuses"):
            ScannerWindow._replace_scan_statuses(
                self,
                ScanStatus(
                    source=SCAN_STATUS_CAPTURE_SOURCE,
                    title="PREVIEW ENDED",
                    summary="Nothing is being recorded.",
                    action="Press Preview to resume framing.",
                    severity="idle",
                    priority=5,
                ),
            )
        self.statusBar().showMessage("preview ended - press Preview to resume", 6000)

    def _preview_failed(self, text: str) -> None:
        LOG.error("preview failed: %s", text.strip().splitlines()[-1] if text.strip() else text)
        self._end_depth_run()
        self._preview_action.setChecked(False)
        self._log("\nPreview failed:\n" + text)
        if hasattr(self, "_scan_statuses"):
            ScannerWindow._replace_scan_statuses(
                self,
                ScanStatus(
                    source=SCAN_STATUS_CAPTURE_SOURCE,
                    title="PREVIEW FAILED",
                    summary="The live camera stream stopped with an error.",
                    action="Open technical details, then try Preview again.",
                    severity="error",
                    priority=100,
                    detail=text,
                ),
            )

    # -- plumbing ----------------------------------------------------------
    def _busy(self, busy: bool) -> None:
        self._capture_action.setEnabled(not busy)
        self._recon_action.setEnabled(not busy)
        self._preview_action.setEnabled(not busy)
        self._stop_action.setEnabled(busy)
        if not busy:
            # Whatever ended - a capture, a reconstruction, a failure - there is
            # no live scan to pause now. Reset rather than left as it was, so a
            # Pause left checked by the run that just finished cannot be
            # inherited by the next one.
            self._end_pause()

    # -- pause and reposition ----------------------------------------------
    def _end_pause(self) -> None:
        """Put the Pause control back to "no capture is running". GUI THREAD.

        The words as well as the state: a run that ended while paused would
        otherwise leave a toolbar offering "Resume" over a scan that has
        finished, which is an offer nothing behind it can honour.
        """
        self._pause = None
        self._aligned_frames = 0
        self._set_pause_toggle(False)
        self._pause_action.setEnabled(False)
        # The tooltip too, for the same reason as the title: a capture that lost
        # tracking leaves both saying so, and after that capture has ended they
        # would be describing a scan that is over. ``_tracking_loss`` itself is
        # deliberately NOT cleared here - the report's closing sentence still
        # has to be the true one.
        self._pause_action.setToolTip(PAUSE_TOOLTIP)

    def _set_pause_toggle(self, paused: bool) -> None:
        """Point the toggle - its words as well as its state - at ``paused``.

        The capture publishes the pause STATE, and a refused pause therefore has
        to put the button back up. Doing that by setting the action would fire
        :meth:`_toggle_pause` and ask the capture to resume something it never
        started, so the signal is blocked across the change.

        The title moves with that state and only with it. A control that renamed
        itself on the click would say "Resume" over a capture that had refused
        the pause and was still writing, and over the seconds between the press
        and the frame boundary the pause is actually taken at. So "waiting for a
        boundary", "relocalizing" and every refusal read ``Pause``, because that
        is what pressing it would do then; only the published ``paused`` state
        reads ``Resume``.

        The title is set unconditionally rather than behind the same early
        return the checked state uses: the two can legitimately disagree - a
        capture republishing ``paused`` while the toggle is already down - and a
        title skipped because the checkmark had not moved is exactly the stale
        word this exists to prevent.
        """
        self._pause_action.setText(RESUME_TITLE if paused else PAUSE_TITLE)
        if self._pause_action.isChecked() == paused:
            return
        blocked = self._pause_action.blockSignals(True)
        try:
            self._pause_action.setChecked(paused)
        finally:
            self._pause_action.blockSignals(blocked)

    def _toggle_pause(self, on: bool) -> None:
        """The operator asked to pause or to resume. GUI THREAD.

        A request, never an effect: the capture applies it at its own frame
        boundary, after the driver buffer in flight has been handed back. What
        the panel says is what the CAPTURE published, not what was clicked, so
        the words on screen can never run ahead of the scan.
        """
        if self._capturing and self._lost_tracking:
            # The toolbar action is disabled the moment the loss is published,
            # so this is only reachable by a route that goes round it. Answered
            # rather than ignored: a control that silently does nothing is
            # indistinguishable from one that is broken, and Pause is exactly
            # the control the operator will try when the model stops moving.
            self._deny_pause_after_loss()
            self.statusBar().showMessage(
                "tracking is lost - Pause cannot repair it. Press Stop.", 6000
            )
            return
        control = self._pause
        if control is None:
            self._set_pause_toggle(False)
            return
        if on:
            LOG.info("pause requested by the operator")
            control.pause()
            self.statusBar().showMessage("pausing at the next frame...", 6000)
        else:
            LOG.info("resume requested by the operator")
            control.resume()
            self.statusBar().showMessage("resuming - collecting frames to place...", 6000)

    def _show_pause_state(self, state: PauseState) -> None:
        """Say where the pause got to, in the capture's own words. GUI THREAD."""
        if state.state == TRACKING_LOST_STATE:
            self._show_tracking_lost(state.detail)
            return
        if state.state == TRACKING_LOSS_UNRECORDED_STATE:
            self._show_tracking_loss_unrecorded(state.detail)
            return
        if state.state == RECOVERING_TRACKING_STATE:
            self._show_recovering_tracking(state.detail)
            return
        if state.state == TRACKING_RECOVERED_STATE:
            self._show_tracking_recovered(state.detail)
            return
        self._set_pause_toggle(state.paused)
        self._pause_label.setText(state.detail)
        ScannerWindow._set_technical_widget_visible(
            self,
            self._pause_label,
            state.state != "running" or bool(state.detail),
        )
        if state.state == "paused":
            ScannerWindow._show_scan_status(
                self,
                ScanStatus(
                    source=SCAN_STATUS_TRACKING_SOURCE,
                    title="SCAN PAUSED",
                    summary="The live views continue, but no frames are being added.",
                    action="Move the camera or object, then press Resume.",
                    severity="info",
                    priority=55,
                    detail=state.detail,
                ),
            )
        elif state.state == "relocalizing":
            ScannerWindow._show_scan_status(
                self,
                ScanStatus(
                    source=SCAN_STATUS_TRACKING_SOURCE,
                    title="MATCHING THE NEW VIEW",
                    summary="New frames are held out until the existing model is recognized.",
                    action="Aim at distinctive surface you already scanned.",
                    severity="warning",
                    priority=70,
                    detail=state.detail,
                ),
            )

    def _show_tracking_lost(self, detail: str) -> None:
        """The scan has stopped knowing where the object is. GUI THREAD.

        One way, because the state is one way: the capture publishes this once
        and publishes nothing after it, so the banner stays up and the Pause
        control is relabelled and taken away rather than greyed for a moment.
        """
        self._tracking_loss = detail
        self._deny_pause_after_loss()
        self._pause_label.setText(detail)
        ScannerWindow._set_technical_widget_visible(self, self._pause_label, True)
        terminal = "NOT recovered" in detail or "no recovery will be attempted" in detail
        ScannerWindow._show_scan_status(
            self,
            ScanStatus(
                source=SCAN_STATUS_TRACKING_SOURCE,
                title="SCAN FAILED — TRACKING LOST" if terminal else "TRACKING LOST",
                summary="The model is frozen at the last frame that aligned.",
                action=(
                    "Press Stop, then start a new scan."
                    if terminal
                    else "Turn back toward the last good view while recovery starts."
                ),
                severity="error",
                priority=100 if terminal else 90,
                detail=detail,
            ),
        )

    def _show_tracking_loss_unrecorded(self, detail: str) -> None:
        """The scan lost tracking and could NOT write the record. GUI THREAD.

        Everything the recorded case does to the window, and none of what it
        says about the disk. The capture is already cancelling itself and the
        job will raise, so a failure dialog follows this; what this is for is
        the seconds before it, during which the toolbar must not still be
        offering a Pause that cannot be honoured.
        """
        self._tracking_loss_unrecorded = detail
        self._deny_pause_after_loss()
        self._pause_label.setText(detail)
        ScannerWindow._set_technical_widget_visible(self, self._pause_label, True)
        ScannerWindow._show_scan_status(
            self,
            ScanStatus(
                source=SCAN_STATUS_TRACKING_SOURCE,
                title="SCAN FAILED — TRACKING LOST",
                summary="The scan stopped and its tracking-loss record could not be written.",
                action="Press Stop. The frames already written are untouched.",
                severity="error",
                priority=100,
                detail=detail,
            ),
        )

    def _show_recovering_tracking(self, detail: str) -> None:
        """The loss is recorded and the scan is looking for the object. GUI THREAD.

        Everything the recorded loss does to the toolbar, and different words on
        it. The loss banner is NOT cleared and ``_tracking_loss`` is NOT
        forgotten: until a recovery has actually been published the capture is
        still a capture that lost tracking, and if it ends here that is the
        ending it must be given.
        """
        self._deny_pause_after_loss()
        self._pause_action.setText(RECOVERING_PAUSE_TITLE)
        self._pause_action.setToolTip(RECOVERING_PAUSE_TOOLTIP)
        self._pause_label.setText(detail)
        ScannerWindow._set_technical_widget_visible(self, self._pause_label, True)
        ScannerWindow._show_scan_status(
            self,
            ScanStatus(
                source=SCAN_STATUS_TRACKING_SOURCE,
                title="TRACKING LOST — TRYING TO RECOVER",
                summary="Candidate frames are being held out while the old model is searched.",
                action="Turn back toward distinctive surface you already scanned.",
                severity="warning",
                priority=95,
                detail=detail,
            ),
        )

    def _show_tracking_recovered(self, detail: str) -> None:
        """A recovery was published and the scan is running again. GUI THREAD.

        The one place the lost state is cleared, and the capture only publishes
        this state once the segment holding the recovered frames and their
        binding is visible on disk. Pause is put back to its ordinary words and
        left DISABLED: it becomes available again the same way it does at the
        start of a capture, when :meth:`_show_coverage` sees enough accepted
        geometry for a relocalization to be measured against.
        """
        self._tracking_loss = None
        self._tracking_loss_unrecorded = None
        self._set_pause_toggle(False)
        self._pause_action.setEnabled(False)
        self._pause_action.setText(PAUSE_TITLE)
        self._pause_action.setToolTip(PAUSE_TOOLTIP)
        self._pause_label.setText(detail)
        ScannerWindow._set_technical_widget_visible(self, self._pause_label, True)
        ScannerWindow._show_scan_status(
            self,
            ScanStatus(
                source=SCAN_STATUS_TRACKING_SOURCE,
                title="TRACKING RECOVERED",
                summary="The new view matched the existing scan and recording has resumed.",
                action="Move slowly and keep previously scanned detail in view.",
                severity="ok",
                priority=50,
                detail=detail,
            ),
        )

    @property
    def _lost_tracking(self) -> bool:
        """Whether this capture has stopped knowing where the object is.

        Either way it stopped: with a durable record, or with a record it could
        not write. Pause is denied and the model is frozen in both cases, and
        only the CLOSING sentence of the run distinguishes them.
        """
        return self._tracking_loss is not None or self._tracking_loss_unrecorded is not None

    def _deny_pause_after_loss(self) -> None:
        """Put Pause up, disable it, and make it say why. GUI THREAD.

        The title is set AFTER :meth:`_set_pause_toggle`, which writes the
        ordinary "Pause"/"Resume" wording unconditionally; doing it the other
        way round would leave the control reading "Pause" over a scan that
        cannot be paused.
        """
        self._set_pause_toggle(False)
        self._pause_action.setEnabled(False)
        self._pause_action.setText(TRACKING_LOST_PAUSE_TITLE)
        self._pause_action.setToolTip(TRACKING_LOST_PAUSE_TOOLTIP)

    def _stop(self) -> None:
        """Ask the running job to stop. Once, and only the once.

        Cancellation is a request the job honours at its own safe boundary - for
        a capture, after the current buffer has been handed back to the driver.
        Asking twice cannot make that boundary arrive sooner, so a second press
        is dropped rather than logged again.
        """
        worker = self._worker
        if worker is None or not worker.isRunning() or worker.cancelled:
            return
        LOG.info("stop requested by the operator")
        worker.cancel()
        self._stop_action.setEnabled(False)
        self._log("\nStopping at the next frame...")

    def _log(self, message: str) -> None:
        self._report.appendPlainText(message)

    def _consume_resume_preview(self) -> bool:
        """Whether preview should come back, clearing the intent as it answers.

        Read exactly once per job, on whichever path that job ended by. Left set,
        it would outlive the capture it belongs to and hand a later operation -
        a reconstruction, or a capture the user stopped - a preview it never
        asked for.
        """
        resume, self._resume_preview_after = self._resume_preview_after, False
        return resume

    def _was_stopped(self) -> None:
        """The operator's Stop, arrived at after the device has been let go.

        Deliberately NOT a completion. Preview is not restarted here even when
        it was running before the capture: the person who pressed Stop asked for
        the scanner to stop, and answering that by putting the live stream back
        up - projector lit, frames flowing - is indistinguishable from the button
        having done nothing. That is exactly how it read to the owner on
        2026-08-26.
        """
        # Ended before the rest, so a frame the stopping capture still has in
        # flight cannot be drawn against whatever comes next.
        self._end_depth_run()
        self._busy(False)
        resume = self._consume_resume_preview()
        self._finish_progress()
        if self._tracking_loss is not None:
            LOG.info("capture stopped by the operator after tracking was lost")
            self._log(_tracking_loss_epilogue(self._capture_dir, "Stopped."))
            self._replace_scan_statuses(
                ScanStatus(
                    source=SCAN_STATUS_CAPTURE_SOURCE,
                    title="SCAN FAILED — TRACKING LOST",
                    summary="The model is frozen at the last frame that aligned.",
                    action="Start a new scan; this folder cannot be reconstructed.",
                    severity="error",
                    priority=100,
                    detail=self._tracking_loss,
                )
            )
        else:
            counts = _frame_counts(self._capture_dir)
            LOG.info(
                "capture stopped by the operator with %d reconstructable frames written and "
                "%d excluded frame(s) kept as evidence",
                counts.frames,
                counts.excluded,
            )
            self._log(
                f"Stopped. {counts.frames} frames were already written and are safe to reconstruct."
            )
            # Only when there are any, and never folded into the number above:
            # a recovered capture holds frames that are on disk and are NOT part
            # of the scan, and calling them either "written and safe to
            # reconstruct" or nothing at all are both untrue.
            excluded = _excluded_note(counts)
            if excluded:
                self._log(excluded)
            self._replace_scan_statuses(
                ScanStatus(
                    source=SCAN_STATUS_CAPTURE_SOURCE,
                    title="SCAN STOPPED",
                    summary=f"{counts.frames} frames are safely stored.",
                    action="Reconstruct this scan, or start another capture.",
                    severity="info",
                    priority=20,
                )
            )
        if resume:
            self._log("Preview was NOT restarted. Press Preview when you want the live view back.")

    def _fail(self, text: str) -> None:
        # Before anything else, and before any preview is put back: this run is
        # over, so its warning goes with it.
        self._end_depth_run()
        self._busy(False)
        # Consumed and dropped: a failed capture is not the moment to reopen the
        # device, and leaving the intent set would spring it on the next job.
        self._consume_resume_preview()
        self._finish_progress()
        self._log("\nFAILED:\n" + text)
        title = (
            "SCAN FAILED — TRACKING LOST"
            if self._tracking_loss_unrecorded is not None
            else "OPERATION FAILED"
        )
        self._replace_scan_statuses(
            ScanStatus(
                source=SCAN_STATUS_CAPTURE_SOURCE,
                title=title,
                summary="The operation stopped with an error.",
                action="Open technical details for the exact cause, then retry.",
                severity="error",
                priority=100,
                detail=text,
            )
        )
        QMessageBox.critical(self, "3D Scanner", text.strip().splitlines()[-1])

    def _show_frame(self, raw: bytes) -> None:
        """Draw one depth frame and say how much of it came back. GUI THREAD.

        ONE :func:`~scanner3d.viewer.render_frame` produces the picture, the
        numbers and the return-health evidence, so the frame is unpacked once
        per displayed frame rather than once for colour and again for
        statistics - and the picture and the sentence beside it are guaranteed
        to describe the same samples.

        The same slot serves preview and capture (both workers connect their
        ``preview`` signal here), so there is exactly one place in this window
        that decides what a depth frame looks like and what is said about it.
        """
        rendered = render_frame(raw)
        stats = rendered.stats
        image = QImage(
            b"".join(rendered.pixels),
            FRAME_WIDTH,
            FRAME_HEIGHT,
            FRAME_WIDTH * 3,
            QImage.Format.Format_RGB888,
        )
        self._view.setPixmap(
            QPixmap.fromImage(image).scaled(
                self._view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._frame_label.setText(
            f"returns {stats.returned} of {stats.samples} "
            f"({100.0 * stats.yield_fraction:.1f}% of the frame)   "
            f"depth codes {stats.min_code} to {stats.max_code}, median {stats.median_code}"
        )
        self._show_return_health(stats)

    def _show_return_health(self, stats: FrameStats) -> None:
        """Restate the return-health warning from this frame. GUI THREAD.

        Advisory only, and it says so by what it does NOT do: nothing here
        accepts or refuses a frame, stops a capture, moves a tracking gate or
        changes a reconstruction threshold. It writes two labels.

        The labels are rewritten on every displayed frame, so a recovered frame
        clears the warning without waiting for anything. The LOG is written
        only when the verdict changes, because a line per frame at ten frames a
        second buries the transition that is the only interesting part of it.
        """
        health = self._return_health.update(stats.yield_fraction)
        titles = {
            NO_RETURNS: "NO DEPTH SIGNAL",
            SEVERELY_SPARSE: "DEPTH SIGNAL TOO LOW",
            COLLAPSING: "DEPTH SIGNAL COLLAPSING",
        }
        title = titles.get(health.verdict, "")
        action = (
            "Stop moving. Turn back toward the last good view and aim the surface more squarely."
        )
        technical = (
            f"{health.advice}\n\n{spatial_summary(stats.spatial)}" if health.is_warning else ""
        )
        self._return_label.setText(
            f"{title} — {100.0 * health.yield_fraction:.1f}% returned" if health.is_warning else ""
        )
        self._return_label.setToolTip(technical)
        self._return_label.setStyleSheet(
            LOUD_RETURN_STYLE if health.is_warning else QUIET_RETURN_STYLE
        )
        self._return_advice.setText(action if health.is_warning else "")
        self._return_advice.setToolTip(technical)
        if hasattr(self, "_scan_statuses"):
            if health.is_warning:
                if health.verdict == COLLAPSING and health.baseline_fraction is not None:
                    summary = (
                        f"{health.yield_fraction:.1%} of this view returned depth, down from "
                        f"{health.baseline_fraction:.1%}."
                    )
                    priority = 80
                elif health.verdict == NO_RETURNS:
                    summary = "This view returned no usable depth."
                    priority = 90
                else:
                    summary = f"Only {health.yield_fraction:.1%} of this view returned depth."
                    priority = 85
                ScannerWindow._show_scan_status(
                    self,
                    ScanStatus(
                        source=SCAN_STATUS_DEPTH_SOURCE,
                        title=title,
                        summary=summary,
                        action=action,
                        severity="warning",
                        priority=priority,
                        detail=technical,
                    ),
                )
            else:
                ScannerWindow._clear_scan_status(self, SCAN_STATUS_DEPTH_SOURCE)
        if health.verdict != self._return_verdict:
            LOG.info(
                "depth return health: %s (%s) - returns %.1f%% of the frame, recent baseline %s",
                health.verdict,
                health.reason,
                100.0 * health.yield_fraction,
                "none yet"
                if health.baseline_fraction is None
                else f"{100.0 * health.baseline_fraction:.1f}%",
            )
            self._return_verdict = health.verdict

    def _reset_return_health(self) -> None:
        """Forget the return-health history, and clear the warning.

        Called wherever a run begins or ends and when the window closes: an old
        scan's collapse is not evidence about a new one, and a warning left
        standing after a run has ended describes a frame that is no longer
        being produced.
        """
        self._return_health.reset()
        self._return_verdict = None
        self._return_label.setText("")
        self._return_label.setToolTip("")
        self._return_label.setStyleSheet(QUIET_RETURN_STYLE)
        self._return_advice.setText("")
        self._return_advice.setToolTip("")
        if hasattr(self, "_scan_statuses"):
            ScannerWindow._clear_scan_status(self, SCAN_STATUS_DEPTH_SOURCE)

    def _begin_depth_run(self) -> None:
        """A preview or a capture is starting. New identity, empty history."""
        self._depth_run += 1
        self._reset_return_health()

    def _end_depth_run(self) -> None:
        """A preview or a capture has ended, by ANY route.

        Manual stop, the stream reaching its own ceiling, normal completion and
        failure all come here. Moving the identity on is what makes a frame
        still in flight from the ended run stop being displayed
        (:meth:`_frame_slot`); clearing the history is what stops the next run
        being judged against this one's last frames; clearing the labels is
        what stops a warning about a stream that is no longer running from
        sitting on screen as though it were current.
        """
        self._depth_run += 1
        self._reset_return_health()

    def _preview_end_slot[T](self, run: int, end: Callable[[T], None]) -> Callable[[T], None]:
        """A preview's completion or failure slot, belonging to ONE run.

        ``finished_ok`` and ``failed`` are queued across threads exactly as
        ``preview`` is. A preview that hit its ceiling, or died, in the instant
        before the operator pressed Stop therefore has its terminal signal
        delivered to the GUI thread AFTER the capture that replaced it has
        started - and unguarded, that stale callback ends the new run: it
        advances the run identity (so the capture's own frames are then
        dropped by :meth:`_frame_slot`), clears the capture's return-health
        warning, and unticks Preview against a stream that is not the one it
        belongs to.

        The run is captured when the signal is connected, so this is an
        identity comparison rather than a guess about elapsed time. A stale
        ending is not silent - a failure in particular is real news about the
        device, and is logged - but it changes nothing about the run that is
        now live.
        """

        def ended(payload: T) -> None:
            if run != self._depth_run:
                # One line, at the end of a run rather than per frame, and
                # trimmed to a failure's last line: the news is kept, the
                # stale run's control over the live one is not.
                lines = payload.strip().splitlines() if isinstance(payload, str) else []
                text = lines[-1] if lines else "ended"
                LOG.info(
                    "ignored the ending of depth run %d; run %d is current: %s",
                    run,
                    self._depth_run,
                    text,
                )
                return
            end(payload)

        return ended

    def _frame_slot(self, run: int) -> Callable[[bytes], None]:
        """A depth-frame slot that belongs to ONE run.

        Qt delivers a worker's ``preview`` emissions to the GUI thread through
        a queue, so a frame emitted by a preview just before it was stopped can
        be delivered after the capture that replaced it has already begun. Left
        unguarded that frame redraws the new run's panel and, worse, becomes
        the first entry in its return-health baseline - the new capture is then
        measured against the run it replaced.

        The run this slot was connected for is captured here, so the check is
        an identity comparison rather than a guess about how much time has
        passed.
        """

        def deliver(raw: bytes) -> None:
            if run != self._depth_run:
                # Not an error, and not worth a line per frame: a couple of
                # frames in flight at a run boundary is the ordinary shape of a
                # stop. DEBUG so it is still answerable from a log.
                LOG.debug(
                    "dropped a depth frame from run %d; run %d is current", run, self._depth_run
                )
                return
            self._show_frame(raw)

        return deliver

    def _show_travel(self, travel: float) -> None:
        verdict, advice = travel_verdict(travel)
        colour = {"good": "#1a7f37", "too slow": "#9a6700", "too fast": "#9a6700"}.get(
            verdict, "#666"
        )
        self._speed_label.setText(f"ROTATION — {verdict.upper()}  ({travel:.1f} per frame)")
        self._speed_label.setStyleSheet(
            f"font-size:15pt; font-weight:bold; padding:4px; color:{colour};"
        )
        short_advice = {
            "too slow": "Speed the rotation up.",
            "too fast": "Slow the rotation down.",
            "no motion": "Start the turntable, or move the scanner.",
        }.get(verdict, "")
        self._speed_label.setToolTip(advice)
        self._speed_advice.setText(short_advice)
        self._speed_advice.setToolTip(advice)
        if verdict in {"too slow", "too fast"} and hasattr(self, "_scan_statuses"):
            ScannerWindow._show_scan_status(
                self,
                ScanStatus(
                    source=SCAN_STATUS_MOTION_SOURCE,
                    title=f"ROTATION {verdict.upper()}",
                    summary="Motion between saved frames is outside the alignment band.",
                    action=short_advice,
                    severity="warning",
                    priority=45,
                    detail=advice,
                ),
            )
        elif hasattr(self, "_scan_statuses"):
            ScannerWindow._clear_scan_status(self, SCAN_STATUS_MOTION_SOURCE)

    def _refresh_capture_progress(self) -> None:
        """Restate the heartbeat from the counters. GUI THREAD."""
        self._capture_progress.setText(
            capture_progress_text(
                self._saved_frames,
                self._model_refreshed_at,
                self._model_refreshes,
                running=self._capturing,
            )
        )

    def _show_saved(self, saved: int) -> None:
        """One frame reached the disk. GUI THREAD.

        The cheapest thing this window does per frame, and the only one that is
        guaranteed to happen on every one of them: a label with a number in it.
        """
        self._saved_frames = saved
        self._refresh_capture_progress()
        self._refresh_scan_metrics()

    def _finish_progress(self) -> None:
        """Put the heartbeat into the past tense, once, when a capture ends.

        A no-op for a job that was not a capture, so a stopped or failed
        reconstruction leaves the last capture's summary standing rather than
        overwriting it with a claim about frames it never wrote.
        """
        if not self._capturing:
            return
        self._capturing = False
        self._refresh_capture_progress()
        self._refresh_scan_metrics()
        self._log_feedback_load()

    def _log_feedback_load(self) -> None:
        """What the run cost the GUI, once, at the end of a capture.

        The readable form of what the per-event DEBUG lines record. It is the
        difference between the three explanations a physical "it froze" report
        cannot otherwise be told apart by: a model that was never PRODUCED (the
        alignment or the build failed), one that was produced and APPLIED (the
        picture really did change and was too subtle to see), and one produced
        and then SUPERSEDED before it could be drawn (the window could not keep
        up, which is the failure this bounding exists to prevent, and which
        nothing in the previous run's evidence could have told apart from the
        other two).
        """
        worker = self._worker
        if worker is None:
            return
        for channel in worker.channels():
            LOG.info("live feedback - %s", channel.summary())

    def _show_coverage(self, snapshot: ModelSnapshot) -> None:
        """Draw one accumulated model, described by ITS OWN counts. GUI THREAD.

        ``snapshot.saved`` rather than ``self._saved_frames``: the frame count
        that belongs to this picture is the one it was built at, and by the time
        it is drawn the live count may have moved on. Reading the live one made
        the panel claim a rebuild that had not happened yet.

        The snapshot also carries whether the view should re-frame itself, and
        the answer is no for every incremental update. Framing a model that has
        just grown normalises the growth away and throws out the pan and zoom
        the operator chose, which together are what made a live model that WAS
        changing look like one that had stopped.
        """
        self._cloud.set_cloud(snapshot.points, snapshot.thin, snapshot.newest, fit=snapshot.reframe)
        # After the cloud, always: set_cloud is what frames a view that has
        # never been framed, and it frames from the POINTS - so a surface
        # applied first would be framed against and then re-framed against
        # something else in the same update.
        # What the panel will be HOLDING when this update is finished, or None.
        # One record, read twice below - by the provenance line and by the
        # coverage summary - because two sentences about one mesh that are
        # assembled separately are two sentences that can disagree.
        evidence: SurfaceEvidence | None = None
        if snapshot.has_surface:
            # WHICH surface this is, from the producer that built it. Refused
            # rather than guessed: the two candidates are the live occupancy
            # preview and the exported signed-distance surface, and each of
            # their sentences is false about the other - so a mesh that arrived
            # without saying which it is has no truthful description here.
            if snapshot.surface_kind is None:
                raise ProvenanceUnusable(
                    "this model carries a surface but does not say which surface it is; "
                    "the producer sets ModelSnapshot.surface_kind"
                )
            # The record that describes the surface the viewer is about to
            # HOLD, taken from the same snapshot the geometry came from and
            # built in the same breath as it. Any other moment - a later
            # refresh, a toggle, a renderer swap - would be a moment at which
            # the description and the mesh could belong to two different models.
            #
            # Built BEFORE the mesh is mounted, so a surface this seam refuses
            # to describe is never drawn under a sentence at all.
            evidence = SurfaceEvidence(
                kind=snapshot.surface_kind,
                faces=len(snapshot.surface_faces),
                cell=snapshot.surface_voxel,
            )
            self._cloud.set_surface(snapshot.surface_vertices, snapshot.surface_faces)
            self._surface_evidence = evidence
        else:
            self._cloud.clear_surface()
            self._surface_evidence = None
        # After both, because this is what re-derives the provenance line, and a
        # line derived before the viewer was given its picture would describe the
        # one before it.
        self._show_surface_control(snapshot.has_surface)
        self._model_refreshes += 1
        self._model_refreshed_at = snapshot.saved
        # Pause becomes available here and nowhere else, because this is where
        # the window learns how much ACCEPTED geometry the scan has - which is
        # exactly what a relocalization would have to recognise the object by.
        # Offering it earlier would be offering a control whose only possible
        # answer is a refusal.
        self._aligned_frames = snapshot.aligned
        self._coverage_well = snapshot.well_covered
        self._coverage_needs = self._cloud.thin_fraction
        ScannerWindow._refresh_scan_metrics(self)
        # Never after a loss. No model refresh should reach here at all once
        # tracking is lost - nothing is added to the model - but re-enabling
        # Pause is the one thing a stray late snapshot must not be able to do.
        if self._capturing and self._pause is not None and not self._lost_tracking:
            self._pause_action.setEnabled(snapshot.aligned >= MIN_PAUSE_REFERENCE_CLOUDS)
        # Said only when there is something to see: a registered model has no
        # newest frame, and a sentence pointing at a highlight that is not there
        # is worse than no sentence.
        highlight = (
            "  The brightest points are the newest frame." if bool(np.any(snapshot.newest)) else ""
        )
        # Said whenever there is a surface, whether or not it is the picture
        # showing, because the coverage percentages either side of it are
        # measured on the POINTS - and an operator looking at a shaded shape
        # while reading a number about a cloud is owed the sentence that says
        # which is which.
        # And when there is NOT one, why - whenever the producer had a reason
        # worth giving. A panel that just stops showing a surface is
        # indistinguishable from one whose scan has stalled, and the operator
        # who reported a blob had no way to tell either from a surface that had
        # quietly been redrawn at half the resolution.
        if evidence is not None:
            # From the SAME record the provenance line above it is written from,
            # so the two cannot disagree about what this mesh is: a summary
            # calling the exported reconstruction an experimental occupancy
            # preview is exactly the contradiction that made the persistent line
            # untrustworthy.
            surface = f"  {surface_summary(evidence)}"
        elif snapshot.surface_note:
            surface = f"  {snapshot.surface_note}"
        else:
            surface = ""
        self._coverage_summary.setText(
            f"{snapshot.note}   {100.0 * snapshot.well_covered:.0f}% of the surface is covered, "
            f"{100.0 * self._cloud.thin_fraction:.0f}% of the POINTS still need scanning"
            f"{highlight}{surface}"
        )
        LOG.info(
            "live model applied: %d points from %d aligned frames of %d saved",
            len(snapshot.points),
            snapshot.aligned,
            snapshot.saved,
        )
        # Restated here so the heartbeat can say when the picture below it last
        # changed - which is the difference between a model that is holding
        # still because nothing is happening and one that is between rebuilds.
        if self._capturing:
            self._refresh_capture_progress()

    def _toggle_surface(self, on: bool) -> None:
        """The operator choosing a picture, remembered as well as applied.

        Remembered because the choice outlives the surface. A capture that
        outgrows the live cell budget has no surface to show and the control
        goes down with it; when a later capture has one again, the operator gets
        back the picture they picked rather than this window's default.

        The provenance line moves with the picture and in the same call, so
        there is no moment - not even one repaint - at which the panel is
        showing one thing and claiming the other.
        """
        self._surface_wanted = on
        self._cloud.set_show_surface(on)
        self._refresh_model_provenance()

    def _show_surface_control(self, has_surface: bool) -> None:
        """Keep the Surface control describing what is actually on the screen.

        A box that stays ticked over a panel with no surface in it is a control
        claiming a picture nobody can see, and it is exactly the state that made
        a withdrawn or coarsened surface hard to tell from a scan that had
        stopped. So no surface means the box goes down and the panel falls back
        to the points - which are authoritative anyway, and are where coverage is
        measured.

        It never ticks the box on the operator's behalf either. The remembered
        preference starts at :data:`SURFACE_SELECTED_BY_DEFAULT`, which is False,
        so a scan producing its first surface makes this control USABLE and
        nothing more; the picture stays on the points until the operator asks for
        the other one, and that asking is what brings it back when a later
        surface appears.

        Moved without it looking like the operator moved it, for the reason
        :meth:`_set_pause_toggle` gives: firing ``toggled`` here would overwrite
        the very preference this is preserving.
        """
        showing = has_surface and self._surface_wanted
        self._surface_toggle.setEnabled(has_surface)
        if self._surface_toggle.isChecked() != showing:
            blocked = self._surface_toggle.blockSignals(True)
            try:
                self._surface_toggle.setChecked(showing)
            finally:
                self._surface_toggle.blockSignals(blocked)
        self._cloud.set_show_surface(showing)
        self._refresh_model_provenance()

    def _mounted_picture(self) -> ModelPicture:
        """Which picture is ACTUALLY on the screen, asked of the viewer itself.

        Not of :attr:`_surface_wanted`, and not of the checkbox. Those two say
        what the operator asked for, which is the thing that can disagree with
        the screen: a preference for the surface survives a capture that has no
        surface to draw, and the panel answers such a capture with the points.
        A label built from the request rather than from the result is exactly
        the sentence that would then be false.

        :attr:`~scanner3d.cloud_viewer.CloudViewer.showing_surface` and
        :attr:`~scanner3d.cloud_viewer.CloudViewer.has_cloud` are forwarded to
        whichever backend is mounted, so this follows a GPU-to-software fallback
        without being told one happened.
        """
        if self._cloud.showing_surface:
            return ModelPicture.SURFACE
        if self._cloud.has_cloud:
            return ModelPicture.POINTS
        return ModelPicture.NOTHING

    def _refresh_model_provenance(self) -> None:
        """Restate what the model panel is showing. GUI THREAD.

        The only writer of that line, called from every event that can change
        which picture is mounted: a published model, the toggle, the control
        following a surface that arrived or went away, and a renderer swap.
        One writer rather than a sentence assembled in each of those handlers,
        because the way this claim goes wrong is two handlers that disagree.

        The surface's numbers are passed only when the surface is what is drawn.
        :func:`~scanner3d.preview_provenance.model_provenance` refuses them
        otherwise, which is what stops a triangle count from ever appearing
        beside the dots.

        The surface control's own title and tooltip are rewritten here too, from
        the same record, because they are the other visible claim about the same
        mesh. A box that says it draws the occupancy grid while the panel holds
        the exported signed-distance surface contradicts the line above it, and a
        line adjacent product text contradicts is not a line an operator can act
        on. The control describes what the panel HOLDS rather than what is
        mounted: it is the control for showing that surface, and it says which
        surface that would be even while the dots are the picture.
        """
        picture = self._mounted_picture()
        evidence = self._surface_evidence if picture is ModelPicture.SURFACE else None
        self._model_provenance.setText(model_provenance(picture, evidence))
        control = surface_control(self._surface_evidence)
        # setText on a checkbox changes no check state and emits no ``toggled``,
        # so this cannot look like the operator moving the control - which is the
        # care :meth:`_show_surface_control` takes for the same reason.
        self._surface_toggle.setText(control.title)
        self._surface_toggle.setToolTip(control.tooltip)

    def _run(self, job: Callable[[Worker], object], done: Callable[[object], None]) -> None:
        worker = Worker(job)
        self._worker = worker
        worker.progress.connect(self._log)
        worker.preview.connect(self._frame_slot(self._depth_run))
        worker.colour.connect(self._show_colour)
        worker.saved.connect(self._show_saved)
        worker.coverage.connect(self._show_coverage)
        worker.travel.connect(self._show_travel)
        worker.pause_state.connect(self._show_pause_state)
        worker.scan_status.connect(self._show_scan_status)
        worker.finished_ok.connect(done)
        worker.stopped.connect(self._was_stopped)
        worker.failed.connect(self._fail)
        worker.start()

    # -- where captures go -------------------------------------------------
    def _effective_captures_dir(self) -> tuple[Path | None, str | None]:
        """The captures root that will actually be used, and what states it.

        Both, always together. The file is not the only thing that answers this
        question, and reporting a value without its source is how someone ends
        up looking for their frames under the folder they just chose while an
        environment variable quietly sends them somewhere else.
        """
        try:
            return configured_captures_dir(), captures_dir_source()
        except ConfigFileUnreadable:
            return None, None

    def _refresh_captures_action(self) -> None:
        """Keep the toolbar honest about where the next capture will land."""
        try:
            configured = configured_captures_dir()
            source = captures_dir_source()
        except ConfigFileUnreadable as exc:
            self._captures_action.setToolTip(f"Captures folder unreadable:\n{exc}")
            return
        if configured is None:
            current = "Not set - you are asked for a folder every time."
        else:
            current = f"Currently: {configured}\nStated by: {source}"
        self._captures_action.setToolTip(
            "Where captures go. Set it once and every capture lands in a fresh\n"
            f"folder of its own under {CAPTURE_PREFIX}/, without being asked where.\n"
            "Each scan is named when you start it, or dated if you leave the\n"
            "name blank.\n\n" + current
        )

    def _choose_captures_dir(self) -> None:
        """Record a captures folder, so captures stop asking where to go."""
        try:
            start = configured_captures_dir()
        except ConfigFileUnreadable:
            start = None
        directory = QFileDialog.getExistingDirectory(
            self, "Where should captures go?", str(start or Path.home())
        )
        if not directory:
            return
        try:
            target = set_captures_dir(Path(directory)).path
        except (ConfigFileUnreadable, ConfigWriteRefused) as exc:
            LOG.error("captures folder not recorded: %s", exc)
            QMessageBox.warning(self, "3D Scanner", str(exc))
            return
        self._refresh_captures_action()
        # Reported from the setting read back, never from the folder that was
        # chosen: an environment variable outranks the file, and saying
        # "captures will go under <what you picked>" when they will not is
        # exactly the surprise this whole setting exists to remove.
        effective, source = self._effective_captures_dir()
        if source == CAPTURES_DIR_ENV_VAR:
            message = (
                f"Recorded {target}, but {CAPTURES_DIR_ENV_VAR} is set in this session and "
                f"wins: captures go under {effective}/{CAPTURE_PREFIX}, in a new folder each "
                "time, named when you start the scan."
            )
        else:
            message = (
                f"Captures will go under {effective}/{CAPTURE_PREFIX}, in a new folder each "
                "time, named when you start the scan."
            )
        self._log(message)
        self.statusBar().showMessage(f"captures folder: {effective}", 6000)

    def _capture_name(self) -> str | None:
        """What to call this scan: a validated leaf name, "" for dated, or None.

        None means "do not capture" and covers both ways of saying it - the
        dialog was cancelled, or the name it returned cannot be a folder name.
        Both are the same instruction to the caller, and both happen before
        anything is created, so there is nothing for the caller to undo.

        Asked before the preview is released and long before a device is
        opened, because a cancelled prompt must cost nothing: no directory, no
        interrupted preview, no scanner touched.
        """
        typed, accepted = QInputDialog.getText(
            self,
            "Start capture",
            "Name this scan (optional).\n"
            "Leave it blank and the folder is named with today's date and time.\n"
            "One folder name, please: no '/' or '\\', and no tabs or line breaks.",
        )
        if not accepted:
            return None
        try:
            return validate_capture_name(typed)
        except CaptureDirUnavailable as exc:
            # CaptureNameInvalid, in practice. Caught by the base class so that
            # a name refused for a reason added later still stops the capture
            # here rather than escaping into a traceback.
            LOG.warning("scan name refused: %s", exc)
            QMessageBox.warning(self, "3D Scanner", str(exc))
            return None

    def _capture_target(self) -> Path | None:
        """The directory for the capture about to start, or None if there is none.

        Always a directory created here and now, one level down inside the
        captures root's ``scan`` namespace, that no earlier capture can already
        have written into. Never the captures root itself and never a folder
        somebody merely pointed at: this window writes ``frame_0000.raw``
        upward, so a second capture sharing a directory would overwrite the
        first frame by frame.

        A recorded captures folder is the whole point of having one: it answers
        where without a dialog. Without one, the question still has to be
        asked - and what the answer names is the captures ROOT, with the fresh
        per-scan folder allocated underneath it exactly as if it had been
        recorded.

        A folder that is recorded but cannot be allocated is refused rather
        than quietly demoted to the dialog: the answer to "I cannot write where
        you told me to" is to say so, not to put the frames somewhere else.
        """
        name = self._capture_name()
        if name is None:
            return None
        try:
            root = configured_captures_dir()
        except ConfigFileUnreadable as exc:
            LOG.warning("captures folder unreadable, asking instead: %s", exc)
            root = None
        if root is None:
            directory = QFileDialog.getExistingDirectory(
                self, "Which folder should captures go under?"
            )
            if not directory:
                return None
            root = Path(directory)
        try:
            return allocate_capture_dir(CAPTURE_PREFIX, root=root, name=name)
        except CaptureDirUnavailable as exc:
            LOG.error("no capture folder could be created: %s", exc)
            QMessageBox.warning(self, "3D Scanner", str(exc))
            return None

    # -- capture -----------------------------------------------------------
    def _start_capture(self) -> None:
        target = self._capture_target()
        if target is None:
            return
        # Preview holds the same V4L2 node a capture needs. Released here, and
        # waited for, before anything tries to open it again.
        was_previewing = self._preview_action.isChecked()
        self._preview_action.setChecked(False)
        self._stop_preview()
        self._resume_preview_after = was_previewing
        self._capture_dir = target
        # Same reason as the counters below: this capture's return health is
        # measured against this capture's own frames, starting with none - and
        # the preview just stopped above may still have a frame in flight, which
        # the new run identity keeps out of this capture entirely.
        self._begin_depth_run()
        # The heartbeat describes THIS capture. Left where the last one finished
        # it would open by claiming frames that are not in this folder.
        self._saved_frames = 0
        self._model_refreshes = 0
        self._model_refreshed_at = None
        self._coverage_well = None
        self._coverage_needs = None
        # And neither does the last capture's SURFACE describe this one. It is
        # left in the viewer rather than thrown away - a fallback would only
        # have to rebuild it - but it stops being the picture, so a new scan
        # cannot open on a derived shell whose triangle count and cell edge
        # belong to a capture that has already finished. The operator's
        # remembered choice is untouched, so the first surface THIS capture
        # builds brings their picture back with it.
        self._show_surface_control(False)
        self._capturing = True
        # One control per capture, made here and handed to the job below, so a
        # Pause pressed against a finished scan has nothing to reach. The action
        # stays disabled until the model panel reports enough accepted geometry
        # to be a reference - see :meth:`_show_coverage`.
        self._pause = PauseControl()
        self._aligned_frames = 0
        # A new capture is a capture that has not lost tracking. Left set from
        # the last one it would disable this one's Pause and end it with the
        # wrong sentence.
        self._tracking_loss = None
        self._tracking_loss_unrecorded = None
        self._set_pause_toggle(False)
        self._pause_action.setEnabled(False)
        self._refresh_capture_progress()
        self._refresh_scan_metrics()
        self._replace_scan_statuses(
            ScanStatus(
                source=SCAN_STATUS_CAPTURE_SOURCE,
                title="STARTING SCAN",
                summary="Waiting for the first depth frame to align.",
                action="Rotate steadily and keep previously scanned detail in view.",
                severity="info",
                priority=20,
            )
        )
        frames = self._frames_spin.value()
        decimate = self._decimate_spin.value()
        hold_cadence = self._hold_cadence.isChecked()
        backend = self._backend.currentText()
        self._report.clear()
        self._log(f"Capturing to {target}")
        self._log("Move all the way around, and come BACK to where you started.")
        self._busy(True)
        # Read here rather than on the worker, once, and used for both things
        # that need an identity: which scanner the readiness wait waits for and
        # whether the colour camera can be pumped alongside the depth stream. A
        # scanner that will not say who it is is still perfectly capturable -
        # None means "any scanner" to the wait and "no colour" to the pump.
        serial = self._scanner_serial()
        control = self._pause

        def job(worker: Worker) -> object:
            from scanner3d.capture_sequence import (
                FRAME_TEMPLATE,
                TRACKING_INPUT_EMPTY_TYPE,
                TRACKING_INPUT_UNUSABLE_TYPE,
                AlignmentFailure,
                TrackingLoss,
                publish_recovery_segment,
                publish_segment,
                publish_tracking_loss,
                read_tracking_loss,
            )
            from scanner3d.geometry import Rigid
            from scanner3d.tracking import (
                ADJACENT_ONLY_POLICY,
                ANCHOR_ORIGIN,
                MODEL_ORIGIN,
                PRIOR_SEED,
                RECOVERY_ORIGIN,
                TRACK_NO_CANDIDATE_TYPE,
                IncrementalTracker,
                TrackDecision,
            )

            try:
                self._await_device(serial)
            except StreamDeviceNotReady:
                # The operator pressed Stop while that wait was running and it
                # then timed out. They are no longer waiting for the scanner,
                # so answering with a permission failure they never asked about
                # would be a dialog about a capture they had already abandoned.
                # A Stop that did NOT happen leaves check() a no-op and the
                # readiness failure propagates untouched.
                worker.check()
                raise
            # Stop pressed while that wait was running - and it can be a long
            # wait: the previous stream has to end and the device has to come
            # back on the bus and become usable. Without this the capture would
            # go on to open the node, take a buffer and write one frame for a
            # sweep the operator has already abandoned. Raising is safe HERE
            # precisely because nothing has been opened yet: no V4L2 buffer is
            # owned, so there is no VIDIOC_QBUF to skip, and no artifact has
            # been written.
            worker.check()

            target.mkdir(parents=True, exist_ok=True)
            kept = 0
            clouds: list[tuple[tuple[float, float, float], ...]] = []
            poses: list[Rigid] = []
            #: What actually decides where every ordinary frame goes. The SAME
            #: module and the same policy the offline reconstruction runs, so a
            #: pose this capture accepts and a pose the reconstruction accepts
            #: are decided by one implementation rather than by two that agree
            #: today. ``clouds`` and ``poses`` above are this window's mirror of
            #: what the tracker accepted - the live model view, the pause
            #: reference and the export all read them - and never a second place
            #: a pose is worked out.
            tracker = IncrementalTracker(backend=backend)
            #: The frame spacing, while the operator asked for it to be held.
            #: ``None`` for the ordinary fixed-cadence capture, in which case
            #: nothing below it does anything at all and the stream is handed
            #: the same plain integer it always was.
            cadence = CadenceController(decimate) if hold_cadence else None
            #: Every cadence change, for the line the operator is shown and for
            #: the run's own summary. A count and the reason rather than a list
            #: per frame: "it moved four times, twice because frames were
            #: refused" is the status; a line per decision is a wall of text at
            #: the rate this delivers.
            cadence_changes: list[object] = []
            #: How many frames each kind of reference placed, so the operator is
            #: told which one is carrying the scan rather than only that it is
            #: still going. Reported on the model-refresh line rather than per
            #: frame: at ten frames a second a per-frame line is not a status,
            #: it is a wall of text.
            placements: dict[str, int] = {}
            #: Whether the operator has already been told, once, that the scan
            #: is being held by something other than the previous frame. The
            #: first time is news; the two hundredth is noise.
            said_model_placement = False
            said_rescue = False
            said_coarse_recovery = False
            said_motion_prior = False
            #: Aligned clouds at the last SUCCESSFUL model refresh; 0 for never.
            #: A count rather than a flag, so a refresh that failed to build
            #: leaves the next frame due for another attempt instead of moving
            #: the goalposts past it.
            refreshed_at = 0
            #: The monotonic time before which no further refresh may be taken,
            #: set from what the last one actually cost. Zero until one has run,
            #: so the first is never held off. See :data:`LIVE_MODEL_DUTY`.
            next_refresh_at = 0.0
            #: The live surface's grid: one occupancy set, grown by every
            #: accepted frame and never re-read. See
            #: :class:`~scanner3d.mesh.SurfaceAccumulator` for why this can be
            #: incremental at all, and for the limitation that matters most -
            #: the cells were placed by the poses that were accepted AT THE TIME
            #: and no pose correction reaches back into them.
            #:
            #: ``None`` until the first refresh, because its cell edge is
            #: measured from the geometry rather than chosen in advance; see
            #: :data:`LIVE_MESH_SPACING_MULTIPLE`.
            accumulator: SurfaceAccumulator | None = None
            #: Aligned clouds already folded into it. The one number that keeps
            #: a frame from being added twice or missed once, and the reason
            #: folding happens at refresh time rather than at each of the two
            #: places a cloud is appended: one place to be right instead of two.
            accumulated = 0
            #: Cell-edge doublings spent because a march came back empty, for
            #: this capture and not for this refresh. See
            #: :data:`LIVE_MESH_COARSEN_ATTEMPTS`.
            retried = 0
            #: Why the live surface was given up on, in the operator's words, or
            #: empty while there is still one to build. Latched deliberately: it
            #: is set when the grid outgrows :data:`LIVE_MESH_CELLS`, and an
            #: occupancy union never shrinks, so retrying would be spending the
            #: whole fold and measure again on every refresh to learn the same
            #: answer. A new capture gets a new one of these, as it gets a new
            #: accumulator.
            withdrawn = ""
            #: Cells in the grid as it was last measured, for the log. Read from
            #: the accumulator where it is already being measured rather than
            #: recomputed at the log line, which over a withdrawn grid would be
            #: an extent pass per refresh answering a question already settled.
            surface_cells = 0
            failures = {"alignment": 0, "model": 0, "relocalization": 0, "surface": 0}
            #: Ordinary live-alignment failures SINCE THE LAST SUCCESS, in the
            #: order they happened. The list rather than a counter because the
            #: durable record needs the frames and the messages themselves, and
            #: because "how long is it" and "what were they" must not be two
            #: variables that can disagree.
            #:
            #: Emptied by any successful alignment. Deliberately not
            #: ``failures["alignment"]``, which is cumulative, rises on a scan
            #: that is working fine and must never decide this.
            streak: list[AlignmentFailure] = []
            #: Whether this capture has lost tracking. Terminal once set: there
            #: is no path back, on purpose, because recovering from a loss means
            #: relating the frames after it to the ones before it and nothing
            #: here can do that. Owned by the stream thread alone, like ``mode``.
            tracking_lost = False
            #: Whether the capture is currently trying to find the object again
            #: by itself. True only between a DURABLE loss and its answer: an
            #: accepted recovery clears both this and ``tracking_lost``, and a
            #: bounded search that runs out clears this alone and leaves the
            #: loss standing. Owned by the stream thread, like ``mode``.
            recovering = False
            #: The loss a recovery must close, as it was read back off disk, and
            #: the run it was recorded in. Read back rather than remembered:
            #: what binds a recovery is the record's own bytes, so the capture
            #: works from the file it just published and not from the values it
            #: believes it wrote.
            recovery_loss: TrackingLoss | None = None
            recovery_dir = target
            #: The refusal to raise once the device has been let go, when the
            #: loss could not be made durable or a recovery could not be
            #: published. Either one is a capture whose disk does not say what
            #: happened, which is the exact silent state all of this exists to
            #: prevent, so it fails the capture instead of ending it quietly.
            loss_failure: BaseException | None = None
            #: Selected frames that arrived after the loss and were dropped.
            #: The 2026-08-28 capture wrote 120-odd of these; counting them is
            #: how a later reader sees that this run did not.
            dropped_after_loss = 0
            #: ``"running"``, ``"paused"`` or ``"relocalizing"``. Owned by this
            #: thread alone; the operator's REQUEST lives in ``control`` and is
            #: read here, which is what keeps the state and the request from
            #: ever being the same variable.
            mode = "running"
            #: The pause epoch this capture has answered. A pause and a resume
            #: that both land between two frames leave the state unchanged and
            #: the epoch advanced, and the epoch is what makes that still count.
            handled_epoch = 0
            #: Where ordinary frames are written NOW: the capture root until a
            #: relocalization is accepted, and that boundary's own directory
            #: afterwards. Global frame indices carry on regardless.
            current_dir = target
            #: Global index of the last frame that both reached disk and
            #: aligned. A pause may only take effect immediately after one of
            #: those, because the bridge is measured against its pose and the
            #: durable record relates it to the frame IMMEDIATELY before the
            #: segment - if the two were different frames the record would be
            #: describing a pair that does not exist.
            last_aligned = -1
            #: Frames a wanted pause has waited for a boundary to arrive.
            waiting_to_pause = 0
            tentative = TentativeSegment.empty()
            #: The thread the global search runs on, made on first use so an
            #: ordinary capture that is never paused never starts one.
            searcher: Executor | None = None
            #: The search currently running, or ``None``. A search in flight is
            #: the only thing that stops another starting, and it also freezes
            #: the held segment: a selected frame that arrives while one is
            #: running is skipped entirely rather than added underneath it, so
            #: the answer always describes the frames it was asked about.
            searching: Future[Relocalization] | None = None
            #: Searches abandoned mid-flight, for the log. An answer for a
            #: segment the operator has already thrown away reaches nobody:
            #: ``searching`` is cleared, so nothing ever asks the future for its
            #: result, and the frames it was about no longer exist.
            abandoned = 0
            #: The epoch of the resume that started the current tentative
            #: segment. Recorded once, at the resume, rather than read off
            #: whichever frame happens to be in hand when the answer arrives.
            resume_epoch = 0
            #: Set by ``on_frame`` when the frame in hand belongs to the
            #: tentative path, and consumed by ``after_frame``. This is the whole
            #: shape of the fix: the pre-requeue callback decides, and every
            #: expensive thing that follows from the decision happens after the
            #: driver has its buffer back.
            tentative_pending = False

            def note_failure(stage: str, exc: BaseException | str) -> None:
                """Say why a live-preview stage failed, without flooding the log.

                These failures were previously swallowed whole. That is defensible
                for the CAPTURE - a frame that will not align is already safely on
                disk and Reconstruct will do the job properly later - but it left
                a stalled-looking model view with no recorded reason anywhere,
                which is precisely the state the owner had to guess about.

                Takes a plain reason as well as an exception: a frame that
                deprojects to nothing raises nothing, and it is still a live
                stage that did not produce what it exists to produce.
                """
                failures[stage] += 1
                count = failures[stage]
                if count <= FAILURES_REPORTED:
                    LOG.warning(
                        "live %s failed on frame %d (%d so far): %s", stage, kept, count, exc
                    )
                elif count % FAILURE_SUMMARY_EVERY == 0:
                    LOG.info("live %s has now failed %d times; latest: %s", stage, count, exc)

            def publish_alignment_status(
                decision: TrackDecision, frame_index: int, *, after_dropout: bool
            ) -> None:
                """Publish only the current tracking answer, never its evidence wall."""
                origin = decision.origin
                detail = decision.message()
                if origin == RECOVERY_ORIGIN:
                    status = ScanStatus(
                        source=SCAN_STATUS_TRACKING_SOURCE,
                        title="TRACKING RECOVERED",
                        summary=f"Frame {frame_index} was recovered from the existing model.",
                        action="Slow down and keep previously scanned detail in view.",
                        severity="warning",
                        priority=55,
                        detail=detail,
                    )
                elif decision.rescue or decision.dissenting:
                    status = ScanStatus(
                        source=SCAN_STATUS_TRACKING_SOURCE,
                        title="TRACKING WEAK",
                        summary=f"Frame {frame_index} aligned, but the available views disagreed.",
                        action="Slow down and turn back toward surface already scanned.",
                        severity="warning",
                        priority=50,
                        detail=detail,
                    )
                elif after_dropout:
                    status = ScanStatus(
                        source=SCAN_STATUS_TRACKING_SOURCE,
                        title="TRACKING RESTORED",
                        summary=f"Frame {frame_index} aligned; the scan is connected again.",
                        action="Continue slowly with overlap to the previous view.",
                        severity="ok",
                        priority=35,
                        detail=detail,
                    )
                else:
                    status = ScanStatus(
                        source=SCAN_STATUS_TRACKING_SOURCE,
                        title="SCANNING — TRACKING OK",
                        summary=f"Frame {frame_index} aligned • {kept} frames saved.",
                        action="Keep rotating steadily.",
                        severity="ok",
                        priority=20,
                        detail=detail,
                    )
                worker.scan_status.emit(status)

            def say_placement(decision: object) -> None:
                """Tell the operator, once, when the scan stops resting on the
                previous frame alone. NEVER raises.

                The distinction matters to the person holding the scanner and it
                is invisible without this. "Still tracking" while every pose
                comes from the accepted model means the individual views have
                gone weak - a flat panel, a fast sweep - and the scan is being
                held up by geometry captured minutes ago. That is not a failure,
                but it is the last warning before a refusal, and an operator who
                is told it can slow down or move back into the overlap.

                Said once per kind per capture. At ten frames a second, a line
                per frame is not a status.
                """
                nonlocal said_model_placement, said_rescue, said_coarse_recovery
                nonlocal said_motion_prior
                origin = getattr(decision, "origin", None)
                # FIRST, and it does not return: this is a statement about how
                # the frame was reached rather than about which reference placed
                # it, so it can be true at the same time as any of the three
                # below. Each of the four is said once in a whole capture, so
                # two lines on one frame is two facts, not a repeat.
                #
                # It is also the most ACTIONABLE of them. The others say the
                # scan is leaning on older geometry; this one says the scanner
                # moved far enough between two frames that the previous one
                # could no longer place this one from a standing start, and the
                # operator can do something about that in the next second.
                if not said_motion_prior and any(
                    getattr(candidate, "accepted", False)
                    and getattr(candidate, "seed", None) == PRIOR_SEED
                    for candidate in getattr(decision, "candidates", ())
                ):
                    said_motion_prior = True
                    message = (
                        "That frame took a second attempt: the previous frame could not place "
                        "it from a standing start, and only did once the solve was started from "
                        "how the scanner was already moving. The scan is still tracking, but "
                        "you are moving far enough between frames that alignment is no longer "
                        "finding the object where it starts looking. Slow down."
                    )
                    LOG.info(
                        "a frame was placed by the adjacent retry seeded from the motion prior"
                    )
                    worker.progress.emit("\n" + message)
                    say(mode, message)
                if origin == RECOVERY_ORIGIN:
                    # Said BEFORE the rescue and model lines, and it is a
                    # different event from either: the ordinary references did
                    # not agree with each other, and this frame is only here
                    # because a coarse search found a placement that two
                    # reference sets sharing no accepted frame both confirmed.
                    # Telling the operator "poses are coming from the accepted
                    # model" for that would describe a scan that is going well.
                    if said_coarse_recovery:
                        return
                    said_coarse_recovery = True
                    message = (
                        "The references disagreed about where that frame went, and a bounded "
                        "coarse search recovered it - two independent parts of the model agreed "
                        "on the placement. The scan is still connected, but this is the last "
                        "warning before a refusal: slow down, and move back over surface you "
                        "have already scanned."
                    )
                    LOG.info("tracking held by a bounded coarse recovery")
                    worker.progress.emit("\n" + message)
                    say(mode, message)
                    return
                if getattr(decision, "rescue", False) and not said_rescue:
                    said_rescue = True
                    message = (
                        "The previous frame could not place this one on its own; the accepted "
                        "model did, against several earlier viewpoints. The scan is still "
                        "tracking, but individual views have gone weak - move more slowly, and "
                        "keep surface you have already scanned in view."
                    )
                    LOG.info("tracking held by a model-only rescue")
                    worker.progress.emit("\n" + message)
                    say(mode, message)
                    return
                if origin == MODEL_ORIGIN and not said_model_placement:
                    said_model_placement = True
                    LOG.info("tracking is being carried by the accepted model reference")
                    worker.progress.emit(
                        "\nPoses are now coming from the accepted model rather than from the "
                        "previous frame alone. That is the tracker doing its job on weaker "
                        "views; the scan is still connected."
                    )

            def note_cadence(travel: float | None) -> None:
                """Offer one measurement to the frame spacing. NEVER raises.

                ``travel`` is an accepted frame's measured surface travel, or
                ``None`` for a frame that could not be placed at all. Both are
                inputs; see :mod:`scanner3d.cadence` for why the second one is.

                Does nothing at all when the operator did not ask for the
                spacing to be held, which is the default - so an ordinary
                capture runs the code path it always did and hands the stream
                the same fixed number.

                Never raises for the same reason its callers do not: this runs
                inside the stream's own loop, between the driver handing a
                buffer over and getting it back.
                """
                if cadence is None:
                    return
                try:
                    change = cadence.placed(travel) if travel is not None else cadence.refused()
                except Exception as exc:
                    # A controller that cannot decide leaves the spacing where
                    # it is, which is a working capture at the operator's own
                    # setting. It must never be the thing that ends one.
                    LOG.warning("the cadence controller refused a measurement: %s", exc)
                    return
                if not change.changed:
                    return
                cadence_changes.append(change)
                LOG.info(
                    "frame spacing %d -> %d (%s)", change.previous, change.current, change.reason
                )

            def note_tracking_failure(frame_index: int, kind: str, message: str) -> None:
                """Extend the consecutive-failure streak, and act if it is long
                enough. NEVER raises.

                One failure is a bad frame and two is a bad moment; see
                :data:`~scanner3d.capture_sequence.TRACKING_LOSS_STREAK` for why
                the third in a row is not.

                Every way a selected frame can fail to be placed comes through
                here, not only a failed alignment: a frame that deprojects to
                nothing, and a deprojection that raises, leave the model exactly
                as stationary as a singular ICP solve does, and a mechanism that
                counted only the last of the three would let the same frozen
                model run forever on either of the other two.

                A frame index that does not follow the previous one RESTARTS the
                streak. Ordinarily it cannot happen - every selected frame is
                either accepted or recorded here - but a frame skipped for any
                other reason (a Stop landing mid-frame) breaks the run, and a
                streak is a run. It is also what keeps the published record's
                own shape true: the reader requires one consecutive run of
                failures, so a streak that had a hole in it could not be
                published at all.

                Nothing is recorded once the capture is cancelling. A Stop is
                not a tracking failure, and a frame dropped because the operator
                ended the run must never be a third of a loss.
                """
                if worker.cancelled:
                    return
                # BEFORE the streak, and after the cancel guard. Every way a
                # selected frame can fail to be placed already funnels through
                # here, which is exactly what the cadence wants: a refusal is
                # the one speed signal that arrives before the travel
                # measurement that would have explained it, because a frame that
                # was not placed has no travel. A Stop is not one, which the
                # guard above has already settled.
                note_cadence(None)
                if streak and frame_index != streak[-1].frame_index + 1:
                    streak.clear()
                streak.append(
                    AlignmentFailure(frame_index=frame_index, failure_type=kind, message=message)
                )
                failed = len(streak)
                safe_through = (
                    f"frame {last_aligned}" if last_aligned >= 0 else "no aligned frame yet"
                )
                worker.scan_status.emit(
                    ScanStatus(
                        source=SCAN_STATUS_TRACKING_SOURCE,
                        title=(
                            "TRACKING LOST"
                            if failed >= TRACKING_LOSS_STREAK
                            else f"TRACKING WARNING — {failed} OF {TRACKING_LOSS_STREAK}"
                        ),
                        summary=(
                            f"Frame {frame_index} did not align; the scan is safe through "
                            f"{safe_through}."
                        ),
                        action="Stop moving and turn back toward the last good view.",
                        severity="error" if failed >= TRACKING_LOSS_STREAK else "warning",
                        priority=90 if failed >= TRACKING_LOSS_STREAK else 60 + failed,
                        detail=message,
                    )
                )
                if len(streak) >= TRACKING_LOSS_STREAK:
                    enter_tracking_lost()

            def note_alignment_failure(frame_index: int, exc: BaseException) -> None:
                """One live alignment that raised, as a tracking failure. NEVER raises."""
                note_tracking_failure(frame_index, type(exc).__name__, str(exc))

            def enter_tracking_lost() -> None:
                """Stop the scan pretending it knows where the object is. NEVER raises.

                Three steps, and the ORDER is the whole of this function:

                1. The containment flag is set FIRST, before the record is even
                   attempted. From this line on no frame is written, no count
                   moves, no pose is placed and no model is refreshed, whatever
                   the disk does next. This runs on the stream thread, which is
                   the only thread that reads it, so setting it is the whole of
                   "atomically".
                2. The record is published.
                3. ONLY THEN is the formal ``tracking_lost`` state reported.

                Containment and the reported loss are deliberately two things.
                Stopping is always safe and always right, so it happens first
                and unconditionally. But "this capture lost tracking" is a claim
                about what is ON DISK - it sends the operator to a file and it
                tells every later reader the folder explains itself - and that
                claim may not be made before the file it refers to exists. So a
                publication that fails reports :data:`TRACKING_LOSS_UNRECORDED_STATE`
                instead, which says the opposite in as many words.

                A publication that fails does NOT leave the capture running
                either. An unmarked capture that has silently stopped tracking
                is the 2026-08-28 failure exactly, and it is strictly worse than
                a capture that ends with an error on screen - so the capture is
                stopped and ``loss_failure`` fails it once the device is let go.
                """
                nonlocal tracking_lost, loss_failure
                tracking_lost = True
                frames_failed = ", ".join(str(item.frame_index) for item in streak)
                frozen = (
                    f"frame {last_aligned}"
                    if last_aligned >= 0
                    else "nothing at all - no frame in this capture ever aligned"
                )
                try:
                    # Into the run that actually lost tracking, which is the
                    # capture root until a boundary has been accepted and that
                    # boundary's own directory afterwards. A record always
                    # written at the root would be overwritten - and therefore
                    # refused - by a second loss in a long capture, and would
                    # describe frames that are not in the directory it sits in.
                    record = publish_tracking_loss(
                        current_dir,
                        last_aligned_frame_index=last_aligned,
                        failures=streak,
                    )
                except Exception as exc:
                    loss_failure = exc
                    LOG.error(
                        "tracking lost after %d saved frames (failed frames %s) AND the loss "
                        "record could not be written: %s",
                        kept,
                        frames_failed,
                        exc,
                    )
                    detail = (
                        f"TRACKING LOST after {kept} frames, and the record of it could NOT be "
                        f"written: {exc}. There is NO {TRACKING_LOSS_RECORD_NAME} in this "
                        "folder and nothing on disk says where these frames stop being one "
                        "scan. Nothing more is being written. This capture is stopping now and "
                        "will report that failure; the frames already on disk are untouched."
                    )
                    worker.progress.emit("\n" + detail)
                    # Never TRACKING_LOST_STATE. That state is the capture's
                    # claim that a durable record exists, and here one does not.
                    say(TRACKING_LOSS_UNRECORDED_STATE, detail)
                    # The capture cannot be left running: every frame it went on
                    # to write would be an unmarked frame in a sequence nothing
                    # records as broken.
                    worker.cancel()
                    return
                LOG.error(
                    "tracking lost after %d saved frames: %d consecutive live alignment "
                    "failures on frames %s; model frozen at %s; recorded in %s",
                    kept,
                    len(streak),
                    frames_failed,
                    frozen,
                    record,
                )
                detail = (
                    f"TRACKING LOST after {kept} frames. Live alignment failed "
                    f"{len(streak)} times in a row, on frames {frames_failed}, and the scan no "
                    f"longer knows where the object is. The model is FROZEN at {frozen}; the "
                    "frames after it are NOT a valid continuation of the scan and nothing more "
                    "is being written. Every frame written before the loss is safe on disk and "
                    f"{record.name} beside them records what happened. Reconstruct will refuse "
                    "this folder until the loss is closed."
                )
                # The prospect of a recovery is decided BEFORE the loss is
                # announced, so that the operator is told once what has
                # happened and what happens next, rather than being told the
                # scan is over and then contradicted a line later.
                loss, why_not = recovery_prospect(record)
                worker.progress.emit("\n" + detail + why_not)
                say(TRACKING_LOST_STATE, detail + why_not)
                if loss is not None:
                    enter_recovering(loss, record.parent)

            def recovery_prospect(record: Path) -> tuple[TrackingLoss | None, str]:
                """Whether an automatic recovery can honestly be tried. NEVER raises.

                Returns the loss to recover from, read back off the disk, or
                ``None`` and the sentence saying why not. Read back rather than
                remembered: what binds a recovery is the record's own bytes, so
                the capture works from the file it has just published and not
                from the values it believes it wrote.
                """
                if len(clouds) < MIN_PAUSE_REFERENCE_CLOUDS:
                    # Below this there is no model to recognise the object by,
                    # so a search could only ever end in a refusal - after
                    # spending the whole bounded budget to get there, and after
                    # telling the operator something was being tried.
                    return None, (
                        f" Only {len(clouds)} frames had aligned before the loss, under the "
                        f"{MIN_PAUSE_REFERENCE_CLOUDS} needed for the scan to recognise the "
                        "object again, so nothing can be placed against them and no recovery "
                        "will be attempted. Press Stop and start a new scan."
                    )
                try:
                    loss = read_tracking_loss(record.parent)
                except Exception as exc:  # pragma: no cover - just published
                    note_failure("relocalization", exc)
                    loss = None
                if loss is None:
                    return None, (
                        " The record was written and could not be read back, so nothing can be "
                        "bound to it and no recovery will be attempted. Press Stop."
                    )
                return loss, ""

            def enter_recovering(loss: TrackingLoss, loss_dir: Path) -> None:
                """Begin trying to find the object again. NEVER raises.

                Reached only from the line after a successful publication, and
                that ordering is the contract: there is nothing to recover FROM
                until the loss is durable, and a capture that began collecting
                against an unwritten loss could publish a segment binding a
                record that does not exist.

                Nothing is relaxed here. The frames are held in exactly the
                bounded container a resume uses, chained by exactly the same
                judged local edges, and placed - if they are placed at all - by
                exactly the same strict global relocalization against the model
                already accepted. What is different is only that the operator
                did not ask for it.
                """
                nonlocal recovering, recovery_loss, recovery_dir
                recovery_loss = loss
                recovery_dir = loss_dir
                recovering = True
                abandon_search()
                tentative.clear(now=time.monotonic())
                LOG.info(
                    "recovering: collecting up to %d tentative frames to relocalize against "
                    "%d accepted clouds",
                    MAX_TENTATIVE_FRAMES,
                    len(clouds),
                )
                detail = (
                    "Trying to find the object again. The scan is collecting a small segment "
                    "of frames and looking for the model it had already accepted. None of "
                    "those frames are in the scan and none of them are on disk. Move back "
                    "over a part you have already scanned."
                )
                worker.progress.emit("\n" + detail)
                say(RECOVERING_TRACKING_STATE, detail)

            def on_delivery(buf: bytes, _record: object) -> None:
                """Show the depth camera at the rate it delivers. NEVER raises.

                Every complete delivery, including the ones decimation does not
                keep. The picture is a one-slot emission that costs the producer
                a lock and a pointer swap, while a keep costs a half-megabyte
                write, a deprojection and an alignment - so tying the first to
                the second bought nothing and cost the operator two thirds of
                the depth view's motion at the default decimation of 3.

                A capture that is stopping still shows what it is being handed:
                this is the frame the scanner delivered a moment ago, drawing it
                is free, and a view that goes dark before the stream does is a
                Stop that looks like a crash.
                """
                worker.preview.emit(buf)

            def say(state: str, detail: str, *, paused: bool = False) -> None:
                """Publish where the pause has got to, in one place. NEVER raises."""
                worker.pause_state.emit(PauseState(state=state, detail=detail, paused=paused))

            def enter_paused() -> None:
                """Stop committing frames. Previews keep running; nothing is kept."""
                nonlocal mode, waiting_to_pause
                mode = "paused"
                waiting_to_pause = 0
                # A search still running for the segment being thrown away is
                # abandoned rather than waited for: its answer is about frames
                # that no longer exist, and the operator is holding the object.
                abandon_search()
                tentative.clear()
                detail = (
                    f"PAUSED after {kept} frames. Nothing is being written or added to the "
                    f"model. Reposition the object, then press {RESUME_TITLE} to carry on - "
                    "the scan will need to see surface it has already scanned, and a part of "
                    "it that does not look like every other part, to work out where the "
                    "object went."
                )
                LOG.info("capture paused at %d saved frames, %d aligned", kept, len(clouds))
                worker.progress.emit(f"\nPaused. Reposition, then press {RESUME_TITLE}.")
                say("paused", detail, paused=True)

            def refuse_pause(epoch: int, why: str) -> None:
                """Answer a pause that cannot be honoured, and put the button back up."""
                nonlocal mode, handled_epoch, waiting_to_pause
                mode = "running"
                waiting_to_pause = 0
                handled_epoch = epoch
                abandon_search()
                control.resume()
                LOG.info("pause refused: %s", why)
                worker.progress.emit(f"\nPause not taken: {why}")
                say("running", f"Pause not taken: {why}", paused=False)

            def enter_relocalizing(epoch: int) -> None:
                """Begin a tentative mini-segment. Nothing here reaches the scan."""
                nonlocal mode, resume_epoch
                mode = "relocalizing"
                resume_epoch = epoch
                abandon_search()
                tentative.clear(now=time.monotonic())
                LOG.info("relocalizing: collecting up to %d tentative frames", MAX_TENTATIVE_FRAMES)
                worker.progress.emit(
                    "\nResumed. Collecting a few frames and looking for surface already "
                    "scanned, to work out where the object is now."
                )
                say("relocalizing", "Resuming - collecting frames to place the new position.")

            def say_tentative(detail: str) -> None:
                """Where the held frames have got to, on whichever state owns them.

                One tentative buffer, two reasons to be filling it: a resume the
                operator asked for, and a recovery nobody did. The window drives
                a different label and a different Pause title from each, so
                every line about the held frames has to travel on the state that
                is actually in play - a recovery reporting itself as an ordinary
                relocalization would put the Pause control back up mid-loss.
                """
                say(RECOVERING_TRACKING_STATE if recovering else "relocalizing", detail)

            def recovery_expired() -> str | None:
                """Why the automatic recovery's own clock has run out, or ``None``.

                An automatic recovery has no operator behind it: nobody pressed
                anything to start it and nobody is going to press anything to
                end it, so the only thing that can end one that is not working
                is this clock. It is an ACCEPTANCE BOUNDARY rather than a
                watchdog for a hung search, so it is checked BEFORE any further
                work on a post-loss delivery and BEFORE any search result is
                looked at - not only after a frame has successfully deprojected,
                chained and entered the buffer, which is the one path a capture
                stuck on unusable or locally inconsistent frames never reaches,
                and not only when a future is unfinished, which is the one state
                a search that comes back accepted too late is never in.

                Only while ``recovering``. An operator's resume is bounded by
                the same constant in the same place it always was, because there
                a person is watching, holding the object and able to press
                Pause again; changing that here would change Pause/Resume.
                """
                if not recovering:
                    return None
                spent = time.monotonic() - tentative.started
                if spent <= MAX_TENTATIVE_SECONDS:
                    return None
                return (
                    f"Still cannot tell where the object is {spent:.0f}s after tracking was "
                    f"lost, past the {MAX_TENTATIVE_SECONDS:.0f}s this search is allowed, over "
                    f"{tentative.attempts} attempt(s) and {tentative.frames} held frame(s)."
                )

            def report_collecting() -> None:
                if recovering:
                    say_tentative(
                        f"Looking for where the object went: {tentative.frames} of at least "
                        f"{MIN_TENTATIVE_FRAMES} frames collected, none of them in the scan. "
                        "Move back over a part you have already scanned."
                    )
                    return
                say(
                    "relocalizing",
                    f"Placing the new position: {tentative.frames} of at least "
                    f"{MIN_TENTATIVE_FRAMES} frames collected. None of them are in the scan "
                    "yet. Move back over a part you have already scanned.",
                )

            def accept_segment(result: Relocalization, epoch: int) -> bool:
                """Publish the buffered batch and join it to the model. NEVER raises.

                Everything or nothing, in this order: the raw frames and the
                boundary record become visible as one directory rename first,
                and only once that has returned do the counts move, the poses
                grow and a model reach the screen. A publication that fails
                leaves the segment tentative and the capture folder untouched,
                which is the state it was already in.
                """
                nonlocal kept, refreshed_at, mode, handled_epoch, current_dir, last_aligned
                bridge = result.bridge
                if bridge is None:
                    return False
                batch = [(kept + offset, raw) for offset, raw in enumerate(tentative.raws)]
                # Relative to the frame IMMEDIATELY before the segment, which is
                # the last aligned frame because a pause is only taken at such a
                # boundary. That is the direction a sequential pose chain
                # composes, so a reconstruction drops it in exactly where it
                # would have dropped in a fitted pair.
                relative = poses[-1].inverse().compose(bridge)
                # The global decision AND the chain it was decided over. A
                # record that carried only the first says where a segment was
                # put and nothing about whether the segment was a segment; the
                # local edges are what a later reader needs to re-judge that.
                evidence = dict(result.as_dict())
                evidence["local_chain"] = tentative.chain_record()
                try:
                    publication = publish_segment(
                        target,
                        frames=batch,
                        predecessor_index=kept - 1,
                        transform=relative,
                        evidence=evidence,
                        limitations=result.limitations,
                    )
                except Exception as exc:
                    note_failure("relocalization", exc)
                    return False
                current_dir = publication.directory
                for offset, (cloud, local) in enumerate(
                    zip(tentative.clouds, tentative.poses, strict=True)
                ):
                    placed = bridge.compose(local)
                    clouds.append(cloud)
                    poses.append(placed)
                    # Seeded, never refitted. The boundary was decided by the
                    # relocalization against evidence in its own record, and the
                    # tracker continues from there against the same accepted
                    # model - which is exactly what the offline reconstruction
                    # does with the bridge this publication just wrote.
                    tracker.seed(cloud, placed, anchor=offset == 0)
                kept += len(batch)
                last_aligned = kept - 1
                mode = "running"
                handled_epoch = epoch
                summary = evidence_summary(result)
                tentative.clear()
                worker.saved.emit(kept)
                LOG.info(
                    "relocalization accepted: %s; published %d frames as %s",
                    summary,
                    len(batch),
                    publication.directory.name,
                )
                worker.progress.emit(
                    f"\nPlaced. {len(batch)} frames joined the scan through "
                    f"{publication.directory.name} - {summary}"
                )
                say(
                    "running",
                    f"Placed: {len(batch)} frames joined the scan ({summary}).",
                )
                # Immediately, and not at the next scheduled refresh: the whole
                # point of the wait was to find out where the new frames go, and
                # an operator who has just been told it worked has to be able to
                # SEE it worked.
                refresh_model(force=True)
                return True

            def accept_recovery(result: Relocalization) -> bool:
                """Close the loss with a published recovery, or fail. NEVER raises.

                The same order ``accept_segment`` uses, for the same reason: the
                frames, the bridge AND the binding to the loss record become
                visible as one directory rename first, and only once that has
                returned do the counts move, the poses grow, the lost state
                clear and a model reach the screen.

                The one deliberate difference is what a failed publication
                means. A pause boundary that cannot be published leaves the
                segment tentative and the capture exactly as it was, which is a
                state the operator can act on. A recovery that cannot be
                published is not that: the capture has a durable loss, it has
                just been told it found the object, and carrying on would leave
                a folder whose loss looks closed. So it FAILS the capture, and
                it never reports a recovery it did not write.
                """
                nonlocal kept, mode, current_dir, last_aligned
                nonlocal recovering, recovery_loss, tracking_lost, loss_failure
                bridge = result.bridge
                loss = recovery_loss
                if bridge is None or loss is None:
                    return False
                batch = [(kept + offset, raw) for offset, raw in enumerate(tentative.raws)]
                # Against the last frame that ALIGNED, which is the frame the
                # loss record names - not the last frame on disk, which is the
                # end of the failed streak and has no pose at all.
                relative = poses[-1].inverse().compose(bridge)
                evidence = dict(result.as_dict())
                evidence["local_chain"] = tentative.chain_record()
                excluded = len(loss.failures)
                try:
                    publication = publish_recovery_segment(
                        target,
                        loss_dir=recovery_dir,
                        loss=loss,
                        frames=batch,
                        transform=relative,
                        evidence=evidence,
                        limitations=result.limitations,
                    )
                except Exception as exc:
                    note_failure("relocalization", exc)
                    loss_failure = exc
                    recovering = False
                    abandon_search()
                    tentative.clear()
                    LOG.error(
                        "tracking was relocalized after the loss and the recovery could NOT be "
                        "published: %s",
                        exc,
                    )
                    detail = (
                        "The scan found the object again, and the recovery could NOT be "
                        f"written: {exc}. NOTHING was published, so the loss still stands and "
                        "none of the held frames are on disk. This capture is stopping now and "
                        "will report that failure; every frame written before the loss is "
                        "untouched."
                    )
                    worker.progress.emit("\n" + detail)
                    say(TRACKING_LOST_STATE, detail)
                    worker.cancel()
                    return False
                current_dir = publication.directory
                for offset, (cloud, local) in enumerate(
                    zip(tentative.clouds, tentative.poses, strict=True)
                ):
                    placed = bridge.compose(local)
                    clouds.append(cloud)
                    poses.append(placed)
                    # Seeded, never refitted - see ``accept_segment``.
                    tracker.seed(cloud, placed, anchor=offset == 0)
                kept += len(batch)
                last_aligned = kept - 1
                # Only now, and only in this order: the loss is closed on disk,
                # so the capture may stop being a capture that lost tracking.
                recovering = False
                tracking_lost = False
                recovery_loss = None
                streak.clear()
                mode = "running"
                summary = evidence_summary(result)
                tentative.clear()
                worker.saved.emit(kept)
                LOG.info(
                    "tracking recovered: %s; published %d frames as %s, excluding %d failed "
                    "frame(s)",
                    summary,
                    len(batch),
                    publication.directory.name,
                    excluded,
                )
                detail = (
                    f"TRACKING RECOVERED. {len(batch)} frames joined the scan through "
                    f"{publication.directory.name} - {summary}. The {excluded} frame(s) written "
                    "while tracking was lost are still on disk as evidence and are NOT part of "
                    "the scan. Scanning has carried on."
                )
                worker.progress.emit("\n" + detail)
                say(TRACKING_RECOVERED_STATE, detail)
                refresh_model(force=True)
                return True

            def abandon_recovery(reason: str) -> None:
                """The bounded search did not find the object. NEVER raises.

                Nothing is written, nothing is retried, and the loss stays
                exactly as it was recorded. That is the whole answer: an
                automatic recovery that kept trying would spend the rest of the
                capture proving the same thing, and one that lowered a gate to
                succeed would produce the confident wrong model this exists to
                prevent.
                """
                nonlocal recovering
                recovering = False
                abandon_search()
                held = tentative.frames
                tentative.clear()
                frozen = (
                    f"frame {last_aligned}"
                    if last_aligned >= 0
                    else "nothing at all - no frame in this capture ever aligned"
                )
                detail = (
                    f"TRACKING LOST and NOT recovered. {reason} The {held} frames held while "
                    "looking were never written and are gone; nothing has been added to the "
                    f"scan and nothing more will be. The model is FROZEN at {frozen}, the "
                    "frames written while tracking was lost are still on disk as evidence, and "
                    f"{TRACKING_LOSS_RECORD_NAME} records what happened. Press Stop and rescan "
                    "- Reconstruct will refuse this folder while the loss is unresolved."
                )
                LOG.warning("automatic recovery gave up: %s", reason)
                worker.progress.emit("\n" + detail)
                say(TRACKING_LOST_STATE, detail)

            def grow_surface() -> Mesh | None:
                """Fold every frame not yet folded in, and march. NEVER raises.

                Returns ``None`` rather than propagating, because a surface is
                the part of the picture the operator can do without: a capture
                whose mesh fails must still show its points, still write its
                frames and still finish.

                Every accepted frame reaches the grid, not the
                :data:`LIVE_MODEL_CLOUDS` the point picture samples. That is
                affordable here precisely because it is a grid: folding a frame
                in costs the frame, and what it costs to MARCH is bounded by the
                object's own extent rather than by how many frames have been
                folded.

                Also returns ``None``, permanently, once the object outgrows
                :data:`LIVE_MESH_CELLS`. It does NOT coarsen to fit: see
                ``withdrawn`` below.
                """
                nonlocal accumulator, accumulated, retried, withdrawn, surface_cells
                # Bounded by construction rather than by trying again cheaply: an
                # occupancy union only ever grows, and the grid extent with it,
                # so a build that did not fit cannot come to fit. Returning here
                # is what stops a scan that outgrew the budget from folding and
                # re-measuring its whole grid every few frames for the rest of
                # the capture.
                if withdrawn:
                    return None
                try:
                    # Into the UPRIGHT presentation frame, the same one the
                    # points are drawn in and the exported files are written in.
                    # A surface and a cloud in different frames would be two
                    # objects on one screen.
                    pending = [
                        tuple(upright_pose(poses[index]).apply_all(clouds[index]))
                        for index in range(accumulated, len(clouds))
                    ]
                    if accumulator is None:
                        # Sized from the geometry the capture has, the first time
                        # it has any. Before this there is no grid at all, so a
                        # refresh that arrives too early costs nothing and the
                        # next one measures again.
                        flat = [point for cloud in pending for point in cloud]
                        if not flat:
                            return None
                        edge = estimate_spacing(flat).spacing * LIVE_MESH_SPACING_MULTIPLE
                        accumulator = SurfaceAccumulator(voxel_size=edge)
                        LOG.info(
                            "live surface grid sized at voxel %.3f from %d points of %d frames",
                            edge,
                            len(flat),
                            len(pending),
                        )
                    for cloud in pending:
                        accumulator.add(cloud)
                    accumulated = len(clouds)
                    if accumulator.is_empty:
                        return None
                    # The estimate above is a guess about gaps made from a
                    # measurement of typical distances, so it is checked against
                    # what it produced rather than trusted. An empty march means
                    # the cells are isolated; a coarser one joins them up.
                    while True:
                        surface_cells = accumulator.grid_cells
                        if surface_cells > LIVE_MESH_CELLS:
                            # The whole point of this task. What used to happen
                            # here was coarsen_to_fit: the sole fine accumulator
                            # was destructively re-cut at twice the cell edge and
                            # the panel went on showing the result as though the
                            # fidelity were unchanged. The mesher's surface
                            # already stands up to a voxel outside its samples,
                            # so that DOUBLED the error, swallowed thin features
                            # and produced the blob the owner reported.
                            #
                            # There is no coarser answer worth showing, so no
                            # answer is shown. The points are the authoritative
                            # picture and they are still current.
                            nx, ny, nz = accumulator.grid_shape
                            # The closing sentence is measured against what
                            # Reconstruct does rather than against how much
                            # better it sounds. It reads every captured frame
                            # and rebuilds from the corrected poses, and its
                            # mesh is fused from the full-resolution frames -
                            # but the voxel it fuses at is still derived from
                            # the placed extent and RECONSTRUCT_VOXEL_BUDGET
                            # rather than from the depth image, and the cloud it
                            # writes is still thinned to RECONSTRUCT_POINTS with
                            # isolated samples dropped. So it is not "full
                            # resolution" either, and a message whose whole
                            # purpose is to refuse an overstated picture cannot
                            # end on one.
                            withdrawn = (
                                f"No live surface: the scan now needs a {nx}x{ny}x{nz} grid of "
                                f"{surface_cells} cells at cell {accumulator.voxel_size:.2f}, "
                                f"past this preview's {LIVE_MESH_CELLS}-cell budget. The points "
                                "are the picture from here on; Reconstruct still reads every "
                                "captured frame, rebuilds from the corrected poses, and sizes "
                                "its own mesh resolution under its own separate budget."
                            )
                            LOG.warning(
                                "live surface withdrawn after %d folded frames: %s "
                                "The grid is NOT coarsened to fit - doubling the cell edge "
                                "doubles how far the surface stands from the samples - and it "
                                "is not attempted again for this capture, because an occupancy "
                                "union only grows.",
                                accumulated,
                                withdrawn,
                            )
                            # Once, in the running report as well as beside the
                            # coverage numbers. The panel sentence says what is
                            # true NOW and is easy to have already been looking
                            # away from; this one says WHEN it stopped being a
                            # surface, which is what an operator comparing this
                            # scan with the last one needs.
                            worker.progress.emit(f"\n{withdrawn}")
                            return None
                        marched = accumulator.surface(max_grid_cells=LIVE_MESH_CELLS)
                        if marched.faces or retried >= LIVE_MESH_COARSEN_ATTEMPTS:
                            return marched
                        retried += 1
                        LOG.info(
                            "live surface marched to no faces at voxel %.3f over %d cells; "
                            "doubling the cell edge (%d of %d allowed for this capture)",
                            accumulator.voxel_size,
                            accumulator.occupied_cells,
                            retried,
                            LIVE_MESH_COARSEN_ATTEMPTS,
                        )
                        accumulator.coarsen()
                except Exception as exc:
                    note_failure("surface", exc)
                    return None

            def refresh_model(*, force: bool = False) -> None:
                """Build and publish one accumulated model, if one is due. NEVER raises."""
                nonlocal refreshed_at, next_refresh_at
                aligned = len(clouds)
                if aligned < MIN_COVERAGE_CLOUDS:
                    return
                due = force or refreshed_at == 0 or aligned - refreshed_at >= COVERAGE_EVERY
                if not due or worker.cancelled:
                    return
                started = time.monotonic()
                # Held off rather than skipped, and ``refreshed_at`` is left
                # exactly where it is so the next aligned frame asks again. This
                # is what bounds the stream thread's share of a refresh that has
                # become expensive because the object is large - see
                # :data:`LIVE_MODEL_DUTY`. Never against a forced refresh: that
                # one exists because the operator has just been told something
                # happened and has to be able to see it.
                if not force and started < next_refresh_at:
                    return
                surface = grow_surface()
                try:
                    snapshot = _live_snapshot(
                        clouds,
                        poses,
                        saved=kept,
                        aligned=aligned,
                        surface=surface,
                        surface_voxel=0.0 if accumulator is None else accumulator.voxel_size,
                        surface_note=withdrawn,
                        # The first model this capture has produced, and the
                        # only one entitled to move the camera: the previous
                        # capture's framing describes a different object, and
                        # every refresh after this one is an addition to the
                        # picture the operator is already looking at.
                        reframe=refreshed_at == 0,
                    )
                except Exception as exc:
                    # ``refreshed_at`` deliberately unchanged: the next aligned
                    # frame tries again. A model that fails to build once must
                    # not be a model that never builds.
                    note_failure("model", exc)
                    return
                refreshed_at = aligned
                elapsed = time.monotonic() - started
                # Measured here and applied to the NEXT one, which is the only
                # honest way round: what a refresh costs is a property of the
                # grid the object has grown, and nothing knows it in advance. A
                # refresh that cost less than a frame does arms nothing at all,
                # so the ordinary cadence is exactly what it always was.
                next_refresh_at = (
                    time.monotonic() + elapsed * (1.0 / LIVE_MODEL_DUTY - 1.0)
                    if elapsed > LIVE_MODEL_REFRESH_FLOOR
                    else 0.0
                )
                # "Produced" is the half of the story this side of the boundary
                # can vouch for. Whether it was ever DRAWN is the window's to
                # report, and the two being separate lines is what lets a
                # physical report distinguish a model that failed to build from
                # one the GUI could not keep up with. Once every
                # COVERAGE_EVERY alignments, so this is not per-frame spam.
                # The placement breakdown rides on this line rather than one of
                # its own, and that is not tidiness: this is at INFO precisely
                # because it does not fire per frame, and a second line on the
                # same cadence would double what a long capture writes to say
                # one thing. "Aligned 40 frames" does not distinguish a capture
                # whose every pose came from its immediate neighbour from one
                # being held up entirely by geometry captured minutes ago, and
                # only the second is about to stop working.
                LOG.info(
                    "live model produced: %d points from %d of %d aligned frames "
                    "in %.3fs, at %d saved; surface %d faces over %d cells at voxel %.3f%s; "
                    "placed by %s over %d keyframes",
                    len(snapshot.points),
                    snapshot.sampled,
                    aligned,
                    elapsed,
                    kept,
                    len(snapshot.surface_faces),
                    surface_cells,
                    0.0 if accumulator is None else accumulator.voxel_size,
                    " (surface withdrawn)" if withdrawn else "",
                    ", ".join(f"{count} {origin}" for origin, count in sorted(placements.items()))
                    or "nothing yet",
                    len(tracker.keyframes),
                )
                worker.coverage.emit(snapshot)

            def abandon_search() -> None:
                """Stop caring about a search still running. NEVER blocks.

                Cancelling only takes effect if the worker has not picked the
                task up yet, and that is fine: the snapshots it was handed are
                tuples nobody else will touch, so a search nobody is waiting for
                finishes harmlessly into a result that is dropped. Waiting for
                it is what must not happen - the operator has just pressed
                something, and this runs on the stream thread.
                """
                nonlocal searching, abandoned
                if searching is not None:
                    # Cancelling only works if the worker has not started it.
                    # Either way this stops caring: ``abandoned`` is what makes
                    # an answer that arrives anyway reach nobody.
                    searching.cancel()
                    abandoned += 1
                searching = None

            def start_search() -> None:
                """Ask, off this thread, where the held segment belongs. NEVER raises."""
                nonlocal searching, searcher
                # Snapshots, taken here on the stream thread, of everything the
                # search reads. The accepted model is not mutated while a resume
                # is in play - no frame is committed until the segment is - but
                # a search reading lists another thread OWNS is a race waiting
                # for the first exception to that rule, and a tuple costs
                # pointers.
                segment_clouds = tuple(tentative.clouds)
                segment_poses = tuple(tentative.poses)
                model_clouds = tuple(clouds)
                model_poses = tuple(poses)

                def search() -> Relocalization:
                    reference = bounded_reference(model_clouds, model_poses)
                    return relocalize(
                        list(segment_clouds),
                        list(segment_poses),
                        reference,
                        backend=backend,
                    )

                try:
                    if searcher is None:
                        searcher = _relocalization_executor()
                    searching = searcher.submit(search)
                except Exception as exc:
                    # A thread that could not be started is a refusal to search,
                    # not a capture that ends. The frames stay held and the next
                    # one tries again.
                    note_failure("relocalization", exc)
                    searching = None
                    report_collecting()
                    return
                say_tentative(
                    f"Looking for where the object went: attempt {tentative.attempts} of "
                    f"{MAX_ATTEMPTS}, over {tentative.frames} held frames. The scanner keeps "
                    "running while it looks, and none of those frames are in the scan."
                )

            def take_search_result() -> bool:
                """Apply a finished search, if one finished. NEVER raises.

                Returns whether the search slot is free - false means one is
                still running, and the caller must not start another or add to
                the segment underneath it.
                """
                nonlocal searching
                if searching is None:
                    return True
                if not searching.done():
                    return False
                finished = searching
                searching = None
                try:
                    result = finished.result()
                except CancelledError:
                    return True
                except Exception as exc:
                    note_failure("relocalization", exc)
                    report_collecting()
                    return True
                if result.accepted:
                    if recovering:
                        # Publishes its own state whichever way it went, and a
                        # failure has already stopped the capture, so there is
                        # never a "still collecting" line to print after it.
                        accept_recovery(result)
                        return True
                    if accept_segment(result, resume_epoch):
                        return True
                    report_collecting()
                    return True
                reason = result.refusals[0] if result.refusals else "not enough evidence yet"
                LOG.info(
                    "relocalization attempt %d of %d refused: %s",
                    tentative.attempts,
                    MAX_ATTEMPTS,
                    reason,
                )
                worker.progress.emit(f"  not placed yet ({tentative.frames} frames): {reason}")
                say_tentative(
                    f"Not placed yet, after {tentative.attempts} of {MAX_ATTEMPTS} attempts "
                    f"over {tentative.frames} held frames: {reason}"
                )
                return True

            def give_up(reason: str) -> None:
                """Record that this resume is over, once. NEVER raises.

                A recovery ends differently from a resume: there is no Pause to
                press twice and no unpaused scan to go back to, so it gets its
                own ending rather than this one with the words changed.
                """
                if recovering:
                    abandon_recovery(reason)
                    return
                tentative.exhausted = True
                tentative.exhausted_reason = reason
                detail = (
                    f"{reason} Nothing has been added to the scan. Move back over a part you "
                    "have already scanned and press Pause twice to try again, or press Stop - "
                    "every frame written before the pause is safe."
                )
                LOG.warning("relocalization gave up: %s", reason)
                worker.progress.emit("\n" + detail)
                say("relocalizing", detail)

            def collect_tentative(buf: bytes) -> None:
                """Hold one frame back and try to place the segment. NEVER raises.

                POST-REQUEUE, always. Everything below - deprojecting half a
                megabyte, an adjacent ICP solve, and the decision to start a
                global search - used to run between the driver handing this
                buffer over and getting it back, which froze the depth view for
                the whole of it and made Stop wait for a search to finish.
                """
                # Before the deprojection, and that is the whole point of the
                # order: an exhausted attempt used to keep deprojecting and
                # locally aligning every selected frame forever, spending the
                # cost of a resume that had already been declared over. Only the
                # operator restarts it, by pausing and resuming again.
                if tentative.exhausted:
                    return
                # Before the deprojection too, and for the reason the check
                # after ``tentative.add`` is not enough: a frame that yields an
                # empty cloud, raises on the way out of the deprojection, throws
                # from the adjacent ICP or is refused by the local chain returns
                # from BELOW this line and above that one. A stream of those is
                # exactly what a lost scanner pointed at nothing delivers, and
                # without this the capture would sit in ``recovering_tracking``
                # for the rest of the run.
                expired = recovery_expired()
                if expired is not None:
                    abandon_recovery(expired)
                    return
                try:
                    cloud = _cloud_from_bytes(buf, PREVIEW_POINTS)
                except Exception as exc:
                    note_failure("relocalization", exc)
                    return
                if not cloud:
                    return

                edge: LocalEdge | None = None
                if tentative.poses:
                    # The SAME tracker, under a policy that names what a
                    # tentative segment does not have: it is chained against
                    # itself, so there is no accepted model and no keyframe set
                    # to refer to, and :data:`~scanner3d.tracking.ADJACENT_ONLY_POLICY`
                    # switches those two references off rather than loosening a
                    # gate. Built per link over exactly the previous held frame,
                    # because the segment's buffer slides and a tracker holding
                    # its own copy of it could disagree with it.
                    link = IncrementalTracker(policy=ADJACENT_ONLY_POLICY, backend=backend)
                    try:
                        link.seed(tentative.clouds[-1], tentative.poses[-1])
                        decision = link.offer(cloud)
                    except Exception as exc:
                        # The tentative segment failed to chain to ITSELF, which
                        # is ordinary adjacent alignment and nothing to do with
                        # the reposition. The frame is dropped from the segment;
                        # it was never on disk and never counted.
                        note_failure("relocalization", exc)
                        report_collecting()
                        return
                    # A returned fit is not a link. It is judged against named
                    # gates in the same vocabulary as everything else here, and
                    # a link that fails them is dropped rather than chained -
                    # otherwise a global match could place a segment whose own
                    # chain nothing had ever looked at.
                    adjacent = next(iter(decision.candidates), None)
                    if adjacent is not None:
                        edge = replace(adjacent.edge, frame=tentative.frames)
                    if not decision.accepted or decision.pose is None:
                        LOG.info(
                            "tentative frame rejected by the local chain: %s",
                            "; ".join(decision.refusals),
                        )
                        say_tentative(
                            f"That frame did not line up with the one before it, so it was not "
                            f"held: {decision.refusals[0]} Still holding {tentative.frames} "
                            "frames, none of them in the scan."
                        )
                        return
                    pose = decision.pose
                else:
                    pose = Rigid.identity()

                refused = tentative.add(buf, cloud, pose, edge)
                if refused:
                    give_up(refused)
                    return
                if worker.cancelled:
                    return
                if tentative.frames < MIN_TENTATIVE_FRAMES:
                    report_collecting()
                    return
                spent = time.monotonic() - tentative.started
                if tentative.attempts >= MAX_ATTEMPTS or spent > MAX_TENTATIVE_SECONDS:
                    give_up(
                        f"Still cannot tell where the object is, after {tentative.attempts} "
                        f"attempts over {spent:.0f}s."
                    )
                    return
                # Only ever with a NEW frame in the buffer, which is what stops
                # this from being a loop that re-runs the same failed decision
                # at the frame rate.
                tentative.attempts += 1
                start_search()
                # An answer that is ALREADY here is applied now rather than at
                # the next frame. There is nothing to wait for in that case, and
                # a segment that could be published one frame earlier should be;
                # a search that is genuinely still running is untouched by this
                # and is picked up when it finishes.
                take_search_result()

            def on_frame(buf: bytes, _record: object) -> None:
                """Save the frame, then keep the model view honest about it.

                Never raises, and specifically never raises
                :class:`Stopped`: this runs between the driver handing a buffer
                over and ``stream_frames`` handing it back, and an exception here
                would skip that ``VIDIOC_QBUF`` and leave userspace owning a
                buffer at teardown - which this hardware answers by
                re-enumerating. Cancellation is read, not thrown.

                The pause state machine sits at the top of it, before anything
                is written, because everything below this is what a pause exists
                to prevent: a paused capture writes no file, moves no count,
                places no pose and draws no model, and a resumed one does none
                of those either until the frames it is holding have been placed
                against the model by evidence.
                """
                nonlocal kept, refreshed_at, mode, handled_epoch, waiting_to_pause, last_aligned
                nonlocal tentative_pending, dropped_after_loss
                tentative_pending = False
                if tracking_lost:
                    # Terminal, and checked before anything else this callback
                    # does. The depth preview is fed from ``on_delivery`` and
                    # the colour view from the pump, neither of which is on this
                    # path, and the stream polls for Stop after every re-queue -
                    # so both stay live and bounded. What stops here is
                    # everything that would extend a scan that has come apart:
                    # no frame is written, no count moves, no pose is placed, no
                    # model or surface is refreshed, and no relocalization is
                    # started. Pause is disabled in the window rather than
                    # answered here, because the machinery a pause needs is
                    # exactly the machinery that has stopped working.
                    #
                    # While a recovery is in play the frame goes down the
                    # TENTATIVE path instead - the same one a resume uses, held
                    # in the same bounded buffer - and the expensive half of
                    # that runs in ``after_frame``, once the driver has its
                    # buffer back. It is still not written, still not counted
                    # and still not in the model; the only thing that can move
                    # it out of memory is a published recovery.
                    if recovering:
                        tentative_pending = True
                        return
                    dropped_after_loss += 1
                    return
                paused_now, epoch = control.state()
                #: A pause that has been asked for and is waiting for a frame it
                #: can be taken at. The frame in hand is committed as usual, and
                #: it must NOT be mistaken for a resume by the epoch check
                #: below - the request has not been answered yet, it has been
                #: deferred, and answering it here would start a relocalization
                #: against a pause that never happened.
                deferring = False

                if paused_now and mode != "paused":
                    if len(clouds) < MIN_PAUSE_REFERENCE_CLOUDS:
                        refuse_pause(
                            epoch,
                            f"only {len(clouds)} frames have aligned so far, under the "
                            f"{MIN_PAUSE_REFERENCE_CLOUDS} needed for the scan to recognise "
                            "the object again after it is moved. Keep scanning, then Pause.",
                        )
                        paused_now = False
                    elif last_aligned != kept - 1:
                        # The bridge is measured against the last aligned pose
                        # and RECORDED against the frame immediately before the
                        # segment, so a pause is only safe where those are the
                        # same frame. Normally the very next frame is one.
                        waiting_to_pause += 1
                        if waiting_to_pause > PAUSE_BOUNDARY_GRACE:
                            refuse_pause(
                                epoch,
                                f"the last {waiting_to_pause} frames would not align, so there "
                                "is no reliable position to pause from. Hold still until the "
                                "model is moving again, then Pause.",
                            )
                            paused_now = False
                        else:
                            deferring = True
                            say(
                                "running",
                                "Pausing at the next frame that aligns - the scan has to know "
                                "where it is before it can be moved.",
                                paused=True,
                            )
                        # Committed anyway, on both branches. The pause is
                        # waiting for a frame that ALIGNS, and dropping frames
                        # while it waits would guarantee there is never another
                        # one - the wait would then always end in the refusal
                        # above, which is the opposite of deferring it.
                        paused_now = False
                    else:
                        enter_paused()
                        return
                if paused_now:
                    # Paused, and staying paused. The depth and colour previews
                    # are fed from ``on_delivery`` and the colour pump, neither
                    # of which is on this path, so the operator can frame the
                    # new position with everything still moving.
                    return

                if not deferring and (
                    mode == "paused" or (mode == "running" and epoch != handled_epoch)
                ):
                    enter_relocalizing(epoch)
                if mode == "relocalizing":
                    # Nothing else here. The frame is not written, not counted
                    # and not drawn, and the work of deciding where it belongs
                    # is handed to ``after_frame`` - which runs once the driver
                    # has this buffer back, so a global search that takes a
                    # second costs the device and the previews nothing.
                    tentative_pending = True
                    return

                path = current_dir / FRAME_TEMPLATE.format(index=kept)
                path.write_bytes(buf)
                kept += 1
                # Before anything that can be skipped, and never skipped itself:
                # the frame IS on disk, so the count must say so.
                worker.saved.emit(kept)

                # Everything below is the live model, and all of it is expensive
                # enough to be worth not doing for a capture that is ending.
                # Checked at each stage rather than once, because a Stop pressed
                # during deprojection should not still pay for the alignment
                # after it.
                if worker.cancelled:
                    return

                # Live coverage is chained alignment only - no loop closure, so
                # it drifts. It answers "have I pointed at that yet", nothing
                # more, and Reconstruct recomputes it honestly afterwards.
                # Both ways this can fail to produce something placeable are
                # tracking failures, not silent skips. Before this, a frame that
                # deprojected to nothing - or one whose deprojection raised, which
                # also escaped a callback documented never to - simply returned
                # here, leaving the model stationary without extending the streak.
                # A capture doing that on every frame is the 2026-08-28 failure
                # with a different cause and no bound at all.
                #
                # Cancellation is checked at each step and is never one of them.
                try:
                    cloud = _load_cloud(path, PREVIEW_POINTS)
                except Exception as exc:
                    note_failure("alignment", exc)
                    note_tracking_failure(
                        kept - 1,
                        TRACKING_INPUT_UNUSABLE_TYPE,
                        "preparing the frame for tracking failed before alignment was "
                        f"attempted: {type(exc).__name__}: {exc}",
                    )
                    return
                # One statement, two different reasons to stop here, and only
                # one of them is a tracking failure. A Stop that landed during
                # the deprojection above ends the frame and is classified as
                # nothing; a frame that yielded no points is the streak.
                if not cloud or worker.cancelled:
                    if not cloud:
                        unusable = (
                            "the frame deprojected to no usable points, so there was nothing "
                            "to align it by and nothing to add to the model"
                        )
                        note_failure("alignment", unusable)
                        # Records nothing if the capture is cancelling: the
                        # guard for that lives in one place, inside it.
                        note_tracking_failure(kept - 1, TRACKING_INPUT_EMPTY_TYPE, unusable)
                    return
                # One decision, three references, and none of the arithmetic
                # here. Where the frame goes is decided by the tracker against
                # the previous accepted frame, a selected keyframe and a bounded
                # sample of the accepted model, and a placement only two of them
                # agree on. A frame the tracker refuses is dropped from the
                # PREVIEW only; the raw frame is already safely on disk.
                #
                # ``offer`` never raises for ordinary data, and a refusal
                # mutates nothing - so a refused frame leaves the tracker, the
                # keyframes and the model exactly as the last accepted frame
                # left them. Exceptions are still caught, because "never raises"
                # is a property of this repository's code and this callback runs
                # between the driver handing a buffer over and getting it back.
                try:
                    decision = tracker.offer(cloud)
                except Exception as exc:
                    note_failure("alignment", exc)
                    note_alignment_failure(kept - 1, exc)
                    return
                if not decision.accepted or decision.pose is None:
                    note_failure("alignment", decision.message())
                    # The cumulative count above says how the capture has been
                    # going. THIS says whether the chain is still connected, and
                    # on the third consecutive failure it ends the capture's
                    # claim to know where the object is.
                    note_tracking_failure(
                        kept - 1,
                        decision.failure_type or TRACK_NO_CANDIDATE_TYPE,
                        decision.message(),
                    )
                    return
                poses.append(decision.pose)
                clouds.append(cloud)
                placements[decision.origin or ""] = placements.get(decision.origin or "", 0) + 1
                if decision.origin != ANCHOR_ORIGIN:
                    # Measured from the placement that was actually accepted, so
                    # it reflects the motion the registration actually sees.
                    worker.travel.emit(decision.travel)
                    say_placement(decision)
                    # The same measurement the readout above shows, handed to
                    # the thing that can act on it. Fed here rather than beside
                    # the write, because a travel is only a travel once a
                    # placement was accepted - the anchor has none, and a
                    # refused frame's is deliberately infinite.
                    note_cadence(decision.travel)

                # One alignment that worked ends the streak, whatever came
                # before it: an isolated failure, or two, is a dropout and not a
                # loss, and a scan that recovers by itself must not be carrying
                # two thirds of a verdict into the rest of the capture.
                publish_alignment_status(decision, kept - 1, after_dropout=bool(streak))
                streak.clear()
                # This frame both reached disk and aligned, so it is a position
                # a pause may be taken from. Set AFTER the alignment, never
                # beside the write: a frame on disk with no pose is not a place
                # a bridge can be measured against.
                last_aligned = kept - 1
                # The first useful model as soon as there is one, then on a
                # fixed alignment interval. ``>=`` rather than equality so a
                # refresh that was skipped or failed is caught up on rather than
                # lost until the next multiple comes round.
                refresh_model()

            def after_frame(buf: bytes, _record: object) -> None:
                """The expensive half of a resume, with the buffer already back.

                NEVER raises, for the same reason ``on_frame`` never does: this
                runs inside the stream's own loop, and an exception here would
                unwind through it. It cannot skip a ``VIDIOC_QBUF`` - that has
                already happened, which is the point of this callback existing -
                but the stream's cleanup guarantees are still the stream's.

                Nothing in here waits for the global search. A search still
                running means this frame is skipped: it is not written, not
                counted and not held, and the depth preview and the Stop poll
                both carry on at the rate the scanner delivers.
                """
                nonlocal tentative_pending
                if not tentative_pending:
                    return
                tentative_pending = False
                if worker.cancelled or not (recovering or mode == "relocalizing"):
                    return
                # BEFORE the result is even looked at. The clock is an
                # acceptance boundary, not a watchdog for a hung search: a
                # search that comes back accepted after the recovery was allowed
                # to end must not place a segment, because by then nobody can
                # say the object is still where that answer says it is. Checking
                # it only when the future is unfinished would let exactly that
                # late acceptance through. An already-done future is abandoned
                # here as readily as a running one - ``abandon_search`` never
                # waits either way - and its answer reaches nobody.
                expired = recovery_expired()
                if expired is not None:
                    abandon_recovery(expired)
                    return
                if not take_search_result():
                    # A search that is still going is left alone rather than
                    # waited for: this frame is simply not written, not counted
                    # and not held, and the deadline above is what ends the
                    # recovery if the answer never comes.
                    say_tentative(
                        f"Still looking for where the object went (attempt "
                        f"{tentative.attempts} of {MAX_ATTEMPTS}, {tentative.frames} frames "
                        "held). Nothing is being written."
                    )
                    return
                # ``take_search_result`` can have ACCEPTED, which puts the
                # capture back into ordinary running - in which case this frame
                # belongs to the resumed scan and not to a segment. It is
                # dropped rather than committed here: committing it would mean
                # writing a frame from inside the tentative path, and the very
                # next delivery takes the ordinary route. The same is true of a
                # recovery that has just been published, and of one that has
                # just run out of attempts.
                if not (recovering or mode == "relocalizing"):
                    return
                collect_tentative(buf)

            try:
                result = stream_with_colour(
                    worker,
                    serial,
                    on_frame,
                    frames=frames,
                    decimate=(lambda: cadence.decimate) if cadence is not None else decimate,
                    duration_s=900.0,
                    after_frame=after_frame,
                    on_delivery=on_delivery,
                )
            finally:
                # ``wait=False``: a Stop must not sit behind a global search
                # that has already started, and it does not have to - the search
                # holds only the tuples it was handed, publishes nothing and
                # touches no device. ``cancel_futures`` stops one that has not
                # begun.
                if searcher is not None:
                    searcher.shutdown(wait=False, cancel_futures=True)
                if tracking_lost:
                    LOG.info(
                        "capture ended with tracking lost after %d saved frames; %d selected "
                        "frame(s) arrived afterwards and were deliberately not written",
                        kept,
                        dropped_after_loss,
                    )
                if abandoned:
                    LOG.info(
                        "%d relocalization search(es) were abandoned mid-flight; none of them "
                        "could reach the scan",
                        abandoned,
                    )
                if cadence is not None:
                    # Said at the end rather than per change, for the same
                    # reason the placement lines are said once: at this
                    # scanner's rate a line per decision is a wall of text. And
                    # said AT ALL, rather than left in the log, because the
                    # frame spacing is the one capture setting the operator
                    # typed in and then did not get - a folder whose frames are
                    # unevenly spaced is a folder they should know about.
                    summary = cadence.record()
                    refusals = sum(
                        1
                        for change in cadence_changes
                        if getattr(change, "reason", None) == REFUSAL_REASON
                    )
                    LOG.info("frame spacing over the capture: %s", summary)
                    if summary["changes"]:
                        line = (
                            f"Frame spacing moved {summary['changes']} time(s) during this "
                            f"capture: you set 1 in {summary['nominal']}, and it ran between "
                            f"1 in {summary['lowest']} and 1 in {summary['highest']}, "
                            f"finishing at 1 in {summary['final']}."
                        )
                        if refusals:
                            line += (
                                f" {refusals} of those were frames the tracker could not place, "
                                "which is the scan telling you it was going too fast."
                            )
                        worker.progress.emit("\n" + line)
                    else:
                        worker.progress.emit(
                            f"\nFrame spacing held at 1 in {summary['nominal']} for the whole "
                            "capture - your speed stayed inside the band that registers well."
                        )
            # Only now, with the stream returned and its finally block done -
            # STREAMOFF, unmap, buffers released, node closed. Raising this any
            # earlier would unwind through the callback and skip the re-queue
            # that keeps the device healthy; raising it here simply routes the
            # finished job to the stopped outcome instead of the completed one.
            #
            # The stream's own result is the authority, and deliberately the
            # ONLY thing consulted. ``worker.cancelled`` is not: a Stop pressed
            # in the moment between a normal return and this line would relabel
            # a capture that ran to its full frame count - or out its duration -
            # as stopped, printing "frames were already written" about a sweep
            # that in fact finished. The stream polls ``should_stop`` after
            # every re-queue, so a stop it could act on is one it reports.
            # Before the Stop check, deliberately. The capture cancelled
            # ITSELF to get here, so the stream reports a caller stop and the
            # operator pressed nothing; routing this to the quiet "stopped"
            # outcome would file the one failure that must never be quiet under
            # the one ending that says nothing at all.
            if loss_failure is not None:
                raise loss_failure
            if result.stopped_early and result.stop_reason == CALLER_STOP_REASON:
                raise Stopped
            return result

        self._run(job, self._capture_done)

    def _capture_done(self, result: object) -> None:
        """A capture that ran to its own end. Never the operator's Stop.

        Stop arrives at :meth:`_was_stopped` instead, which is what makes
        resuming the preview here safe: this is the path where the capture
        finished on its own terms and the live view is the natural next thing to
        see.
        """
        # First: resuming the preview below STARTS a run, and this one has to
        # have ended before it does.
        self._end_depth_run()
        self._busy(False)
        if self._consume_resume_preview():
            self._preview_action.setChecked(True)
        self._finish_progress()
        # A capture can reach its full frame count with tracking already lost -
        # the 2026-08-28 sweep ran on for another 120-odd frames. Ending it with
        # "press Reconstruct" would be the same silent lie in the last sentence
        # of the run, so the loss owns this ending too.
        if self._tracking_loss is not None:
            self._log(_tracking_loss_epilogue(self._capture_dir, "\nThe capture ended."))
            self._replace_scan_statuses(
                ScanStatus(
                    source=SCAN_STATUS_CAPTURE_SOURCE,
                    title="SCAN FAILED — TRACKING LOST",
                    summary="The model is frozen at the last frame that aligned.",
                    action="Start a new scan; this folder cannot be reconstructed.",
                    severity="error",
                    priority=100,
                    detail=self._tracking_loss,
                )
            )
            return
        counts = _frame_counts(self._capture_dir)
        self._log(f"\nCaptured {counts.frames} frames to {self._capture_dir}")
        excluded = _excluded_note(counts)
        if excluded:
            self._log(excluded)
        # Said out loud only when it happened. The stream drops a buffer the
        # device delivered short of a whole frame, which is the right thing to
        # do with it and is also invisible from here: the folder simply holds
        # fewer files than were asked for. Without this line that gap has no
        # explanation, and the obvious guess - that the scan failed - is wrong.
        short = getattr(result, "short_frames", 0)
        if short:
            self._log(
                f"{short} incomplete frame(s) arrived from the scanner and were skipped, "
                "so there are fewer here than you asked for. Every file above is a whole "
                "frame; a partial one is not usable and is never written."
            )
        self._log("Press Reconstruct to register them and get a score you can trust.")
        self._replace_scan_statuses(
            ScanStatus(
                source=SCAN_STATUS_CAPTURE_SOURCE,
                title="SCAN COMPLETE",
                summary=f"{counts.frames} frames are ready to reconstruct.",
                action="Press Reconstruct + export.",
                severity="ok",
                priority=20,
            )
        )

    # -- reconstruct -------------------------------------------------------
    def _start_reconstruct(self) -> None:
        """Register a folder of captured frames, and export what comes out.

        The folder dialog asks for INPUT - the frames a capture already wrote -
        and every word the operator sees says so, because this action both reads
        a folder and creates one and the two are easy to confuse. The one it
        creates is made on the worker, after registration has succeeded, and it
        is named in the report when it is finished; cancelling the dialog writes
        nothing anywhere.
        """
        # This capture first, then the folder they all live in: both beat
        # opening the file dialog wherever the process happens to have started.
        try:
            configured = configured_captures_dir()
        except ConfigFileUnreadable:
            configured = None
        start = str(self._capture_dir or configured or "")
        directory = QFileDialog.getExistingDirectory(self, RECONSTRUCT_DIALOG_CAPTION, start)
        if not directory:
            # Nothing has been created, opened or written at this point, and
            # nothing will be: the output folder is made by the export itself.
            return
        frames_dir = Path(directory)
        backend = self._backend.currentText()
        workers = self._workers.value()
        self._report.clear()
        self._replace_scan_statuses(
            ScanStatus(
                source=SCAN_STATUS_CAPTURE_SOURCE,
                title="RECONSTRUCTING",
                summary=f"Registering the frames in {frames_dir.name}.",
                action="Wait for registration, scoring and export to finish.",
                severity="info",
                priority=30,
                detail=str(frames_dir),
            )
        )
        self._log(f"Reading the captured frames in {frames_dir}")
        self._log(
            "They are the INPUT. Registering them, then writing a registered point cloud, a "
            "triangle mesh as PLY and OBJ, and a record of all three into a NEW folder inside "
            "that one. Nothing already there is changed.\n"
        )
        self._busy(True)

        def job(worker: Worker) -> object:
            from scanner3d.cli import DEPTH_SCALE_METRES
            from scanner3d.pipeline import reconstruct_sweep
            from scanner3d.simulate import FACTORY_INTRINSICS

            # The authoritative reader, not a glob. A capture that was
            # repositioned mid-sweep keeps its later frames in the boundary
            # directories, and it carries a recorded transform for each seam
            # that MUST be used instead of aligning that pair - a glob would
            # return neither.
            sequence = read_capture_sequence(frames_dir)
            paths = sequence.paths
            boundaries = sequence.boundaries()
            if boundaries:
                worker.progress.emit(
                    f"  {len(boundaries)} recorded relocalization bridge(s) in this capture; "
                    "those pairs are placed from their records rather than aligned."
                )
            # The raw bytes are kept beside the thinned cloud, not re-read
            # later: the export fuses its final surface from the FULL-resolution
            # frames and registers from the thinned ones, and reading each file
            # twice would leave a window in which the two halves of one export
            # could come from different bytes.
            clouds = []
            fusion_frames: list[FusionFrame] = []
            prepared: list[PreparedCloud] = []
            for index, path in enumerate(paths):
                worker.check()
                raw = path.read_bytes()
                frame = _prepare_frame(raw, RECONSTRUCT_POINTS)
                prepared.append(frame)
                clouds.append(frame.points)
                # The SAME plane, applied a second time to the frame that will
                # be FUSED. Registration and fusion read this capture through
                # two different paths - a decimated cloud and the full-
                # resolution samples - and a support surface removed from only
                # the first of them stays in the exported mesh, which is the
                # artifact the operator opens. Derived in memory: the file this
                # was read from is not written to, and a frame the filter
                # withdrew on is fused byte for byte as captured.
                fusion_frames.append(
                    mask_support_band(
                        raw,
                        frame.support,
                        intrinsics=FACTORY_INTRINSICS,
                        depth_scale=DEPTH_SCALE_METRES,
                    )
                )
                if index % PREPARATION_PROGRESS_EVERY == 0:
                    worker.progress.emit(f"  read {index}/{len(paths)} input frames")
                    # Bounded to the same cadence as the read line. A frame whose
                    # support plane was NOT removed is the interesting one - it
                    # still contains whatever the object was standing on - so the
                    # running tally names withdrawals rather than only successes.
                    worker.progress.emit(f"    {PreparationSummary.of(prepared).summary_line()}")
            summary = PreparationSummary.of(prepared)
            fusion = FusionMaskSummary.of(fusion_frames)
            raw_frames = [frame.raw for frame in fusion_frames]
            worker.progress.emit(f"  {len(clouds)} input frames loaded")
            worker.progress.emit(f"  {summary.summary_line()}")
            worker.progress.emit(f"  {fusion.summary_line()}")

            def report_progress(message: str) -> None:
                worker.check()
                worker.progress.emit(message)

            result = reconstruct_sweep(
                clouds,
                backend=backend,
                workers=workers,
                bridges=boundaries,
                on_progress=report_progress,
            )
            report_progress(
                "exporting the registered cloud from the filtered points, and a mesh fused "
                f"from all {len(raw_frames)} full-resolution frames, "
                f"{fusion.masked_frames} of them with a proved support band masked out"
            )
            # Every frame and every point that was registered, into a folder of
            # its own. This happens BEFORE the model reaches the screen: a
            # picture of a reconstruction that failed to export is a picture of
            # something the operator cannot open anywhere else, and it would be
            # the only evidence they were given that the run succeeded.
            try:
                export = export_reconstruction(
                    frames_dir,
                    clouds,
                    list(result.poses),
                    raw_frames=raw_frames,
                    intrinsics=FACTORY_INTRINSICS,
                    registration=result,
                    unit=_depth_unit(),
                    depth_scale=DEPTH_SCALE_METRES,
                    calibration_source=CALIBRATION_SOURCE,
                    points_per_frame_target=RECONSTRUCT_POINTS,
                    per_frame_filter=filter_statement(summary, fusion),
                    preparation=summary.as_dict(fusion),
                    voxel_budget=RECONSTRUCT_VOXEL_BUDGET,
                    on_progress=report_progress,
                )
            except ReconstructionRefused as refusal:
                # Every failed gate, in the report, in the same words the
                # command line prints - and BEFORE the exception carries the
                # last of them into the failure dialog. A refusal is the most
                # useful thing this action can tell an operator when it is true,
                # and a one-line dialog is not room for four reasons and four
                # pieces of advice.
                #
                # Then re-raised, unchanged. Nothing was exported and nothing
                # may be drawn: the model panel is refreshed further down, and a
                # picture of a reconstruction that was refused is exactly the
                # plausible-looking result this gate exists to prevent.
                for line in refusal.readiness.refusal_text().splitlines():
                    worker.progress.emit(line)
                raise
            # Every frame, every point: this is the one the operator is being
            # asked to trust, it is built once at the end rather than a dozen
            # times a minute, and nothing about the live preview's bounds
            # applies to it.
            #
            # Through the SAME upright presentation frame the files were written
            # in, composed with each accepted pose by the one function that owns
            # that composition - so what is on the screen and what is in the
            # exported PLY are the same arithmetic rather than two versions of
            # it that could disagree about which way up the object is.
            points, thin, well_covered, _newest = _coloured_cloud(
                clouds, [upright_pose(pose) for pose in result.poses]
            )
            surface_vertices, surface_faces = _surface_arrays(export.surface)
            worker.coverage.emit(
                ModelSnapshot(
                    points=points,
                    thin=thin,
                    # Deliberately discarded, and deliberately not the flags the
                    # builder just computed. "Newest" means "this arrived while
                    # you were watching"; a reconstruction places every frame at
                    # once, hours after the sweep if that is when it was run, and
                    # highlighting its last file would be an animation cue
                    # attached to nothing that happened.
                    newest=np.zeros(len(points), dtype=bool),
                    well_covered=well_covered,
                    aligned=len(clouds),
                    saved=len(clouds),
                    sampled=len(clouds),
                    note="registered -",
                    # A different object from whatever was on screen: every
                    # frame, every point, and poses the optimiser moved. This is
                    # the other boundary that is entitled to frame the view.
                    reframe=True,
                    # The surface that was just WRITTEN, not a second one
                    # reconstructed for the screen. The export kept it so this
                    # panel and the two mesh files in the folder cannot be
                    # different surfaces, and this one is built from the
                    # corrected poses - which the live preview's is not.
                    surface_vertices=surface_vertices,
                    surface_faces=surface_faces,
                    surface_voxel=export.resolution.voxel_size,
                    # And said so, in the type the panel reads rather than in a
                    # comment: this is the fused signed-distance surface, so the
                    # sentence beside it, the coverage clause under it and the
                    # control that draws it all describe THAT rather than the
                    # occupancy preview they would otherwise have described.
                    surface_kind=SurfaceKind.RECONSTRUCTED_TSDF,
                )
            )
            return ReconstructionOutcome(registration=result, export=export)

        self._run(job, self._reconstruct_done)

    def _reconstruct_done(self, result: object) -> None:
        """Say where the scan went, then what the evidence says about it.

        The folder first, and before the score, because "where is my scan" is
        the question this action was pressed to answer and a report that opens
        with three angles in degrees buries it.
        """
        self._busy(False)
        export = getattr(result, "export", None)
        # Tolerated rather than required, so that a caller holding a bare
        # registration - which is what this method took before there was
        # anything to export - still gets its score reported.
        registration = getattr(result, "registration", result)
        if isinstance(export, ReconstructionExport):
            self._log(f"\nEXPORTED to {export.directory}")
            for path in export.paths:
                self._log(f"  {path.name}")
            closed = "measured closed" if export.watertight else "measured NOT closed"
            # Both halves, and which points each one came from. They are built
            # from different inputs - the cloud from the thinned and filtered
            # points, the surface from every full-resolution frame - and a line
            # that said "a mesh" left an operator to assume the file they can
            # measure was fitted to the file they can see. "Full-resolution"
            # rather than "raw": a frame whose support plane was proved is
            # fused with that band masked out, which the progress above and the
            # exported record both say in full.
            self._log(
                f"\n{export.point_count} registered points (thinned and filtered), and a TSDF "
                f"surface of {export.vertex_count} vertices and {export.face_count} faces "
                f"({closed}) fused from {export.integration.frames_offered} full-resolution "
                "frames "
                f"at {export.resolution.summary}, truncated at "
                f"{export.resolution.truncation!r}."
            )
            self._log(
                "The mesh and the cloud are in the same right-handed upright frame, so they "
                "open the right way up; reconstruction.json states the axis mapping back to "
                "the camera's own frame, every file's SHA-256, and what none of it establishes."
            )
            self.statusBar().showMessage(f"exported: {export.summary}", 15000)
        optimised = getattr(registration, "optimised_score", None)
        chained = getattr(registration, "chained_score", None)
        self._log("")
        self._log(f"frames {getattr(registration, 'frames', '?')}")
        self._log(f"revisited viewpoints found: {getattr(registration, 'revisits_found', 0)}")
        # Said where the score is said, because it is part of reading it: those
        # seams were not measured by this run, and the limitations below name
        # what they were measured by instead.
        bridges = getattr(registration, "bridges", ())
        if bridges:
            self._log(
                f"recorded relocalization bridges used: {', '.join(bridges)} - those pairs "
                "were placed from the evidence saved during the capture, not aligned here."
            )
        if optimised is None or chained is None:
            self._log("\nNOT SCORED, and not corrected.")
            self._log(
                "Too few revisited viewpoints. Without somewhere the scanner returned to, a "
                "drifting scan cannot be caught, so no number here would mean anything. "
                "Capture a path that comes back to where it started."
            )
        else:
            self._log("\nHELD-OUT CLOSURE ERROR (scored against constraints never fitted):")
            self._log(
                f"  before  median {math.degrees(chained.median_error_radians):7.3f} deg"
                f"   worst {math.degrees(chained.worst_error_radians):7.3f}"
            )
            self._log(
                f"  after   median {math.degrees(optimised.median_error_radians):7.3f} deg"
                f"   worst {math.degrees(optimised.worst_error_radians):7.3f}"
            )
            self._log(
                f"  {optimised.used_for_fitting} constraints fitted, "
                f"{optimised.held_out} withheld and used only to score"
            )
        for limitation in getattr(registration, "limitations", ()):
            self._log(f"\n  - {limitation}")
        if isinstance(export, ReconstructionExport):
            self._replace_scan_statuses(
                ScanStatus(
                    source=SCAN_STATUS_CAPTURE_SOURCE,
                    title="RECONSTRUCTION COMPLETE",
                    summary=f"Point cloud and mesh exported to {export.directory}.",
                    action="Review the measured points first; the mesh is derived geometry.",
                    severity="ok",
                    priority=20,
                    detail=export.summary,
                )
            )
        else:
            self._replace_scan_statuses(
                ScanStatus(
                    source=SCAN_STATUS_CAPTURE_SOURCE,
                    title="REGISTRATION COMPLETE",
                    summary="The capture finished registering.",
                    action="Open technical details for its score and limitations.",
                    severity="ok",
                    priority=20,
                )
            )

    def closeEvent(self, event: object) -> None:
        """Stop every worker before the window goes away.

        Qt destroys a running QThread noisily and the process can die on the
        way out - which, for someone who has just finished a capture, looks
        like the application crashing rather than closing. The waits here are
        bounded: a worker that will not stop must not hold the window open
        forever either.
        """
        LOG.info("window closing, stopping workers")
        for worker in (self._preview_worker, self._worker, self._previous_stream):
            if worker is not None and worker.isRunning():
                worker.cancel()
        for worker in (self._preview_worker, self._worker, self._previous_stream):
            if worker is not None and worker.isRunning():
                worker.wait(4000)
        LOG.info("workers stopped")
        # The window is going: whatever run was on screen ends with it, and a
        # frame still queued from it must not reach a torn-down panel.
        self._end_depth_run()
        super().closeEvent(event)  # type: ignore[arg-type]


def main(argv: list[str] | None = None) -> int:
    path = configure_logging()
    LOG.info("scanner3d-gui starting; log at %s", path)
    app = QApplication(argv if argv is not None else sys.argv)
    window = ScannerWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
