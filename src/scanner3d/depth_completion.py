"""Confidence-aware spatial completion of raw no-return depth, inside one frame.

What this is
------------
A depth frame from the scanner is full of holes. Some are speckle - one sample
the sensor did not return, surrounded on every side by samples it did. Some are
the shadow of a depth edge, a silhouette, or a gap between two objects, where
there is nothing behind the hole to recover and the honest answer is that the
frame does not know.

This module recovers the first kind and refuses the second. It takes one
rectangle of raw uint16 depth codes, and returns the same rectangle with a
provenance state, a finite confidence and a named list of supporting *original
measured* samples for every position in it. Nothing else about the frame
changes: the raw codes it was handed come back byte-identical under
:data:`STATE_MEASURED`, and a hole this module refuses comes back as raw code
zero, confidence zero and :data:`STATE_MISSING`.

What it is not
--------------
* It is not a smoother, a denoiser or a preview filter. A returned sample is
  never edited, averaged or replaced - see :func:`complete_patch`.
* It is not temporal. Every reconstruction here names support inside the very
  same frame; recovering what *another view* saw is a different problem with a
  different failure mode, and it is deliberately not attempted here.
* It is not wired into capture, the live preview, the tracker, the TSDF or the
  mesh. Nothing in :mod:`scanner3d` calls this module yet. It exists as a
  primitive those stages can adopt once each of them has decided what a
  reconstructed sample means to it - which is a per-stage decision, because a
  reconstruction is evidence of a different kind from a measurement and
  averaging the two without saying so is how a scan quietly stops being one.

Zero is not a depth
-------------------
:data:`scanner3d.depth_metric.NO_RETURN_SAMPLE` is the whole reason this module
can exist: a raw code of zero is *no measurement*, not a point at the origin.
So a hole is identifiable, a fill can only ever occupy one, and a sample this
module reconstructs can always be told apart from one the sensor returned. The
one invariant every other rule here is built on is that the two never merge.

The method, and why each bound is there
---------------------------------------
Deterministic, bounded, single-pass, and edge-aware by construction:

1. **Holes are grouped, not just counted.** Four-connected components of raw
   zeros. A component larger than :attr:`CompletionPolicy.max_component_samples`
   or longer than :attr:`CompletionPolicy.max_component_span` in either
   direction is refused whole. That is what keeps a silhouette's background, a
   band of dropout hugging a depth step, a column collapse and the gap between
   two objects out: in every one of those the "surface" a local method would
   invent spans further than any local evidence reaches.

2. **Support is original measurement only.** The support of a reconstruction is
   drawn from the codes this module was *given*, never from a value it
   produced, so no reconstruction is ever evidence for another one and a fill
   cannot creep across a frame one sample at a time. Support is the nearest
   :attr:`CompletionPolicy.max_support` measured samples inside a disc of
   :attr:`CompletionPolicy.radius`, ordered by distance and then by position so
   the choice is total and stable.

3. **The support has to surround the sample.** At least
   :attr:`CompletionPolicy.min_support` of them, with measured samples on both
   sides along both axes, and a second-moment spread of at least
   :attr:`CompletionPolicy.min_spread` in its weaker direction. Interpolation
   inside evidence, never extrapolation past its edge, and never along a single
   line of samples that fixes no surface at all.

4. **Incompatible depth is detected before it is averaged.** The support's
   codes are sorted; a gap between consecutive values wider than
   :attr:`CompletionPolicy.cluster_gap_codes` means the neighbourhood contains
   two surfaces rather than one, and the sample is refused rather than placed
   in the empty band between them. A total spread wider than
   :attr:`CompletionPolicy.max_support_span_codes` is refused for the same
   reason at a coarser scale.

5. **The fit has to actually fit.** A least-squares plane in ``(column, row,
   code)`` over the support, refused if its worst residual exceeds
   :attr:`CompletionPolicy.max_residual_codes`. On a flat surface this is
   near-exact; on a strongly curved one, or across a boundary the gap test did
   not already catch, it is not, and the sample stays missing. The prediction
   must then itself land inside the support's own code range widened by
   :attr:`CompletionPolicy.max_overshoot_codes`, and inside
   ``[min_code, max_code]``.

6. **Work is bounded by construction, and then by a ceiling.** One pass, no
   iteration to convergence, no queue that can grow. Every hole is examined at
   most once and costs at most ``(2 * radius + 1) ** 2`` neighbourhood visits,
   so :attr:`CompletionWork.work_budget` is a number derivable from the input
   before the run and :attr:`CompletionWork.work_units` is what was actually
   spent. Neither is a wall-clock assertion, which would measure the machine.

   A policy's own bounds are not enough, because a finite Python integer is
   not a cap: ``CompletionPolicy(radius=10**9)`` states a perfectly finite
   radius and publishes a neighbourhood of four quintillion samples. So every
   bound has an absolute product ceiling - :data:`MAX_RADIUS`,
   :data:`MAX_SUPPORT`, :data:`MAX_COMPONENT_SAMPLES`,
   :data:`MAX_COMPONENT_SPAN`, :data:`MAX_SPREAD`, :data:`MAX_PATCH_DIMENSION`
   and :data:`MAX_PATCH_SAMPLES` - and :func:`complete_patch` derives the whole
   call's worst case before it scans anything, refusing it with
   :class:`WorkTooLarge` if it exceeds :data:`MAX_WORK_UNITS`. A whole 640x400
   frame in which every sample is a hole is inside that ceiling; nothing much
   larger is.

Provenance is checked against the frame, not against the labels
---------------------------------------------------------------
:class:`CompletedPatch` holds the immutable :class:`DepthPatch` it was derived
from, and every claim it makes is validated against *that* rather than against
its own per-sample states. The difference matters: a result that calls a sample
``"measured"`` has said nothing about whether the sensor returned it, and
accepting a support index because the result labels that index measured is
letting a claim vouch for its own evidence. With the source in hand there is
exactly one shape a valid result can have, and it is spelled out in
:class:`CompletedPatch`.

And the source is not only checked for *shape*. A result whose every label,
code and support index is legal can still carry a reconstructed code of 65000
where its own evidence gives 100, a confidence of 0.99 where the formula gives
0.31, a legal-but-different support selection, a work record that spent
nothing, or a rejection taxonomy that moved every refusal onto a reason that
never fired. So :class:`CompletedPatch` re-derives the whole run from the
immutable source and the policy it publishes, and requires the result to *be*
that derivation: every reconstructed code, every confidence, every support
selection, every hole that stayed missing, the exact ``work_units`` and the
exact count of each rejection reason. What a result publishes is then a
consequence of two things it cannot change rather than a claim it makes about
itself.

The one thing that binding does not establish is that the published policy is
the policy the caller passed: a policy edit that changes nothing this frame is
sensitive to leaves an identical derivation and is accepted. Every published
value still reproduces exactly under the policy the result carries, which is
what a reader of that result needs; "these were the bounds someone typed" is
not a fact a single result contains, and this module does not claim it.

The same rule applies to mutability. These types are advertised as immutable,
so they copy every sequence and mapping they are handed at construction: a list
the caller still holds is not this module's own, and a value that can change
after validation was never validated. And to arithmetic: ``True == 1`` and
``1.0 == 1``, so every number here whose identity is load-bearing - a schema
version, the dimensions of a result - has its *type* checked before its value,
because an equality test that runs first accepts a boolean width for being
numerically right.

Confidence
----------
:attr:`CompletedPatch.confidence` is a fraction of 1 derived from named
evidence, and it is **not** a camera confidence: it says nothing about the
sensor and never turns a reconstruction into a measurement. It is the geometric
mean of five terms, each in ``[0, 1]`` and each *monotone in the direction its
name implies* - see :func:`support_confidence` for the exact formula:

* more supporting samples raises it;
* wider directional spread of that support raises it;
* support further away from the sample lowers it;
* a worse worst-residual on the fit lowers it;
* wider disagreement among the supporting codes lowers it.

A measured sample carries confidence ``1.0`` and a missing one ``0.0``, so the
three states are separable by state and never by threshold alone. Thresholding
above :attr:`CompletionPolicy.min_confidence` is applied by this module itself,
so a caller that ignores confidence still gets the conservative answer.

Backends
--------
The dependency-free ``"python"`` reference is the only implementation, and it
is the authority. ``"auto"`` selects it. Naming any other backend is an error
rather than a silent downgrade, on the same rule as :mod:`scanner3d.compute`:
a caller who selected a backend and got a different one cannot interpret their
own result. No vectorised backend is written here - adding one would need a
parity test against this reference, and the reference is what the benchmark
scores.

Units
-----
Everything in this module is in **raw depth codes** and in **samples**. It
never multiplies by a depth scale, never deprojects and never sees intrinsics:
turning a code into a length is :mod:`scanner3d.depth_metric`'s job and stating
what the length means is the caller's. That keeps this module honest about what
it is - integer image-space reasoning about which holes are supported - and
keeps every physical claim somewhere that has the provenance to make one.
"""

from __future__ import annotations

import functools
import itertools
import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

from scanner3d.compute import ComputeBackendUnavailable, ComputeBackendUnknown
from scanner3d.depth_metric import NO_RETURN_SAMPLE
from scanner3d.errors import ScannerError

#: The schema of every public type in this module. A caller that persists a
#: result records this beside it; a reader that does not understand it refuses
#: rather than guessing which fields it is looking at.
SCHEMA_VERSION: Final = 1

#: The highest raw code a uint16 depth sample can carry.
MAX_CODE: Final = 0xFFFF

# --- the absolute ceilings -----------------------------------------------------------
#
# Every bound below this line is a *product* maximum, not a default. A policy is
# free to be more conservative than any of them and none of them is reachable by
# the shipped defaults; what they exist for is that a finite Python integer is
# not a work cap. ``CompletionPolicy(radius=10**9)`` states a perfectly finite
# bound and publishes a neighbourhood of 4,000,000,004,000,000,001 samples, and
# a call that entered a neighbourhood loop at that radius would not be a bounded
# product operation whatever its type annotations said. So the ceilings are
# checked on the policy, on the patch, and once more on the run - each before
# any loop that could spend the work.

#: The furthest a reconstruction may ever look for support. Thirty-two samples
#: is already four times the widest default disc and gives a 65x65
#: neighbourhood; support that far from a hole is not local evidence about it
#: on any sensor this pipeline targets.
MAX_RADIUS: Final = 32

#: The most supporting samples one reconstruction may ever rest on. A 65x65
#: disc holds about 3,200 samples; 256 is the most a least-squares plane in
#: three unknowns can be said to be *fitted* to rather than averaged over.
MAX_SUPPORT: Final = 256

#: The largest hole component a policy may ever attempt, and the longest it may
#: ever be in either direction. Both are far past the defaults and both are
#: bounded by the patch ceiling below in any case; they are stated so that a
#: policy cannot declare the whole frame one recoverable component.
MAX_COMPONENT_SAMPLES: Final = 4096
MAX_COMPONENT_SPAN: Final = 512

#: The largest second-moment spread a policy may require, in samples squared.
#: The most any support inside :data:`MAX_RADIUS` can have.
MAX_SPREAD: Final = float(MAX_RADIUS * MAX_RADIUS)

#: The largest patch this module will accept, in samples and in either
#: direction. A 1024x1024 patch is four times the 640x400 depth frame the
#: pipeline actually reads, so a whole frame is comfortably inside it and a
#: dimension that is really a corrupted header is not.
MAX_PATCH_DIMENSION: Final = 4096
MAX_PATCH_SAMPLES: Final = 1024 * 1024

#: The most neighbourhood visits one call may be *permitted* to spend, derived
#: from its input before it runs. A whole 640x400 frame in which every single
#: sample is a hole costs 256,000 x 81 = 20,736,000 visits at the default
#: radius; this ceiling is set just above that, so the worst case the pipeline
#: can actually present is admitted and nothing larger is. A call whose derived
#: worst case exceeds it is refused before the first neighbourhood is scanned,
#: rather than discovered to be unbounded while it runs.
MAX_WORK_UNITS: Final = 25_000_000

#: What this module says about one sample. Stored compactly as one byte each -
#: see :attr:`CompletedPatch.state_codes` - and read back through
#: :meth:`CompletedPatch.state_at`.
#:
#: There is no temporal state here on purpose. This module reconstructs only
#: from the frame it was given, so a state it cannot produce is a state it does
#: not define; the benchmark that scores it carries the wider vocabulary.
STATE_MISSING: Final = 0
STATE_MEASURED: Final = 1
STATE_SPATIALLY_RECONSTRUCTED: Final = 2
STATE_NAMES: Final[tuple[str, ...]] = (
    "missing",
    "measured",
    "spatially_reconstructed",
)

#: Every reason a hole can be left missing, in the order they are tested. The
#: vocabulary is closed and every reason is always reported, at zero when it
#: never fired, so a run's conservatism is a stable shape rather than whichever
#: keys happened to be non-empty.
REJECTION_REASONS: Final[tuple[str, ...]] = (
    "component_too_large",
    "component_too_long",
    "insufficient_support",
    "one_sided_support",
    "collinear_support",
    "incompatible_clusters",
    "support_span_too_wide",
    "degenerate_fit",
    "residual_too_large",
    "prediction_out_of_range",
    "confidence_below_threshold",
)

#: The reference backend, and the only one implemented here.
PYTHON_BACKEND: Final = "python"
AUTO_BACKEND: Final = "auto"

#: Backends this module recognises the name of and does not provide. Named
#: rather than lumped in with a typo, so refusing one says why.
_UNIMPLEMENTED_BACKENDS: Final[tuple[str, ...]] = ("numpy", "scipy", "gpu")


class CompletionNotAllowed(ScannerError):
    """A completion that was refused before it ran, and the reason."""


class PatchUnusable(CompletionNotAllowed):
    """A depth patch whose dimensions, byte count or codes are not a patch."""


class PolicyUnusable(CompletionNotAllowed):
    """A completion policy that states a bound nothing could be held to."""


class ResultUnusable(CompletionNotAllowed):
    """A completion result whose own account of itself does not hold."""


class WorkTooLarge(CompletionNotAllowed):
    """A call whose worst-case work, derived before it runs, is past the ceiling."""


# --- the input: a fixed-denominator rectangle of raw codes ----------------------------


def _frozen_sequence(
    value: object, what: str, error: type[CompletionNotAllowed]
) -> tuple[object, ...]:
    """A caller-owned sequence, copied into a tuple this module owns.

    Declared over ``object`` on purpose: a frozen dataclass field annotated
    ``tuple[int, ...]`` is a promise the type checker keeps and a caller can
    still break, and the whole point of this function is the case where the
    promise was broken.
    """
    if isinstance(value, tuple):
        return value
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise error(what)
    return tuple(value)


def _frozen_bytes(value: object, what: str) -> bytes:
    """One byte per sample, copied out of whatever buffer the caller still holds."""
    if isinstance(value, bytes):
        return value
    if not isinstance(value, bytearray | memoryview):
        raise ResultUnusable(what)
    return bytes(value)


@dataclass(frozen=True, slots=True)
class DepthPatch:
    """One rectangle of raw uint16 depth codes, immutable and fully populated.

    A rectangle rather than a whole frame because completion is a local
    operation and a caller may want only part of a frame - but a *fixed* one:
    ``len(codes)`` is always ``width * height``, so every count this module
    reports has a denominator that the caller chose in advance and the
    completion cannot move. A result that "improved" by looking at fewer
    samples would be arithmetic about nothing.
    """

    width: int
    height: int
    codes: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.width, bool) or isinstance(self.height, bool):
            raise PatchUnusable(f"a patch is {self.width!r}x{self.height!r}, which is not a size")
        if not isinstance(self.width, int) or not isinstance(self.height, int):
            raise PatchUnusable(f"a patch is {self.width!r}x{self.height!r}, which is not a size")
        if self.width <= 0 or self.height <= 0:
            raise PatchUnusable(f"a patch of {self.width}x{self.height} spans nothing")
        if self.width > MAX_PATCH_DIMENSION or self.height > MAX_PATCH_DIMENSION:
            raise PatchUnusable(
                f"a patch of {self.width}x{self.height} is past the {MAX_PATCH_DIMENSION}-sample "
                "ceiling this module states in either direction"
            )
        if self.width * self.height > MAX_PATCH_SAMPLES:
            raise PatchUnusable(
                f"a patch of {self.width}x{self.height} holds {self.width * self.height} "
                f"sample(s), past the ceiling of {MAX_PATCH_SAMPLES} this module states"
            )
        # The one place a caller-owned sequence becomes this module's own. A
        # frozen dataclass freezes the *binding*, not a list somebody else still
        # holds, so the copy has to happen here or "immutable" is a word rather
        # than a property. Validation runs on the copy for the same reason.
        object.__setattr__(
            self,
            "codes",
            _frozen_sequence(
                self.codes,
                f"a patch carries {self.codes!r}, which is not a sequence of codes",
                PatchUnusable,
            ),
        )
        if len(self.codes) != self.width * self.height:
            raise PatchUnusable(
                f"a {self.width}x{self.height} patch carries {len(self.codes)} code(s) and not "
                f"{self.width * self.height}; the denominator is the rectangle"
            )
        for index, code in enumerate(self.codes):
            if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= MAX_CODE:
                raise PatchUnusable(
                    f"a patch carries {code!r} at index {index}, which is not a uint16 depth code"
                )

    @classmethod
    def of(cls, width: int, height: int, codes: Sequence[int]) -> DepthPatch:
        """A patch from any sequence of codes, copied so it cannot change under us."""
        return cls(width=width, height=height, codes=tuple(codes))

    @classmethod
    def from_frame(
        cls,
        raw: bytes,
        *,
        frame_width: int,
        frame_height: int,
        u0: int = 0,
        v0: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> DepthPatch:
        """One rectangle read out of a whole little-endian uint16 depth frame.

        The frame's own dimensions are named by the caller rather than assumed:
        this module has no opinion about which sensor produced the bytes, and a
        frame whose byte count does not match what it was told is refused
        instead of being read at an offset that means nothing.
        """
        if frame_width <= 0 or frame_height <= 0:
            raise PatchUnusable(f"a frame of {frame_width}x{frame_height} spans nothing")
        if frame_width > MAX_PATCH_DIMENSION or frame_height > MAX_PATCH_DIMENSION:
            raise PatchUnusable(
                f"a frame of {frame_width}x{frame_height} is past the {MAX_PATCH_DIMENSION}-sample "
                "ceiling this module states in either direction; a dimension that large is a "
                "corrupted header rather than a depth frame, and it is refused before the bytes "
                "behind it are indexed"
            )
        expected = frame_width * frame_height * 2
        if len(raw) != expected:
            raise PatchUnusable(
                f"a {frame_width}x{frame_height} frame is {expected} byte(s) and this one is "
                f"{len(raw)}"
            )
        span_width = frame_width if width is None else width
        span_height = frame_height if height is None else height
        if span_width <= 0 or span_height <= 0:
            raise PatchUnusable(f"a patch of {span_width}x{span_height} spans nothing")
        if u0 < 0 or v0 < 0:
            raise PatchUnusable(f"a patch starts outside the frame at ({u0}, {v0})")
        if u0 + span_width > frame_width or v0 + span_height > frame_height:
            raise PatchUnusable(
                f"a {span_width}x{span_height} patch at ({u0}, {v0}) leaves the "
                f"{frame_width}x{frame_height} frame"
            )
        codes: list[int] = []
        for row in range(span_height):
            offset = ((v0 + row) * frame_width + u0) * 2
            codes.extend(struct.unpack_from(f"<{span_width}H", raw, offset))
        return cls(width=span_width, height=span_height, codes=tuple(codes))

    @property
    def sample_count(self) -> int:
        return self.width * self.height

    def position_of(self, index: int) -> tuple[int, int]:
        """The ``(column, row)`` of one index, within this patch."""
        row, column = divmod(index, self.width)
        return column, row

    def index_of(self, column: int, row: int) -> int | None:
        """The index of one position, or ``None`` when it is outside the patch."""
        if not (0 <= column < self.width and 0 <= row < self.height):
            return None
        return row * self.width + column

    def is_measured(self, index: int) -> bool:
        """Whether the sensor returned anything at this sample. Zero never did."""
        return self.codes[index] != NO_RETURN_SAMPLE

    def measured_indices(self) -> tuple[int, ...]:
        return tuple(index for index in range(self.sample_count) if self.is_measured(index))

    def hole_indices(self) -> tuple[int, ...]:
        return tuple(index for index in range(self.sample_count) if not self.is_measured(index))


# --- the policy: every bound, stated where a reviewer can read it ---------------------


def _positive_int(value: object, name: str, maximum: int) -> int:
    """A count, inside the product ceiling stated for it.

    ``maximum`` is not advisory and not a default: a bound past it is refused
    here rather than reached later, because the only thing finiteness buys is
    that the number can be printed.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyUnusable(f"a completion policy states {name} {value!r}, which is not a count")
    if value > maximum:
        raise PolicyUnusable(
            f"a completion policy states {name} {value!r}, past the ceiling of {maximum} this "
            "module states. A finite integer is not a bound on work; this is"
        )
    return value


def _positive_number(value: object, name: str, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PolicyUnusable(f"a completion policy states {name} {value!r}, which is not a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise PolicyUnusable(
            f"a completion policy states {name} {value!r}; every bound here is finite and "
            "positive, and an unbounded one is not a bound"
        )
    if number > maximum:
        raise PolicyUnusable(
            f"a completion policy states {name} {value!r}, past the ceiling of {maximum} this "
            "module states"
        )
    return number


@dataclass(frozen=True, slots=True)
class CompletionPolicy:
    """Every bound :func:`complete_patch` is held to, in codes and in samples.

    Immutable, validated on construction, and published in the result: an
    accepted sample is reproducible from its named support and these numbers
    alone, which is what makes "it recovered that hole" checkable rather than
    assertable. Every field is a *conservatism* knob - tightening any of them
    refuses more and invents less - and each default is justified in its own
    comment.

    One of them is not a free choice and is called out here rather than left
    in a comment: :attr:`max_residual_codes` has to be set from the sensor's
    own noise floor. The default of two raw codes suits a quantisation-limited
    frame, where a plane fit on flat geometry misses its own support by well
    under one code and a fit that misses by two is describing something that is
    not a local plane. A noisier sensor needs a larger value, and pays for it
    by admitting curvature that a plane cannot represent. It is the one bound
    a caller must revisit before using this on real frames.
    """

    #: How far, in samples, a reconstruction may look for support. Four gives
    #: a 9x9 disc: enough that a sample missing its whole 3x3 neighbourhood
    #: still has evidence, and small enough that the surface across it is a
    #: local question.
    radius: int = 4
    #: The fewest supporting samples a reconstruction may rest on. Eight is a
    #: full immediate ring; fewer than that has not established a surface.
    min_support: int = 8
    #: The most support one reconstruction uses. A cap, not a target: it keeps
    #: the fit local when the neighbourhood is dense, and it bounds the work.
    max_support: int = 16
    #: The smallest second-moment spread, in samples squared, the support may
    #: have in its weaker direction. One sample: support strung along a single
    #: line fixes no surface, however much of it there is.
    min_spread: float = 1.0
    #: The largest gap between consecutive sorted support codes before the
    #: neighbourhood is called two surfaces rather than one.
    cluster_gap_codes: float = 64.0
    #: The largest total spread of support codes. A neighbourhood wider than
    #: this is refused even when it has no single wide gap in it.
    max_support_span_codes: float = 384.0
    #: The worst residual the least-squares plane may leave on its own support.
    #: See the class docstring: this is the sensor-dependent bound, and two raw
    #: codes is the quantisation-limited setting.
    max_residual_codes: float = 2.0
    #: How far outside the support's own code range a prediction may land.
    max_overshoot_codes: float = 32.0
    #: The most samples one refusable hole component may contain, and the
    #: longest it may be in either direction. A component past either bound is
    #: refused whole: that is the silhouette, the edge band, the collapsed
    #: column and the gap between two objects, none of which one frame can
    #: settle.
    max_component_samples: int = 48
    max_component_span: int = 10
    #: The lowest confidence a reconstruction may be published with. Applied
    #: here so a caller that ignores confidence still gets the conservative
    #: answer.
    min_confidence: float = 0.2
    #: The code range a prediction must land inside. One is the lowest code
    #: that is a return at all, because zero is no measurement.
    min_code: int = 1
    max_code: int = MAX_CODE
    #: How many passes over the holes are made. Exactly one, and this field
    #: exists to say so and to refuse anything else: a second pass could only
    #: differ by using the first pass's own output as evidence.
    passes: int = 1

    def __post_init__(self) -> None:
        radius = _positive_int(self.radius, "radius", MAX_RADIUS)
        min_support = _positive_int(self.min_support, "min_support", MAX_SUPPORT)
        max_support = _positive_int(self.max_support, "max_support", MAX_SUPPORT)
        _positive_int(self.max_component_samples, "max_component_samples", MAX_COMPONENT_SAMPLES)
        _positive_int(self.max_component_span, "max_component_span", MAX_COMPONENT_SPAN)
        min_code = _positive_int(self.min_code, "min_code", MAX_CODE)
        max_code = _positive_int(self.max_code, "max_code", MAX_CODE)
        _positive_number(self.min_spread, "min_spread", MAX_SPREAD)
        _positive_number(self.cluster_gap_codes, "cluster_gap_codes", float(MAX_CODE))
        _positive_number(self.max_support_span_codes, "max_support_span_codes", float(MAX_CODE))
        _positive_number(self.max_residual_codes, "max_residual_codes", float(MAX_CODE))
        _positive_number(self.max_overshoot_codes, "max_overshoot_codes", float(MAX_CODE))
        if isinstance(self.min_confidence, bool) or not isinstance(
            self.min_confidence, int | float
        ):
            raise PolicyUnusable(
                f"a completion policy states min_confidence {self.min_confidence!r}"
            )
        confidence = float(self.min_confidence)
        if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
            raise PolicyUnusable(
                f"a completion policy states min_confidence {self.min_confidence!r}; it is a "
                "fraction of 1 above zero, because a reconstruction nothing supports is not "
                "published at any confidence"
            )
        if max_support < min_support:
            raise PolicyUnusable(
                f"a completion policy allows at most {max_support} support sample(s) and requires "
                f"at least {min_support}"
            )
        if min_support > (2 * radius + 1) ** 2 - 1:
            raise PolicyUnusable(
                f"a completion policy requires {min_support} support sample(s) inside a radius of "
                f"{radius}, which holds at most {(2 * radius + 1) ** 2 - 1}"
            )
        if self.cluster_gap_codes > self.max_support_span_codes:
            raise PolicyUnusable(
                "a completion policy allows a single support gap wider than the whole span it "
                "allows, so the coarser bound could never fire"
            )
        if max_code > MAX_CODE or min_code >= max_code:
            raise PolicyUnusable(
                f"a completion policy predicts into codes {min_code}..{max_code}, which is not a "
                f"range inside 1..{MAX_CODE}"
            )
        if self.passes != 1 or isinstance(self.passes, bool):
            raise PolicyUnusable(
                f"a completion policy states {self.passes!r} passes; this method makes exactly "
                "one, because a later pass could only use the earlier pass's own output as "
                "evidence and no reconstruction is evidence"
            )

    @property
    def neighbourhood_samples(self) -> int:
        """The most samples one hole's neighbourhood scan can visit."""
        return (2 * self.radius + 1) ** 2

    def worst_case_work(self, holes: int) -> int:
        """The most neighbourhood visits ``holes`` hole(s) can cost under this policy.

        Exact rather than an estimate, and derivable from the input before the
        run: every hole is examined at most once and each examination scans one
        neighbourhood. :func:`complete_patch` refuses a call whose value here
        exceeds :data:`MAX_WORK_UNITS` before it scans anything.
        """
        if isinstance(holes, bool) or not isinstance(holes, int) or holes < 0:
            raise PolicyUnusable(f"{holes!r} is not a number of holes")
        return holes * self.neighbourhood_samples

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "radius": self.radius,
            "min_support": self.min_support,
            "max_support": self.max_support,
            "min_spread": self.min_spread,
            "cluster_gap_codes": self.cluster_gap_codes,
            "max_support_span_codes": self.max_support_span_codes,
            "max_residual_codes": self.max_residual_codes,
            "max_overshoot_codes": self.max_overshoot_codes,
            "max_component_samples": self.max_component_samples,
            "max_component_span": self.max_component_span,
            "min_confidence": self.min_confidence,
            "min_code": self.min_code,
            "max_code": self.max_code,
            "passes": self.passes,
        }


#: The policy every caller gets unless it states otherwise.
DEFAULT_POLICY: Final = CompletionPolicy()


# --- what a run cost, and how much of it it refused -----------------------------------


def _exact_int(value: object, name: str) -> int:
    """One integer that has to be an integer, checked before it is compared.

    ``True == 1`` and ``1.0 == 1`` in Python, so an equality or a range test
    that runs first happily accepts a boolean width and a float schema version
    and only the type check refuses them. Every number in this module that a
    reader would treat as a fixed identity - a schema version, the dimensions
    of a result - is therefore type-checked here before its value is looked at.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResultUnusable(
            f"{name} is {value!r}, which is not an integer. ``True`` and ``1.0`` both compare "
            "equal to 1, so the type is checked before the value and not after it"
        )
    return value


def _work_count(value: object, name: str) -> int:
    """One count in a work record: a real non-negative integer, never a bool.

    ``True`` is an ``int`` in Python and would otherwise pass every arithmetic
    check below as a one. A count that arrived as a boolean is a bug in whatever
    produced it, and a work record is exactly where such a bug would be read as
    a fact about how conservative a run was.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ResultUnusable(f"work states {name} {value!r}, which is not a count")
    if value < 0:
        raise ResultUnusable(f"work states {name} {value!r}; a count of something is not negative")
    return value


@dataclass(frozen=True, slots=True)
class CompletionWork:
    """What one completion actually did: its cost, and every refusal by reason.

    Reported so conservatism is inspectable rather than inferred from the
    output. A completion that recovered little because the frame was hard and
    one that recovered little because it never looked are different facts, and
    without these counts they produce the same picture.
    """

    backend: str
    policy: CompletionPolicy
    holes: int
    attempted: int
    accepted: int
    rejected: int
    rejections: Mapping[str, int]
    support_samples: int
    work_units: int
    work_budget: int
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _exact_int(self.schema_version, "a work record's schema version")
        if self.schema_version != SCHEMA_VERSION:
            raise ResultUnusable(
                f"work states schema version {self.schema_version!r}; this module writes "
                f"{SCHEMA_VERSION} and a reader that does not know a version refuses rather "
                "than guessing which fields it is looking at"
            )
        if self.backend != PYTHON_BACKEND:
            raise ResultUnusable(
                f"work names backend {self.backend!r}; {PYTHON_BACKEND!r} is the only "
                "implementation in this module and so the only one that can have produced a run"
            )
        if not isinstance(self.policy, CompletionPolicy):
            raise ResultUnusable(
                f"work states policy {self.policy!r}, which is not a CompletionPolicy; the "
                "bounds a run was held to are the validated type or they are unchecked prose"
            )
        for name in ("holes", "attempted", "accepted", "rejected", "support_samples"):
            _work_count(getattr(self, name), name)
        for name in ("work_units", "work_budget"):
            _work_count(getattr(self, name), name)
        # The same copy-on-construction rule as DepthPatch.codes: a mapping the
        # caller still holds is not this record's own, and a count that can move
        # after validation was never validated.
        stated: object = self.rejections
        if not isinstance(stated, Mapping):
            raise ResultUnusable(f"work states rejections {stated!r}, which is not a map")
        for reason, count in stated.items():
            if not isinstance(reason, str):
                raise ResultUnusable(
                    f"work states rejection reason {reason!r}, which is not a name"
                )
            _work_count(count, f"rejections[{reason!r}]")
        object.__setattr__(self, "rejections", MappingProxyType(dict(stated)))
        if self.attempted != self.accepted + self.rejected:
            raise ResultUnusable(
                f"work attempted {self.attempted} hole(s) and accounts for "
                f"{self.accepted + self.rejected}"
            )
        if self.attempted != self.holes:
            raise ResultUnusable(
                f"work attempted {self.attempted} of {self.holes} hole(s); every hole is examined "
                "once, and one that was not looked at is a refusal with no reason"
            )
        if sorted(self.rejections) != sorted(REJECTION_REASONS):
            raise ResultUnusable(
                "work states rejection reasons that are not exactly the declared vocabulary"
            )
        if sum(self.rejections.values()) != self.rejected:
            raise ResultUnusable(
                f"work rejected {self.rejected} hole(s) and gives reasons for "
                f"{sum(self.rejections.values())}"
            )
        if self.work_units > self.work_budget:
            raise ResultUnusable(
                f"work spent {self.work_units} unit(s) against a budget of {self.work_budget}; "
                "the budget is derived from the input before the run and is not advisory"
            )
        bounds = (self.accepted * self.policy.min_support, self.accepted * self.policy.max_support)
        if not bounds[0] <= self.support_samples <= bounds[1]:
            raise ResultUnusable(
                f"work names {self.support_samples} support sample(s) for {self.accepted} "
                f"reconstruction(s), and its own policy allows {bounds[0]}..{bounds[1]}"
            )

    @property
    def within_budget(self) -> bool:
        return self.work_units <= self.work_budget

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "policy": self.policy.as_dict(),
            "holes": self.holes,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "rejections": {reason: self.rejections[reason] for reason in REJECTION_REASONS},
            "support_samples": self.support_samples,
            "work_units": self.work_units,
            "work_budget": self.work_budget,
            "within_budget": self.within_budget,
        }


# --- the output -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CompletedPatch:
    """One completed rectangle: a code, a state, a confidence and support per sample.

    The whole rectangle, always. Every array here is exactly
    ``width * height`` long and every index in it is accounted for, so a caller
    can count what happened against a denominator it fixed rather than one the
    completion chose.

    ``state_codes`` is ``bytes`` rather than a tuple of strings: one byte per
    sample keeps a full frame's provenance to half a megabyte instead of a
    tuple of a quarter of a million interned strings, and
    :meth:`state_at` is the only way anything reads it.

    The source is the point
    -----------------------
    :attr:`source` is the immutable :class:`DepthPatch` this result claims to
    preserve, and it is a *field* rather than a construction-time argument that
    is thrown away. Without it a result could only be checked against its own
    labels, and a label is exactly what a forged result would change: calling a
    sample ``"measured"`` proves nothing about whether the sensor returned it,
    and believing a support index because the result itself says that index is
    measured is letting a claim vouch for its own evidence.

    So every rule below is stated against ``source.codes``:

    * a source sample the sensor returned comes back with the same code, the
      state ``"measured"``, confidence ``1.0`` and no support - byte-for-byte
      identity, not similarity;
    * a source sample the sensor did not return is either exactly missing
      (code zero, confidence zero, no support) or a spatial reconstruction with
      a nonzero code, a finite confidence strictly between zero and one, and
      distinct support indices *whose source codes are nonzero*;
    * no other combination exists, so there is no state a forged result can
      relabel its way into.
    """

    width: int
    height: int
    codes: tuple[int, ...]
    state_codes: bytes
    confidence: tuple[float, ...]
    support: Mapping[int, tuple[int, ...]]
    work: CompletionWork
    #: The exact raw patch this result is derived from and preserves. Immutable,
    #: and the authority for every provenance claim the result makes.
    source: DepthPatch
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _exact_int(self.schema_version, "a result's schema version")
        if self.schema_version != SCHEMA_VERSION:
            raise ResultUnusable(
                f"a result states schema version {self.schema_version!r}; this module writes "
                f"{SCHEMA_VERSION}"
            )
        _exact_int(self.width, "a result's width")
        _exact_int(self.height, "a result's height")
        source = self.source
        if not isinstance(source, DepthPatch):
            raise ResultUnusable(
                f"a result names source {source!r}, which is not a DepthPatch. A result that "
                "carries no source can only be checked against its own labels, and a label is "
                "what a forged result changes"
            )
        samples = source.sample_count
        if (self.width, self.height) != (source.width, source.height):
            raise ResultUnusable(
                f"a {self.width}x{self.height} result claims to complete a "
                f"{source.width}x{source.height} patch; the denominator is the rectangle the "
                "caller handed over and a completion does not get to move it"
            )
        # Copy every caller-owned container before anything is validated. A
        # frozen dataclass freezes the binding; the list, bytearray or dict on
        # the other end of it stays the caller's unless it is copied here, and
        # a value that can change after validation was never validated.
        object.__setattr__(
            self,
            "codes",
            _frozen_sequence(
                self.codes,
                f"a result carries {self.codes!r}, which is not a sequence of codes",
                ResultUnusable,
            ),
        )
        object.__setattr__(
            self,
            "state_codes",
            _frozen_bytes(
                self.state_codes,
                f"a result carries states {self.state_codes!r}, which is not one byte per sample",
            ),
        )
        object.__setattr__(
            self,
            "confidence",
            _frozen_sequence(
                self.confidence,
                f"a result carries confidence {self.confidence!r}, which is not a sequence",
                ResultUnusable,
            ),
        )
        given: object = self.support
        if not isinstance(given, Mapping):
            raise ResultUnusable(f"a result carries support {given!r}, which is not a mapping")
        frozen_support: dict[int, tuple[int, ...]] = {}
        for index, sources in given.items():
            at = f"a result names support for {index!r}"
            if isinstance(index, bool) or not isinstance(index, int):
                raise ResultUnusable(f"{at}, which is not a sample index")
            named = _frozen_sequence(
                sources, f"{at} as {sources!r}, which is not a list of indices", ResultUnusable
            )
            indices: list[int] = []
            for source_index in named:
                if isinstance(source_index, bool) or not isinstance(source_index, int):
                    raise ResultUnusable(f"{at} with {source_index!r}, which is not an index")
                indices.append(source_index)
            frozen_support[index] = tuple(indices)
        object.__setattr__(self, "support", MappingProxyType(frozen_support))

        counts = (len(self.codes), len(self.state_codes), len(self.confidence))
        if counts != (samples,) * 3:
            raise ResultUnusable(
                f"a {self.width}x{self.height} result carries {counts} value(s) for {samples} "
                "sample(s); a result that is not the whole rectangle cannot be counted against it"
            )
        for index, code in enumerate(self.codes):
            if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code <= MAX_CODE:
                raise ResultUnusable(f"a result carries depth code {code!r} at index {index}")
        for index, state in enumerate(self.state_codes):
            if state >= len(STATE_NAMES):
                raise ResultUnusable(f"a result carries state {state!r} at index {index}")
        for index, value in enumerate(self.confidence):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ResultUnusable(f"a result carries confidence {value!r} at index {index}")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ResultUnusable(
                    f"a result carries confidence {value!r} at index {index}; confidence is a "
                    "finite fraction of 1"
                )
        for index in self.support:
            if index not in range(samples):
                raise ResultUnusable(f"a result names support for index {index}, which it has not")
        self._bind_to_source(source, samples)
        self._bind_to_work(source, samples)
        self._bind_to_replay(source)

    def _bind_to_source(self, source: DepthPatch, samples: int) -> None:
        """Every sample of the result, against the raw sample it came from.

        The whole point of holding the source. Nothing here reads one of the
        result's own state labels as evidence for anything: what a sample is
        allowed to be follows from ``source.codes``, which the result cannot
        change, and the labels are then required to agree with that.
        """
        for index in range(samples):
            raw = source.codes[index]
            state = self.state_codes[index]
            code = self.codes[index]
            value = float(self.confidence[index])
            sources = self.support.get(index, ())
            if raw != NO_RETURN_SAMPLE:
                if state != STATE_MEASURED:
                    raise ResultUnusable(
                        f"a result calls index {index} {self.state_at(index)!r}, and the frame it "
                        f"was given returned code {raw} there. A measurement stays a measurement"
                    )
                if code != raw:
                    raise ResultUnusable(
                        f"a result carries code {code} at index {index}, where the frame it was "
                        f"given carries {raw}. Raw measurements come back byte-for-byte or this "
                        "is not a completion"
                    )
                if value != 1.0:
                    raise ResultUnusable(
                        f"a result publishes measured index {index} at confidence {value!r}; a "
                        "sample the sensor returned is carried at 1.0 and nothing else"
                    )
                if sources:
                    raise ResultUnusable(
                        f"a result names support for measured index {index}; a measurement rests "
                        "on the sensor and not on its neighbours"
                    )
                continue
            if state == STATE_MISSING:
                if code != NO_RETURN_SAMPLE or value != 0.0 or sources:
                    raise ResultUnusable(
                        f"a result calls index {index} missing and carries code {code!r} at "
                        f"confidence {value!r} with {len(sources)} support sample(s); a refused "
                        "hole is exactly code zero, confidence zero and no evidence"
                    )
                continue
            if state != STATE_SPATIALLY_RECONSTRUCTED:
                raise ResultUnusable(
                    f"a result calls index {index} {self.state_at(index)!r}, and the frame it was "
                    "given returned nothing there. A reconstruction may occupy a no-return "
                    "sample; it may never be relabelled as one of the sensor's own"
                )
            if code == NO_RETURN_SAMPLE:
                raise ResultUnusable(
                    f"a result reconstructs index {index} and carries code zero there, which is "
                    "no measurement rather than a depth"
                )
            if not 0.0 < value < 1.0:
                raise ResultUnusable(
                    f"a result publishes reconstructed index {index} at confidence {value!r}; a "
                    "reconstruction is worth more than nothing and strictly less than a "
                    "measurement, whose 1.0 is reserved for what the sensor returned"
                )
            if not sources:
                raise ResultUnusable(
                    f"a result reconstructs index {index} and names no support for it; a value "
                    "with no named evidence is invention, whatever its confidence says"
                )
            if len(set(sources)) != len(sources):
                raise ResultUnusable(
                    f"a result supports index {index} with the same sample twice; counting one "
                    "measurement as two is how thin evidence is made to look like enough"
                )
            for support_index in sources:
                if support_index not in range(samples) or support_index == index:
                    raise ResultUnusable(
                        f"a result supports index {index} with {support_index}, which is not "
                        "another sample of the same patch"
                    )
                if source.codes[support_index] == NO_RETURN_SAMPLE:
                    raise ResultUnusable(
                        f"a result supports index {index} with index {support_index}, where the "
                        "frame it was given returned nothing. Only an original measurement is "
                        "evidence, and neither a reconstruction nor a label saying otherwise is"
                    )

    def _bind_to_work(self, source: DepthPatch, samples: int) -> None:
        """The work record, against the source, the result and the published policy.

        Every count here is one the source and the result already determine, so
        a run cannot state a conservatism it did not practise or a budget that
        was not derived from what it was handed.
        """
        work = self.work
        if not isinstance(work, CompletionWork):
            raise ResultUnusable(f"a result carries work {work!r}, which is not a CompletionWork")
        holes = sum(1 for code in source.codes if code == NO_RETURN_SAMPLE)
        if work.holes != holes or work.attempted != holes:
            raise ResultUnusable(
                f"a result's work faced {work.holes} hole(s) and attempted {work.attempted} of "
                f"them, beside a source frame with {holes}. How many holes there were is the "
                "input's fact and not the run's"
            )
        reconstructed = sum(
            1 for state in self.state_codes if state == STATE_SPATIALLY_RECONSTRUCTED
        )
        if work.accepted != reconstructed or work.rejected != holes - reconstructed:
            raise ResultUnusable(
                f"a result carries {reconstructed} reconstruction(s) beside work that accepted "
                f"{work.accepted} and rejected {work.rejected} of {holes}"
            )
        named = sum(len(sources) for sources in self.support.values())
        if named != work.support_samples:
            raise ResultUnusable(
                f"a result names {named} support sample(s) and its own work states "
                f"{work.support_samples}"
            )
        budget = work.policy.worst_case_work(holes)
        if work.work_budget != budget:
            raise ResultUnusable(
                f"a result's work states a budget of {work.work_budget} for {holes} hole(s) at "
                f"radius {work.policy.radius}, which derives {budget}. The budget is the input "
                "and the policy, not a number the run chose"
            )
        if budget > MAX_WORK_UNITS:
            raise ResultUnusable(
                f"a result's work derives a worst case of {budget} unit(s), past the ceiling of "
                f"{MAX_WORK_UNITS} this module states"
            )
        if samples > MAX_PATCH_SAMPLES:  # pragma: no cover - DepthPatch refuses this first
            raise ResultUnusable(f"a result of {samples} sample(s) is past the patch ceiling")

    def _bind_to_replay(self, source: DepthPatch) -> None:
        """Every published value, against what this exact source and policy derive.

        The rules above establish that a result has a *shape* the source
        permits: that a measurement came back unedited, that a reconstruction
        occupies a no-return sample, that its support indices were things the
        sensor returned. None of that says the published number is the one this
        method produces. A result can satisfy every one of them and still carry
        a reconstructed code of 65000 where the evidence gives 100, a
        confidence of 0.99 where the formula gives 0.31, support that is a
        different legal selection from the one the policy names, a work record
        that spent nothing, or a rejection taxonomy that moved every refusal
        from the reason that fired to a reason that did not.

        So the whole run is re-derived here from ``source`` and
        ``work.policy`` - the two immutable things the result names - and the
        result is required to *be* that derivation. Every reconstructed code,
        every confidence, every support selection, every hole that was refused
        and every count of what the run cost is then a deterministic
        consequence of facts the result cannot change, rather than a claim it
        makes about itself.

        Nothing here reads the result. The derivation is a pure function of the
        immutable source patch and the published policy, so no output of the
        completion is ever evidence for another part of it, and a forged field
        is refused by recomputation rather than by consistency with its
        neighbours.
        """
        derived = _completion_of(source, self.work.policy)
        for index in range(source.sample_count):
            if self.state_codes[index] != derived.states[index]:
                raise ResultUnusable(
                    f"a result calls index {index} {self.state_at(index)!r}, and this source "
                    f"under the policy it publishes derives "
                    f"{STATE_NAMES[derived.states[index]]!r}. Which holes are recoverable is a "
                    "consequence of the frame and the bounds, not a decision a result reports"
                )
            if self.codes[index] != derived.codes[index]:
                raise ResultUnusable(
                    f"a result carries code {self.codes[index]} at index {index}, where this "
                    f"source under the policy it publishes derives {derived.codes[index]}. A "
                    "published reconstruction is the outcome of its own named evidence or it is "
                    "a number with a provenance label attached to it"
                )
            if float(self.confidence[index]) != derived.confidence[index]:
                raise ResultUnusable(
                    f"a result publishes index {index} at confidence "
                    f"{float(self.confidence[index])!r}, where this source under the policy it "
                    f"publishes derives {derived.confidence[index]!r}. Confidence is computed "
                    "from named evidence and is not a number a result may state"
                )
            named = self.support.get(index, ())
            expected = derived.support.get(index, ())
            if named != expected:
                raise ResultUnusable(
                    f"a result supports index {index} with {named}, where this source under the "
                    f"policy it publishes selects {expected}. Changed, omitted and extra support "
                    "are all refused: the support of a reconstruction is chosen by the policy "
                    "from the frame, and a result that got to choose it could name evidence for "
                    "a value that evidence does not give"
                )
        work = self.work
        if work.work_units != derived.work_units:
            raise ResultUnusable(
                f"a result's work states {work.work_units} neighbourhood visit(s), and this "
                f"source under the policy it publishes spends {derived.work_units}. What a run "
                "cost is what it did, not a total that merely fits under the budget"
            )
        for reason in REJECTION_REASONS:
            if work.rejections[reason] != derived.rejections[reason]:
                raise ResultUnusable(
                    f"a result's work refuses {work.rejections[reason]} hole(s) for {reason!r}, "
                    f"and this source under the policy it publishes refuses "
                    f"{derived.rejections[reason]}. Every refusal is counted against the reason "
                    "that actually fired; a taxonomy that only adds up to the right total says "
                    "nothing about why a frame came back sparse"
                )
        if work.accepted != derived.accepted or work.support_samples != derived.support_samples:
            raise ResultUnusable(  # pragma: no cover - the per-sample checks refuse this first
                f"a result's work accepted {work.accepted} hole(s) on {work.support_samples} "
                f"support sample(s), and this source under the policy it publishes derives "
                f"{derived.accepted} on {derived.support_samples}"
            )

    @property
    def sample_count(self) -> int:
        return self.width * self.height

    def state_at(self, index: int) -> str:
        """The provenance of one sample, by name."""
        return STATE_NAMES[self.state_codes[index]]

    def states(self) -> tuple[str, ...]:
        """Every provenance state, by name, in index order."""
        return tuple(STATE_NAMES[state] for state in self.state_codes)

    def support_at(self, index: int) -> tuple[int, ...]:
        """The original measured samples one reconstruction rests on. Empty otherwise."""
        return self.support.get(index, ())

    def position_of(self, index: int) -> tuple[int, int]:
        row, column = divmod(index, self.width)
        return column, row

    def reconstructed_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, state in enumerate(self.state_codes)
            if state == STATE_SPATIALLY_RECONSTRUCTED
        )

    def as_dict(self) -> dict[str, object]:
        """A JSON-shaped summary. Counts and work, never a quarter-million codes."""
        states = self.states()
        return {
            "schema_version": self.schema_version,
            "width": self.width,
            "height": self.height,
            "sample_count": self.sample_count,
            "measured": states.count(STATE_NAMES[STATE_MEASURED]),
            "spatially_reconstructed": states.count(STATE_NAMES[STATE_SPATIALLY_RECONSTRUCTED]),
            "missing": states.count(STATE_NAMES[STATE_MISSING]),
            "work": self.work.as_dict(),
        }


# --- the support of one sample --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupportSample:
    """One original measurement a reconstruction rests on, and where it is."""

    index: int
    column_offset: int
    row_offset: int
    code: int

    @property
    def distance(self) -> float:
        return math.hypot(self.column_offset, self.row_offset)


@dataclass(frozen=True, slots=True)
class SupportFit:
    """The least-squares plane one neighbourhood's support determines.

    Carried out of the fit rather than reduced to a number, because every one
    of these fields is an input to :func:`support_confidence` and to the
    reproduction check: an accepted sample can be recomputed from its support
    and the policy alone, and this is what that recomputation produces.
    """

    predicted: float
    worst_residual: float
    mean_distance: float
    code_span: float
    weaker_spread: float
    support_count: int


def _solve_symmetric_3(
    matrix: Sequence[Sequence[float]], rhs: Sequence[float]
) -> tuple[float, float, float] | None:
    """Gaussian elimination with partial pivoting on a 3x3 system, or ``None``.

    ``None`` rather than an exception: a singular normal-matrix means the
    support determines no plane, which is an ordinary refusal reason and not an
    error. The pivot floor is relative to the largest entry, so it scales with
    the neighbourhood rather than assuming a magnitude.
    """
    rows = [
        [float(value) for value in row] + [float(rhs[index])] for index, row in enumerate(matrix)
    ]
    scale = max((abs(value) for row in rows for value in row[:3]), default=0.0)
    if scale <= 0.0:
        return None
    floor = scale * 1e-12
    for column in range(3):
        pivot = max(range(column, 3), key=lambda index: abs(rows[index][column]))
        if abs(rows[pivot][column]) <= floor:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for other in range(column + 1, 3):
            factor = rows[other][column] / rows[column][column]
            for step in range(column, 4):
                rows[other][step] -= factor * rows[column][step]
    solution = [0.0, 0.0, 0.0]
    for column in reversed(range(3)):
        total = rows[column][3] - sum(
            rows[column][step] * solution[step] for step in range(column + 1, 3)
        )
        solution[column] = total / rows[column][column]
    return solution[0], solution[1], solution[2]


def fit_support(support: Sequence[SupportSample]) -> SupportFit | None:
    """The plane ``code = a*column + b*row + c`` the support determines, at the hole.

    ``None`` when the normal equations are singular - support along one line,
    or repeated at one position, determines no plane. The offsets are relative
    to the sample being reconstructed, so ``c`` is the prediction *at* it and
    no separate evaluation step can disagree with the fit.
    """
    if not support:
        return None
    sum_cc = sum(float(item.column_offset) ** 2 for item in support)
    sum_rr = sum(float(item.row_offset) ** 2 for item in support)
    sum_cr = sum(float(item.column_offset) * item.row_offset for item in support)
    sum_c = sum(float(item.column_offset) for item in support)
    sum_r = sum(float(item.row_offset) for item in support)
    count = float(len(support))
    sum_zc = sum(float(item.code) * item.column_offset for item in support)
    sum_zr = sum(float(item.code) * item.row_offset for item in support)
    sum_z = sum(float(item.code) for item in support)
    solution = _solve_symmetric_3(
        ((sum_cc, sum_cr, sum_c), (sum_cr, sum_rr, sum_r), (sum_c, sum_r, count)),
        (sum_zc, sum_zr, sum_z),
    )
    if solution is None:
        return None
    slope_c, slope_r, predicted = solution
    if not all(math.isfinite(value) for value in (slope_c, slope_r, predicted)):
        return None
    worst = 0.0
    for item in support:
        fitted = slope_c * item.column_offset + slope_r * item.row_offset + predicted
        residual = abs(float(item.code) - fitted)
        worst = max(worst, residual)
    codes = [float(item.code) for item in support]
    return SupportFit(
        predicted=predicted,
        worst_residual=worst,
        mean_distance=sum(item.distance for item in support) / count,
        code_span=max(codes) - min(codes),
        weaker_spread=_weaker_spread(support),
        support_count=len(support),
    )


def _weaker_spread(support: Sequence[SupportSample]) -> float:
    """The smaller eigenvalue of the support's second-moment matrix, about the hole.

    About the hole rather than about the support's own centroid on purpose:
    what has to be established is that the sample being reconstructed is
    surrounded, and a tight cluster of support all on one side has a perfectly
    healthy spread about *its own* centre.
    """
    sum_cc = sum(float(item.column_offset) ** 2 for item in support)
    sum_rr = sum(float(item.row_offset) ** 2 for item in support)
    sum_cr = sum(float(item.column_offset) * item.row_offset for item in support)
    count = float(len(support))
    if count <= 0.0:
        return 0.0
    a, b, d = sum_cc / count, sum_cr / count, sum_rr / count
    trace, determinant = a + d, a * d - b * b
    discriminant = max(trace * trace / 4.0 - determinant, 0.0)
    return trace / 2.0 - math.sqrt(discriminant)


def support_confidence(fit: SupportFit, policy: CompletionPolicy) -> float:
    """How much this reconstruction is worth, from the evidence that produced it.

    Five terms, each a fraction of 1, combined as their geometric mean so that
    every one of them can veto and none of them can be traded away:

    ``amount``
        ``(count - min_support + 1) / (max_support - min_support + 1)``,
        clamped. More supporting measurements raises it. The ``+ 1`` is
        deliberate: the least support the policy accepts is still support, and
        scoring it at exactly zero would make the acceptance bound and the
        confidence floor the same test twice, refusing every reconstruction
        that rests on the minimum rather than publishing it as the weakest one.
    ``spread``
        ``sqrt(weaker_spread) / radius``, clamped. Support distributed around
        the sample raises it; support strung along one direction lowers it.
    ``proximity``
        ``1 - mean_distance / (radius + 1)``. Support further away lowers it.
    ``fit``
        ``1 - worst_residual / max_residual_codes``. A plane that fits its own
        support badly lowers it.
    ``agreement``
        ``1 - code_span / max_support_span_codes``. Support that disagrees
        about the depth lowers it.

    Each term is monotone in the direction its name implies, so the geometric
    mean is too: worsening the residual, moving the support further away or
    widening the disagreement can only lower the result, never raise it. That
    monotonicity is the property a caller thresholding on confidence depends
    on, and it is tested directly.

    The result is a property of the reconstruction's evidence and **not** a
    camera confidence. It says nothing about the sensor, and no value of it
    ever makes a reconstructed sample a measurement.
    """
    spare = float(max(policy.max_support - policy.min_support, 0) + 1)
    amount = _clamp((fit.support_count - policy.min_support + 1) / spare)
    spread = _clamp(math.sqrt(max(fit.weaker_spread, 0.0)) / float(policy.radius))
    proximity = _clamp(1.0 - fit.mean_distance / float(policy.radius + 1))
    quality = _clamp(1.0 - fit.worst_residual / policy.max_residual_codes)
    agreement = _clamp(1.0 - fit.code_span / policy.max_support_span_codes)
    terms = (amount, spread, proximity, quality, agreement)
    if any(term <= 0.0 for term in terms):
        return 0.0
    return _clamp(math.exp(sum(math.log(term) for term in terms) / len(terms)))


def _clamp(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


# --- hole components ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HoleComponent:
    """One four-connected run of no-return samples, and how big it got."""

    indices: tuple[int, ...]
    column_span: int
    row_span: int

    @property
    def sample_count(self) -> int:
        return len(self.indices)


def hole_components(patch: DepthPatch) -> tuple[HoleComponent, ...]:
    """Every four-connected component of no-return samples, in index order.

    Four-connected rather than eight: two holes that touch only at a corner are
    two holes, and the diagonal between them is measured surface that separates
    them. Deterministic - components come out ordered by their lowest index and
    each one's indices are sorted - so nothing downstream depends on iteration
    order.
    """
    holes = set(patch.hole_indices())
    seen: set[int] = set()
    components: list[HoleComponent] = []
    for start in sorted(holes):
        if start in seen:
            continue
        seen.add(start)
        stack = [start]
        found: list[int] = []
        while stack:
            index = stack.pop()
            found.append(index)
            column, row = patch.position_of(index)
            for step_column, step_row in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = patch.index_of(column + step_column, row + step_row)
                if neighbour is not None and neighbour in holes and neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        columns = [patch.position_of(index)[0] for index in found]
        rows = [patch.position_of(index)[1] for index in found]
        components.append(
            HoleComponent(
                indices=tuple(sorted(found)),
                column_span=max(columns) - min(columns) + 1,
                row_span=max(rows) - min(rows) + 1,
            )
        )
    return tuple(components)


# --- the completion -------------------------------------------------------------------


def select_backend(name: str = AUTO_BACKEND) -> str:
    """The backend that will actually run, or an error naming why not.

    Exact, never a downgrade, on :mod:`scanner3d.compute`'s rule: ``"auto"``
    and ``"python"`` both give the reference, and every other recognised name
    is refused with a reason rather than silently replaced by a different
    implementation whose numbers a caller would then misattribute.
    """
    if name in (AUTO_BACKEND, PYTHON_BACKEND):
        return PYTHON_BACKEND
    if name in _UNIMPLEMENTED_BACKENDS:
        raise ComputeBackendUnavailable(
            f"scanner3d.depth_completion has no {name!r} backend. The dependency-free "
            "'python' reference is the only implementation and is the authority this "
            "module's benchmark scores; a vectorised backend would have to be parity-tested "
            "against it before it could be selected"
        )
    raise ComputeBackendUnknown(
        f"{name!r} is not a completion backend; this module states {[AUTO_BACKEND, PYTHON_BACKEND]}"
    )


@dataclass(slots=True)
class _Tally:
    """Mutable bookkeeping for one run. Never escapes :func:`complete_patch`."""

    accepted: int = 0
    support_samples: int = 0
    work_units: int = 0
    rejections: dict[str, int] = field(default_factory=lambda: dict.fromkeys(REJECTION_REASONS, 0))

    def reject(self, reason: str, count: int = 1) -> None:
        self.rejections[reason] += count


def _gather_support(
    patch: DepthPatch, index: int, policy: CompletionPolicy
) -> tuple[list[SupportSample], int]:
    """The nearest original measurements around one hole, and what looking cost.

    Ordered by distance and then by position, so ties are broken by a total
    order and the chosen support of a given patch is a function of the patch
    alone. Only samples the caller handed over are eligible: this reads
    ``patch``, which is immutable, and never the result being built.
    """
    column, row = patch.position_of(index)
    radius = policy.radius
    found: list[SupportSample] = []
    visits = 0
    for step_row in range(-radius, radius + 1):
        for step_column in range(-radius, radius + 1):
            visits += 1
            if step_row == 0 and step_column == 0:
                continue
            if step_column * step_column + step_row * step_row > radius * radius:
                continue
            neighbour = patch.index_of(column + step_column, row + step_row)
            if neighbour is None or not patch.is_measured(neighbour):
                continue
            found.append(
                SupportSample(
                    index=neighbour,
                    column_offset=step_column,
                    row_offset=step_row,
                    code=patch.codes[neighbour],
                )
            )
    found.sort(
        key=lambda item: (
            item.column_offset * item.column_offset + item.row_offset * item.row_offset,
            item.row_offset,
            item.column_offset,
        )
    )
    return found[: policy.max_support], visits


def _surrounds(support: Sequence[SupportSample]) -> bool:
    """Whether the support has measurements on both sides along both axes.

    The difference between interpolating inside evidence and extrapolating past
    its edge. A hole at the rim of a silhouette has support on one side only,
    and every value a fit would put there is a guess about a surface nothing
    measured.
    """
    return (
        any(item.column_offset < 0 for item in support)
        and any(item.column_offset > 0 for item in support)
        and any(item.row_offset < 0 for item in support)
        and any(item.row_offset > 0 for item in support)
    )


def _widest_gap(support: Sequence[SupportSample]) -> float:
    codes = sorted(float(item.code) for item in support)
    return max((later - earlier for earlier, later in itertools.pairwise(codes)), default=0.0)


def _refuse_component(component: HoleComponent, policy: CompletionPolicy) -> str | None:
    if component.sample_count > policy.max_component_samples:
        return "component_too_large"
    if max(component.column_span, component.row_span) > policy.max_component_span:
        return "component_too_long"
    return None


def _attempt(
    patch: DepthPatch, index: int, policy: CompletionPolicy, tally: _Tally
) -> tuple[int, float, tuple[int, ...]] | None:
    """One hole, taken as far as the policy allows. ``None`` means it stays missing.

    Every refusal is counted against its own reason before returning, so the
    reason a frame came back sparse is in the work record and not only in the
    absence of a value.
    """
    support, visits = _gather_support(patch, index, policy)
    tally.work_units += visits
    if len(support) < policy.min_support:
        tally.reject("insufficient_support")
        return None
    if not _surrounds(support):
        tally.reject("one_sided_support")
        return None
    if _widest_gap(support) > policy.cluster_gap_codes:
        tally.reject("incompatible_clusters")
        return None
    codes = [float(item.code) for item in support]
    if max(codes) - min(codes) > policy.max_support_span_codes:
        tally.reject("support_span_too_wide")
        return None
    fit = fit_support(support)
    if fit is None:
        tally.reject("degenerate_fit")
        return None
    if fit.weaker_spread < policy.min_spread:
        tally.reject("collinear_support")
        return None
    if fit.worst_residual > policy.max_residual_codes:
        tally.reject("residual_too_large")
        return None
    # The finiteness test comes first because everything after it rounds:
    # fit_support already refuses a non-finite solution, and this is the guard
    # that keeps that guarantee local rather than borrowed.
    low = min(codes) - policy.max_overshoot_codes
    high = max(codes) + policy.max_overshoot_codes
    if not math.isfinite(fit.predicted) or not low <= fit.predicted <= high:
        tally.reject("prediction_out_of_range")
        return None
    predicted = round(fit.predicted)
    if not policy.min_code <= predicted <= policy.max_code:
        tally.reject("prediction_out_of_range")
        return None
    confidence = support_confidence(fit, policy)
    if confidence < policy.min_confidence:
        tally.reject("confidence_below_threshold")
        return None
    return predicted, confidence, tuple(sorted(item.index for item in support))


@dataclass(frozen=True, slots=True)
class _Completion:
    """Everything one patch and one policy deterministically produce.

    Internal, and the single definition of what this method does. Both
    :func:`complete_patch` and :meth:`CompletedPatch._bind_to_replay` read it,
    so the value that is published and the value a result is checked against
    cannot drift into two different answers.
    """

    codes: tuple[int, ...]
    states: bytes
    confidence: tuple[float, ...]
    support: Mapping[int, tuple[int, ...]]
    accepted: int
    support_samples: int
    work_units: int
    rejections: Mapping[str, int]


def _derive(patch: DepthPatch, policy: CompletionPolicy) -> _Completion:
    """The one pass, over the components of one immutable patch. Pure.

    Reads ``patch`` and ``policy`` and nothing else - no clock, no randomness,
    no I/O and, crucially, none of the values it is producing: support comes
    from ``patch.codes``, so no reconstruction is ever evidence for another.
    """
    codes = list(patch.codes)
    states = bytearray(
        STATE_MEASURED if patch.is_measured(index) else STATE_MISSING
        for index in range(patch.sample_count)
    )
    confidence = [1.0 if patch.is_measured(index) else 0.0 for index in range(patch.sample_count)]
    support: dict[int, tuple[int, ...]] = {}
    tally = _Tally()
    for component in hole_components(patch):
        refusal = _refuse_component(component, policy)
        if refusal is not None:
            tally.reject(refusal, component.sample_count)
            continue
        for index in component.indices:
            attempt = _attempt(patch, index, policy, tally)
            if attempt is None:
                continue
            code, value, sources = attempt
            codes[index] = code
            states[index] = STATE_SPATIALLY_RECONSTRUCTED
            confidence[index] = value
            support[index] = sources
            tally.accepted += 1
            tally.support_samples += len(sources)
    return _Completion(
        codes=tuple(codes),
        states=bytes(states),
        confidence=tuple(confidence),
        support=MappingProxyType(support),
        accepted=tally.accepted,
        support_samples=tally.support_samples,
        work_units=tally.work_units,
        rejections=MappingProxyType(dict(tally.rejections)),
    )


#: How many derivations are remembered at once. Two, deliberately: the only hit
#: that has to land is the one a result's own validation makes immediately after
#: :func:`complete_patch` derived the same thing, so a completion costs one pass
#: rather than two. A patch can be a quarter of a million samples, and a cache
#: that remembered many of them would trade a bounded module for a growing one.
_REMEMBERED_DERIVATIONS: Final = 2


@functools.lru_cache(maxsize=_REMEMBERED_DERIVATIONS)
def _completion_of(patch: DepthPatch, policy: CompletionPolicy) -> _Completion:
    """:func:`_derive`, memoised on the two immutable values it reads.

    Memoising is sound here only because both arguments are frozen, hashable
    and deeply copied at construction, and because :func:`_derive` is pure: the
    same patch and policy have exactly one answer, so remembering it is an
    optimisation and never a semantic. Callers must have checked
    :data:`MAX_WORK_UNITS` against ``policy.worst_case_work`` before arriving
    here; this function does no work bounding of its own.
    """
    return _derive(patch, policy)


def complete_patch(
    patch: DepthPatch,
    *,
    policy: CompletionPolicy = DEFAULT_POLICY,
    backend: str = AUTO_BACKEND,
) -> CompletedPatch:
    """Recover the holes of one raw depth patch that this frame actually supports.

    Every returned sample comes back exactly as it was given, labelled
    ``"measured"`` with confidence ``1.0``. Every no-return sample is examined
    exactly once; the ones the policy accepts come back
    ``"spatially_reconstructed"`` with a finite confidence and the indices of
    the original measurements they rest on, and every other one comes back at
    raw code zero, confidence zero and ``"missing"`` with its refusal counted
    by reason in :attr:`CompletedPatch.work`.

    Pure: the same patch and policy always give the same result, no clock, no
    randomness and no I/O. Bounded: one pass, at most
    ``holes * (2 * radius + 1) ** 2`` neighbourhood visits, which
    :attr:`CompletionWork.work_budget` states before the run.
    """
    resolved = select_backend(backend)
    if not isinstance(policy, CompletionPolicy):
        raise PolicyUnusable(f"a completion needs a CompletionPolicy, not {policy!r}")
    if not isinstance(patch, DepthPatch):
        raise PatchUnusable(f"a completion needs a DepthPatch, not {patch!r}")
    holes = patch.hole_indices()
    # Before the first neighbourhood is scanned, and derived rather than
    # measured: the worst this call can cost is one neighbourhood per hole, and
    # a call past the ceiling is refused rather than discovered to be unbounded
    # while it runs.
    budget = policy.worst_case_work(len(holes))
    if budget > MAX_WORK_UNITS:
        raise WorkTooLarge(
            f"completing {len(holes)} hole(s) at radius {policy.radius} is worth up to {budget} "
            f"neighbourhood visit(s), past the ceiling of {MAX_WORK_UNITS} this module states. "
            "Tighten the radius or complete a smaller patch; a bound that large is not one"
        )
    derived = _completion_of(patch, policy)
    work = CompletionWork(
        backend=resolved,
        policy=policy,
        holes=len(holes),
        attempted=len(holes),
        accepted=derived.accepted,
        rejected=len(holes) - derived.accepted,
        rejections=dict(derived.rejections),
        support_samples=derived.support_samples,
        work_units=derived.work_units,
        work_budget=budget,
    )
    return CompletedPatch(
        width=patch.width,
        height=patch.height,
        codes=derived.codes,
        state_codes=derived.states,
        confidence=derived.confidence,
        support=derived.support,
        work=work,
        source=patch,
    )


def reconstructed_samples(result: CompletedPatch) -> Iterator[tuple[int, int, float]]:
    """Every reconstruction, as ``(index, code, confidence)``, in index order."""
    for index in result.reconstructed_indices():
        yield index, result.codes[index], result.confidence[index]
