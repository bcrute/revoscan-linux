"""What a depth frame is allowed to be worth to :mod:`scanner3d.tsdf`.

Why this module exists
----------------------
:meth:`scanner3d.tsdf.TsdfVolume.integrate` takes raw frames, and every sample
the sensor returned weighs exactly the same as every other. That is the honest
default while the only thing a frame carries is measurement: the running
weighted mean a TSDF keeps is the minimum-variance estimator precisely when the
observations are equally uncertain.

:mod:`scanner3d.depth_completion` and :mod:`scanner3d.depth_temporal` now
produce something a raw frame cannot express - a sample the sensor did NOT
return, standing where independent posed views agree a surface is, carrying an
exact confidence derived from that evidence. Handing such a sample to
``integrate`` would fuse it as a measurement, which is the one thing this
repository has spent three tasks refusing to do. Dropping it loses evidence
that other views really did measure.

So this module defines the third option, and only the third option: a frame
whose every sample carries the PROVENANCE it came from and a finite FUSION
WEIGHT derived from that provenance. It converts nothing, estimates nothing and
completes nothing - the two accepted completion modules own every derivation
here, and this module is refused the right to duplicate them.

The weighting contract, stated once
-----------------------------------
Four provenances, and exactly four. The weight of each is fixed here and is not
a caller's parameter, because a weight a caller could raise is not evidence:

``measured``
    The sensor returned this sample. Weight :data:`MEASURED_WEIGHT`, which is
    ``1.0`` - the same full weight the raw path has always given it, so a
    weighted frame built from raw bytes fuses byte-for-byte identically to
    those bytes.
``temporally_reconstructed``
    :mod:`scanner3d.depth_temporal` accepted this sample from other posed
    frames that measured it. Weight
    ``TEMPORAL_WEIGHT_CEILING * confidence``, where ``confidence`` is that
    module's own :func:`~scanner3d.depth_temporal.temporal_confidence` value,
    republished exactly. Strictly increasing in the confidence, strictly
    positive, and strictly below :data:`MEASURED_WEIGHT` for every confidence
    that module can produce - so a reconstruction can move a voxel and can
    never outvote the measurement beside it.
``spatially_excluded``
    :mod:`scanner3d.depth_completion` proposed a value for this sample from
    NEIGHBOURS INSIDE THE SAME FRAME, and no independent view confirmed it.
    Weight exactly ``0.0``, and the code offered to fusion is
    :data:`~scanner3d.depth_metric.NO_RETURN_SAMPLE`. Task 029's
    aperture-ambiguity case is why: a one-sample gap in a surface and a
    one-sample dropout ON that surface are, at that size and from one frame,
    the same input bytes. A spatial estimate under that ambiguity is a
    hypothesis, and a hypothesis that can add weight to a voxel is invented
    geometry.
``missing``
    Nothing returned it and nothing reconstructed it. Weight exactly ``0.0``
    and code :data:`~scanner3d.depth_metric.NO_RETURN_SAMPLE`, which is what a
    raw no-return already was.

A weight of zero is *identical to absence*: a sample carrying it takes the
existing no-return path through integration, so it cannot update a voxel and
cannot turn unobserved space into observed space. That equivalence is not an
implementation detail, it is the property the two zero-weight provenances
exist to have, and :mod:`tests.test_tsdf` proves it directly.

What is derived here, and what is only carried
----------------------------------------------
Derived here: the mapping from a provenance to a weight, and nothing else.

Carried, never re-derived: which sample has which provenance, what code a
reconstruction has, what confidence it earned and what it rests on. Those come
out of an ALREADY VALIDATED
:class:`~scanner3d.depth_completion.CompletedPatch` or
:class:`~scanner3d.depth_temporal.TemporalPatch`, each of which has bound
itself to its own immutable raw inputs and re-derived its own run. A frame
built here holds that result as a field, so every provenance claim it makes
can be checked against the accepted result rather than believed - and a caller
cannot hand this module a bag of labels and have them fused.

What this is NOT
----------------
Not a product path. Nothing in :mod:`scanner3d` calls it: the GUI, the live
preview, the capture path, the tracker and
:mod:`scanner3d.reconstruction_export` are unchanged, and the export still
fuses its original raw frames. This is the fusion INPUT contract and the
synthetic evidence for it; routing an accepted weighted frame into an export is
a later, separately reviewed task.

Not a physical claim. Every number this module can produce comes from
synthetic frames; no weighting here has been shown to improve a scan of a real
object on the attached device.
"""

from __future__ import annotations

import array
import math
import struct
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Final

from scanner3d import depth_completion, depth_temporal
from scanner3d.depth_completion import CompletedPatch
from scanner3d.depth_metric import NO_RETURN_SAMPLE
from scanner3d.depth_temporal import TemporalPatch
from scanner3d.errors import DoctorStatus, ScannerError
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES, FRAME_HEIGHT, FRAME_WIDTH

#: What a caller may hand over as one frame's bytes. Any of the three is
#: copied on the way in - a frozen dataclass freezes the binding and not the
#: buffer on the other end of it - so a caller that still holds a bytearray
#: cannot change a frame after it has been checked.
type RawFrame = bytes | bytearray | memoryview

#: The schema every frame and summary this module writes stamps.
SCHEMA_VERSION: Final = 1

#: Samples in one frame of the one depth mode this repository captures.
FRAME_SAMPLES: Final = FRAME_WIDTH * FRAME_HEIGHT

#: What a returned sample is worth. The raw path's implicit weight, named.
MEASURED_WEIGHT: Final = 1.0

#: The most a temporal reconstruction at the highest confidence
#: :mod:`scanner3d.depth_temporal` will publish can be worth. Half a
#: measurement, and the reason is not tuning: at this ceiling two agreeing
#: reconstructions still weigh less than the one measurement they disagree
#: with, so a voxel a camera actually saw is never carried away from what it
#: saw by reconstructions alone. It is a fixed property of this contract and
#: not a caller's parameter, because a weight a caller could raise is not
#: evidence about anything.
TEMPORAL_WEIGHT_CEILING: Final = 0.5

#: What a sample no evidence supports is worth. Exactly zero, which integration
#: treats as absence.
EXCLUDED_WEIGHT: Final = 0.0

#: Where one sample offered to fusion came from. One byte per sample.
PROVENANCE_MEASURED: Final = 0
PROVENANCE_TEMPORALLY_RECONSTRUCTED: Final = 1
PROVENANCE_SPATIALLY_EXCLUDED: Final = 2
PROVENANCE_MISSING: Final = 3

#: The four names, in byte order. A provenance nobody defined is a claim
#: nobody checked, so anything else is refused rather than defaulted.
PROVENANCE_NAMES: Final[tuple[str, ...]] = (
    "measured",
    "temporally_reconstructed",
    "spatially_excluded",
    "missing",
)

#: The provenances that carry no weight at all, named so the two reasons stay
#: distinguishable in the evidence even though their arithmetic is identical.
UNWEIGHTED_PROVENANCES: Final[tuple[int, ...]] = (
    PROVENANCE_SPATIALLY_EXCLUDED,
    PROVENANCE_MISSING,
)

#: Which accepted producer a frame came through. A frame that names one it did
#: not come through is refused: the adapter is what says which result type is
#: the authority for its provenance.
ADAPTER_RAW: Final = "raw"
ADAPTER_SPATIAL: Final = "spatial_completion"
ADAPTER_TEMPORAL: Final = "temporal_completion"
ADAPTERS: Final[tuple[str, ...]] = (ADAPTER_RAW, ADAPTER_SPATIAL, ADAPTER_TEMPORAL)

#: The part of a weighted frame's storage that does not depend on what is in
#: it: the raw frame it preserves, the codes it offers and one provenance byte
#: per sample. 1.25 MiB, exactly, on every frame.
FIXED_FRAME_BYTES: Final = 2 * EXPECTED_FRAME_BYTES + FRAME_SAMPLES

#: The most temporally reconstructed samples one frame may carry. A frame is
#: the rectangle, so this is the rectangle: it exists so that the sparse weight
#: mapping cannot be handed indices the frame does not have.
MAX_RECONSTRUCTIONS: Final = FRAME_SAMPLES

#: What one reconstruction costs a frame, exactly: a little-endian ``uint32``
#: sample index beside a little-endian ``float64`` fusion weight, both held in
#: :class:`bytes`. These are byte counts of the encoding this module writes and
#: not an estimate of a Python object's footprint - a dict entry holding a
#: boxed ``int`` key and a boxed ``float`` value is several times this and is
#: not something a portable contract can state a number for, which is why the
#: weights are not held in one.
INDEX_ENTRY_BYTES: Final = 4
WEIGHT_ENTRY_BYTES: Final = 8
RECONSTRUCTION_ENTRY_BYTES: Final = INDEX_ENTRY_BYTES + WEIGHT_ENTRY_BYTES

#: Every byte one weighted frame can own, at the worst frame this contract
#: allows: the fixed part plus one entry for every sample in the rectangle,
#: which is what a frame whose every sample is a temporal reconstruction costs.
#: 4352000 bytes, and :meth:`WeightedDepthFrame.owned_bytes` is at most this on
#: every frame that can be built. Stated as a number a reader can check against
#: the buffers themselves rather than as the word "bounded": a long offline
#: sweep costs frames times this constant and never a growing per-frame
#: structure.
MAX_FRAME_BYTES: Final = FIXED_FRAME_BYTES + MAX_RECONSTRUCTIONS * RECONSTRUCTION_ENTRY_BYTES


class WeightedInputNotAllowed(ScannerError):
    """A weighted frame was offered that this module refuses to build or accept.

    Always raised before the frame exists, so a refused input leaves no
    half-built provenance behind it and nothing downstream can be handed a
    partially checked frame.
    """

    status = DoctorStatus.UNSUPPORTED_MODE


class FrameUnusable(WeightedInputNotAllowed):
    """The bytes, the provenance or the weights are not one whole frame."""


class ProvenanceUnusable(WeightedInputNotAllowed):
    """A sample's provenance, code or weight is not one this contract allows."""


class SourceUnusable(WeightedInputNotAllowed):
    """The accepted completion result does not belong to these raw bytes."""


class SummaryUnusable(WeightedInputNotAllowed):
    """The published counts are not the counts the frame actually carries."""


def temporal_fusion_weight(confidence: float) -> float:
    """What a temporal reconstruction of this exact confidence is worth.

    ``TEMPORAL_WEIGHT_CEILING * confidence``, and deliberately nothing more
    elaborate. Three properties are wanted and this is the simplest function
    that has all three:

    * **strictly increasing** in the confidence, so lowering an accepted
      reconstruction's confidence can only lower its influence on the field;
    * **strictly positive** for every confidence
      :mod:`scanner3d.depth_temporal` can accept, which is bounded below by
      its own ``min_confidence`` and above by
      :data:`~scanner3d.depth_temporal.MAX_RECONSTRUCTION_CONFIDENCE`;
    * **strictly below** :data:`MEASURED_WEIGHT` at every one of those
      confidences, with the widest margin at the top rather than the narrowest.

    One multiplication, so the result is exact in the sense that matters: two
    runs on the same confidence produce the identical float, on either backend.

    Refuses anything that is not a confidence. A boolean is not a number here -
    ``True`` would silently weigh as much as a perfect reconstruction - and a
    non-finite or out-of-range value has no place on a weight that multiplies
    into a running mean.
    """
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        raise ProvenanceUnusable(
            f"a reconstruction offers confidence {confidence!r}, which is not a number. A weight "
            "is derived from the accepted confidence or it is not derived at all"
        )
    value = float(confidence)
    if not math.isfinite(value) or not 0.0 < value <= depth_temporal.MAX_RECONSTRUCTION_CONFIDENCE:
        raise ProvenanceUnusable(
            f"a reconstruction offers confidence {value!r}; scanner3d.depth_temporal publishes a "
            f"confidence above 0 and at most "
            f"{depth_temporal.MAX_RECONSTRUCTION_CONFIDENCE}, and a weight is never derived from "
            "a value that module could not have produced"
        )
    return TEMPORAL_WEIGHT_CEILING * value


@dataclass(frozen=True, slots=True)
class WeightedFrameSummary:
    """Exactly what one weighted frame offers fusion, counted once.

    A partition: the four counts add to the frame's sample count, which is the
    denominator the caller fixed and this module cannot move. ``total_weight``
    is the sum of every sample's fusion weight, taken with :func:`math.fsum`,
    so it is the correctly rounded sum of those floats rather than an
    order-dependent accumulation - the same number whichever order the samples
    were visited in.
    """

    measured: int
    temporally_reconstructed: int
    spatially_excluded: int
    missing: int
    total_weight: float

    @property
    def sample_count(self) -> int:
        return (
            self.measured + self.temporally_reconstructed + self.spatially_excluded + self.missing
        )

    @property
    def weighted_samples(self) -> int:
        """Samples that can update a voxel at all: measured plus temporal."""
        return self.measured + self.temporally_reconstructed

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "measured": self.measured,
            "temporally_reconstructed": self.temporally_reconstructed,
            "spatially_excluded": self.spatially_excluded,
            "missing": self.missing,
            "sample_count": self.sample_count,
            "weighted_samples": self.weighted_samples,
            "total_weight": self.total_weight,
            "total_weight_is": (
                "math.fsum over every sample's fusion weight: correctly rounded and independent "
                "of the order the samples were visited in"
            ),
            "partitions": (
                "measured plus temporally_reconstructed plus spatially_excluded plus missing "
                "equals sample_count, which is the whole frame"
            ),
        }


def _exact_int(value: object, what: str) -> int:
    """An integer, and never a boolean wearing one's clothes."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FrameUnusable(f"{what} is {value!r}, which is not an integer")
    return value


def _frozen_bytes(value: object, what: str) -> bytes:
    """A caller's buffer, copied. A frozen binding is not a frozen bytearray."""
    if isinstance(value, bytes):
        return value
    if not isinstance(value, bytearray | memoryview):
        raise FrameUnusable(f"{what} is {value!r}, which is not a buffer of bytes")
    return bytes(value)


def _codes_of(frame: bytes) -> array.array[int]:
    """One frame's little-endian uint16 codes, on any host."""
    codes: array.array[int] = array.array("H")
    codes.frombytes(frame)
    if sys.byteorder != "little":  # pragma: no cover - x86/ARM hosts are little-endian
        codes.byteswap()
    return codes


#: Maps a raw byte to the provenance of a sample whose bytes are all zero
#: (missing) or not (measured). Used through :meth:`bytes.translate` so that
#: deriving a whole frame's baseline provenance is one C-level pass rather than
#: a quarter of a million interpreted iterations.
_BASE_PROVENANCE: Final = bytes([PROVENANCE_MISSING] + [PROVENANCE_MEASURED] * 255)


def _baseline_provenance(raw: bytes) -> bytes:
    """One provenance byte per sample, from the raw bytes and nothing else.

    A sample is measured exactly when its uint16 code is not
    :data:`~scanner3d.depth_metric.NO_RETURN_SAMPLE`, which is exactly when at
    least one of its two bytes is non-zero - so the two byte planes are ORed
    together as big integers and the result translated. Endianness does not
    enter: "either byte is set" is the same statement in both orders.
    """
    low = raw[0::2]
    high = raw[1::2]
    merged = (int.from_bytes(low, "big") | int.from_bytes(high, "big")).to_bytes(len(low), "big")
    return merged.translate(_BASE_PROVENANCE)


class SparseWeights(Mapping[int, float]):
    """``{sample index: fusion weight}`` for reconstructions, held in two buffers.

    A :class:`Mapping` to read and two :class:`bytes` objects to store: the
    sample indices as ascending little-endian ``uint32``, and the fusion
    weights as little-endian ``float64`` in the same order. Nothing else is
    kept, so what one of these costs is
    ``len(self) * RECONSTRUCTION_ENTRY_BYTES`` and that is a number rather than
    an estimate - :attr:`owned_bytes` reads it off the buffers.

    Why not a dict
    --------------
    Because the storage bound this module publishes has to be true at the worst
    frame it allows, and a frame may be entirely reconstruction: 256000
    entries. A ``dict`` of that size holds 256000 boxed integers and 256000
    boxed floats behind a table that grows in powers of two, and no portable
    exact byte count can be stated for it - :func:`sys.getsizeof` measures the
    table alone, on one interpreter, and is not a contract. Two ``bytes``
    objects are 3072000 bytes at that population on every interpreter, which is
    a bound a reader can check.

    Exactness is preserved, not traded for the compactness. ``float64`` is the
    width a Python float already is, and ``struct`` round-trips one unchanged -
    so a weight read back out of here is the identical float
    :func:`temporal_fusion_weight` derived from the accepted confidence, and
    both TSDF backends see the same value.

    Immutable, and not by convention: :class:`bytes` cannot be written to, the
    slots refuse rebinding, and a caller's mapping is copied on the way in.
    Lookup is a binary search over the ascending index buffer, which is
    ``O(log n)`` against a dict's ``O(1)`` - paid only at the reconstructions
    themselves, which is where fusion reads a weight at all.
    """

    __slots__ = ("_indices", "_weights")

    _indices: bytes
    _weights: bytes

    def __init__(self, entries: Mapping[int, float]) -> None:
        if not isinstance(entries, Mapping):
            raise FrameUnusable(
                f"a weighted frame carries weights {entries!r}, which is not a mapping of sample "
                "index to fusion weight"
            )
        if len(entries) > MAX_RECONSTRUCTIONS:
            raise FrameUnusable(
                f"a weighted frame carries {len(entries)} reconstruction weight(s) for a frame "
                f"of {MAX_RECONSTRUCTIONS} sample(s)"
            )
        checked: list[tuple[int, float]] = []
        for index, weight in entries.items():
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < FRAME_SAMPLES
            ):
                raise FrameUnusable(
                    f"a weighted frame carries a weight for {index!r}, which is not a sample of "
                    f"a {FRAME_WIDTH}x{FRAME_HEIGHT} frame"
                )
            if isinstance(weight, bool) or not isinstance(weight, int | float):
                raise ProvenanceUnusable(
                    f"a weighted frame carries weight {weight!r} at index {index}, which is not "
                    "a number. A boolean is not a weight: True would weigh as much as a perfect "
                    "reconstruction"
                )
            number = float(weight)
            if not math.isfinite(number) or not 0.0 < number < MEASURED_WEIGHT:
                raise ProvenanceUnusable(
                    f"a weighted frame carries weight {number!r} at index {index}. A "
                    f"reconstruction weighs strictly more than {EXCLUDED_WEIGHT} - a zero weight "
                    f"is absence and is recorded as absence - and strictly less than the "
                    f"{MEASURED_WEIGHT} a measurement weighs"
                )
            checked.append((int(index), number))
        checked.sort()
        count = len(checked)
        object.__setattr__(
            self, "_indices", struct.pack(f"<{count}I", *(index for index, _ in checked))
        )
        object.__setattr__(
            self, "_weights", struct.pack(f"<{count}d", *(weight for _, weight in checked))
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"{type(self).__name__} is immutable and {name!r} cannot be rebound; a weight that "
            "can change after the frame carrying it was validated was never validated"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable and {name!r} cannot be removed")

    def __len__(self) -> int:
        return len(self._weights) // WEIGHT_ENTRY_BYTES

    def __iter__(self) -> Iterator[int]:
        return iter(struct.unpack(f"<{len(self)}I", self._indices))

    def _position(self, index: int) -> int:
        """Where ``index`` sits in the ascending index buffer, or ``-1``."""
        low, high = 0, len(self)
        while low < high:
            middle = (low + high) // 2
            at = INDEX_ENTRY_BYTES * middle
            found = int.from_bytes(self._indices[at : at + INDEX_ENTRY_BYTES], "little")
            if found == index:
                return middle
            if found < index:
                low = middle + 1
            else:
                high = middle
        return -1

    def __contains__(self, key: object) -> bool:
        if isinstance(key, bool) or not isinstance(key, int):
            return False
        return self._position(key) >= 0

    def __getitem__(self, key: int) -> float:
        if isinstance(key, bool) or not isinstance(key, int):
            raise KeyError(key)
        position = self._position(key)
        if position < 0:
            raise KeyError(key)
        (weight,) = struct.unpack_from("<d", self._weights, WEIGHT_ENTRY_BYTES * position)
        return float(weight)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SparseWeights):
            return self._indices == other._indices and self._weights == other._weights
        if isinstance(other, Mapping):
            return len(self) == len(other) and all(
                key in other and other[key] == weight for key, weight in self.items()
            )
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self)!r})"

    @property
    def owned_bytes(self) -> int:
        """Every byte this mapping holds, read off the buffers it holds them in."""
        return len(self._indices) + len(self._weights)


@dataclass(frozen=True, slots=True)
class WeightedDepthFrame:
    """One frame offered to fusion, with a provenance and a weight per sample.

    Deeply immutable. Every buffer is copied on the way in, the sparse weight
    mapping is re-encoded into the two :class:`bytes` buffers a
    :class:`SparseWeights` holds, and the accepted completion result it binds
    to is itself a frozen, self-validating type - so nothing a caller still
    holds can change what this frame means after it has been checked.

    What it costs is a number rather than a word: :attr:`owned_bytes` is the
    length of those buffers added up, at most :data:`MAX_FRAME_BYTES`.

    The raw frame is the point
    --------------------------
    :attr:`raw` is the exact 512000 bytes the sensor (or the fixture) produced,
    and it is a *field* rather than a construction-time argument that is thrown
    away. Every rule below is stated against it:

    * a sample the raw frame returned is ``measured``, offers that identical
      code, and weighs exactly :data:`MEASURED_WEIGHT`. A weighted frame cannot
      change, drop or re-weight a measurement;
    * a sample the raw frame did not return is one of the other three
      provenances, and which one it may be follows from the accepted completion
      result this frame carries - not from a label it was handed.

    So a forged frame has nothing to forge with. Relabelling a sample
    ``measured`` contradicts :attr:`raw`; relabelling one
    ``temporally_reconstructed`` contradicts :attr:`completion`, which has
    already bound itself to its own raw inputs and re-derived its own run.

    How it is checked
    -----------------
    Not field by field, but by re-derivation: the whole frame that
    :attr:`raw`, :attr:`completion` and :attr:`origin` imply is rebuilt here
    and the offered codes, provenance, weights and summary are required to
    *be* it, exactly. The work is one C-level pass over the frame plus one
    interpreted pass over the completed rectangle, which is bounded by the
    rectangle the caller fixed.
    """

    #: The exact raw frame this result preserves. 512000 little-endian uint16
    #: bytes, immutable, and the authority for every measured sample.
    raw: bytes
    #: The codes actually offered to fusion. Identical to :attr:`raw` except at
    #: temporally reconstructed samples, which carry the accepted result's code.
    codes: bytes
    #: One :data:`PROVENANCE_NAMES` byte per sample, in frame order.
    provenance: bytes
    #: ``{sample index: fusion weight}`` for temporally reconstructed samples
    #: only, held as a :class:`SparseWeights`. Sparse because it is the only
    #: provenance whose weight varies: measured is always
    #: :data:`MEASURED_WEIGHT` and the other two are always
    #: :data:`EXCLUDED_WEIGHT`, so storing those densely would be a megabyte of
    #: constants per frame. Any mapping may be handed over; what is kept is the
    #: compact encoding, whose cost is
    #: :data:`RECONSTRUCTION_ENTRY_BYTES` per entry exactly.
    weights: Mapping[int, float]
    #: The exact counts and total weight this frame offers.
    summary: WeightedFrameSummary
    #: Which accepted producer built it, from :data:`ADAPTERS`.
    adapter: str
    #: The accepted completion result every reconstruction here came from, or
    #: ``None`` for a frame that carries only raw measurement.
    completion: CompletedPatch | TemporalPatch | None = None
    #: Where the completed rectangle's first sample sits in the frame, as
    #: ``(u0, v0)``. Ignored - and required to be the origin - when there is no
    #: completion result.
    origin: tuple[int, int] = (0, 0)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if _exact_int(self.schema_version, "a weighted frame's schema version") != SCHEMA_VERSION:
            raise FrameUnusable(
                f"a weighted frame states schema version {self.schema_version!r}; this module "
                f"writes {SCHEMA_VERSION}"
            )
        if self.adapter not in ADAPTERS:
            raise FrameUnusable(
                f"a weighted frame names adapter {self.adapter!r}; this module implements "
                f"{list(ADAPTERS)} and refuses a producer it does not know rather than treating "
                "an unknown one as raw measurement"
            )
        # Copy every caller-owned buffer before anything is validated. A frozen
        # dataclass freezes the binding; the bytearray, memoryview or dict on
        # the other end of it stays the caller's unless it is copied here, and a
        # value that can change after validation was never validated.
        object.__setattr__(self, "raw", _frozen_bytes(self.raw, "a weighted frame's raw bytes"))
        object.__setattr__(
            self, "codes", _frozen_bytes(self.codes, "a weighted frame's offered codes")
        )
        object.__setattr__(
            self, "provenance", _frozen_bytes(self.provenance, "a weighted frame's provenance")
        )
        for name, buffer, expected in (
            ("raw bytes", self.raw, EXPECTED_FRAME_BYTES),
            ("offered codes", self.codes, EXPECTED_FRAME_BYTES),
            ("provenance", self.provenance, FRAME_SAMPLES),
        ):
            if len(buffer) != expected:
                raise FrameUnusable(
                    f"a weighted frame's {name} are {len(buffer)} byte(s) and one "
                    f"{FRAME_WIDTH}x{FRAME_HEIGHT} frame is {expected}; a frame that is not the "
                    "whole rectangle cannot be counted against it"
                )
        object.__setattr__(self, "weights", _frozen_weights(self.weights))
        origin = self.origin
        if (
            not isinstance(origin, tuple)
            or len(origin) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in origin)
        ):
            raise FrameUnusable(
                f"a weighted frame places its completed rectangle at {origin!r}, which is not a "
                "(u0, v0) pair of integers"
            )
        object.__setattr__(self, "origin", (int(origin[0]), int(origin[1])))
        if not isinstance(self.summary, WeightedFrameSummary):
            raise SummaryUnusable(
                f"a weighted frame carries summary {self.summary!r}, which is not a "
                "WeightedFrameSummary; a count published as loose data is a count nothing checked"
            )
        self._bind_to_source()

    # -- validation -------------------------------------------------------------

    def _bind_to_source(self) -> None:
        """Rebuild the whole frame from the raw bytes and the accepted result.

        The one check this type is for. Nothing here reads one of the frame's
        own labels as evidence: what each sample is allowed to be follows from
        :attr:`raw`, which the frame cannot change, and from :attr:`completion`,
        which validated and re-derived itself against its own immutable inputs.
        The offered codes, the provenance, the weights and the summary are then
        required to be that derivation exactly.
        """
        completion = self.completion
        expected_adapter = (
            ADAPTER_RAW
            if completion is None
            else ADAPTER_SPATIAL
            if isinstance(completion, CompletedPatch)
            else ADAPTER_TEMPORAL
        )
        if completion is not None and not isinstance(completion, CompletedPatch | TemporalPatch):
            raise SourceUnusable(
                f"a weighted frame names completion {completion!r}, which is neither an accepted "
                "CompletedPatch nor an accepted TemporalPatch. A reconstruction is carried "
                "through a validated result or it is an unchecked claim"
            )
        if self.adapter != expected_adapter:
            raise SourceUnusable(
                f"a weighted frame names adapter {self.adapter!r} and carries "
                f"{type(completion).__name__ if completion is not None else 'no'} completion "
                f"result; the adapter that applies here is {expected_adapter!r}. The adapter is "
                "what names which accepted module is the authority for this frame's provenance"
            )
        if completion is None:
            if self.origin != (0, 0):
                raise SourceUnusable(
                    f"a weighted frame carries no completion result and places a rectangle at "
                    f"{self.origin}; there is no rectangle to place"
                )
            patch_codes: tuple[int, ...] = ()
            states: bytes = b""
            result_codes: tuple[int, ...] = ()
            confidence: tuple[float, ...] = ()
            width = height = 0
        else:
            source = (
                completion.source
                if isinstance(completion, CompletedPatch)
                else completion.target.patch
            )
            patch_codes = source.codes
            states = completion.state_codes
            result_codes = completion.codes
            confidence = completion.confidence
            width, height = completion.width, completion.height
        u0, v0 = self.origin
        if u0 < 0 or v0 < 0 or u0 + width > FRAME_WIDTH or v0 + height > FRAME_HEIGHT:
            raise SourceUnusable(
                f"a {width}x{height} completed rectangle at ({u0}, {v0}) leaves the "
                f"{FRAME_WIDTH}x{FRAME_HEIGHT} frame it claims to be part of"
            )

        raw_codes = _codes_of(self.raw)
        expected_provenance = bytearray(_baseline_provenance(self.raw))
        expected_codes = bytearray(self.raw)
        expected_weights: dict[int, float] = {}

        for patch_index in range(width * height):
            row, column = divmod(patch_index, width)
            index = (v0 + row) * FRAME_WIDTH + (u0 + column)
            raw_code = raw_codes[index]
            if raw_code != patch_codes[patch_index]:
                raise SourceUnusable(
                    f"a weighted frame's raw sample {index} is code {raw_code} and the accepted "
                    f"completion result it carries was derived from code "
                    f"{patch_codes[patch_index]} there. A result about other bytes says nothing "
                    "about these"
                )
            state = states[patch_index]
            if state == depth_completion.STATE_MEASURED:
                # The raw code already stands in expected_codes, and the
                # baseline provenance already says measured. Nothing to write:
                # a measurement is preserved by not being touched.
                continue
            if state == depth_completion.STATE_MISSING:
                continue
            if state == depth_completion.STATE_SPATIALLY_RECONSTRUCTED:
                expected_provenance[index] = PROVENANCE_SPATIALLY_EXCLUDED
                continue
            # Only depth_temporal has a fourth state, and only a TemporalPatch
            # can be carrying one: the spatial result's own validation forbids
            # it. Reached through the accepted result, so the value, the
            # confidence and the support behind it are that module's.
            code = result_codes[patch_index]
            expected_codes[2 * index : 2 * index + 2] = struct.pack("<H", code)
            expected_provenance[index] = PROVENANCE_TEMPORALLY_RECONSTRUCTED
            expected_weights[index] = temporal_fusion_weight(float(confidence[patch_index]))

        if bytes(expected_codes) != self.codes:
            raise ProvenanceUnusable(
                "a weighted frame offers codes that are not the ones its raw frame and its "
                "accepted completion result imply. Measured samples are republished byte for "
                "byte, temporal reconstructions carry the accepted result's code, and everything "
                f"else offers {NO_RETURN_SAMPLE}"
            )
        if bytes(expected_provenance) != self.provenance:
            raise ProvenanceUnusable(
                "a weighted frame's provenance is not the provenance its raw frame and its "
                "accepted completion result imply. Provenance is derived from those two, never "
                "declared beside them"
            )
        if self.weights != expected_weights:
            raise ProvenanceUnusable(
                "a weighted frame's fusion weights are not the weights its provenance implies. "
                f"Only temporally reconstructed samples carry one, each exactly "
                f"{TEMPORAL_WEIGHT_CEILING} times the accepted confidence; a measured sample "
                f"weighs {MEASURED_WEIGHT} and everything else exactly {EXCLUDED_WEIGHT}"
            )
        self._bind_to_summary()

    def _bind_to_summary(self) -> None:
        """The published counts, against the provenance actually carried."""
        counted = summarise(self.provenance, self.weights)
        if counted != self.summary:
            raise SummaryUnusable(
                f"a weighted frame publishes {self.summary.as_dict()} and the provenance it "
                f"carries is {counted.as_dict()}. The summary is the frame counted, never the "
                "frame described"
            )

    # -- reading one sample -----------------------------------------------------

    @property
    def sample_count(self) -> int:
        return FRAME_SAMPLES

    def provenance_at(self, index: int) -> str:
        """Where one sample came from, by name."""
        return PROVENANCE_NAMES[self.provenance[index]]

    def weight_at(self, index: int) -> float:
        """One sample's fusion weight. Zero is absence, and is never negative."""
        state = self.provenance[index]
        if state == PROVENANCE_MEASURED:
            return MEASURED_WEIGHT
        if state == PROVENANCE_TEMPORALLY_RECONSTRUCTED:
            return self.weights[index]
        return EXCLUDED_WEIGHT

    @property
    def owned_bytes(self) -> int:
        """Every byte this frame owns, read off the buffers it owns.

        The three fixed buffers plus the reconstruction encoding: not a
        constant that ignores the sparse part, and not an estimate of the
        Python objects around them. At most :data:`MAX_FRAME_BYTES`, which is
        this number at the worst frame the contract allows - every sample a
        temporal reconstruction.
        """
        weights = self.weights
        sparse = (
            weights.owned_bytes
            if isinstance(weights, SparseWeights)
            else len(weights) * RECONSTRUCTION_ENTRY_BYTES
        )
        return len(self.raw) + len(self.codes) + len(self.provenance) + sparse

    def code_at(self, index: int) -> int:
        """The uint16 code offered to fusion at one sample."""
        return int.from_bytes(self.codes[2 * index : 2 * index + 2], "little")

    def as_dict(self) -> dict[str, object]:
        """A JSON-shaped account. Counts and provenance, never 256000 codes."""
        return {
            "schema_version": self.schema_version,
            "adapter": self.adapter,
            "origin": list(self.origin),
            "frame": {"width": FRAME_WIDTH, "height": FRAME_HEIGHT, "samples": FRAME_SAMPLES},
            "weighting": {
                "measured_weight": MEASURED_WEIGHT,
                "temporal_weight_ceiling": TEMPORAL_WEIGHT_CEILING,
                "excluded_weight": EXCLUDED_WEIGHT,
                "temporal_weight_is": (
                    "temporal_weight_ceiling times the accepted reconstruction confidence: "
                    "strictly increasing in it, strictly positive, and strictly below "
                    "measured_weight at every confidence scanner3d.depth_temporal can publish"
                ),
            },
            "summary": self.summary.as_dict(),
            "storage": {
                "owned_bytes": self.owned_bytes,
                "fixed_bytes": FIXED_FRAME_BYTES,
                "reconstruction_entry_bytes": RECONSTRUCTION_ENTRY_BYTES,
                "reconstruction_bytes": len(self.weights) * RECONSTRUCTION_ENTRY_BYTES,
                "max_owned_bytes": MAX_FRAME_BYTES,
                "owned_bytes_is": (
                    "the raw frame, the offered codes, one provenance byte per sample and "
                    "reconstruction_entry_bytes per temporal reconstruction, read off the "
                    "buffers themselves. Every byte this frame owns, at most max_owned_bytes, "
                    "which is the same count at a frame that is entirely reconstruction"
                ),
            },
        }


def _frozen_weights(value: object) -> SparseWeights:
    """A caller's weight mapping, copied into the compact immutable encoding.

    Every check lives in :class:`SparseWeights` itself, so a weight mapping
    that reached a frame by any route was checked by the same code.
    """
    if isinstance(value, SparseWeights):
        # Already the encoding, already checked, and immutable - so there is
        # nothing a caller could still be holding to copy away from.
        return value
    if not isinstance(value, Mapping):
        raise FrameUnusable(
            f"a weighted frame carries weights {value!r}, which is not a mapping of sample index "
            "to fusion weight"
        )
    return SparseWeights(value)


def summarise(provenance: bytes, weights: Mapping[int, float]) -> WeightedFrameSummary:
    """Count one frame's provenance. Bookkeeping, and no judgement at all.

    Four C-level passes over the provenance bytes and one :func:`math.fsum`
    over the reconstruction weights, so the cost is the frame and does not
    depend on what is in it.
    """
    measured = provenance.count(PROVENANCE_MEASURED)
    temporal = provenance.count(PROVENANCE_TEMPORALLY_RECONSTRUCTED)
    excluded = provenance.count(PROVENANCE_SPATIALLY_EXCLUDED)
    missing = provenance.count(PROVENANCE_MISSING)
    total = math.fsum((measured * MEASURED_WEIGHT, *weights.values()))
    return WeightedFrameSummary(
        measured=measured,
        temporally_reconstructed=temporal,
        spatially_excluded=excluded,
        missing=missing,
        total_weight=total,
    )


# --- the adapters -------------------------------------------------------------


def _build(
    raw: RawFrame,
    completion: CompletedPatch | TemporalPatch | None,
    *,
    adapter: str,
    origin: tuple[int, int],
) -> WeightedDepthFrame:
    """The one construction path, so a frame cannot be built two ways.

    Every adapter below derives the same three buffers here and then hands
    them to :class:`WeightedDepthFrame`, which re-derives them from the same
    inputs and refuses anything that does not match. The duplication is the
    point: an adapter that got it wrong is caught by the type rather than
    trusted because it is in this module.
    """
    raw_bytes = _frozen_bytes(raw, "a weighted frame's raw bytes")
    if len(raw_bytes) != EXPECTED_FRAME_BYTES:
        raise FrameUnusable(
            f"a {FRAME_WIDTH}x{FRAME_HEIGHT} depth frame is {EXPECTED_FRAME_BYTES} byte(s) and "
            f"this one is {len(raw_bytes)}"
        )
    provenance = bytearray(_baseline_provenance(raw_bytes))
    codes = bytearray(raw_bytes)
    weights: dict[int, float] = {}
    if completion is not None:
        u0, v0 = origin
        width, height = completion.width, completion.height
        if u0 < 0 or v0 < 0 or u0 + width > FRAME_WIDTH or v0 + height > FRAME_HEIGHT:
            raise SourceUnusable(
                f"a {width}x{height} completed rectangle at ({u0}, {v0}) leaves the "
                f"{FRAME_WIDTH}x{FRAME_HEIGHT} frame it would be part of"
            )
        for patch_index in range(width * height):
            row, column = divmod(patch_index, width)
            index = (v0 + row) * FRAME_WIDTH + (u0 + column)
            state = completion.state_codes[patch_index]
            if state == depth_completion.STATE_SPATIALLY_RECONSTRUCTED:
                # Excluded, and excluded ALL the way: the code offered to
                # fusion becomes a no-return, so a spatial hypothesis cannot
                # reach the field even through a defect in the weighting.
                provenance[index] = PROVENANCE_SPATIALLY_EXCLUDED
                codes[2 * index : 2 * index + 2] = struct.pack("<H", NO_RETURN_SAMPLE)
            elif state == depth_temporal.STATE_TEMPORALLY_RECONSTRUCTED:
                provenance[index] = PROVENANCE_TEMPORALLY_RECONSTRUCTED
                codes[2 * index : 2 * index + 2] = struct.pack("<H", completion.codes[patch_index])
                weights[index] = temporal_fusion_weight(float(completion.confidence[patch_index]))
    frozen_codes = bytes(codes)
    frozen_provenance = bytes(provenance)
    return WeightedDepthFrame(
        raw=raw_bytes,
        codes=frozen_codes,
        provenance=frozen_provenance,
        weights=weights,
        summary=summarise(frozen_provenance, weights),
        adapter=adapter,
        completion=completion,
        origin=origin,
    )


def weighted_raw_frame(raw: RawFrame) -> WeightedDepthFrame:
    """One raw frame as weighted input: every return full weight, every zero none.

    The identity adapter, and the one that makes the whole contract checkable:
    a frame built here fuses byte-for-byte identically to handing the same
    bytes to :meth:`scanner3d.tsdf.TsdfVolume.integrate`, because every
    measured sample weighs :data:`MEASURED_WEIGHT` and every no-return weighs
    :data:`EXCLUDED_WEIGHT`, which integration treats as absence.
    """
    return _build(raw, None, adapter=ADAPTER_RAW, origin=(0, 0))


def weighted_completed_frame(
    raw: RawFrame, result: CompletedPatch, *, origin: tuple[int, int] = (0, 0)
) -> WeightedDepthFrame:
    """An accepted single-frame completion as weighted input. It adds nothing.

    Deliberately, and this is the point of running it at all. Every spatial
    reconstruction :mod:`scanner3d.depth_completion` produces comes through as
    ``spatially_excluded`` at weight :data:`EXCLUDED_WEIGHT` with a no-return
    code, so the field this produces is identical to the field the raw frame
    produces - and the *count* of what was excluded is published rather than
    lost.

    Task 029's aperture-ambiguity case is the reason. A one-sample gap in a
    surface and a one-sample dropout on that surface present byte-identical
    neighbourhoods to a single frame, so a spatial estimate under that
    ambiguity cannot be told from invented geometry from inside the frame that
    made it. It becomes eligible for weight only when an independent view
    confirms it, at which point :mod:`scanner3d.depth_temporal` republishes it
    with that view's own value and support and it arrives here as a temporal
    reconstruction.
    """
    if not isinstance(result, CompletedPatch):
        raise SourceUnusable(
            f"a spatial weighted frame was offered {result!r}, which is not an accepted "
            "scanner3d.depth_completion.CompletedPatch"
        )
    return _build(raw, result, adapter=ADAPTER_SPATIAL, origin=origin)


def weighted_temporal_frame(
    raw: RawFrame, result: TemporalPatch, *, origin: tuple[int, int] = (0, 0)
) -> WeightedDepthFrame:
    """An accepted pose-compensated completion as weighted input.

    The only adapter that can add weight to a voxel the sensor did not reach.
    A sample :mod:`scanner3d.depth_temporal` accepted - other posed frames
    measured it, from enough distinct viewpoints, agreeing closely enough,
    through poses good enough - arrives as ``temporally_reconstructed`` at
    :func:`temporal_fusion_weight` of that module's own confidence.

    A spatial estimate the temporal stage never observed stays
    ``spatially_excluded`` and weighs nothing, exactly as it does through
    :func:`weighted_completed_frame`. That is the whole confirmation rule:
    :meth:`~scanner3d.depth_temporal.TemporalPatch.confirmation_at` says
    ``confirmed`` or ``contradicted`` only where independent evidence settled
    the sample, and in both of those cases the accepted result has already
    republished the sample with the INDEPENDENT value, confidence and support,
    which is what is weighted here. Where it says ``not_observed`` the sample
    is still the single-frame hypothesis and still weighs nothing.
    """
    if not isinstance(result, TemporalPatch):
        raise SourceUnusable(
            f"a temporal weighted frame was offered {result!r}, which is not an accepted "
            "scanner3d.depth_temporal.TemporalPatch"
        )
    return _build(raw, result, adapter=ADAPTER_TEMPORAL, origin=origin)
