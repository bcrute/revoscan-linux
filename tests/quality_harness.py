"""The deterministic reconstruction-quality harness, as one command.

``tests/test_quality.py`` proves the :mod:`scanner3d.quality` primitives are
load-bearing in Python. This is the other half: the known geometry lives on
disk as data (``tests/fixtures/quality/cases/``), one command runs all of it
through those same primitives and through the real
:mod:`scanner3d.reconstruction_readiness` reader, and the run leaves ONE strict
JSON artifact naming every case and every expected-versus-observed gate::

    .venv/bin/python -m tests.quality_harness --out quality-harness-report.json

Exit codes, kept apart so a caller can tell them apart:

* ``0`` - every case produced exactly the outcome its fixture declares.
* ``1`` - the harness ran and at least one case did not. An unexpected pass, a
  missing expected failure, an extra failure, a malformed fixture, an unusable
  measurement and an exception raised while measuring are all this.
* ``2`` - the harness could not run at all: the corpus is missing, its manifest
  is not the shape a manifest must be, a case file is not where the manifest
  says, the requested output path is not one a report may be written to, or the
  artifact could not be published.

Independent cases continue after a failure - including a case whose measurement
raises, which is recorded as malformed while every later case still runs. The
first broken case is rarely the informative one, and a report that stopped at it
would hide every other regression in the same change.

The requested ``--out`` path is cleared before any of that begins, so an
invocation that ends at exit 2 leaves no file there rather than the previous
run's report, which would otherwise be read as this one's answer.

**A deliberately bad fixture is a harness PASS only when the exact expected set
of gates fails.** Not "something failed": the whole point of the bad cases is
that a missing surface and an invented one fail *different* gates, and a check
that accepted any failure would pass a reconstruction that broke in a new way.

Nothing here touches a scanner, a capture, a display server, a GPU, the network
or the clock. The compute backend is the pure-Python reference, the corpus is
committed data, and two runs over an unchanged corpus produce byte-identical
JSON - which is why no duration and no machine identity appears anywhere in the
artifact. Measure the time beside the harness, not inside it.

This is a CORRECTNESS oracle over synthetic truth. It proves a change did not
break the geometry the harness measures. It makes no claim about the physical
accuracy of the scanner - see ``docs/quality-harness.md`` for what would.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import scanner3d.quality as quality
from scanner3d.errors import ScannerError
from scanner3d.geometry import Point
from scanner3d.reconstruction_readiness import (
    REASONS,
    ReadinessRecordUnusable,
    ReconstructionReadiness,
)

#: The corpus this harness runs, and the manifest that names its contents.
CORPUS: Final = Path(__file__).parent / "fixtures" / "quality" / "cases"
MANIFEST_NAME: Final = "manifest.json"

#: Schema of the fixtures read and of the artifact written. Both are refused if
#: they state anything else: a reader that guessed at a neighbouring version
#: would be reading fields that mean something different.
FIXTURE_SCHEMA_VERSION: Final = 1
CORPUS_KIND: Final = "scanner3d.quality-harness-corpus"
CASE_KIND: Final = "scanner3d.quality-harness-case"

#: Version 2 adds ``gate_bindings`` beside every case and holds every embedded
#: measurement to the exact schema of its own kind. A version-1 reader would
#: accept a measurement this one refuses, so the number moves rather than the
#: meaning of the old one.
REPORT_SCHEMA_VERSION: Final = 2
REPORT_KIND: Final = "scanner3d.quality-harness-report"

#: A fixture file name. No separator, no dot segment, no leading dash, so a
#: manifest cannot name a path outside the corpus however it is written.
FIXTURE_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*\.json$")

#: One step of a gate's path into its measurement. A plain key: no separator,
#: no dot segment, nothing empty, so a binding names a field of the
#: measurement beside it and can never be read as a path to somewhere else.
PATH_STEP = re.compile(r"^[a-z][a-z0-9_]*$")

#: Who read a frozen readiness record. Written into the measurement and
#: checked when it is read back, so an artifact cannot claim the real
#: authority read a record some other code produced.
READINESS_READ_BY: Final = "scanner3d.reconstruction_readiness.ReconstructionReadiness.from_dict"

#: Every measurement kind this harness emits. The report reader holds each one
#: to its own exact schema, so an unknown kind is a measurement nobody checked.
MEASUREMENT_KINDS: Final[tuple[str, ...]] = (
    "cloud_agreement",
    "plane_roughness",
    "dimension_errors",
    "landmark_offsets",
    "readiness_record",
)

#: How a case may declare its expected outcome.
OUTCOMES: Final = ("pass", "fail")

#: What every case status can be. "malformed" is deliberately NOT folded into
#: "fail": a fixture that could not be read is a different fact from a
#: reconstruction that failed a gate, and the artifact keeps them apart.
STATUSES: Final = ("pass", "fail", "malformed")

#: What this harness does not establish, carried in every artifact it writes.
LIMITATIONS: Final[tuple[str, ...]] = (
    "every case here is SYNTHETIC. The truth is constructed, not measured, so a passing run "
    "says a change did not break the geometry these instruments measure and says nothing about "
    "the physical accuracy of the scanner",
    "the readiness cases exercise the real strict reader over FROZEN records. They prove today's "
    "refusal rules still read those records the same way; they do not re-run a capture, a "
    "fusion or an export",
    "no duration, host, path or environment appears in this artifact. Two runs over an unchanged "
    "corpus are byte-identical, and a number that changed with the machine would not be",
    "a gate is the fixture's own stated limit beside the truth it governs. This harness owns no "
    "notion of good and applies no threshold the corpus does not state",
)

#: The comparison names a fixture may use, mapped to the authority's enum.
COMPARISONS: Final[Mapping[str, quality.Comparison]] = {
    "<=": quality.Comparison.AT_MOST,
    ">=": quality.Comparison.AT_LEAST,
    "is": quality.Comparison.IS,
}


class HarnessRefused(ScannerError):
    """The harness could not run at all. Exit 2, and no artifact is published."""


class FixtureUnusable(ScannerError):
    """One case fixture is not the shape a case must be. A harness failure."""


class ReportUnusable(ScannerError):
    """A harness artifact is not the shape this schema states."""


# --- strict reading helpers ---------------------------------------------------------


def _mapping(
    value: object, what: str, *, error: type[ScannerError] = FixtureUnusable
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error(f"{what} is not a JSON object")
    for key in value:
        if not isinstance(key, str):
            raise error(f"{what} has a non-string key {key!r}")
    return value


def _exact_keys(
    record: Mapping[str, object],
    allowed: Sequence[str],
    what: str,
    *,
    error: type[ScannerError] = FixtureUnusable,
) -> None:
    """Exactly these keys: no missing one, no unknown one.

    Unknown keys are refused rather than ignored because the ignored one is
    where a fixture quietly stops meaning what it says - a misspelled
    ``expected_failures`` beside a correct-looking file is a case that asserts
    nothing.
    """
    present, expected = set(record), set(allowed)
    missing, unknown = sorted(expected - present), sorted(present - expected)
    if missing or unknown:
        raise error(f"{what}: missing {missing}, unknown {unknown}")


def _text(record: Mapping[str, object], key: str, what: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise FixtureUnusable(f"{what} {key!r} is not a non-empty string")
    return value


def _bool(record: Mapping[str, object], key: str, what: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise FixtureUnusable(f"{what} {key!r} is {value!r}, which is not a boolean")
    return value


def _number(record: Mapping[str, object], key: str, what: str) -> float:
    return _as_number(record.get(key), f"{what} {key!r}")


def _as_number(value: object, what: str) -> float:
    """A real, finite number. A boolean is not one, whatever Python says.

    ``isinstance(True, int)`` is true, so a schema that accepted "a number"
    would accept ``true`` for a tolerance and compare it as 1.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FixtureUnusable(f"{what} is {value!r}, which is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise FixtureUnusable(f"{what} is {value!r}, which is not finite")
    return number


def _strings(record: Mapping[str, object], key: str, what: str) -> tuple[str, ...]:
    value = record.get(key)
    if not isinstance(value, list):
        raise FixtureUnusable(f"{what} {key!r} is not a list")
    for entry in value:
        if not isinstance(entry, str):
            raise FixtureUnusable(f"{what} {key!r} holds {entry!r}, which is not a string")
    return tuple(str(entry) for entry in value)


def _point(value: object, what: str) -> Point:
    if not isinstance(value, list) or len(value) != 3:
        raise FixtureUnusable(f"{what} is not a list of three coordinates")
    x, y, z = (_as_number(entry, f"{what} coordinate {index}") for index, entry in enumerate(value))
    return (x, y, z)


def _points(record: Mapping[str, object], key: str, what: str) -> list[Point]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise FixtureUnusable(f"{what} {key!r} is not a non-empty list of points")
    return [_point(entry, f"{what} {key!r} point {index}") for index, entry in enumerate(value)]


def _named_numbers(record: Mapping[str, object], key: str, what: str) -> dict[str, float]:
    entries = _mapping(record.get(key), f"{what} {key!r}")
    return {name: _as_number(value, f"{what} {key!r}[{name!r}]") for name, value in entries.items()}


def _named_points(record: Mapping[str, object], key: str, what: str) -> dict[str, Point]:
    entries = _mapping(record.get(key), f"{what} {key!r}")
    return {name: _point(value, f"{what} {key!r}[{name!r}]") for name, value in entries.items()}


def _loads(text: str, what: str, *, error: type[ScannerError] = FixtureUnusable) -> object:
    """Parse strictly: no NaN, no Infinity, no duplicate object keys.

    Python's decoder accepts all three - the first two are not JSON at all, and
    a duplicate key silently keeps the last one, which is how a document ends
    up asserting something other than what a reviewer read. This is the only
    place either can be caught: once an ordinary decoder has built a mapping,
    the discarded duplicate is gone and no validator over that mapping can see
    that it was ever there.
    """

    def _constant(token: str) -> float:
        raise error(
            f"{what} contains {token}, which is not JSON. Python's decoder accepts it by "
            "default; this one does not, because a non-finite value in evidence is a value no "
            "gate can be judged against"
        )

    def _object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise error(f"{what} names {key!r} twice in one object")
            seen.add(key)
        return dict(pairs)

    try:
        return json.loads(text, parse_constant=_constant, object_pairs_hook=_object_pairs)
    except json.JSONDecodeError as decode_error:
        raise error(f"{what} is not valid JSON: {decode_error}") from decode_error


# --- the corpus ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FixtureFile:
    """One case file, with the identity the manifest records for it."""

    name: str
    sha256: str
    bytes_: int
    case_id: str


@dataclass(frozen=True, slots=True)
class Corpus:
    """A manifest that has been checked against the directory beside it."""

    root: Path
    manifest_sha256: str
    files: tuple[FixtureFile, ...]


def _safe_child(root: Path, name: str) -> Path:
    """The file ``name`` inside ``root``, or a refusal.

    Three separate refusals, because they are three separate ways in: a name
    that is not a plain fixture name, a path that leaves the corpus once
    resolved, and a symlink - which passes both of the first two and can still
    point anywhere on the machine.
    """
    if not FIXTURE_NAME.match(name):
        raise HarnessRefused(
            f"{name!r} is not a fixture file name; a case is a plain lower-case .json file "
            "directly inside the corpus"
        )
    path = root / name
    if path.is_symlink():
        raise HarnessRefused(
            f"{name!r} is a symbolic link. A fixture is evidence and must be the bytes in the "
            "corpus, not a pointer to bytes somewhere else"
        )
    resolved = path.resolve()
    if resolved.parent != root.resolve():
        raise HarnessRefused(f"{name!r} resolves to {resolved}, which is outside {root}")
    if not resolved.is_file():
        raise HarnessRefused(f"{name!r} is named by the manifest but is not a file in {root}")
    return path


def load_corpus(root: Path = CORPUS) -> Corpus:
    """Read and check the manifest against what is actually on disk.

    Both directions. A manifest entry with no file behind it is a case that
    silently stopped running; a file the manifest does not name is a case
    nobody reviewed. Either one makes the corpus untrustworthy as a whole, so
    both are refusals rather than case failures.
    """
    if not root.is_dir():
        raise HarnessRefused(f"{root} is not a directory; there is no corpus to run")
    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise HarnessRefused(f"{manifest_path} is not a regular file")
    manifest_bytes = manifest_path.read_bytes()
    try:
        document = _mapping(
            _loads(manifest_bytes.decode("utf-8"), str(manifest_path)), str(manifest_path)
        )
        _exact_keys(
            document,
            ("schema_version", "kind", "what", "why", "units", "generated_by", "files"),
            str(manifest_path),
        )
        version = document.get("schema_version")
        if version != FIXTURE_SCHEMA_VERSION or isinstance(version, bool):
            raise FixtureUnusable(
                f"{manifest_path} states schema version {version!r}; this harness understands "
                f"exactly version {FIXTURE_SCHEMA_VERSION} and does not guess at another"
            )
        if _text(document, "kind", str(manifest_path)) != CORPUS_KIND:
            raise FixtureUnusable(f"{manifest_path} is not a {CORPUS_KIND}")
        entries = document.get("files")
        if not isinstance(entries, list) or not entries:
            raise FixtureUnusable(f"{manifest_path} names no files")
        files: list[FixtureFile] = []
        for index, entry in enumerate(entries):
            what = f"{manifest_path} entry {index}"
            record = _mapping(entry, what)
            _exact_keys(record, ("name", "case_id", "bytes", "sha256"), what)
            digest = _text(record, "sha256", what)
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise FixtureUnusable(f"{what} sha256 {digest!r} is not a hex digest")
            size = record.get("bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise FixtureUnusable(f"{what} bytes {size!r} is not a positive integer")
            files.append(
                FixtureFile(
                    name=_text(record, "name", what),
                    sha256=digest,
                    bytes_=size,
                    case_id=_text(record, "case_id", what),
                )
            )
    except FixtureUnusable as error:
        raise HarnessRefused(f"the corpus manifest is unusable: {error}") from error

    named = [entry.name for entry in files]
    duplicates = sorted({name for name in named if named.count(name) > 1})
    if duplicates:
        raise HarnessRefused(f"{manifest_path} names {duplicates} more than once")
    identifiers = [entry.case_id for entry in files]
    repeated = sorted({case for case in identifiers if identifiers.count(case) > 1})
    if repeated:
        raise HarnessRefused(
            f"{manifest_path} gives case id(s) {repeated} to more than one file; two cases with "
            "one name cannot both be reported"
        )
    for entry in files:
        _safe_child(root, entry.name)
    on_disk = sorted(
        path.name for path in root.iterdir() if path.name != MANIFEST_NAME and path.is_file()
    )
    unlisted = sorted(set(on_disk) - set(named))
    if unlisted:
        raise HarnessRefused(
            f"{root} holds {unlisted}, which the manifest does not name. A case file nobody "
            "listed is a case nobody reviewed"
        )
    return Corpus(
        root=root,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        files=tuple(sorted(files, key=lambda entry: entry.name)),
    )


# --- one case -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Expectation:
    """What a fixture declares its own outcome must be."""

    outcome: str
    failed_gates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateBinding:
    """Where one gate's ``measured`` value came from.

    :class:`scanner3d.quality.Gate` records the value and not its provenance,
    and that schema belongs to the authority rather than to this harness. So
    the binding lives beside the gate in the ARTIFACT: the measurement the
    fixture named and the path it walked into that measurement's own
    serialised form. A reader can then resolve the path itself and refuse a
    report whose gate value is not the value its measurement carries - which
    is the one thing a changed measurement beside an unchanged gate cannot
    survive.
    """

    gate: str
    measurement: str
    path: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"gate": self.gate, "measurement": self.measurement, "path": list(self.path)}


@dataclass(frozen=True, slots=True)
class ReadinessMeasurement:
    """A frozen readiness record, re-read by the real strict reader.

    The record is not summarised here and no rule of
    :mod:`scanner3d.reconstruction_readiness` is repeated: everything below is
    read off the :class:`~scanner3d.reconstruction_readiness.ReconstructionReadiness`
    that module's own reader produced. ``reason_present`` carries every reason
    code the authority defines, so a fixture can gate on a code being ABSENT -
    which is what stops "refused, for some reason" from counting as the
    refusal a fixture was written for.
    """

    name: str
    ready: bool
    reasons: tuple[str, ...]
    reason_present: Mapping[str, bool]
    surface_measured: bool | None
    summary: str

    @property
    def kind(self) -> str:
        return "readiness_record"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "kind": self.kind,
            "name": self.name,
            "ready": self.ready,
            "reasons": list(self.reasons),
            "reason_count": len(self.reasons),
            "reason_present": dict(self.reason_present),
            "surface_measured": self.surface_measured,
            "summary": self.summary,
            "read_by": READINESS_READ_BY,
        }


def _readiness_measurement(spec: Mapping[str, object], what: str) -> ReadinessMeasurement:
    _exact_keys(spec, ("kind", "name", "record"), what)
    name = _text(spec, "name", what)
    record = _mapping(spec.get("record"), f"{what} 'record'")
    try:
        readiness = ReconstructionReadiness.from_dict(record, what=f"{what} 'record'")
    except ReadinessRecordUnusable as error:
        raise FixtureUnusable(
            f"{what}: the real readiness reader refused this record: {error}"
        ) from error
    reasons = readiness.reasons
    return ReadinessMeasurement(
        name=name,
        ready=readiness.ready,
        reasons=reasons,
        reason_present={reason: reason in set(reasons) for reason in REASONS},
        surface_measured=None if readiness.surface is None else readiness.surface.measured,
        summary=readiness.summary(),
    )


def _measurement(spec: Mapping[str, object], what: str) -> quality.QualityMeasurement:
    """Build one measurement, through the authority that owns it.

    Every branch here calls straight into :mod:`scanner3d.quality` or
    :mod:`scanner3d.reconstruction_readiness`. The harness supplies the data
    and the caller's stated unit; it computes nothing itself, so a case cannot
    be passed by a scoring rule that lives here.
    """
    kind = _text(spec, "kind", what)
    if kind == "readiness_record":
        return _readiness_measurement(spec, what)
    if kind == "cloud_agreement":
        _exact_keys(
            spec, ("kind", "name", "unit", "coverage_tolerance", "observed", "reference"), what
        )
        return quality.cloud_agreement(
            _points(spec, "observed", what),
            _points(spec, "reference", what),
            name=_text(spec, "name", what),
            coverage_tolerance=_number(spec, "coverage_tolerance", what),
            unit=_text(spec, "unit", what),
        )
    if kind == "plane_roughness":
        _exact_keys(spec, ("kind", "name", "unit", "region", "points"), what)
        return quality.plane_roughness(
            _points(spec, "points", what),
            name=_text(spec, "name", what),
            region=_text(spec, "region", what),
            unit=_text(spec, "unit", what),
        )
    if kind == "dimension_errors":
        _exact_keys(spec, ("kind", "name", "unit", "truth_source", "measured", "truth"), what)
        return quality.dimension_errors(
            _named_numbers(spec, "measured", what),
            _named_numbers(spec, "truth", what),
            name=_text(spec, "name", what),
            truth_source=_text(spec, "truth_source", what),
            unit=_text(spec, "unit", what),
        )
    if kind == "landmark_offsets":
        _exact_keys(spec, ("kind", "name", "unit", "truth_source", "measured", "truth"), what)
        return quality.landmark_offsets(
            _named_points(spec, "measured", what),
            _named_points(spec, "truth", what),
            name=_text(spec, "name", what),
            truth_source=_text(spec, "truth_source", what),
            unit=_text(spec, "unit", what),
        )
    raise FixtureUnusable(
        f"{what} states measurement kind {kind!r}, which this harness does not know how to build"
    )


def _resolve(value: object, path: Sequence[str], what: str) -> object:
    """Walk a gate's declared path into a measurement's own serialised form.

    The path is read out of what the measurement serialises rather than off
    the object, so a gate names exactly what a reader of the artifact sees.
    """
    current = value
    for step in path:
        if not isinstance(current, Mapping) or step not in current:
            raise FixtureUnusable(f"{what}: no {step!r} at {list(path)}")
        current = current[step]
    return current


def _gate(
    spec: Mapping[str, object],
    measurements: Mapping[str, quality.QualityMeasurement],
    what: str,
) -> tuple[quality.Gate, GateBinding]:
    """One gate, built through :mod:`scanner3d.quality`'s own factories.

    Returns the gate and the :class:`GateBinding` that says where its value
    came from, so the artifact can carry the provenance the gate itself does
    not record.
    """
    comparison_name = _text(spec, "comparison", what)
    if comparison_name not in COMPARISONS:
        raise FixtureUnusable(
            f"{what} compares with {comparison_name!r}; this harness states "
            f"{sorted(COMPARISONS)} and refuses one it cannot evaluate"
        )
    comparison = COMPARISONS[comparison_name]
    if comparison is quality.Comparison.IS:
        _exact_keys(spec, ("name", "measurement", "path", "comparison", "limit", "context"), what)
    else:
        _exact_keys(
            spec, ("name", "measurement", "path", "comparison", "limit", "unit", "context"), what
        )
    name = _text(spec, "name", what)
    source = _text(spec, "measurement", what)
    if source not in measurements:
        raise FixtureUnusable(f"{what} reads measurement {source!r}, which the case does not make")
    path = _strings(spec, "path", what)
    if not path:
        raise FixtureUnusable(f"{what} states an empty path")
    for step in path:
        if not PATH_STEP.match(step):
            raise FixtureUnusable(
                f"{what} walks {step!r}, which is not a plain measurement field name. A gate "
                "reads a field of the measurement it names and nothing else"
            )
    measured = _resolve(measurements[source].as_dict(), path, what)
    binding = GateBinding(gate=name, measurement=source, path=path)
    context = _text(spec, "context", what)
    if comparison is quality.Comparison.IS:
        if not isinstance(measured, bool):
            raise FixtureUnusable(
                f"{what} compares {measured!r} with 'is', but it is not a boolean. An ordering "
                "belongs under <= or >="
            )
        return quality.must_be(name, measured, _bool(spec, "limit", what), context=context), binding
    unit = _text(spec, "unit", what)
    value = _as_number(measured, f"{what} measured value")
    limit = _number(spec, "limit", what)
    if comparison is quality.Comparison.AT_MOST:
        return quality.at_most(name, value, limit, unit=unit, context=context), binding
    return quality.at_least(name, value, limit, unit=unit, context=context), binding


@dataclass(frozen=True, slots=True)
class Case:
    """A fixture that has been read, validated and turned into real gates."""

    case_id: str
    what: str
    why: str
    expectation: Expectation
    report: quality.CaseReport
    #: One per gate, in the gates' own order.
    bindings: tuple[GateBinding, ...]


def load_case(path: Path, *, entry: FixtureFile) -> Case:
    """Read one fixture and evaluate it. Raises :class:`FixtureUnusable` only.

    The digest is checked first and against the manifest, so a case whose
    bytes changed is refused before its own claims about itself are read.
    """
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry.sha256 or len(raw) != entry.bytes_:
        raise FixtureUnusable(
            f"{path.name} is {len(raw)} byte(s) with sha256 {digest}; the manifest records "
            f"{entry.bytes_} byte(s) with sha256 {entry.sha256}. A fixture that has been edited "
            "is no longer the evidence that was reviewed"
        )
    what = path.name
    document = _mapping(_loads(raw.decode("utf-8"), what), what)
    _exact_keys(
        document,
        (
            "schema_version",
            "kind",
            "case_id",
            "what",
            "why",
            "expectation",
            "notes",
            "measurements",
            "gates",
        ),
        what,
    )
    version = document.get("schema_version")
    if version != FIXTURE_SCHEMA_VERSION or isinstance(version, bool):
        raise FixtureUnusable(
            f"{what} states schema version {version!r}; this harness understands exactly version "
            f"{FIXTURE_SCHEMA_VERSION}"
        )
    if _text(document, "kind", what) != CASE_KIND:
        raise FixtureUnusable(f"{what} is not a {CASE_KIND}")
    case_id = _text(document, "case_id", what)
    if case_id != entry.case_id:
        raise FixtureUnusable(
            f"{what} calls itself {case_id!r} and the manifest calls it {entry.case_id!r}"
        )

    expectation_record = _mapping(document.get("expectation"), f"{what} 'expectation'")
    _exact_keys(expectation_record, ("outcome", "failed_gates"), f"{what} 'expectation'")
    outcome = _text(expectation_record, "outcome", f"{what} 'expectation'")
    if outcome not in OUTCOMES:
        raise FixtureUnusable(f"{what} expects outcome {outcome!r}; it must be one of {OUTCOMES}")
    expected_failures = _strings(expectation_record, "failed_gates", f"{what} 'expectation'")
    if len(set(expected_failures)) != len(expected_failures):
        raise FixtureUnusable(f"{what} names an expected failing gate twice")
    if (outcome == "fail") != bool(expected_failures):
        raise FixtureUnusable(
            f"{what} expects {outcome!r} beside {len(expected_failures)} named failing gate(s). "
            "A failing case must say which gates fail, and a passing one must name none - "
            "otherwise 'something failed' would count as the expected failure"
        )

    specs = document.get("measurements")
    if not isinstance(specs, list) or not specs:
        raise FixtureUnusable(f"{what} makes no measurements")
    measurements: dict[str, quality.QualityMeasurement] = {}
    built: list[quality.QualityMeasurement] = []
    for index, spec in enumerate(specs):
        measurement = _measurement(
            _mapping(spec, f"{what} measurement {index}"), f"{what} measurement {index}"
        )
        if measurement.name in measurements:
            raise FixtureUnusable(f"{what} makes two measurements called {measurement.name!r}")
        measurements[measurement.name] = measurement
        built.append(measurement)

    gate_specs = document.get("gates")
    if not isinstance(gate_specs, list) or not gate_specs:
        raise FixtureUnusable(
            f"{what} states no gates. A case that applies no gate passes trivially and is not "
            "evidence of anything"
        )
    gates: list[quality.Gate] = []
    bindings: list[GateBinding] = []
    seen: set[str] = set()
    for index, spec in enumerate(gate_specs):
        gate, binding = _gate(
            _mapping(spec, f"{what} gate {index}"), measurements, f"{what} gate {index}"
        )
        if gate.name in seen:
            raise FixtureUnusable(
                f"{what} states two gates called {gate.name!r}; an expected failure naming one of "
                "them could not say which"
            )
        seen.add(gate.name)
        gates.append(gate)
        bindings.append(binding)
    unknown = sorted(set(expected_failures) - seen)
    if unknown:
        raise FixtureUnusable(
            f"{what} expects gate(s) {unknown} to fail, and states no such gate. An expectation "
            "naming a gate that does not exist can never be satisfied or violated"
        )

    return Case(
        case_id=case_id,
        what=_text(document, "what", what),
        why=_text(document, "why", what),
        expectation=Expectation(outcome=outcome, failed_gates=expected_failures),
        report=quality.CaseReport(
            name=case_id,
            measurements=tuple(built),
            gates=tuple(gates),
            notes=_strings(document, "notes", what),
        ),
        bindings=tuple(bindings),
    )


def _mismatch_lines(
    expected_outcome: str,
    expected_failures: Sequence[str],
    observed_outcome: str,
    observed_failures: Sequence[str],
) -> list[str]:
    """Every way an observed result differs from a declared one.

    All of them, not the first: "expected this to fail and it passed" and
    "something else failed too" are different findings and a reader needs
    both.

    One function, called by the harness when it writes a case and by the reader
    when it reads one back. That is deliberate: the mismatch prose is DERIVED
    from expected-versus-observed, so an artifact carrying invented findings -
    or carrying none where the two disagree - is refused rather than believed.
    """
    lines: list[str] = []
    if observed_outcome != expected_outcome:
        lines.append(f"expected the case to {expected_outcome} and it did {observed_outcome}")
    expected, observed = set(expected_failures), set(observed_failures)
    lines.extend(
        f"gate {name!r} was expected to fail and passed" for name in sorted(expected - observed)
    )
    lines.extend(
        f"gate {name!r} failed and was not expected to" for name in sorted(observed - expected)
    )
    return lines


def _mismatches(case: Case) -> list[str]:
    """The mismatches of one evaluated case."""
    return _mismatch_lines(
        case.expectation.outcome,
        case.expectation.failed_gates,
        "pass" if case.report.passed else "fail",
        [gate.name for gate in case.report.failures],
    )


# --- the artifact -------------------------------------------------------------------


def _case_entry(entry: FixtureFile, case: Case | None, error: str | None) -> dict[str, object]:
    if case is None:
        return {
            "case_id": entry.case_id,
            "fixture": {"name": entry.name, "sha256": entry.sha256, "bytes": entry.bytes_},
            "status": "malformed",
            "what": None,
            "why": None,
            "expected": None,
            "observed": None,
            "mismatches": [],
            "error": error,
            "quality": None,
            "gate_bindings": None,
        }
    mismatches = _mismatches(case)
    return {
        "case_id": case.case_id,
        "fixture": {"name": entry.name, "sha256": entry.sha256, "bytes": entry.bytes_},
        "status": "pass" if not mismatches else "fail",
        "what": case.what,
        "why": case.why,
        "expected": {
            "outcome": case.expectation.outcome,
            "failed_gates": sorted(case.expectation.failed_gates),
        },
        "observed": {
            "outcome": "pass" if case.report.passed else "fail",
            "failed_gates": sorted(gate.name for gate in case.report.failures),
        },
        "mismatches": mismatches,
        "error": None,
        "quality": case.report.as_dict(),
        "gate_bindings": [binding.as_dict() for binding in case.bindings],
    }


def run(root: Path = CORPUS) -> dict[str, object]:
    """Run every case in the corpus and return the artifact as a mapping.

    Cases are independent and every one of them is attempted: an exception out
    of a measurement is recorded against its own case and the next case still
    runs, so the artifact preserves all failures rather than the first.

    ``Exception`` and not one enumerated family of them. A measurement
    authority that raises something nobody anticipated - an ``OSError``, an
    ``ArithmeticError``, a ``TypeError`` from a refactor half-done - is still
    one case's problem, and stopping the run there would hide every later case
    behind it. ``BaseException`` is deliberately NOT caught: a
    ``KeyboardInterrupt`` or a ``SystemExit`` is the operator ending the run,
    not a case failing, and recording it as a malformed fixture would be a lie.
    """
    corpus = load_corpus(root)
    entries: list[dict[str, object]] = []
    for fixture in corpus.files:
        path = _safe_child(root, fixture.name)
        try:
            case = load_case(path, entry=fixture)
        except Exception as error:
            entries.append(_case_entry(fixture, None, f"{type(error).__name__}: {error}"))
            continue
        entries.append(_case_entry(fixture, case, None))

    statuses = [str(entry["status"]) for entry in entries]
    totals = {
        "cases": len(entries),
        "passed": statuses.count("pass"),
        "failed": statuses.count("fail"),
        "malformed": statuses.count("malformed"),
    }
    document: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "corpus": {
            "kind": CORPUS_KIND,
            "schema_version": FIXTURE_SCHEMA_VERSION,
            "manifest_sha256": corpus.manifest_sha256,
            "case_count": len(corpus.files),
        },
        "cases": entries,
        "totals": totals,
        "decision": "pass" if totals["passed"] == totals["cases"] and entries else "fail",
        "limitations": list(LIMITATIONS),
    }
    # Written through the same reader CI and a developer read it back with, so
    # an artifact this harness could not itself accept is never published.
    validate_report(document)
    return document


def loads_report(text: str, *, what: str = "harness report") -> Mapping[str, object]:
    """Read an artifact from its TEXT, strictly, and validate it.

    The entry point a consumer of a published report should use. Reading the
    text here rather than accepting an already-decoded mapping is the whole
    point: ``{"decision": "fail", "decision": "pass"}`` is one mapping by the
    time an ordinary decoder is done with it, and no validator over that
    mapping can tell that the file said both. ``NaN`` and ``Infinity`` are
    refused for the same reason - Python writes and reads them happily, and
    they are not JSON.
    """
    return validate_report(_loads(text, what, error=ReportUnusable), what=what)


def read_report(path: Path, *, what: str | None = None) -> Mapping[str, object]:
    """Read an artifact from disk through :func:`loads_report`."""
    where = what or str(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReportUnusable(f"{where} could not be read: {error}") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReportUnusable(f"{where} is not UTF-8 text: {error}") from error
    return loads_report(text, what=where)


def validate_report(document: object, *, what: str = "harness report") -> Mapping[str, object]:
    """Refuse an artifact that is not exactly this schema. Fail closed.

    Every check here is one a corrupted, hand-edited or deliberately forged
    report could otherwise pass. The rule throughout is that nothing a report
    ASSERTS about itself is believed where the report also carries what that
    assertion is derived from: a gate's verdict is recomputed from its own
    measured value, comparison and limit; a case's ``passed`` and
    ``gate_count`` from its gates; its observed outcome and failing set from
    the gates it reports; its mismatch prose and status from expected versus
    observed; and the totals and decision from the cases. A statement that
    only agreed with itself at the top level is exactly the shape a forged
    artifact takes.
    """
    record = _mapping(document, what, error=ReportUnusable)
    try:
        _exact_keys(
            record,
            ("schema_version", "kind", "corpus", "cases", "totals", "decision", "limitations"),
            what,
            error=ReportUnusable,
        )
        version = record.get("schema_version")
        if version != REPORT_SCHEMA_VERSION or isinstance(version, bool):
            raise ReportUnusable(f"{what} states schema version {version!r}")
        if record.get("kind") != REPORT_KIND:
            raise ReportUnusable(f"{what} states kind {record.get('kind')!r}, not {REPORT_KIND!r}")
        if record.get("limitations") != list(LIMITATIONS):
            raise ReportUnusable(
                f"{what} does not carry this version's limitations; they are the artifact's own "
                "statement of what it does not establish and are not optional"
            )
        corpus = _validate_corpus(record.get("corpus"), what)
        cases = record.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ReportUnusable(f"{what} carries no cases")
        seen: set[str] = set()
        counts = {"pass": 0, "fail": 0, "malformed": 0}
        for index, entry in enumerate(cases):
            case = _mapping(entry, f"{what} case {index}", error=ReportUnusable)
            _exact_keys(
                case,
                (
                    "case_id",
                    "fixture",
                    "status",
                    "what",
                    "why",
                    "expected",
                    "observed",
                    "mismatches",
                    "error",
                    "quality",
                    "gate_bindings",
                ),
                f"{what} case {index}",
                error=ReportUnusable,
            )
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ReportUnusable(f"{what} case {index} has no case id")
            if case_id in seen:
                raise ReportUnusable(f"{what} reports case {case_id!r} twice")
            seen.add(case_id)
            status = case.get("status")
            if status not in STATUSES:
                raise ReportUnusable(f"{what} case {case_id!r} has status {status!r}")
            counts[str(status)] += 1
            _validate_case_body(case, case_id, what)
        totals = _mapping(record.get("totals"), f"{what} 'totals'", error=ReportUnusable)
        _exact_keys(
            totals,
            ("cases", "passed", "failed", "malformed"),
            f"{what} 'totals'",
            error=ReportUnusable,
        )
        stated = {
            "cases": _count(totals, "cases", what),
            "passed": _count(totals, "passed", what),
            "failed": _count(totals, "failed", what),
            "malformed": _count(totals, "malformed", what),
        }
        derived = {
            "cases": len(cases),
            "passed": counts["pass"],
            "failed": counts["fail"],
            "malformed": counts["malformed"],
        }
        if stated != derived:
            raise ReportUnusable(f"{what} states totals {stated} beside cases that are {derived}")
        if stated["cases"] != _count(corpus, "case_count", f"{what} 'corpus'"):
            raise ReportUnusable(f"{what} reports fewer cases than the corpus it names")
        decision = record.get("decision")
        if decision not in OUTCOMES:
            raise ReportUnusable(f"{what} states decision {decision!r}")
        if (decision == "pass") != (stated["passed"] == stated["cases"]):
            raise ReportUnusable(
                f"{what} decides {decision!r} with {stated['passed']} of {stated['cases']} "
                "case(s) passing; a decision that contradicts its own totals is refused"
            )
    except FixtureUnusable as error:  # pragma: no cover - helpers reused across both readers
        raise ReportUnusable(str(error)) from error
    return record


def _validate_corpus(value: object, what: str) -> Mapping[str, object]:
    """The corpus the report names, checked rather than taken on trust.

    A forged kind, version or manifest digest is how a report claims to be
    evidence about a corpus it never ran, so each is refused on its own terms
    rather than merely being present.
    """
    where = f"{what} 'corpus'"
    corpus = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        corpus,
        ("kind", "schema_version", "manifest_sha256", "case_count"),
        where,
        error=ReportUnusable,
    )
    if corpus.get("kind") != CORPUS_KIND:
        raise ReportUnusable(f"{where} states kind {corpus.get('kind')!r}, not {CORPUS_KIND!r}")
    version = corpus.get("schema_version")
    if version != FIXTURE_SCHEMA_VERSION or isinstance(version, bool):
        raise ReportUnusable(
            f"{where} states fixture schema version {version!r}; this report can only have been "
            f"produced from version {FIXTURE_SCHEMA_VERSION}"
        )
    _digest(corpus, "manifest_sha256", where)
    _count(corpus, "case_count", where)
    return corpus


def _count(record: Mapping[str, object], key: str, what: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ReportUnusable(f"{what} {key!r} is {value!r}, which is not a count")
    return value


def _digest(record: Mapping[str, object], key: str, what: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ReportUnusable(f"{what} {key!r} is {value!r}, which is not a sha256")
    return value


def _report_text(record: Mapping[str, object], key: str, what: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReportUnusable(f"{what} {key!r} is {value!r}, which is not a non-empty string")
    return value


def _report_finite(value: object, what: str) -> float:
    """A real, finite number. ``True`` is not one, whatever Python says."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReportUnusable(f"{what} is {value!r}, which is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ReportUnusable(f"{what} is {value!r}, which is not finite")
    return number


def _no_nonfinite(value: object, what: str) -> None:
    """Walk a subtree and refuse any non-finite number anywhere in it.

    ``allow_nan=False`` stops one being written and :func:`loads_report` stops
    one being read from text, so this is for the third way in: a mapping handed
    straight to the validator, which is how the harness checks its own document
    before it is ever serialised.
    """
    if isinstance(value, Mapping):
        for key, entry in value.items():
            _no_nonfinite(entry, f"{what}[{key!r}]")
    elif isinstance(value, list):
        for index, entry in enumerate(value):
            _no_nonfinite(entry, f"{what}[{index}]")
    elif not isinstance(value, bool) and isinstance(value, float) and not math.isfinite(value):
        raise ReportUnusable(f"{what} is {value!r}, which is not finite")


def _outcome_block(value: object, where: str) -> tuple[str, list[str]]:
    """One ``{outcome, failed_gates}`` block, in its one canonical form.

    Canonical because the harness writes exactly one form of it: an outcome
    from :data:`OUTCOMES`, a sorted list of distinct non-empty gate names, and
    ``fail`` if and only if that list is non-empty. Anything else - ``null``, a
    duplicate, an unsorted list, ``pass`` beside named failures - is a block
    this harness did not write.
    """
    block = _mapping(value, where, error=ReportUnusable)
    _exact_keys(block, ("outcome", "failed_gates"), where, error=ReportUnusable)
    outcome = block.get("outcome")
    if outcome not in OUTCOMES:
        raise ReportUnusable(f"{where} states outcome {outcome!r}; it must be one of {OUTCOMES}")
    failed = block.get("failed_gates")
    if not isinstance(failed, list):
        raise ReportUnusable(f"{where} 'failed_gates' is {failed!r}, which is not a list")
    names: list[str] = []
    for entry in failed:
        if not isinstance(entry, str) or not entry.strip():
            raise ReportUnusable(f"{where} 'failed_gates' holds {entry!r}, which is not a name")
        names.append(entry)
    if len(set(names)) != len(names):
        raise ReportUnusable(
            f"{where} names a failing gate more than once: {names}. One gate fails once, and a "
            "repeated name makes a failure set look larger than it is"
        )
    if names != sorted(names):
        raise ReportUnusable(f"{where} names {names}, which is not the sorted form this writes")
    if (outcome == "fail") != bool(names):
        raise ReportUnusable(
            f"{where} states outcome {outcome!r} beside {len(names)} failing gate(s); a failing "
            "result names them and a passing one names none"
        )
    return str(outcome), names


#: How close a derived number has to be to the one the artifact carries. The
#: report's values were produced by real floating-point arithmetic, so a
#: derivation that repeats it in a different order lands within rounding and
#: not on the same bits. Wide enough to survive that, far too narrow to admit
#: a value anybody edited.
_TOLERANCE: Final = 1e-9


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=_TOLERANCE, abs_tol=1e-12)


def _report_number(record: Mapping[str, object], key: str, what: str) -> float:
    return _report_finite(record.get(key), f"{what} {key!r}")


def _report_bool(record: Mapping[str, object], key: str, what: str) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise ReportUnusable(f"{what} {key!r} is {value!r}, which is not a boolean")
    return value


def _positive_count(record: Mapping[str, object], key: str, what: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ReportUnusable(f"{what} {key!r} is {value!r}, which is not a positive count")
    return value


def _report_point(value: object, what: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ReportUnusable(f"{what} is {value!r}, which is not three coordinates")
    x, y, z = (_report_finite(entry, f"{what}[{index}]") for index, entry in enumerate(value))
    return (x, y, z)


def _report_strings(value: object, what: str) -> list[str]:
    if not isinstance(value, list):
        raise ReportUnusable(f"{what} is {value!r}, which is not a list")
    names: list[str] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, str) or not entry.strip():
            raise ReportUnusable(f"{what}[{index}] is {entry!r}, which is not a name")
        names.append(entry)
    return names


def _entries(record: Mapping[str, object], key: str, what: str) -> list[object]:
    value = record.get(key)
    if not isinstance(value, list) or not value:
        raise ReportUnusable(f"{what} {key!r} is {value!r}, which is not a non-empty list")
    return list(value)


def _sorted_distinct(names: Sequence[str], what: str) -> None:
    """Named entries appear once each, in the order the authority writes them.

    Both measurements that carry named entries build them from
    ``tuple(sorted(truth))``, so a repeated name or an out-of-order one is a
    list this harness cannot have produced - and a repeated name is how a
    summary of N entries could be made to describe a different N.
    """
    if len(set(names)) != len(names):
        raise ReportUnusable(f"{what} names an entry more than once: {list(names)}")
    if list(names) != sorted(names):
        raise ReportUnusable(
            f"{what} names {list(names)}, which is not the sorted form this writes"
        )


# --- the exact schema of every measurement kind this harness emits -------------------


def _validate_distribution(
    value: object, where: str, *, count: int, unit: str
) -> Mapping[str, object]:
    """One :class:`scanner3d.quality.Distribution`, held to what it summarises.

    ``count`` is not read and believed either: it is the number of values the
    measurement around this block was computed from, so a summary describing a
    different population is refused. The order invariants below hold for any
    set of magnitudes whatever - median at or under p95, p95 at or under the
    worst, and a root mean square that cannot exceed the largest magnitude in
    it - so a value edited upward or downward on its own breaks one of them.
    """
    block = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        block, ("count", "median", "rms", "p95", "worst", "unit"), where, error=ReportUnusable
    )
    stated = _positive_count(block, "count", where)
    if stated != count:
        raise ReportUnusable(
            f"{where} summarises {stated} value(s) beside the {count} the measurement around it "
            "reports having"
        )
    stated_unit = _report_text(block, "unit", where)
    if stated_unit != unit:
        raise ReportUnusable(f"{where} is in {stated_unit!r} and its measurement is in {unit!r}")
    median = _report_number(block, "median", where)
    rms = _report_number(block, "rms", where)
    p95 = _report_number(block, "p95", where)
    worst = _report_number(block, "worst", where)
    for key, magnitude in (("median", median), ("rms", rms), ("p95", p95), ("worst", worst)):
        if magnitude < 0.0:
            raise ReportUnusable(
                f"{where} {key!r} is {magnitude!r}; this block summarises magnitudes, and a "
                "negative one is not a magnitude"
            )
    if not median <= p95 <= worst:
        raise ReportUnusable(
            f"{where} orders median {median!r}, p95 {p95!r} and worst {worst!r}, which no set of "
            "magnitudes produces"
        )
    if rms > worst * (1.0 + _TOLERANCE) + _TOLERANCE:
        raise ReportUnusable(
            f"{where} states rms {rms!r} above its worst value {worst!r}; a root mean square "
            "cannot exceed the largest magnitude in it"
        )
    if stated == 1 and not (_close(median, worst) and _close(p95, worst) and _close(rms, worst)):
        raise ReportUnusable(
            f"{where} summarises a single value as {median!r}, {rms!r}, {p95!r} and {worst!r}, "
            "which one value cannot be"
        )
    return block


def _validate_directed(
    value: object, where: str, *, direction: str, unit: str
) -> tuple[int, int, float]:
    """One :class:`scanner3d.quality.DirectedDistance`.

    ``coverage_fraction`` is a count over ``query_count`` by construction, so
    it must land on a whole number of queries. A fraction adjusted to move a
    coverage gate almost never does.
    """
    block = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        block,
        (
            "direction",
            "query_count",
            "target_count",
            "distances",
            "coverage_fraction",
            "coverage_tolerance",
            "unit",
        ),
        where,
        error=ReportUnusable,
    )
    if block.get("direction") != direction:
        raise ReportUnusable(
            f"{where} calls itself {block.get('direction')!r} and is carried as {direction!r}"
        )
    queries = _positive_count(block, "query_count", where)
    targets = _positive_count(block, "target_count", where)
    stated_unit = _report_text(block, "unit", where)
    if stated_unit != unit:
        raise ReportUnusable(f"{where} is in {stated_unit!r} and its measurement is in {unit!r}")
    _validate_distribution(block.get("distances"), f"{where} 'distances'", count=queries, unit=unit)
    fraction = _report_number(block, "coverage_fraction", where)
    if not 0.0 <= fraction <= 1.0:
        raise ReportUnusable(
            f"{where} states coverage fraction {fraction!r}, which is not in [0, 1]"
        )
    covered = fraction * queries
    if abs(covered - round(covered)) > _TOLERANCE:
        raise ReportUnusable(
            f"{where} states coverage {fraction!r} of {queries} query point(s), which is "
            f"{covered!r} points and not a whole number of them"
        )
    tolerance = _report_number(block, "coverage_tolerance", where)
    if tolerance <= 0.0:
        raise ReportUnusable(
            f"{where} states coverage tolerance {tolerance!r}; the authority refuses a "
            "non-positive one, so this measurement cannot have been made with it"
        )
    return queries, targets, tolerance


def _validate_cloud_agreement(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "observed_to_reference",
            "reference_to_observed",
            "search_gate",
            "resolved_backend",
            "unit",
        ),
        where,
        error=ReportUnusable,
    )
    unit = _report_text(record, "unit", where)
    _report_text(record, "resolved_backend", where)
    gate = _report_number(record, "search_gate", where)
    if gate <= 0.0:
        raise ReportUnusable(f"{where} states a search gate of {gate!r}, which spans nothing")
    forward = _validate_directed(
        record.get("observed_to_reference"),
        f"{where} 'observed_to_reference'",
        direction="observed_to_reference",
        unit=unit,
    )
    backward = _validate_directed(
        record.get("reference_to_observed"),
        f"{where} 'reference_to_observed'",
        direction="reference_to_observed",
        unit=unit,
    )
    if (forward[0], forward[1]) != (backward[1], backward[0]):
        raise ReportUnusable(
            f"{where} measures {forward[0]} against {forward[1]} point(s) one way and "
            f"{backward[0]} against {backward[1]} the other; one pair of clouds has one size"
        )
    if forward[2] != backward[2]:
        raise ReportUnusable(
            f"{where} states coverage tolerances {forward[2]!r} and {backward[2]!r}; one "
            "measurement is made at one tolerance"
        )
    for key in ("observed_to_reference", "reference_to_observed"):
        directed = _mapping(record.get(key), f"{where} {key!r}", error=ReportUnusable)
        distances = _mapping(directed.get("distances"), f"{where} {key!r}", error=ReportUnusable)
        worst = _report_number(distances, "worst", f"{where} {key!r} 'distances'")
        if worst > gate * (1.0 + _TOLERANCE):
            raise ReportUnusable(
                f"{where} {key!r} reports a nearest neighbour {worst!r} away, outside the "
                f"{gate!r} search gate every one of them was found inside"
            )


def _validate_plane_roughness(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "region",
            "point_count",
            "normal",
            "offset",
            "residuals",
            "signed_min",
            "signed_max",
            "unit",
            "limitation",
        ),
        where,
        error=ReportUnusable,
    )
    unit = _report_text(record, "unit", where)
    _report_text(record, "region", where)
    count = _positive_count(record, "point_count", where)
    if count < 3:
        raise ReportUnusable(
            f"{where} fits a plane through {count} point(s); the authority requires three"
        )
    normal = _report_point(record.get("normal"), f"{where} 'normal'")
    length = math.sqrt(math.fsum(axis * axis for axis in normal))
    if not _close(length, 1.0):
        raise ReportUnusable(
            f"{where} states a normal of length {length!r}; a fitted plane normal is a unit vector"
        )
    _report_number(record, "offset", where)
    low = _report_number(record, "signed_min", where)
    high = _report_number(record, "signed_max", where)
    if low > high:
        raise ReportUnusable(f"{where} states signed residuals from {low!r} to {high!r}")
    residuals = _validate_distribution(
        record.get("residuals"), f"{where} 'residuals'", count=count, unit=unit
    )
    worst = _report_number(residuals, "worst", f"{where} 'residuals'")
    if not _close(worst, max(abs(low), abs(high))):
        raise ReportUnusable(
            f"{where} states a worst residual of {worst!r} beside signed extremes {low!r} and "
            f"{high!r}; the residuals are the magnitudes of the signed values"
        )
    if record.get("limitation") != quality.PLANE_LIMITATION:
        raise ReportUnusable(
            f"{where} does not carry the authority's own plane limitation, which travels with "
            "every roughness measurement and is not optional"
        )


def _validate_dimension_errors(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "entries",
            "absolute",
            "relative",
            "unit",
            "truth_source",
        ),
        where,
        error=ReportUnusable,
    )
    unit = _report_text(record, "unit", where)
    _report_text(record, "truth_source", where)
    names: list[str] = []
    absolutes: list[float] = []
    relatives: list[float] = []
    for index, value in enumerate(_entries(record, "entries", where)):
        at = f"{where} entry {index}"
        entry = _mapping(value, at, error=ReportUnusable)
        _exact_keys(
            entry,
            ("name", "measured", "truth", "error", "absolute_error", "relative_error"),
            at,
            error=ReportUnusable,
        )
        names.append(_report_text(entry, "name", at))
        measured = _report_number(entry, "measured", at)
        truth = _report_number(entry, "truth", at)
        error = _report_number(entry, "error", at)
        absolute = _report_number(entry, "absolute_error", at)
        if not _close(error, measured - truth):
            raise ReportUnusable(
                f"{at} states error {error!r} for {measured!r} against truth {truth!r}"
            )
        if absolute != abs(error):
            raise ReportUnusable(f"{at} states absolute error {absolute!r} beside error {error!r}")
        absolutes.append(absolute)
        relative = entry.get("relative_error")
        if truth == 0.0:
            if relative is not None:
                raise ReportUnusable(
                    f"{at} states relative error {relative!r} against a truth of zero, which the "
                    "authority reports as null rather than inventing a number for"
                )
        else:
            value_ = _report_finite(relative, f"{at} 'relative_error'")
            if not _close(value_, error / truth):
                raise ReportUnusable(
                    f"{at} states relative error {value_!r} for error {error!r} against truth "
                    f"{truth!r}"
                )
            relatives.append(abs(value_))
    _sorted_distinct(names, f"{where} 'entries'")
    absolute_block = _validate_distribution(
        record.get("absolute"), f"{where} 'absolute'", count=len(names), unit=unit
    )
    worst = _report_number(absolute_block, "worst", f"{where} 'absolute'")
    if not _close(worst, max(absolutes)):
        raise ReportUnusable(
            f"{where} 'absolute' states a worst error of {worst!r} beside entries whose largest "
            f"is {max(absolutes)!r}"
        )
    relative_block = record.get("relative")
    if len(relatives) != len(names):
        if relative_block is not None:
            raise ReportUnusable(
                f"{where} summarises relative error while {len(names) - len(relatives)} entry/ies "
                "have none; the authority summarises it only when every truth is non-zero"
            )
        return
    if relative_block is None:
        raise ReportUnusable(
            f"{where} gives every entry a relative error and summarises none of them"
        )
    summary = _validate_distribution(
        relative_block, f"{where} 'relative'", count=len(names), unit=quality.FRACTION_UNIT
    )
    relative_worst = _report_number(summary, "worst", f"{where} 'relative'")
    if not _close(relative_worst, max(relatives)):
        raise ReportUnusable(
            f"{where} 'relative' states a worst error of {relative_worst!r} beside entries whose "
            f"largest is {max(relatives)!r}"
        )


def _validate_landmark_offsets(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        ("schema_version", "kind", "name", "entries", "offsets", "unit", "truth_source"),
        where,
        error=ReportUnusable,
    )
    unit = _report_text(record, "unit", where)
    _report_text(record, "truth_source", where)
    names: list[str] = []
    offsets: list[float] = []
    for index, value in enumerate(_entries(record, "entries", where)):
        at = f"{where} entry {index}"
        entry = _mapping(value, at, error=ReportUnusable)
        _exact_keys(entry, ("name", "measured", "truth", "offset"), at, error=ReportUnusable)
        names.append(_report_text(entry, "name", at))
        measured = _report_point(entry.get("measured"), f"{at} 'measured'")
        truth = _report_point(entry.get("truth"), f"{at} 'truth'")
        offset = _report_number(entry, "offset", at)
        if offset < 0.0:
            raise ReportUnusable(f"{at} states a negative offset {offset!r}")
        # The two points are both in the artifact, so the distance between
        # them is derivable from it rather than being independent truth.
        if not _close(offset, math.dist(measured, truth)):
            raise ReportUnusable(
                f"{at} states offset {offset!r} between {list(measured)} and {list(truth)}, which "
                f"are {math.dist(measured, truth)!r} apart"
            )
        offsets.append(offset)
    _sorted_distinct(names, f"{where} 'entries'")
    summary = _validate_distribution(
        record.get("offsets"), f"{where} 'offsets'", count=len(names), unit=unit
    )
    worst = _report_number(summary, "worst", f"{where} 'offsets'")
    if not _close(worst, max(offsets)):
        raise ReportUnusable(
            f"{where} 'offsets' states a worst offset of {worst!r} beside entries whose largest "
            f"is {max(offsets)!r}"
        )


def _validate_readiness_record(record: Mapping[str, object], where: str) -> None:
    _exact_keys(
        record,
        (
            "schema_version",
            "kind",
            "name",
            "ready",
            "reasons",
            "reason_count",
            "reason_present",
            "surface_measured",
            "summary",
            "read_by",
        ),
        where,
        error=ReportUnusable,
    )
    ready = _report_bool(record, "ready", where)
    reasons = _report_strings(record.get("reasons"), f"{where} 'reasons'")
    unknown = sorted(set(reasons) - set(REASONS))
    if unknown:
        raise ReportUnusable(
            f"{where} names reason code(s) {unknown}, which scanner3d.reconstruction_readiness "
            "does not define"
        )
    if len(set(reasons)) != len(reasons):
        raise ReportUnusable(f"{where} names a reason code more than once: {reasons}")
    canonical = [reason for reason in REASONS if reason in set(reasons)]
    if reasons != canonical:
        raise ReportUnusable(
            f"{where} lists reasons {reasons}, which is not the canonical order {canonical} the "
            "authority publishes"
        )
    stated_count = record.get("reason_count")
    if stated_count != len(reasons) or isinstance(stated_count, bool):
        raise ReportUnusable(
            f"{where} states {stated_count!r} reason(s) and carries {len(reasons)}"
        )
    if ready != (not reasons):
        raise ReportUnusable(
            f"{where} states ready {ready!r} beside {len(reasons)} reason(s); readiness is "
            "exactly 'no gate failed'"
        )
    present = _mapping(
        record.get("reason_present"), f"{where} 'reason_present'", error=ReportUnusable
    )
    _exact_keys(present, REASONS, f"{where} 'reason_present'", error=ReportUnusable)
    for reason in REASONS:
        flag = _report_bool(present, reason, f"{where} 'reason_present'")
        if flag != (reason in set(reasons)):
            raise ReportUnusable(
                f"{where} 'reason_present' says {reason!r} is {flag!r} beside reasons {reasons}"
            )
    surface = record.get("surface_measured")
    if surface is not None and not isinstance(surface, bool):
        raise ReportUnusable(
            f"{where} states surface_measured {surface!r}, which is neither a boolean nor null"
        )
    expected_summary = (
        "readiness: ready, every applicable gate passed"
        if ready
        else f"readiness: REFUSED ({', '.join(reasons)})"
    )
    if record.get("summary") != expected_summary:
        raise ReportUnusable(
            f"{where} summarises itself as {record.get('summary')!r}; its own verdict and reasons "
            f"give {expected_summary!r}"
        )
    if record.get("read_by") != READINESS_READ_BY:
        raise ReportUnusable(
            f"{where} says it was read by {record.get('read_by')!r}, not {READINESS_READ_BY!r}"
        )


_MEASUREMENT_SCHEMAS: Final[Mapping[str, object]] = {
    "cloud_agreement": _validate_cloud_agreement,
    "plane_roughness": _validate_plane_roughness,
    "dimension_errors": _validate_dimension_errors,
    "landmark_offsets": _validate_landmark_offsets,
    "readiness_record": _validate_readiness_record,
}


def _validate_measurement(value: object, where: str) -> tuple[str, Mapping[str, object]]:
    """One embedded measurement, held to the exact schema of its own kind.

    Not "it has a kind and a name". A measurement is the evidence a gate's
    value came from, and a reader that accepted an unknown field, a missing
    one or a changed number inside it would accept an artifact whose gates no
    longer follow from anything. Every field of every kind this harness emits
    is named here, with the invariants that kind's own construction
    guarantees - counts against the population they summarise, fractions
    against the number of queries they count, orders that any set of
    magnitudes satisfies, and the values one field derives from another.
    """
    record = _mapping(value, where, error=ReportUnusable)
    kind = _report_text(record, "kind", where)
    if kind not in MEASUREMENT_KINDS:
        raise ReportUnusable(
            f"{where} states measurement kind {kind!r}; this schema states "
            f"{list(MEASUREMENT_KINDS)} and refuses one it has no schema for"
        )
    name = _report_text(record, "name", where)
    # The readiness measurement is this harness's own view of a record the
    # readiness authority read; the other four are the quality authority's.
    expected_version = (
        FIXTURE_SCHEMA_VERSION if kind == "readiness_record" else quality.SCHEMA_VERSION
    )
    version = record.get("schema_version")
    if version != expected_version or isinstance(version, bool):
        raise ReportUnusable(
            f"{where} states schema version {version!r}; a {kind!r} measurement this harness "
            f"wrote states {expected_version}"
        )
    validator = _MEASUREMENT_SCHEMAS[kind]
    assert callable(validator)
    validator(record, where)
    return name, record


# --- the gate bindings ----------------------------------------------------------------


def _validate_bindings(
    value: object,
    measurements: Mapping[str, Mapping[str, object]],
    gates: Sequence[Mapping[str, object]],
    where: str,
) -> None:
    """Every gate's value, resolved from the measurement the fixture named.

    :class:`scanner3d.quality.Gate` duplicates its measured value and records
    nothing about where it came from, so a report carrying only gates cannot
    tell a measurement that changed from one that did not. These bindings
    close that: one per gate, in the gates' own order, naming the measurement
    and the path into its serialised form, and the value at that path must be
    the value the gate compared - as the same kind of thing, so a boolean
    ``true`` cannot stand in for a number.
    """
    entries = value
    if not isinstance(entries, list):
        raise ReportUnusable(f"{where} is {entries!r}, which is not a list of gate bindings")
    bound: list[str] = []
    for index, entry in enumerate(entries):
        at = f"{where}[{index}]"
        binding = _mapping(entry, at, error=ReportUnusable)
        _exact_keys(binding, ("gate", "measurement", "path"), at, error=ReportUnusable)
        gate_name = _report_text(binding, "gate", at)
        source = _report_text(binding, "measurement", at)
        path = _report_strings(binding.get("path"), f"{at} 'path'")
        if not path:
            raise ReportUnusable(
                f"{at} binds gate {gate_name!r} to an empty path; a gate reads a named field of "
                "its measurement, and a binding to nothing proves nothing"
            )
        for step in path:
            if not PATH_STEP.match(step):
                raise ReportUnusable(
                    f"{at} walks {step!r}, which is not a plain measurement field name"
                )
        if source not in measurements:
            raise ReportUnusable(
                f"{at} binds gate {gate_name!r} to measurement {source!r}, which this case does "
                f"not report; it reports {sorted(measurements)}"
            )
        bound.append(gate_name)
    if len(set(bound)) != len(bound):
        raise ReportUnusable(f"{where} binds a gate more than once: {bound}")

    named = [str(gate["name"]) for gate in gates]
    unbound = sorted(set(named) - set(bound))
    if unbound:
        raise ReportUnusable(
            f"{where} leaves gate(s) {unbound} unbound; a gate whose value names no measurement "
            "is a number with no provenance"
        )
    invented = sorted(set(bound) - set(named))
    if invented:
        raise ReportUnusable(f"{where} binds gate(s) {invented}, which this case does not report")
    if bound != named:
        raise ReportUnusable(
            f"{where} binds {bound}, and the case reports gates {named}; the bindings are written "
            "in the gates' own order"
        )

    by_name = {str(gate["name"]): gate for gate in gates}
    for index, entry in enumerate(entries):
        binding = _mapping(entry, f"{where}[{index}]", error=ReportUnusable)
        gate_name = str(binding["gate"])
        source = str(binding["measurement"])
        path = [str(step) for step in _report_strings(binding.get("path"), f"{where}[{index}]")]
        at = f"{where}[{index}] gate {gate_name!r}"
        resolved: object = measurements[source]
        for step in path:
            if not isinstance(resolved, Mapping) or step not in resolved:
                raise ReportUnusable(
                    f"{at} reads {path} of measurement {source!r}, which has no {step!r} there"
                )
            resolved = resolved[step]
        measured = by_name[gate_name].get("measured")
        if isinstance(measured, bool) != isinstance(resolved, bool):
            raise ReportUnusable(
                f"{at} compared {measured!r} and its measurement reports {resolved!r} at {path}; "
                "a boolean and a number are not the same kind of thing"
            )
        if isinstance(measured, bool):
            if measured is not resolved:
                raise ReportUnusable(
                    f"{at} compared {measured!r} and measurement {source!r} reports {resolved!r} "
                    f"at {path}"
                )
            continue
        left = _report_finite(measured, f"{at} 'measured'")
        right = _report_finite(resolved, f"{at} measurement {source!r} at {path}")
        if left != right:
            raise ReportUnusable(
                f"{at} compared {left!r} and measurement {source!r} reports {right!r} at {path}. "
                "A measurement changed beside the gate that read it is exactly what this binding "
                "exists to catch"
            )


def _validate_gate(value: object, where: str) -> tuple[str, bool]:
    """One gate, with its verdict RECOMPUTED from what it carries.

    ``passed`` is not read and believed. A forged ``true`` beside a measured
    value that is over its limit is the single most useful lie an artifact
    could tell, and it is the one thing the gate itself contains enough
    information to refute.
    """
    gate = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        gate,
        ("name", "measured", "comparison", "limit", "unit", "passed", "context"),
        where,
        error=ReportUnusable,
    )
    name = _report_text(gate, "name", where)
    _report_text(gate, "unit", where)
    _report_text(gate, "context", where)
    comparison = gate.get("comparison")
    if not isinstance(comparison, str) or comparison not in COMPARISONS:
        raise ReportUnusable(
            f"{where} gate {name!r} compares with {comparison!r}; this schema states "
            f"{sorted(COMPARISONS)}"
        )
    measured, limit = gate.get("measured"), gate.get("limit")
    if COMPARISONS[comparison] is quality.Comparison.IS:
        if not isinstance(measured, bool) or not isinstance(limit, bool):
            raise ReportUnusable(
                f"{where} gate {name!r} compares {measured!r} to {limit!r} with 'is', and they "
                "are not both booleans"
            )
        verdict = measured is limit
    else:
        if isinstance(measured, bool) != isinstance(limit, bool):
            raise ReportUnusable(
                f"{where} gate {name!r} compares {measured!r} to {limit!r}; a boolean and a "
                "number are not the same kind of thing"
            )
        left = _report_finite(measured, f"{where} gate {name!r} 'measured'")
        right = _report_finite(limit, f"{where} gate {name!r} 'limit'")
        verdict = left <= right if comparison == "<=" else left >= right
    passed = gate.get("passed")
    if not isinstance(passed, bool):
        raise ReportUnusable(f"{where} gate {name!r} states passed {passed!r}, not a boolean")
    if passed != verdict:
        raise ReportUnusable(
            f"{where} gate {name!r} states passed {passed!r}, and {measured!r} {comparison} "
            f"{limit!r} is {verdict!r}. A verdict that does not follow from the values beside it "
            "is refused"
        )
    return name, verdict


@dataclass(frozen=True, slots=True)
class _QualityView:
    """What a validated embedded quality report yields to its caller."""

    passed: bool
    failing: list[str]
    measurements: Mapping[str, Mapping[str, object]]
    gates: tuple[Mapping[str, object], ...]


def _validate_quality(value: object, case_id: str, where: str) -> _QualityView:
    """The embedded :class:`~scanner3d.quality.CaseReport`, checked against itself.

    Returns the recomputed verdict, the sorted names of the gates that
    actually failed, and the validated measurements and gates, so the caller
    can hold the case's observed block and its gate bindings to them.
    """
    report = _mapping(value, where, error=ReportUnusable)
    _exact_keys(
        report,
        (
            "schema_version",
            "case",
            "passed",
            "gate_count",
            "measurements",
            "gates",
            "failures",
            "notes",
        ),
        where,
        error=ReportUnusable,
    )
    version = report.get("schema_version")
    if version != quality.SCHEMA_VERSION or isinstance(version, bool):
        raise ReportUnusable(f"{where} states quality schema version {version!r}")
    if report.get("case") != case_id:
        raise ReportUnusable(
            f"{where} calls itself {report.get('case')!r} inside the report of case {case_id!r}"
        )
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or not measurements:
        raise ReportUnusable(f"{where} carries no measurements")
    by_name: dict[str, Mapping[str, object]] = {}
    for index, entry in enumerate(measurements):
        name, measurement = _validate_measurement(entry, f"{where} measurement {index}")
        if name in by_name:
            raise ReportUnusable(f"{where} reports measurement {name!r} twice")
        by_name[name] = measurement
    _no_nonfinite(measurements, f"{where} 'measurements'")
    gates = report.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ReportUnusable(f"{where} carries no gates")
    names: set[str] = set()
    failing: list[str] = []
    for index, entry in enumerate(gates):
        name, verdict = _validate_gate(entry, f"{where} gate {index}")
        if name in names:
            raise ReportUnusable(f"{where} reports gate {name!r} twice")
        names.add(name)
        if not verdict:
            failing.append(name)
    stated_count = report.get("gate_count")
    if not isinstance(stated_count, int) or isinstance(stated_count, bool):
        raise ReportUnusable(f"{where} states gate_count {stated_count!r}, which is not a count")
    if stated_count != len(gates):
        raise ReportUnusable(
            f"{where} states {stated_count} gate(s) and carries {len(gates)}; the count is what a "
            "reader uses to tell 'nothing failed' from 'nothing was asked'"
        )
    failures = report.get("failures")
    if failures != [gate for gate in gates if not gate["passed"]]:
        raise ReportUnusable(
            f"{where} lists failures that are not exactly the gates it reports as failing"
        )
    passed = report.get("passed")
    if not isinstance(passed, bool):
        raise ReportUnusable(f"{where} states passed {passed!r}, which is not a boolean")
    if passed != (not failing):
        raise ReportUnusable(
            f"{where} states passed {passed!r} beside {len(failing)} failing gate(s) {failing}"
        )
    notes = report.get("notes")
    if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
        raise ReportUnusable(f"{where} notes is not a list of strings")
    return _QualityView(
        passed=passed,
        failing=sorted(failing),
        measurements=by_name,
        gates=tuple(
            _mapping(gate, f"{where} gate {index}", error=ReportUnusable)
            for index, gate in enumerate(gates)
        ),
    )


def _validate_case_body(case: Mapping[str, object], case_id: str, what: str) -> None:
    """One reported case, every claim of which is checked against its evidence."""
    where = f"{what} case {case_id!r}"
    fixture = _mapping(case.get("fixture"), f"{where} 'fixture'", error=ReportUnusable)
    _exact_keys(fixture, ("name", "sha256", "bytes"), f"{where} 'fixture'", error=ReportUnusable)
    name = fixture.get("name")
    if not isinstance(name, str) or not FIXTURE_NAME.match(name):
        raise ReportUnusable(
            f"{where} names fixture {name!r}, which is not a fixture file name; a case this "
            "harness ran came from a plain lower-case .json file inside the corpus"
        )
    _digest(fixture, "sha256", f"{where} 'fixture'")
    size = fixture.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ReportUnusable(f"{where} fixture bytes {size!r} is not a positive integer")

    status = case.get("status")
    mismatches = case.get("mismatches")
    if not isinstance(mismatches, list) or any(not isinstance(line, str) for line in mismatches):
        raise ReportUnusable(f"{where} mismatches is not a list of strings")
    error = case.get("error")
    if (status == "malformed") != (error is not None):
        raise ReportUnusable(f"{where} is {status!r} beside error {error!r}")

    if status == "malformed":
        if not isinstance(error, str) or not error.strip():
            raise ReportUnusable(f"{where} is malformed and says nothing about why")
        carried = {
            key: case.get(key)
            for key in ("what", "why", "expected", "observed", "quality", "gate_bindings")
            if case.get(key) is not None
        }
        if carried or mismatches:
            raise ReportUnusable(
                f"{where} is malformed and still carries {sorted(carried) or mismatches}. A "
                "fixture that could not be read produced no evidence, and an entry holding both "
                "is not a thing this harness can have written"
            )
        return

    _report_text(case, "what", where)
    _report_text(case, "why", where)
    expected_outcome, expected_failures = _outcome_block(
        case.get("expected"), f"{where} 'expected'"
    )
    observed_outcome, observed_failures = _outcome_block(
        case.get("observed"), f"{where} 'observed'"
    )
    view = _validate_quality(case.get("quality"), case_id, f"{where} 'quality'")
    _validate_bindings(
        case.get("gate_bindings"), view.measurements, view.gates, f"{where} 'gate_bindings'"
    )
    if observed_outcome != ("pass" if view.passed else "fail"):
        raise ReportUnusable(
            f"{where} observed {observed_outcome!r} beside a quality report that "
            f"{'passed' if view.passed else 'failed'}"
        )
    if observed_failures != view.failing:
        raise ReportUnusable(
            f"{where} observed failing {observed_failures} beside gates that actually failed "
            f"{view.failing}"
        )
    derived = _mismatch_lines(
        expected_outcome, expected_failures, observed_outcome, observed_failures
    )
    if list(mismatches) != derived:
        raise ReportUnusable(
            f"{where} states mismatches {list(mismatches)}; expected versus observed gives "
            f"{derived}. The prose is derived, not asserted, so invented or omitted findings are "
            "refused"
        )
    if (status == "pass") != (not derived):
        raise ReportUnusable(
            f"{where} is {status!r} beside {len(derived)} mismatch(es); a passing case has none "
            "and a failing one names them"
        )


def to_json(document: Mapping[str, object]) -> str:
    """The one serialisation. Strict, stable and byte-identical between runs.

    ``allow_nan=False`` because ``NaN`` and ``Infinity`` are not JSON and a
    consumer that accepted them would read a value that compares false to
    everything. ``sort_keys=False`` because the insertion order above is the
    order a person reads it in, and it is fixed by the code rather than by the
    data.
    """
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=False) + "\n"


def staging_path(path: Path) -> Path:
    """Where the artifact is written before it is renamed over ``path``."""
    return path.parent / f".{path.name}.partial"


def clear_destination(path: Path) -> Path:
    """Empty the exact requested destination, and only it, before any work.

    A requested output path must never be left holding an EARLIER report after
    an invocation that could not produce one. Nothing else makes an artifact
    dangerous in the way a plausible stale one does: it carries a real digest,
    a real decision and a real set of cases, and nothing in it says it is not
    this run's answer.

    So the destination and its staging path are removed up front - explicitly,
    by name, and nowhere else on the filesystem - and every later failure
    (a corpus that will not load, a document that will not serialise, a write
    or rename that fails) leaves no file rather than the old one. Refused
    rather than removed: a directory, a symbolic link, or a parent that is not
    a directory. Deleting either of those would be an action on something that
    is not this harness's artifact.
    """
    parent = path.parent
    if not parent.is_dir():
        raise HarnessRefused(f"{parent} is not a directory; the artifact has nowhere to go")
    if not path.name or path.name in (".", ".."):
        raise HarnessRefused(f"{path} does not name a file the artifact could be written to")
    staging = staging_path(path)
    for target in (path, staging):
        if target.is_symlink():
            raise HarnessRefused(
                f"{target} is a symbolic link. The report is written where it was asked for, not "
                "through a pointer to somewhere else"
            )
        if target.is_dir():
            raise HarnessRefused(f"{target} is a directory, not a report this run may replace")
        try:
            target.unlink(missing_ok=True)
        except OSError as error:
            raise HarnessRefused(f"{target} could not be cleared: {error}") from error
    return staging


def publish(document: Mapping[str, object], path: Path) -> None:
    """Write the artifact atomically, or leave nothing behind.

    The destination is cleared first (see :func:`clear_destination`), then the
    document is serialised in full, written to a temporary file in the SAME
    directory, flushed, fsynced and renamed over the destination. A failure at
    any point leaves no file at ``path`` and no partial one beside it: a reader
    therefore never sees half an artifact, and never sees an older one
    presented as this run's.
    """
    staging = clear_destination(path)
    text = to_json(document)
    try:
        with staging.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        staging.replace(path)
    except OSError as error:
        staging.unlink(missing_ok=True)
        raise HarnessRefused(f"the artifact could not be published to {path}: {error}") from error


def render(document: Mapping[str, object]) -> str:
    """The whole report as text, for a CI log. Deterministic.

    Every gate of every case, not only the failures: the passing ones are what
    say which failure mode was ruled out, and a log showing only failures makes
    a case that failed one gate look like a case that only had one.
    """
    cases = document["cases"]
    assert isinstance(cases, list)
    totals = document["totals"]
    assert isinstance(totals, Mapping)
    decision = str(document["decision"]).upper()
    lines = [
        f"[{decision}] reconstruction quality harness: {totals['passed']} of {totals['cases']} "
        f"case(s) produced exactly the outcome their fixture declares "
        f"({totals['failed']} mismatched, {totals['malformed']} malformed)"
    ]
    for entry in cases:
        assert isinstance(entry, Mapping)
        status = str(entry["status"]).upper()
        lines.append(f"[{status}] {entry['case_id']} ({entry['fixture']['name']})")
        if entry["status"] == "malformed":
            lines.append(f"  fixture unusable: {entry['error']}")
            continue
        expected, observed = entry["expected"], entry["observed"]
        assert isinstance(expected, Mapping) and isinstance(observed, Mapping)
        lines.append(
            f"  expected {expected['outcome']} failing {list(expected['failed_gates'])}; "
            f"observed {observed['outcome']} failing {list(observed['failed_gates'])}"
        )
        report = entry["quality"]
        assert isinstance(report, Mapping)
        gates = report["gates"]
        assert isinstance(gates, list)
        for gate in gates:
            assert isinstance(gate, Mapping)
            verdict = "PASS" if gate["passed"] else "FAIL"
            lines.append(
                f"  {verdict} {gate['name']}: {gate['measured']} {gate['comparison']} "
                f"{gate['limit']} ({gate['unit']})"
            )
        notes = report["notes"]
        assert isinstance(notes, list)
        lines.extend(f"  note: {note}" for note in notes)
        mismatches = entry["mismatches"]
        assert isinstance(mismatches, list)
        lines.extend(f"  MISMATCH: {line}" for line in mismatches)
    lines.append("")
    lines.extend(f"limitation: {line}" for line in LIMITATIONS)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tests.quality_harness",
        description=(
            "Run the deterministic reconstruction-quality fixture corpus through "
            "scanner3d.quality and the real readiness reader, and write one strict JSON report."
        ),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS,
        help="the fixture corpus to run (default: tests/fixtures/quality/cases)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to publish the strict JSON report, written atomically",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the rendered report; the exit code and --out still say everything",
    )
    arguments = parser.parse_args(argv)

    # Before the corpus is even opened. A run that refuses at exit 2 must not
    # leave the previous run's report sitting at the path this invocation was
    # asked to write, where the next reader would take it for this answer.
    if arguments.out is not None:
        try:
            clear_destination(arguments.out)
        except HarnessRefused as error:
            print(f"the quality harness could not use that output path: {error}", file=sys.stderr)
            return 2
    try:
        document = run(arguments.corpus)
    except HarnessRefused as error:
        print(f"the quality harness could not run: {error}", file=sys.stderr)
        return 2
    if arguments.out is not None:
        try:
            publish(document, arguments.out)
        except (HarnessRefused, ValueError) as error:
            print(f"the quality harness could not publish its report: {error}", file=sys.stderr)
            return 2
    if not arguments.quiet:
        print(render(document))
    return 0 if document["decision"] == "pass" else 1


if __name__ == "__main__":  # pragma: no cover - the command line entry point
    raise SystemExit(main())
