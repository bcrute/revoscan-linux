"""The CI workflow, read as data and held to the promises its comments make.

`.github/workflows/ci.yml` is the one file in this repository that runs code on
a machine nobody here owns, with a token GitHub mints. Its safety properties -
read-only, secret-free, hosted, pinned, bounded, hardware-free - are all
conventions, and every one of them is a single careless line away from being
untrue. A reviewer would have to notice; these tests notice instead.

The assertions are written generically over every job and every step rather
than against the five jobs that exist today. A future job that quietly asks
for a self-hosted runner, a write permission or an unpinned action is exactly
the change worth catching, and a test that only knew about `static`, `tests`,
`harness`, `depth-completeness` and `surface-oracle` would let it through.

The three named correctness-oracle jobs are the exception, and deliberately so.
Each one exists to run one exact command and keep one exact artifact, and
`HARNESSES` below states all three pairs once. Generic assertions cannot notice a
job that runs the wrong harness, uploads the other one's report, or quietly
disappears.

There is no YAML library here and none may be added, so the parsing goes
through `tests.workflow_yaml`, which refuses anything outside its subset rather
than silently dropping it. One claim - the version comment beside each pinned
action - is a property of the text rather than of the parsed value, so that one
reads the source directly. The rest look at parsed values only, so that a
comment explaining why a tool is not run cannot read as a use of it.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any

import pytest

from tests.workflow_yaml import WorkflowYamlUnusable, parse

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

# A full commit object name, lower case: the only `uses` reference that names
# code no one can change after review.
COMMIT_SHA_RE = re.compile(r"\A[0-9a-f]{40}\Z")
USES_LINE_RE = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+)\s*(?P<comment>#.*)?$")

# Tooling that needs a scanner, a capture on this developer's disk, or an
# opt-in benchmark run. None of it belongs on a hosted runner.
PHYSICAL_ONLY_TOKENS = (
    "quality_replay",
    "readiness_replay",
    "preview_health_replay",
    "tsdf_benchmark",
    "/dev/video",
    "/home/ben/scans",
    "usbmon",
    "--capture",
)

# Long enough for the offline suite, short enough that a hung job gives the
# runner back. The ordinary suite is what sets this ceiling: it measured about
# 29.5 minutes on the owner's machine, so the 30 minutes this used to allow was
# not a bound with margin - it was a bound the suite was already sitting on,
# and a hosted runner, which is not that machine and may be slower, would have
# begun reporting timeouts instead of test results. An hour is twice that
# measured cost. A gate that needs longer than an hour has stopped being an
# offline gate over a small repository and is a different thing, which someone
# should have to argue for rather than raise this number for.
MAX_TIMEOUT_MINUTES = 60
MAX_RETENTION_DAYS = 90

# What the two long jobs actually cost, in minutes, measured on the owner's
# machine: the frozen full suite at 29 minutes 30 seconds, and the whole
# surface-oracle corpus at 3 minutes 11 seconds. The bounds below are stated as
# multiples of these rather than as round numbers near them, because a hosted
# runner is a different and possibly slower machine.
MEASURED_FULL_SUITE_MINUTES = 29.5
MEASURED_SURFACE_ORACLE_MINUTES = 3.2

# The bound each of those jobs carries, so a change to either is a decision
# somebody makes here rather than a number that drifted in the workflow.
JUSTIFIED_TIMEOUT_MINUTES = {"tests": 60, "surface-oracle": 20}

# The one interpreter every job is pinned to, and the one way the environment
# may be built from it: `--frozen` fails rather than re-resolving when uv.lock
# and pyproject.toml disagree, so a job that dropped it would be testing an
# environment the lockfile never described.
PYTHON_VERSION = "3.14"
FROZEN_SYNC = "uv sync --frozen --extra dev"

# The three CPU correctness oracles that run as their own named jobs: the job
# key, the exact command README.md documents, and the artifact that command
# writes. Stated once here and read back out of the workflow, so a job that
# runs another harness, renames its report, or vanishes is a failure rather
# than a silently narrower CI.
HARNESSES = {
    "harness": (
        ".venv/bin/python -m tests.quality_harness --out quality-harness-report.json",
        "quality-harness-report.json",
    ),
    "depth-completeness": (
        ".venv/bin/python -m tests.depth_completeness_harness --out depth-completeness-report.json",
        "depth-completeness-report.json",
    ),
    "surface-oracle": (
        ".venv/bin/python -m tests.surface_oracle_harness --out surface-oracle-report.json",
        "surface-oracle-report.json",
    ),
}

# A hosted runner has no GPU and the oracles do not want one. Every backend any
# of them selects is a CPU implementation over committed synthetic truth: the
# quality harness and the depth-completeness benchmark run the pure-Python
# references throughout, and the surface oracle runs the pure-Python reference
# for its six declared grid cells and the numpy CPU backend for the one
# shipped-policy arm whose two-million-voxel field is past the pure-Python
# ceiling. No accelerated implementation is selected or required anywhere, so a
# label or a flag reaching for acceleration would add an execution path that
# improves no correctness evidence, and would mean the job had stopped being
# the reproduction of the README's offline gates that it claims to be. `lan`
# and `heavy` are the owner's names for the machine this work must never move
# to.
ACCELERATOR_TOKENS = ("gpu", "cuda", "nvidia", "opencl", "lan", "heavy")

# Keys and step fields through which a workflow stops being read-only: a
# deployment, a release, an environment with its own secrets, or a package
# push. None of them belong in a workflow whose whole output is a log and three
# JSON reports.
DEPLOYMENT_KEYS = ("environment", "deployments", "release", "publish", "deploy")


def _read_workflow() -> tuple[str, dict[str, Any]]:
    """Return the workflow source and its parsed form, or fail with the reason."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    try:
        document = parse(text)
    except WorkflowYamlUnusable as unusable:  # pragma: no cover - a broken workflow
        pytest.fail(f"{WORKFLOW_PATH} is not parseable: {unusable}")
    if not isinstance(document, dict):
        pytest.fail(f"{WORKFLOW_PATH} is not a mapping at the top level")
    return text, document


RAW_TEXT, WORKFLOW = _read_workflow()


def _jobs() -> dict[str, dict[str, Any]]:
    """Return every job as a mapping, failing rather than skipping a malformed one."""
    jobs = WORKFLOW.get("jobs")
    assert isinstance(jobs, dict) and jobs, "the workflow declares no jobs"
    result: dict[str, dict[str, Any]] = {}
    for name, job in jobs.items():
        assert isinstance(job, dict), f"job {name} is not a mapping"
        result[name] = job
    return result


JOBS = _jobs()


def _steps(job_name: str) -> list[dict[str, Any]]:
    """Return the steps of one job, each as a mapping."""
    steps = JOBS[job_name].get("steps")
    assert isinstance(steps, list) and steps, f"job {job_name} declares no steps"
    result: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        assert isinstance(step, dict), f"job {job_name} step {index} is not a mapping"
        result.append(step)
    return result


def _all_steps() -> list[tuple[str, dict[str, Any]]]:
    """Return every step in the workflow, paired with the job it belongs to."""
    return [(name, step) for name in JOBS for step in _steps(name)]


def _flatten(value: object) -> list[str]:
    """Return every key and scalar inside `value` as text, so nested blocks are searched too."""
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in [str(key), *_flatten(item)]]
    if isinstance(value, list):
        return [text for item in value for text in _flatten(item)]
    return [] if value is None else [str(value)]


# Everything the workflow actually says, comments excluded: a comment explaining
# why a tool is NOT run must not read as a use of it.
WORKFLOW_VALUES = _flatten(WORKFLOW)


def _run_commands() -> list[tuple[str, str]]:
    """Return each `run` script in the workflow, paired with its job name."""
    commands: list[tuple[str, str]] = []
    for job_name, step in _all_steps():
        script = step.get("run")
        if script is not None:
            assert isinstance(script, str), f"job {job_name} has a non-string run"
            commands.append((job_name, script))
    return commands


def _harness_out_path(script: str) -> str:
    """Return the value the harness invocation passes to `--out`."""
    tokens = shlex.split(script)
    for index, token in enumerate(tokens):
        if token == "--out":
            assert index + 1 < len(tokens), "--out has no value"
            return tokens[index + 1]
        if token.startswith("--out="):
            return token.removeprefix("--out=")
    pytest.fail(f"the harness invocation names no --out artifact: {script}")


def _harness_step(job_name: str, command: str) -> dict[str, Any]:
    """Return the one step of `job_name` whose `run` is `command`, exactly."""
    assert job_name in JOBS, f"the {job_name} job is gone"
    wanted = shlex.split(command)
    matches = [
        step
        for step in _steps(job_name)
        if isinstance(step.get("run"), str) and shlex.split(str(step["run"])) == wanted
    ]
    scripts = [step.get("run") for step in _steps(job_name) if "run" in step]
    assert len(matches) == 1, (
        f"job {job_name} does not run {command!r} exactly once; it runs {scripts!r}"
    )
    return matches[0]


def _upload_steps() -> list[tuple[str, dict[str, Any]]]:
    """Return every artifact-upload step, paired with the job it belongs to."""
    return [
        (job_name, step)
        for job_name, step in _all_steps()
        if isinstance(step.get("uses"), str) and "actions/upload-artifact@" in str(step["uses"])
    ]


def test_the_workflow_parses_and_every_job_runs_on_a_hosted_ubuntu_runner() -> None:
    """A self-hosted, containerised or service-backed job would run our CI on a machine
    whose contents and lifetime we do not control, and would quietly stop being the
    clean-room reproduction of the README's offline gates that this workflow claims to be.
    """
    assert JOBS, "the workflow parsed but declares no jobs"
    for name, job in JOBS.items():
        assert job.get("runs-on") == "ubuntu-latest", f"job {name} is not on ubuntu-latest"
        assert "container" not in job, f"job {name} runs in a container"
        assert "services" not in job, f"job {name} starts service containers"
    for text in WORKFLOW_VALUES:
        assert "self-hosted" not in text, f"the workflow asks for a self-hosted runner: {text}"


def test_no_job_widens_the_read_only_workflow_permissions() -> None:
    """The token GitHub mints is scoped by these keys alone. Anything beyond `contents: read`
    would let a workflow triggered by a fork pull request write to this repository, and this
    workflow has nothing to write: it builds no release, publishes nothing and touches no
    packages, pages or checks.
    """
    assert WORKFLOW.get("permissions") == {"contents": "read"}
    for name, job in JOBS.items():
        permissions = job.get("permissions")
        if permissions is None:
            continue
        assert isinstance(permissions, dict), f"job {name} has non-mapping permissions"
        for scope, level in permissions.items():
            allowed = ("read", "none") if scope == "contents" else ("none",)
            assert level in allowed, f"job {name} grants {scope}: {level}"


def test_nothing_in_the_workflow_asks_for_a_secret() -> None:
    """A read-only public CI run needs no credential, so any `secrets` reference here is
    either dead weight or a new way for a fork pull request to reach something private.
    The sweep over every parsed value, and not just over `run`, `with` and `env`, is what
    catches an expression that arrives through a key or a block this test never named.
    """
    for name, job in JOBS.items():
        assert "secrets" not in job, f"job {name} passes secrets"
    for job_name, step in _all_steps():
        for field in ("run", "with", "env"):
            for text in _flatten(step.get(field)):
                assert "${{ secrets." not in text, f"job {job_name} reads a secret in {field}"
    for text in WORKFLOW_VALUES:
        assert "${{ secrets." not in text, f"the workflow references a secret: {text}"


def test_every_action_is_pinned_to_a_reviewed_commit_sha() -> None:
    """A tag or branch reference resolves to whatever that name points at when the job runs,
    so an upstream compromise becomes our compromise without a commit here. The `# v...`
    comment is the other half: a bare SHA says nothing about which release was reviewed,
    and without it nobody can tell an upgrade from a downgrade.
    """
    uses_lines = [
        match for line in RAW_TEXT.splitlines() if (match := USES_LINE_RE.match(line)) is not None
    ]
    assert uses_lines, "the workflow uses no actions, which this test cannot be right about"
    for match in uses_lines:
        reference = match.group("ref")
        _, separator, revision = reference.partition("@")
        assert separator, f"{reference} names no revision"
        assert COMMIT_SHA_RE.match(revision), f"{reference} is not pinned to a commit sha"
        comment = match.group("comment")
        assert comment is not None and comment.startswith("# v"), (
            f"{reference} carries no '# v...' comment naming the reviewed version"
        )
    parsed_uses = [step["uses"] for _, step in _all_steps() if "uses" in step]
    assert len(parsed_uses) == len(uses_lines), "a `uses` was parsed that the text scan missed"


def test_every_job_is_bounded_by_a_timeout() -> None:
    """Without `timeout-minutes` a hung job holds a runner for GitHub's six-hour default.
    The ceiling also states an expectation: these are offline gates over a small repository,
    so a job that needs longer than an hour has changed into something else. An hour rather
    than the half hour this used to allow, because the ordinary suite alone now measures about
    29.5 minutes locally and a bound a job is already sitting on is not a bound.
    """
    for name, job in JOBS.items():
        timeout = job.get("timeout-minutes")
        assert isinstance(timeout, int) and not isinstance(timeout, bool), (
            f"job {name} sets no integer timeout-minutes"
        )
        assert 0 < timeout <= MAX_TIMEOUT_MINUTES, f"job {name} allows {timeout} minutes"


def test_the_two_long_jobs_carry_the_bounds_their_measured_costs_justify() -> None:
    """A ceiling every job passes says nothing about the two jobs that are actually long, so
    those two are checked against what they cost rather than against the ceiling. The ordinary
    suite measured about 29 minutes 30 seconds on the owner's machine, and a GitHub-hosted
    runner is not that machine: at 30 minutes the job would have been reporting timeouts, which
    is not evidence, instead of test results. Its bound is now twice the measured cost. The
    surface oracle's whole corpus measured 3 minutes 11 seconds, and its 20 minutes is the
    deliberately larger multiple that catches an oracle which hangs while letting a slow hosted
    runner finish. Both numbers are asserted exactly, so raising either is a decision somebody
    makes rather than a drift nobody reads.
    """
    for job_name, expected in JUSTIFIED_TIMEOUT_MINUTES.items():
        assert job_name in JOBS, f"the {job_name} job is gone"
        assert JOBS[job_name].get("timeout-minutes") == expected, (
            f"job {job_name} allows {JOBS[job_name].get('timeout-minutes')} minutes, "
            f"not the justified {expected}"
        )
    tests_timeout = JOBS["tests"]["timeout-minutes"]
    assert tests_timeout >= 2 * MEASURED_FULL_SUITE_MINUTES, (
        f"the ordinary suite's {tests_timeout} minutes is less than twice the "
        f"{MEASURED_FULL_SUITE_MINUTES} it measures locally, so a slower hosted runner "
        f"has no margin"
    )
    oracle_timeout = JOBS["surface-oracle"]["timeout-minutes"]
    assert oracle_timeout >= 4 * MEASURED_SURFACE_ORACLE_MINUTES, (
        f"the surface oracle's {oracle_timeout} minutes is less than four times the "
        f"{MEASURED_SURFACE_ORACLE_MINUTES} its corpus measures locally"
    )


def test_each_named_harness_job_runs_its_exact_command_once() -> None:
    """`HARNESSES` is the contract: one job, one command, run through the locked environment.
    The comparison is over `shlex` tokens rather than raw text, so reflowed whitespace is not a
    failure while a changed flag, a changed module or a second invocation is. A harness invoked
    from more than one job would double the cost and make it ambiguous which run produced the
    artifact that was kept.
    """
    missing = sorted(set(HARNESSES) - set(JOBS))
    assert not missing, f"the workflow no longer declares the oracle jobs {missing}"
    for job_name, (command, _) in HARNESSES.items():
        _harness_step(job_name, command)
    for command, _ in HARNESSES.values():
        module = shlex.split(command)[shlex.split(command).index("-m") + 1]
        source = WORKFLOW_PATH.parents[2] / (module.replace(".", "/") + ".py")
        assert source.is_file(), f"the workflow runs {module}, which is not a module in this tree"
        invocations = [script for _, script in _run_commands() if f"-m {module}" in script]
        assert len(invocations) == 1, f"{module} runs {len(invocations)} times, not once"


def test_no_harness_step_swallows_the_exit_code_that_is_the_whole_point() -> None:
    """A harness says what it found only through its exit status: `1` is a candidate that did
    not do what its fixture declares, `2` is a corpus that could not be run at all. `||
    true`, a trailing `; exit 0` or `continue-on-error` would leave the job green over exactly
    the runs worth catching, and a job-level `continue-on-error` would do it one level up
    where the step still looks honest.
    """
    for job_name, job in JOBS.items():
        assert job.get("continue-on-error") is not True, f"job {job_name} cannot fail"
    for job_name, step in _all_steps():
        assert step.get("continue-on-error") is not True, (
            f"a step of job {job_name} cannot fail: {step.get('name')}"
        )
    for job_name, (command, _) in HARNESSES.items():
        step = _harness_step(job_name, command)
        assert "if" not in step, f"job {job_name} runs its harness only conditionally"
    for job_name, script in _run_commands():
        for swallow in ("|| true", "|| :", "exit 0", "set +e"):
            assert swallow not in script, f"job {job_name} discards a failure with {swallow!r}"


def test_each_harness_job_uploads_the_file_its_own_command_wrote() -> None:
    """The report is only evidence if the file the harness writes is the file that leaves the
    runner. The `--out` value is read back out of the run command rather than written here a
    second time, because a test that hard-codes the path agrees with itself while the workflow
    uploads something that was never created. Crossing the artifacts - any one of the three jobs
    uploading another's report - is the mistake this pairing exists to refuse.
    """
    uploads = dict(_upload_steps())
    assert len(uploads) == len(_upload_steps()), "a job uploads more than one artifact"
    assert set(uploads) == set(HARNESSES), (
        f"the jobs that upload artifacts are {sorted(uploads)}, not {sorted(HARNESSES)}"
    )
    for job_name, (command, expected_artifact) in HARNESSES.items():
        artifact = _harness_out_path(str(_harness_step(job_name, command)["run"]))
        assert artifact == expected_artifact, (
            f"job {job_name} writes {artifact!r}, not {expected_artifact!r}"
        )
        options = uploads[job_name].get("with")
        assert isinstance(options, dict), f"job {job_name} upload step passes no `with` block"
        assert options.get("path") == artifact, (
            f"job {job_name} uploads {options.get('path')!r}, not its own report {artifact!r}"
        )


def test_every_uploaded_artifact_has_its_own_name() -> None:
    """Two jobs uploading under one name is a collision, not two pieces of evidence: whichever
    lands second is the one a reader downloads, and the run that failed can be the one that
    disappeared. The names are also checked against the reports themselves, so an artifact
    cannot be named after a file some other job produced.
    """
    names: dict[str, str] = {}
    for job_name, step in _upload_steps():
        options = step.get("with")
        assert isinstance(options, dict), f"job {job_name} upload step passes no `with` block"
        name = options.get("name")
        assert isinstance(name, str) and name, f"job {job_name} uploads an unnamed artifact"
        assert name not in names, f"jobs {names[name]} and {job_name} both upload {name!r}"
        names[name] = job_name
    assert set(names.values()) == set(HARNESSES)
    for job_name, (_, artifact) in HARNESSES.items():
        name = next(key for key, owner in names.items() if owner == job_name)
        assert Path(artifact).stem == name, (
            f"job {job_name} uploads {artifact!r} under the unrelated name {name!r}"
        )


def test_the_harness_report_is_uploaded_even_when_the_harness_fails() -> None:
    """`always()` is the load-bearing part, and `success()` would defeat the whole step: the
    failing run is the one whose report matters, and the harness exits nonzero exactly then.
    `if-no-files-found: error` is the same argument once more - a report that was never
    written is a fact worth failing over, not a silent skip - and a bounded retention keeps
    the artifacts from becoming an archive nobody prunes.
    """
    uploads = _upload_steps()
    assert uploads, "nothing uploads a harness report"
    for job_name, step in uploads:
        condition = step.get("if")
        assert isinstance(condition, str), f"job {job_name} upload step has no `if`"
        assert condition.replace(" ", "") == "always()", (
            f"job {job_name} uploads only when {condition}"
        )
        options = step.get("with")
        assert isinstance(options, dict), f"job {job_name} upload step passes no `with` block"
        assert options.get("if-no-files-found") == "error"
        retention = options.get("retention-days")
        assert isinstance(retention, int) and not isinstance(retention, bool), (
            f"job {job_name} sets no integer retention-days"
        )
        assert 0 < retention <= MAX_RETENTION_DAYS, f"job {job_name} retains for {retention} days"


def test_every_job_builds_its_environment_from_the_lockfile_on_one_interpreter() -> None:
    """A job that installed with a bare `uv sync`, or that let uv fetch its own CPython beside
    the one setup-python installed, would be reporting on an environment nobody pinned - and
    the harnesses would then be oracles over whatever resolved that morning. Every job is held
    to the same interpreter and the same frozen sync, including any job added later.
    """
    assert WORKFLOW.get("env", {}).get("UV_PYTHON_PREFERENCE") == "only-system"
    for job_name in JOBS:
        setups = [
            step
            for step in _steps(job_name)
            if isinstance(step.get("uses"), str) and "actions/setup-python@" in str(step["uses"])
        ]
        assert len(setups) == 1, f"job {job_name} sets up Python {len(setups)} times"
        options = setups[0].get("with")
        assert isinstance(options, dict), f"job {job_name} setup-python passes no `with` block"
        assert options.get("python-version") == PYTHON_VERSION, (
            f"job {job_name} installs Python {options.get('python-version')!r}"
        )
        syncs = [
            script
            for script in [step.get("run") for step in _steps(job_name)]
            if isinstance(script, str) and script.split()[:2] == ["uv", "sync"]
        ]
        assert len(syncs) == 1, f"job {job_name} runs `uv sync` {len(syncs)} times"
        assert syncs[0].split() == FROZEN_SYNC.split(), (
            f"job {job_name} syncs with {syncs[0]!r}, not {FROZEN_SYNC!r}"
        )


def test_every_job_pins_the_same_reviewed_action_revisions() -> None:
    """Pinning per job is not enough on its own: two jobs on two revisions of checkout means
    one of them is running code that was never reviewed here, and the drift is invisible in a
    diff that only added a job. One action name, one commit, across the whole workflow.
    """
    revisions: dict[str, set[str]] = {}
    for _, step in _all_steps():
        reference = step.get("uses")
        if not isinstance(reference, str):
            continue
        name, _, revision = reference.partition("@")
        revisions.setdefault(name, set()).add(revision)
    assert revisions, "the workflow uses no actions"
    for name, seen in revisions.items():
        assert len(seen) == 1, (
            f"{name} is pinned to {len(seen)} different revisions: {sorted(seen)}"
        )


def test_nothing_asks_for_an_accelerator_or_the_owners_heavy_machine() -> None:
    """Every harness here is a correctness oracle over committed synthetic truth, and every
    backend any of them selects is a CPU implementation: pure Python throughout for the quality
    and depth-completeness corpora, and pure Python for the surface oracle's six declared cells
    with the numpy CPU backend for its one shipped-policy arm, whose field is past the
    pure-Python voxel ceiling. Nothing here selects or needs an accelerated implementation, so
    an accelerator would add a different execution path without improving the correctness
    evidence. The owner's decision is that this public work stays on free GitHub-hosted runners,
    and a label, a container image or a flag reaching for the LAN or heavy machine is how that
    decision would be reversed without anyone saying so.
    """
    for text in WORKFLOW_VALUES:
        lowered = text.lower()
        for token in ACCELERATOR_TOKENS:
            assert not re.search(rf"(?<![a-z]){token}(?![a-z])", lowered), (
                f"the workflow reaches for {token}: {text}"
            )


def test_the_workflow_deploys_publishes_and_releases_nothing() -> None:
    """Its entire output is a log and three JSON artifacts. An `environment:` would attach
    secrets and protection rules to a job that needs neither, and a release, deployment or
    package push from a workflow that fork pull requests can trigger is the failure mode the
    read-only token exists to prevent - but only while nothing here asks for one.
    """
    for job_name, job in JOBS.items():
        for key in DEPLOYMENT_KEYS:
            assert key not in job, f"job {job_name} declares {key}"
    for job_name, step in _all_steps():
        reference = step.get("uses")
        if isinstance(reference, str):
            action = reference.partition("@")[0].lower()
            for key in DEPLOYMENT_KEYS:
                assert key not in action, f"job {job_name} uses the {action} action"
    for job_name, script in _run_commands():
        for command in ("gh release", "twine", "uv publish", "docker push", "git push"):
            assert command not in script, f"job {job_name} runs {command}"


def test_the_ordinary_test_suite_still_runs() -> None:
    """Each oracle is additional named evidence over a synthetic corpus, not a replacement for
    the suite. Losing `.venv/bin/pytest` while the oracles kept passing would leave CI green
    over a fraction of what it used to check, and so would keeping it while narrowing it: `-k`,
    `-m`, `--ignore` or a path argument is how the surface tests - which run the oracle's corpus
    twice inside the ordinary suite - would be split away from it without the job disappearing.
    The run has to sit in the `tests` job, because that is the job whose timeout was justified
    against the whole suite's measured cost.
    """
    pytest_runs = [
        (job_name, script) for job_name, script in _run_commands() if ".venv/bin/pytest" in script
    ]
    assert pytest_runs, "no job runs .venv/bin/pytest"
    for job_name, script in pytest_runs:
        assert job_name == "tests", f"the ordinary suite runs in job {job_name}, not `tests`"
        assert shlex.split(script) == [".venv/bin/pytest"], (
            f"the ordinary suite is narrowed to a selection: {script!r}"
        )


def test_no_job_reaches_for_hardware_or_the_opt_in_replay_tools() -> None:
    """Every tool named here needs a scanner, a capture sitting on one developer's disk, or a
    benchmark whose numbers are meaningless on a shared runner. A hosted job that invoked one
    would fail for reasons unrelated to the change under test, so its red would stop meaning
    anything. The `hardware` pytest marker is the same failure by selection rather than path.
    """
    for text in WORKFLOW_VALUES:
        for token in PHYSICAL_ONLY_TOKENS:
            assert token not in text, f"the workflow references {token}: {text}"
        assert "hardware" not in text, f"the workflow selects hardware-marked work: {text}"


def test_checkout_leaves_no_credential_behind() -> None:
    """actions/checkout stores the run's token in `.git/config` unless told not to, where any
    later step - including one added by a dependency's own tooling - can read it. Nothing in
    this workflow talks to the remote after checkout, so there is no reason to keep it.
    """
    checkouts = [
        (job_name, step)
        for job_name, step in _all_steps()
        if isinstance(step.get("uses"), str) and "actions/checkout@" in str(step["uses"])
    ]
    assert len(checkouts) == len(JOBS), "not every job checks the repository out"
    for job_name, step in checkouts:
        options = step.get("with")
        assert isinstance(options, dict), f"job {job_name} checkout passes no `with` block"
        assert options.get("persist-credentials") is False, (
            f"job {job_name} leaves the run token in .git/config"
        )
