"""Distribution-artifact tests.

These exist because the runtime SDK manifest is *data*, not code: a wheel that
ships only ``.py`` files installs a CLI whose ``--sdk-manifest`` default points
at a file that is not there, so the oracle silently cannot be verified. The
sdist assertions are the mirror image - coordination state and agent
instructions are repository working files and must not be redistributed.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import scanner3d
from scanner3d.reference import DEFAULT_MANIFEST, PINNED_FILES, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(next(iter(scanner3d.__path__))).resolve()
SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


# --- manifest location (runs everywhere, including from an install) -----------


def test_default_manifest_lives_inside_the_package() -> None:
    """Repo-root-relative resolution breaks once the package is installed."""
    assert DEFAULT_MANIFEST.is_relative_to(PACKAGE_ROOT)
    assert DEFAULT_MANIFEST.is_file()


def test_default_manifest_pins_every_file_the_adapter_hashes() -> None:
    pinned = load_manifest(DEFAULT_MANIFEST)
    assert set(pinned) == set(PINNED_FILES)
    for name, digest in pinned.items():
        assert SHA256_RE.match(digest), f"{name} is not a sha256 hex digest"


# --- built artifacts ----------------------------------------------------------


@pytest.fixture(scope="session")
def built_artifacts(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build the wheel and sdist once, or skip if no build frontend is present."""
    if not (REPO_ROOT / "pyproject.toml").is_file():
        pytest.skip("not running from a source tree")
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed; cannot build distribution artifacts")
    out = tmp_path_factory.mktemp("dist")
    result = subprocess.run(
        [uv, "build", "--out-dir", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        pytest.fail(f"uv build failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    wheels = list(out.glob("*.whl"))
    sdists = list(out.glob("*.tar.gz"))
    assert len(wheels) == 1 and len(sdists) == 1, f"unexpected artifacts: {list(out.iterdir())}"
    return wheels[0], sdists[0]


def test_wheel_ships_the_pinned_sdk_manifest(built_artifacts: tuple[Path, Path]) -> None:
    wheel, _ = built_artifacts
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "scanner3d/data/sdk-manifest.json" in names, names
        payload = archive.read("scanner3d/data/sdk-manifest.json")
    assert payload == DEFAULT_MANIFEST.read_bytes(), "wheel manifest drifted from the source one"


def test_wheel_excludes_repository_working_files(built_artifacts: tuple[Path, Path]) -> None:
    wheel, _ = built_artifacts
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    leaked = [n for n in names if n.startswith(".coordination/") or n in {"AGENTS.md", "CLAUDE.md"}]
    assert leaked == [], f"wheel leaked repository-only files: {leaked}"


def test_sdist_contains_sources_docs_tests_and_reference(
    built_artifacts: tuple[Path, Path],
) -> None:
    _, sdist = built_artifacts
    names = _sdist_members(sdist)
    for expected in (
        "src/scanner3d/reference.py",
        "src/scanner3d/rgb_exposure_info.py",
        "src/scanner3d/rgb_exposure_length.py",
        "src/scanner3d/data/sdk-manifest.json",
        "tests/test_reference.py",
        "tests/test_rgb_exposure_info.py",
        "tests/test_rgb_exposure_length.py",
        "docs/architecture.md",
        "reference/probe/probe.cpp",
        "README.md",
        "pyproject.toml",
    ):
        assert expected in names, f"sdist is missing {expected}"


def test_sdist_excludes_coordination_state_and_agent_instructions(
    built_artifacts: tuple[Path, Path],
) -> None:
    _, sdist = built_artifacts
    names = _sdist_members(sdist)
    leaked = sorted(
        n for n in names if n.startswith(".coordination/") or n in {"AGENTS.md", "CLAUDE.md"}
    )
    assert leaked == [], f"sdist leaked repository-only files: {leaked}"


def _sdist_members(sdist: Path) -> set[str]:
    """Member paths with the ``name-version/`` prefix stripped."""
    with tarfile.open(sdist) as archive:
        raw = archive.getnames()
    return {name.split("/", 1)[1] for name in raw if "/" in name}


# --- installed behaviour ------------------------------------------------------


def test_installed_wheel_can_verify_the_oracle(
    built_artifacts: tuple[Path, Path], tmp_path: Path
) -> None:
    """Install the wheel away from the repo and load the manifest from there.

    ``--target`` keeps this offline: the project declares no dependencies.
    """
    wheel, _ = built_artifacts
    uv = shutil.which("uv")
    assert uv is not None  # the fixture already skipped otherwise
    target = tmp_path / "site"
    install = subprocess.run(
        [uv, "pip", "install", "--target", str(target), "--no-deps", str(wheel)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if install.returncode != 0:
        pytest.fail(f"install failed:\n{install.stdout}\n{install.stderr}")

    probe = (
        "import json, sys\n"
        "from scanner3d.reference import DEFAULT_MANIFEST, load_manifest\n"
        "assert DEFAULT_MANIFEST.is_file(), DEFAULT_MANIFEST\n"
        "print(json.dumps({'path': str(DEFAULT_MANIFEST), 'pinned': sorted(load_manifest())}))\n"
    )
    # cwd=tmp_path and an explicit sys.path keep the source tree out of reach.
    run = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(target)!r})\n{probe}"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert run.returncode == 0, f"installed package could not load its manifest:\n{run.stderr}"
    assert str(REPO_ROOT) not in run.stdout, "installed run resolved back into the source tree"
    assert sorted(PINNED_FILES) == json.loads(run.stdout)["pinned"]


def test_installed_wheel_exposes_rgb_get_len_help_and_dry_json(
    built_artifacts: tuple[Path, Path], tmp_path: Path
) -> None:
    """Run the new CLI from an isolated wheel install, never the source tree."""
    wheel, _ = built_artifacts
    uv = shutil.which("uv")
    assert uv is not None
    target = tmp_path / "site"
    install = subprocess.run(
        [uv, "pip", "install", "--target", str(target), "--no-deps", str(wheel)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert install.returncode == 0, f"install failed:\n{install.stdout}\n{install.stderr}"

    help_code = (
        "import pathlib, sys\n"
        "target = pathlib.Path(sys.argv[1]).resolve()\n"
        "sys.path.insert(0, str(target))\n"
        "import scanner3d\n"
        "assert pathlib.Path(scanner3d.__file__).resolve().is_relative_to(target)\n"
        "from scanner3d.cli import main\n"
        "raise SystemExit(main(['rgb-exposure-length', '--help']))\n"
    )
    help_run = subprocess.run(
        [sys.executable, "-I", "-c", help_code, str(target)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert help_run.returncode == 0, help_run.stderr
    assert "--execute-rgb-exposure-length-read" in help_run.stdout
    assert "GET_LEN" in help_run.stdout
    assert str(REPO_ROOT) not in help_run.stdout + help_run.stderr

    dry_code = (
        "import pathlib, sys\n"
        "target = pathlib.Path(sys.argv[1]).resolve()\n"
        "sys.path.insert(0, str(target))\n"
        "import scanner3d\n"
        "assert pathlib.Path(scanner3d.__file__).resolve().is_relative_to(target)\n"
        "from scanner3d.cli import main\n"
        "raise SystemExit(main(['rgb-exposure-length', '--json']))\n"
    )
    dry_run = subprocess.run(
        [sys.executable, "-I", "-c", dry_code, str(target)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    payload = json.loads(dry_run.stdout)
    assert payload["status"] == "not_attempted"
    assert payload["length_outcome"] == "not_attempted"
    assert payload["transfers"] == []
    assert payload["succeeded"] is False
    assert str(REPO_ROOT) not in dry_run.stdout + dry_run.stderr


def test_installed_wheel_exposes_rgb_get_info_help_and_dry_json(
    built_artifacts: tuple[Path, Path], tmp_path: Path
) -> None:
    """Run GET_INFO help/dry from an isolated wheel target, never the source tree."""
    wheel, _ = built_artifacts
    uv = shutil.which("uv")
    assert uv is not None
    target = tmp_path / "site"
    install = subprocess.run(
        [uv, "pip", "install", "--target", str(target), "--no-deps", str(wheel)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert install.returncode == 0, f"install failed:\n{install.stdout}\n{install.stderr}"

    help_code = (
        "import pathlib, sys\n"
        "target = pathlib.Path(sys.argv[1]).resolve()\n"
        "sys.path.insert(0, str(target))\n"
        "import scanner3d\n"
        "assert pathlib.Path(scanner3d.__file__).resolve().is_relative_to(target)\n"
        "from scanner3d.cli import main\n"
        "raise SystemExit(main(['rgb-exposure-info', '--help']))\n"
    )
    help_run = subprocess.run(
        [sys.executable, "-I", "-c", help_code, str(target)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert help_run.returncode == 0, help_run.stderr
    assert "--execute-rgb-exposure-info-read" in help_run.stdout
    assert "GET_INFO" in help_run.stdout
    assert str(REPO_ROOT) not in help_run.stdout + help_run.stderr

    dry_code = (
        "import pathlib, sys\n"
        "target = pathlib.Path(sys.argv[1]).resolve()\n"
        "sys.path.insert(0, str(target))\n"
        "import scanner3d\n"
        "assert pathlib.Path(scanner3d.__file__).resolve().is_relative_to(target)\n"
        "from scanner3d.cli import main\n"
        "raise SystemExit(main(['rgb-exposure-info', '--json']))\n"
    )
    dry_run = subprocess.run(
        [sys.executable, "-I", "-c", dry_code, str(target)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert dry_run.returncode == 0, dry_run.stderr
    payload = json.loads(dry_run.stdout)
    assert payload["status"] == "not_attempted"
    assert payload["info_outcome"] == "not_attempted"
    assert payload["transfers"] == []
    assert payload["succeeded"] is False
    assert str(REPO_ROOT) not in dry_run.stdout + dry_run.stderr
