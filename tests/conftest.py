"""Fixture builders that synthesize a fake sysfs/dev tree on disk.

Every discovery test drives the real code paths against one of these trees, so
the tests exercise the same readers used against ``/sys`` rather than mocks.
The recorded descriptor blob in ``tests/fixtures/`` is the byte-for-byte
configuration descriptor of the physical Revopoint MINI used for this task.
"""

from __future__ import annotations

import atexit
import builtins
import io
import os
import shutil
import tempfile
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REAL_SERIAL = "EXAMPLESERIAL0001"


def _isolate_config_before_collection() -> Path:
    """Point the whole process at an empty configuration, at import time.

    This runs when pytest imports this conftest, which is BEFORE it imports a
    single test module - and that ordering is the entire point. Several modules
    evaluate ``AUTHORIZED_SERIAL = stated_serial()`` while they are being
    imported, so a fixture, however early, is already too late: by the time it
    runs, the constant has captured whatever the machine's own
    ``~/.config/scanner3d/config.toml`` said. On a developer who has recorded
    their own scanner's serial, tests unrelated to their change start failing,
    and a test asserting a default passes or fails depending on somebody's home
    directory.

    Only the process environment is touched, and only here. Tests that
    deliberately set these variables still override this with ``monkeypatch``,
    which is what makes the environment-precedence tests possible at all.
    """
    home = Path(tempfile.mkdtemp(prefix="scanner3d-config-isolation-"))
    os.environ["XDG_CONFIG_HOME"] = str(home)
    os.environ.pop("SCANNER3D_SERIAL", None)
    os.environ.pop("SCANNER3D_CAPTURES_DIR", None)
    atexit.register(shutil.rmtree, home, ignore_errors=True)
    return home


#: The empty config home every import-time constant is read against.
ISOLATED_CONFIG_HOME = _isolate_config_before_collection()


@pytest.fixture(autouse=True)
def no_live_scanner_opens(monkeypatch):
    """A missing fake must fail here, never operate the connected scanner.

    Explicit fake-driver fixtures replace these wrappers afterwards. This is
    a regression guard for Python device opens, not a subprocess/FFI sandbox.
    Physical diagnostics live in tools with their explicit execution gates.
    """

    def guarded(original):
        def opened(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                name = os.fsdecode(path)
                if name.startswith(("/dev/video", "/dev/bus/usb/", "/dev/media")):
                    raise AssertionError(
                        f"Unit test attempted live scanner open: {name}; install a fake"
                    )
            return original(path, *args, **kwargs)

        return opened

    monkeypatch.setattr(os, "open", guarded(os.open))
    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))


def descriptor_blob() -> bytes:
    return (FIXTURES / "mini-config-descriptor.bin").read_bytes()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class SysfsBuilder:
    """Builds a minimal sysfs + /dev tree under a tmp_path."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.sysfs = root / "sys"
        self.dev = root / "dev"
        self.devices = self.sysfs / "bus" / "usb" / "devices"
        self.devices.mkdir(parents=True, exist_ok=True)
        self.dev.mkdir(parents=True, exist_ok=True)

    def device_node(self, name: str, mode: int = 0o660) -> Path:
        """Create a stand-in device node (a regular file) under /dev."""
        path = self.dev / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        path.chmod(mode)
        return path

    def add_usb_device(
        self,
        usb_path: str = "1-4.1",
        *,
        vendor: str = "2207",
        product: str = "110c",
        serial: str | None = REAL_SERIAL,
        manufacturer: str | None = "Chishine3d",
        product_name: str | None = "ZXProduct",
        speed: str | None = "480",
        busnum: str = "1",
        devnum: str = "24",
        devpath: str = "4.1",
        bcd_device: str = "0419",
        version: str = "2.00",
        descriptors: bytes | None = None,
    ) -> Path:
        base = self.devices / usb_path
        base.mkdir(parents=True, exist_ok=True)
        attrs = {
            "idVendor": vendor,
            "idProduct": product,
            "busnum": busnum,
            "devnum": devnum,
            "devpath": devpath,
            "bcdDevice": bcd_device,
            "version": version,
        }
        if serial is not None:
            attrs["serial"] = serial
        if manufacturer is not None:
            attrs["manufacturer"] = manufacturer
        if product_name is not None:
            attrs["product"] = product_name
        if speed is not None:
            attrs["speed"] = speed
        for key, value in attrs.items():
            _write(base / key, value + "\n")
        blob = descriptor_blob() if descriptors is None else descriptors
        (base / "descriptors").write_bytes(blob)
        return base

    def add_interface(
        self,
        base: Path,
        number: int,
        *,
        iface_class: str = "0e",
        iface_subclass: str = "01",
        interface_string: str | None = None,
        driver: str | None = "uvcvideo",
        video_nodes: tuple[tuple[str, int], ...] = (),
        hidraw_nodes: tuple[str, ...] = (),
        node_mode: int = 0o660,
        v4l_name: str | None = None,
    ) -> Path:
        iface = base / f"{base.name}:1.{number}"
        iface.mkdir(parents=True, exist_ok=True)
        _write(iface / "bInterfaceNumber", f"{number:02x}\n")
        _write(iface / "bInterfaceClass", iface_class + "\n")
        _write(iface / "bInterfaceSubClass", iface_subclass + "\n")
        if interface_string is not None:
            _write(iface / "interface", interface_string + "\n")
        if driver is not None:
            driver_target = self.sysfs / "bus" / "usb" / "drivers" / driver
            driver_target.mkdir(parents=True, exist_ok=True)
            link = iface / "driver"
            if not link.exists():
                link.symlink_to(driver_target)
        for node_name, index in video_nodes:
            node_dir = iface / "video4linux" / node_name
            node_dir.mkdir(parents=True, exist_ok=True)
            _write(node_dir / "index", f"{index}\n")
            _write(node_dir / "name", (v4l_name or f"ZXProduct: {node_name}") + "\n")
            self.device_node(node_name, mode=node_mode)
        for hid_name in hidraw_nodes:
            hid_dir = iface / "0003:2207:110C.000F" / "hidraw" / hid_name
            hid_dir.mkdir(parents=True, exist_ok=True)
            self.device_node(hid_name, mode=0o600)
        return iface

    def add_usbfs(self, bus: int = 1, device: int = 24, mode: int = 0o660) -> Path:
        path = self.dev / "bus" / "usb" / f"{bus:03d}" / f"{device:03d}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        path.chmod(mode)
        return path

    def add_healthy_mini(self, **device_kwargs: object) -> Path:
        """The full happy-path device as observed on real hardware."""
        serial = device_kwargs.get("serial", REAL_SERIAL)
        base = self.add_usb_device(**device_kwargs)  # type: ignore[arg-type]
        self.add_usbfs()
        self.add_interface(
            base,
            0,
            interface_string=f"DepthCam{serial}",
            video_nodes=(("video0", 0), ("video1", 1)),
        )
        self.add_interface(base, 1, iface_subclass="02", interface_string="Video Streaming")
        self.add_interface(
            base,
            2,
            interface_string=f"RGBCam{serial}",
            video_nodes=(("video2", 0), ("video3", 1)),
        )
        self.add_interface(base, 3, iface_subclass="02", interface_string="Video Streaming")
        self.add_interface(
            base,
            4,
            iface_class="03",
            iface_subclass="00",
            interface_string="HID Interface",
            driver="usbhid",
            hidraw_nodes=("hidraw10",),
        )
        return base


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path_factory):
    """Give every test its own empty configuration, and no environment overrides.

    :func:`_isolate_config_before_collection` has already detached this process
    from the host's configuration; this gives each test a fresh directory on top
    of that, so a test that writes a config file cannot change what the next one
    reads. Both are needed: that one runs early enough for import-time
    constants, this one runs often enough for isolation between tests.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("config-home")))
    monkeypatch.delenv("SCANNER3D_SERIAL", raising=False)
    monkeypatch.delenv("SCANNER3D_CAPTURES_DIR", raising=False)


@pytest.fixture
def readiness_stood_down(monkeypatch):
    """Let a test past the reconstruction readiness gates, ON PURPOSE and by name.

    :mod:`scanner3d.reconstruction_readiness` refuses a reconstruction whose
    poses do not describe a scan, and small synthetic fixtures usually do not:
    nine renders across twenty-four degrees, or four identical frames placed at
    the identity, are manufactured pose sequences and the gate is right about
    them. That verdict is correct and completely beside the point for a test
    asking what the CLI put in its JSON payload or what the shared preparation
    path reported.

    So this replaces the evaluation with a ready verdict. Every number in it is
    one a real evaluation could have produced, because the strict reader
    derives the required reason set from the evidence: a stand-in whose
    evidence would have obliged a refusal is not a ready record however empty
    its failure list is. It is deliberately NOT autouse: the real gate has to
    be the default everywhere, and what it does on each front end is proved
    against fixtures built to be judged - in
    :mod:`tests.test_reconstruction_readiness`,
    :mod:`tests.test_reconstruction_export_readiness`,
    :mod:`tests.test_gui_readiness_refusal` and
    :mod:`tests.test_cli_readiness`.
    """
    from scanner3d import reconstruction_readiness

    policy = reconstruction_readiness.DEFAULT_POLICY

    def ready(*_args: object, **_kwargs: object) -> object:
        return reconstruction_readiness.ReconstructionReadiness(
            policy=policy,
            tracking=reconstruction_readiness.TrackingEvidence(0, 0, unresolved=False, detail=""),
            poses=reconstruction_readiness.PoseEvidence(
                poses=4,
                adjacent_steps=3,
                moving_steps=3,
                stationary_steps=0,
                travel_median=10.0 * policy.noise_floor,
                travel_worst=10.0 * policy.noise_floor,
                distinct_viewpoints=4,
                viewpoints_capped=False,
                excursion=20.0 * policy.noise_floor,
                probe_points=policy.probe_points,
            ),
            coverage=reconstruction_readiness.CoverageEvidence(
                measured=True,
                frames_offered=4,
                frames_used=4,
                points_used=400,
                cell_size=1.0,
                total_cells=10,
                supported_cells=10,
                supported_fraction=1.0,
                detail="stood down by the readiness_stood_down fixture",
            ),
            surface=None,
            failures=(),
        )

    monkeypatch.setattr(reconstruction_readiness, "evaluate_readiness", ready)


@pytest.fixture
def builder(tmp_path: Path) -> SysfsBuilder:
    return SysfsBuilder(tmp_path)


@pytest.fixture
def healthy(builder: SysfsBuilder) -> SysfsBuilder:
    builder.add_healthy_mini()
    return builder


@pytest.fixture
def empty_sysfs(builder: SysfsBuilder) -> SysfsBuilder:
    return builder
