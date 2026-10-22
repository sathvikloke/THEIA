"""Measured macOS storage, backing-image, power and session limits for NLST."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import threading
import time

try:
    from .nlst_resume_core import Refusal, utcnow
except ImportError:
    from nlst_resume_core import Refusal, utcnow


class StopSession(RuntimeError):
    """End work while preserving every attempt and partial."""


def run_command(argv: list[str], *, timeout_seconds: float = 15) -> bytes:
    try:
        return subprocess.check_output(argv, stderr=subprocess.PIPE, timeout=timeout_seconds)
    except (OSError, subprocess.SubprocessError) as exc:
        raise Refusal("Required resource probe failed: " + argv[0]) from exc


def power_is_ac(*, timeout_seconds: float = 15) -> bool:
    return b"Now drawing from 'AC Power'" in run_command(
        ["/usr/bin/pmset", "-g", "batt"], timeout_seconds=timeout_seconds,
    )


def mounted_volume(root: Path, expected_uuid: str) -> dict:
    root = root.resolve(strict=True)
    # diskutil accepts mount points/device identifiers, not arbitrary directories.
    requested_mount = root
    while not requested_mount.is_mount() and requested_mount != requested_mount.parent:
        requested_mount = requested_mount.parent
    info = plistlib.loads(run_command([
        "/usr/sbin/diskutil", "info", "-plist", str(requested_mount),
    ]))
    if info.get("VolumeUUID", "").lower() != expected_uuid.lower():
        raise Refusal("Volume UUID mismatch for " + str(root))
    mount = Path(info.get("MountPoint", "/")).resolve(strict=True)
    if mount == Path("/") or not root.is_relative_to(mount):
        raise Refusal("Expected an external mounted volume for " + str(root))
    device = root.stat().st_dev
    if device == Path("/").stat().st_dev or device != mount.stat().st_dev:
        raise Refusal("Storage path is not on the expected external volume")
    if not root.is_dir() or not os.access(root, os.W_OK):
        raise Refusal("Storage root is not writable: " + str(root))
    return {
        "root": str(root), "mount_point": str(mount), "device": device,
        "volume_uuid": info["VolumeUUID"],
        "free_bytes": shutil.disk_usage(root).free,
    }


def verify_backing_image(theia: dict, research: dict, backing_image: Path) -> str:
    image_path = backing_image.resolve(strict=True)
    if not image_path.is_relative_to(Path(research["root"])):
        raise Refusal("THEIA backing image is not under the verified Research root")
    if image_path.stat().st_dev != research["device"]:
        raise Refusal("Backing image is not on the verified Research device")
    info = plistlib.loads(run_command(["/usr/bin/hdiutil", "info", "-plist"]))
    for image in info.get("images", []):
        candidate = image.get("image-path")
        if not candidate or Path(candidate).resolve() != image_path:
            continue
        if any(
            entity.get("mount-point") == theia["mount_point"]
            for entity in image.get("system-entities", [])
        ):
            return str(image_path)
    raise Refusal("Mounted THEIA volume does not match the supplied backing image")


class ResourceGuard:
    def __init__(
        self, *, theia_root: Path, research_root: Path, backing_image: Path,
        theia_uuid: str, research_uuid: str, cutoff: dt.datetime,
        deadline: dt.datetime, reserve_theia: int, reserve_research: int,
        internal_root: Path | None = None, reserve_internal: int = 0,
    ):
        if reserve_theia < 0 or reserve_research < 0 or reserve_internal < 0:
            raise Refusal("Storage reserves cannot be negative")
        if not utcnow() < cutoff < deadline:
            raise Refusal("Require current UTC < cutoff UTC < deadline UTC")
        self.cutoff = cutoff
        self.deadline = deadline
        self.theia = mounted_volume(theia_root, theia_uuid)
        self.research = mounted_volume(research_root, research_uuid)
        self.backing_image = verify_backing_image(self.theia, self.research, backing_image)
        self.reserve_theia = reserve_theia
        self.reserve_research = reserve_research
        self.reserve_internal = reserve_internal
        self.internal = None
        if internal_root is not None:
            internal_root = internal_root.resolve(strict=True)
            self.internal = {
                "root": str(internal_root), "device": internal_root.stat().st_dev,
            }
        self.active: dict[str, int] = {}
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.stop_lock = threading.Lock()
        self.stop_reason: str | None = None
        self.last_power_check = float("-inf")
        self.check()

    def request_stop(self, reason: str) -> None:
        # Independent of the resource lock: a socket deadline must be able to
        # interrupt I/O while another worker is inside a resource probe.
        with self.stop_lock:
            if not self.stop.is_set():
                self.stop_reason = reason
                self.stop.set()

    def _fail(self, reason: str, cause: BaseException | None = None) -> None:
        self.request_stop(reason)
        raise StopSession(self.stop_reason or reason) from cause

    def _check_time_and_stop(self, new_work: bool) -> None:
        if self.stop.is_set():
            raise StopSession(self.stop_reason or "stop_requested")
        current = utcnow()
        if current >= self.deadline:
            self._fail("deadline_reached")
        if new_work and current >= self.cutoff:
            # Admission closes without stopping already admitted transfers.
            raise StopSession("cutoff_reached")

    def check(self, *, new_work: bool = False, extra_bytes: int = 0) -> dict:
        with self.lock:
            self._check_time_and_stop(new_work)
            if time.monotonic() - self.last_power_check >= 5:
                bound = self.cutoff if new_work else self.deadline
                budget = min(15.0, (bound - utcnow()).total_seconds())
                self._check_time_and_stop(new_work)
                try:
                    is_ac = power_is_ac(timeout_seconds=max(0.001, budget))
                except (OSError, Refusal) as exc:
                    self._check_time_and_stop(new_work)
                    self._fail("power_probe_failed", exc)
                self._check_time_and_stop(new_work)
                if not is_ac:
                    self._fail("ac_power_unavailable")
                self.last_power_check = time.monotonic()
            reservation = sum(self.active.values()) + extra_bytes
            snapshot = {"reserved_transfer_bytes": reservation}
            volumes = [
                ("theia", self.theia, self.reserve_theia, reservation),
                ("research", self.research, self.reserve_research, reservation),
            ]
            if self.internal is not None:
                volumes.append(("internal", self.internal, self.reserve_internal, 0))
            for label, volume, reserve, reserved in volumes:
                root = Path(volume["root"])
                try:
                    if root.stat().st_dev != volume["device"]:
                        self._fail(label + "_mount_identity_changed")
                    free = shutil.disk_usage(root).free
                except OSError as exc:
                    self._fail(label + "_mount_unavailable", exc)
                self._check_time_and_stop(new_work)
                if free < reserve + reserved:
                    self._fail(label + "_capacity_reserve")
                snapshot[label + "_free_bytes"] = free
            self._check_time_and_stop(new_work)
            return snapshot

    def reserve(self, key: str, byte_limit: int) -> None:
        with self.lock:
            if type(byte_limit) is not int or byte_limit < 1:
                raise Refusal("Transfer reservation must be a positive integer")
            self.check(new_work=True, extra_bytes=byte_limit)
            if key in self.active:
                raise Refusal("Duplicate active transfer reservation")
            self.active[key] = byte_limit

    def release(self, key: str) -> None:
        with self.lock:
            self.active.pop(key, None)

    def receipt(self) -> dict:
        return {
            "theia": self.theia, "research": self.research,
            "backing_image": self.backing_image,
            "reserve_theia_bytes": self.reserve_theia,
            "reserve_research_bytes": self.reserve_research,
            "internal": self.internal, "reserve_internal_bytes": self.reserve_internal,
            "ac_power_required": True, "capacity": self.check(),
        }
