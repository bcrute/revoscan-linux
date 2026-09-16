"""Opt-in GPU compute experiment, never imported by the live application.

Uses the existing Qt/OpenGL driver plus optional PyOpenGL. No ROCm, privilege
change or scanner access. Reference target buffers stay resident across ICP
iterations; reported wall time includes uploads, dispatch, synchronization,
readback and Python conversion. Both kernels use float64, not float32.

Example: .venv/bin/python tools/gpu_tracking_probe.py --capture CAPTURE --frames 90
The capture is read-only. --output creates a NEW JSON report, refusing overwrite.
This intentionally does not register or enable a production compute backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scanner3d import icp_plane
from scanner3d.capture_sequence import TRACKING_LOSS_STREAK
from scanner3d.compute import ComputeBackend, NearestHit, _ensure_gate, _ScipyBackend
from scanner3d.geometry import Point, normalize
from scanner3d.gui import PREVIEW_POINTS
from scanner3d.preparation import prepare_frame
from scanner3d.tracking import IncrementalTracker, TrackDecision
from scanner3d.v4l2_capture import EXPECTED_FRAME_BYTES

NEAREST_SHADER = """#version 430
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer Q { dvec4 queries[]; };
layout(std430,binding=1) readonly buffer T { dvec4 targets[]; };
layout(std430,binding=2) writeonly buffer O { dvec2 results[]; };
uniform uint target_count;
uniform double gate_squared;
shared double distances[64];
shared uint indices[64];
void main() {
    uint row = gl_WorkGroupID.x, lane = gl_LocalInvocationID.x;
    precise double best = gate_squared;
    uint winner = 0xffffffffu;
    for (uint j=lane; j<target_count; j+=64u) {
        precise dvec3 delta = queries[row].xyz - targets[j].xyz;
        precise double squared = (delta.x*delta.x + delta.y*delta.y) + delta.z*delta.z;
        if (squared < best || (squared == best && winner != 0xffffffffu && j < winner)) {
            best = squared; winner = j;
        }
    }
    distances[lane] = best; indices[lane] = winner;
    barrier();
    for (uint stride=32u; stride>0u; stride/=2u) {
        if (lane < stride) {
            uint other = lane + stride;
            if (distances[other] < distances[lane] ||
                (distances[other] == distances[lane] && indices[other] < indices[lane])) {
                distances[lane] = distances[other]; indices[lane] = indices[other];
            }
        }
        barrier();
    }
    if (lane == 0u) results[row] = dvec2(distances[0], double(indices[0]));
}
"""

# Cyclic rotations match the CPU _jacobi algorithm and order. The sum used
# solely for the convergence stop is not Python math.fsum; this is one reason
# this remains an accuracy-tested experiment rather than a claimed exact path.
JACOBI_SHADER = """#version 430
layout(local_size_x=64) in;
layout(std430,binding=0) readonly buffer M { double matrices[]; };
layout(std430,binding=2) writeonly buffer O { double results[]; };
uniform uint matrix_count;
// On this driver native fp64 division differed from CPU by about 1e-7
// relative, whereas sqrt/fma agreed on the tested inputs. Refine division
// with explicit fused residuals. This is measured, not a general proof of
// bitwise equivalence for arbitrary input, drivers, or subnormal values.
double divide64(double x, double y) {
    precise double q = x / y;
    for (int i=0; i<3; ++i) {
        precise double correction = fma(-q, y, x) / y;
        q = q + correction;
    }
    return q;
}
double sqrt64(double x) {
    return sqrt(x);
}
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= matrix_count) return;
    precise double a[9], v[9];
    for (int i=0; i<9; ++i) { a[i]=matrices[row*9u+uint(i)]; v[i]=(i%4==0)?1.0:0.0; }
    for (int sweep=0; sweep<64; ++sweep) {
        precise double off = (a[1]*a[1]+a[2]*a[2])+a[5]*a[5];
        if (off < 1.e-30) break;
        for (int p=0; p<2; ++p) for (int q=p+1; q<3; ++q) {
            if (abs(a[p*3+q]) < 1.e-30) continue;
            precise double theta = divide64(a[q*3+q]-a[p*3+p], 2.0*a[p*3+q]);
            precise double sign_theta = theta >= 0.0 ? 1.0 : -1.0;
            // 'precise' does not propagate through a function call. Keep
            // these intermediates explicit to prevent multiply/add
            // contraction in helper arguments from changing a pose.
            precise double theta_squared_plus_one = theta*theta+1.0;
            precise double t = divide64(sign_theta, abs(theta)+sqrt64(theta_squared_plus_one));
            precise double t_squared_plus_one = t*t+1.0;
            precise double c = divide64(1.0, sqrt64(t_squared_plus_one)), s = t*c;
            for (int k=0; k<3; ++k) {
                precise double akp=a[k*3+p], akq=a[k*3+q];
                a[k*3+p]=c*akp-s*akq; a[k*3+q]=s*akp+c*akq;
            }
            for (int k=0; k<3; ++k) {
                precise double apk=a[p*3+k], aqk=a[q*3+k];
                a[p*3+k]=c*apk-s*aqk; a[q*3+k]=s*apk+c*aqk;
            }
            for (int k=0; k<3; ++k) {
                precise double vkp=v[k*3+p], vkq=v[k*3+q];
                v[k*3+p]=c*vkp-s*vkq; v[k*3+q]=s*vkp+c*vkq;
            }
        }
    }
    for (int i=0; i<3; ++i) results[row*12u+uint(i)]=a[i*3+i];
    for (int i=0; i<9; ++i) results[row*12u+3u+uint(i)]=v[i];
}
"""


class GpuProbe:
    """Single-thread, explicitly closed offscreen context; no software fallback."""

    def __init__(self, *, surface: Any = None) -> None:
        import numpy as np
        from OpenGL import GL  # type: ignore[import-untyped]
        from OpenGL.GL.shaders import compileProgram, compileShader  # type: ignore[import-untyped]
        from PySide6.QtGui import (
            QGuiApplication,
            QOffscreenSurface,
            QOpenGLContext,
            QSurfaceFormat,
        )

        self.np, self.gl = np, GL
        self.app = QGuiApplication.instance() or QGuiApplication([])
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        self.context = QOpenGLContext()
        self.context.setFormat(fmt)
        if not self.context.create():
            raise RuntimeError("OpenGL 4.3 context creation failed")
        self.surface = surface
        if self.surface is None:
            # A live worker must receive a surface made on the GUI thread.
            self.surface = QOffscreenSurface()
            self.surface.setFormat(self.context.format())
            self.surface.create()
        if not self.context.makeCurrent(self.surface):
            raise RuntimeError("Cannot activate GPU context")
        self.renderer = GL.glGetString(GL.GL_RENDERER).decode()
        if any(name in self.renderer.lower() for name in ("llvmpipe", "softpipe", "swrast")):
            raise RuntimeError(f"Refusing software renderer: {self.renderer}")
        self.nearest_program = compileProgram(compileShader(NEAREST_SHADER, GL.GL_COMPUTE_SHADER))
        self.jacobi_program = compileProgram(compileShader(JACOBI_SHADER, GL.GL_COMPUTE_SHADER))
        self.input_buffer, self.output_buffer = map(int, GL.glGenBuffers(2))
        self.target_buffers: OrderedDict[tuple[Point, ...], int] = OrderedDict()
        self.calls = {"nearest": 0, "jacobi": 0, "target_uploads": 0}
        self.seconds = {"nearest": 0.0, "jacobi": 0.0}
        self._closed = False

    def _upload(self, buffer: int, binding: int, data: Any) -> None:
        gl = self.gl
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, data.nbytes, data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, binding, buffer)

    def _output(self, count: int) -> None:
        gl = self.gl
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.output_buffer)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, count * 8, None, gl.GL_STREAM_READ)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 2, self.output_buffer)

    def _read(self, count: int) -> Any:
        gl = self.gl
        gl.glMemoryBarrier(gl.GL_BUFFER_UPDATE_BARRIER_BIT)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.output_buffer)
        raw = gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, count * 8)
        return self.np.frombuffer(raw, dtype=self.np.float64)

    def _packed_points(self, points: Sequence[Point]) -> Any:
        packed = self.np.zeros((len(points), 4), dtype=self.np.float64)
        packed[:, :3] = points
        return packed

    def nearest_neighbors(
        self, queries: Sequence[Point], target: Sequence[Point], *, gate: float
    ) -> list[NearestHit]:
        _ensure_gate(gate)
        if not queries or not target:
            return [None for _ in queries]
        started = time.perf_counter()
        gl = self.gl
        key = tuple(target)
        target_buffer = self.target_buffers.get(key)
        if target_buffer is None:
            target_buffer = int(gl.glGenBuffers(1))
            self._upload(target_buffer, 1, self._packed_points(target))
            self.target_buffers[key] = target_buffer
            self.calls["target_uploads"] += 1
            if len(self.target_buffers) > 8:
                _, old = self.target_buffers.popitem(last=False)
                gl.glDeleteBuffers(1, [old])
        else:
            self.target_buffers.move_to_end(key)
            gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 1, target_buffer)
        self._upload(self.input_buffer, 0, self._packed_points(queries))
        self._output(len(queries) * 2)
        gl.glUseProgram(self.nearest_program)
        gl.glUniform1ui(gl.glGetUniformLocation(self.nearest_program, "target_count"), len(target))
        gl.glUniform1d(gl.glGetUniformLocation(self.nearest_program, "gate_squared"), gate * gate)
        gl.glDispatchCompute(len(queries), 1, 1)
        values = self._read(len(queries) * 2).reshape(-1, 2)
        result = [
            None if int(index) == 0xFFFFFFFF else (math.sqrt(squared), target[int(index)])
            for squared, index in values
        ]
        self.calls["nearest"] += 1
        self.seconds["nearest"] += time.perf_counter() - started
        return result

    def jacobi(
        self, matrices: Sequence[list[list[float]]]
    ) -> list[tuple[list[float], list[list[float]]]]:
        if not matrices:
            return []
        started = time.perf_counter()
        gl = self.gl
        packed = self.np.asarray(matrices, dtype=self.np.float64)
        self._upload(self.input_buffer, 0, packed)
        self._output(len(matrices) * 12)
        gl.glUseProgram(self.jacobi_program)
        gl.glUniform1ui(gl.glGetUniformLocation(self.jacobi_program, "matrix_count"), len(matrices))
        gl.glDispatchCompute((len(matrices) + 63) // 64, 1, 1)
        values = self._read(len(matrices) * 12).reshape(-1, 12)
        result = [(row[:3].tolist(), row[3:].reshape(3, 3).tolist()) for row in values]
        self.calls["jacobi"] += 1
        self.seconds["jacobi"] += time.perf_counter() - started
        return result

    def estimate_normals(
        self,
        points: Sequence[Point],
        *,
        neighbours: int = icp_plane.DEFAULT_NORMAL_NEIGHBOURS,
        backend: str | ComputeBackend = "scipy",
    ) -> icp_plane.NormalField:
        """CPU neighbourhoods/covariances, batched GPU eigensolves, CPU classification.

        Limited to the benchmark's supported cloud/parameter sizes. All sparse
        and invalid inputs delegate to the unmodified estimator. No new gates.
        """
        if (
            len(points) - 1 < icp_plane.MIN_NORMAL_NEIGHBOURS
            or neighbours < icp_plane.MIN_NORMAL_NEIGHBOURS
        ):
            return ORIGINAL_NORMALS(points, neighbours=neighbours, backend=backend)
        _, compute = icp_plane._resolve_backend(backend)
        if compute is None:
            return ORIGINAL_NORMALS(points, neighbours=neighbours, backend=backend)
        indices = compute.k_nearest_neighbors(points, neighbours)
        matrices = []
        for point, row in zip(points, indices, strict=True):
            local = [point, *(points[j] for j in row)]
            centre = (
                math.fsum(p[0] for p in local) / len(local),
                math.fsum(p[1] for p in local) / len(local),
                math.fsum(p[2] for p in local) / len(local),
            )
            matrices.append(icp_plane._covariance(local, centre))
        estimates = []
        coincident = isotropic = 0
        for point, row, (values, vectors) in zip(
            points, indices, self.jacobi(matrices), strict=True
        ):
            order = sorted(range(3), key=lambda i: values[i])
            smallest, largest = values[order[0]], values[order[2]]
            normal = None
            if largest <= 0.0:
                coincident += 1
            elif smallest / largest > icp_plane.NORMAL_ISOTROPY_RATIO:
                isotropic += 1
            else:
                normal = normalize(
                    (vectors[0][order[0]], vectors[1][order[0]], vectors[2][order[0]])
                )
            estimates.append(icp_plane.NormalEstimate(point, normal, len(row)))
        limitations = list(NORMAL_LIMITATIONS)
        if coincident:
            limitations.append(
                f"{coincident} of {len(points)} points had an entirely coincident neighbourhood."
            )
        if isotropic:
            limitations.append(
                f"{isotropic} of {len(points)} points had no dominant flat direction (ratio above "
                f"{icp_plane.NORMAL_ISOTROPY_RATIO}) - likely noise or a corner "
                "where surfaces meet - and "
                "were left without a normal."
            )
        return icp_plane.NormalField(tuple(estimates), neighbours, tuple(limitations))

    def close(self) -> None:
        if self._closed:
            return
        gl = self.gl
        buffers = [self.input_buffer, self.output_buffer, *self.target_buffers.values()]
        gl.glDeleteBuffers(len(buffers), buffers)
        gl.glDeleteProgram(self.nearest_program)
        gl.glDeleteProgram(self.jacobi_program)
        self.target_buffers.clear()
        self.context.doneCurrent()
        self._closed = True


ORIGINAL_NORMALS = icp_plane.estimate_normals
NORMAL_LIMITATIONS = ORIGINAL_NORMALS([], backend="python").limitations


def replay(
    clouds: Sequence[Sequence[Point]], gpu: GpuProbe, mode: str
) -> tuple[list[TrackDecision], list[float]]:
    icp_plane._cached_normals.cache_clear()
    tracker = IncrementalTracker(backend="scipy")
    decisions, elapsed = [], []
    streak = 0
    with ExitStack() as stack:
        if mode in ("nearest", "both"):
            stack.enter_context(
                patch.object(
                    _ScipyBackend,
                    "nearest_neighbors",
                    lambda _, *a, **k: gpu.nearest_neighbors(*a, **k),
                )
            )
        if mode in ("normals", "both"):
            stack.enter_context(patch.object(icp_plane, "estimate_normals", gpu.estimate_normals))
        for cloud in clouds:
            started = time.perf_counter()
            decision = tracker.offer(cloud)
            elapsed.append(time.perf_counter() - started)
            decisions.append(decision)
            streak = 0 if decision.accepted else streak + 1
            if streak == TRACKING_LOSS_STREAK:
                break
    icp_plane._cached_normals.cache_clear()
    return decisions, elapsed


def comparison(
    reference: Sequence[TrackDecision],
    candidate: Sequence[TrackDecision],
    clouds: Sequence[Sequence[Point]],
) -> dict[str, object]:
    disagreements, pose_errors = [], []
    for index, (a, b) in enumerate(zip(reference, candidate, strict=False)):
        if (a.accepted, a.origin, a.failure_type) != (b.accepted, b.origin, b.failure_type):
            disagreements.append(index)
        if a.pose is not None and b.pose is not None:
            pose_errors.append(
                max(math.dist(a.pose.apply(p), b.pose.apply(p)) for p in clouds[index])
            )
    return {
        "decision_disagreements": disagreements,
        "same_offered_count": len(reference) == len(candidate),
        "max_surface_pose_difference_mm": max(pose_errors, default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["cpu", "nearest", "normals", "both"],
        default=["cpu", "nearest", "normals", "both"],
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames <= 0 or args.modes[0] != "cpu":
        parser.error("Positive frame count and a CPU-first comparison are required")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; refusing overwrite")
    paths = sorted(args.capture.glob("frame_*.raw"))[: args.frames]
    if not paths:
        parser.error("No capture frames found")
    clouds, preparation_seconds = [], []
    for path in paths:
        raw = path.read_bytes()
        if len(raw) != EXPECTED_FRAME_BYTES:
            continue
        started = time.perf_counter()
        clouds.append(prepare_frame(raw, target_points=PREVIEW_POINTS).points)
        preparation_seconds.append(time.perf_counter() - started)
    gpu = GpuProbe()
    report: dict[str, Any] = {
        "capture": str(args.capture),
        "renderer": gpu.renderer,
        "jacobi_shader_sha256": hashlib.sha256(JACOBI_SHADER.encode()).hexdigest(),
        "nearest_shader_sha256": hashlib.sha256(NEAREST_SHADER.encode()).hexdigest(),
        "precision": "float64; refined division; explicit non-contracted intermediates",
        "scope": "experimental normal eigensolves and/or nearest search; other work remains CPU",
        "prepared": len(clouds),
        "preparation_seconds": preparation_seconds,
        "runs": [],
    }
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("runs", "preparation_seconds")}),
        flush=True,
    )
    try:
        # Compile/execute each kernel before measured work; cold cloud uploads
        # are still included in every actual tracking measurement.
        gpu.nearest_neighbors([(0.0, 0.0, 0.0)], [(1.0, 0.0, 0.0)], gate=2.0)
        gpu.jacobi([[[2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]])
        reference = None
        for mode in args.modes:
            previous_calls, previous_seconds = dict(gpu.calls), dict(gpu.seconds)
            decisions, elapsed = replay(clouds, gpu, mode)
            if reference is None:
                reference = decisions
            run = {
                "mode": mode,
                "offered": len(decisions),
                "accepted": sum(d.accepted for d in decisions),
                "median_tracking_s": statistics.median(elapsed),
                "total_tracking_s": sum(elapsed),
                "tracking_seconds": elapsed,
                "gpu_calls": {k: v - previous_calls[k] for k, v in gpu.calls.items()},
                "gpu_wall_seconds": {k: v - previous_seconds[k] for k, v in gpu.seconds.items()},
                **comparison(reference, decisions, clouds),
                "decisions": [d.as_dict() for d in decisions],
            }
            report["runs"].append(run)
            print(
                json.dumps(
                    {k: v for k, v in run.items() if k not in ("decisions", "tracking_seconds")}
                ),
                flush=True,
            )
    finally:
        gpu.close()
    if args.output is not None:
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)


if __name__ == "__main__":
    main()
