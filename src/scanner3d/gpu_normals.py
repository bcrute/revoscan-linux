"""Optional live FP64 normal eigensolves using the desktop OpenGL driver.

The surface is created/destroyed on the GUI thread; its context is owned and
closed by the capture worker. Neighbour selection, covariance arithmetic,
classification and all tracking gates retain their CPU reference definitions.
No device control, global monkey-patching, or compute-backend registration.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from scanner3d import icp_plane
from scanner3d.compute import ComputeBackend, select_backend
from scanner3d.geometry import Point, normalize

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

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


class GpuNormals:
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
        self.jacobi_program = self.input_buffer = self.output_buffer = 0
        self._closed = False
        try:
            self.renderer = GL.glGetString(GL.GL_RENDERER).decode()
            if any(name in self.renderer.lower() for name in ("llvmpipe", "softpipe", "swrast")):
                raise RuntimeError(f"Refusing software renderer: {self.renderer}")
            self.jacobi_program = compileProgram(compileShader(JACOBI_SHADER, GL.GL_COMPUTE_SHADER))
            self.input_buffer, self.output_buffer = map(int, GL.glGenBuffers(2))
        except BaseException:
            self.close()
            raise
        self.calls = {"jacobi": 0}
        self.seconds = {"jacobi": 0.0}

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

    def jacobi(
        self, matrices: Sequence[list[list[float]]] | NDArray[np.float64]
    ) -> list[tuple[list[float], list[list[float]]]]:
        if len(matrices) == 0:
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
        if not self.np.all(self.np.isfinite(values)):
            raise RuntimeError("GPU eigensolve returned non-finite values")
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
        patches, centres = [], []
        for point, row in zip(points, indices, strict=True):
            local = [point, *(points[j] for j in row)]
            centre = (
                math.fsum(p[0] for p in local) / len(local),
                math.fsum(p[1] for p in local) / len(local),
                math.fsum(p[2] for p in local) / len(local),
            )
            patches.append(local)
            centres.append(centre)
        # Keep fsum centroids, then evaluate the same covariance products and
        # sequential additions in batches. No pairwise sum or BLAS reordering.
        np = self.np
        delta = np.asarray(patches) - np.asarray(centres)[:, None, :]
        covariance = np.empty((len(points), 3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(a, 3):
                values = np.add.accumulate(delta[:, :, a] * delta[:, :, b], axis=1)[:, -1]
                covariance[:, a, b] = covariance[:, b, a] = values
        estimates = []
        coincident = isotropic = 0
        for point, row, (values, vectors) in zip(
            # Upload the already packed array directly. The previous list
            # round trip allocated four Python lists per normal and rebuilt
            # the identical float64 buffer in jacobi(), adding GC work.
            points,
            indices,
            self.jacobi(covariance),
            strict=True,
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
        try:
            buffers = [value for value in (self.input_buffer, self.output_buffer) if value]
            if buffers:
                gl.glDeleteBuffers(len(buffers), buffers)
            if self.jacobi_program:
                gl.glDeleteProgram(self.jacobi_program)
        finally:
            self.context.doneCurrent()
            self._closed = True


ORIGINAL_NORMALS = icp_plane.estimate_normals
NORMAL_LIMITATIONS = ORIGINAL_NORMALS([], backend="python").limitations


@contextmanager
def live_normals(surface: Any, notify: Callable[[str], None]) -> Iterator[None]:
    """Enable validated GPU normals on this worker; CPU fallback is explicit."""
    if select_backend("auto").resolved != "scipy":
        notify("GPU normals require SciPy; keeping the selected CPU processing path.")
        yield
        return
    gpu = None
    try:
        gpu = GpuNormals(surface=surface)
        # Driver arithmetic is not portable. Refuse acceleration unless this
        # context reproduces a deterministic set of reference eigensolves.
        rng = gpu.np.random.default_rng(51)
        matrices = []
        for _ in range(128):
            patch = rng.normal(size=(12, 3))
            matrices.append((patch.T @ patch).tolist())
        if gpu.jacobi(matrices) != [icp_plane._jacobi(m) for m in matrices]:
            raise RuntimeError("FP64 normal self-test disagrees with the CPU reference")
    except Exception as error:
        if gpu is not None:
            gpu.close()
        notify(f"GPU normals unavailable; CPU processing: {type(error).__name__}: {error}")
        yield
        return
    notify(f"GPU-assisted tracking: {gpu.renderer}; FP64 normals, SciPy correspondence search.")
    logging.getLogger(__name__).info("Live FP64 normal accelerator: %s", gpu.renderer)
    failed = False

    def estimate(points: tuple[Point, ...], neighbours: int) -> icp_plane.NormalField:
        nonlocal failed
        if not failed:
            try:
                return gpu.estimate_normals(points, neighbours=neighbours, backend="scipy")
            except Exception as error:
                failed = True
                notify(f"GPU normals failed; continuing on CPU: {type(error).__name__}: {error}")
        return ORIGINAL_NORMALS(points, neighbours=neighbours, backend="scipy")

    try:
        with icp_plane.normal_acceleration(estimate):
            yield
    finally:
        logging.getLogger(__name__).info(
            "GPU normal dispatches: %s; seconds: %s", gpu.calls, gpu.seconds
        )
        gpu.close()
