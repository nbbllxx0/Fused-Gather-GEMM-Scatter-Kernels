"""
filter_r2.py
------------
Structured-grid matrix-free cone density filter (task T5).

An explicit sparse filter matrix can be built from a
cKDTree neighbour query.  That is what makes the filter, not the operator,
the memory wall at large mesh sizes: at a *fixed physical* radius the cone
support grows as (r/h)^3, so an explicit filter needs roughly

    216k  ->  0.03 GB        4.9M ->  17.7 GB
    512k  ->  0.19 GB        8M   ->  46.8 GB
    1M    ->  0.73 GB        16M  -> 187 GB

On a uniform Cartesian grid none of that has to be stored.  The weights
w_ij = max(0, r_min - |x_i - x_j|) depend only on the offset between two
element centres, so the filter is a stencil and is applied on the fly.

Two operations are needed, and they are the same kernel:

    forward     rho_bar = (H rho) / s,      s_i = sum_j w_ij
    transpose   dc/drho = H (g / s)

because w is symmetric in i,j.  The row sums s are position-dependent only
through the domain boundary, so they are computed once per radius.

`r_min` is accepted in *element* units.  Which convention a study uses is
not cosmetic and is easy to leave undocumented: the continuation
schedule ramps r_min from 1.5 to 1.20 elements, so the physical filter
radius shrinks as the mesh is refined and no two rungs of the scaling ladder
solve the same regularised problem.  `physical_rmin_elements()` converts a
fixed physical radius into the element-unit radius for a given mesh, which
is what the mesh-comparability series uses.
"""

from __future__ import annotations

from .cuda_fused_matvec import _ascii

_TPL_STENCIL = r"""
extern "C" __global__ void __NAME__(
    const __T__* __restrict__ x,
    __T__*       __restrict__ y,
    const int nelx, const int nely, const int nelz,
    const int R, const __T__ rmin
) {
    const int n_elem = nelx * nely * nelz;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;

    const int ez = e % nelz;
    const int te = e / nelz;
    const int ey = te % nely;
    const int ex = te / nely;

    __T__ acc = (__T__)0;
    for (int dx = -R; dx <= R; ++dx) {
        const int jx = ex + dx;
        if (jx < 0 || jx >= nelx) continue;
        for (int dy = -R; dy <= R; ++dy) {
            const int jy = ey + dy;
            if (jy < 0 || jy >= nely) continue;
            for (int dz = -R; dz <= R; ++dz) {
                const int jz = ez + dz;
                if (jz < 0 || jz >= nelz) continue;
                const __T__ d = sqrt((__T__)(dx*dx + dy*dy + dz*dz));
                const __T__ w = rmin - d;
                if (w > (__T__)0) {
                    acc += w * x[(jx*nely + jy)*nelz + jz];
                }
            }
        }
    }
    y[e] = acc;
}
"""


def _instantiate(tpl, name, ctype):
    return tpl.replace("__NAME__", name).replace("__T__", ctype)


_SRC_STENCIL = (_instantiate(_TPL_STENCIL, "cone_stencil_fp32", "float")
                + _instantiate(_TPL_STENCIL, "cone_stencil_fp64", "double"))


def physical_rmin_elements(r_physical, domain_length, n_elements_along):
    """Element-unit radius that corresponds to a fixed physical radius."""
    h = float(domain_length) / float(n_elements_along)
    return float(r_physical) / h


class ConeFilter:
    """Matrix-free cone filter on a structured hexahedral grid."""

    BLOCK = 128

    def __init__(self, nelx, nely, nelz, rmin, dtype=None):
        import cupy as cp
        self.nelx, self.nely, self.nelz = int(nelx), int(nely), int(nelz)
        self.n_elem = self.nelx * self.nely * self.nelz
        self.dtype = dtype or cp.float64

        opts = ("-std=c++14",)
        mod = cp.RawModule(code=_ascii(_SRC_STENCIL), options=opts)
        self._k32 = mod.get_function("cone_stencil_fp32")
        self._k64 = mod.get_function("cone_stencil_fp64")
        self._mod = mod

        self._rmin = None
        self._s = None
        self.set_rmin(rmin)

    # -- radius ------------------------------------------------------------
    def set_rmin(self, rmin):
        """Set the radius (element units) and recompute the row sums."""
        import cupy as cp
        rmin = float(rmin)
        if self._rmin is not None and abs(rmin - self._rmin) < 1e-12:
            return
        self._rmin = rmin
        # Any offset with a component >= ceil(rmin) is at distance >= rmin,
        # so its weight is exactly zero; R = ceil(rmin) - 1 is therefore an
        # exact support bound, not a truncation.
        import math
        self._R = max(0, int(math.ceil(rmin)) - 1)
        ones = cp.ones(self.n_elem, dtype=self.dtype)
        self._s = self._stencil(ones)
        # s > 0 always: the centre offset (0,0,0) has weight rmin > 0.

    @property
    def rmin(self):
        return self._rmin

    @property
    def support_radius(self):
        return self._R

    def neighbours_per_element(self):
        """Number of non-zero weights for an interior element -- the size the
        equivalent explicit sparse filter would have to store."""
        import math
        R = self._R
        cnt = 0
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                for dz in range(-R, R + 1):
                    if self._rmin - math.sqrt(dx * dx + dy * dy + dz * dz) > 0:
                        cnt += 1
        return cnt

    # -- kernels -----------------------------------------------------------
    def _stencil(self, x):
        import cupy as cp
        import numpy as np
        x = cp.ascontiguousarray(x, dtype=self.dtype)
        y = cp.empty(self.n_elem, dtype=self.dtype)
        k = self._k64 if self.dtype == cp.float64 else self._k32
        grid = ((self.n_elem + self.BLOCK - 1) // self.BLOCK,)
        k(grid, (self.BLOCK,),
          (x, y, self.nelx, self.nely, self.nelz,
           self._R, np.dtype(self.dtype).type(self._rmin)))
        return y

    def forward(self, rho):
        """rho_bar = (H rho) / s."""
        return self._stencil(rho) / self._s

    def transpose(self, g):
        """Chain rule: given dc/d(rho_bar), return dc/d(rho)."""
        return self._stencil(g / self._s)

    # -- self-test ---------------------------------------------------------
    def check_adjoint(self, seed=0):
        """<H a, b> == <a, H^T b> to round-off -- guards the chain rule."""
        import cupy as cp
        rng = cp.random.RandomState(seed)
        a = rng.rand(self.n_elem).astype(self.dtype)
        b = rng.rand(self.n_elem).astype(self.dtype)
        lhs = float(cp.dot(self.forward(a), b))
        rhs = float(cp.dot(a, self.transpose(b)))
        denom = max(abs(lhs), abs(rhs), 1e-300)
        return abs(lhs - rhs) / denom
