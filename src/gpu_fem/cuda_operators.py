"""
cuda_operators.py
-----------------
Matrix-free operator suite: every mapping of the K*v product the paper
compares, behind one interface.

Comparing a fused FP32 kernel against a three-stage FP64 path measures two
things at once -- fusion and halved precision -- and cannot attribute the
result to either.  So the suite spans the cross product deliberately:

  * matched precision.  Three-stage FP64, three-stage FP32, fused FP64 and
    fused FP32 exist as four separate paths, so a fusion speedup can be read
    at fixed precision rather than inferred from a mixture.
  * node-owned mapping.  A thread owns an output node and gathers from its
    incident elements.  There is no atomic scatter, no element-to-DOF table,
    and the accumulation order is fixed, so the result is bitwise
    reproducible.
  * analytic indexing.  On a structured Cartesian grid the 24 DOF indices
    follow from the element coordinates, so the edof table -- 96 B/element of
    storage and of read traffic -- is not needed at all.
  * declared precision per array.  The natural CuPy three-stage FP32 path
    scatters through cp.bincount, which returns float64 whatever the weights
    are; that path is therefore FP32 only in its GEMM.  A true-FP32-scatter
    variant is provided alongside it so the FP32 comparison is against a
    genuinely single-precision path.

Every path exposes the same signature so the microbenchmark and the parity
harness can treat them uniformly:

    suite = OperatorSuite(nelx, nely, nelz, KE_unit, edof=None)
    y_full = suite.matvec_full(u_full, E_e, path="fused_fp64")

`edof` is optional: the analytic-index and node-owned paths never touch it,
and building it is what limits the largest mesh that fits on a 24 GB card.

Index convention (must match pub_simp_solver._edof_table_3d exactly):
    element e = ex*(nely*nelz) + ey*nelz + ez
    node   n  = ix*(nely+1)*(nelz+1) + iy*(nelz+1) + iz
    local node order (ox,oy,oz):
        0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
        4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
    dof edof[e, 3*m+c] = 3*node_m + c
"""

from __future__ import annotations

from .cuda_fused_matvec import _ascii


# ─────────────────────────────────────────────────────────────────────────────
# T2 -- fused element-owned kernel, consistently FP64
#
# Same mapping as the shipped FP32 kernel: one thread per element, KE_unit
# broadcast to shared memory once per block, 24 atomic accumulations per
# element.  Only the arithmetic and the vector/KE/E precisions change, which
# is the whole point -- it isolates fusion from precision.
# ─────────────────────────────────────────────────────────────────────────────

_SRC_FUSED_FP64 = r"""
extern "C" __global__ void fused_matvec_fp64(
    const int*    __restrict__ edof,       // (n_elem, 24) int32
    const double* __restrict__ KE_global,  // (24, 24)     float64
    const double* __restrict__ E,          // (n_elem,)    float64
    const double* __restrict__ u_full,     // (ndof,)      float64
    double*       __restrict__ y_full,     // (ndof,)      float64, zeroed by caller
    const int n_elem
) {
    __shared__ double KE_s[24*24];          // 4608 B
    for (int idx = threadIdx.x; idx < 24*24; idx += blockDim.x) {
        KE_s[idx] = KE_global[idx];
    }
    __syncthreads();

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;

    int    dofs[24];
    double u_e[24];
    const int edof_base = e * 24;
    #pragma unroll
    for (int j = 0; j < 24; ++j) {
        int d   = edof[edof_base + j];
        dofs[j] = d;
        u_e[j]  = u_full[d];
    }

    const double Ee = E[e];
    #pragma unroll 4
    for (int i = 0; i < 24; ++i) {
        double acc = 0.0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) {
            acc += KE_s[i*24 + j] * u_e[j];
        }
        atomicAdd(&y_full[dofs[i]], Ee * acc);
    }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# T3 -- fused element-owned kernel with analytic indexing (no edof at all)
#
# The eight node indices of a structured hexahedral element are an affine
# function of its (ex,ey,ez) coordinates, so the 96 B/element index table is
# replaced by ~10 integer operations.
# ─────────────────────────────────────────────────────────────────────────────

_TPL_FUSED_AI = r"""
extern "C" __global__ void __NAME__(
    const __T__* __restrict__ KE_global,
    const __T__* __restrict__ E,
    const __T__* __restrict__ u_full,
    __T__*       __restrict__ y_full,
    const int n_elem, const int nelx, const int nely, const int nelz
) {
    __shared__ __T__ KE_s[24*24];
    for (int idx = threadIdx.x; idx < 24*24; idx += blockDim.x) {
        KE_s[idx] = KE_global[idx];
    }
    __syncthreads();

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;

    const int nny = nely + 1, nnz = nelz + 1;
    const int ez  = e % nelz;
    const int te  = e / nelz;
    const int ey  = te % nely;
    const int ex  = te / nely;

    const int sx = nny * nnz;
    const int sy = nnz;
    const int nb = ex*sx + ey*sy + ez;

    int nodes[8];
    nodes[0] = nb;            nodes[1] = nb + sx;
    nodes[2] = nb + sx + sy;  nodes[3] = nb + sy;
    nodes[4] = nb + 1;        nodes[5] = nb + sx + 1;
    nodes[6] = nb + sx+sy+1;  nodes[7] = nb + sy + 1;

    __T__ u_e[24];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int d0 = 3 * nodes[m];
        u_e[3*m + 0] = u_full[d0 + 0];
        u_e[3*m + 1] = u_full[d0 + 1];
        u_e[3*m + 2] = u_full[d0 + 2];
    }

    const __T__ Ee = E[e];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int d0 = 3 * nodes[m];
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            const int i = 3*m + c;
            __T__ acc = (__T__)0;
            #pragma unroll
            for (int j = 0; j < 24; ++j) {
                acc += KE_s[i*24 + j] * u_e[j];
            }
            atomicAdd(&y_full[d0 + c], Ee * acc);
        }
    }
}
"""


def _instantiate(tpl, name, ctype):
    return tpl.replace("__NAME__", name).replace("__T__", ctype)


_SRC_FUSED_AI = (_instantiate(_TPL_FUSED_AI, "fused_matvec_ai_fp32", "float")
                 + _instantiate(_TPL_FUSED_AI, "fused_matvec_ai_fp64", "double"))


# ─────────────────────────────────────────────────────────────────────────────
# Node-owned kernel
#
# One thread owns one output node and loops over the (at most) eight elements
# incident on it, accumulating the three output components in registers.  The
# consequences follow from the ownership rule itself, not from tuning:
#
#   * no atomicAdd anywhere -- each output DOF is written by exactly one
#     thread, so y_full does not even need to be zeroed first;
#   * bitwise run-to-run reproducibility, because the summation order is
#     fixed by the loop nest rather than by atomic arrival order;
#   * no element-to-DOF table.
#
# The cost is on the read side: the owner re-reads the displacement of every
# node in its 3x3x3 neighbourhood once per incident element, so the logical
# read volume is ~8x that of the element-owned mapping.  Those reads are
# heavily reused across neighbouring threads, so whether the trade is
# favourable is a measurement -- the operator microbenchmark and the
# end-to-end ablation -- rather than an argument.
#
# Owner's local-node index within an incident element, keyed by the offset
# (dx,dy,dz) = (ix-ex, iy-ey, iz-ez) packed as dx*4+dy*2+dz:
#     (0,0,0)->0 (0,0,1)->4 (0,1,0)->3 (0,1,1)->7
#     (1,0,0)->1 (1,0,1)->5 (1,1,0)->2 (1,1,1)->6
# ─────────────────────────────────────────────────────────────────────────────

_TPL_NODE = r"""
extern "C" __global__ void __NAME__(
    const __T__* __restrict__ KE_global,
    const __T__* __restrict__ E,
    const __T__* __restrict__ u_full,
    __T__*       __restrict__ y_full,
    const int nelx, const int nely, const int nelz
) {
    __shared__ __T__ KE_s[24*24];
    for (int idx = threadIdx.x; idx < 24*24; idx += blockDim.x) {
        KE_s[idx] = KE_global[idx];
    }
    __syncthreads();

    const int nnx = nelx + 1, nny = nely + 1, nnz = nelz + 1;
    const int n_node = nnx * nny * nnz;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_node) return;

    const int iz = n % nnz;
    const int tn = n / nnz;
    const int iy = tn % nny;
    const int ix = tn / nny;

    const int sx = nny * nnz;
    const int sy = nnz;

    const int LUT[8] = {0, 4, 3, 7, 1, 5, 2, 6};

    __T__ acc0 = (__T__)0, acc1 = (__T__)0, acc2 = (__T__)0;

    #pragma unroll
    for (int a = 0; a < 2; ++a) {
      const int ex = ix - 1 + a;
      if (ex >= 0 && ex < nelx) {
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
          const int ey = iy - 1 + b;
          if (ey >= 0 && ey < nely) {
            #pragma unroll
            for (int d = 0; d < 2; ++d) {
              const int ez = iz - 1 + d;
              if (ez >= 0 && ez < nelz) {
                const int e  = (ex*nely + ey)*nelz + ez;
                const __T__ Ee = E[e];
                const int li = LUT[(1-a)*4 + (1-b)*2 + (1-d)];
                const int r0 = (3*li + 0) * 24;
                const int r1 = (3*li + 1) * 24;
                const int r2 = (3*li + 2) * 24;
                const int nb = ex*sx + ey*sy + ez;
                int en[8];
                en[0] = nb;           en[1] = nb + sx;
                en[2] = nb + sx + sy; en[3] = nb + sy;
                en[4] = nb + 1;       en[5] = nb + sx + 1;
                en[6] = nb + sx+sy+1; en[7] = nb + sy + 1;
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                  const int d0 = 3 * en[m];
                  const __T__ u0 = u_full[d0 + 0];
                  const __T__ u1 = u_full[d0 + 1];
                  const __T__ u2 = u_full[d0 + 2];
                  const int j = 3 * m;
                  acc0 += Ee*(KE_s[r0+j]*u0 + KE_s[r0+j+1]*u1 + KE_s[r0+j+2]*u2);
                  acc1 += Ee*(KE_s[r1+j]*u0 + KE_s[r1+j+1]*u1 + KE_s[r1+j+2]*u2);
                  acc2 += Ee*(KE_s[r2+j]*u0 + KE_s[r2+j+1]*u1 + KE_s[r2+j+2]*u2);
                }
              }
            }
          }
        }
      }
    }

    const int d0 = 3 * n;
    y_full[d0 + 0] = acc0;
    y_full[d0 + 1] = acc1;
    y_full[d0 + 2] = acc2;
}
"""

_SRC_NODE = (_instantiate(_TPL_NODE, "node_matvec_fp32", "float")
             + _instantiate(_TPL_NODE, "node_matvec_fp64", "double"))


# ─────────────────────────────────────────────────────────────────────────────
# Fused element-compliance / sensitivity kernel
#
# c_e = u_e^T K_e^unit u_e, evaluated with analytic indexing and no
# intermediate element arrays.  This is the one kernel in the suite that
# exists only because there is a design problem: it is the adjoint pass of
# the compliance objective, and nothing outside optimization needs it.
#
# The shipped solver computes it as a materialised three-stage FP64 path
# (solver_v2.py:2310-2313):
#     Ue = U_gpu[edof]; KUe = Ue @ KE_unit; ce = (KUe*Ue).sum(1)
# i.e. exactly the pattern the paper removes for K*v and leaves in place for
# the gradient -- two (n_elem, 24) FP64 temporaries, ~970 B/element, once per
# design iteration.  Fusing it removes both temporaries and the index table.
# ─────────────────────────────────────────────────────────────────────────────

_TPL_CE = r"""
extern "C" __global__ void __NAME__(
    const __T__* __restrict__ KE_global,
    const __T__* __restrict__ u_full,
    __T__*       __restrict__ ce,
    const int n_elem, const int nelx, const int nely, const int nelz
) {
    __shared__ __T__ KE_s[24*24];
    for (int idx = threadIdx.x; idx < 24*24; idx += blockDim.x) {
        KE_s[idx] = KE_global[idx];
    }
    __syncthreads();

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;

    const int nny = nely + 1, nnz = nelz + 1;
    const int ez = e % nelz;
    const int te = e / nelz;
    const int ey = te % nely;
    const int ex = te / nely;
    const int sx = nny * nnz, sy = nnz;
    const int nb = ex*sx + ey*sy + ez;

    int nodes[8];
    nodes[0] = nb;            nodes[1] = nb + sx;
    nodes[2] = nb + sx + sy;  nodes[3] = nb + sy;
    nodes[4] = nb + 1;        nodes[5] = nb + sx + 1;
    nodes[6] = nb + sx+sy+1;  nodes[7] = nb + sy + 1;

    __T__ u_e[24];
    #pragma unroll
    for (int m = 0; m < 8; ++m) {
        const int d0 = 3 * nodes[m];
        u_e[3*m + 0] = u_full[d0 + 0];
        u_e[3*m + 1] = u_full[d0 + 1];
        u_e[3*m + 2] = u_full[d0 + 2];
    }

    __T__ acc = (__T__)0;
    #pragma unroll
    for (int i = 0; i < 24; ++i) {
        __T__ row = (__T__)0;
        #pragma unroll
        for (int j = 0; j < 24; ++j) {
            row += KE_s[i*24 + j] * u_e[j];
        }
        acc += row * u_e[i];
    }
    ce[e] = acc;
}
"""

_SRC_CE = (_instantiate(_TPL_CE, "elem_compliance_fp32", "float")
           + _instantiate(_TPL_CE, "elem_compliance_fp64", "double"))


# ─────────────────────────────────────────────────────────────────────────────
# Jacobi diagonal, node-owned and analytically indexed.
#
# diag(K)[3n+c] = sum over elements incident on node n of
#                 E_e * KE^unit[3*li+c, 3*li+c],
# where li is the owner's local-node index in that element.  Written this way
# it needs no edof table and no atomics, so it works for every path in the
# suite including the ones that never build an index table.
# ─────────────────────────────────────────────────────────────────────────────

_TPL_DIAG = r"""
extern "C" __global__ void __NAME__(
    const __T__* __restrict__ KE_global,
    const __T__* __restrict__ E,
    __T__*       __restrict__ diag,
    const int nelx, const int nely, const int nelz
) {
    const int nnx = nelx + 1, nny = nely + 1, nnz = nelz + 1;
    const int n_node = nnx * nny * nnz;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_node) return;

    const int iz = n % nnz;
    const int tn = n / nnz;
    const int iy = tn % nny;
    const int ix = tn / nny;

    const int LUT[8] = {0, 4, 3, 7, 1, 5, 2, 6};
    __T__ acc0 = (__T__)0, acc1 = (__T__)0, acc2 = (__T__)0;

    #pragma unroll
    for (int a = 0; a < 2; ++a) {
      const int ex = ix - 1 + a;
      if (ex >= 0 && ex < nelx) {
        #pragma unroll
        for (int b = 0; b < 2; ++b) {
          const int ey = iy - 1 + b;
          if (ey >= 0 && ey < nely) {
            #pragma unroll
            for (int d = 0; d < 2; ++d) {
              const int ez = iz - 1 + d;
              if (ez >= 0 && ez < nelz) {
                const int e  = (ex*nely + ey)*nelz + ez;
                const __T__ Ee = E[e];
                const int li = LUT[(1-a)*4 + (1-b)*2 + (1-d)];
                const int i0 = 3*li + 0, i1 = 3*li + 1, i2 = 3*li + 2;
                acc0 += Ee * KE_global[i0*24 + i0];
                acc1 += Ee * KE_global[i1*24 + i1];
                acc2 += Ee * KE_global[i2*24 + i2];
              }
            }
          }
        }
      }
    }
    const int d0 = 3 * n;
    diag[d0 + 0] = acc0;
    diag[d0 + 1] = acc1;
    diag[d0 + 2] = acc2;
}
"""

_SRC_DIAG = (_instantiate(_TPL_DIAG, "jacobi_diag_fp32", "float")
             + _instantiate(_TPL_DIAG, "jacobi_diag_fp64", "double"))


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Traffic model
#
# An arithmetic intensity is only meaningful next to the byte convention that
# produced it, and it is easy to end up quoting several.  Charging 8 B per
# gathered and scattered value while describing an FP32 kernel, then quoting
# an intensity that corresponds to the 4-byte model, then a third figure for a
# variant path, gives several accounts of one quantity and no way to tell which
# is meant.
#
# So: one convention, stated once, applied to every path and derived in code
# rather than typed in by hand.  Two models are reported, because they
# answer different questions and the difference between them is the whole of
# the cache story:
#
#   LOGICAL     every array access that crosses the kernel boundary is
#               counted, with no reuse assumed.  This is what Eq. (6) was
#               trying to express.  It is an upper bound on DRAM traffic.
#
#   COMPULSORY  every distinct array element crosses the boundary exactly
#               once.  This is a lower bound on DRAM traffic, reached only
#               with perfect caching.
#
# Measured DRAM bytes (Nsight `dram__bytes_*`) lie between the two, and are
# collected separately by ncu_dram_profile.py.  Neither model may be called
# "measured bandwidth" -- that is the reporting rule adopted here.
#
# The node-owned path is the reason both bounds are needed: its logical count
# is the largest of any path (it re-reads its whole 3x3x3 node neighbourhood
# once per incident element) while its compulsory count is the smallest, and
# it is in fact the fastest FP32 path measured.  Quoting either bound alone
# would misrepresent it by an order of magnitude.
# ─────────────────────────────────────────────────────────────────────────────

_PREC_BYTES = {"fp32": 4, "fp64": 8, "int32": 4, "int64": 8, "none": 0}

#: (vectors, KE_unit, E_e, index dtype, ownership)
PATH_ARRAYS = {
    "three_stage_fp64":     ("fp64", "fp64", "fp64", "int32", "three_stage"),
    "three_stage_fp64_e32": ("fp64", "fp64", "fp32", "int32", "three_stage"),
    "three_stage_fp32":     ("fp32", "fp32", "fp32", "int32", "three_stage"),
    "three_stage_fp32_s64": ("fp32", "fp32", "fp32", "int64", "three_stage"),
    "fused_fp64":           ("fp64", "fp64", "fp64", "int32", "element"),
    "fused_fp32":           ("fp32", "fp32", "fp32", "int32", "element"),
    "fused_ai_fp64":        ("fp64", "fp64", "fp64", "none",  "element"),
    "fused_ai_fp32":        ("fp32", "fp32", "fp32", "none",  "element"),
    "node_fp64":            ("fp64", "fp64", "fp64", "none",  "node"),
    "node_fp32":            ("fp32", "fp32", "fp32", "none",  "node"),
}


def traffic_terms(path):
    """Per-element byte breakdown for one path, both bounds, itemised.

    Per element, and using n_node ~= n_elem on a large grid so that a node
    quantity of 3 DOF is 3 values per element:

      element-owned fused : read 24 u, write 24 y (atomic), read 24 indices,
                            read 1 modulus
      three-stage         : additionally materialises u_elem and f_elem, so
                            the 24-vector is touched four more times, and the
                            index table is read twice (gather and scatter)
      node-owned          : reads 8 x 24 u (its 3x3x3 neighbourhood, once per
                            incident element), reads 8 moduli, writes 3 y,
                            reads no indices
    """
    vec, ke, ee, idx, own = PATH_ARRAYS[path]
    b = _PREC_BYTES[vec]
    be = _PREC_BYTES[ee]
    bi = _PREC_BYTES[idx]

    if own == "element":
        logical = {"u_read": 24 * b, "y_write": 24 * b,
                   "index_read": 24 * bi, "E_read": be}
    elif own == "three_stage":
        logical = {"u_read": 24 * b, "u_elem_write": 24 * b,
                   "u_elem_read": 24 * b, "f_elem_write": 24 * b,
                   "f_elem_read": 24 * b, "y_write": 24 * b,
                   "index_read": 2 * 24 * bi, "E_read": be}
    elif own == "node":
        logical = {"u_read": 8 * 24 * b, "y_write": 3 * b,
                   "index_read": 0, "E_read": 8 * be}
    else:                                                # pragma: no cover
        raise AssertionError(own)

    # Compulsory: each distinct value crosses once.  u and y are nodal (3 per
    # element-equivalent); the index table, where it exists, is genuinely
    # 24 entries of unique data per element.
    compulsory = {"u_read": 3 * b, "y_write": 3 * b,
                  "index_read": 24 * bi, "E_read": be}
    if own == "three_stage":
        # the two element arrays are unique per element and cannot be reused
        compulsory = dict(compulsory)
        compulsory["u_elem"] = 2 * 24 * b
        compulsory["f_elem"] = 2 * 24 * b

    return {
        "logical_terms": logical,
        "logical_B_per_elem": sum(logical.values()),
        "compulsory_terms": compulsory,
        "compulsory_B_per_elem": sum(compulsory.values()),
        # 576 madds = 1152 FLOP for the 24x24 local matvec, plus the 24
        # modulus-scaling multiplies.  A single 1152 denominator is not valid
        # across paths, so the count is carried per path.
        "FLOP_per_elem": 1176,
        "vectors": vec, "KE": ke, "E_e": ee, "index": idx, "ownership": own,
    }


#: Back-compatible tuple view used by the drivers.
PATH_SPEC = {
    p: (a[0], a[1], a[2], a[3],
        traffic_terms(p)["logical_B_per_elem"],
        traffic_terms(p)["FLOP_per_elem"])
    for p, a in PATH_ARRAYS.items()
}

#: Paths that need an explicit edof table on the device.
NEEDS_EDOF = {
    "three_stage_fp64", "three_stage_fp64_e32", "three_stage_fp32",
    "three_stage_fp32_s64", "fused_fp64", "fused_fp32",
}

ALL_PATHS = list(PATH_SPEC.keys())


class OperatorSuite:
    """Uniform access to every matrix-free K*v mapping compared here."""

    BLOCK_FP32 = 128
    BLOCK_FP64 = 128
    BLOCK_NODE = 128

    def __init__(self, nelx, nely, nelz, KE_unit, ndof=None, edof=None,
                 build_edof=True):
        import cupy as cp
        import numpy as np

        self.nelx, self.nely, self.nelz = int(nelx), int(nely), int(nelz)
        self.n_elem = self.nelx * self.nely * self.nelz
        self.nnx, self.nny, self.nnz = nelx + 1, nely + 1, nelz + 1
        self.n_node = self.nnx * self.nny * self.nnz
        self.ndof = int(ndof) if ndof is not None else 3 * self.n_node

        self.KE64 = cp.asarray(KE_unit, dtype=cp.float64)
        self.KE32 = self.KE64.astype(cp.float32)

        self._edof32 = None
        self._edof64 = None
        if edof is not None:
            self._edof32 = cp.asarray(edof, dtype=cp.int32).ravel()
        elif build_edof:
            self._edof32 = self.build_edof_device()

        opts = ("-std=c++14",)
        self._k = {}
        self._k["fused_fp64"] = cp.RawKernel(
            _ascii(_SRC_FUSED_FP64), "fused_matvec_fp64", options=opts)
        mod_ai = cp.RawModule(code=_ascii(_SRC_FUSED_AI), options=opts)
        self._k["fused_ai_fp32"] = mod_ai.get_function("fused_matvec_ai_fp32")
        self._k["fused_ai_fp64"] = mod_ai.get_function("fused_matvec_ai_fp64")
        mod_nd = cp.RawModule(code=_ascii(_SRC_NODE), options=opts)
        self._k["node_fp32"] = mod_nd.get_function("node_matvec_fp32")
        self._k["node_fp64"] = mod_nd.get_function("node_matvec_fp64")
        mod_ce = cp.RawModule(code=_ascii(_SRC_CE), options=opts)
        self._k["ce_fp32"] = mod_ce.get_function("elem_compliance_fp32")
        self._k["ce_fp64"] = mod_ce.get_function("elem_compliance_fp64")
        mod_dg = cp.RawModule(code=_ascii(_SRC_DIAG), options=opts)
        self._k["diag_fp32"] = mod_dg.get_function("jacobi_diag_fp32")
        self._k["diag_fp64"] = mod_dg.get_function("jacobi_diag_fp64")
        self._mod_ai, self._mod_nd = mod_ai, mod_nd
        self._mod_ce, self._mod_dg = mod_ce, mod_dg

        # Reuse the shipped FP32 fused kernel so the ablation's FP32 cell is
        # literally the kernel that was measured, not a re-write.
        from .cuda_fused_matvec import _KERNEL_SRC_FP32
        self._k["fused_fp32"] = cp.RawKernel(
            _ascii(_KERNEL_SRC_FP32), "fused_matvec_fp32", options=opts)

        self._buf = {}

    # -- index table -------------------------------------------------------
    def build_edof_device(self):
        """edof (n_elem*24,) int32 on device, matching _edof_table_3d exactly.

        Built on the GPU in chunks: the host-side meshgrid construction in
        pub_simp_solver allocates several int64 temporaries of size
        n_elem*24, which is what puts the 8 M mesh close to the memory wall
        before the solve even starts.
        """
        import cupy as cp
        nely, nelz, nny, nnz = self.nely, self.nelz, self.nny, self.nnz
        e = cp.arange(self.n_elem, dtype=cp.int32)
        ez = e % nelz
        te = e // nelz
        ey = te % nely
        ex = te // nely
        sx, sy = nny * nnz, nnz
        nb = ex * sx + ey * sy + ez
        offs = cp.asarray([0, sx, sx + sy, sy, 1, sx + 1, sx + sy + 1, sy + 1],
                          dtype=cp.int32)
        nodes = nb[:, None] + offs[None, :]                 # (n_elem, 8)
        edof = (3 * nodes[:, :, None]
                + cp.asarray([0, 1, 2], dtype=cp.int32)[None, None, :])
        return edof.reshape(self.n_elem, 24).ravel().astype(cp.int32)

    def edof_int64(self):
        """int64 copy required by cp.bincount -- the duplicate that costs
        1.5 GB at 8 M elements and that the analytic-index paths remove."""
        import cupy as cp
        if self._edof64 is None:
            self._edof64 = self._edof32.astype(cp.int64)
        return self._edof64

    def free_edof(self):
        self._edof32 = None
        self._edof64 = None

    # -- buffers -----------------------------------------------------------
    def _y(self, dt):
        import cupy as cp
        key = ("y", str(dt))
        if key not in self._buf:
            self._buf[key] = cp.zeros(self.ndof, dtype=dt)
        return self._buf[key]

    # -- the operator ------------------------------------------------------
    def matvec_full(self, u_full, E_e, path="fused_fp32", zero_output=True):
        """y = K(E) * u for the whole (unconstrained) DOF vector.

        Caller is responsible for zeroing u at fixed DOFs, exactly as in the
        shipped MatrixFreeKff.matvec.
        """
        import cupy as cp
        if path not in PATH_SPEC:
            raise ValueError(f"unknown path {path!r}; choose from {ALL_PATHS}")
        vec_prec = PATH_SPEC[path][0]
        dt = cp.float64 if vec_prec == "fp64" else cp.float32
        u = cp.ascontiguousarray(u_full, dtype=dt)
        E = cp.ascontiguousarray(E_e, dtype=dt)

        if path.startswith("three_stage"):
            return self._three_stage(u, E, path)

        y = self._y(dt)
        if zero_output and not path.startswith("node"):
            y.fill(0)

        KE = self.KE64 if dt == cp.float64 else self.KE32
        ne = self.n_elem

        if path in ("fused_fp32", "fused_fp64"):
            blk = self.BLOCK_FP32 if dt == cp.float32 else self.BLOCK_FP64
            grid = ((ne + blk - 1) // blk,)
            self._k[path](grid, (blk,),
                          (self._edof32, KE, E, u, y, ne))
        elif path in ("fused_ai_fp32", "fused_ai_fp64"):
            blk = self.BLOCK_FP32 if dt == cp.float32 else self.BLOCK_FP64
            grid = ((ne + blk - 1) // blk,)
            self._k[path](grid, (blk,),
                          (KE, E, u, y, ne, self.nelx, self.nely, self.nelz))
        elif path in ("node_fp32", "node_fp64"):
            blk = self.BLOCK_NODE
            grid = ((self.n_node + blk - 1) // blk,)
            self._k[path](grid, (blk,),
                          (KE, E, u, y, self.nelx, self.nely, self.nelz))
        else:
            raise AssertionError(path)
        return y

    # -- three-stage reference paths --------------------------------------
    def _three_stage(self, u, E, path):
        """Gather / batched GEMM / scatter, mirroring MatrixFreeKff.matvec.

        Three scatter conventions are exposed, because "FP32 three-stage" is
        ambiguous until the scatter's accumulation type is pinned down:

          three_stage_fp64      float64 state, bincount scatter (float64)
          three_stage_fp32_s64  float32 state, bincount scatter -- bincount
                                returns float64 regardless of the weights
                                dtype, so this path silently accumulates in
                                double.  It is the obvious way to write an
                                "FP32" three-stage path in CuPy, and it is not
                                one.
          three_stage_fp32      float32 state, true FP32 atomic scatter.
        """
        import cupy as cp
        import cupyx
        ne = self.n_elem
        dt = u.dtype
        KE = self.KE64 if dt == cp.float64 else self.KE32

        u_elem = u[self._edof32.reshape(ne, 24)]           # gather
        f_elem = (KE @ u_elem.T * E[None, :]).T            # batched GEMM

        if path == "three_stage_fp32":
            y = self._y(dt)
            y.fill(0)
            cupyx.scatter_add(y, self._edof32, f_elem.ravel())
            return y
        y = cp.bincount(self.edof_int64(),
                        weights=f_elem.ravel(),
                        minlength=self.ndof)
        return y.astype(dt, copy=False)

    # -- adjoint / element compliance -------------------------------------
    def elem_compliance(self, u_full, fused=True, dtype=None):
        """c_e = u_e^T K_e^unit u_e for every element.

        `fused=True` uses the analytic-index single-kernel path; `fused=False`
        reproduces the shipped materialised three-stage FP64 formulation, so
        the two can be compared for both accuracy and cost.
        """
        import cupy as cp
        dt = dtype or u_full.dtype
        u = cp.ascontiguousarray(u_full, dtype=dt)
        KE = self.KE64 if dt == cp.float64 else self.KE32

        if not fused:
            Ue = u[self._edof32.reshape(self.n_elem, 24)]
            KUe = Ue @ KE
            return (KUe * Ue).sum(axis=1)

        ce = cp.empty(self.n_elem, dtype=dt)
        k = self._k["ce_fp64" if dt == cp.float64 else "ce_fp32"]
        blk = 128
        grid = ((self.n_elem + blk - 1) // blk,)
        k(grid, (blk,), (KE, u, ce, self.n_elem,
                         self.nelx, self.nely, self.nelz))
        return ce

    # -- Jacobi diagonal ---------------------------------------------------
    def diagonal(self, E_e, path="fused_fp32", scatter=False):
        """diag(K), needed once per design iteration for the Jacobi preconditioner.

        Default is the node-owned analytic kernel: no index table, no atomics,
        deterministic.  `scatter=True` selects the element-owned scatter form
        for cross-checking the two against each other.

        Note for the scope reply: rebuilding this diagonal every time the
        coefficient field changes is *not* topology-optimization-specific.
        Any repeated-solve workload whose coefficients vary -- parametric
        studies, reliability analysis, inverse problems -- rebuilds it too.
        """
        import cupy as cp
        import cupyx
        dt = cp.float64 if PATH_SPEC[path][0] == "fp64" else cp.float32
        E = cp.ascontiguousarray(E_e, dtype=dt)
        KE = self.KE64 if dt == cp.float64 else self.KE32

        if scatter:
            contrib = (E[:, None] * KE.diagonal()[None, :]).ravel()
            d = cp.zeros(self.ndof, dtype=dt)
            cupyx.scatter_add(d, self._edof32, contrib)
            return d

        d = cp.empty(self.ndof, dtype=dt)
        k = self._k["diag_fp64" if dt == cp.float64 else "diag_fp32"]
        blk = 128
        grid = ((self.n_node + blk - 1) // blk,)
        k(grid, (blk,), (KE, E, d, self.nelx, self.nely, self.nelz))
        return d
