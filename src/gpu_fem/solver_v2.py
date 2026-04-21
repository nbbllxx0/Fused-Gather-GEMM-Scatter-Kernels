"""
solver_v2.py
------------
Phase 1 solver stack for GPU-FEM-Accel (paper 2).

Drop-in replacement for GPUFEMSolver._solve_cupy / _solve_torch.
All improvements are behind feature flags so that compliance parity
can be verified one step at a time before the next flag is enabled.

Phase 1 techniques implemented here
------------------------------------
1.1  Warm-start CG (enable_warm_start=True, default ON)
     Persist u_prev between SIMP iterations.  The density field changes
     slowly, so u_{k-1} is a very good x0 for the next CG solve.
     Expected gain: 30–60% fewer CG iterations on iterations 2+.

1.2B HexGridGMG — assembled Galerkin geometric multigrid for structured
     3D hex grids (enable_gmg=True, default OFF until parity verified).
     Trilinear prolongation, Galerkin coarse operator built once per SIMP
     iteration via GPU sparse triple product, damped-Jacobi smoother,
     GPU throughout (CuPy CSR).
     Expected gain: 3–8× over Jacobi once validated.

1.3  Mixed-precision iterative refinement (enable_mixed_precision=True,
     default OFF until parity verified).
     FP32 inner CG exploits RTX 4090's 82 TFLOPS FP32 ceiling.
     FP64 outer residual correction preserves accuracy.
     Expected gain: 1.5–2× on top of 1.2.

1.4  CUDA graph capture — TODO (week 4 of Phase 1).

Usage
-----
    from gpu_fem.solver_v2 import SolverV2

    solver = SolverV2(
        edof=edof, row_idx=row_idx, col_idx=col_idx, KE_UNIT=KE_UNIT,
        free=free, F=F, ndof=ndof,
        grid_dims=(nelx, nely, nelz),   # required for enable_gmg=True
        enable_warm_start=True,
        enable_gmg=False,               # flip to True after parity check
        enable_mixed_precision=False,
    )
    compliance, dc = solver.solve(rho_phys, penal)
"""

from __future__ import annotations

import gc
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .fem_gpu import GPUFEMSolver


# ─────────────────────────────────────────────────────────────────────────────
# § 1.2B  Geometric Multigrid: intergrid transfer operators
# ─────────────────────────────────────────────────────────────────────────────

def _build_scalar_prolongation(
    nelx_f: int, nely_f: int, nelz_f: int,
    nelx_c: int, nely_c: int, nelz_c: int,
) -> sp.csr_matrix:
    """
    Build trilinear prolongation from coarse-node to fine-node scalar field.

    Returns scipy CSR matrix P of shape (n_fine_nodes, n_coarse_nodes).

    For 3-component vector fields use kron(P, I_3) — see _build_vector_prolongation.

    Node ordering (both levels):
        node(i, j, k) = i * (ny+1) * (nz+1) + j * (nz+1) + k
        where nx = nelx, ny = nely, nz = nelz (element counts, not node counts).

    Trilinear interpolation weights:
        - fine node at even (i,j,k): weight 1 for the one parent coarse node
        - fine node at odd  i: weight 0.5 for coarse nodes at floor(i/2) and ceil(i/2)
        same for j, k independently → up to 2^3 = 8 coarse parents per fine node.
    """
    nx_fn, ny_fn, nz_fn = nelx_f + 1, nely_f + 1, nelz_f + 1
    nx_cn, ny_cn, nz_cn = nelx_c + 1, nely_c + 1, nelz_c + 1
    n_fine   = nx_fn * ny_fn * nz_fn
    n_coarse = nx_cn * ny_cn * nz_cn

    # Vectorised mesh of fine node indices
    ii, jj, kk = np.meshgrid(
        np.arange(nx_fn, dtype=np.int32),
        np.arange(ny_fn, dtype=np.int32),
        np.arange(nz_fn, dtype=np.int32),
        indexing='ij',
    )
    fine_idx = (ii * ny_fn + jj) * nz_fn + kk   # shape (nx_fn, ny_fn, nz_fn)

    rows_list, cols_list, data_list = [], [], []

    # Iterate over 8 (di, dj, dk) ∈ {0,1}^3 combinations
    # di=0: lower coarse parent in x, di=1: upper coarse parent (only if i is odd)
    for di in (0, 1):
        ic = (ii >> 1) + di                          # floor(i/2) + di
        # Weight in x:
        #   even i → wx = 1.0 (di=0) or 0.0 (di=1)
        #   odd  i → wx = 0.5 for both di
        wx = np.where(ii & 1, 0.5, np.where(di == 0, 1.0, 0.0))
        # Valid: even i contributes only when di=0; odd i contributes for both
        valid_x = (ii & 1).astype(bool) | (di == 0)
        in_x    = (ic >= 0) & (ic < nx_cn)

        for dj in (0, 1):
            jc = (jj >> 1) + dj
            wy = np.where(jj & 1, 0.5, np.where(dj == 0, 1.0, 0.0))
            valid_y = (jj & 1).astype(bool) | (dj == 0)
            in_y    = (jc >= 0) & (jc < ny_cn)

            for dk in (0, 1):
                kc = (kk >> 1) + dk
                wz = np.where(kk & 1, 0.5, np.where(dk == 0, 1.0, 0.0))
                valid_z = (kk & 1).astype(bool) | (dk == 0)
                in_z    = (kc >= 0) & (kc < nz_cn)

                mask = valid_x & valid_y & valid_z & in_x & in_y & in_z
                if not mask.any():
                    continue

                coarse_idx = (ic * ny_cn + jc) * nz_cn + kc
                rows_list.append(fine_idx[mask].ravel())
                cols_list.append(coarse_idx[mask].ravel())
                data_list.append((wx * wy * wz)[mask].ravel())

    rows = np.concatenate(rows_list).astype(np.int32)
    cols = np.concatenate(cols_list).astype(np.int32)
    data = np.concatenate(data_list).astype(np.float64)

    P = sp.csr_matrix((data, (rows, cols)), shape=(n_fine, n_coarse), dtype=np.float64)
    # Deduplicate (shouldn't be needed with the valid_x/y/z masks, but is safe)
    P.sum_duplicates()

    # Row-normalize so every row sums to 1.
    # Without normalization, fine nodes at an odd boundary index (e.g. iz=7 when
    # nz_c = nz_f // 2 = 3) can only reach one of their two coarse parents (the
    # other is out-of-bounds), leaving the row sum at 0.5 instead of 1.0.
    # This breaks the partition-of-unity property and degrades the V-cycle,
    # causing CG non-convergence at multiple levels when n_levels ≥ 4.
    # The fix: clamp each row to sum to 1 by scaling with 1/row_sum where positive.
    row_sums = np.asarray(P.sum(axis=1)).ravel()
    row_sums_safe = np.where(row_sums > 1e-12, row_sums, 1.0)
    P = sp.diags(1.0 / row_sums_safe, format="csr") @ P
    return P


def _coarse_free_dofs(
    P_vec: sp.csr_matrix,
    free_fine: np.ndarray,
) -> np.ndarray:
    """
    Identify free DOFs at the coarse level.

    A coarse DOF is free iff it has at least one fine free-DOF descendant
    in the prolongation (P_vec[free_fine, coarse_dof].max() > 0).

    Returns sorted int32 array of coarse DOF indices in [0, n_coarse_dof).
    """
    n_fine_dof = P_vec.shape[0]
    fine_mask  = np.zeros(n_fine_dof, dtype=np.float64)
    fine_mask[free_fine] = 1.0
    coarse_contribution = P_vec.T @ fine_mask       # (n_coarse_dof,)
    return np.where(coarse_contribution > 1e-9)[0].astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# § 1.2B  HexGridGMG class
# ─────────────────────────────────────────────────────────────────────────────

class HexGridGMG:
    """
    Assembled Galerkin geometric multigrid for structured 3D hexahedral grids.

    Construction
    ------------
    Builds static trilinear prolongation/restriction matrices for each level
    pair (CPU, scipy CSR).  These are uploaded to GPU on the first setup() call.

    Per-SIMP-iteration: setup(K_fine)
    ---------------------------------
    Builds coarse Galerkin operators K_c = P^T K_f P via CuPy sparse triple
    products.  Stores diagonal inverse arrays for damped-Jacobi smoother.

    Per-CG-iteration: vcycle(r) → z
    --------------------------------
    One full V-cycle (pre-smooth → restrict → coarse solve → prolongate →
    post-smooth) entirely on GPU.  Coarsest level solved via CuPy spsolve
    (GPU direct) or scipy spsolve (CPU fallback).

    Parameters
    ----------
    nelx, nely, nelz : int
        Fine-level element counts.
    free : (n_free,) int32
        Free DOF indices into the fine-level global DOF vector
        (3*(nelx+1)*(nely+1)*(nelz+1) DOFs total).
    n_levels : int
        Number of levels (including fine).  Default 3.
    n_smooth : int
        Damped-Jacobi pre/post-smoothing sweeps per level.  Default 2.
    omega : float
        Jacobi damping factor.  2/3 is standard for regular hex grids.
    """

    def __init__(
        self,
        nelx: int, nely: int, nelz: int,
        free: np.ndarray,
        n_levels: int = 3,
        n_smooth: int = 2,
        omega: float = 2.0 / 3.0,
    ) -> None:
        self._n_levels = n_levels
        self._n_smooth = n_smooth
        self._omega    = omega

        # ── Compute element counts at each level ──────────────────────────
        dims: list[tuple[int, int, int]] = [(nelx, nely, nelz)]
        for _ in range(1, n_levels):
            nx, ny, nz = dims[-1]
            dims.append((max(2, nx // 2), max(2, ny // 2), max(2, nz // 2)))
        self._dims = dims

        # ── Build prolongation matrices and free-DOF lists per level ──────
        self._free_per_level: list[np.ndarray] = [free]
        self._n_free: list[int] = [len(free)]
        self._P_scipy: list[sp.csr_matrix] = []

        for lv in range(n_levels - 1):
            nx_f, ny_f, nz_f = dims[lv]
            nx_c, ny_c, nz_c = dims[lv + 1]
            free_f = self._free_per_level[lv]

            # Scalar node prolongation: fine → coarse (shape: n_fine_nodes × n_coarse_nodes)
            P_sc = _build_scalar_prolongation(nx_f, ny_f, nz_f, nx_c, ny_c, nz_c)

            # Lift to DOF prolongation via kron(P_sc, I_3)
            P_vec = sp.kron(P_sc, sp.eye(3, format='csr', dtype=np.float64), format='csr')
            del P_sc

            # Determine free DOFs at coarse level
            free_c = _coarse_free_dofs(P_vec, free_f)
            self._free_per_level.append(free_c)
            self._n_free.append(len(free_c))

            # Restrict prolongation to free-DOF subspace: P_free[free_f, :][:, free_c]
            P_free = P_vec[free_f, :][:, free_c]
            self._P_scipy.append(P_free.tocsr())
            del P_vec, P_free
            gc.collect()

        # ── GPU arrays (populated on first setup() call) ──────────────────
        self._P_gpu  : list[Optional[object]] = [None] * (n_levels - 1)
        self._R_gpu  : list[Optional[object]] = [None] * (n_levels - 1)
        self._gpu_uploaded = False

        # Per-SIMP-iteration state
        self._K_gpu        : list[Optional[object]] = [None] * n_levels
        self._diag_inv_gpu : list[Optional[object]] = [None] * n_levels
        self._coarse_K_cpu : Optional[sp.csr_matrix] = None
        # Cached LU factorization of the coarsest matrix.
        # Computed once per SIMP iteration in setup(); reused across all V-cycles
        # within that iteration.  Avoids O(n^2.7) refactorization per V-cycle.
        self._coarse_lu    = None   # callable: rhs -> solution (scipy SuperLU)
        self._is_setup     = False

    # ------------------------------------------------------------------
    # Per-SIMP-iteration setup
    # ------------------------------------------------------------------

    def setup(self, K_fine_cupy, fine_diag=None) -> None:
        """
        Build coarse Galerkin operators for the current stiffness matrix.

        K_fine_cupy : cupyx.scipy.sparse.csr_matrix, shape (n_free, n_free).
        fine_diag   : optional precomputed diagonal of K_fine_cupy (cp.ndarray).
                      Pass Kff.data[diag_pos_gpu] from SolverV2 to skip the slow
                      O(nnz) .diagonal() search.
        Called once per SIMP iteration (when K changes with the density field).
        """
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        # Upload static P matrices to GPU once
        if not self._gpu_uploaded:
            for lv in range(self._n_levels - 1):
                self._P_gpu[lv] = cpsp.csr_matrix(self._P_scipy[lv])
                self._R_gpu[lv] = self._P_gpu[lv].T.tocsr()
            self._gpu_uploaded = True

        # Level 0: the given fine-level matrix
        self._K_gpu[0] = K_fine_cupy
        # Use caller-supplied diagonal when available (avoids O(nnz) CSR search).
        diag0 = fine_diag if fine_diag is not None else K_fine_cupy.diagonal()
        diag0 = cp.where(cp.abs(diag0) > 1e-12, diag0, cp.ones_like(diag0))
        self._diag_inv_gpu[0] = 1.0 / diag0

        # Galerkin coarse operators: K_{l+1} = R_l K_l P_l
        # Memory management: cuSPARSE SpGEMM allocates a large workspace (~1–2 GB for
        # the fine level) that CuPy retains in its pool.  Without explicit release, the
        # pool accumulates ~2 GB per SIMP iteration and exhausts 24 GB VRAM in ~6 iters.
        # Fix: delete intermediates immediately and release the pool after each level.
        pool = cp.get_default_memory_pool()
        for lv in range(self._n_levels - 1):
            K_f = self._K_gpu[lv]
            P   = self._P_gpu[lv]
            R   = self._R_gpu[lv]
            # Triple product on GPU via CuPy cuSPARSE.
            # Parenthesise right-first: K_f @ P is (n_fine × n_coarse) with bounded
            # sparsity; R @ that is the cheap (n_coarse × n_coarse) final product.
            # Left-first (R @ K_f) would create an (n_coarse × n_fine) intermediate
            # which has far more nonzeros.
            KfP = K_f @ P               # intermediate — delete ASAP
            K_c = (R @ KfP).tocsr()
            del KfP                     # free the large intermediate from the pool
            pool.free_all_blocks()      # return freed blocks to the OS / driver

            self._K_gpu[lv + 1] = K_c
            diag_c = K_c.diagonal()
            diag_c = cp.where(cp.abs(diag_c) > 1e-12, diag_c, cp.ones_like(diag_c))
            self._diag_inv_gpu[lv + 1] = 1.0 / diag_c

        # Keep coarsest K on CPU; precompute its LU factorization (SuperLU).
        # Factorizing here (once per SIMP iter) lets each V-cycle use only
        # cheap triangular solves instead of re-factorizing from scratch.
        K_coarsest_gpu = self._K_gpu[self._n_levels - 1]
        self._coarse_K_cpu = sp.csr_matrix((
            cp.asnumpy(K_coarsest_gpu.data),
            cp.asnumpy(K_coarsest_gpu.indices),
            cp.asnumpy(K_coarsest_gpu.indptr),
        ), shape=K_coarsest_gpu.shape)
        from scipy.sparse.linalg import factorized as _factorized
        self._coarse_lu = _factorized(self._coarse_K_cpu)   # callable: b -> x
        self._is_setup = True

    # ------------------------------------------------------------------
    # V-cycle
    # ------------------------------------------------------------------

    def vcycle(self, r_gpu) -> "cp.ndarray":
        """Apply one V-cycle: z ≈ K_fine^{-1} r.  r_gpu is a CuPy vector."""
        if not self._is_setup:
            raise RuntimeError("HexGridGMG.setup() must be called before vcycle().")
        return self._vcycle(0, r_gpu)

    def _vcycle(self, lv: int, r) -> "cp.ndarray":
        import cupy as cp

        K        = self._K_gpu[lv]
        d_inv    = self._diag_inv_gpu[lv]
        omega    = self._omega
        ns       = self._n_smooth

        # ── Coarsest level: cached LU triangular solve (CPU) ─────────────
        # _coarse_lu is pre-factorized once in setup() via scipy.factorized.
        # Each V-cycle uses only the cheap triangular solve, not re-factorization.
        if lv == self._n_levels - 1:
            r_cpu = cp.asnumpy(r)
            x_cpu = self._coarse_lu(r_cpu)
            return cp.asarray(x_cpu)

        # ── Pre-smooth: damped Jacobi ─────────────────────────────────────
        x = cp.zeros_like(r)
        for _ in range(ns):
            x = x + omega * d_inv * (r - K @ x)

        # ── Restrict residual to coarse level ─────────────────────────────
        r_c = self._R_gpu[lv] @ (r - K @ x)

        # ── Coarse-level solve (recursion) ────────────────────────────────
        e_c = self._vcycle(lv + 1, r_c)

        # ── Prolongate correction ─────────────────────────────────────────
        x = x + self._P_gpu[lv] @ e_c

        # ── Post-smooth: damped Jacobi ────────────────────────────────────
        for _ in range(ns):
            x = x + omega * d_inv * (r - K @ x)

        return x

    def as_linear_operator(self):
        """Return a CuPy LinearOperator wrapping this V-cycle preconditioner."""
        import cupy as cp
        import cupyx.scipy.sparse.linalg as cpspla
        n = self._n_free[0]
        return cpspla.LinearOperator(
            (n, n),
            matvec=lambda v: self.vcycle(v),
            dtype=cp.float64,
        )


# ─────────────────────────────────────────────────────────────────────────────
# § 1.3  Mixed-Precision Iterative Refinement
# ─────────────────────────────────────────────────────────────────────────────

def _mixed_precision_solve(
    Kff_fp64,           # cupyx CSR float64
    F_free_fp64,        # cp.ndarray float64
    M_op,               # preconditioner LinearOperator (fp64)
    x0,                 # initial guess (fp64 cp.ndarray or None)
    tol: float,
    maxiter: int,
    refine_iters: int = 2,
) -> "cp.ndarray":
    """
    Mixed-precision iterative refinement.

    Algorithm
    ---------
        x = x0  (FP64 initial guess or zero)
        for outer in range(refine_iters):
            r = F - K x                              # FP64 residual
            if ||r||/||F|| < tol: break
            delta = CG_FP32(K_fp32, r_fp32, M_fp32) # FP32 inner solve
            x = x + delta                            # FP64 correction

    The inner CG runs in FP32, exploiting the RTX 4090's 82 TFLOPS FP32
    throughput vs 1.3 TFLOPS FP64 for the matvec-heavy inner loop.  The
    outer residual correction in FP64 drives the solution to full precision.

    Notes
    -----
    - FP32 single-step CG tol is floored at 1e-6 (FP32 machine epsilon ~1e-7).
    - If CG breaks down (info < 0) the current x is returned as-is.
    """
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cpspla

    x = cp.zeros(F_free_fp64.shape[0], dtype=cp.float64) if x0 is None else x0.copy()

    # Build FP32 version of K and a simple Jacobi preconditioner for FP32
    Kff_fp32 = Kff_fp64.astype(cp.float32)
    diag32   = Kff_fp32.diagonal()
    diag32   = cp.where(cp.abs(diag32) > 1e-8, diag32, cp.ones_like(diag32))
    M32_inv  = 1.0 / diag32
    M32_op   = cpspla.LinearOperator(
        Kff_fp32.shape,
        matvec=lambda v: M32_inv * v.astype(cp.float32),
        dtype=cp.float32,
    )

    F_norm = float(cp.linalg.norm(F_free_fp64))
    if F_norm < 1e-15:
        return x

    for _ in range(max(1, refine_iters)):
        r64 = F_free_fp64 - Kff_fp64 @ x
        rel_res = float(cp.linalg.norm(r64)) / F_norm
        if rel_res < tol:
            break

        r32    = r64.astype(cp.float32)
        delta, info = cpspla.cg(
            Kff_fp32, r32,
            tol=max(tol * 0.1, 1e-6),
            maxiter=maxiter,
            M=M32_op,
        )
        if info < 0:
            break   # CG breakdown: return best x so far
        x = x + delta.astype(cp.float64)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# § 1.3b  Typed CG — explicit FP32-safe conjugate gradient
# ─────────────────────────────────────────────────────────────────────────────

def _cupy_cg(A, b, x0=None, tol=1e-5, maxiter=1000, M_inv=None, history=None):
    """
    Typed Jacobi-preconditioned CG, explicit dtype management.

    Replaces cupyx.scipy.sparse.linalg.cg for FP32 solves.  CuPy's cg()
    crashes on FP32 because cp.dot(fp32, fp32) returns an fp64 scalar on
    some CuPy builds; that fp64 scalar then promotes alpha * p to fp64,
    which either crashes or silently defeats the bandwidth gain.

    Fix strategy: keep alpha/beta as 0-d CuPy arrays (via .astype(dt) to
    normalise any cp.dot fp64 promotion).  0-d CuPy × 1-d CuPy stays on
    GPU with no host sync.  Only the convergence check (one cp.dot→float
    per iteration) touches the host — identical to CuPy's own cg() loop.

    Parameters
    ----------
    A      : cupyx CSR matrix OR callable (v → A·v) — matrix or operator
    b      : cp.ndarray — RHS; dtype sets compute precision
    x0     : cp.ndarray or None — initial guess
    tol    : float — relative residual tolerance
    maxiter: int
    M_inv  : cp.ndarray or None — diagonal preconditioner (1/diag(A))
    history: list or None — if provided, per-iteration ||r||/||b|| is appended
             (first entry is the initial residual ratio; adds one host sync
             at init but no extra sync inside the loop since r_sq is already
             computed for the convergence test).

    Returns
    -------
    x         : cp.ndarray (same dtype as b)
    iters     : int — CG iterations performed
    converged : bool
    """
    import cupy as cp
    dt = b.dtype
    _Aop = A if callable(A) else (lambda v, _A=A: _A @ v)

    x = cp.zeros_like(b) if x0 is None else x0.astype(dt, copy=True)
    r = b - _Aop(x)
    z = M_inv * r if M_inv is not None else r.copy()
    p = z.copy()

    # 0-d GPU arrays for rz; .astype(dt) normalises any fp64 promotion from cp.dot
    rz      = cp.dot(r, z).astype(dt)
    b_sq    = float(cp.dot(b, b))                 # one-time host sync at init
    tol_sq  = (tol ** 2) * b_sq
    if history is not None:
        r0_sq = float(cp.dot(r, r))
        history.append((r0_sq / b_sq) ** 0.5 if b_sq > 0 else 0.0)

    converged = False
    iters = 0
    for _ in range(maxiter):
        Ap    = _Aop(p)
        pAp   = cp.dot(p, Ap).astype(dt)          # 0-d fp32 GPU scalar
        alpha = rz / pAp                           # 0-d fp32 / 0-d fp32 = 0-d fp32
        x     = x + alpha * p                      # fp32 throughout
        r     = r - alpha * Ap
        iters += 1
        r_sq = float(cp.dot(r, r))                 # ONE host sync per iter (convergence check)
        if history is not None:
            history.append((r_sq / b_sq) ** 0.5 if b_sq > 0 else 0.0)
        if r_sq <= tol_sq:
            converged = True
            break
        z      = M_inv * r if M_inv is not None else r.copy()
        rz_new = cp.dot(r, z).astype(dt)
        beta   = rz_new / rz                       # 0-d fp32
        p      = z + beta * p
        rz     = rz_new

    return x, iters, converged


def _cupy_pcg(A, b, M_op, x0=None, tol=1e-5, maxiter=1000):
    """
    Preconditioned CG with upfront convergence check.

    Unlike cupyx.scipy.sparse.linalg.cg, which applies the preconditioner to
    r0 before the first convergence check, this implementation checks ||r0||/||b||
    < tol BEFORE the first M_op application.  This is critical for warm-start
    iterations where x0 ≈ exact solution: the initial residual is already below
    tolerance and we can return in ~2 ms (residual computation + norm) instead of
    paying for a full GMG V-cycle (~100 ms) unnecessarily.

    Parameters
    ----------
    A      : cupyx CSR matrix OR callable (v → A·v) — matrix or operator
    b      : cp.ndarray (float64) — RHS
    M_op   : callable (cp.ndarray → cp.ndarray) — preconditioner apply
    x0     : cp.ndarray or None — warm-start initial guess
    tol    : float — relative residual tolerance
    maxiter: int

    Returns
    -------
    x         : cp.ndarray
    iters     : int — preconditioner applications performed
    converged : bool
    """
    import cupy as cp
    _Aop = A if callable(A) else (lambda v, _A=A: _A @ v)

    x = x0.copy() if x0 is not None else cp.zeros_like(b)
    r = b - _Aop(x)

    b_norm_sq  = float(cp.dot(b, b))          # one host sync
    tol_sq     = tol ** 2 * b_norm_sq

    # ── Upfront convergence check (avoids one M_op application on warm-start) ──
    if float(cp.dot(r, r)) <= tol_sq:
        return x, 0, True

    # ── Standard PCG ──────────────────────────────────────────────────────────
    z    = M_op(r)                              # first preconditioner application
    p    = z.copy()
    rz   = float(cp.dot(r, z))                 # host sync

    converged = False
    iters     = 1   # one M_op already applied above
    for _ in range(maxiter - 1):
        Ap    = _Aop(p)
        pAp   = float(cp.dot(p, Ap))
        alpha = rz / pAp
        x     = x + alpha * p
        r     = r - alpha * Ap
        r_sq  = float(cp.dot(r, r))            # host sync for convergence check
        if r_sq <= tol_sq:
            converged = True
            break
        z      = M_op(r)
        iters += 1
        rz_new = float(cp.dot(r, z))
        beta   = rz_new / rz
        p      = z + beta * p
        rz     = rz_new

    return x, iters, converged


# ─────────────────────────────────────────────────────────────────────────────
# § 1.2C  Re-discretization geometric multigrid
# ─────────────────────────────────────────────────────────────────────────────

class _CoarseLevel:
    """
    Sparse assembly infrastructure for one re-discretization coarse level.
    Holds precomputed GPU arrays for O(n_kept) per-iteration Kff assembly.
    """
    __slots__ = (
        "nelx", "nely", "nelz", "n_elem", "n_free",
        "Kff_indptr", "Kff_indices", "elem_sorted", "KE0_sorted",
        "diag_pos", "diag_row", "E0", "Emin",
    )


def _build_coarse_level(
    nelx_c: int, nely_c: int, nelz_c: int,
    free_c: "np.ndarray",
    E0: float, Emin: float,
    KE_UNIT: "np.ndarray",
) -> _CoarseLevel:
    """
    Build GPU-ready sparse assembly arrays for one coarse mesh level.

    Replicates GPUFEMSolver._precompute_sorted_csr for an arbitrary
    (nelx_c, nely_c, nelz_c, free_c) without constructing a full solver.
    On return the _CoarseLevel holds GPU arrays for O(n_kept) assembly:
        Kff_data = E_e[elem_sorted] * KE0_sorted
        Kff = csr_matrix((Kff_data, Kff_indices, Kff_indptr))
    """
    import cupy as cp
    from .pub_simp_solver import _edof_table_3d, _build_sparse_indices

    ndof_c   = 3 * (nelx_c + 1) * (nely_c + 1) * (nelz_c + 1)
    n_free_c = len(free_c)

    edof_c           = _edof_table_3d(nelx_c, nely_c, nelz_c)   # (n_elem_c, 24)
    row_idx, col_idx = _build_sparse_indices(edof_c)             # (n_elem_c*576,)

    # Filter to free–free DOF pairs
    free_mask = np.zeros(ndof_c, dtype=bool)
    free_mask[free_c] = True
    keep     = free_mask[row_idx] & free_mask[col_idx]
    keep_idx = np.nonzero(keep)[0]

    # Remap global DOF indices → [0, n_free_c)
    free_local = np.full(ndof_c, -1, dtype=np.int32)
    free_local[free_c] = np.arange(n_free_c, dtype=np.int32)
    rows_local = free_local[row_idx[keep_idx]]
    cols_local = free_local[col_idx[keep_idx]]

    # Element and KE entry indices (each element contributes 576 = 24² entries)
    elem_of_kept = (keep_idx // 576).astype(np.int32)
    KE0_of_kept  = KE_UNIT.ravel()[keep_idx % 576]

    # Sort by (row, col) → zero-sort-cost per-iteration CSR construction
    sort_idx = np.lexsort((cols_local, rows_local))
    rows_s   = rows_local[sort_idx]
    cols_s   = cols_local[sort_idx]
    elem_s   = elem_of_kept[sort_idx]
    KE0_s    = KE0_of_kept[sort_idx]

    indptr = np.searchsorted(
        rows_s.astype(np.int64), np.arange(n_free_c + 1, dtype=np.int64), side="left",
    ).astype(np.int32)

    # Fast diagonal scatter (cp.bincount trick, mirrors SolverV2)
    row_expanded = np.repeat(np.arange(n_free_c, dtype=np.int32), np.diff(indptr))
    is_diag  = (cols_s == row_expanded)
    diag_pos = np.nonzero(is_diag)[0].astype(np.int32)
    diag_row = row_expanded[is_diag].astype(np.int32)

    lv             = _CoarseLevel()
    lv.nelx        = nelx_c
    lv.nely        = nely_c
    lv.nelz        = nelz_c
    lv.n_elem      = nelx_c * nely_c * nelz_c
    lv.n_free      = n_free_c
    lv.E0          = E0
    lv.Emin        = Emin
    lv.Kff_indptr  = cp.asarray(indptr)
    lv.Kff_indices = cp.asarray(cols_s.astype(np.int32))
    lv.elem_sorted = cp.asarray(elem_s)
    lv.KE0_sorted  = cp.asarray(KE0_s)
    lv.diag_pos    = cp.asarray(diag_pos)
    lv.diag_row    = cp.asarray(diag_row)
    return lv


def _coarse_free_dofs_injection(
    nelx_f: int, nely_f: int, nelz_f: int,
    nelx_c: int, nely_c: int, nelz_c: int,
    free_f: "np.ndarray",
) -> "np.ndarray":
    """
    Identify free DOFs at the coarse level using injection (not prolongation).

    Coarse node (ix_c, iy_c, iz_c) maps injectively to fine node
    (2·ix_c, 2·iy_c, 2·iz_c).  A coarse DOF is free iff its injected
    fine DOF is free.  This correctly handles face Dirichlet BCs: a coarse
    node on the fixed face (e.g. ix_c=0) maps to a fixed fine node (ix_f=0)
    and is correctly identified as fixed.

    The prolongation-based _coarse_free_dofs wrongly marks boundary coarse
    nodes as free because they have fractional weights (0.5) to free interior
    fine nodes — but those are not "owned" by the boundary coarse node.
    Re-discretization GMG assembles K_c independently, so if these DOFs are
    incorrectly included in free_c, the coarse solve produces spurious
    boundary corrections that cause CG divergence.
    """
    ndof_c = 3 * (nelx_c + 1) * (nely_c + 1) * (nelz_c + 1)
    ndof_f = 3 * (nelx_f + 1) * (nely_f + 1) * (nelz_f + 1)

    # Injection: coarse node (ix_c, iy_c, iz_c) → fine node (2ix_c, 2iy_c, 2iz_c)
    ix_c, iy_c, iz_c = np.meshgrid(
        np.arange(nelx_c + 1), np.arange(nely_c + 1), np.arange(nelz_c + 1),
        indexing="ij",
    )
    fine_node = (
        2 * ix_c * (nely_f + 1) * (nelz_f + 1)
        + 2 * iy_c * (nelz_f + 1)
        + 2 * iz_c
    ).ravel()   # (n_coarse_nodes,) — injected fine node index per coarse node

    free_mask_f = np.zeros(ndof_f, dtype=bool)
    free_mask_f[free_f] = True

    # Check all 3 DOF components (x, y, z) per node
    is_free_c = np.zeros(ndof_c, dtype=bool)
    n_cn = len(fine_node)
    for d in range(3):
        fine_dofs   = 3 * fine_node + d                      # (n_cn,)
        coarse_dofs = np.arange(n_cn, dtype=np.int32) * 3 + d  # (n_cn,)
        valid = fine_dofs < ndof_f
        is_free_c[coarse_dofs[valid]] = free_mask_f[fine_dofs[valid]]

    return np.where(is_free_c)[0].astype(np.int32)


def _coarsen_density(
    rho_f, nelx_f: int, nely_f: int, nelz_f: int,
    nelx_c: int, nely_c: int, nelz_c: int,
):
    """
    Average fine element density to coarse (2:1 in each dimension).

    Works on both NumPy and CuPy arrays.  Truncates fine dims to the
    nearest even multiple so odd-dimension grids are handled gracefully.
    """
    nx2, ny2, nz2 = nelx_c * 2, nely_c * 2, nelz_c * 2
    rho_3d = rho_f.reshape(nelx_f, nely_f, nelz_f)[:nx2, :ny2, :nz2]
    return rho_3d.reshape(nelx_c, 2, nely_c, 2, nelz_c, 2).mean(axis=(1, 3, 5)).reshape(-1)


def _fixed_pcg_coarse(K, b, d_inv, n_iters: int = 40):
    """
    Fixed-count Jacobi-preconditioned CG for the coarsest-level solve.

    Zero initial guess; exactly n_iters iterations (no tolerance check).
    Deterministic → the V-cycle M^{-1} is a fixed polynomial of K, hence
    a symmetric linear operator, which is required for the outer PCG to
    converge in theory.

    Stays entirely on GPU (no cp.cuda.stream.synchronize / float() calls),
    so WDDM overhead is paid only once at the outer CG's sync points.
    """
    import cupy as cp
    x  = cp.zeros_like(b)
    r  = b.copy()
    z  = d_inv * r
    p  = z.copy()
    rz = r.dot(z)                     # 0-dim CuPy array (stays on GPU)
    for _ in range(n_iters):
        Ap    = K @ p
        pAp   = p.dot(Ap)
        alpha = rz / (pAp + cp.array(1e-300, dtype=b.dtype))
        x     = x + alpha * p
        r     = r - alpha * Ap
        z     = d_inv * r
        rz_n  = r.dot(z)
        beta  = rz_n / (rz + cp.array(1e-300, dtype=b.dtype))
        p     = z + beta * p
        rz    = rz_n
    return x


class RedisCGMG:
    """
    Galerkin geometric multigrid for structured 3D hex grids.

    Uses Galerkin coarsening: K_c = P^T K_f P at every level.  This
    guarantees the coarse operator is SPD and consistent with the fine
    operator for ANY density distribution, including high-contrast SIMP
    densities where re-discretization (assembling K_c from averaged rho)
    fails (V-cycle ratio > 1, PCG diverges).

    Fill-in at coarse levels (~125 nnz/row at level 1 vs 26 for re-discr.)
    is tolerable: Jacobi smoothing only needs the diagonal and one SpMV,
    and the matrices are much smaller than the fine level.

    Construction
    ------------
    Builds prolongation matrices (trilinear interpolation, free-DOF subspace)
    and computes free DOFs at each coarse level via injection.
    No coarse K precomputation — GPU arrays are built on first setup().

    Per-SIMP-iteration: setup(K_f, rho_f, penal, fine_diag)
    --------------------------------------------------------
    Computes K_c = P^T K_f P at each level via GPU SpGEMM (cuSPARSE).
    Coarsest level: dense Cholesky if n_free <= DENSE_CHOL_MAX (5 000),
    else fixed-count Jacobi-PCG (deterministic, preserves V-cycle symmetry).

    Per-CG-iteration: vcycle(r) → z
    --------------------------------
    Standard V-cycle (pre-smooth → restrict → coarse solve → prolongate →
    post-smooth) on GPU.  Identical to HexGridGMG._vcycle.

    Parameters
    ----------
    nelx, nely, nelz : int  Fine-level element counts.
    free  : (n_free,) int32  Fine-level free DOF indices.
    E0, Emin : float  Material moduli.
    KE_UNIT  : (24, 24) float64  Unit element stiffness matrix.
    n_levels : int  Number of levels including fine.  Default 3.
    n_smooth : int  Jacobi pre/post-smoothing sweeps per level.  Default 2.
    omega    : float  Jacobi damping factor.  Default 2/3.
    """

    def __init__(
        self,
        nelx: int, nely: int, nelz: int,
        free: "np.ndarray",
        E0: float, Emin: float,
        KE_UNIT: "np.ndarray",
        n_levels: int = 3,
        n_smooth: int = 2,
        omega: float = 2.0 / 3.0,
    ) -> None:
        self._n_levels = n_levels
        self._n_smooth = n_smooth
        self._omega    = omega

        # Element counts per level (halve each dim, floor to 1)
        dims: list[tuple[int, int, int]] = [(nelx, nely, nelz)]
        for _ in range(1, n_levels):
            nx, ny, nz = dims[-1]
            dims.append((max(1, nx // 2), max(1, ny // 2), max(1, nz // 2)))
        self._dims = dims

        # Free DOFs per level and prolongation matrices (scipy CSR, free-DOF subspace)
        self._free_per_level: list["np.ndarray"] = [free]
        self._n_free: list[int] = [len(free)]
        self._P_scipy: list["sp.csr_matrix"] = []

        for lv in range(n_levels - 1):
            nx_f, ny_f, nz_f = dims[lv]
            nx_c, ny_c, nz_c = dims[lv + 1]
            free_f = self._free_per_level[lv]

            # Injection: coarse DOF free iff its injected fine DOF is free.
            # Galerkin K_c = P^T K_f P inherits BCs automatically, but we
            # still need free_c to build P_free (restriction to free subspace)
            # and for the next level's prolongation.
            free_c = _coarse_free_dofs_injection(nx_f, ny_f, nz_f, nx_c, ny_c, nz_c, free_f)
            self._free_per_level.append(free_c)
            self._n_free.append(len(free_c))

            # Prolongation: trilinear interpolation, restricted to free subspace
            P_sc  = _build_scalar_prolongation(nx_f, ny_f, nz_f, nx_c, ny_c, nz_c)
            P_vec = sp.kron(P_sc, sp.eye(3, format="csr", dtype=np.float64), format="csr")
            del P_sc
            P_free = P_vec[free_f, :][:, free_c].tocsr()
            self._P_scipy.append(P_free)
            del P_vec, P_free
            gc.collect()

        # GPU P/R matrices (lazily uploaded on first setup())
        self._P_gpu: list["Optional[object]"] = [None] * (n_levels - 1)
        self._R_gpu: list["Optional[object]"] = [None] * (n_levels - 1)
        self._gpu_uploaded = False

        # Per-SIMP-iteration GPU state
        self._K_gpu:           list["Optional[object]"] = [None] * n_levels
        self._diag_inv_gpu:    list["Optional[object]"] = [None] * n_levels
        self._coarse_chol_L    = None   # dense lower-triangular factor (when coarsest is small)
        self._coarse_chol_mode = "none" # "dense" | "iterative"
        self._is_setup         = False

        # Rho-change cache — skip recomputation when density unchanged
        self._rho_fine_cache: "Optional[np.ndarray]" = None

    # ------------------------------------------------------------------
    # Per-SIMP-iteration: assemble coarse operators
    # ------------------------------------------------------------------

    # Threshold: use dense Cholesky when coarsest level has <= this many DOFs.
    # Above this, fall back to fixed-count iterative PCG (avoids huge dense alloc).
    _DENSE_CHOL_MAX = 5000

    def setup(
        self,
        K_fine_cupy,
        rho_fine: "np.ndarray",
        penal: float,
        fine_diag=None,
    ) -> None:
        """
        Build Galerkin coarse operators K_c = P^T K_f P for each level.

        K_fine_cupy : cupyx CSR (n_free_fine, n_free_fine) — fine K.
        rho_fine    : np.ndarray (n_elem_fine,) — element densities.
                      Used ONLY for the fast-path cache check; coarse K is
                      computed via Galerkin, not from density.
        penal       : unused (kept for API compatibility).
        fine_diag   : optional precomputed cp.ndarray diagonal of K_fine_cupy.

        Caching
        -------
        If rho_fine equals the previous call's value, all coarse operators
        are still valid (K_fine is deterministic given rho_fine) → return early.
        """
        import cupy as cp
        import cupyx.scipy.sparse as cpsp

        # --- GPU P/R upload (first call only) ---
        if not self._gpu_uploaded:
            for lv in range(self._n_levels - 1):
                self._P_gpu[lv] = cpsp.csr_matrix(self._P_scipy[lv])
                self._R_gpu[lv] = self._P_gpu[lv].T.tocsr()
            self._gpu_uploaded = True

        # --- Level 0: accept caller's fine K ---
        self._K_gpu[0] = K_fine_cupy
        diag0 = fine_diag if fine_diag is not None else K_fine_cupy.diagonal()
        diag0 = cp.where(cp.abs(diag0) > 1e-12, diag0, cp.ones_like(diag0))
        self._diag_inv_gpu[0] = 1.0 / diag0

        # --- Fast path: rho unchanged → coarse ops still valid ---
        if (self._is_setup
                and self._rho_fine_cache is not None
                and np.array_equal(rho_fine, self._rho_fine_cache)):
            return

        # --- Galerkin coarsening: K_c = P^T K_f P at each level ---
        # cuSPARSE SpGEMM handles any density distribution correctly:
        # the coarse operator is guaranteed SPD and consistent with K_fine
        # regardless of stiffness contrast (high-contrast SIMP safe).
        for lv in range(1, self._n_levels):
            K_f = self._K_gpu[lv - 1]
            P   = self._P_gpu[lv - 1]   # (n_free_f × n_free_c)
            R   = self._R_gpu[lv - 1]   # (n_free_c × n_free_f)  = P^T
            K_c = R @ (K_f @ P)          # Galerkin triple product
            self._K_gpu[lv] = K_c
            diag_c = K_c.diagonal()
            diag_c = cp.where(cp.abs(diag_c) > 1e-12, diag_c, cp.ones_like(diag_c))
            self._diag_inv_gpu[lv] = 1.0 / diag_c

        # --- Coarsest-level direct factor (dense Cholesky or iterative fallback) ---
        K_coarsest = self._K_gpu[self._n_levels - 1]
        n_coarsest = K_coarsest.shape[0]
        if n_coarsest <= self._DENSE_CHOL_MAX:
            K_dense_gpu = K_coarsest.toarray()
            self._coarse_chol_L    = cp.linalg.cholesky(K_dense_gpu)
            self._coarse_chol_mode = "dense"
        else:
            # Too large for 1 GB dense alloc — use fixed-count PCG at coarsest level.
            # The V-cycle remains a well-defined symmetric operator (zero-start CG
            # with fixed iteration count is a polynomial of K, hence symmetric).
            self._coarse_chol_L    = None
            self._coarse_chol_mode = "iterative"

        self._rho_fine_cache = rho_fine.copy()
        self._is_setup = True

    # ------------------------------------------------------------------
    # V-cycle (identical structure to HexGridGMG._vcycle)
    # ------------------------------------------------------------------

    def vcycle(self, r_gpu) -> "cp.ndarray":
        """Apply one V-cycle: z ≈ K_fine^{-1} r."""
        if not self._is_setup:
            raise RuntimeError("RedisCGMG.setup() must be called before vcycle().")
        return self._vcycle(0, r_gpu)

    def _vcycle(self, lv: int, r) -> "cp.ndarray":
        import cupy as cp

        K     = self._K_gpu[lv]
        d_inv = self._diag_inv_gpu[lv]

        # Coarsest level: direct solve (dense Cholesky) or fixed-count iterative PCG.
        if lv == self._n_levels - 1:
            if self._coarse_chol_mode == "dense":
                import cupyx.scipy.linalg as _cpla
                y = _cpla.solve_triangular(self._coarse_chol_L, r,  lower=True)
                return _cpla.solve_triangular(self._coarse_chol_L, y, lower=True, trans='T')
            else:
                # Fixed-count Jacobi-PCG (zero initial guess, n_iters fixed for determinism).
                # Deterministic → V-cycle is a polynomial of K → symmetric linear operator.
                # Avoids 1 GB dense allocation for n_coarsest > 5 000.
                return _fixed_pcg_coarse(K, r, d_inv, n_iters=40)

        # Pre-smooth
        x = cp.zeros_like(r)
        for _ in range(self._n_smooth):
            x = x + self._omega * d_inv * (r - K @ x)

        # Restrict residual to coarse level
        r_c = self._R_gpu[lv] @ (r - K @ x)

        # Coarse solve (recursive)
        e_c = self._vcycle(lv + 1, r_c)

        # Prolongate correction
        x = x + self._P_gpu[lv] @ e_c

        # Post-smooth
        for _ in range(self._n_smooth):
            x = x + self._omega * d_inv * (r - K @ x)

        return x

    def as_linear_operator(self):
        """Return a CuPy LinearOperator wrapping this V-cycle preconditioner."""
        import cupy as cp
        import cupyx.scipy.sparse.linalg as cpspla
        n = self._n_free[0]
        return cpspla.LinearOperator(
            (n, n), matvec=lambda v: self.vcycle(v), dtype=cp.float64,
        )


# ─────────────────────────────────────────────────────────────────────────────
# § 1.1  Warm-start CG for the Torch backend
# ─────────────────────────────────────────────────────────────────────────────

def _torch_cg_warm(
    A,
    b: "torch.Tensor",
    x0: "Optional[torch.Tensor]" = None,
    tol: float = 1e-5,
    maxiter: int = 1000,
) -> "torch.Tensor":
    """
    Jacobi-preconditioned CG with optional warm-start initial guess.
    Identical to fem_gpu._torch_cg except for the x0 parameter.
    """
    import torch

    def _mv(M, v):
        return (M @ v.unsqueeze(1)).squeeze(1)

    # Vectorised Jacobi diagonal extraction from CSR
    crow = A.crow_indices()
    ccol = A.col_indices()
    vals = A.values()
    n    = A.shape[0]
    counts       = crow[1:] - crow[:-1]
    row_of_entry = torch.repeat_interleave(
        torch.arange(n, device=b.device), counts
    )
    diag_mask = (ccol == row_of_entry)
    diag = torch.zeros(n, dtype=b.dtype, device=b.device)
    diag.index_add_(0, ccol[diag_mask], vals[diag_mask])
    diag  = torch.where(diag.abs() > 1e-12, diag, torch.ones_like(diag))
    M_inv = 1.0 / diag

    x  = x0.clone() if x0 is not None else torch.zeros_like(b)
    r  = b - _mv(A, x)
    z  = M_inv * r
    p  = z.clone()
    rz = torch.dot(r, z)

    b_norm = torch.norm(b)
    for _ in range(maxiter):
        Ap    = _mv(A, p)
        alpha = rz / (torch.dot(p, Ap) + 1e-30)
        x     = x + alpha * p
        r     = r - alpha * Ap
        if torch.norm(r) < tol * b_norm:
            break
        z      = M_inv * r
        rz_new = torch.dot(r, z)
        beta   = rz_new / (rz + 1e-30)
        p      = z + beta * p
        rz     = rz_new

    return x


# ─────────────────────────────────────────────────────────────────────────────
# § Phase 2 — Matrix-free K_ff · u operator
# ─────────────────────────────────────────────────────────────────────────────

class MatrixFreeKff:
    """
    Matrix-free K_ff · u operator for structured 3D hex meshes.

    Replaces the stored-CSR SpMV with on-the-fly element recomputation.
    Storage: O(n_elem * ke_size) integers for edof + O(ke_size²) for KE_UNIT.
    No sparse matrix is assembled or held in memory.

    For a Q1 hex element: ke_size=24 DOFs.  KE_UNIT (24×24) is shared by all
    elements on a regular mesh — only element moduli E_e vary each SIMP iteration.

    Matvec K_ff · u_free = (n_free,) → (n_free,) via five steps:

      1. Expand:   u_full[free] = u_free;  u_full[fixed] = 0
      2. Gather:   u_elem = u_full[edof]                       (n_elem, ke_size)
      3. GEMM:     f_tmp  = KE_UNIT @ u_elem.T                 (ke_size, n_elem)
                   f_elem = (E_e * f_tmp).T                    (n_elem, ke_size)
                   (One cuBLAS DGEMM; no loop over elements.)
      4. Scatter:  y_full = bincount(edof.ravel(), f_elem.ravel(), n=ndof)
      5. Restrict: y = y_full[free]                            (n_free,)

    The diagonal of K_ff (for Jacobi preconditioner) is extracted without
    assembling the matrix via the same scatter pattern:
      K_ff_diag[d] = Σ_{(e,j): edof[e,j]=d} E_e[e] * KE_UNIT[j,j]

    VRAM cost (216k elements, FP64):
      edof     : n_elem * ke_size * 4B ≈  20 MB
      KE_UNIT  : ke_size² * 8B         ≈   5 KB
      KE_diag  : ke_size * 8B          ≈   0.2 KB
      (vs canonical sparse Kff:        ≈  11.4 GB)
    """

    def __init__(
        self,
        edof_gpu,       # (n_elem, ke_size) cp.ndarray int32 — global DOF indices
        KE_unit_gpu,    # (ke_size, ke_size) cp.ndarray float64 — unit stiffness
        free_gpu,       # (n_free,) cp.ndarray int32 — free DOF global indices
        n_free: int,
        ndof: int,
    ) -> None:
        self._edof_gpu  = edof_gpu        # (n_elem, ke_size)
        self._KE_unit   = KE_unit_gpu     # (ke_size, ke_size)
        self._free_gpu  = free_gpu        # (n_free,)
        self._n_free    = n_free
        self._ndof      = ndof
        self._n_elem, self._ke_size = int(edof_gpu.shape[0]), int(edof_gpu.shape[1])

        # Precompute for scatter step
        # Pre-cast to int64: cp.bincount requires int64 indices; casting once here
        # avoids an O(n_elem*ke_size) cast inside every matvec call.
        self._edof_flat = edof_gpu.ravel().astype('int64')  # (n_elem * ke_size,) int64
        self._KE_diag   = KE_unit_gpu.diagonal().copy()     # (ke_size,) float64

        # FP32 cache — built lazily on first FP32 matvec call
        self._KE_unit_f32 = None
        self._KE_diag_f32 = None

    # ------------------------------------------------------------------
    # Core matvec
    # ------------------------------------------------------------------

    def matvec(self, u_free, E_e):
        """
        Compute K_ff · u_free without assembling K_ff.

        Parameters
        ----------
        u_free : cp.ndarray (n_free,)  — displacement (free DOFs)
        E_e    : cp.ndarray (n_elem,)  — element Young's moduli, same dtype as u_free

        Returns
        -------
        y : cp.ndarray (n_free,)  — K_ff · u_free
        """
        import cupy as cp
        dt = u_free.dtype

        # 1. Zero-padded expansion: fixed DOFs stay 0
        u_full = cp.zeros(self._ndof, dtype=dt)
        u_full[self._free_gpu] = u_free

        # 2. Gather: u_elem[e, j] = u_full[edof[e, j]]
        u_elem = u_full[self._edof_gpu]          # (n_elem, ke_size)

        # 3. Batched GEMV as one GEMM:
        #    KE_UNIT @ u_elem.T → (ke_size, n_elem), scale by E_e, transpose
        KE = self._KE_unit if dt == cp.float64 else self._ke_unit_f32(cp)
        f_elem = (KE @ u_elem.T * E_e[None, :]).T       # (n_elem, ke_size)

        # 4. Scatter-add: y_full[k] = Σ f_elem[e,j] where edof[e,j]=k
        # NOTE: cp.bincount always returns float64 regardless of weights dtype.
        # Cast back to dt so the CG loop (and any downstream FP32 arithmetic)
        # stays in the requested precision.  astype(..., copy=False) is a no-op
        # when dt is already float64, so there is no extra copy on the FP64 path.
        y_full = cp.bincount(
            self._edof_flat,          # pre-cast int64 (no runtime cast overhead)
            weights=f_elem.ravel(),
            minlength=self._ndof,
        ).astype(dt, copy=False)

        # 5. Restrict to free DOFs
        return y_full[self._free_gpu]

    # ------------------------------------------------------------------
    # Jacobi diagonal extraction (no K assembly)
    # ------------------------------------------------------------------

    def extract_diagonal(self, E_e):
        """
        Extract diag(K_ff) without assembling K_ff.

        For each free DOF d:
          K_ff_diag[local(d)] = Σ_{(e,j): edof[e,j]=d} E_e[e] * KE_UNIT[j,j]

        Parameters
        ----------
        E_e : cp.ndarray (n_elem,) float64

        Returns
        -------
        diag : cp.ndarray (n_free,) float64
        """
        import cupy as cp
        # diag_contrib[e, j] = E_e[e] * KE_diag[j]
        diag_contrib = E_e[:, None] * self._KE_diag[None, :]   # (n_elem, ke_size)
        y_full = cp.bincount(
            self._edof_flat.astype(cp.int64),
            weights=diag_contrib.ravel(),
            minlength=self._ndof,
        )
        return y_full[self._free_gpu]                           # (n_free,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ke_unit_f32(self, cp):
        """Lazy FP32 cast of KE_UNIT (built once, reused)."""
        if self._KE_unit_f32 is None:
            self._KE_unit_f32 = self._KE_unit.astype(cp.float32)
        return self._KE_unit_f32


# ─────────────────────────────────────────────────────────────────────────────
# § Phase 2 — 3×3 node-block Jacobi preconditioner (matrix-free compatible)
# ─────────────────────────────────────────────────────────────────────────────

class NodeBlockJacobiMF:
    """
    Matrix-free 3×3 node-block Jacobi preconditioner.

    Each global mesh node gets a 3×3 diagonal block of K_ff assembled via
    element scatter-add — no full sparse matrix is ever formed.  Applying
    the block solve captures axial-shear coupling (u_x, u_y, u_z at each
    node) that scalar Jacobi ignores, giving a substantially better
    conditioned system for bending-dominated BVPs (MBB, bridge, bracket).

    Mathematical identity
    ---------------------
    For element e and its local node i (0…7):
      K_node[global_node(e,i)] += E_e · KE_UNIT[3i:3i+3, 3i:3i+3]

    The 8×(3×3) node-diagonal blocks of KE_UNIT are precomputed once;
    per-SIMP-iteration assembly is nine cp.bincount calls.
    After assembly each all-free-node block is inverted exactly (3×3 has
    a closed-form inverse; cp.linalg.inv on the batch is O(n_nodes)).

    Parameters
    ----------
    edof_gpu    : (n_elem, 24) cp.ndarray int32  — global DOF indices
    KE_unit_gpu : (24, 24)    cp.ndarray float64 — unit element stiffness
    free_gpu    : (n_free,)   cp.ndarray int32   — free DOF global indices
    n_free, ndof : int

    Assumption
    ----------
    Global DOF ordering is  DOF = 3·node_index + component  (standard FEM
    convention).  This means edof[e, 3·i] // 3 == global_node_i.
    """

    def __init__(
        self,
        edof_gpu,
        KE_unit_gpu,
        free_gpu,
        n_free: int,
        ndof: int,
    ) -> None:
        import cupy as cp

        self._n_free  = n_free
        self._ndof    = ndof
        n_nodes       = ndof // 3
        self._n_nodes = n_nodes
        n_elem        = int(edof_gpu.shape[0])

        # ── 8×(3×3) node-diagonal blocks of KE_UNIT (precomputed once) ────────
        KE_cpu = cp.asnumpy(KE_unit_gpu)
        ke_blocks_cpu = np.stack(
            [KE_cpu[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] for i in range(8)]
        )  # (8, 3, 3)
        self._ke_blocks_gpu = cp.asarray(ke_blocks_cpu)  # (8, 3, 3)

        # ── Flat (n_elem×8) pair tables ───────────────────────────────────────
        edof_cpu = cp.asnumpy(edof_gpu)
        # global node index for element e, local node i = edof[e, 3i] // 3
        node_of_pair_cpu = edof_cpu[:, [3 * i for i in range(8)]] // 3  # (n_elem, 8)
        local_node_flat_cpu  = np.tile(np.arange(8, dtype=np.int32), n_elem)  # (n_elem*8,)
        global_node_flat_cpu = node_of_pair_cpu.ravel().astype(np.int32)      # (n_elem*8,)
        self._local_node_flat_gpu  = cp.asarray(local_node_flat_cpu)
        self._global_node_flat_gpu = cp.asarray(global_node_flat_cpu)

        # ── Free-DOF layout per node ───────────────────────────────────────────
        free_cpu  = cp.asnumpy(free_gpu)
        free_mask = np.zeros(ndof, dtype=bool)
        free_mask[free_cpu] = True
        node_free = free_mask.reshape(n_nodes, 3)   # (n_nodes, 3), DOF = 3n + d

        all_free_mask       = node_free.all(axis=1)  # all 3 components free
        all_free_nodes_cpu  = np.where(all_free_mask)[0].astype(np.int32)
        self._n_all_free    = len(all_free_nodes_cpu)
        self._all_free_nodes_gpu = cp.asarray(all_free_nodes_cpu)

        # Reverse map: global DOF → local free index (or -1 if fixed)
        full_to_free = np.full(ndof, -1, dtype=np.int32)
        full_to_free[free_cpu] = np.arange(n_free, dtype=np.int32)

        # (n_all_free, 3) local free-DOF indices for gather/scatter
        global_dofs = (
            3 * all_free_nodes_cpu[:, None] + np.arange(3, dtype=np.int32)[None, :]
        )  # (n_all_free, 3)
        all_free_local_cpu = full_to_free[global_dofs]  # (n_all_free, 3)
        assert (all_free_local_cpu >= 0).all(), (
            "NodeBlockJacobiMF: node flagged all-free has a fixed DOF — "
            "check that DOF ordering is 3·node + component."
        )
        self._all_free_local_gpu = cp.asarray(all_free_local_cpu)  # (n_all_free, 3)

        # ── Partial-freedom nodes (e.g. pin_x, roller_x BCs) ─────────────────
        partial_mask = node_free.any(axis=1) & ~all_free_mask
        partial_nodes_cpu = np.where(partial_mask)[0].astype(np.int32)
        partial_local_list: list[int] = []
        for n in partial_nodes_cpu:
            for d in range(3):
                if node_free[n, d]:
                    partial_local_list.append(int(full_to_free[3 * n + d]))
        self._partial_local_gpu = (
            cp.asarray(np.array(partial_local_list, dtype=np.int32))
            if partial_local_list
            else cp.empty(0, dtype=cp.int32)
        )

        # Per-SIMP-iteration state (set by update())
        self._block_inv_gpu       = None   # (n_all_free, 3, 3)
        self._partial_diag_inv_gpu = None  # (n_partial,)

    # ------------------------------------------------------------------
    # Per-SIMP-iteration: assemble and invert node blocks
    # ------------------------------------------------------------------

    def update(self, E_e, scalar_diag=None) -> None:
        """
        Assemble K_node[n] = Σ E_e·KE_UNIT[3i:3i+3,3i:3i+3] and invert.

        Parameters
        ----------
        E_e         : cp.ndarray (n_elem,) float64 — element moduli
        scalar_diag : cp.ndarray (n_free,) float64 or None
            Jacobi diagonal for partial-freedom nodes (fallback).
            Pass mf.extract_diagonal(E_e) from the matfree operator.
            If None, partial-freedom DOFs receive identity preconditioning.
        """
        import cupy as cp

        n_nodes          = self._n_nodes
        global_node_long = self._global_node_flat_gpu.astype(cp.int64)
        E_per_pair       = cp.repeat(E_e, 8)   # (n_elem*8,)

        # Assemble all 9 scalar components of the 3×3 node block-K
        block_K = cp.zeros((n_nodes, 3, 3), dtype=cp.float64)
        for r in range(3):
            for c in range(3):
                ke_rc  = self._ke_blocks_gpu[self._local_node_flat_gpu, r, c]
                contrib = E_per_pair * ke_rc
                block_K[:, r, c] = cp.bincount(
                    global_node_long, weights=contrib, minlength=n_nodes
                )

        # Batch-invert all-free node blocks (closed-form for 3×3 on GPU)
        self._block_inv_gpu = cp.linalg.inv(
            block_K[self._all_free_nodes_gpu]   # (n_all_free, 3, 3)
        )

        # Scalar Jacobi fallback for partial-freedom DOFs (MBB, bridge pin BCs)
        if self._partial_local_gpu.size > 0:
            if scalar_diag is not None:
                diag_vals = scalar_diag[self._partial_local_gpu]
                diag_safe = cp.where(
                    cp.abs(diag_vals) > 1e-12, diag_vals, cp.ones_like(diag_vals)
                )
                self._partial_diag_inv_gpu = 1.0 / diag_safe
            else:
                self._partial_diag_inv_gpu = None

    # ------------------------------------------------------------------
    # Per-CG-iteration: apply block preconditioner
    # ------------------------------------------------------------------

    def apply(self, r):
        """
        Apply M^{-1} r.

        r : cp.ndarray (n_free,) — residual (any dtype; result matches dtype)
        Returns cp.ndarray (n_free,) — preconditioned vector.
        """
        import cupy as cp

        z = cp.zeros(self._n_free, dtype=cp.float64)

        # ── All-free nodes: exact 3×3 block solve ─────────────────────────
        if self._n_all_free > 0 and self._block_inv_gpu is not None:
            r_blocks = r[self._all_free_local_gpu].astype(cp.float64)   # (n_all_free, 3)
            z_blocks = cp.einsum('nij,nj->ni', self._block_inv_gpu, r_blocks)
            z[self._all_free_local_gpu] = z_blocks

        # ── Partial-freedom nodes: scalar Jacobi fallback ──────────────────
        if self._partial_local_gpu.size > 0 and self._partial_diag_inv_gpu is not None:
            pd = self._partial_local_gpu
            z[pd] = self._partial_diag_inv_gpu * r[pd].astype(cp.float64)

        return z.astype(r.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# § SolverV2 — Phase 1 drop-in replacement for GPUFEMSolver
# ─────────────────────────────────────────────────────────────────────────────

class SolverV2(GPUFEMSolver):
    """
    Phase 1 solver: warm-start CG + geometric multigrid + mixed-precision IR.

    Drop-in replacement for GPUFEMSolver.  Inherits assembly, CSR layout,
    and the CPU/Torch fallback paths unchanged.  Only _solve_cupy and
    _solve_torch are overridden.

    All Phase 1 improvements are behind flags so that compliance parity can
    be verified independently before stacking.

    Parameters
    ----------
    *args, **kwargs
        Passed through to GPUFEMSolver.__init__() unchanged.
    grid_dims : (nelx, nely, nelz) tuple
        Required when enable_gmg=True.
    enable_warm_start : bool
        1.1 — persist u_prev between SIMP iterations.  Default True.
    enable_gmg : bool
        1.2B — use HexGridGMG as preconditioner.  Default False.
        Enable only after parity check passes.
    gmg_levels : int
        Number of multigrid levels.  Default 3.
    gmg_smooth_iters : int
        Damped-Jacobi pre/post-smoothing sweeps per level.  Default 2.
    enable_mixed_precision : bool
        1.3 — FP32 inner CG + FP64 residual correction.  Default False.
        Enable only after GMG parity check passes.
    mixed_precision_refine_iters : int
        Number of outer MPIR refinement steps.  Default 2.
    gmg_cg_threshold : int
        Adaptive preconditioner switch.  When > 0 and enable_gmg=True:
        if the PREVIOUS iteration's CG count < gmg_cg_threshold, skip GMG
        setup for the current iteration and use Jacobi instead.
        Rationale: GMG setup costs ~0.3–1 s/iter; when CG already converges
        fast (<threshold iters) with Jacobi, GMG is not worth its overhead.
        Recommended: 50 (skip GMG when previous CG < 50 iters).
        Default: 0 (always use GMG when enable_gmg=True).
    enable_matrix_free : bool
        2.0 — replace stored-CSR SpMV with on-the-fly MatrixFreeKff.
        Eliminates the ~11 GB Kff allocation at 216k; enables 1M+ elements.
        Preconditioner: Jacobi only (GMG + matrix-free interaction is Phase 2.5).
        Incompatible with enable_gmg / enable_rediscr_gmg (raises ValueError).
        Default: False.
    enable_block_jacobi : bool
        2.1 — upgrade the matrix-free preconditioner from scalar Jacobi to
        3×3 node-block Jacobi (NodeBlockJacobiMF).  Captures x/y/z coupling
        at each node without assembling the global K; substantially better
        conditioned for bending/torsion BVPs where Jacobi is insufficient
        (MBB, bridge, bracket, torsion at 512k+).
        Requires enable_matrix_free=True.  Default: False.
    enable_fused_cuda : bool
        3.0 — replace the 3-pass CuPy matvec (gather + GEMV + bincount scatter)
        with a single fused CUDA kernel.  Measured 5.8-6.2× speedup over
        CuPy FP32 matfree at 64k-5M elements (FP64 parity 6.98e-8).  Requires
        enable_matrix_free=True and enable_mixed_precision=True; falls back to
        CuPy matvec when either is disabled.  Default: False.
    """

    def __init__(
        self,
        *args,
        grid_dims: Optional[Tuple[int, int, int]] = None,
        enable_warm_start: bool = True,
        enable_gmg: bool = False,
        gmg_levels: int = 3,
        gmg_smooth_iters: int = 2,
        enable_mixed_precision: bool = False,
        mixed_precision_refine_iters: int = 2,
        enable_profiling: bool = False,
        gmg_cg_threshold: int = 0,
        enable_rediscr_gmg: bool = False,
        rediscr_gmg_levels: int = 3,
        enable_matrix_free: bool = False,
        enable_block_jacobi: bool = False,
        enable_fused_cuda: bool = False,
        fused_dtype: str = "fp32",
        bf16_ir_outer_tol: float = 1e-5,
        bf16_ir_inner_tol: float = 1e-3,
        bf16_ir_max_outer: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(*args, skip_sparse_assembly=enable_matrix_free, **kwargs)

        self._enable_warm_start      = enable_warm_start
        self._enable_gmg             = enable_gmg
        self._enable_mixed_precision = enable_mixed_precision
        self._mp_refine_iters        = mixed_precision_refine_iters
        self._enable_profiling       = enable_profiling
        self._gmg_cg_threshold       = gmg_cg_threshold   # 0 = always GMG
        self._enable_rediscr_gmg     = enable_rediscr_gmg
        self._enable_matrix_free     = enable_matrix_free
        self._enable_block_jacobi    = enable_block_jacobi and enable_matrix_free
        self._enable_fused_cuda      = (enable_fused_cuda
                                        and enable_matrix_free
                                        and enable_mixed_precision)
        # "fp32" | "bf16" (raw, typically stalls) | "bf16_ir" (FP32 iterative refinement)
        self._fused_dtype            = fused_dtype
        self._bf16_ir_outer_tol      = bf16_ir_outer_tol
        self._bf16_ir_inner_tol      = bf16_ir_inner_tol
        self._bf16_ir_max_outer      = bf16_ir_max_outer
        # Lazy-init: will hold a FusedMatvec instance (see _solve_cupy_matfree)
        self._fused_op = None

        if enable_fused_cuda and not (enable_matrix_free and enable_mixed_precision):
            raise ValueError(
                "enable_fused_cuda requires both enable_matrix_free=True and "
                "enable_mixed_precision=True.  The fused CUDA kernel is FP32-only."
            )

        if fused_dtype not in ("fp32", "bf16", "bf16_ir"):
            raise ValueError(
                f"fused_dtype must be 'fp32', 'bf16', or 'bf16_ir'; got {fused_dtype!r}"
            )
        if fused_dtype != "fp32" and not enable_fused_cuda:
            raise ValueError(
                "fused_dtype='bf16*' requires enable_fused_cuda=True"
            )

        if enable_block_jacobi and not enable_matrix_free:
            raise ValueError(
                "enable_block_jacobi requires enable_matrix_free=True.  "
                "Block-Jacobi is a matrix-free preconditioner; it has no effect on the "
                "sparse assembled path."
            )

        if enable_matrix_free and (enable_gmg or enable_rediscr_gmg):
            raise ValueError(
                "enable_matrix_free is incompatible with enable_gmg / enable_rediscr_gmg "
                "in Phase 2.  GMG + matrix-free interaction is deferred to Phase 2.5."
            )

        # § 1.1 Warm-start buffers (one per backend)
        self._u_prev_cupy  = None   # cp.ndarray | None
        self._u_prev_torch = None   # torch.Tensor | None
        self._prev_penal   = None   # float | None — detect schedule transitions

        # Diagnostics — updated every solve() call
        self.last_cg_iters: int = 0   # CG iterations in most recent solve
        # Residual-history capture: set self.capture_cg_history = True to have
        # the solver populate self.last_cg_history with ||r||/||b|| per CG iter
        # (first entry = initial residual ratio). Only the scalar-Jacobi path
        # in _solve_cupy_matfree supports capture right now (the path used by
        # enable_matrix_free + enable_mixed_precision + enable_fused_cuda).
        self.capture_cg_history: bool = False
        self.last_cg_history: list = []
        # Per-component timing (only meaningful when enable_profiling=True).
        # Keys: 'assemble_ms', 'gmg_setup_ms', 'cg_ms', 'sensitivity_ms'
        self.last_timing: dict = {}

        if not enable_matrix_free:
            # Fast diagonal extraction: precompute which data-array positions contribute
            # to the diagonal of K_ff.  The CSR stored in _Kff_indptr_cpu/_Kff_cols_cpu
            # uses a pre-summation duplicate layout (multiple element contributions per
            # (i,j) pair).  _diag_scatter_{pos,row} find ALL diagonal entries so that
            # diag[i] = sum(Kff.data[_diag_scatter_pos[j]] for j where _diag_scatter_row[j]==i)
            # which is then efficiently computed with cp.bincount each SIMP iteration.
            _indptr = self._Kff_indptr_cpu   # (n_free+1,) int32
            _cols   = self._Kff_cols_cpu     # (nnz_dup,)  int32 — with duplicates
            _n      = self._n_free
            _row    = np.repeat(np.arange(_n, dtype=np.int32), np.diff(_indptr))
            _is_diag = (_cols == _row)
            self._diag_scatter_pos_cpu = np.nonzero(_is_diag)[0].astype(np.int32)
            self._diag_scatter_row_cpu = _row[_is_diag].astype(np.int32)
            # GPU arrays lazily uploaded on first _solve_cupy call
            self._diag_scatter_pos_gpu = None   # cp.ndarray int32
            self._diag_scatter_row_gpu = None   # cp.ndarray int32

            # § Canonical Kff scatter index — avoids 91 ms sum_duplicates on each SpMV
            self._precompute_canonical_kff_layout(_indptr, _cols, _row, _n)
            # GPU arrays for the canonical Kff — lazily initialised on first _solve_cupy
            self._Kff_gpu          = None   # persistent cpsp.csr_matrix (canonical, FP64)
            self._Kff_scatter_gpu  = None   # cp.ndarray int32  (nnz_dup,)
            self._Kff_gpu_ready    = False
        else:
            # Matrix-free: none of the sparse Kff structures are needed.
            self._diag_scatter_pos_cpu = None
            self._diag_scatter_row_cpu = None
            self._diag_scatter_pos_gpu = None
            self._diag_scatter_row_gpu = None
            self._Kff_gpu          = None
            self._Kff_scatter_gpu  = None
            self._Kff_gpu_ready    = False

        # § 1.2B Geometric multigrid
        if enable_gmg:
            if grid_dims is None:
                raise ValueError(
                    "grid_dims=(nelx, nely, nelz) is required when enable_gmg=True."
                )
            nelx, nely, nelz = grid_dims
            print(
                f"[SolverV2] Building HexGridGMG: {nelx}×{nely}×{nelz}, "
                f"{gmg_levels} levels, {gmg_smooth_iters} smooth iters/level …"
            )
            self._gmg = HexGridGMG(
                nelx, nely, nelz,
                free=self._free,
                n_levels=gmg_levels,
                n_smooth=gmg_smooth_iters,
            )
            print(
                f"[SolverV2] GMG ready.  Free DOFs per level: "
                + str(self._gmg._n_free)
            )
        else:
            self._gmg = None

        # § 1.2C Re-discretization GMG
        if enable_rediscr_gmg:
            if grid_dims is None:
                raise ValueError(
                    "grid_dims=(nelx, nely, nelz) is required when enable_rediscr_gmg=True."
                )
            from .pub_simp_solver import KE_UNIT_3D as _KE_UNIT_3D
            nelx, nely, nelz = grid_dims
            print(
                f"[SolverV2] Building RedisCGMG: {nelx}×{nely}×{nelz}, "
                f"{rediscr_gmg_levels} levels, {gmg_smooth_iters} smooth iters/level …"
            )
            self._rediscr_gmg = RedisCGMG(
                nelx, nely, nelz,
                free=self._free,
                E0=self.E0, Emin=self.Emin,
                KE_UNIT=_KE_UNIT_3D,
                n_levels=rediscr_gmg_levels,
                n_smooth=gmg_smooth_iters,
            )
            print(
                f"[SolverV2] RedisCGMG ready.  Free DOFs per level: "
                + str(self._rediscr_gmg._n_free)
            )
            # Pre-warm cuSOLVER JIT kernels (cholesky + solve_triangular carry
            # ~150 ms cold-start overhead each; pay it once during construction).
            try:
                import cupy as _cp
                import cupyx.scipy.linalg as _cpla
                _A = _cp.eye(10, dtype=_cp.float64)
                _L = _cp.linalg.cholesky(_A)
                _v = _cp.zeros(10, dtype=_cp.float64)
                _  = _cpla.solve_triangular(_L, _v, lower=True)
                _  = _cpla.solve_triangular(_L, _v, lower=True, trans='T')
                # Warm up cp.dot + float() for n_free-sized vectors.
                # Without this, the first float(cp.dot(v, v)) call after a
                # stream synchronize() triggers ~86 ms of JIT compilation,
                # inflating the cg_ms measurement on the first warm-start iter.
                _vn = _cp.zeros(len(self._free), dtype=_cp.float64)
                _ = float(_cp.dot(_vn, _vn))
                _cp.cuda.stream.get_current_stream().synchronize()
                del _A, _L, _v, _vn, _
            except Exception:
                pass  # non-CuPy backend: warmup is a no-op
        else:
            self._rediscr_gmg = None

        # § 2.0 Matrix-free operator — built lazily on first CuPy solve
        # (edof_gpu / KE_unit_gpu / free_gpu not yet uploaded if backend=cpu)
        self._matfree_op: Optional[MatrixFreeKff] = None   # set in _solve_cupy_matfree

        # § 2.1 Node-block Jacobi — built lazily alongside MatrixFreeKff
        self._block_jacobi_op: Optional[NodeBlockJacobiMF] = None

    # ------------------------------------------------------------------
    # § Canonical Kff layout precomputation
    # ------------------------------------------------------------------

    def _precompute_canonical_kff_layout(self, _indptr, _cols, _row, _n):
        """Precompute the scatter index from duplicate-entry CSR to canonical CSR.

        The FEM assembler stores duplicate (i,j) contributions in sorted CSR form.
        cuSPARSE requires canonical CSR (unique (i,j) pairs, sum of duplicates).
        Without this, the first SpMV on a freshly assembled Kff triggers
        sum_duplicates() inside CuPy which costs ~91 ms per SIMP iteration.

        This method computes _Kff_scatter_idx_cpu (int32, nnz_dup,) such that:
            canonical_data[scatter_idx[j]] += full_data[j]  for all j

        Per-iter usage (one GPU kernel):
            Kff.data[:] = cp.bincount(scatter_idx_gpu, weights=full_data, ...)

        One-time CPU cost: O(nnz_dup) — run-length encoding on the globally-sorted
        duplicate-entry CSR.  No scipy needed.  ~0.5 s at 64k, ~1.5 s at 216k.
        """
        _nnz_dup = len(_cols)

        # The duplicate-entry CSR is globally sorted by (row, col) — an invariant
        # of the FEM assembly in fem_gpu.py.  Equal (row,col) pairs are therefore
        # CONSECUTIVE.  Run-length encoding directly gives the canonical scatter index
        # without any sorting or searching — O(nnz_dup) instead of O(nnz log nnz).
        _dup_key = _row.astype(np.int64) * _n + _cols.astype(np.int64)

        # Detect transitions between unique (row,col) pairs
        _is_new = np.empty(_nnz_dup, dtype=bool)
        _is_new[0] = True
        _is_new[1:] = (_dup_key[1:] != _dup_key[:-1])

        # Scatter index: position in canonical data array for each duplicate entry
        self._Kff_scatter_idx_cpu = (np.cumsum(_is_new) - 1).astype(np.int32)
        self._Kff_nnz_canon = int(_is_new.sum())

        # Canonical CSR structure: indptr and column indices
        _canon_mask       = _is_new                           # first occurrence of each pair
        _canon_rows       = _row[_canon_mask].astype(np.int32)
        _canon_cols       = _cols[_canon_mask].astype(np.int32)

        # Build canonical indptr from per-row entry counts
        _counts = np.bincount(_canon_rows, minlength=_n).astype(np.int32)
        self._Kff_canon_indptr = np.zeros(_n + 1, dtype=np.int32)
        np.cumsum(_counts, out=self._Kff_canon_indptr[1:])
        self._Kff_canon_indices = _canon_cols

        # Canonical diagonal positions — used to extract Kff diagonal each iter.
        # In canonical CSR, each (i,i) entry appears exactly once (FEM: K_ii > 0).
        # _Kff_canon_diag_pos_cpu[k]  = position in canonical data array
        # _Kff_canon_diag_row_cpu[k]  = free-DOF row index (= col index) for that entry
        _is_diag = (_canon_cols == _canon_rows)
        self._Kff_canon_diag_pos_cpu = np.where(_is_diag)[0].astype(np.int32)
        self._Kff_canon_diag_row_cpu = _canon_rows[_is_diag].astype(np.int32)
        # GPU arrays — uploaded lazily alongside _Kff_gpu init
        self._Kff_canon_diag_pos_gpu = None
        self._Kff_canon_diag_row_gpu = None

    # ------------------------------------------------------------------
    # § CuPy solve — override with Phase 1/2 stack
    # ------------------------------------------------------------------

    def _solve_cupy(self, rho_phys: np.ndarray, penal: float) -> tuple:
        if self._enable_matrix_free:
            return self._solve_cupy_matfree(rho_phys, penal)
        import time as _time
        cp = self._cp
        import cupyx.scipy.sparse    as cpsp
        import cupyx.scipy.sparse.linalg as cpspla

        def _sync_time():
            """Synchronise GPU stream and return wall-clock timestamp.
            Sync only happens when profiling is active; otherwise we just
            snapshot perf_counter() and let async GPU work continue.
            The actual accuracy requirement is met by the mandatory syncs
            inside float() calls on GPU scalars."""
            if self._enable_profiling:
                cp.cuda.Stream.null.synchronize()
            return _time.perf_counter()

        E0, Emin = self.E0, self.Emin

        # ── Lazy init: persistent canonical Kff + cuSPARSE descriptor ────────
        # First call only: upload scatter index, create canonical CSR matrix,
        # and warm up the cuSPARSE SpMV descriptor (avoids 91 ms per-iter
        # sum_duplicates cost on freshly assembled duplicate-entry matrices).
        if not self._Kff_gpu_ready:
            self._Kff_scatter_gpu = cp.asarray(self._Kff_scatter_idx_cpu)
            _canon_data_init = cp.ones(self._Kff_nnz_canon, dtype=cp.float64)
            self._Kff_gpu = cpsp.csr_matrix(
                (_canon_data_init,
                 cp.asarray(self._Kff_canon_indices),
                 cp.asarray(self._Kff_canon_indptr)),
                shape=(self._n_free, self._n_free),
            )
            # Upload canonical diagonal lookup arrays
            self._Kff_canon_diag_pos_gpu = cp.asarray(self._Kff_canon_diag_pos_cpu)
            self._Kff_canon_diag_row_gpu = cp.asarray(self._Kff_canon_diag_row_cpu)
            # Warm up cuSPARSE descriptor: first SpMV establishes the handle so
            # all subsequent SpMVs (pre-t2 residual, CG) can dispatch in <1 ms.
            _dummy_v = cp.zeros(self._n_free, dtype=cp.float64)
            _ = self._Kff_gpu @ _dummy_v
            cp.cuda.Stream.null.synchronize()
            del _dummy_v, _
            self._Kff_gpu_ready = True

        rho_gpu  = cp.asarray(rho_phys, dtype=cp.float64)
        E_e      = Emin + (E0 - Emin) * rho_gpu ** penal

        t0 = _sync_time()

        # ── § 1.3  Mixed-precision: choose CG dtype ───────────────────────
        # FP32 inner CG: 2× bandwidth reduction (smaller Kff), 64× faster
        # compute on RTX 4090 tensor cores.  cg_tol=1e-5 is safely above
        # FP32 machine epsilon (~1e-7), so no iterative refinement needed.
        _cg_dt = cp.float32 if self._enable_mixed_precision else cp.float64

        # ── Build Kff via scatter-add into persistent canonical CSR ───────────
        # Instead of creating a new cpsp.csr_matrix each iteration (which forces
        # a 91 ms sum_duplicates() on the first SpMV), scatter-add the duplicate
        # element contributions directly into the canonical data array using
        # cp.bincount.  The persistent _Kff_gpu's cuSPARSE descriptor was
        # pre-warmed above, so SpMV dispatch is <1 ms every iteration.
        Kff_data_f64 = E_e[self._elem_sorted_gpu] * self._KE0_sorted_gpu
        cp.copyto(
            self._Kff_gpu.data,
            cp.bincount(
                self._Kff_scatter_gpu,
                weights=Kff_data_f64,
                minlength=self._Kff_nnz_canon,
            ),
        )
        Kff = self._Kff_gpu   # canonical, descriptor pre-warmed → SpMV <1 ms

        # FP32 mixed-precision: wrap the canonical FP64 data as FP32 matrix.
        # Reuses already-uploaded GPU index arrays; only the data cast is new.
        if _cg_dt == cp.float32:
            Kff = cpsp.csr_matrix(
                (self._Kff_gpu.data.astype(cp.float32),
                 self._Kff_gpu.indices,
                 self._Kff_gpu.indptr),
                shape=(self._n_free, self._n_free),
            )

        t1 = _sync_time()

        # ── Fast diagonal extraction from canonical Kff ──────────────────────
        # Kff is now in canonical CSR (no duplicate (i,j) entries).
        # The diagonal K_ii corresponds to the canonical entry where col == row.
        # Precomputed canonical diag positions allow direct indexed extraction —
        # one GPU gather op instead of the old bincount over 36M duplicate entries.
        _kff_diag = cp.zeros(self._n_free, dtype=cp.float64)
        _kff_diag[self._Kff_canon_diag_row_gpu] = \
            self._Kff_gpu.data[self._Kff_canon_diag_pos_gpu]
        if _cg_dt == cp.float32:
            _kff_diag = _kff_diag.astype(cp.float32)

        # ── Preconditioner ────────────────────────────────────────────────
        _use_gmg = (
            self._enable_gmg
            and self._gmg is not None
            and (
                self._gmg_cg_threshold <= 0
                or self.last_cg_iters == 0
                or self.last_cg_iters >= self._gmg_cg_threshold
            )
        )
        _use_rediscr_gmg = (
            self._enable_rediscr_gmg
            and self._rediscr_gmg is not None
            and not _use_gmg   # Galerkin GMG takes precedence if both enabled
        )

        # ── § 1.1  Warm-start initial guess (pre-t2 setup) ───────────────
        # Penal-jump check must happen before GPU work is queued so that
        # a stale x0 doesn't get used for the pre-t2 residual computation.
        _penal_jumped = (
            self._prev_penal is not None
            and abs(penal - self._prev_penal) > 0.5
        )
        if _penal_jumped and self._u_prev_cupy is not None:
            self._u_prev_cupy = None   # discard stale warm-start
        self._prev_penal = penal

        x0_f64 = self._u_prev_cupy if (self._enable_warm_start and self._u_prev_cupy is not None) else None

        # ── Pre-t2 residual overlap: RedisCGMG warm-start convergence check ──
        # Queue  x0  →  F - Kff @ x0  and the norms on the GPU *before*
        # the RedisCGMG.setup / HexGridGMG.setup / Jacobi-setup calls.
        # Those setup kernels run concurrently on the GPU while Python queues
        # the SpMV.  The t2 sync then covers ALL of this work "for free".
        # After t2 the GPU is warm, so float(r0_sq_gpu) costs ~0.3 ms instead
        # of the ~19 ms cold-GPU WDDM overhead inside _cupy_pcg.
        _pre_t2_ws_skip = False        # True if warm-start already checked
        _r0_sq_gpu      = None         # GPU scalar; read after t2
        _b_norm_sq_gpu  = None         # GPU scalar; read after t2 (first call only)
        _x0_pre         = None         # CuPy x0 cast to _cg_dt
        _F_cg_pre       = None         # CuPy F cast to _cg_dt
        if _use_rediscr_gmg and x0_f64 is not None:
            _x0_pre   = x0_f64.astype(_cg_dt)           # async GPU dtype cast / copy
            _F_cg_pre = self._F_free_gpu.astype(_cg_dt)  # async GPU dtype cast / copy
            _r0_prelim  = _F_cg_pre - Kff @ _x0_pre     # async SpMV using current Kff
            _r0_sq_gpu  = cp.dot(_r0_prelim, _r0_prelim) # async reduction
            if not hasattr(self, '_b_norm_sq_cached') or self._b_norm_sq_cached is None:
                _b_norm_sq_gpu = cp.dot(_F_cg_pre, _F_cg_pre)  # async; only once

        _diag_tol = 1e-8 if _cg_dt == cp.float32 else 1e-12
        _M_inv = None   # populated in Jacobi branch; used by custom FP32 CG
        if _use_gmg:
            self._gmg.setup(Kff, fine_diag=_kff_diag)
            M_op = self._gmg.as_linear_operator()
        elif _use_rediscr_gmg:
            # Lazy: defer Galerkin SpGEMM setup until AFTER warm-start check.
            # When warm-start already converges (0 CG iters needed), we skip
            # the ~50 ms setup entirely.  For iters that DO need CG, setup()
            # is called just before _cupy_pcg below.
            M_op = None   # not used for _use_rediscr_gmg path (uses vcycle directly)
        else:
            _diag  = cp.where(cp.abs(_kff_diag) > _diag_tol, _kff_diag, cp.ones_like(_kff_diag))
            _M_inv = (1.0 / _diag).astype(_cg_dt)   # explicit cast prevents FP64 promotion
            M_op   = cpspla.LinearOperator(
                (self._n_free, self._n_free),
                matvec=lambda v: _M_inv * v.astype(_cg_dt),
                dtype=_cg_dt,
            )

        t2 = _sync_time()

        # ── Post-t2: read pre-computed residual norms (GPU warm → ~0.3 ms) ──
        # Use pre-t2 x0 (already cast to _cg_dt) when available (RedisCGMG path);
        # otherwise fall back to the raw FP64 warm-start (Jacobi / HexGMG paths).
        if _x0_pre is not None:
            x0 = _x0_pre                                       # already _cg_dt
        elif x0_f64 is not None:
            x0 = x0_f64.astype(_cg_dt) if _cg_dt != cp.float64 else x0_f64
        else:
            x0 = None
        F_cg = _F_cg_pre if _F_cg_pre is not None else self._F_free_gpu.astype(_cg_dt)
        if _r0_sq_gpu is not None:
            if _b_norm_sq_gpu is not None:
                self._b_norm_sq_cached = float(_b_norm_sq_gpu)    # cache; paid once
            _r0_sq = float(_r0_sq_gpu)                            # ~0.3 ms warm sync
            _tol_sq = self.cg_tol ** 2 * self._b_norm_sq_cached
            if _r0_sq <= _tol_sq:
                # x0 is already converged for the current K/F — skip CG entirely.
                _pre_t2_ws_skip = True

        # ── CG solve ──────────────────────────────────────────────────────
        _iters = [0]

        if _cg_dt == cp.float32 and not _use_gmg and not _use_rediscr_gmg:
            # Custom typed CG: avoids CuPy's fp32→fp64 scalar promotion in cp.dot
            U_free_cg, n_cg, converged = _cupy_cg(
                Kff, F_cg, x0=x0,
                tol=self.cg_tol, maxiter=self.cg_maxiter, M_inv=_M_inv,
            )
            if not converged and x0 is not None:
                # Warm-start failed: retry from zero
                U_free_cg, n_cg, converged = _cupy_cg(
                    Kff, F_cg, x0=None,
                    tol=self.cg_tol, maxiter=self.cg_maxiter, M_inv=_M_inv,
                )
            if not converged:
                print(
                    f"[SolverV2] WARNING: FP32 CG did not converge in "
                    f"{self.cg_maxiter} iters (n_free={self._n_free})"
                )
            _iters[0] = n_cg
        elif _use_rediscr_gmg:
            if _pre_t2_ws_skip:
                # Warm-start already converged → skip both setup() AND CG.
                # Lazy setup avoids ~50 ms Galerkin SpGEMM on every warm-start iter.
                U_free_cg = _x0_pre   # already _cg_dt
                n_cg      = 0
                converged = True
            else:
                # Lazy Galerkin setup: runs here (after warm-start check fails).
                # For iters that DO need CG, this costs ~50 ms but is necessary.
                self._rediscr_gmg.setup(Kff, rho_phys, penal, fine_diag=_kff_diag)
                # Full PCG: either no warm-start or warm-start didn't converge.
                _vcycle_fn = self._rediscr_gmg.vcycle
                U_free_cg, n_cg, converged = _cupy_pcg(
                    Kff, F_cg, M_op=_vcycle_fn, x0=x0,
                    tol=self.cg_tol, maxiter=self.cg_maxiter,
                )
                if not converged and x0 is not None:
                    # Warm-start degraded convergence: retry from zero
                    U_free_cg, n_cg, converged = _cupy_pcg(
                        Kff, F_cg, M_op=_vcycle_fn, x0=None,
                        tol=self.cg_tol, maxiter=self.cg_maxiter,
                    )
                if not converged:
                    print(
                        f"[SolverV2] WARNING: RedisCGMG CG did not converge in "
                        f"{self.cg_maxiter} iters (n_free={self._n_free})"
                    )
            _iters[0] = n_cg
        else:
            def _cg_callback(_xk):
                _iters[0] += 1

            U_free_cg, info = cpspla.cg(
                Kff, F_cg,
                x0=x0, tol=self.cg_tol, maxiter=self.cg_maxiter, M=M_op,
                callback=_cg_callback,
            )
            if info < 0:
                raise RuntimeError(f"CuPy CG breakdown (info={info})")
            if info > 0 and x0 is not None:
                # Warm-start degraded convergence (common at schedule transitions
                # that weren't caught by penal-jump detection, e.g. beta changes).
                # Retry from zero — cheap since we pay maxiter only once.
                _iters[0] = 0
                U_free_cg, info = cpspla.cg(
                    Kff, F_cg, x0=None,
                    tol=self.cg_tol, maxiter=self.cg_maxiter, M=M_op,
                    callback=_cg_callback,
                )
            if info > 0:
                print(
                    f"[SolverV2] WARNING: CuPy CG did not converge in "
                    f"{self.cg_maxiter} iters (n_free={self._n_free})"
                )

        # Upcast to FP64 for compliance, sensitivity, and warm-start storage.
        U_free_gpu = U_free_cg.astype(cp.float64) if _cg_dt == cp.float32 else U_free_cg

        self.last_cg_iters = _iters[0]

        t3 = _sync_time()

        # ── § 1.1  Save warm-start buffer ─────────────────────────────────
        if self._enable_warm_start:
            self._u_prev_cupy = U_free_gpu.copy()

        # ── Compliance and sensitivity (unchanged from paper-1) ───────────
        compliance = float(cp.dot(self._F_free_gpu, U_free_gpu).get())

        U_gpu = cp.zeros(self.ndof, dtype=cp.float64)
        U_gpu[self._free_gpu] = U_free_gpu

        Ue  = U_gpu[self._edof_gpu]
        KUe = Ue @ self._KE_unit_gpu
        ce  = (KUe * Ue).sum(axis=1)
        dc_phys_gpu = -penal * (E0 - Emin) * rho_gpu ** (penal - 1) * ce

        t4 = _sync_time()

        if self._enable_profiling:
            self.last_timing = {
                "assemble_ms":   (t1 - t0) * 1e3,
                "gmg_setup_ms":  (t2 - t1) * 1e3,   # 0 if Jacobi fallback
                "cg_ms":         (t3 - t2) * 1e3,
                "sensitivity_ms":(t4 - t3) * 1e3,
                "total_ms":      (t4 - t0) * 1e3,
                "used_gmg":      _use_gmg,
            }

        return compliance, cp.asnumpy(dc_phys_gpu)

    # ------------------------------------------------------------------
    # § 2.0 Matrix-free CuPy solve
    # ------------------------------------------------------------------

    def _solve_cupy_matfree(self, rho_phys: np.ndarray, penal: float) -> tuple:
        """
        Phase 2 matrix-free solve: K_ff · u via on-the-fly element recomputation.

        No sparse Kff is assembled or stored.  VRAM usage is O(n_elem) instead
        of O(nnz_Kff):
          - 216k elements: ~20 MB (edof) vs ~11.4 GB (canonical CSR Kff)
          - 1M   elements: ~90 MB (edof) — fits on 24 GB card, sparse impossible

        Preconditioner: Jacobi (diag(K_ff) extracted element-wise, no K assembly).
        GMG preconditioner is deferred to Phase 2.5.

        Warm-start and penal-jump detection are preserved from Phase 1.
        Mixed-precision (FP32 matvec) is available via enable_mixed_precision.
        """
        import time as _time
        cp = self._cp

        def _sync_time():
            if self._enable_profiling:
                cp.cuda.Stream.null.synchronize()
            return _time.perf_counter()

        E0, Emin = self.E0, self.Emin

        # ── Lazy MatrixFreeKff init (first call) ─────────────────────────
        if self._matfree_op is None:
            self._matfree_op = MatrixFreeKff(
                edof_gpu   = self._edof_gpu,
                KE_unit_gpu= self._KE_unit_gpu,
                free_gpu   = self._free_gpu,
                n_free     = self._n_free,
                ndof       = self.ndof,
            )
            print(
                f"[SolverV2] MatrixFreeKff ready — "
                f"n_elem={self._n_elem:,}  ke_size={self._matfree_op._ke_size}  "
                f"n_free={self._n_free:,}  ndof={self.ndof:,}"
            )

        mf = self._matfree_op

        # ── Lazy FusedMatvec init (Phase 3 — fused CUDA kernel) ───────────
        if self._enable_fused_cuda and self._fused_op is None:
            from gpu_fem.cuda_fused_matvec import FusedMatvec
            self._fused_op = FusedMatvec(
                edof_gpu    = self._edof_gpu,
                KE_unit_gpu = self._KE_unit_gpu,
                ndof        = self.ndof,
            )
            print(
                f"[SolverV2] FusedMatvec ready — fused gather/GEMV/scatter "
                f"(FP32, {self._fused_op.BLOCK_X} threads/block)"
            )

        # ── Lazy NodeBlockJacobiMF init (first call) ──────────────────────
        if self._enable_block_jacobi and self._block_jacobi_op is None:
            self._block_jacobi_op = NodeBlockJacobiMF(
                edof_gpu    = self._edof_gpu,
                KE_unit_gpu = self._KE_unit_gpu,
                free_gpu    = self._free_gpu,
                n_free      = self._n_free,
                ndof        = self.ndof,
            )
            print(
                f"[SolverV2] NodeBlockJacobiMF ready — "
                f"n_all_free_nodes={self._block_jacobi_op._n_all_free:,}  "
                f"n_partial_dofs={self._block_jacobi_op._partial_local_gpu.size}"
            )

        # ── Penal-jump warm-start reset ───────────────────────────────────
        _penal_jumped = (
            self._prev_penal is not None
            and abs(penal - self._prev_penal) > 0.5
        )
        if _penal_jumped and self._u_prev_cupy is not None:
            self._u_prev_cupy = None
        self._prev_penal = penal

        rho_gpu = cp.asarray(rho_phys, dtype=cp.float64)
        E_e     = Emin + (E0 - Emin) * rho_gpu ** penal    # (n_elem,) float64

        _cg_dt = cp.float32 if self._enable_mixed_precision else cp.float64
        E_e_cg = E_e.astype(_cg_dt)

        t0 = _sync_time()

        # ── Matrix-free operator (closure over E_e_cg) ───────────────────
        # For bf16_ir we also need a separate FP32 operator for the outer
        # residual — so build both flavors up front and pick the primary.
        _Kop_fp32_fused = None
        _Kop_bf16_fused = None
        if self._enable_fused_cuda and self._fused_op is not None:
            _fused = self._fused_op
            _free  = self._free_gpu
            def _Kop_fp32_fused(v, _fused=_fused, _free=_free):
                return _fused.matvec(v, E_e_cg, _free, dtype="fp32")
            def _Kop_bf16_fused(v, _fused=_fused, _free=_free):
                return _fused.matvec(v, E_e_cg, _free, dtype="bf16")

            if self._fused_dtype == "fp32":
                _Kop = _Kop_fp32_fused
            elif self._fused_dtype == "bf16":
                _Kop = _Kop_bf16_fused
            else:   # "bf16_ir" — inner loop uses bf16, outer residual uses fp32
                _Kop = _Kop_bf16_fused
        else:
            def _Kop(v):
                return mf.matvec(v, E_e_cg)

        # ── Preconditioner (Jacobi diagonal — always needed) ─────────────
        _kff_diag = mf.extract_diagonal(E_e)       # always FP64 for accuracy
        _diag_tol = 1e-8 if _cg_dt == cp.float32 else 1e-12
        _diag     = cp.where(cp.abs(_kff_diag) > _diag_tol,
                             _kff_diag, cp.ones_like(_kff_diag))
        _M_inv    = (1.0 / _diag).astype(_cg_dt)

        # ── § 2.1  Node-block Jacobi (optional upgrade over scalar Jacobi) ─
        if self._enable_block_jacobi and self._block_jacobi_op is not None:
            self._block_jacobi_op.update(E_e, scalar_diag=_kff_diag)
            _bj_op = self._block_jacobi_op  # local alias for closure
            _M_op  = lambda v: _bj_op.apply(v)
        else:
            _M_op = None   # Jacobi path uses _M_inv directly in _cupy_cg

        t1 = _sync_time()

        # ── Warm-start initial guess ──────────────────────────────────────
        x0_f64 = (
            self._u_prev_cupy
            if (self._enable_warm_start and self._u_prev_cupy is not None)
            else None
        )
        x0 = x0_f64.astype(_cg_dt) if x0_f64 is not None else None
        F_cg = self._F_free_gpu.astype(_cg_dt)

        t2 = _sync_time()

        # ── CG solve ─────────────────────────────────────────────────────
        if _M_op is not None:
            # Block-Jacobi path: use _cupy_pcg (general operator interface)
            U_free_cg, n_cg, converged = _cupy_pcg(
                _Kop, F_cg.astype(cp.float64), _M_op,
                x0=x0_f64,
                tol=self.cg_tol, maxiter=self.cg_maxiter,
            )
            U_free_cg = U_free_cg.astype(_cg_dt)
            if not converged and x0_f64 is not None:
                U_free_cg, n_cg, converged = _cupy_pcg(
                    _Kop, F_cg.astype(cp.float64), _M_op,
                    x0=None,
                    tol=self.cg_tol, maxiter=self.cg_maxiter,
                )
                U_free_cg = U_free_cg.astype(_cg_dt)
        else:
            # Scalar Jacobi path (original)
            _hist = [] if self.capture_cg_history else None
            if self._fused_dtype == "bf16_ir" and _Kop_fp32_fused is not None:
                # ── Mixed-precision iterative refinement ─────────────────
                # Inner CG uses BF16 WMMA matvec; outer residual uses FP32
                # matvec.  Loop until ||r||/||b|| < outer_tol or max_outer
                # reached.  Total inner iterations reported as n_cg.
                n_cg = 0
                b_norm = cp.sqrt(cp.dot(F_cg, F_cg))
                x_acc = cp.zeros_like(F_cg) if x0 is None else x0.copy()
                # Initial residual (FP32): r = F - K_fp32 @ x_acc
                r = F_cg - _Kop_fp32_fused(x_acc)
                converged = False
                _verbose_ir = getattr(self, "_bf16_ir_verbose", False)
                for _outer in range(self._bf16_ir_max_outer):
                    r_norm = float(cp.sqrt(cp.dot(r, r)).get())
                    ratio  = r_norm / float(b_norm.get())
                    if _hist is not None:
                        _hist.append(ratio)
                    if _verbose_ir:
                        print(f"    [IR outer {_outer}] ||r||/||b||={ratio:.3e}")
                    if ratio < self._bf16_ir_outer_tol:
                        converged = True
                        break
                    # Inner solve: _Kop_bf16 · dx = r, loose tol
                    dx, n_inner, _conv_inner = _cupy_cg(
                        _Kop_bf16_fused, r, x0=None,
                        tol=self._bf16_ir_inner_tol,
                        maxiter=self.cg_maxiter,
                        M_inv=_M_inv, history=None,
                    )
                    n_cg += n_inner
                    if _verbose_ir:
                        print(f"    [IR outer {_outer}] inner_cg={n_inner} conv={_conv_inner}")
                    x_acc = x_acc + dx
                    # Recompute FP32 residual for next outer iter
                    r = F_cg - _Kop_fp32_fused(x_acc)
                U_free_cg = x_acc
            else:
                U_free_cg, n_cg, converged = _cupy_cg(
                    _Kop, F_cg, x0=x0,
                    tol=self.cg_tol, maxiter=self.cg_maxiter, M_inv=_M_inv,
                    history=_hist,
                )
                if not converged and x0 is not None:
                    # Warm-start degraded convergence — retry from zero
                    U_free_cg, n_cg, converged = _cupy_cg(
                        _Kop, F_cg, x0=None,
                        tol=self.cg_tol, maxiter=self.cg_maxiter, M_inv=_M_inv,
                        history=_hist,
                    )
            if self.capture_cg_history:
                self.last_cg_history = _hist
        if not converged:
            prec_name = "block-Jacobi" if _M_op is not None else "Jacobi"
            print(
                f"[SolverV2] WARNING: matrix-free CG ({prec_name}) did not converge in "
                f"{self.cg_maxiter} iters (n_free={self._n_free})"
            )

        self.last_cg_iters = n_cg

        # Upcast to FP64 for compliance, sensitivity, and warm-start storage
        U_free_gpu = U_free_cg.astype(cp.float64) if _cg_dt == cp.float32 else U_free_cg

        t3 = _sync_time()

        # ── § 1.1  Save warm-start buffer ────────────────────────────────
        if self._enable_warm_start:
            self._u_prev_cupy = U_free_gpu.copy()

        # ── Compliance and sensitivity (unchanged from Phase 1) ───────────
        compliance = float(cp.dot(self._F_free_gpu, U_free_gpu).get())

        U_gpu = cp.zeros(self.ndof, dtype=cp.float64)
        U_gpu[self._free_gpu] = U_free_gpu

        Ue  = U_gpu[self._edof_gpu]
        KUe = Ue @ self._KE_unit_gpu
        ce  = (KUe * Ue).sum(axis=1)
        dc_phys_gpu = -penal * (E0 - Emin) * rho_gpu ** (penal - 1) * ce

        t4 = _sync_time()

        if self._enable_profiling:
            self.last_timing = {
                "assemble_ms":   0.0,             # no assembly in matrix-free
                "diag_ms":       (t1 - t0) * 1e3, # diagonal extraction + block update
                "gmg_setup_ms":  (t2 - t1) * 1e3, # warm-start cast (nearly 0)
                "cg_ms":         (t3 - t2) * 1e3,
                "sensitivity_ms":(t4 - t3) * 1e3,
                "total_ms":      (t4 - t0) * 1e3,
                "used_gmg":      False,
                "matrix_free":   True,
                "block_jacobi":  self._enable_block_jacobi,
                "cg_iters":      n_cg,
            }

        return compliance, cp.asnumpy(dc_phys_gpu)

    # ------------------------------------------------------------------
    # § Torch solve — warm-start only (GMG / MPIR are CuPy-only for now)
    # ------------------------------------------------------------------

    def _solve_torch(self, rho_phys: np.ndarray, penal: float) -> tuple:
        self._ensure_torch_backend()
        torch = self._torch
        dev   = self._torch_device
        E0, Emin = self.E0, self.Emin

        rho_gpu = torch.tensor(rho_phys, dtype=torch.float64, device=dev)
        E_e     = Emin + (E0 - Emin) * rho_gpu ** penal

        Kff_data = E_e[self._elem_sorted_torch] * self._KE0_sorted_torch
        Kff = torch.sparse_csr_tensor(
            self._Kff_indptr_torch.to(torch.int64),
            self._Kff_indices_torch.to(torch.int64),
            Kff_data,
            size=(self._n_free, self._n_free),
            dtype=torch.float64,
            device=dev,
        )

        # § 1.1 Warm-start
        x0_torch = (
            self._u_prev_torch
            if (self._enable_warm_start and self._u_prev_torch is not None)
            else None
        )

        U_free = _torch_cg_warm(
            Kff, self._F_free_gpu,
            x0=x0_torch,
            tol=self.cg_tol, maxiter=self.cg_maxiter,
        )

        if self._enable_warm_start:
            self._u_prev_torch = U_free.detach().clone()

        compliance = float(torch.dot(self._F_free_gpu, U_free).cpu())

        U_gpu = torch.zeros(self.ndof, dtype=torch.float64, device=dev)
        U_gpu[self._free_gpu] = U_free

        Ue  = U_gpu[self._edof_gpu]
        KUe = Ue @ self._KE_unit_gpu
        ce  = (KUe * Ue).sum(dim=1)
        dc_phys_gpu = -penal * (E0 - Emin) * rho_gpu ** (penal - 1) * ce

        return compliance, dc_phys_gpu.cpu().numpy()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Reset warm-start buffers and cached factorizations (e.g. on rmin change)."""
        super().invalidate()
        self._u_prev_cupy  = None
        self._u_prev_torch = None
        # GMG coarse matrices will be rebuilt on the next solve() call automatically.
        # NodeBlockJacobiMF state is per-SIMP-iteration — no reset needed (update()
        # is called every solve, so stale block_inv is always overwritten).
