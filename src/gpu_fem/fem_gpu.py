"""
fem_gpu.py
----------
GPU-accelerated FEM solver for 3D SIMP topology optimization.

Backend priority:
  1. CuPy sparse (fastest — uses GPU sparse BLAS + cuSPARSE)
  2. PyTorch sparse CG (fallback — pure PyTorch, no cupy required)
  3. CPU PyAMG / SciPy (last resort — matches existing TO3D behavior)

All paths expose the same interface:
  GPUFEMSolver.solve(rho_phys, penal) → (compliance, dc_phys)

Memory design (v2 — avoids cuSPARSE COO sort workspace):
  At __init__:
    * The kept COO triplets (free DOF pairs only) are sorted once by (row, col)
      on GPU (fast) or CPU (fallback).
    * Pre-sorted CSR column indices and row-pointer array (indptr) are uploaded
      once and held permanently.
    * For each kept entry the unit KE value (KE_flat[ke_local]) and the element
      index are also stored permanently.
  At each solve:
    * A single gather-multiply: Kff_data = E_e[elem_sorted] * KE0_sorted
    * The CSR matrix is constructed directly from (Kff_data, indices, indptr)
      without any sorting — no cuSPARSE sort workspace, no broadcast intermediate.
    * Peak per-solve VRAM ≈ n_kept × 8B (Kff_data) + small CG temporaries.

For 216k elements this drops peak VRAM from ~13.5 GB to ~4-5 GB.
For 512k elements the solver should fit comfortably in 24 GB.
"""

from __future__ import annotations

import gc
import sys
from typing import Optional

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

from .paths import ensure_local_paths

ensure_local_paths(__file__)

try:
    from .pub_simp_solver import _AMGSolver
    _PYAMG_AVAILABLE = True
except ImportError:
    _PYAMG_AVAILABLE = False


# ── Backend detection ─────────────────────────────────────────────────────────

def detect_gpu_backend() -> str:
    """Detect best available GPU backend. Returns 'cupy', 'torch_cuda', or 'cpu'."""
    try:
        import cupy as cp
        cp.array([1.0])
        return "cupy"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            return "torch_cuda"
    except Exception:
        pass
    return "cpu"


VALID_BACKEND_REQUESTS = {"auto", "cuda", "cupy", "torch_cuda", "cpu"}


def resolve_backend_choice(requested: str = "auto") -> tuple[str, Optional[str]]:
    if requested not in VALID_BACKEND_REQUESTS:
        raise ValueError(
            f"Unknown backend/device '{requested}'. "
            f"Expected one of {sorted(VALID_BACKEND_REQUESTS)}."
        )
    detected = detect_gpu_backend()
    if requested == "auto":
        return detected, None
    if requested == "cuda":
        if detected == "cpu":
            return "cpu", "Requested CUDA, but no GPU available. Using CPU fallback."
        return detected, None
    if requested == "cupy":
        if _cupy_available():
            return "cupy", None
        return "cpu", "Requested CuPy, but CuPy unavailable. Using CPU fallback."
    if requested == "torch_cuda":
        if _torch_cuda_available():
            return "torch_cuda", None
        return "cpu", "Requested Torch CUDA, but CUDA unavailable. Using CPU fallback."
    return "cpu", None


def backend_fallback_order(resolved_backend: str) -> list[str]:
    if resolved_backend == "cupy":
        return ["cupy", "torch_cuda", "cpu"]
    if resolved_backend == "torch_cuda":
        return ["torch_cuda", "cupy", "cpu"]
    return ["cpu"]


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
        return True
    except ImportError:
        return False


def _torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def gpu_mem_used_mb() -> float:
    """Return device-wide GPU memory used (MB) via nvidia-smi, or -1 if unavailable."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        return float(out[0])
    except Exception:
        return -1.0


def cupy_pool_used_mb() -> float:
    """Return CuPy memory pool used bytes (MB), or -1 if CuPy unavailable."""
    try:
        import cupy as cp
        return cp.get_default_memory_pool().used_bytes() / (1024 ** 2)
    except Exception:
        return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# GPU FEM Solver
# ─────────────────────────────────────────────────────────────────────────────

class GPUFEMSolver:
    """
    GPU-accelerated FEM solver. Drop-in replacement for _fea_compute + _AMGSolver.

    Parameters
    ----------
    edof : (n_elem, ke_size) int
    row_idx, col_idx : (n_elem * ke_size²,) int  — full COO sparse indices
    KE_UNIT : (ke_size, ke_size) float
    free : (n_free,) int
    F : (ndof,) float
    ndof : int
    backend : 'auto' | 'cupy' | 'torch_cuda' | 'cpu'
    """

    def __init__(
        self,
        edof: np.ndarray,
        row_idx: np.ndarray,
        col_idx: np.ndarray,
        KE_UNIT: np.ndarray,
        free: np.ndarray,
        F: np.ndarray,
        ndof: int,
        E0: float = 1.0,
        Emin: float = 1e-9,
        backend: str = "auto",
        cg_tol: float = 1e-5,
        cg_maxiter: int = 1000,
        amg_ndof_threshold: int = 3000,
        amg_rebuild_tol: float = 0.15,
        skip_sparse_assembly: bool = False,
    ) -> None:
        self.ndof = ndof
        self.E0 = E0
        self.Emin = Emin
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter

        self._requested_backend = backend
        self._backend, _note = resolve_backend_choice(backend)
        if _note:
            print(f"[GPUFEMSolver] {_note}")
        print(
            f"[GPUFEMSolver] Requested={self._requested_backend} "
            f"Resolved={self._backend}  ndof={ndof:,}"
        )

        # CPU AMG fallback
        if _PYAMG_AVAILABLE:
            self._cpu_solver = _AMGSolver(
                ndof_threshold=amg_ndof_threshold,
                rebuild_tol=amg_rebuild_tol,
            )
        else:
            self._cpu_solver = None

        self._n_free = len(free)
        self._ke_size = KE_UNIT.shape[0]
        self._n_elem = len(edof)

        # Flat KE (unit stiffness, before E_e scaling)
        self._KE_flat = KE_UNIT.ravel().astype(np.float64)

        # ── Free-DOF mask ──────────────────────────────────────────────────
        self._free = free
        self._F_free_cpu = F[free].astype(np.float64)
        self._F_cpu = F.astype(np.float64)
        self._edof = edof

        if not skip_sparse_assembly:
            free_mask = np.zeros(ndof, dtype=bool)
            free_mask[free] = True

            # Keep only COO entries where both row and col are free
            keep = free_mask[row_idx] & free_mask[col_idx]

            # Compact (0..n_free-1) DOF remapping
            free_local = np.full(ndof, -1, dtype=np.int32)
            free_local[free] = np.arange(len(free), dtype=np.int32)
            kff_rows_local = free_local[row_idx[keep]].astype(np.int32)
            kff_cols_local = free_local[col_idx[keep]].astype(np.int32)

            ke_sq = self._ke_size * self._ke_size
            n_kept = int(keep.sum())

            # CPU arrays for fallback solver
            self._row_idx_cpu = row_idx.astype(np.int32, copy=False)
            self._col_idx_cpu = col_idx.astype(np.int32, copy=False)

            # ── Precompute sorted CSR structure (one-time cost) ────────────
            # For each kept entry: which element produced it, and its unit KE value.
            # Then sort by (row_local, col_local) → build CSR indptr permanently.
            # At solve time: Kff_data = E_e[elem_sorted] * KE0_sorted (one gather).
            # No COO sort per iteration, no cuSPARSE sort workspace.
            self._precompute_sorted_csr(keep, kff_rows_local, kff_cols_local, ke_sq, n_kept)
        else:
            # Matrix-free mode: no sparse K needed; skip the O(nnz) setup entirely.
            self._row_idx_cpu = None
            self._col_idx_cpu = None
            self._Kff_indptr_cpu  = None
            self._Kff_cols_cpu    = None
            self._elem_sorted_cpu = None
            self._KE0_sorted_cpu  = None

        # ── Upload static arrays to GPU ────────────────────────────────────
        self._upload_static(edof, KE_UNIT, free, F, skip_sparse=skip_sparse_assembly)

    # ------------------------------------------------------------------
    # One-time sorted-CSR precomputation (CPU + optional GPU sort)
    # ------------------------------------------------------------------

    def _precompute_sorted_csr(
        self,
        keep: np.ndarray,
        kff_rows_local: np.ndarray,
        kff_cols_local: np.ndarray,
        ke_sq: int,
        n_kept: int,
    ) -> None:
        """
        Build sorted CSR descriptor for Kff once at init.

        Stores on self:
          _Kff_indptr_cpu  : (n_free+1,) int32 — CSR row pointers
          _Kff_cols_cpu    : (n_kept,) int32   — CSR column indices (sorted)
          _elem_sorted_cpu : (n_kept,) int32   — element index per kept entry
          _KE0_sorted_cpu  : (n_kept,) float64 — unit KE value per kept entry
        """
        n_elem = self._n_elem

        # ── Element index for each kept entry ────────────────────────────
        # Counts of kept entries per element, then repeat element indices
        keep_counts = keep.reshape(n_elem, ke_sq).sum(axis=1)          # (n_elem,) int
        elem_of_kept = np.repeat(
            np.arange(n_elem, dtype=np.int32), keep_counts
        )                                                               # (n_kept,)

        # ── Unit KE value for each kept entry ────────────────────────────
        # ke_local in [0, ke_sq) cycles per element: tile, then mask
        ke_local_of_kept = np.tile(
            np.arange(ke_sq, dtype=np.int32), n_elem
        )[keep]                                                         # (n_kept,)
        KE0_of_kept = self._KE_flat[ke_local_of_kept]                  # (n_kept,) float64
        del ke_local_of_kept

        # ── Sort kept entries by (row, col) ──────────────────────────────
        # GPU sort (fast, a few seconds even for 300M entries).
        # Falls back to CPU lexsort if CuPy not available.
        sort_idx = self._sort_kept(kff_rows_local, kff_cols_local, n_kept)

        rows_sorted = kff_rows_local[sort_idx].astype(np.int32)
        cols_sorted = kff_cols_local[sort_idx].astype(np.int32)
        self._elem_sorted_cpu = elem_of_kept[sort_idx]
        self._KE0_sorted_cpu  = KE0_of_kept[sort_idx]
        self._Kff_cols_cpu    = cols_sorted
        del sort_idx, elem_of_kept, KE0_of_kept, keep_counts

        # ── Build CSR indptr via searchsorted on sorted row indices ───────
        self._Kff_indptr_cpu = np.searchsorted(
            rows_sorted.astype(np.int64),
            np.arange(self._n_free + 1, dtype=np.int64),
            side="left",
        ).astype(np.int32)
        del rows_sorted
        gc.collect()

    def _sort_kept(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        n_kept: int,
    ) -> np.ndarray:
        """
        Return an int array `sort_idx` such that (rows[sort_idx], cols[sort_idx])
        is sorted lexicographically.  Uses GPU sort if CuPy is available.
        """
        try:
            import cupy as cp
            rows_cp = cp.asarray(rows.astype(np.int64))
            cols_cp = cp.asarray(cols.astype(np.int64))
            key = rows_cp * int(self._n_free) + cols_cp
            idx = cp.argsort(key)
            result = cp.asnumpy(idx)
            del rows_cp, cols_cp, key, idx
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            return result
        except Exception:
            return np.lexsort((cols, rows))

    # ------------------------------------------------------------------
    # GPU upload (called once at construction)
    # ------------------------------------------------------------------

    def _upload_static(self, edof, KE_UNIT, free, F, skip_sparse: bool = False) -> None:
        """Upload static arrays to GPU. Only uploads for the resolved backend."""
        if self._backend == "cupy":
            self._upload_cupy(edof, KE_UNIT, free, F, skip_sparse=skip_sparse)
        elif self._backend == "torch_cuda":
            self._upload_torch(edof, KE_UNIT, free, F, skip_sparse=skip_sparse)

    def _upload_cupy(self, edof, KE_UNIT, free, F, skip_sparse: bool = False) -> None:
        import cupy as cp
        self._cp = cp
        self._edof_gpu         = cp.asarray(edof, dtype=cp.int32)
        self._KE_unit_gpu      = cp.asarray(KE_UNIT, dtype=cp.float64)
        self._F_free_gpu       = cp.asarray(self._F_free_cpu, dtype=cp.float64)
        self._F_full_gpu       = cp.asarray(self._F_cpu, dtype=cp.float64)
        self._free_gpu         = cp.asarray(free, dtype=cp.int32)
        if not skip_sparse:
            # Pre-sorted CSR descriptor — skip for matrix-free mode (no Kff needed)
            self._Kff_indptr_gpu   = cp.asarray(self._Kff_indptr_cpu,   dtype=cp.int32)
            self._Kff_indices_gpu  = cp.asarray(self._Kff_cols_cpu,     dtype=cp.int32)
            self._elem_sorted_gpu  = cp.asarray(self._elem_sorted_cpu,  dtype=cp.int32)
            self._KE0_sorted_gpu   = cp.asarray(self._KE0_sorted_cpu,   dtype=cp.float64)
        else:
            self._Kff_indptr_gpu  = None
            self._Kff_indices_gpu = None
            self._elem_sorted_gpu = None
            self._KE0_sorted_gpu  = None

    def _upload_torch(self, edof, KE_UNIT, free, F, skip_sparse: bool = False) -> None:
        import torch
        dev = torch.device("cuda")
        self._torch        = torch
        self._torch_device = dev
        self._edof_gpu          = torch.tensor(edof,              dtype=torch.long,    device=dev)
        self._KE_unit_gpu       = torch.tensor(KE_UNIT,           dtype=torch.float64, device=dev)
        self._F_free_gpu        = torch.tensor(self._F_free_cpu,  dtype=torch.float64, device=dev)
        self._F_full_gpu        = torch.tensor(self._F_cpu,       dtype=torch.float64, device=dev)
        self._free_gpu          = torch.tensor(free,              dtype=torch.long,    device=dev)
        if not skip_sparse:
            # Pre-sorted CSR descriptor — skip for matrix-free mode
            self._Kff_indptr_torch  = torch.tensor(self._Kff_indptr_cpu,  dtype=torch.int32,   device=dev)
            self._Kff_indices_torch = torch.tensor(self._Kff_cols_cpu,    dtype=torch.int32,   device=dev)
            self._elem_sorted_torch = torch.tensor(self._elem_sorted_cpu, dtype=torch.long,    device=dev)
            self._KE0_sorted_torch  = torch.tensor(self._KE0_sorted_cpu,  dtype=torch.float64, device=dev)
        else:
            self._Kff_indptr_torch  = None
            self._Kff_indices_torch = None
            self._elem_sorted_torch = None
            self._KE0_sorted_torch  = None

    def _ensure_torch_backend(self) -> None:
        """Lazily initialise Torch CUDA tensors (used when falling back from CuPy)."""
        if getattr(self, "_torch_device", None) is not None:
            return
        edof    = self._edof
        KE_UNIT = self._KE_flat.reshape(self._ke_size, self._ke_size)
        F       = self._F_cpu
        free    = self._free
        self._upload_torch(edof, KE_UNIT, free, F)

    # ------------------------------------------------------------------
    # Public solve interface
    # ------------------------------------------------------------------

    def solve(self, rho_phys: np.ndarray, penal: float) -> tuple[float, np.ndarray]:
        """Full FEA on GPU. Returns (compliance, dc_phys). Falls back transparently."""
        last_exc: Exception | None = None
        for bk in backend_fallback_order(self._backend):
            try:
                if bk == "cupy" and _cupy_available():
                    self._backend = "cupy"
                    return self._solve_cupy(rho_phys, penal)
                if bk == "torch_cuda" and _torch_cuda_available():
                    self._backend = "torch_cuda"
                    return self._solve_torch(rho_phys, penal)
                if bk == "cpu":
                    self._backend = "cpu"
                    return self._solve_cpu(rho_phys, penal)
            except Exception as exc:
                last_exc = exc
                print(f"[GPUFEMSolver] {bk} failed ({exc}), trying fallback…")

        if last_exc:
            print(f"[GPUFEMSolver] All GPU backends failed. Last: {last_exc}")
        self._backend = "cpu"
        return self._solve_cpu(rho_phys, penal)

    # ------------------------------------------------------------------
    # CuPy backend  — O(n_kept) gather, zero sort per iteration
    # ------------------------------------------------------------------

    def _solve_cupy(self, rho_phys: np.ndarray, penal: float) -> tuple[float, np.ndarray]:
        cp  = self._cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cpspla

        E0, Emin = self.E0, self.Emin
        rho_gpu = cp.asarray(rho_phys, dtype=cp.float64)

        # Element moduli
        E_e = Emin + (E0 - Emin) * rho_gpu ** penal            # (n_elem,)

        # Kff values: gather E_e per kept entry, multiply by unit KE value
        # Memory: one float64 array of n_kept — no broadcast intermediate
        Kff_data = E_e[self._elem_sorted_gpu] * self._KE0_sorted_gpu  # (n_kept,)

        # Build Kff directly from pre-sorted CSR descriptor — no cuSPARSE sort
        Kff = cpsp.csr_matrix(
            (Kff_data, self._Kff_indices_gpu, self._Kff_indptr_gpu),
            shape=(self._n_free, self._n_free),
        )

        # Jacobi preconditioner: M^{-1} = diag(Kff)^{-1}
        # Without this, unpreconditioned CG silently hits maxiter on
        # ill-conditioned 3D problems (n_free ≳ 200k) and returns a
        # partially-converged solution that the SIMP loop then uses to
        # drive the design — this was the root cause of the cupy/torch
        # compliance divergence at 216k. The torch backend already does
        # the same thing in _torch_cg below.
        diag = Kff.diagonal()
        diag = cp.where(cp.abs(diag) > 1e-12, diag, cp.ones_like(diag))
        M_inv = 1.0 / diag
        M_op = cpspla.LinearOperator(
            (self._n_free, self._n_free),
            matvec=lambda v: M_inv * v,
            dtype=cp.float64,
        )

        # Solve Kff U = F_free
        U_free_gpu, info = cpspla.cg(
            Kff, self._F_free_gpu,
            tol=self.cg_tol, maxiter=self.cg_maxiter, M=M_op,
        )
        if info < 0:
            raise RuntimeError(f"CuPy CG breakdown (info={info})")
        if info > 0:
            print(
                f"[GPUFEMSolver] WARNING: CuPy CG did not converge in "
                f"{self.cg_maxiter} iterations (n_free={self._n_free}); "
                f"residual may be loose"
            )

        compliance = float(cp.dot(self._F_free_gpu, U_free_gpu).get())

        # Reconstruct full U (fixed DOFs = 0)
        U_gpu = cp.zeros(self.ndof, dtype=cp.float64)
        U_gpu[self._free_gpu] = U_free_gpu

        # Sensitivity: dc_phys = -penal * (E0-Emin) * rho^(p-1) * ce
        Ue  = U_gpu[self._edof_gpu]               # (n_elem, ke_size)
        KUe = Ue @ self._KE_unit_gpu              # (n_elem, ke_size)
        ce  = (KUe * Ue).sum(axis=1)             # (n_elem,)
        dc_phys_gpu = -penal * (E0 - Emin) * rho_gpu ** (penal - 1) * ce

        return compliance, cp.asnumpy(dc_phys_gpu)

    # ------------------------------------------------------------------
    # PyTorch CUDA backend — similarly uses pre-sorted CSR
    # ------------------------------------------------------------------

    def _solve_torch(self, rho_phys: np.ndarray, penal: float) -> tuple[float, np.ndarray]:
        self._ensure_torch_backend()
        torch = self._torch
        dev   = self._torch_device
        E0, Emin = self.E0, self.Emin

        rho_gpu = torch.tensor(rho_phys, dtype=torch.float64, device=dev)
        E_e = Emin + (E0 - Emin) * rho_gpu ** penal

        # Kff values via gather
        Kff_data = E_e[self._elem_sorted_torch] * self._KE0_sorted_torch  # (n_kept,)

        # Build sparse CSR directly (no COO conversion → no sort cost)
        Kff = torch.sparse_csr_tensor(
            self._Kff_indptr_torch.to(torch.int64),
            self._Kff_indices_torch.to(torch.int64),
            Kff_data,
            size=(self._n_free, self._n_free),
            dtype=torch.float64,
            device=dev,
        )

        U_free = _torch_cg(Kff, self._F_free_gpu, tol=self.cg_tol, maxiter=self.cg_maxiter)

        compliance = float(torch.dot(self._F_free_gpu, U_free).cpu())

        U_gpu = torch.zeros(self.ndof, dtype=torch.float64, device=dev)
        U_gpu[self._free_gpu] = U_free

        Ue  = U_gpu[self._edof_gpu]
        KUe = Ue @ self._KE_unit_gpu
        ce  = (KUe * Ue).sum(dim=1)
        dc_phys_gpu = -penal * (E0 - Emin) * rho_gpu ** (penal - 1) * ce

        return compliance, dc_phys_gpu.cpu().numpy()

    # ------------------------------------------------------------------
    # CPU fallback
    # ------------------------------------------------------------------

    def _solve_cpu(self, rho_phys: np.ndarray, penal: float) -> tuple[float, np.ndarray]:
        E0, Emin = self.E0, self.Emin
        E_e     = Emin + (E0 - Emin) * rho_phys ** penal
        KE_vals = (E_e[:, None] * self._KE_flat[None, :]).ravel()

        K = csc_matrix(
            (KE_vals, (self._row_idx_cpu, self._col_idx_cpu)),
            shape=(self.ndof, self.ndof),
        )

        if self._cpu_solver is not None:
            U = self._cpu_solver.solve(K, self._F_cpu, self._free, E_e, self.ndof)
        else:
            U = np.zeros(self.ndof, dtype=np.float64)
            Kff = K[self._free][:, self._free]
            U[self._free] = spsolve(Kff, self._F_free_cpu)

        compliance = float(self._F_cpu @ U)

        Ue    = U[self._edof]
        KUe   = Ue @ self._KE_flat.reshape(self._ke_size, self._ke_size)
        ce    = np.einsum("ei,ei->e", KUe, Ue)
        dc    = -penal * (E0 - Emin) * rho_phys ** (penal - 1) * ce
        return compliance, dc

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """Reset cached factorizations (call when rmin changes)."""
        if self._cpu_solver is not None:
            self._cpu_solver.invalidate()

    def free_gpu_cache(self) -> None:
        """Release CuPy / Torch memory pool blocks back to CUDA allocator."""
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Standalone function interface
# ─────────────────────────────────────────────────────────────────────────────

def fea_compute_gpu(
    rho_phys: np.ndarray,
    penal: float,
    solver: GPUFEMSolver,
) -> tuple[float, np.ndarray]:
    """GPU FEA via a pre-built GPUFEMSolver. Returns (compliance, dc_phys)."""
    return solver.solve(rho_phys, penal)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch CG helper — works with CSR tensors directly
# ─────────────────────────────────────────────────────────────────────────────

def _torch_cg(
    A,              # torch CSR sparse tensor (n_free × n_free)
    b: "torch.Tensor",
    tol: float = 1e-5,
    maxiter: int = 1000,
) -> "torch.Tensor":
    """
    Conjugate gradient with Jacobi preconditioner.
    A must be a sparse CSR tensor — no COO conversion.
    Diagonal extraction is fully vectorised (O(nnz), no Python loops).
    """
    import torch

    def _mv(M, v):
        # Sparse CSR × dense vector: works in PyTorch ≥ 2.0
        return (M @ v.unsqueeze(1)).squeeze(1)

    # Fully vectorised Jacobi diagonal extraction from CSR
    crow = A.crow_indices()                       # (n+1,)
    ccol = A.col_indices()                        # (nnz,)
    vals = A.values()                             # (nnz,)
    n    = A.shape[0]
    counts       = crow[1:] - crow[:-1]           # entries per row
    row_of_entry = torch.repeat_interleave(
        torch.arange(n, device=b.device), counts  # O(nnz) GPU scatter
    )
    diag_mask = (ccol == row_of_entry)
    diag = torch.zeros(n, dtype=b.dtype, device=b.device)
    diag.index_add_(0, ccol[diag_mask], vals[diag_mask])
    diag  = torch.where(diag.abs() > 1e-12, diag, torch.ones_like(diag))
    M_inv = 1.0 / diag

    x  = torch.zeros_like(b)
    r  = b - _mv(A, x)
    z  = M_inv * r
    p  = z.clone()
    rz = torch.dot(r, z)

    for _ in range(maxiter):
        Ap    = _mv(A, p)
        alpha = rz / (torch.dot(p, Ap) + 1e-30)
        x     = x + alpha * p
        r     = r - alpha * Ap
        if torch.norm(r) < tol * torch.norm(b):
            break
        z      = M_inv * r
        rz_new = torch.dot(r, z)
        beta   = rz_new / (rz + 1e-30)
        p      = z + beta * p
        rz     = rz_new

    return x
