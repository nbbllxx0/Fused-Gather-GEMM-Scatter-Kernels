"""
simp_gpu.py
-----------
GPU-accelerated SIMP topology optimizer with neural surrogate sensitivity injection.

Adapted from TO3D simp_core.py with GPU-resident hot path:
  - K assembly on GPU
  - Density filter as GPU matvec
  - Heaviside projection on GPU
  - OC update (bisection outer loop on CPU, vectorized inner ops on GPU)
  - Surrogate predict_gpu() for zero CPU round-trip on surrogate iterations

Multi-agent loop (unchanged from TO3D):
  Each iteration a RouterAgent decides: FEM (full GPU KU=F) or surrogate (MLP predict).
  OC update applied regardless.  Surrogate trained online from FEM observations.

Core FEM building blocks imported from AutoSIMP's pub_simp_solver.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csc_matrix

# ── AutoSIMP building blocks ─────────────────────────────────────────────────
from .pub_simp_solver import (
    _build_density_filter,
    _checkerboard_3d,
    _grayness,
    _heaviside,
    _heaviside_deriv,
    _oc_update,
    KE_UNIT_3D,
    KE_UNIT_2D,
    _edof_table_3d,
    _edof_table_2d,
    _build_sparse_indices,
    StepState,
)

from .fem_gpu import GPUFEMSolver
from .fem_gpu import backend_fallback_order
from .surrogate_gpu import SensitivitySurrogateGPU


# ─────────────────────────────────────────────────────────────────────────────
# Data structures (identical to TO3D simp_core.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TO3DParams:
    """Topology optimizer parameters."""
    nelx: int = 24
    nely: int = 12
    nelz: int = 12
    volfrac: float = 0.3
    penal: float = 3.0
    rmin: float = 1.5
    move: float = 0.2
    max_iter: int = 100
    tol: float = 0.01
    compliance_tol: float = 5e-4
    compliance_window: int = 5
    min_iter: int = 30
    use_heaviside: bool = True
    beta_init: float = 1.0
    beta_max: float = 32.0
    eta: float = 0.5
    min_penal_for_best: float = 3.0
    max_gray_for_best: float = 0.25
    amg_ndof_threshold: int = 3000
    amg_rebuild_tol: float = 0.15


@dataclass
class IterResult:
    iteration: int
    compliance: float
    rho_new: np.ndarray
    rho_phys: np.ndarray
    dc_phys: np.ndarray
    change: float
    used_surrogate: bool
    surrogate_uncertainty: float
    elapsed_sec: float


# ─────────────────────────────────────────────────────────────────────────────
# GPU density filter + Heaviside helpers
# ─────────────────────────────────────────────────────────────────────────────

class GPUDensityFilter:
    """
    GPU-resident density filter.  Wraps scipy sparse H_mat with a GPU matvec.
    Falls back to CPU scipy if GPU not available.
    """

    def __init__(self, H_mat, backend: str = "auto") -> None:
        self._H_cpu = H_mat
        self._backend = backend
        self._H_gpu = None
        self._upload(H_mat, backend)

    def _upload(self, H_mat, backend: str) -> None:
        self._H_T_gpu = None   # transpose; lazily populated for CuPy OC step
        if backend in ("cupy", "auto"):
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cpsp
                self._H_gpu = cpsp.csr_matrix(H_mat)
                self._H_T_gpu = cpsp.csr_matrix(H_mat.T.tocsr())
                self._backend = "cupy"
                return
            except Exception:
                pass
        if backend in ("torch_cuda", "auto", "cupy"):
            try:
                import torch
                if torch.cuda.is_available():
                    # Convert scipy CSR to torch sparse CSR
                    H_csr = H_mat.tocsr()
                    indptr = torch.tensor(H_csr.indptr, dtype=torch.long, device="cuda")
                    indices = torch.tensor(H_csr.indices, dtype=torch.long, device="cuda")
                    data = torch.tensor(H_csr.data, dtype=torch.float64, device="cuda")
                    self._H_gpu = torch.sparse_csr_tensor(
                        indptr, indices, data,
                        size=H_csr.shape, dtype=torch.float64, device="cuda"
                    )
                    self._H_gpu_coo = self._H_gpu.to_sparse_coo()
                    self._backend = "torch_cuda"
                    return
            except Exception:
                pass
        self._backend = "cpu"

    def rebuild(self, H_mat) -> None:
        """Rebuild filter when rmin changes."""
        self._H_cpu = H_mat
        self._upload(H_mat, self._backend)

    def apply(self, rho: np.ndarray) -> np.ndarray:
        """Apply density filter H_mat @ rho. Returns numpy array."""
        for backend in backend_fallback_order(self._backend):
            try:
                if backend == "cupy" and self._H_gpu is not None and self._backend == "cupy":
                    import cupy as cp
                    rho_gpu = cp.asarray(rho, dtype=cp.float64)
                    return cp.asnumpy(self._H_gpu.dot(rho_gpu))
                if backend == "torch_cuda":
                    if self._backend != "torch_cuda" or self._H_gpu is None:
                        self._upload(self._H_cpu, "torch_cuda")
                    if self._backend == "torch_cuda" and self._H_gpu is not None:
                        import torch
                        rho_t = torch.tensor(rho, dtype=torch.float64, device="cuda")
                        H = getattr(self, "_H_gpu_coo", None)
                        if H is None:
                            H = self._H_gpu.to_sparse_coo()
                            self._H_gpu_coo = H
                        return torch.sparse.mm(H, rho_t.unsqueeze(1)).squeeze(1).cpu().numpy()
                if backend == "cpu":
                    return self._H_cpu.dot(rho)
            except Exception as exc:
                print(f"[GPUDensityFilter] {backend} backend failed ({exc}), trying fallback...")
                self._backend = backend

        return self._H_cpu.dot(rho)


# ─────────────────────────────────────────────────────────────────────────────
# OC step (CPU bisection, vectorized inner ops)
# ─────────────────────────────────────────────────────────────────────────────

def _oc_step_gpu(
    rho_design: np.ndarray,
    rho_f: np.ndarray,
    dc_phys: np.ndarray,
    beta: float,
    eta: float,
    volfrac: float,
    move: float,
    H_mat,
    n_elem: int,
    use_heaviside: bool,
) -> tuple[np.ndarray, float]:
    """
    OC update — identical logic to TO3D _oc_step.
    dc_phys arrives as numpy (downloaded from GPU if needed).
    """
    dh = _heaviside_deriv(rho_f, beta, eta) if (use_heaviside and beta > 1e-6) else np.ones_like(rho_f)
    dc_f = dc_phys * dh
    dc_design = H_mat.T.dot(dc_f)
    dv_design = H_mat.T.dot(dh)

    rho_new = _oc_update(
        rho_design, dc_design, dv_design,
        volfrac, move, n_elem,
        beta, eta, use_heaviside, H_mat,
    )
    change = float(np.max(np.abs(rho_new - rho_design)))
    return rho_new, change


# ─────────────────────────────────────────────────────────────────────────────
# GPU OC step — replaces CPU scipy bisection with GPU SpMV (cuPy only)
# ─────────────────────────────────────────────────────────────────────────────

def _oc_step_cupy(
    rho_design: np.ndarray,
    rho_f: np.ndarray,
    dc_phys: np.ndarray,
    beta: float,
    eta: float,
    volfrac: float,
    move: float,
    H_gpu,      # cupyx.scipy.sparse.csr_matrix — H_mat on GPU
    H_T_gpu,    # cupyx.scipy.sparse.csr_matrix — H_mat.T on GPU (pre-transposed)
    n_elem: int,
    use_heaviside: bool,
) -> tuple[np.ndarray, float]:
    """
    GPU-accelerated OC update for the CuPy backend.

    Replaces the 100× CPU scipy H_mat.dot calls in _oc_step_gpu / _oc_update
    with GPU SpMVs via cuPy.  At 216k elements this cuts OC time from ~544 ms
    to ~25 ms per SIMP iteration (≈22× speedup), reducing it from the dominant
    cost (71% of wall time) to a minor one.

    The bisection scalar comparison still requires one GPU→CPU float() per iter
    (~0.2 ms on Windows WDDM); 60 bisection iters × 0.2 ms = ~12 ms.
    The GPU SpMVs themselves cost ~0.05 ms each (4M nnz at 1 TB/s).

    All vectors stay on GPU throughout; only the final rn is downloaded once.
    """
    import cupy as cp

    # Upload inputs to GPU (each ~1.7 MB; all three in ~0.3 ms)
    rho_gpu = cp.asarray(rho_design, dtype=cp.float64)
    dc_gpu  = cp.asarray(dc_phys,   dtype=cp.float64)

    # Sensitivity and volume sensitivity in the design variable space
    if use_heaviside and beta > 1e-6:
        rho_f_gpu = cp.asarray(rho_f, dtype=cp.float64)
        # GPU Heaviside derivative: beta*(1 - tanh²(beta*(rho_f - eta))) / denom
        # denom = tanh(beta*eta) + tanh(beta*(1 - eta))  (scalar, computed once)
        import math
        denom = math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta))
        dh_gpu = cp.float64(beta) * (1.0 - cp.tanh(cp.float64(beta) * (rho_f_gpu - eta))**2) / denom
        dc_f_gpu = dc_gpu * dh_gpu
        dc_design_gpu = H_T_gpu.dot(dc_f_gpu)     # H^T @ (dc * dh)
        dv_design_gpu = H_T_gpu.dot(dh_gpu)       # H^T @ dh
    else:
        dc_design_gpu = H_T_gpu.dot(dc_gpu)
        dv_design_gpu = cp.ones(n_elem, dtype=cp.float64)

    dc_s = cp.minimum(dc_design_gpu, cp.float64(-1e-12))
    dv_s = cp.maximum(dv_design_gpu, cp.float64(1e-12))
    volfrac_n = volfrac * n_elem    # threshold for bisection: sum(rn_phys) vs volfrac_n

    l1, l2 = 0.0, 1e9
    rn_gpu = rho_gpu   # initialise; overwritten every iter

    for _ in range(100):
        lm   = 0.5 * (l1 + l2)
        rn_gpu = cp.clip(
            cp.clip(rho_gpu * cp.sqrt(-dc_s / (cp.float64(lm) * dv_s)),
                    rho_gpu - move, rho_gpu + move),
            1e-3, 1.0,
        )
        rn_f_gpu = H_gpu.dot(rn_gpu)   # GPU SpMV: ~0.05 ms
        if use_heaviside and beta > 1e-6:
            # GPU Heaviside: (tanh(beta*eta) + tanh(beta*(x - eta))) / denom
            rn_phys_sum = float(
                (cp.tanh(cp.float64(beta) * cp.float64(eta))
                 + cp.tanh(cp.float64(beta) * (rn_f_gpu - cp.float64(eta))))
                .sum()
            ) / denom        # scalar GPU→CPU sync (~0.2 ms on Windows WDDM)
        else:
            rn_phys_sum = float(rn_f_gpu.sum())

        if rn_phys_sum > volfrac_n:
            l1 = lm
        else:
            l2 = lm
        if l2 - l1 < 1e-9:
            break

    rn = cp.asnumpy(rn_gpu)
    change = float(np.max(np.abs(rn - rho_design)))
    return rn, change


# ─────────────────────────────────────────────────────────────────────────────
# Main GPU SIMP loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simp_surrogate_gpu(
    params: TO3DParams,
    fixed: np.ndarray,
    free: np.ndarray,
    F: np.ndarray,
    ndof: int,
    surrogate: Optional[SensitivitySurrogateGPU],
    router: Callable,
    device: str = "auto",
    param_controller: Optional[Callable] = None,
    passive_mask: Optional[np.ndarray] = None,
    verbose: bool = False,
    progress_callback: Optional[Callable] = None,
    solver_class=None,
    solver_kwargs: Optional[dict] = None,
) -> dict:
    """
    GPU-accelerated SIMP topology optimization with surrogate sensitivity injection.

    Same interface as TO3D run_simp_surrogate() with added `device` parameter.
    Returns same dict format for compatibility.

    Parameters
    ----------
    device : str
        "auto" (detect best GPU), "cuda", "cupy", "torch_cuda", or "cpu".
    solver_class : type, optional
        Solver class to instantiate.  Defaults to GPUFEMSolver (paper-1 baseline).
        Pass SolverV2 (from solver_v2.py) to enable Phase 1 improvements.
    solver_kwargs : dict, optional
        Extra keyword arguments forwarded to solver_class.__init__().
        E.g. for SolverV2: {'grid_dims': (nelx, nely, nelz), 'enable_gmg': True}.
    """
    t_start = time.time()

    nelx, nely, nelz = params.nelx, params.nely, params.nelz
    is_3d = nelz > 0
    n_elem = nelx * nely * (nelz if is_3d else 1)
    E0, Emin = 1.0, 1e-9

    if is_3d:
        edof = _edof_table_3d(nelx, nely, nelz)
        KE_UNIT = KE_UNIT_3D
    else:
        edof = _edof_table_2d(nelx, nely)
        KE_UNIT = KE_UNIT_2D

    H_mat = _build_density_filter(nelx, nely, params.rmin, nelz if is_3d else 0)

    # ── Build GPU FEM solver ─────────────────────────────────────────────
    _SolverClass = solver_class if solver_class is not None else GPUFEMSolver
    _extra_kwargs = solver_kwargs if solver_kwargs is not None else {}

    # Skip the O(nnz_K) sparse index build when matrix-free is requested.
    # At 1M elements nnz_K = 576M; building row_idx/col_idx wastes ~4.6 GB RAM
    # and triggers a GPU sort that alone OOMs the 24 GB card.
    _matrix_free = _extra_kwargs.get("enable_matrix_free", False)
    if _matrix_free:
        row_idx = np.empty(0, dtype=np.int32)
        col_idx = np.empty(0, dtype=np.int32)
    else:
        row_idx, col_idx = _build_sparse_indices(edof)

    gpu_solver = _SolverClass(
        edof=edof,
        row_idx=row_idx,
        col_idx=col_idx,
        KE_UNIT=KE_UNIT,
        free=free,
        F=F,
        ndof=ndof,
        E0=E0,
        Emin=Emin,
        backend=device,
        amg_ndof_threshold=params.amg_ndof_threshold,
        amg_rebuild_tol=params.amg_rebuild_tol,
        **_extra_kwargs,
    )

    # ── GPU density filter ────────────────────────────────────────────────
    gpu_filter = GPUDensityFilter(H_mat, backend=gpu_solver._backend)

    # ── Initialize density ────────────────────────────────────────────────
    rho = params.volfrac * np.ones(n_elem)
    if passive_mask is not None:
        rho[passive_mask == 1] = 1e-3
        rho[passive_mask == 2] = 1.0

    penal = params.penal
    rmin = params.rmin
    move = params.move
    beta = params.beta_init
    eta = params.eta
    prev_rmin = rmin

    best_rho = rho.copy()
    best_compliance = np.inf
    best_iteration = 0
    best_is_valid = False
    best_grayness = _grayness(rho)
    compliance_hist: list[float] = []
    params_log: list[dict] = []
    last_compliance = np.inf

    fem_calls = 0
    surrogate_calls = 0

    for iteration in range(1, params.max_iter + 1):
        _iter_t0 = time.time()
        # ── Rebuild filter + solver if rmin changed ─────────────────────
        if abs(rmin - prev_rmin) > 1e-6:
            H_mat = _build_density_filter(nelx, nely, rmin, nelz if is_3d else 0)
            gpu_filter.rebuild(H_mat)
            gpu_solver.invalidate()
            prev_rmin = rmin

        # ── GPU density filter + Heaviside ──────────────────────────────
        rho_f = gpu_filter.apply(rho)
        rho_phys = (
            _heaviside(rho_f, beta, eta)
            if (params.use_heaviside and beta > 1e-6)
            else rho_f.copy()
        )

        iter_ratio = iteration / params.max_iter
        use_fem = True

        # ── Route: GPU FEM or GPU surrogate? ────────────────────────────
        if surrogate is not None:
            unc = surrogate.uncertainty
            use_fem = router(
                iteration, unc, surrogate.is_ready, beta,
                rho=rho_phys, compliance=last_compliance,
            )

        if use_fem:
            compliance, dc_phys = gpu_solver.solve(rho_phys, penal)
            last_compliance = compliance
            fem_calls += 1

            if surrogate is not None:
                surrogate.update(rho_phys, rho_f, penal, iter_ratio, dc_phys)
        else:
            dc_pred, unc = surrogate.predict(rho_phys, rho_f, penal, iter_ratio)
            if dc_pred is None:
                # Surrogate not ready — fall back to FEM
                compliance, dc_phys = gpu_solver.solve(rho_phys, penal)
                last_compliance = compliance
                fem_calls += 1
                surrogate.update(rho_phys, rho_f, penal, iter_ratio, dc_phys)
            else:
                dc_phys = dc_pred
                compliance = last_compliance
                surrogate_calls += 1

        # ── OC update ────────────────────────────────────────────────────
        # Use GPU bisection (cuPy) when available — replaces the 100× CPU
        # scipy H_mat.dot calls with GPU SpMVs (~22× faster at 216k).
        if gpu_filter._backend == "cupy" and gpu_filter._H_T_gpu is not None:
            rho_new, change = _oc_step_cupy(
                rho, rho_f, dc_phys, beta, eta,
                params.volfrac, move,
                gpu_filter._H_gpu, gpu_filter._H_T_gpu,
                n_elem, params.use_heaviside,
            )
        else:
            rho_new, change = _oc_step_gpu(
                rho, rho_f, dc_phys, beta, eta,
                params.volfrac, move, H_mat, n_elem, params.use_heaviside,
            )

        if passive_mask is not None:
            rho_new[passive_mask == 1] = 1e-3
            rho_new[passive_mask == 2] = 1.0

        # ── Track best ───────────────────────────────────────────────────
        gray_now = _grayness(rho_phys)
        gate_ok = penal >= params.min_penal_for_best and gray_now < params.max_gray_for_best
        if compliance < best_compliance and gate_ok:
            best_compliance = compliance
            best_rho = rho_phys.copy()
            best_iteration = iteration
            best_grayness = gray_now
            best_is_valid = True

        compliance_hist.append(compliance)

        # ── Convergence ──────────────────────────────────────────────────
        win = compliance_hist[-params.compliance_window:]
        conv_rho = change < params.tol
        conv_c = (
            len(win) == params.compliance_window
            and abs(win[-1] - win[0]) / max(abs(win[0]), 1e-10) < params.compliance_tol
        )
        converged = conv_rho and conv_c and iteration >= params.min_iter

        # ── Beta schedule ────────────────────────────────────────────────
        if iteration % 20 == 0 and beta < params.beta_max:
            beta = min(beta * 2.0, params.beta_max)

        # ── Optional parameter controller (AutoSIMP-compatible) ──────────
        if param_controller is not None:
            try:
                rel1 = 0.0 if len(compliance_hist) < 2 else (
                    (compliance_hist[-1] - compliance_hist[-2])
                    / max(abs(compliance_hist[-2]), 1e-10)
                )
                checker = _checkerboard_3d(rho_phys, nelx, nely, nelz) if is_3d else 0.0
                state = StepState(
                    iteration=iteration, compliance=compliance,
                    best_compliance=best_compliance, best_iteration=best_iteration,
                    volume_fraction=float(rho_phys.sum() / n_elem),
                    grayness=gray_now, best_grayness=best_grayness,
                    checkerboard=checker, obj_slope=0.0,
                    rel_change_1=rel1, rel_change_5=0.0,
                    stagnation_counter=0,
                    penal=penal, rmin=rmin, move=move, beta=beta,
                    converged=converged, best_is_valid=best_is_valid,
                    compliance_history=list(compliance_hist),
                )
                action = param_controller(state, rho_new)
                if action:
                    if "penal" in action:
                        penal = float(np.clip(action["penal"], 1.0, 5.0))
                    if "rmin" in action:
                        rmin = float(np.clip(action["rmin"], 1.1, 4.0))
                    if "move" in action:
                        move = float(np.clip(action["move"], 0.03, 0.4))
                    if "beta" in action:
                        beta = float(np.clip(action["beta"], 1.0, 64.0))
                    if action.get("restart", False) and best_is_valid:
                        rho_new = best_rho.copy()
                        gpu_solver.invalidate()
            except Exception:
                pass

        params_log.append({
            "iter": iteration, "penal": penal, "rmin": rmin, "move": move, "beta": beta,
            "compliance": compliance, "grayness": gray_now,
            "used_surrogate": not use_fem,
            "surrogate_uncertainty": surrogate.uncertainty if surrogate else 0.0,
            "cg_iters": int(getattr(gpu_solver, "last_cg_iters", -1)),
            "change": float(change),
            "volume": float(rho_phys.sum() / n_elem),
            "iter_wall_sec": time.time() - _iter_t0,
        })

        if verbose:
            src = "SUR" if (not use_fem and surrogate is not None) else "GPU"
            print(
                f"  [{src}] iter {iteration:3d}  C={compliance:8.4f}"
                f"  gray={gray_now:.3f}  p={penal:.2f}  β={beta:.1f}"
                f"  chg={change:.4f}"
                + (f"  unc={surrogate.uncertainty:.3f}" if surrogate else "")
            )

        if progress_callback is not None:
            progress_callback({
                "iteration": iteration, "max_iter": params.max_iter,
                "compliance": compliance, "grayness": gray_now,
                "used_surrogate": not use_fem,
                "uncertainty": surrogate.uncertainty if surrogate else 0.0,
                "beta": beta, "change": change,
                "penal": penal, "rmin": rmin, "move": move,
                "cg_iters": getattr(gpu_solver, "last_cg_iters", -1),
            })

        rho = rho_new

        if converged:
            if verbose:
                print(f"  Converged at iteration {iteration}.")
            break

    rho_final = best_rho if best_is_valid else rho_phys
    reported_final = compliance_hist[-1] if compliance_hist else np.inf
    if best_is_valid and reported_final > best_compliance * 2.0:
        reported_final = best_compliance

    total_calls = fem_calls + surrogate_calls
    return {
        "nelx": nelx, "nely": nely, "nelz": nelz,
        "n_iter": iteration,
        "final_compliance": reported_final,
        "best_compliance": best_compliance,
        "best_iteration": best_iteration,
        "final_grayness": _grayness(rho_final),
        "best_grayness": best_grayness,
        "best_is_valid": best_is_valid,
        "rho_final": rho_final,
        "rho_best": best_rho,
        "compliance_history": compliance_hist,
        "fem_calls": fem_calls,
        "surrogate_calls": surrogate_calls,
        "surrogate_used_fraction": surrogate_calls / max(total_calls, 1),
        "is_3d": is_3d,
        "params_log": params_log,
        "wall_time_sec": time.time() - t_start,
        "gpu_backend": gpu_solver._backend,
    }
