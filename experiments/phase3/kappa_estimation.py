"""
kappa_estimation.py
-------------------
Estimate the condition number kappa(K) of the SIMP stiffness matrix for the
canonical condition-number bundle.

Canonical rows:
  - uniform-density initialization rho = 0.5
  - penalization levels p in {3.0, 5.0}
  - mesh sizes 64k, 216k, and 512k

Optional exploratory rows can add mid- and late-continuation SIMP states for
appendix-only checks or debugging.

The BF16 comparison uses the unit roundoff epsilon_BF16 = 2^-8 and the paper's
BF16 stall criterion kappa(K) >= 1 / epsilon_BF16.

Method
------
Power iteration / inverse iteration using the matrix-free matvec:
  lambda_max via power iteration
  lambda_min via inverse iteration using CG
  kappa(K) = lambda_max / lambda_min

No explicit K assembly is required; the same MatrixFreeKff operator used by the
solver is reused here.

Usage
-----
    python experiments/phase3/kappa_estimation.py
    python experiments/phase3/kappa_estimation.py --include-mid
    python experiments/phase3/kappa_estimation.py --include-late
    python experiments/phase3/kappa_estimation.py --sizes 64k,216k,512k
"""
from __future__ import annotations
import argparse
import csv
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _prefer_pytorch_env() -> None:
    root        = Path(__file__).resolve().parents[2]
    runtime_tmp = root / ".runtime_tmp"
    cupy_cache  = root / ".cupy_cache"
    runtime_tmp.mkdir(exist_ok=True)
    cupy_cache.mkdir(exist_ok=True)
    os.environ["TMP"]            = str(runtime_tmp)
    os.environ["TEMP"]           = str(runtime_tmp)
    os.environ["CUPY_CACHE_DIR"] = str(cupy_cache)
    os.environ["CUPY_TEMPDIR"]   = str(runtime_tmp)
    tempfile.tempdir             = str(runtime_tmp)

    preferred_python = os.environ.get("GPU_FEM_PYTHON")
    if not preferred_python:
        return
    env_python = Path(preferred_python).expanduser()
    if os.environ.get("GPU_FEM_ENV_BOOTSTRAPPED") == "1":
        return
    if not env_python.exists():
        return
    env = os.environ.copy()
    env["GPU_FEM_ENV_BOOTSTRAPPED"] = "1"
    os.execve(str(env_python), [str(env_python), __file__, *sys.argv[1:]], env)


_prefer_pytorch_env()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "phase3"))

import numpy as np

EPS_BF16 = 2**-8          # BF16 unit roundoff used in the paper
STALL_THRESHOLD = 1.0 / EPS_BF16   # kappa must stay below this for BF16-IR to converge

SIZE_LADDER = {
    "64k":  (80,  40,  20),
    "216k": (120, 60,  30),
    "512k": (160, 80,  40),
}


def _free_gpu_memory():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


def build_cantilever_problem(nelx, nely, nelz):
    from gpu_fem.problem_spec import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator import generate_bc

    spec = ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
        rmin=1.5,
    )
    bc = generate_bc(spec)
    return dict(
        spec=spec,
        nelx=nelx, nely=nely, nelz=nelz,
        n_elem=nelx * nely * nelz,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=bc.F, ndof=bc.ndof,
    )


def build_matfree_op(prob):
    """Build matrix-free Kff operator (FP64 by default)."""
    import cupy as cp
    from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
    from gpu_fem.solver_v2 import MatrixFreeKff

    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    edof = _edof_table_3d(nelx, nely, nelz)

    edof_gpu    = cp.asarray(edof)
    KE_unit_gpu = cp.asarray(KE_UNIT_3D)
    free_gpu    = cp.asarray(prob["free"])

    mf = MatrixFreeKff(
        edof_gpu=edof_gpu,
        KE_unit_gpu=KE_unit_gpu,
        free_gpu=free_gpu,
        n_free=len(prob["free"]),
        ndof=prob["ndof"],
    )
    return mf, edof


def compute_E_e(rho, penal, E0=1.0, Emin=1e-9):
    """Compute per-element Young's moduli from SIMP interpolation."""
    return Emin + (1.0 - Emin) * rho**penal


def power_iteration_largest(mf_op, E_e_gpu, n_free, n_iters=50, tol=1e-6):
    """
    Power iteration: estimate 閳ユ湗閳ユ牑鍊?= 浣峗max(K).

    Returns (lambda_max, n_iters_used, converged)
    """
    import cupy as cp

    # Random unit start vector
    rng = cp.random.default_rng(seed=42)
    v = rng.standard_normal(n_free, dtype=cp.float64)
    v /= cp.linalg.norm(v)

    lam_prev = 0.0
    for i in range(n_iters):
        w = mf_op.matvec(v, E_e_gpu)
        lam = float(cp.dot(v, w))
        v = w / cp.linalg.norm(w)
        if i > 0 and abs(lam - lam_prev) / max(abs(lam_prev), 1e-30) < tol:
            return lam, i + 1, True
        lam_prev = lam
    return lam_prev, n_iters, False


def inverse_iteration_smallest(mf_op, E_e_gpu, n_free, n_iters=30,
                                cg_tol=1e-8, cg_maxiter=2000):
    """
    Inverse iteration: estimate 閳ユ湗閳﹀鍏夆偓鏍ゅ€?= 1/浣峗min(K).

    Uses CG for the inner solve K璺痺 = v at each step.
    Returns (lambda_min, n_iters_used, converged)
    """
    import cupy as cp
    import scipy.sparse.linalg as spla

    # Initial unit vector
    rng = cp.random.default_rng(seed=123)
    v = rng.standard_normal(n_free, dtype=cp.float64)
    v /= cp.linalg.norm(v)

    lam_inv_prev = 0.0
    for i in range(n_iters):
        # Solve K璺痺 = v via CG
        def matvec(x_flat):
            x_gpu = cp.asarray(x_flat, dtype=cp.float64)
            y_gpu = mf_op.matvec(x_gpu, E_e_gpu)
            return cp.asnumpy(y_gpu)

        n_f = n_free
        LinOp = spla.LinearOperator((n_f, n_f), matvec=matvec, dtype=np.float64)

        v_np = cp.asnumpy(v)
        # SciPy 閳?.12 renamed tol閳姰tol; support both versions
        try:
            w_np, info = spla.cg(LinOp, v_np, rtol=cg_tol, maxiter=cg_maxiter,
                                  atol=cg_tol)
        except TypeError:
            w_np, info = spla.cg(LinOp, v_np, tol=cg_tol, maxiter=cg_maxiter,
                                  atol=cg_tol)
        w = cp.asarray(w_np)

        norm_w = float(cp.linalg.norm(w))
        if norm_w < 1e-30:
            break
        lam_inv = norm_w     # Rayleigh quotient ~ 閳ユ湗^{-1}v閳?/ 閳ユ潳閳?at convergence
        v = w / norm_w

        if i > 0 and abs(lam_inv - lam_inv_prev) / max(lam_inv_prev, 1e-30) < 1e-4:
            lam_min = 1.0 / lam_inv if lam_inv > 0 else float("nan")
            return lam_min, i + 1, True
        lam_inv_prev = lam_inv

    lam_min = 1.0 / lam_inv_prev if lam_inv_prev > 0 else float("nan")
    return lam_min, n_iters, False


def estimate_kappa(mf_op, E_e, n_free, power_iters=50, inv_iters=20):
    """Estimate 榄?K) = 浣峗max / 浣峗min."""
    import cupy as cp

    E_e_gpu = cp.asarray(E_e, dtype=cp.float64)
    t0 = time.perf_counter()
    lam_max, pi_iters, pi_conv = power_iteration_largest(
        mf_op, E_e_gpu, n_free, n_iters=power_iters
    )
    lam_min, ii_iters, ii_conv = inverse_iteration_smallest(
        mf_op, E_e_gpu, n_free, n_iters=inv_iters
    )
    elapsed = time.perf_counter() - t0

    kappa = lam_max / lam_min if lam_min > 0 else float("inf")
    bf16_product = EPS_BF16 * kappa
    stalls = bf16_product >= 1.0

    return {
        "lam_max":   lam_max,
        "lam_min":   lam_min,
        "kappa":     kappa,
        "eps_bf16_kappa": bf16_product,
        "bf16_stalls": stalls,
        "stall_threshold": STALL_THRESHOLD,
        "pi_iters":  pi_iters,
        "ii_iters":  ii_iters,
        "elapsed_s": elapsed,
    }


def get_simp_density_at_iter(prob, target_iter, schedule="standard"):
    """
    Run SIMP for up to target_iter steps and return (rho_best, penal_best).
    Uses the fused solver so it's fast even at 216k.
    """
    from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
    from gpu_fem.solver_v2 import SolverV2
    from gpu_fem.simp_gpu import run_simp_surrogate_gpu, TO3DParams
    from gpu_fem.local_agents import PureFEMRouter
    from gpu_fem.pub_baseline_controller import ScheduleOnlyController

    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    params = TO3DParams(
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=prob["spec"].volfrac,
        rmin=1.5,
        max_iter=target_iter,
    )
    solver_kwargs = dict(
        grid_dims=(nelx, nely, nelz),
        enable_warm_start=True,
        enable_matrix_free=True,
        enable_mixed_precision=True,
        enable_fused_cuda=True,
    )
    result = run_simp_surrogate_gpu(
        params=params,
        fixed=prob["fixed"], free=prob["free"],
        F=prob["F"], ndof=prob["ndof"],
        surrogate=None,
        router=PureFEMRouter(),
        device="auto",
        param_controller=ScheduleOnlyController(),
        verbose=False,
        solver_class=SolverV2,
        solver_kwargs=solver_kwargs,
    )
    # Prefer selected best iterate; fall back to final or last iterate.
    # Note: best_rho requires penal閳?.0 and grayness<0.25 閳?not met at iter<16.
    rho_best = (result.get("best_rho")
                or result.get("final_rho")
                or result.get("rho")
                or result.get("last_rho"))
    penal_best = result.get("best_penal", result.get("final_penal",
                  result.get("penal", 3.0)))
    if rho_best is None:
        # Last resort: reconstruct from final_x if available
        rho_best = result.get("final_x") or result.get("x")
    if rho_best is None:
        raise RuntimeError(
            f"SIMP result does not contain density field. "
            f"Available keys: {list(result.keys())}"
        )
    return np.asarray(rho_best, dtype=np.float64), float(penal_best)


def run_kappa_study(size_tag, power_iters=50, inv_iters=20, include_mid=False,
                    include_late=False):
    """
    Full kappa study for one size: canonical uniform rows, with optional
    exploratory mid- and late-SIMP states.
    """
    print(f"\n{'='*70}")
    print(f"  kappa(K) estimation - {size_tag} cantilever")
    print(f"{'='*70}")

    nelx, nely, nelz = SIZE_LADDER[size_tag]
    prob = build_cantilever_problem(nelx, nely, nelz)
    n_elem = prob["n_elem"]
    n_free = len(prob["free"])

    print(f"  n_elem={n_elem:,}  n_free={n_free:,}")
    print(f"  epsilon_BF16={EPS_BF16:.3e}  stall threshold kappa>{STALL_THRESHOLD:.0f}\n")

    mf_op, edof = build_matfree_op(prob)
    rows = []

    # State 1: uniform density rho=0.5, p=3 (the Table 6 reference state)
    for penal, label in [(3.0, "uniform_p3"), (5.0, "uniform_p5")]:
        rho = np.full(n_elem, 0.5)
        E_e = compute_E_e(rho, penal)
        print(f"  [{label}] p={penal}  rho=0.5 (uniform)")
        stats = estimate_kappa(mf_op, E_e, n_free,
                               power_iters=power_iters, inv_iters=inv_iters)
        row = dict(
            size=size_tag, n_elem=n_elem, state=label,
            rho_mean=0.5, penal=penal,
            **stats,
        )
        rows.append(row)
        _print_kappa_row(row)

    if include_mid:
        # State 2: exploratory mid-SIMP iterate (iter 40)
        print(f"\n  [mid_simp_iter40] Running SIMP to iter 40")
        try:
            rho_mid, penal_mid = get_simp_density_at_iter(prob, target_iter=40)
            E_e_mid = compute_E_e(rho_mid, penal_mid)
            stats_mid = estimate_kappa(mf_op, E_e_mid, n_free,
                                       power_iters=power_iters, inv_iters=inv_iters)
            row_mid = dict(
                size=size_tag, n_elem=n_elem, state="mid_simp_iter40",
                rho_mean=float(np.mean(rho_mid)), penal=penal_mid,
                **stats_mid,
            )
            rows.append(row_mid)
            _print_kappa_row(row_mid)
            _free_gpu_memory()
        except Exception as e:
            print(f"  WARNING: mid-SIMP run failed: {e}")

    if include_late:
        # State 3: exploratory late-SIMP iterate (iter 100)
        print(f"\n  [late_simp_iter100] Running SIMP to iter 100")
        try:
            rho_late, penal_late = get_simp_density_at_iter(prob, target_iter=100)
            E_e_late = compute_E_e(rho_late, penal_late)
            stats_late = estimate_kappa(mf_op, E_e_late, n_free,
                                        power_iters=power_iters, inv_iters=inv_iters)
            row_late = dict(
                size=size_tag, n_elem=n_elem, state="late_simp_iter100",
                rho_mean=float(np.mean(rho_late)), penal=penal_late,
                **stats_late,
            )
            rows.append(row_late)
            _print_kappa_row(row_late)
            _free_gpu_memory()
        except Exception as e:
            print(f"  WARNING: late-SIMP run failed: {e}")

    del mf_op
    _free_gpu_memory()
    return rows
def _print_kappa_row(row):
    stall_str = "STALLS" if row["bf16_stalls"] else "OK"
    print(
        f"    榄?= {row['kappa']:.3e}   "
        f"钄歘BF16璺瓘 = {row['eps_bf16_kappa']:.2f}   "
        f"[{stall_str}]   "
        f"浣峗max={row['lam_max']:.3e}   浣峗min={row['lam_min']:.3e}   "
        f"t={row['elapsed_s']:.1f}s"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",        default="64k,216k", help="Comma-separated size tags")
    parser.add_argument("--power-iters",  type=int, default=50)
    parser.add_argument("--inv-iters",    type=int, default=20)
    parser.add_argument("--include-mid",  action="store_true",
                        help="Include exploratory mid-SIMP state")
    parser.add_argument("--include-late", action="store_true",
                        help="Include exploratory late-SIMP state")
    parser.add_argument("--out",          default=None)
    args = parser.parse_args()
    sizes = [s.strip() for s in args.sizes.split(",")]
    all_rows = []
    for sz in sizes:
        if sz not in SIZE_LADDER:
            print(f"Unknown size {sz}, skipping.")
            continue
        rows = run_kappa_study(
            sz,
            power_iters=args.power_iters,
            inv_iters=args.inv_iters,
            include_mid=args.include_mid,
            include_late=args.include_late,
        )
        all_rows.extend(rows)

    # 閳光偓閳光偓 Summary table 閳光偓閳光偓
    print(f"\n{'='*90}")
    print("  SUMMARY: kappa(K) by size and SIMP state")
    print(f"{'='*90}")
    print(f"  {'Size':>5}  {'State':>25}  {'p':>4}  {'rho_mean':>8}  "
          f"{'kappa(K)':>10}  {'eps*k':>8}  {'BF16?':>7}")
    print(f"  {'-'*82}")
    for r in all_rows:
        stall = "STALLS" if r["bf16_stalls"] else "OK"
        print(
            f"  {r['size']:>5}  {r['state']:>25}  {r['penal']:>4.1f}  "
            f"{r['rho_mean']:>6.3f}  {r['kappa']:>10.3e}  "
            f"{r['eps_bf16_kappa']:>6.2f}  {stall:>7}"
        )

    # 閳光偓閳光偓 Save outputs 閳光偓閳光偓
    out_dir = ROOT / "experiments" / "phase3"
    csv_path  = out_dir / "kappa_estimation.csv"
    json_path = out_dir / "kappa_estimation.json"

    # Build JSON-safe copy
    def _to_json_safe(v):
        if isinstance(v, (np.floating, float)):
            if np.isnan(v) or np.isinf(v):
                return str(v)
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, bool):
            return bool(v)
        return v

    safe_rows = [{k: _to_json_safe(v) for k, v in r.items()} for r in all_rows]

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in safe_rows:
                writer.writerow(r)
        with open(json_path, "w") as f:
            json.dump(safe_rows, f, indent=2)
        print(f"\n  Saved: {csv_path}")
        print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
