"""
bf16_extended_study.py
----------------------
Extended BF16-IR convergence study across:
  - Multiple SIMP states (uniform density, selected mid-SIMP, selected late-SIMP)
  - Multiple penalization exponents (p = 3, 4.5, 5)
  - Multiple sizes (64k, 216k, 512k, 1M)
  - Multiple volume fractions (V_f = 0.3, 0.5)

The canonical Table 6 only covers uniform density (锜?0.5, p=3) at 64k and
216k. This study extends the evidence that the BF16 stall is not an artefact
of the uniform-density initialization but persists through the optimization
trajectory 閳?consistent with 榄?K) growing as penalization increases.

Usage
-----
    python experiments/phase3/bf16_extended_study.py
    python experiments/phase3/bf16_extended_study.py --sizes 64k,216k --quick
    python experiments/phase3/bf16_extended_study.py --sizes 64k,216k,512k,1M
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

SIZE_LADDER = {
    "64k":  (80,  40,  20),
    "216k": (120, 60,  30),
    "512k": (160, 80,  40),
    "1M":   (200, 100, 50),
}

EPS_BF16 = 2**-8


def _free_gpu_memory():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


def build_cantilever_problem(nelx, nely, nelz, volfrac=0.3, rmin=1.5):
    from gpu_fem.problem_spec import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator import generate_bc

    spec = ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=volfrac,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
        rmin=rmin,
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


def get_simp_rho_at_iter(prob, target_iter):
    """Fast SIMP run using fused solver to obtain density field at target_iter."""
    from gpu_fem.solver_v2 import SolverV2
    from gpu_fem.simp_gpu import run_simp_surrogate_gpu, TO3DParams
    from gpu_fem.local_agents import PureFEMRouter
    from gpu_fem.pub_baseline_controller import ScheduleOnlyController

    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    params = TO3DParams(
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=prob["spec"].volfrac,
        rmin=prob["spec"].rmin,
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
    rho = result.get("best_rho", result.get("rho"))
    if rho is None:
        raise RuntimeError("SIMP result missing density field")
    return np.asarray(rho, dtype=np.float64)


def run_bf16_probe(prob, rho, penal, fused_dtype, inner_tol=1e-3, max_outer=8):
    """Single cold-start linear solve with given fused_dtype and density."""
    from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
    from gpu_fem.solver_v2 import SolverV2

    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    edof = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    solver = SolverV2(
        edof=edof, row_idx=row_idx, col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D, free=prob["free"], F=prob["F"], ndof=prob["ndof"],
        backend="auto",
        grid_dims=(nelx, nely, nelz),
        enable_warm_start=False,
        enable_matrix_free=True,
        enable_mixed_precision=True,
        enable_fused_cuda=True,
        fused_dtype=fused_dtype,
        bf16_ir_inner_tol=inner_tol,
        bf16_ir_max_outer=max_outer,
    )
    t0 = time.perf_counter()
    c, _ = solver.solve(rho, penal)
    dt = time.perf_counter() - t0
    cg_iters = solver.last_cg_iters
    del solver
    _free_gpu_memory()
    return float(c), int(cg_iters), float(dt)


def run_extended_study(size_tag, penals, volfrac, simp_iters_for_state,
                       quick=False):
    """Run extended BF16 study for one size."""
    nelx, nely, nelz = SIZE_LADDER[size_tag]
    prob = build_cantilever_problem(nelx, nely, nelz, volfrac=volfrac)
    n_elem = prob["n_elem"]
    rows = []

    # FP32 reference at each state
    rho_states = {}
    rho_states["uniform"] = np.full(n_elem, 0.5)

    if not quick:
        try:
            print("    Building selected mid-SIMP state (best-valid snapshot near iter 40) ...")
            rho_mid = get_simp_rho_at_iter(prob, 40)
            rho_states["selected_iter40"] = rho_mid
            _free_gpu_memory()
        except Exception as e:
            print(f"    WARNING: selected mid-SIMP failed: {e}")

        try:
            print("    Building selected late-SIMP state (best-valid snapshot near iter 100) ...")
            rho_late = get_simp_rho_at_iter(prob, 100)
            rho_states["selected_iter100"] = rho_late
            _free_gpu_memory()
        except Exception as e:
            print(f"    WARNING: selected late-SIMP failed: {e}")

    for state_name, rho in rho_states.items():
        for penal in penals:
            rho_mean = float(np.mean(rho))
            print(f"    [{size_tag} vf={volfrac} state={state_name} p={penal}]")

            # FP32 reference
            c_fp32, cg_fp32, t_fp32 = run_bf16_probe(prob, rho, penal, "fp32")
            print(f"      fp32:    c={c_fp32:.6f}  CG={cg_fp32}  t={t_fp32*1e3:.1f}ms")

            # BF16 raw
            c_bf16, cg_bf16, t_bf16 = run_bf16_probe(prob, rho, penal, "bf16")
            err_bf16 = abs(c_bf16 - c_fp32) / max(abs(c_fp32), 1e-30)
            print(f"      bf16:    c={c_bf16:.6f}  CG={cg_bf16}  铻朿={err_bf16:.3f}  t={t_bf16*1e3:.1f}ms")

            # BF16-IR
            c_ir, cg_ir, t_ir = run_bf16_probe(prob, rho, penal, "bf16_ir",
                                                inner_tol=1e-3, max_outer=8)
            err_ir = abs(c_ir - c_fp32) / max(abs(c_fp32), 1e-30)
            print(f"      bf16_ir: c={c_ir:.6f}  CG={cg_ir}  铻朿={err_ir:.3f}  t={t_ir*1e3:.1f}ms")

            stalls_bf16 = err_bf16 > 0.05
            stalls_ir   = err_ir   > 0.05

            rows.append(dict(
                size=size_tag, n_elem=n_elem,
                volfrac=volfrac, state=state_name, penal=penal,
                rho_mean=rho_mean,
                c_fp32=c_fp32,   cg_fp32=cg_fp32,   t_fp32_ms=t_fp32 * 1e3,
                c_bf16=c_bf16,   cg_bf16=cg_bf16,   err_bf16=err_bf16,
                c_bf16_ir=c_ir,  cg_bf16_ir=cg_ir,  err_bf16_ir=err_ir,
                bf16_stalls=stalls_bf16,
                bf16_ir_stalls=stalls_ir,
                eps_bf16_kappa_estimate=(EPS_BF16 * err_bf16 / 0.5
                                         if err_bf16 > 0 else float("nan")),
            ))

    del prob
    _free_gpu_memory()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",   default="64k,216k,512k")
    parser.add_argument("--penals",  default="3.0,4.5")
    parser.add_argument("--volfrac", type=float, default=0.3)
    parser.add_argument("--quick",   action="store_true",
                        help="Only test uniform-density state (no SIMP runs)")
    args = parser.parse_args()

    sizes  = [s.strip() for s in args.sizes.split(",")]
    penals = [float(p) for p in args.penals.split(",")]

    print("=" * 70)
    print(f"  Extended BF16-IR study")
    print(f"  sizes={sizes}  penals={penals}  volfrac={args.volfrac}  quick={args.quick}")
    print("=" * 70)

    all_rows = []
    for sz in sizes:
        if sz not in SIZE_LADDER:
            print(f"  WARN: unknown size {sz}, skipping.")
            continue
        print(f"\n  {'='*60}")
        print(f"  {sz}  volfrac={args.volfrac}")
        print(f"  {'='*60}")
        rows = run_extended_study(
            sz, penals, args.volfrac,
            simp_iters_for_state=100,
            quick=args.quick,
        )
        all_rows.extend(rows)

    # 閳光偓閳光偓 Summary table 閳光偓閳光偓
    print(f"\n{'='*110}")
    print("  SUMMARY: BF16 stall by size/state/penalization")
    print(f"{'='*110}")
    print(f"  {'Size':>5}  {'State':>15}  {'p':>4}  {'c_fp32':>8}  "
          f"{'铻朿_bf16':>9}  {'铻朿_ir':>9}  {'BF16?':>7}  {'IR?':>7}")
    print(f"  {'閳光偓'*90}")
    for r in all_rows:
        b16 = "STALLS" if r["bf16_stalls"]    else "OK"
        ir  = "STALLS" if r["bf16_ir_stalls"] else "OK"
        print(f"  {r['size']:>5}  {r['state']:>15}  {r['penal']:>4.1f}  "
              f"{r['c_fp32']:>8.4f}  {r['err_bf16']:>9.4f}  {r['err_bf16_ir']:>9.4f}  "
              f"{b16:>7}  {ir:>7}")

    # 閳光偓閳光偓 Save 閳光偓閳光偓
    out_dir   = ROOT / "experiments" / "phase3"
    csv_path  = out_dir / "bf16_extended_study.csv"
    json_path = out_dir / "bf16_extended_study.json"

    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        with open(json_path, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\n  Saved: {csv_path}")
        print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
