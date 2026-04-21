"""
fully_converged_study.py
------------------------
Run SIMP-120 with a high CG maxiter cap (5000) to obtain fully-converged
linear solves and verify that the selected-iterate compliance reporting
in Table 3 does not mask convergence issues.

The canonical paper runs cap CG at 1000 iterations; at 2M/4.9M elements
the majority of iterations hit this cap.  This study checks that for
smaller sizes (where the cap is not typically hit) the selected compliances
match a truly-converged run to 閳?0.1%.

Sizes: 216k, 512k (where cap is rarely hit)
Paths: fp64, fused

Usage
-----
    python experiments/phase3/fully_converged_study.py
    python experiments/phase3/fully_converged_study.py --sizes 216k --maxiter 10000
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
    "216k": (120, 60,  30),
    "512k": (160, 80,  40),
    "1M":   (200, 100, 50),
}

CANONICAL_MAXITER = 1000   # cap used in paper
HIGH_MAXITER      = 5000   # "fully converged" cap for this study


def _free_gpu_memory():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


def _vram_gb():
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        return (total - free) / 1024**3
    except Exception:
        return float("nan")


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


def run_simp_with_maxiter(size_tag, path, n_simp_iters, cg_maxiter, verbose=False):
    """Run SIMP-n_simp_iters with given CG maxiter cap."""
    from gpu_fem.solver_v2 import SolverV2
    from gpu_fem.simp_gpu import run_simp_surrogate_gpu, TO3DParams
    from gpu_fem.local_agents import PureFEMRouter
    from gpu_fem.pub_baseline_controller import ScheduleOnlyController

    nelx, nely, nelz = SIZE_LADDER[size_tag]
    prob = build_cantilever_problem(nelx, nely, nelz)

    params = TO3DParams(
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=prob["spec"].volfrac,
        rmin=1.5,
        max_iter=n_simp_iters,
    )
    solver_kwargs = dict(
        grid_dims=(nelx, nely, nelz),
        enable_warm_start=True,
        enable_matrix_free=True,
        enable_rediscr_gmg=False,
        cg_maxiter=cg_maxiter,
    )
    if path == "fp64":
        solver_kwargs.update(enable_mixed_precision=False, enable_fused_cuda=False)
    elif path == "fused":
        solver_kwargs.update(enable_mixed_precision=True, enable_fused_cuda=True)
    else:
        raise ValueError(f"Unknown path {path}")

    t0 = time.perf_counter()
    result = run_simp_surrogate_gpu(
        params=params,
        fixed=prob["fixed"], free=prob["free"],
        F=prob["F"], ndof=prob["ndof"],
        surrogate=None,
        router=PureFEMRouter(),
        device="auto",
        param_controller=ScheduleOnlyController(),
        verbose=verbose,
        solver_class=SolverV2,
        solver_kwargs=solver_kwargs,
    )
    wall_s = time.perf_counter() - t0
    c_sel  = result.get("best_compliance", result.get("final_compliance", float("nan")))
    vram   = _vram_gb()
    _free_gpu_memory()
    return float(wall_s), float(c_sel), vram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",   default="216k,512k")
    parser.add_argument("--paths",   default="fp64,fused")
    parser.add_argument("--maxiter", type=int, default=HIGH_MAXITER)
    parser.add_argument("--niters",  type=int, default=120)
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    paths = [p.strip() for p in args.paths.split(",")]

    print("=" * 70)
    print(f"  Fully-converged study 閳?SIMP-{args.niters}")
    print(f"  Canonical cap={CANONICAL_MAXITER}  High cap={args.maxiter}")
    print(f"  sizes={sizes}  paths={paths}")
    print("=" * 70)

    all_rows = []

    for sz in sizes:
        if sz not in SIZE_LADDER:
            print(f"  WARN: unknown size {sz}, skipping.")
            continue
        for path in paths:
            print(f"\n  [{path}] {sz} - canonical cap={CANONICAL_MAXITER} ...")
            wall_can, c_can, _ = run_simp_with_maxiter(
                sz, path, args.niters, CANONICAL_MAXITER
            )
            print(f"    canonical: wall={wall_can:.2f}s  c={c_can:.6f}")

            print(f"  [{path}] {sz} - high cap={args.maxiter} ...")
            wall_hi, c_hi, _ = run_simp_with_maxiter(
                sz, path, args.niters, args.maxiter
            )
            print(f"    high cap: wall={wall_hi:.2f}s  c={c_hi:.6f}")

            rel_diff = abs(c_can - c_hi) / max(abs(c_hi), 1e-30)
            print(f"    |铻朿|/c_hi = {rel_diff:.2e}")

            all_rows.append(dict(
                size=sz, path=path, n_simp_iters=args.niters,
                canonical_maxiter=CANONICAL_MAXITER,
                high_maxiter=args.maxiter,
                wall_canonical_s=wall_can, compliance_canonical=c_can,
                wall_high_s=wall_hi,       compliance_high=c_hi,
                rel_diff=rel_diff,
                agreement=(rel_diff < 1e-3),
            ))

    # 閳光偓閳光偓 Summary 閳光偓閳光偓
    print(f"\n{'='*90}")
    print("  FULLY-CONVERGED vs CANONICAL SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Size':>5}  {'Path':>6}  {'c_canon':>9}  {'c_high':>9}  "
          f"{'|铻朿|/c':>9}  {'OK?':>6}")
    print(f"  {'閳光偓'*60}")
    for r in all_rows:
        ok = "YES" if r["agreement"] else "NO"
        print(f"  {r['size']:>5}  {r['path']:>6}  "
              f"{r['compliance_canonical']:>9.5f}  {r['compliance_high']:>9.5f}  "
              f"{r['rel_diff']:>9.2e}  {ok:>6}")

    # 閳光偓閳光偓 Save 閳光偓閳光偓
    out_dir   = ROOT / "experiments" / "phase3"
    csv_path  = out_dir / "fully_converged_study.csv"
    json_path = out_dir / "fully_converged_study.json"

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
