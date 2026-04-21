"""
determinism_study.py
--------------------
Quantify non-determinism in the fused FP32 CG path caused by atomicAdd
floating-point reordering.

Runs N identical cold-start CG linear solves on the cantilever benchmark
and measures compliance variation.  Expected: 閳?1e-5 relative deviation
across all seeds of non-determinism (problem is deterministically seeded
with uniform rho=0.5 and zero warm-start, but atomic reordering can produce
run-to-run differences).

Published claim: "the fused operator introduces no visible additional error"
(鎼?.9, Correctness Verification).  This script provides the supporting data.

Usage
-----
    python experiments/phase3/determinism_study.py
    python experiments/phase3/determinism_study.py --sizes 64k,216k,512k --nreps 10
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


def make_fused_solver(prob):
    from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
    from gpu_fem.solver_v2 import SolverV2

    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    edof = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    return SolverV2(
        edof=edof, row_idx=row_idx, col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D, free=prob["free"], F=prob["F"], ndof=prob["ndof"],
        backend="auto", grid_dims=(prob["nelx"], prob["nely"], prob["nelz"]),
        enable_warm_start=False,          # cold start every time
        enable_matrix_free=True,
        enable_mixed_precision=True,
        enable_fused_cuda=True,
        fused_dtype="fp32",
    )


def make_fp64_solver(prob):
    from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
    from gpu_fem.solver_v2 import SolverV2

    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    edof = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    return SolverV2(
        edof=edof, row_idx=row_idx, col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D, free=prob["free"], F=prob["F"], ndof=prob["ndof"],
        backend="auto", grid_dims=(prob["nelx"], prob["nely"], prob["nelz"]),
        enable_warm_start=False,
        enable_matrix_free=True,
        enable_mixed_precision=False,
        enable_fused_cuda=False,
    )


def run_determinism(size_tag, nreps, penal=3.0):
    print(f"\n{'='*70}")
    print(f"  Determinism study 閳?{size_tag} cantilever  (fused FP32 vs FP64)")
    print(f"{'='*70}")

    nelx, nely, nelz = SIZE_LADDER[size_tag]
    prob = build_cantilever_problem(nelx, nely, nelz)
    n_elem = prob["n_elem"]
    rho = np.full(n_elem, 0.5)

    # FP64 reference (single run 閳?deterministic by construction)
    solver_fp64 = make_fp64_solver(prob)
    c_ref, _ = solver_fp64.solve(rho, penal)
    c_ref = float(c_ref)
    del solver_fp64
    _free_gpu_memory()
    print(f"  FP64 reference compliance: {c_ref:.8f}")

    # Fused FP32 閳?nreps identical cold-start runs
    solver_fused = make_fused_solver(prob)
    compliances_fused = []
    cg_iters_fused    = []
    times_fused       = []

    # Warmup (compile kernel)
    _ = solver_fused.solve(rho, penal)

    for rep in range(nreps):
        t0 = time.perf_counter()
        c, _ = solver_fused.solve(rho, penal)
        dt = time.perf_counter() - t0
        cit = solver_fused.last_cg_iters
        compliances_fused.append(float(c))
        cg_iters_fused.append(int(cit))
        times_fused.append(float(dt))
        rel_err = abs(float(c) - c_ref) / c_ref
        print(f"    rep {rep+1:2d}: c={float(c):.8f}  铻朿_rel={rel_err:.2e}  "
              f"CG={cit}  t={dt*1e3:.1f}ms")

    del solver_fused
    _free_gpu_memory()

    arr = np.array(compliances_fused)
    spread = float(np.max(arr) - np.min(arr))
    spread_rel = spread / c_ref
    rel_errs = np.abs(arr - c_ref) / c_ref
    max_rel_err = float(np.max(rel_errs))
    mean_rel_err = float(np.mean(rel_errs))

    print(f"\n  Spread across {nreps} fused runs: max-min = {spread:.2e}  "
          f"(relative {spread_rel:.2e})")
    print(f"  Max  |c_fused - c_fp64| / c_fp64 = {max_rel_err:.2e}")
    print(f"  Mean |c_fused - c_fp64| / c_fp64 = {mean_rel_err:.2e}")
    deterministic = spread_rel < 1e-4
    print(f"  Verdict: {'DETERMINISTIC (spread < 1e-4)' if deterministic else 'NON-DETERMINISTIC spread detected'}")

    return dict(
        size=size_tag, n_elem=n_elem, nreps=nreps,
        c_fp64_ref=c_ref,
        compliances_fused=compliances_fused,
        cg_iters_fused=cg_iters_fused,
        spread_abs=spread,
        spread_rel=spread_rel,
        max_rel_err=max_rel_err,
        mean_rel_err=mean_rel_err,
        is_deterministic=deterministic,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="64k,216k",
                        help="Comma-separated size tags")
    parser.add_argument("--nreps", type=int, default=10)
    parser.add_argument("--penal", type=float, default=3.0)
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    all_rows = []

    for sz in sizes:
        if sz not in SIZE_LADDER:
            print(f"Unknown size {sz}, skipping.")
            continue
        row = run_determinism(sz, args.nreps, args.penal)
        all_rows.append(row)

    # 閳光偓閳光偓 Summary 閳光偓閳光偓
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Size':>5}  {'n_elem':>10}  {'spread_rel':>12}  "
          f"{'max_err':>10}  {'verdict':>20}")
    for r in all_rows:
        v = "OK (< 1e-4)" if r["is_deterministic"] else "WARN spread large"
        print(f"  {r['size']:>5}  {r['n_elem']:>10,}  "
              f"{r['spread_rel']:>12.2e}  {r['max_rel_err']:>10.2e}  {v:>20}")

    # 閳光偓閳光偓 Save 閳光偓閳光偓
    out_dir  = ROOT / "experiments" / "phase3"
    csv_path  = out_dir / "determinism_study.csv"
    json_path = out_dir / "determinism_study.json"

    flat = []
    for r in all_rows:
        for i, (c, cg, t) in enumerate(zip(
            r["compliances_fused"], r["cg_iters_fused"],
            [None] * len(r["compliances_fused"]),   # times not stored per-rep in row
        )):
            flat.append(dict(
                size=r["size"], n_elem=r["n_elem"],
                rep=i + 1, compliance_fused=c, cg_iters=cg,
                c_fp64_ref=r["c_fp64_ref"],
                rel_err=abs(c - r["c_fp64_ref"]) / r["c_fp64_ref"],
            ))

    if flat:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            writer.writeheader()
            writer.writerows(flat)

        json_out = []
        for r in all_rows:
            r2 = {k: v for k, v in r.items()}
            json_out.append(r2)
        with open(json_path, "w") as f:
            json.dump(json_out, f, indent=2)

        print(f"\n  Saved: {csv_path}")
        print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
