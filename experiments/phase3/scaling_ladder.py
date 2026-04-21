"""
scaling_ladder.py
-----------------
Phase 3 閳?Headline scaling study for the fused CUDA kernel.

Three comparison paths:
  A) CuPy FP64 matfree              (baseline)
  B) CuPy FP32 matfree              (session-2 fix)
  C) Fused FP32 CUDA kernel         (session-3 contribution)

Two benchmark modes:
  FEA-only (uniform rho=0.5, cold start)   閳?isolates matvec speedup
  SIMP-120 (warm-start, full pipeline)     閳?end-to-end productive speedup

Sizes tested:
  64k  =  80鑴?0鑴?0
  216k = 120鑴?0鑴?0
  512k = 160鑴?0鑴?0
  1M   = 200鑴?00鑴?0
  2M   = 252鑴?26鑴?3
  5M   = 340鑴?70鑴?5
  8M   = 400鑴?00鑴?00   (only if --with-8m passed)

Usage:
    python experiments/phase3/scaling_ladder.py --mode fea   --sizes 216k,512k,1M
    python experiments/phase3/scaling_ladder.py --mode simp  --sizes 216k,512k,1M
    python experiments/phase3/scaling_ladder.py --mode all                # both, full ladder
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


# 閳光偓閳光偓 Bootstrap to the Pytorch conda env 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

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

import numpy as np


# 閳光偓閳光偓 Size ladder 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

SIZE_LADDER = {
    "64k":  (80,  40,  20),
    "216k": (120, 60,  30),
    "512k": (160, 80,  40),
    "1M":   (200, 100, 50),
    "2M":   (252, 126, 63),
    "5M":   (340, 170, 85),
    "8M":   (400, 200, 100),
}


# 閳光偓閳光偓 VRAM helper 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def _vram_used_gb() -> float:
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        return (total - free) / 1024**3
    except Exception:
        return float("nan")


def _free_gpu_memory() -> None:
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()


# 閳光偓閳光偓 Problem builder 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def build_cantilever_problem(nelx: int, nely: int, nelz: int) -> dict:
    from gpu_fem.problem_spec    import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator    import generate_bc

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
        F=bc.F,
        ndof=bc.ndof,
    )


# 閳光偓閳光偓 Solver factory 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def _make_solver(prob: dict, path: str, warm_start: bool):
    """path in {'fp64', 'fp32', 'fused'}."""
    from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
    from gpu_fem.solver_v2       import SolverV2

    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    edof = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)

    kwargs = dict(
        edof=edof, row_idx=row_idx, col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D, free=prob["free"], F=prob["F"], ndof=prob["ndof"],
        backend="auto",
        grid_dims=(nelx, nely, nelz),
        enable_warm_start=warm_start,
        enable_matrix_free=True,
        enable_rediscr_gmg=False,
    )
    if path == "fp64":
        kwargs.update(enable_mixed_precision=False, enable_fused_cuda=False)
    elif path == "fp32":
        kwargs.update(enable_mixed_precision=True,  enable_fused_cuda=False)
    elif path == "fused":
        kwargs.update(enable_mixed_precision=True,  enable_fused_cuda=True)
    else:
        raise ValueError(f"Unknown path: {path}")
    return SolverV2(**kwargs)


# 閳光偓閳光偓 FEA-only bench 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def bench_fea_only(size_tag: str, path: str, n_calls: int = 3) -> dict:
    nelx, nely, nelz = SIZE_LADDER[size_tag]
    prob = build_cantilever_problem(nelx, nely, nelz)
    n_elem = prob["n_elem"]

    solver = _make_solver(prob, path=path, warm_start=False)
    rho_phys = np.full(n_elem, 0.5)
    penal    = 3.0

    # Warmup (compiles kernels, allocates buffers)
    c_warm, _ = solver.solve(rho_phys, penal)

    times, iters_list = [], []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        c, _ = solver.solve(rho_phys, penal)
        times.append(time.perf_counter() - t0)
        iters_list.append(solver.last_cg_iters)

    mean_ms  = float(np.mean(times) * 1e3)
    mean_it  = float(np.mean(iters_list))
    vram_gb  = _vram_used_gb()

    print(f"  [FEA {path:5s}] {size_tag:>4s}  {n_elem:>9,} elem   "
          f"{mean_ms:8.1f} ms  CG={mean_it:6.0f}  VRAM={vram_gb:.2f} GB   c={c:.5g}")

    del solver
    _free_gpu_memory()

    return dict(
        mode="FEA",  path=path, size=size_tag, n_elem=n_elem,
        n_calls=n_calls, mean_ms=mean_ms, cg_iters=mean_it,
        compliance=float(c), vram_gb=vram_gb,
    )


# 閳光偓閳光偓 SIMP-120 bench (warm-start, full pipeline) 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def bench_simp(size_tag: str, path: str, n_iters: int = 120) -> dict:
    from gpu_fem.simp_gpu               import run_simp_surrogate_gpu, TO3DParams
    from gpu_fem.local_agents           import PureFEMRouter
    from gpu_fem.pub_baseline_controller import ScheduleOnlyController
    from gpu_fem.solver_v2              import SolverV2

    nelx, nely, nelz = SIZE_LADDER[size_tag]
    prob = build_cantilever_problem(nelx, nely, nelz)
    n_elem = prob["n_elem"]

    params = TO3DParams(
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=prob["spec"].volfrac,
        rmin=1.5,
        max_iter=n_iters,
    )

    solver_kwargs = dict(
        grid_dims=(nelx, nely, nelz),
        enable_warm_start=True,
        enable_matrix_free=True,
        enable_rediscr_gmg=False,
        enable_profiling=False,
    )
    if path == "fp64":
        solver_kwargs.update(enable_mixed_precision=False, enable_fused_cuda=False)
    elif path == "fp32":
        solver_kwargs.update(enable_mixed_precision=True,  enable_fused_cuda=False)
    elif path == "fused":
        solver_kwargs.update(enable_mixed_precision=True,  enable_fused_cuda=True)

    t_start = time.perf_counter()
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
    wall_s = time.perf_counter() - t_start

    compliance = result.get("best_compliance", result.get("final_compliance", float("nan")))
    vram_gb = _vram_used_gb()

    print(f"  [SIMP {path:5s}] {size_tag:>4s}  {n_elem:>9,} elem   "
          f"{wall_s:8.2f} s  c={compliance:.5g}  VRAM={vram_gb:.2f} GB")

    _free_gpu_memory()
    return dict(
        mode=f"SIMP-{n_iters}", path=path, size=size_tag, n_elem=n_elem,
        wall_s=float(wall_s), compliance=float(compliance), vram_gb=vram_gb,
    )


# 閳光偓閳光偓 Result table 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def print_fea_table(rows: list[dict]) -> None:
    print(f"\n{'='*88}")
    print("  FEA-only scaling - mean ms per call, FP64 vs FP32 vs Fused CUDA")
    print(f"{'='*88}")
    print(f"  {'Size':>5}  {'n_elem':>10}   {'FP64':>9}   {'FP32':>9}   "
          f"{'Fused':>9}   {'F/FP32':>7}   {'F/FP64':>7}   CG iters")
    print(f"  {'閳光偓'*84}")
    by_size: dict[str, dict] = {}
    for r in rows:
        by_size.setdefault(r["size"], {})[r["path"]] = r
    for size_tag in SIZE_LADDER:
        if size_tag not in by_size:
            continue
        row = by_size[size_tag]
        r64 = row.get("fp64");   r32 = row.get("fp32");   rfu = row.get("fused")
        def _t(r): return r["mean_ms"] if r else float("nan")
        t64, t32, tfu = _t(r64), _t(r32), _t(rfu)
        s_fp32 = (t32 / tfu) if (rfu and r32) else float("nan")
        s_fp64 = (t64 / tfu) if (rfu and r64) else float("nan")
        it = (rfu or r32 or r64).get("cg_iters", float("nan"))
        n_elem = (rfu or r32 or r64)["n_elem"]
        print(f"  {size_tag:>5}  {n_elem:>10,}   "
              f"{t64:>8.1f}   {t32:>8.1f}   {tfu:>8.1f}   "
              f"{s_fp32:>6.2f}x  {s_fp64:>6.2f}x  {it:>6.0f}")
    print(f"{'='*88}\n")


def print_simp_table(rows: list[dict]) -> None:
    print(f"\n{'='*92}")
    print("  SIMP end-to-end scaling - wall seconds, FP64 vs FP32 vs Fused CUDA")
    print(f"{'='*92}")
    print(f"  {'Size':>5}  {'n_elem':>10}   {'FP64':>9}   {'FP32':>9}   "
          f"{'Fused':>9}   {'F/FP32':>7}   {'F/FP64':>7}   {'Compliance':>11}")
    print(f"  {'-'*88}")
    by_size: dict[str, dict] = {}
    for r in rows:
        by_size.setdefault(r["size"], {})[r["path"]] = r
    for size_tag in SIZE_LADDER:
        if size_tag not in by_size:
            continue
        row = by_size[size_tag]
        r64 = row.get("fp64");   r32 = row.get("fp32");   rfu = row.get("fused")
        def _t(r): return r["wall_s"] if r else float("nan")
        t64, t32, tfu = _t(r64), _t(r32), _t(rfu)
        s_fp32 = (t32 / tfu) if (rfu and r32) else float("nan")
        s_fp64 = (t64 / tfu) if (rfu and r64) else float("nan")
        c = (rfu or r32 or r64)["compliance"]
        n_elem = (rfu or r32 or r64)["n_elem"]
        print(f"  {size_tag:>5}  {n_elem:>10,}   "
              f"{t64:>8.2f}   {t32:>8.2f}   {tfu:>8.2f}   "
              f"{s_fp32:>6.2f}x  {s_fp64:>6.2f}x  {c:>10.5f}")
    print(f"{'='*92}\n")


# 閳光偓閳光偓 Main 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 scaling ladder - fused CUDA kernel")
    p.add_argument("--mode", default="fea",
                   choices=["fea", "simp", "all"],
                   help="Which benchmarks to run")
    p.add_argument("--sizes", default="216k,512k,1M",
                   help="Comma-separated size tags (see SIZE_LADDER)")
    p.add_argument("--paths", default="fp64,fp32,fused",
                   help="Comma-separated list of paths to compare")
    p.add_argument("--fea-calls", type=int, default=3,
                   help="FEA-only repeats (default: 3)")
    p.add_argument("--simp-iters", type=int, default=120,
                   help="SIMP iterations for --mode simp/all (default: 120)")
    p.add_argument("--tag", default=None,
                   help="Output CSV/JSON tag suffix")
    return p.parse_args()


def main():
    args = parse_args()
    sizes = [s.strip() for s in args.sizes.split(",") if s.strip()]
    paths = [s.strip() for s in args.paths.split(",") if s.strip()]
    for s in sizes:
        if s not in SIZE_LADDER:
            raise SystemExit(f"Unknown size '{s}' 閳?use one of {list(SIZE_LADDER)}")
    for p in paths:
        if p not in ("fp64", "fp32", "fused"):
            raise SystemExit(f"Unknown path '{p}' - use fp64/fp32/fused")

    import cupy as cp
    import torch
    print(f"\n{'='*88}")
    print(f"  Phase 3 scaling ladder - {torch.cuda.get_device_name(0)}")
    print(f"  Sizes: {sizes}   Paths: {paths}   Mode: {args.mode}")
    print(f"{'='*88}")

    fea_rows, simp_rows = [], []

    if args.mode in ("fea", "all"):
        print(f"\n{'閳光偓'*66}\n  FEA-only scaling  ({args.fea_calls} calls per config)\n{'閳光偓'*66}")
        for size_tag in sizes:
            for path in paths:
                try:
                    r = bench_fea_only(size_tag, path, n_calls=args.fea_calls)
                    fea_rows.append(r)
                except Exception as ex:
                    print(f"  [FEA {path:5s}] {size_tag:>4s}  ERROR: {type(ex).__name__}: {ex}")
                    _free_gpu_memory()
        print_fea_table(fea_rows)

    if args.mode in ("simp", "all"):
        print(f"\n{'閳光偓'*66}\n  SIMP-{args.simp_iters} warm-start scaling\n{'閳光偓'*66}")
        for size_tag in sizes:
            for path in paths:
                try:
                    r = bench_simp(size_tag, path, n_iters=args.simp_iters)
                    simp_rows.append(r)
                except Exception as ex:
                    print(f"  [SIMP {path:5s}] {size_tag:>4s}  ERROR: {type(ex).__name__}: {ex}")
                    _free_gpu_memory()
        print_simp_table(simp_rows)

    # 閳光偓閳光偓 Persist 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    tag = args.tag or args.mode
    out_dir = ROOT / "experiments" / "phase3"
    if fea_rows or simp_rows:
        all_rows = fea_rows + simp_rows
        csv_path = out_dir / f"scaling_ladder_{tag}.csv"
        with open(csv_path, "w", newline="") as f:
            fieldnames = sorted({k for r in all_rows for k in r})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        json_path = out_dir / f"scaling_ladder_{tag}.json"
        json_path.write_text(json.dumps({"fea": fea_rows, "simp": simp_rows}, indent=2))
        print(f"  Saved: {csv_path.relative_to(ROOT)}")
        print(f"  Saved: {json_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
