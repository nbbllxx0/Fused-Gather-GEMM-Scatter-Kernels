"""
statistical_repeats.py
----------------------
Run headline SIMP-120 wall-time benchmarks N times and report mean 鍗?std.

Covers Table 3 of paper3 (cantilever benchmark) at the three sizes where
single-run timings are already deposited.  Adds statistical rigor for
reviewer credibility.

Default: 5 repeats at 216k, 512k, 1M for 'fp64' and 'fused' paths.

Usage
-----
    python experiments/phase3/statistical_repeats.py
    python experiments/phase3/statistical_repeats.py --sizes 216k,512k --nreps 3
    python experiments/phase3/statistical_repeats.py --paths fp64,fp32,fused --nreps 5
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


def _free_gpu_memory():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
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


def run_one_simp(size_tag, path, n_iters=120):
    """Run one full SIMP-120 and return (wall_s, compliance, vram_gb)."""
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
        solver_kwargs.update(enable_mixed_precision=True, enable_fused_cuda=False)
    elif path == "fused":
        solver_kwargs.update(enable_mixed_precision=True, enable_fused_cuda=True)
    else:
        raise ValueError(f"Unknown path: {path}")

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
    c = result.get("best_compliance", result.get("final_compliance", float("nan")))
    vram = _vram_gb()
    _free_gpu_memory()
    return float(wall_s), float(c), float(vram)


def bench_with_repeats(size_tag, path, nreps, n_iters=120):
    print(f"\n  [{path:5s}] {size_tag}  ({nreps} reps) ...")
    times = []
    compliances = []
    for rep in range(nreps):
        wall_s, c, vram = run_one_simp(size_tag, path, n_iters)
        times.append(wall_s)
        compliances.append(c)
        print(f"    rep {rep+1}/{nreps}: wall={wall_s:.2f}s  c={c:.5g}")

    arr = np.array(times)
    mean_s = float(np.mean(arr))
    std_s  = float(np.std(arr, ddof=1))
    cv     = std_s / mean_s if mean_s > 0 else float("nan")
    mean_c = float(np.mean(compliances))
    std_c  = float(np.std(compliances, ddof=1))

    print(f"    閳?mean={mean_s:.2f}s  std={std_s:.3f}s  CV={cv*100:.1f}%  "
          f"c_mean={mean_c:.5g}  c_std={std_c:.2e}")

    return dict(
        size=size_tag, path=path, n_iters=n_iters, nreps=nreps,
        mean_s=mean_s, std_s=std_s, cv=cv,
        mean_compliance=mean_c, std_compliance=std_c,
        times=times, compliances=compliances,
        vram_gb=vram,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",  default="216k,512k,1M",
                        help="Comma-separated size tags")
    parser.add_argument("--paths",  default="fp64,fused",
                        help="Comma-separated solver paths")
    parser.add_argument("--nreps",  type=int, default=5,
                        help="Number of repetitions per (size, path) pair")
    parser.add_argument("--niters", type=int, default=120,
                        help="SIMP iterations per run")
    parser.add_argument("--out",    default=None)
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]
    paths = [p.strip() for p in args.paths.split(",")]

    print("=" * 70)
    print(f"  Statistical repeats 閳?SIMP-{args.niters}")
    print(f"  sizes={sizes}  paths={paths}  nreps={args.nreps}")
    print("=" * 70)

    all_rows = []
    for sz in sizes:
        if sz not in SIZE_LADDER:
            print(f"  WARN: unknown size {sz}, skipping.")
            continue
        for path in paths:
            row = bench_with_repeats(sz, path, args.nreps, args.niters)
            all_rows.append(row)

    # 閳光偓閳光偓 Summary table 閳光偓閳光偓
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Size':>5}  {'Path':>6}  {'mean(s)':>9}  {'std(s)':>8}  "
          f"{'CV%':>6}  {'Speedup(F/FP64)':>16}  {'c_mean':>9}")
    print(f"  {'閳光偓'*84}")

    # Group by size to compute speedup
    by_size: dict = {}
    for r in all_rows:
        by_size.setdefault(r["size"], {})[r["path"]] = r

    for sz in sizes:
        if sz not in by_size:
            continue
        size_rows = by_size[sz]
        fp64_mean = size_rows.get("fp64", {}).get("mean_s", None)
        for path in paths:
            if path not in size_rows:
                continue
            r = size_rows[path]
            speedup = (fp64_mean / r["mean_s"]) if fp64_mean and r["mean_s"] > 0 else float("nan")
            sp_str = f"{speedup:.2f}x" if not np.isnan(speedup) else "  --  "
            print(f"  {r['size']:>5}  {r['path']:>6}  "
                  f"{r['mean_s']:>9.2f}  {r['std_s']:>8.3f}  "
                  f"{r['cv']*100:>6.1f}  {sp_str:>16}  {r['mean_compliance']:>9.5g}")

    # 閳光偓閳光偓 Save 閳光偓閳光偓
    out_dir  = ROOT / "experiments" / "phase3"
    csv_path  = out_dir / "statistical_repeats.csv"
    json_path = out_dir / "statistical_repeats.json"

    flat_rows = []
    for r in all_rows:
        for i, (t, c) in enumerate(zip(r["times"], r["compliances"])):
            flat_rows.append(dict(
                size=r["size"], path=r["path"], n_iters=r["n_iters"],
                rep=i + 1, wall_s=t, compliance=c,
                mean_s=r["mean_s"], std_s=r["std_s"], cv=r["cv"],
            ))

    if flat_rows:
        fieldnames = list(flat_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

        summary_out = []
        for r in all_rows:
            row_out = {k: v for k, v in r.items() if k not in ("times", "compliances")}
            row_out["times"] = r["times"]
            row_out["compliances"] = r["compliances"]
            summary_out.append(row_out)
        with open(json_path, "w") as f:
            json.dump(summary_out, f, indent=2)

        print(f"\n  Saved: {csv_path}")
        print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
