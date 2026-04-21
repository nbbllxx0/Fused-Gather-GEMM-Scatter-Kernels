"""
benchmark_torsion.py
--------------------
Phase 3 - Problem-diversity benchmark. SIMP-120 on the torsion preset.

Why: reviewers will ask for results on a different problem class than the
canonical right-load cantilever. This script benchmarks torsion only;
bridge/MBB hard-problem studies are handled by
experiments/phase3/benchmark_hard_problems.py.
Torsion (fully-fixed left face, balanced couple load at the right face)
converges cleanly under the current setup.

Runs FP64 matfree, FP32 matfree, and fused-FP32 paths on
    torsion_gpu_500k  (165x55x55 = 499,125 elements)

Usage:
    python experiments/phase3/benchmark_torsion.py
    python experiments/phase3/benchmark_torsion.py --paths fused      # fused only
    python experiments/phase3/benchmark_torsion.py --iters 60          # shorter
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
    root = Path(__file__).resolve().parents[2]
    for k, v in [("TMP", ".runtime_tmp"), ("TEMP", ".runtime_tmp"),
                 ("CUPY_CACHE_DIR", ".cupy_cache"), ("CUPY_TEMPDIR", ".cupy_cache")]:
        (root / v).mkdir(exist_ok=True)
        os.environ[k] = str(root / v)
    tempfile.tempdir = str(root / ".runtime_tmp")
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


def run_torsion(preset_name: str, path: str, n_iters: int) -> dict:
    from gpu_fem.presets                  import get_preset
    from gpu_fem.bc_generator             import generate_bc
    from gpu_fem.solver_v2                import SolverV2
    from gpu_fem.simp_gpu                 import run_simp_surrogate_gpu, TO3DParams
    from gpu_fem.local_agents             import PureFEMRouter
    from gpu_fem.pub_baseline_controller  import ScheduleOnlyController

    spec = get_preset(preset_name)
    bc = generate_bc(spec)
    n_elem = spec.nelx * spec.nely * spec.nelz

    params = TO3DParams(
        nelx=spec.nelx, nely=spec.nely, nelz=spec.nelz,
        volfrac=spec.volfrac,
        rmin=1.5,
        max_iter=n_iters,
    )

    solver_kwargs = dict(
        grid_dims=(spec.nelx, spec.nely, spec.nelz),
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
    else:
        raise ValueError(f"Unknown path: {path}")

    t_start = time.perf_counter()
    result = run_simp_surrogate_gpu(
        params=params,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=bc.F, ndof=bc.ndof,
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
    vram = _vram_gb()

    # Save per-iter history to side file for plotting
    hist_path = ROOT / "experiments" / "phase3" / f"torsion_500k_{path}_history.json"
    hist_path.write_text(json.dumps({
        "preset": preset_name, "path": path,
        "n_elem": n_elem, "n_iters": n_iters,
        "wall_s": wall_s, "compliance": compliance,
        "compliance_history": result.get("compliance_history", []),
        "params_log": result.get("params_log", []),
    }, indent=2, default=float))

    print(f"  [TORSION {path:5s}] {preset_name}  n_elem={n_elem:,}  "
          f"wall={wall_s:.2f} s  c={compliance:.5g}  VRAM={vram:.2f} GB")

    _free_gpu_memory()
    return dict(
        mode=f"SIMP-{n_iters}", path=path, preset=preset_name, n_elem=n_elem,
        wall_s=float(wall_s), compliance=float(compliance), vram_gb=vram,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="torsion_gpu_500k",
                   choices=["torsion_gpu_medium", "torsion_gpu_500k", "torsion_gpu_1M"])
    p.add_argument("--iters", type=int, default=120)
    p.add_argument("--paths", default="fp64,fp32,fused",
                   help="Comma-separated list: fp64,fp32,fused")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    paths = [s.strip() for s in args.paths.split(",") if s.strip()]

    import cupy as cp
    import torch
    print(f"\n{'='*88}")
    print(f"  Phase 3 torsion bench - {torch.cuda.get_device_name(0)}")
    print(f"  Preset={args.preset}  iters={args.iters}  paths={paths}")
    print(f"{'='*88}")

    rows = []
    for path in paths:
        try:
            rows.append(run_torsion(args.preset, path, args.iters))
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"  [TORSION {path}] ERROR: {type(ex).__name__}: {ex}")
            _free_gpu_memory()

    # Table
    print(f"\n{'='*88}")
    print(f"  Torsion {args.preset}  SIMP-{args.iters}")
    print(f"{'='*88}")
    print(f"  {'path':<6}  {'wall(s)':>9}  {'compliance':>12}  {'VRAM(GB)':>9}")
    base = next((r for r in rows if r["path"] == "fp64"), None)
    for r in rows:
        speedup = (base["wall_s"] / r["wall_s"]) if base and r["wall_s"] > 0 else float("nan")
        print(f"  {r['path']:<6}  {r['wall_s']:>9.2f}  {r['compliance']:>12.5f}  "
              f"{r['vram_gb']:>9.2f}   speedup={speedup:.2f}x")

    tag = args.tag or args.preset
    out_csv = ROOT / "experiments" / "phase3" / f"benchmark_torsion_{tag}.csv"
    with open(out_csv, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Saved: {out_csv.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

