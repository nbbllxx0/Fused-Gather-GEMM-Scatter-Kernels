"""
benchmark_hard_problems.py
--------------------------
Phase 3 閳?FP32 speedup on hard (non-cantilever) problems.

Research question: Does FP32 show better end-to-end SIMP speedup on MBB/bridge
problems where warm-start is less effective (bending-dominated density fields)?

Cantilever recall (from Phase 3 scaling study):
  FEA-only cold-start 216k: 1.64鑴?FP32 speedup
  Full SIMP 120 iters 216k: 1.03鑴?(warm-start dominates 閳?CG is minor fraction)

MBB hypothesis: bending-dominated optimal topologies have sharper density
gradients, so warm-start provides less re-use. Expect 1.3-1.8鑴?SIMP speedup.

Runs:
  A) FEA-only (uniform rho=0.5, cold start): MBB 187k, Bridge 187k
  B) Partial SIMP 60 iters (no warm-start):  MBB 187k, Bridge 187k
  C) Partial SIMP 60 iters (with warm-start): MBB 187k
  D) Full SIMP 120 iters (with warm-start):  Cantilever 512k (xlarge)

Usage:
    python experiments/phase3/benchmark_hard_problems.py          # all runs
    python experiments/phase3/benchmark_hard_problems.py --mode A
    python experiments/phase3/benchmark_hard_problems.py --mode D
    python experiments/phase3/benchmark_hard_problems.py --preset mbb_gpu_large --mode AB
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

# 閳光偓閳光偓 Bootstrap: prefer the Pytorch conda env 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

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
    current    = Path(sys.executable).resolve()
    if os.environ.get("GPU_FEM_ENV_BOOTSTRAPPED") == "1":
        return
    if not env_python.exists():
        return
    if current == env_python.resolve():
        return
    env = os.environ.copy()
    env["GPU_FEM_ENV_BOOTSTRAPPED"] = "1"
    os.execve(str(env_python), [str(env_python), __file__, *sys.argv[1:]], env)


_prefer_pytorch_env()

# 閳光偓閳光偓 Path setup 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from gpu_fem.presets      import get_preset
from gpu_fem.bc_generator import generate_bc
from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
from gpu_fem.solver_v2    import SolverV2
from gpu_fem.simp_gpu     import run_simp_surrogate_gpu, TO3DParams
from gpu_fem.local_agents import PureFEMRouter
from gpu_fem.pub_baseline_controller import ScheduleOnlyController


# 閳光偓閳光偓 VRAM helpers 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def _vram_used_gb() -> float:
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        return (total - free) / 1024**3
    except Exception:
        return float("nan")


# 閳光偓閳光偓 Problem builder 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def build_problem(preset_name: str) -> dict:
    spec  = get_preset(preset_name)
    bc    = generate_bc(spec)
    nelx, nely, nelz = spec.nelx, spec.nely, spec.nelz
    return dict(
        preset=preset_name,
        spec=spec,
        nelx=nelx, nely=nely, nelz=nelz,
        n_elem=nelx * nely * nelz,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=bc.F,
        ndof=bc.ndof,
    )


# 閳光偓閳光偓 Mode A: FEA-only (uniform rho, cold start) 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def run_fea_only(preset_name: str, n_calls: int = 5, fp32: bool = False) -> dict:
    prob = build_problem(preset_name)
    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    n_elem = prob["n_elem"]
    tag    = "FP32" if fp32 else "FP64"

    edof     = _edof_table_3d(nelx, nely, nelz)
    row_idx, col_idx = _build_sparse_indices(edof)

    solver = SolverV2(
        edof=edof, row_idx=row_idx, col_idx=col_idx,
        KE_UNIT=KE_UNIT_3D, free=prob["free"], F=prob["F"], ndof=prob["ndof"],
        backend="auto",
        grid_dims=(nelx, nely, nelz),
        enable_warm_start=False,   # cold start 閳?isolates single FEA call
        enable_matrix_free=True,
        enable_mixed_precision=fp32,
        enable_rediscr_gmg=False,
    )

    rho_phys = np.full(n_elem, 0.5)
    penal    = 3.0

    # warmup
    c_warm, _ = solver.solve(rho_phys, penal)

    times = []
    iters_list = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        c, _ = solver.solve(rho_phys, penal)
        times.append(time.perf_counter() - t0)
        iters_list.append(solver.last_cg_iters)

    mean_ms   = np.mean(times) * 1e3
    mean_iters = np.mean(iters_list)

    print(f"  [{tag}] {preset_name:30s}  FEA-only: "
          f"{mean_ms:7.1f} ms  CG={mean_iters:.0f}  VRAM={_vram_used_gb():.2f} GB")

    return {
        "preset": preset_name, "mode": "FEA-only", "dtype": tag,
        "n_elem": n_elem, "mean_ms": mean_ms, "cg_iters": mean_iters,
        "compliance": float(c), "vram_gb": _vram_used_gb(),
    }


# 閳光偓閳光偓 Mode B/C: Partial SIMP 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def run_partial_simp(preset_name: str, n_iters: int = 60,
                     warm_start: bool = False, fp32: bool = False) -> dict:
    prob = build_problem(preset_name)
    nelx, nely, nelz = prob["nelx"], prob["nely"], prob["nelz"]
    n_elem = prob["n_elem"]
    tag    = "FP32" if fp32 else "FP64"
    ws_tag = "warm" if warm_start else "cold"

    params = TO3DParams(
        nelx=nelx, nely=nely, nelz=nelz,
        volfrac=prob["spec"].volfrac,
        rmin=prob["spec"].rmin if prob["spec"].rmin is not None else 1.5,
        max_iter=n_iters,
    )

    solver_kwargs = dict(
        grid_dims=(nelx, nely, nelz),
        enable_warm_start=warm_start,
        enable_matrix_free=True,
        enable_mixed_precision=fp32,
        enable_rediscr_gmg=False,
        enable_profiling=False,
    )

    t_start = time.perf_counter()
    result = run_simp_surrogate_gpu(
        params=params,
        fixed=prob["fixed"],
        free=prob["free"],
        F=prob["F"],
        ndof=prob["ndof"],
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

    print(f"  [{tag}] {preset_name:30s}  SIMP-{n_iters:3d} ({ws_tag:4s}): "
          f"{wall_s:7.2f} s  c={compliance:.4f}  VRAM={_vram_used_gb():.2f} GB")

    return {
        "preset": preset_name, "mode": f"SIMP-{n_iters}-{ws_tag}", "dtype": tag,
        "n_elem": n_elem, "wall_s": wall_s,
        "compliance": float(compliance), "vram_gb": _vram_used_gb(),
    }


# 閳光偓閳光偓 Result table printer 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def print_speedup_table(rows: list[dict]) -> None:
    print(f"\n{'='*80}")
    print(f"  Phase 3 FP32 Speedup Summary")
    print(f"{'='*80}")
    print(f"  {'Preset':<32} {'Mode':<20} {'FP64':>10} {'FP32':>10} {'Speedup':>8} {'Parity':>9}")
    print(f"  {'閳光偓'*77}")

    # Pair FP64 and FP32 rows by (preset, mode)
    paired: dict[tuple, dict] = {}
    for r in rows:
        key = (r["preset"], r["mode"])
        if key not in paired:
            paired[key] = {}
        paired[key][r["dtype"]] = r

    for (preset, mode), dtypes in sorted(paired.items()):
        r64 = dtypes.get("FP64")
        r32 = dtypes.get("FP32")
        if r64 is None or r32 is None:
            continue

        # Time in seconds (SIMP) or ms (FEA-only)
        if "FEA" in mode:
            t64 = r64["mean_ms"]
            t32 = r32["mean_ms"]
            unit = "ms"
        else:
            t64 = r64["wall_s"]
            t32 = r32["wall_s"]
            unit = " s"

        speedup = t64 / t32 if t32 > 0 else float("nan")
        c64 = r64["compliance"]
        c32 = r32["compliance"]
        parity = abs(c64 - c32) / abs(c64) * 100 if c64 != 0 else float("nan")
        parity_ok = parity < 1.0

        print(f"  {preset:<32} {mode:<20} "
              f"{t64:>9.1f}{unit} {t32:>9.1f}{unit} "
              f"{speedup:>7.2f}x "
              f"{'PASS' if parity_ok else 'FAIL':>4} ({parity:.3f}%)")

    print(f"{'='*80}\n")


# 閳光偓閳光偓 Main 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 hard-problem FP32 benchmark")
    p.add_argument("--mode",   default="ABCD",
                   help="Which runs to perform: A=FEA-only, B=SIMP-cold, "
                        "C=SIMP-warm, D=512k SIMP. Combine: 'AB', 'ABCD' (default)")
    p.add_argument("--preset", default=None,
                   help="Override preset (applies to modes A/B/C only)")
    p.add_argument("--iters",  type=int, default=60,
                   help="SIMP iterations for modes B/C (default: 60)")
    p.add_argument("--fea-calls", type=int, default=5,
                   help="FEA-only repetitions for mode A (default: 5)")
    return p.parse_args()


def main():
    args   = parse_args()
    modes  = args.mode.upper()
    rows   = []

    hard_presets = [args.preset] if args.preset else [
        "mbb_gpu_large",     # 150鑴?0鑴?5 = 187,500  閳?bending-dominated
        "bridge_gpu_large",  # 150鑴?0鑴?5 = 187,500  閳?distributed load
    ]

    # 閳光偓閳光偓 A: FEA-only cold start 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    if "A" in modes:
        print(f"\n{'閳光偓'*66}")
        print(f"  Mode A: FEA-only (uniform rho=0.5, cold start, {args.fea_calls} calls)")
        print(f"{'閳光偓'*66}")
        for preset in hard_presets:
            for fp32 in [False, True]:
                rows.append(run_fea_only(preset, n_calls=args.fea_calls, fp32=fp32))

    # 閳光偓閳光偓 B: Partial SIMP, no warm-start 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    if "B" in modes:
        print(f"\n{'閳光偓'*66}")
        print(f"  Mode B: Partial SIMP {args.iters} iters 閳?NO warm-start (cold)")
        print(f"{'閳光偓'*66}")
        for preset in hard_presets:
            for fp32 in [False, True]:
                rows.append(run_partial_simp(
                    preset, n_iters=args.iters, warm_start=False, fp32=fp32))

    # 閳光偓閳光偓 C: Partial SIMP, with warm-start 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    if "C" in modes:
        print(f"\n{'閳光偓'*66}")
        print(f"  Mode C: Partial SIMP {args.iters} iters 閳?WITH warm-start")
        print(f"{'閳光偓'*66}")
        for preset in hard_presets:
            for fp32 in [False, True]:
                rows.append(run_partial_simp(
                    preset, n_iters=args.iters, warm_start=True, fp32=fp32))

    # 閳光偓閳光偓 D: Full SIMP 512k cantilever 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    if "D" in modes:
        print(f"\n{'閳光偓'*66}")
        print(f"  Mode D: Full SIMP 120 iters 閳?cantilever_gpu_xlarge (512k)")
        print(f"{'閳光偓'*66}")
        for fp32 in [False, True]:
            rows.append(run_partial_simp(
                "cantilever_gpu_xlarge", n_iters=120, warm_start=True, fp32=fp32))

    # 閳光偓閳光偓 Summary table 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    if rows:
        print_speedup_table(rows)

        # CSV output
        import csv
        out_path = ROOT / "experiments" / "phase3" / "results_hard_problems.csv"
        with open(out_path, "w", newline="") as f:
            fieldnames = sorted({k for r in rows for k in r})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Results saved to: {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
