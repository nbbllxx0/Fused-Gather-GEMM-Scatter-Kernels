"""
ncu_profile.py
--------------
Measure per-kernel hardware performance counters for the fused FP32 and
FP64 baseline matvec kernels using the NVIDIA Nsight Compute CLI (ncu).

Two modes
---------
A) Subprocess mode (recommended):
   This script is invoked BY ncu as a profiled target:
       ncu --set full -o ncu_out --target-processes all \\
           python experiments/phase3/ncu_profile.py --kernel-only

   Or use the helper function that builds the ncu command automatically:
       python experiments/phase3/ncu_profile.py --launch-ncu

B) Python-CUDA-events mode (fallback, no ncu required):
   Uses CuPy CUDA events to measure achieved bandwidth and compute
   utilization estimates for the key kernel at each size.

   python experiments/phase3/ncu_profile.py --cuda-events

Key metrics captured (mode A)
------------------------------
  - l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum  [DRAM read bytes]
  - l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum  [DRAM write bytes]
  - lts__t_bytes.sum                               [L2 traffic]
  - sm__throughput.avg.pct_of_peak_sustained_elapsed  [SM utilization]
  - l2_global_load_bytes  [L2 閳?DRAM load bytes, proxy for DRAM BW]

Usage
-----
    python experiments/phase3/ncu_profile.py --cuda-events
    python experiments/phase3/ncu_profile.py --launch-ncu --sizes 216k,1M
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
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

RTX4090_PEAK_BW_TBs  = 1.008     # TB/s (from spec sheet, GDDR6X)
RTX4090_PEAK_FP32_TF = 82.6      # TFLOP/s


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
        nelx=nelx, nely=nely, nelz=nelz, volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
        rmin=1.5,
    )
    bc = generate_bc(spec)
    return dict(
        nelx=nelx, nely=nely, nelz=nelz,
        n_elem=nelx * nely * nelz,
        fixed=bc.fixed_dofs.astype(np.int32),
        free=bc.free_dofs.astype(np.int32),
        F=bc.F, ndof=bc.ndof,
    )


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# CUDA-events bandwidth measurement (no ncu required)
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def measure_kernel_bandwidth(size_tag, n_warmup=5, n_timed=20):
    """
    Estimate achieved DRAM bandwidth by:
      (1) Timing fused FP32 and FP64-baseline matvec with CuPy CUDA events.
      (2) Computing bytes_transferred / measured_time.

    Theoretical bytes:
      Baseline (3-stage FP64): gather + GEMM + scatter = 768 bytes/elem
      Fused (FP32):            gather + scatter         = 384 bytes/elem
      (See 鎼?.3 Eq 5 and 6 of the paper)
    """
    import cupy as cp
    from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
    from gpu_fem.solver_v2 import MatrixFreeKff
    from gpu_fem.cuda_fused_matvec import FusedMatvec

    nelx, nely, nelz = SIZE_LADDER[size_tag]
    prob = build_cantilever_problem(nelx, nely, nelz)
    n_elem = prob["n_elem"]
    n_free = len(prob["free"])

    edof = _edof_table_3d(nelx, nely, nelz)
    edof_gpu    = cp.asarray(edof)
    KE_unit_gpu = cp.asarray(KE_UNIT_3D)
    free_gpu    = cp.asarray(prob["free"])

    mf = MatrixFreeKff(
        edof_gpu=edof_gpu, KE_unit_gpu=KE_unit_gpu,
        free_gpu=free_gpu, n_free=n_free, ndof=prob["ndof"],
    )

    rho = cp.full(n_elem, 0.5, dtype=cp.float64)
    E_e_fp64 = 1e-9 + (1.0 - 1e-9) * rho ** 3.0
    E_e_fp32 = E_e_fp64.astype(cp.float32)
    v_fp64   = cp.random.default_rng(42).standard_normal(n_free, dtype=cp.float64)
    v_fp32   = v_fp64.astype(cp.float32)

    # Bytes transferred per matvec (theoretical)
    bytes_per_elem_fp64_baseline = (24 * 8 * 4)   # 768 bytes/elem (gather+GEMM_in+GEMM_out+scatter)
    bytes_per_elem_fp32_fused    = (24 * 4 * 2)   # 384 bytes/elem (gather + scatter only, FP32)
    # Also: edof table: 24 * 4 bytes/elem; rho: 4 bytes/elem
    bytes_edof_per_elem          = 24 * 4          # 96 bytes
    bytes_rho_per_elem           = 4               # 4 bytes
    bytes_fp64_total = (bytes_per_elem_fp64_baseline + bytes_edof_per_elem + bytes_rho_per_elem) * n_elem
    bytes_fp32_total = (bytes_per_elem_fp32_fused    + bytes_edof_per_elem + bytes_rho_per_elem) * n_elem

    results = {}

    # 閳光偓閳光偓 FP64 baseline (unfused three-stage Python path) 閳光偓閳光偓
    fused = FusedMatvec(
        edof_gpu=edof_gpu, KE_unit_gpu=KE_unit_gpu,
        ndof=prob["ndof"],
    )

    # Warmup
    for _ in range(n_warmup):
        _ = mf.matvec(v_fp64, E_e_fp64)
    cp.cuda.Stream.null.synchronize()

    # Timed FP64 baseline
    ev_start = cp.cuda.Event(); ev_end = cp.cuda.Event()
    ev_start.record()
    for _ in range(n_timed):
        _ = mf.matvec(v_fp64, E_e_fp64)
    ev_end.record()
    ev_end.synchronize()
    t_fp64_us = cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e3 / n_timed  # 娓璼 per call

    bw_fp64_TBs = bytes_fp64_total / (t_fp64_us * 1e-6) / 1e12
    ai_fp64     = (2 * 24 * 24 * n_elem) / bytes_fp64_total   # FLOP/byte

    results["fp64_baseline"] = dict(
        t_us=t_fp64_us,
        bytes_theory=bytes_fp64_total,
        bw_TBs=bw_fp64_TBs,
        pct_peak_bw=bw_fp64_TBs / RTX4090_PEAK_BW_TBs * 100,
        arithmetic_intensity=ai_fp64,
    )
    print(f"  FP64 baseline: {t_fp64_us:.1f} 娓璼  "
          f"BW={bw_fp64_TBs*1e3:.1f} GB/s  "
          f"({bw_fp64_TBs / RTX4090_PEAK_BW_TBs * 100:.1f}% peak)  "
          f"AI={ai_fp64:.2f} FLOP/B")

    # 閳光偓閳光偓 Fused FP32 kernel 閳光偓閳光偓
    for _ in range(n_warmup):
        _ = fused.matvec(v_fp32, E_e_fp32, free_gpu, dtype="fp32")
    cp.cuda.Stream.null.synchronize()

    ev_start.record()
    for _ in range(n_timed):
        _ = fused.matvec(v_fp32, E_e_fp32, free_gpu, dtype="fp32")
    ev_end.record()
    ev_end.synchronize()
    t_fused_us = cp.cuda.get_elapsed_time(ev_start, ev_end) * 1e3 / n_timed

    bw_fused_TBs = bytes_fp32_total / (t_fused_us * 1e-6) / 1e12
    ai_fused     = (2 * 24 * 24 * n_elem) / bytes_fp32_total

    results["fused_fp32"] = dict(
        t_us=t_fused_us,
        bytes_theory=bytes_fp32_total,
        bw_TBs=bw_fused_TBs,
        pct_peak_bw=bw_fused_TBs / RTX4090_PEAK_BW_TBs * 100,
        arithmetic_intensity=ai_fused,
    )
    print(f"  Fused FP32:    {t_fused_us:.1f} us  "
          f"BW={bw_fused_TBs*1e3:.1f} GB/s  "
          f"({bw_fused_TBs / RTX4090_PEAK_BW_TBs * 100:.1f}% peak)  "
          f"AI={ai_fused:.2f} FLOP/B")
    print(f"  Speedup fused/fp64 = {t_fp64_us/t_fused_us:.2f}x")

    del mf, fused, rho, E_e_fp64, E_e_fp32, v_fp64, v_fp32
    _free_gpu_memory()

    return dict(
        size=size_tag, n_elem=n_elem, n_warmup=n_warmup, n_timed=n_timed,
        **{f"{k}_{m}": v
           for k, res in results.items()
           for m, v in res.items()},
    )


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# ncu subprocess launcher
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def launch_ncu(sizes, python_exe=None, out_prefix="ncu_fused"):
    """Build and run the ncu command to profile this script in --kernel-only mode."""
    if python_exe is None:
        python_exe = sys.executable

    ncu_exe = shutil.which("ncu")
    if ncu_exe is None:
        # Try common Windows install locations
        candidates = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin\ncu.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\ncu.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                ncu_exe = c
                break

    if ncu_exe is None:
        print("ERROR: ncu (Nsight Compute CLI) not found in PATH.")
        print("  Install NVIDIA Nsight Compute or add its bin/ to PATH.")
        return

    out_dir = ROOT / "experiments" / "phase3"
    out_path = out_dir / out_prefix

    sizes_str = ",".join(sizes)
    cmd = [
        ncu_exe,
        "--set", "full",
        "--export", str(out_path),
        "--force-overwrite",
        "--target-processes", "all",
        "--kernel-name", "fused_matvec",
        python_exe, str(Path(__file__).resolve()),
        "--kernel-only",
        "--sizes", sizes_str,
    ]
    print(f"  Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"  ncu profile saved to: {out_path}.ncu-rep")
    except subprocess.CalledProcessError as e:
        print(f"  ncu failed with exit code {e.returncode}")
    except FileNotFoundError:
        print("  ncu executable not found.")


def kernel_only_run(sizes):
    """Run just enough matvec calls to be profiled by ncu (no SIMP)."""
    import cupy as cp
    from gpu_fem.pub_simp_solver import _edof_table_3d, _build_sparse_indices, KE_UNIT_3D
    from gpu_fem.solver_v2 import MatrixFreeKff
    from gpu_fem.cuda_fused_matvec import FusedMatvec

    for sz in sizes:
        if sz not in SIZE_LADDER:
            continue
        nelx, nely, nelz = SIZE_LADDER[sz]
        prob = build_cantilever_problem(nelx, nely, nelz)
        n_elem = prob["n_elem"]
        n_free = len(prob["free"])

        edof = _edof_table_3d(nelx, nely, nelz)
        edof_gpu    = cp.asarray(edof)
        KE_unit_gpu = cp.asarray(KE_UNIT_3D)
        free_gpu    = cp.asarray(prob["free"])

        fused = FusedMatvec(
            edof_gpu=edof_gpu, KE_unit_gpu=KE_unit_gpu,
            free_gpu=free_gpu, n_free=n_free, ndof=prob["ndof"],
        )
        E_e = cp.full(n_elem, 0.5, dtype=cp.float32)
        v   = cp.ones(n_free, dtype=cp.float32)

        # Warmup
        for _ in range(3):
            fused.matvec(v, E_e, free_gpu, dtype="fp32")
        cp.cuda.Stream.null.synchronize()

        # The profiled calls
        for _ in range(10):
            fused.matvec(v, E_e, free_gpu, dtype="fp32")
        cp.cuda.Stream.null.synchronize()
        print(f"  [kernel-only] {sz} ({n_elem:,} elem): done")
        del fused, edof_gpu, KE_unit_gpu, free_gpu, E_e, v
        _free_gpu_memory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda-events",  action="store_true",
                        help="Python CUDA-events bandwidth measurement (no ncu needed)")
    parser.add_argument("--launch-ncu",   action="store_true",
                        help="Launch ncu as subprocess to profile fused kernel")
    parser.add_argument("--kernel-only",  action="store_true",
                        help="Run kernel-only mode (invoked BY ncu)")
    parser.add_argument("--sizes",        default="64k,216k,512k,1M")
    parser.add_argument("--n-warmup",     type=int, default=10)
    parser.add_argument("--n-timed",      type=int, default=50)
    args = parser.parse_args()

    sizes = [s.strip() for s in args.sizes.split(",")]

    if args.kernel_only:
        kernel_only_run(sizes)
        return

    if args.launch_ncu:
        launch_ncu(sizes)
        return

    # Default: CUDA-events mode
    print("=" * 70)
    print(f"  Kernel bandwidth measurement (CUDA events)")
    print(f"  sizes={sizes}  n_warmup={args.n_warmup}  n_timed={args.n_timed}")
    print(f"  RTX 4090 peak BW = {RTX4090_PEAK_BW_TBs*1e3:.0f} GB/s")
    print("=" * 70)

    all_rows = []
    for sz in sizes:
        if sz not in SIZE_LADDER:
            print(f"  WARN: unknown size {sz}, skipping.")
            continue
        print(f"\n  {sz}:")
        row = measure_kernel_bandwidth(sz, n_warmup=args.n_warmup, n_timed=args.n_timed)
        all_rows.append(row)

    # 閳光偓閳光偓 Summary 閳光偓閳光偓
    print(f"\n{'='*90}")
    print("  BANDWIDTH SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Size':>5}  {'t_fp64(娓璼)':>11}  {'BW_fp64(GB/s)':>14}  "
          f"{'t_fused(娓璼)':>12}  {'BW_fused(GB/s)':>15}  {'pct_peak':>9}  {'speedup':>8}")
    print(f"  {'閳光偓'*85}")
    for r in all_rows:
        t64   = r.get("fp64_baseline_t_us", float("nan"))
        bw64  = r.get("fp64_baseline_bw_TBs", float("nan")) * 1e3
        tf    = r.get("fused_fp32_t_us", float("nan"))
        bwf   = r.get("fused_fp32_bw_TBs", float("nan")) * 1e3
        pct   = r.get("fused_fp32_pct_peak_bw", float("nan"))
        sp    = t64 / tf if (t64 > 0 and tf > 0) else float("nan")
        print(f"  {r['size']:>5}  {t64:>11.1f}  {bw64:>14.1f}  "
              f"{tf:>12.1f}  {bwf:>15.1f}  {pct:>9.1f}  {sp:>8.2f}x")

    # 閳光偓閳光偓 Save 閳光偓閳光偓
    out_dir   = ROOT / "experiments" / "phase3"
    csv_path  = out_dir / "ncu_bandwidth.csv"
    json_path = out_dir / "ncu_bandwidth.json"

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
