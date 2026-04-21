"""
probe_vram.py
-------------
Measure peak VRAM and assembly/solve time for a single GPU FEM iteration
on a chosen preset, without going through the full SIMP loop.

Usage:
  python scripts/probe_vram.py --preset cantilever_gpu_large
  python scripts/probe_vram.py --preset cantilever_gpu_xlarge --iters 3
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in [str(SRC), str(SRC / "gpu_fem")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def _peak_mb() -> float:
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        return pool.used_bytes() / (1024 ** 2)
    except Exception:
        return -1.0


def _total_pool_mb() -> float:
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        return pool.total_bytes() / (1024 ** 2)
    except Exception:
        return -1.0


def _nvsmi_used_mb() -> float:
    """Process-agnostic GPU memory used (whole device)."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        return float(out[0])
    except Exception:
        return -1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="cantilever_gpu_large")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--backend", type=str, default="cupy")
    args = parser.parse_args()

    from gpu_fem.presets import ALL_PRESETS, n_elements
    from gpu_fem.fem_gpu import GPUFEMSolver
    from gpu_fem.bc_generator import generate_bc
    from gpu_fem.pub_simp_solver import (
        _edof_table_3d, _build_sparse_indices, KE_UNIT_3D,
    )

    spec = ALL_PRESETS[args.preset]
    n_elem = n_elements(args.preset)
    print(f"[probe] preset={args.preset}  n_elem={n_elem:,}")
    print(f"[probe] grid = {spec.nelx}x{spec.nely}x{spec.nelz}")
    print(f"[probe] device-wide VRAM used (start) = {_nvsmi_used_mb():.0f} MB")

    # Build FEM arrays (CPU side)
    t0 = time.time()
    bc = generate_bc(spec)
    edof = _edof_table_3d(spec.nelx, spec.nely, spec.nelz)
    row_idx, col_idx = _build_sparse_indices(edof)
    KE_UNIT = KE_UNIT_3D
    free = bc.free_dofs
    F = bc.F
    ndof = int(bc.ndof)
    print(f"[probe] FEM arrays built in {time.time() - t0:.2f}s")
    print(f"[probe] ndof={ndof:,}  n_coo={len(row_idx):,}")

    # Build solver
    t0 = time.time()
    solver = GPUFEMSolver(
        edof=edof, row_idx=row_idx, col_idx=col_idx,
        KE_UNIT=KE_UNIT, free=free, F=F, ndof=ndof,
        backend=args.backend,
    )
    t_init = time.time() - t0
    print(f"[probe] solver __init__ took {t_init:.2f}s")
    print(f"[probe] cupy pool used after init  = {_peak_mb():.1f} MB")
    print(f"[probe] cupy pool total after init = {_total_pool_mb():.1f} MB")
    print(f"[probe] device-wide VRAM (after init) = {_nvsmi_used_mb():.0f} MB")

    rho = 0.3 * np.ones(n_elem, dtype=np.float64)

    for it in range(1, args.iters + 1):
        t0 = time.time()
        c, dc = solver.solve(rho, 3.0)
        try:
            import cupy as cp
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass
        dt = time.time() - t0
        print(
            f"[probe] iter {it}: solve={dt:.2f}s  C={c:.4f}  "
            f"pool_used={_peak_mb():.1f} MB  pool_total={_total_pool_mb():.1f} MB  "
            f"nvsmi={_nvsmi_used_mb():.0f} MB"
        )

    # Try to force release
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()
    print(f"[probe] device-wide VRAM (after free_all_blocks) = {_nvsmi_used_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
