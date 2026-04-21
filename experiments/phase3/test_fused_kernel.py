"""
test_fused_kernel.py
--------------------
Phase 3 閳?Validate the fused CUDA kernel against the CuPy matfree baseline.

Checks:
  1. Parity: 閳ユ潾_fused - y_cupy閳?/ 閳ユ潾_cupy閳?< 1e-5 (FP32)
  2. Timing: compares fused kernel vs CuPy matfree at 64k, 216k, 512k, 1M
"""
from __future__ import annotations

import os, sys, tempfile
from pathlib import Path


def _prefer_pytorch_env():
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
    if os.environ.get("GPU_FEM_ENV_BOOTSTRAPPED") == "1": return
    if not env_python.exists(): return
    env = os.environ.copy(); env["GPU_FEM_ENV_BOOTSTRAPPED"] = "1"
    os.execve(str(env_python), [str(env_python), __file__, *sys.argv[1:]], env)

_prefer_pytorch_env()

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def _ct(fn, reps=100):
    """CuPy event timing, returns ms/call."""
    import cupy as cp
    fn(); cp.cuda.Device().synchronize()
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(reps): fn()
    e.record(); e.synchronize()
    return cp.cuda.get_elapsed_time(s, e) / reps


def run_parity_and_timing(nelx: int, nely: int, nelz: int):
    import cupy as cp
    from gpu_fem.pub_simp_solver import _edof_table_3d, KE_UNIT_3D
    from gpu_fem.presets import get_preset
    from gpu_fem.bc_generator import generate_bc
    from gpu_fem.solver_v2 import MatrixFreeKff
    from gpu_fem.cuda_fused_matvec import FusedMatvec

    n_elem = nelx * nely * nelz

    # Use cantilever_gpu_large BC for a realistic free-DOF distribution when size matches
    # otherwise synthesize: left face fixed
    bc_like = None
    for pname in ["cantilever_gpu_medium", "cantilever_gpu_large",
                  "cantilever_gpu_xlarge", "cantilever_gpu_xxlarge"]:
        s = get_preset(pname)
        if (s.nelx, s.nely, s.nelz) == (nelx, nely, nelz):
            bc_like = generate_bc(s)
            break
    if bc_like is None:
        # Synthesize: left-face fixed cantilever
        from gpu_fem.problem_spec import ProblemSpec, EdgeSupport, PointLoad
        s = ProblemSpec(Lx=2.0, Ly=1.0, Lz=0.5,
                        nelx=nelx, nely=nely, nelz=nelz, volfrac=0.3,
                        supports=[EdgeSupport(edge="left", constraint="fixed")],
                        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)])
        bc_like = generate_bc(s)

    ndof   = bc_like.ndof
    free   = cp.asarray(bc_like.free_dofs.astype(np.int32))
    n_free = int(free.size)

    edof_cpu = _edof_table_3d(nelx, nely, nelz)
    edof_gpu = cp.asarray(edof_cpu, dtype=cp.int32)

    KE64 = cp.asarray(KE_UNIT_3D, dtype=cp.float64)
    KE32 = KE64.astype(cp.float32)

    rng = np.random.default_rng(42)
    u_free_f32 = cp.asarray(rng.standard_normal(n_free).astype(np.float32))
    u_free_f64 = u_free_f32.astype(cp.float64)
    E_f32 = cp.full(n_elem, 0.5, dtype=cp.float32)
    E_f64 = E_f32.astype(cp.float64)

    print(f"\n  {nelx}x{nely}x{nelz} = {n_elem:,} elements  ({n_free:,} free DOFs)")
    print(f"  {'='*60}")

    # Baseline matfree (CuPy)
    mf64 = MatrixFreeKff(edof_gpu, KE64, free, n_free, ndof)
    mf32 = MatrixFreeKff(edof_gpu, KE64, free, n_free, ndof)   # same, uses FP32 path via dtype

    y_f64 = mf64.matvec(u_free_f64, E_f64)                    # baseline reference
    y_mf32 = mf32.matvec(u_free_f32, E_f32)                   # CuPy FP32 matfree

    # Fused kernel
    fm = FusedMatvec(edof_gpu, KE32, ndof)
    y_fused = fm.matvec(u_free_f32, E_f32, free, dtype="fp32")

    # Parity checks
    ref64    = y_f64.astype(cp.float32)
    err_mf32 = float(cp.linalg.norm(y_mf32 - ref64) / cp.linalg.norm(ref64))
    err_fusd = float(cp.linalg.norm(y_fused - ref64) / cp.linalg.norm(ref64))
    err_fm_vs_mf32 = float(cp.linalg.norm(y_fused - y_mf32) / cp.linalg.norm(y_mf32))

    print(f"  Parity (rel. L2 norm vs FP64 matfree):")
    print(f"    CuPy FP32 matfree  : {err_mf32:.2e}")
    print(f"    Fused kernel FP32  : {err_fusd:.2e}")
    print("    Fused vs CuPy FP32 : {0:.2e}  (should be ~1e-6 - just reduction order)".format(err_fm_vs_mf32))

    # Timing
    reps = 200 if n_elem <= 300_000 else 80
    t_mf64  = _ct(lambda: mf64.matvec(u_free_f64, E_f64), reps)
    t_mf32  = _ct(lambda: mf32.matvec(u_free_f32, E_f32), reps)
    t_fused = _ct(lambda: fm.matvec(u_free_f32, E_f32, free, dtype="fp32"), reps)

    print(f"  Timing (mean over {reps} reps):")
    print(f"    CuPy FP64 matfree  : {t_mf64*1e3:7.0f} 娓璼")
    print(f"    CuPy FP32 matfree  : {t_mf32*1e3:7.0f} 娓璼   "
          f"({t_mf64/t_mf32:.2f}鑴?over FP64)")
    print(f"    Fused FP32 kernel  : {t_fused*1e3:7.0f} 娓璼   "
          f"({t_mf64/t_fused:.2f}鑴?over FP64, "
          f"{t_mf32/t_fused:.2f}鑴?over CuPy FP32)")

    return {
        "n_elem": n_elem,
        "err_fusd": err_fusd,
        "err_fm_vs_mf32": err_fm_vs_mf32,
        "t_mf64_us": t_mf64*1e3,
        "t_mf32_us": t_mf32*1e3,
        "t_fused_us": t_fused*1e3,
    }


def main():
    import cupy as cp
    print("\n" + "="*68)
    print("  Phase 3 - Fused CUDA kernel parity + timing")
    print("="*68)

    sizes = [
        (80, 40, 20),    # 64k
        (120, 60, 30),   # 216k
        (160, 80, 40),   # 512k
        (200, 100, 50),  # 1M
    ]
    results = []
    for sz in sizes:
        try:
            r = run_parity_and_timing(*sz)
            results.append(r)
        except Exception as ex:
            print(f"  ERROR at {sz}: {ex}")
            break

    print(f"\n  {'='*68}")
    print("  Summary - Fused FP32 vs CuPy FP32 matfree")
    print(f"  {'='*68}")
    print(f"  {'n_elem':>10}  {'CuPy FP64':>10}  {'CuPy FP32':>10}  "
          f"{'Fused FP32':>11}  {'Fused/FP32':>10}  {'Fused/FP64':>10}")
    for r in results:
        print(f"  {r['n_elem']:>10,}  {r['t_mf64_us']:>9.0f}us  "
              f"{r['t_mf32_us']:>9.0f}us  {r['t_fused_us']:>10.0f}us  "
              f"{r['t_mf32_us']/r['t_fused_us']:>9.2f}x "
              f"{r['t_mf64_us']/r['t_fused_us']:>9.2f}x")


if __name__ == "__main__":
    main()
