"""
test_bf16_kernel.py
-------------------
Phase 3 閳?Validate the BF16 WMMA tensor-core fused kernel.

Checks:
  1. Compiles on SM 8.0+ (Ampere / Ada)
  2. Parity: 閳ユ潾_bf16 - y_fp64閳?/ 閳ユ潾_fp64閳?should be below ~3e-3
     (BF16 has 7-bit mantissa, 钄?閳?7.8e-3; in well-conditioned elasticity
     we expect rel error ~1e-3 or better)
  3. Timing: BF16 vs FP32 fused vs CuPy matfree at 64k / 216k / 512k / 1M
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


def _ct(fn, reps=100):
    import cupy as cp
    fn(); cp.cuda.Device().synchronize()
    s, e = cp.cuda.Event(), cp.cuda.Event()
    s.record()
    for _ in range(reps): fn()
    e.record(); e.synchronize()
    return cp.cuda.get_elapsed_time(s, e) / reps


def run(nelx: int, nely: int, nelz: int):
    import cupy as cp
    from gpu_fem.pub_simp_solver    import _edof_table_3d, KE_UNIT_3D
    from gpu_fem.problem_spec       import ProblemSpec, EdgeSupport, PointLoad
    from gpu_fem.bc_generator       import generate_bc
    from gpu_fem.solver_v2          import MatrixFreeKff
    from gpu_fem.cuda_fused_matvec  import FusedMatvec

    n_elem = nelx * nely * nelz

    spec = ProblemSpec(
        Lx=2.0, Ly=1.0, Lz=0.5,
        nelx=nelx, nely=nely, nelz=nelz, volfrac=0.3,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    )
    bc = generate_bc(spec)
    free   = cp.asarray(bc.free_dofs.astype(np.int32))
    n_free = int(free.size)

    edof_cpu = _edof_table_3d(nelx, nely, nelz)
    edof_gpu = cp.asarray(edof_cpu, dtype=cp.int32)
    KE64 = cp.asarray(KE_UNIT_3D, dtype=cp.float64)

    rng = np.random.default_rng(42)
    u_free_f32 = cp.asarray(rng.standard_normal(n_free).astype(np.float32))
    u_free_f64 = u_free_f32.astype(cp.float64)
    E_f32 = cp.full(n_elem, 0.5, dtype=cp.float32)
    E_f64 = E_f32.astype(cp.float64)

    print(f"\n  {nelx}x{nely}x{nelz} = {n_elem:,} elements  ({n_free:,} free DOFs)")
    print(f"  {'='*68}")

    # Reference: FP64 CuPy matfree
    mf = MatrixFreeKff(edof_gpu, KE64, free, n_free, bc.ndof)
    y_fp64 = mf.matvec(u_free_f64, E_f64).astype(cp.float32)

    # Fused FP32 and BF16
    fm = FusedMatvec(edof_gpu, cp.asarray(KE_UNIT_3D, dtype=cp.float32), bc.ndof)
    if not fm._bf16_available:
        print(f"  BF16 kernel compile failed: {fm._bf16_compile_err}")
        return None
    y_f32 = fm.matvec(u_free_f32, E_f32, free, dtype="fp32")
    y_b16 = fm.matvec(u_free_f32, E_f32, free, dtype="bf16")

    nrm = float(cp.linalg.norm(y_fp64))
    err_f32 = float(cp.linalg.norm(y_f32 - y_fp64) / nrm)
    err_b16 = float(cp.linalg.norm(y_b16 - y_fp64) / nrm)
    err_b16_vs_f32 = float(cp.linalg.norm(y_b16 - y_f32) / float(cp.linalg.norm(y_f32)))

    print(f"  Parity (rel L2 vs FP64 matfree):")
    print(f"    Fused FP32       : {err_f32:.2e}")
    print(f"    Fused BF16 WMMA  : {err_b16:.2e}   "
          f"({'PASS' if err_b16 < 5e-3 else 'FAIL'} 閳?target < 5e-3)")
    print(f"    BF16 vs FP32     : {err_b16_vs_f32:.2e}")

    # Timing
    reps = 200 if n_elem <= 300_000 else 80
    t_f32 = _ct(lambda: fm.matvec(u_free_f32, E_f32, free, dtype="fp32"), reps)
    t_b16 = _ct(lambda: fm.matvec(u_free_f32, E_f32, free, dtype="bf16"), reps)
    t_mf  = _ct(lambda: mf.matvec(u_free_f32, E_f32), reps)
    t_mf64= _ct(lambda: mf.matvec(u_free_f64, E_f64), reps)

    print(f"  Timing (mean over {reps} reps):")
    print(f"    CuPy FP64 matfree     : {t_mf64*1e3:7.0f} 娓璼")
    print(f"    CuPy FP32 matfree     : {t_mf  *1e3:7.0f} 娓璼   ({t_mf64/t_mf :.2f}鑴?")
    print(f"    Fused FP32            : {t_f32 *1e3:7.0f} 娓璼   ({t_mf64/t_f32:.2f}鑴?over FP64, "
          f"{t_mf /t_f32:.2f}鑴?over FP32)")
    print(f"    Fused BF16 WMMA       : {t_b16 *1e3:7.0f} 娓璼   ({t_mf64/t_b16:.2f}鑴?over FP64, "
          f"{t_f32/t_b16:.2f}鑴?over Fused FP32)")

    return dict(
        n_elem=n_elem,
        err_f32=err_f32, err_b16=err_b16,
        t_mf64_ms=t_mf64*1e3, t_mf_ms=t_mf*1e3,
        t_f32_ms=t_f32*1e3, t_b16_ms=t_b16*1e3,
    )


def main():
    print("\n" + "="*72)
    print("  Phase 3 - BF16 WMMA tensor-core kernel parity + timing")
    print("="*72)

    sizes = [
        (80, 40, 20),    # 64k
        (120, 60, 30),   # 216k
        (160, 80, 40),   # 512k
        (200, 100, 50),  # 1M
    ]
    results = []
    for sz in sizes:
        try:
            r = run(*sz)
            if r is not None:
                results.append(r)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"  ERROR at {sz}: {type(ex).__name__}: {ex}")
            break

    if not results:
        return
    print(f"\n  {'='*72}")
    print("  Summary - Fused BF16 vs Fused FP32 vs CuPy matfree")
    print(f"  {'='*72}")
    print(f"  {'n_elem':>10}  {'CuPy64':>8}  {'CuPy32':>8}  "
          f"{'F.FP32':>8}  {'F.BF16':>8}  {'B/F':>5}  {'F/C32':>5}  {'B/C64':>5}  {'err BF16':>9}")
    for r in results:
        print(f"  {r['n_elem']:>10,}  "
              f"{r['t_mf64_ms']:>7.0f}ms  {r['t_mf_ms']:>7.0f}ms  "
              f"{r['t_f32_ms']:>7.0f}ms  {r['t_b16_ms']:>7.0f}ms  "
              f"{r['t_f32_ms']/r['t_b16_ms']:>4.2f}x "
              f"{r['t_mf_ms']/r['t_f32_ms']:>4.2f}x "
              f"{r['t_mf64_ms']/r['t_b16_ms']:>4.2f}x "
              f"{r['err_b16']:>9.2e}")


if __name__ == "__main__":
    main()
