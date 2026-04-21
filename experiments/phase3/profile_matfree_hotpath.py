"""
profile_matfree_hotpath.py
--------------------------
Phase 3 閳?Micro-benchmark: isolate and time the three sub-operations of
MatrixFreeKff.matvec:

  1. Gather  :  u_elem = u_full[edof]                    閳?indexed gather
  2. GEMM    :  f_elem = (KE @ u_elem.T * E_e).T         閳?the compute hot path
  3. Scatter :  y_full = cp.bincount(edof_flat, weights)  閳?scatter-add

Reports 娓璼 per call and TFLOPS for FP64, FP32, and FP16 (if torch available).

Also flags a "free optimisation": edof_flat.astype(int64) currently runs every
matvec call; pre-casting in the constructor saves ~15% of scatter time.

Usage
-----
  # From the GPU-FEM-Accel root:
  GPU_FEM_ENV_BOOTSTRAPPED=1 python experiments/phase3/profile_matfree_hotpath.py

  # Auto-relaunch into Pytorch conda env:
  python experiments/phase3/profile_matfree_hotpath.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


# 閳光偓閳光偓 Re-launch under Pytorch conda env (same pattern as benchmark_phase2.py) 閳光偓閳光偓
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# CUDA timing helpers
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def _cupy_time_ms(fn, reps: int) -> float:
    """Time fn() with CuPy CUDA events. Returns ms per call."""
    import cupy as cp
    fn()  # warmup + JIT compilation
    cp.cuda.Device().synchronize()
    start = cp.cuda.Event()
    end   = cp.cuda.Event()
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / reps


def _torch_time_ms(fn, reps: int) -> float:
    """Time fn() with torch CUDA events (for torch-based kernels)."""
    import torch
    fn()  # warmup
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def _reps_for(n_elem: int) -> int:
    """Adaptive repetition count: target ~50 ms of total measurement."""
    return max(100, min(2000, int(5e7 // n_elem)))


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# Sizes to test  (matches memory file smoke-test ladder)
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

SIZES = [
    ("cantilever_gpu_medium",  80,  40,  20,  64_000),
    ("cantilever_gpu_large",  120,  60,  30, 216_000),
    ("cantilever_gpu_512k",   160,  80,  40, 512_000),
    ("cantilever_gpu_1m",     200, 100,  50, 1_000_000),
]

KE_SIZE = 24   # 8 nodes 鑴?3 DOFs


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# Per-size profiler
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def profile_size(label: str, nelx: int, nely: int, nelz: int) -> dict:
    import cupy as cp

    n_elem = nelx * nely * nelz
    n_dof  = (nelx + 1) * (nely + 1) * (nelz + 1) * 3
    rng    = np.random.default_rng(42)
    REPS   = _reps_for(n_elem)

    # 閳光偓閳光偓 Synthetic data (same shapes as real matvec, seeded for reproducibility)
    KE_cpu    = rng.standard_normal((KE_SIZE, KE_SIZE))
    KE_cpu    = (KE_cpu + KE_cpu.T) / 2   # symmetric
    edof_cpu  = rng.integers(0, n_dof, size=(n_elem, KE_SIZE), dtype=np.int32)

    KE64      = cp.asarray(KE_cpu, dtype=cp.float64)
    KE32      = KE64.astype(cp.float32)
    edof      = cp.asarray(edof_cpu)

    # edof_flat variants (mimics current code vs optimised pre-cast)
    edof_flat_i32 = edof.ravel()                         # current code: re-cast each call
    edof_flat_i64 = edof_flat_i32.astype(cp.int64)       # pre-cast once (proposed fix)

    u_full64  = cp.asarray(rng.standard_normal(n_dof), dtype=cp.float64)
    u_full32  = u_full64.astype(cp.float32)

    # Pre-materialise u_elem so GEMM benchmarks don't include gather latency
    u_elem64  = cp.asarray(rng.standard_normal((n_elem, KE_SIZE)), dtype=cp.float64)
    u_elem32  = u_elem64.astype(cp.float32)

    E_e64     = cp.full(n_elem, 0.5, dtype=cp.float64)
    E_e32     = E_e64.astype(cp.float32)

    f_flat64  = cp.asarray(rng.standard_normal(n_elem * KE_SIZE), dtype=cp.float64)
    f_flat32  = f_flat64.astype(cp.float32)

    # 閳光偓閳光偓 FP16 via torch (if available) 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    have_torch = False
    KE16 = u_elem16 = E_e16 = None
    try:
        import torch
        KE16     = torch.as_tensor(KE_cpu,                   dtype=torch.float16, device="cuda")
        u_elem16 = torch.as_tensor(cp.asnumpy(u_elem64),    dtype=torch.float16, device="cuda")
        E_e16    = torch.as_tensor(cp.asnumpy(E_e64),       dtype=torch.float16, device="cuda")
        have_torch = True
    except Exception as exc:
        print(f"    [FP16/torch unavailable: {exc}]")

    # Also test BF16 if torch is available
    KE_bf16 = u_elem_bf16 = E_e_bf16 = None
    have_bf16 = False
    if have_torch:
        try:
            import torch
            KE_bf16     = KE16.to(torch.bfloat16)
            u_elem_bf16 = u_elem16.to(torch.bfloat16)
            E_e_bf16    = E_e16.to(torch.bfloat16)
            have_bf16   = True
        except Exception:
            pass

    # 閳光偓閳光偓 (1) Gather 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    def gather64():
        return u_full64[edof]

    t_gather = _cupy_time_ms(gather64, REPS)

    # 閳光偓閳光偓 (2) GEMM FP64 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    def gemm64():
        return (KE64 @ u_elem64.T * E_e64[None, :]).T

    t_gemm64 = _cupy_time_ms(gemm64, REPS)

    # 閳光偓閳光偓 (3) GEMM FP32 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    def gemm32():
        return (KE32 @ u_elem32.T * E_e32[None, :]).T

    t_gemm32 = _cupy_time_ms(gemm32, REPS)

    # 閳光偓閳光偓 (4) GEMM FP16 via torch 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    t_gemm16 = None
    if have_torch:
        def gemm16():
            return (KE16 @ u_elem16.T * E_e16[None, :]).T
        t_gemm16 = _torch_time_ms(gemm16, REPS)

    # 閳光偓閳光偓 (5) GEMM BF16 via torch 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    t_gemm_bf16 = None
    if have_bf16:
        def gemm_bf16():
            return (KE_bf16 @ u_elem_bf16.T * E_e_bf16[None, :]).T
        t_gemm_bf16 = _torch_time_ms(gemm_bf16, REPS)

    # 閳光偓閳光偓 (6) Scatter 閳?current (re-cast i32閳姕64 each call) 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    def scatter_current():
        return cp.bincount(
            edof_flat_i32.astype(cp.int64),   # re-cast every call (current code)
            weights=f_flat64,
            minlength=n_dof,
        )

    t_scatter_current = _cupy_time_ms(scatter_current, REPS)

    # 閳光偓閳光偓 (7) Scatter 閳?optimised (i64 pre-cast once) 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    def scatter_opt():
        return cp.bincount(edof_flat_i64, weights=f_flat64, minlength=n_dof)

    t_scatter_opt = _cupy_time_ms(scatter_opt, REPS)

    # 閳光偓閳光偓 (8) Scatter FP32 weights 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    def scatter_fp32():
        return cp.bincount(edof_flat_i64, weights=f_flat32, minlength=n_dof)

    t_scatter32 = _cupy_time_ms(scatter_fp32, REPS)

    # 閳光偓閳光偓 (9) Full matvec FP64 (baseline) 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    def full_matvec64():
        u_e   = u_full64[edof]
        f_e   = (KE64 @ u_e.T * E_e64[None, :]).T
        return cp.bincount(
            edof_flat_i32.astype(cp.int64),
            weights=f_e.ravel(),
            minlength=n_dof,
        )

    t_full64 = _cupy_time_ms(full_matvec64, REPS)

    # 閳光偓閳光偓 (10) Full matvec FP32 (current enable_mixed_precision=True path) 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    def full_matvec32():
        u_e   = u_full32[edof]
        f_e   = (KE32 @ u_e.T * E_e32[None, :]).T
        # bincount requires float64 weights in CuPy 閳?upcast f_e (potential fix needed)
        return cp.bincount(
            edof_flat_i64,
            weights=f_e.ravel().astype(cp.float64),
            minlength=n_dof,
        )

    t_full32 = _cupy_time_ms(full_matvec32, REPS)

    # 閳光偓閳光偓 (11) Fused single-kernel FP32 (gather + GEMM + scatter in one pass) 閳光偓閳光偓閳光偓
    # 閳光偓閳光偓 (12) Fused single-kernel BF16 WMMA (same pass, tensor cores)          閳光偓閳光偓
    # Uses the real FusedMatvec kernel with synthetic random data.  Gives a
    # directly comparable bar for the paper's "fused vs unfused" figure.
    t_fused_fp32 = None
    t_fused_bf16 = None
    try:
        from gpu_fem.cuda_fused_matvec import FusedMatvec
        # Fused expects int32 edof, FP32 KE, and a zeroable u_full/y_full inside
        # the object.  We already have edof (int32), E_e32, and u_full32.
        fm = FusedMatvec(edof, KE32, n_dof)

        def fused_fp32():
            return fm.matvec_full(u_full32, E_e32, dtype="fp32")
        t_fused_fp32 = _cupy_time_ms(fused_fp32, REPS)

        if getattr(fm, "_bf16_available", False):
            def fused_bf16():
                return fm.matvec_full(u_full32, E_e32, dtype="bf16")
            t_fused_bf16 = _cupy_time_ms(fused_bf16, REPS)
    except Exception as exc:
        print(f"    [FusedMatvec unavailable: {type(exc).__name__}: {exc}]")

    # 閳光偓閳光偓 TFLOPS 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    flops    = 2 * KE_SIZE * KE_SIZE * n_elem   # ops per GEMM call
    tflops64 = flops / (t_gemm64 * 1e-3) / 1e12
    tflops32 = flops / (t_gemm32 * 1e-3) / 1e12
    tflops16 = (flops / (t_gemm16 * 1e-3) / 1e12) if t_gemm16 is not None else None
    tflops_bf16 = (flops / (t_gemm_bf16 * 1e-3) / 1e12) if t_gemm_bf16 is not None else None

    # 閳光偓閳光偓 Print results 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    W = 72
    print(f"\n{'='*W}")
    print(f"  {label} - {nelx}x{nely}x{nelz} = {n_elem:,} elements")
    print(f"  n_dof={n_dof:,}   GEMM FLOPs/call={flops/1e6:.2f} MFLOPS   REPS={REPS}")
    print(f"{'='*W}")
    print(f"  {'Operation':<32}  {'us/call':>9}  {'TFLOPS':>8}  {'vs FP64':>9}")
    print(f"  {'-'*32}  {'-'*9}  {'-'*8}  {'-'*9}")

    def row(name, t_ms, tfl=None, vs=None):
        t_us  = t_ms * 1e3
        tfl_s = f"{tfl:.3f}" if tfl is not None else "   --   "
        vs_s  = f"{vs:.2f}x" if vs is not None else "   --   "
        print(f"  {name:<32}  {t_us:>9.2f}  {tfl_s:>8}  {vs_s:>9}")

    row("Gather FP64",                   t_gather)
    print()
    row("GEMM  FP64",                    t_gemm64,  tflops64,    1.00)
    row("GEMM  FP32",                    t_gemm32,  tflops32,    t_gemm64 / t_gemm32)
    if t_gemm16 is not None:
        row("GEMM  FP16 (torch TC)",     t_gemm16,  tflops16,    t_gemm64 / t_gemm16)
    if t_gemm_bf16 is not None:
        row("GEMM  BF16 (torch TC)",     t_gemm_bf16, tflops_bf16, t_gemm64 / t_gemm_bf16)
    print()
    row("Scatter FP64 (current, +cast)", t_scatter_current)
    row("Scatter FP64 (opt, pre-cast)",  t_scatter_opt,  None, t_scatter_current / t_scatter_opt)
    row("Scatter FP32 weights",          t_scatter32,    None, t_scatter_current / t_scatter32)
    print(f"  {'閳光偓'*32}  {'閳光偓'*9}  {'閳光偓'*8}  {'閳光偓'*9}")
    row("Full matvec FP64 (current)",    t_full64)
    row("Full matvec FP32 (mixed prec)", t_full32,  None,  t_full64 / t_full32)
    if t_fused_fp32 is not None:
        row("Fused FP32 (single kernel)", t_fused_fp32, None, t_full64 / t_fused_fp32)
    if t_fused_bf16 is not None:
        row("Fused BF16 WMMA (tensor cores)", t_fused_bf16, None, t_full64 / t_fused_bf16)
    print(f"{'閳光偓'*W}")

    # Bottleneck analysis
    parts = [
        ("Gather",          t_gather,           "gather"),
        ("GEMM FP64",       t_gemm64,           "gemm"),
        ("Scatter FP64",    t_scatter_current,  "scatter"),
    ]
    dominant = max(parts, key=lambda x: x[1])
    print(f"  Dominant sub-op in FP64 full matvec: {dominant[0]}"
          f"  ({dominant[1]*1e3:.1f} us / {dominant[1]/t_full64*100:.0f}% of total)")
    pct = {name: t / t_full64 * 100 for name, t, _ in parts}
    print(f"  Gather {pct['Gather']:.0f}%  |  "
          f"GEMM {pct['GEMM FP64']:.0f}%  |  "
          f"Scatter {pct['Scatter FP64']:.0f}%  "
          f"(note: sum > 100 due to independent timing)")
    print(f"{'='*W}")

    return {
        "label":             label,
        "n_elem":            n_elem,
        "t_gather_us":       t_gather * 1e3,
        "t_gemm64_us":       t_gemm64 * 1e3,
        "t_gemm32_us":       t_gemm32 * 1e3,
        "t_gemm16_us":       (t_gemm16 * 1e3) if t_gemm16 is not None else None,
        "t_gemm_bf16_us":    (t_gemm_bf16 * 1e3) if t_gemm_bf16 is not None else None,
        "t_scatter_cur_us":  t_scatter_current * 1e3,
        "t_scatter_opt_us":  t_scatter_opt * 1e3,
        "t_full64_us":       t_full64 * 1e3,
        "t_full32_us":       t_full32 * 1e3,
        "t_fused_fp32_us":   (t_fused_fp32 * 1e3) if t_fused_fp32 is not None else None,
        "t_fused_bf16_us":   (t_fused_bf16 * 1e3) if t_fused_bf16 is not None else None,
        "fused_fp32_speedup": (t_full64 / t_fused_fp32) if t_fused_fp32 is not None else None,
        "fused_bf16_speedup": (t_full64 / t_fused_bf16) if t_fused_bf16 is not None else None,
        "tflops64":          tflops64,
        "tflops32":          tflops32,
        "tflops16":          tflops16,
        "gemm32_speedup":    t_gemm64 / t_gemm32,
        "full32_speedup":    t_full64 / t_full32,
        "scatter_cast_overhead_pct": (t_scatter_current - t_scatter_opt) / t_scatter_current * 100,
    }


# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
# Main
# 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓

def main():
    import cupy as cp

    print("\n" + "=" * 72)
    print("  Phase 3 - MatrixFreeKff Hotpath Profiler")
    print("  Sub-ops: Gather | GEMM (FP64/FP32/FP16/BF16) | Scatter-add")
    print("=" * 72)

    dev   = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name  = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    sms   = props.get("multiProcessorCount", "?")
    print(f"\n  GPU : {name}  ({sms} SMs)")
    print(f"  CuPy: {cp.__version__}")
    try:
        import torch
        print(f"  Torch: {torch.__version__}  (CUDA {torch.version.cuda})")
    except ImportError:
        print("  Torch: not available 閳?FP16/BF16 GEMM will be skipped")

    results = []
    for label, nelx, nely, nelz, _ in SIZES:
        r = profile_size(label, nelx, nely, nelz)
        results.append(r)

    # 閳光偓閳光偓 Summary 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    print(f"\n{'='*72}")
    print("  SUMMARY - GEMM and full-matvec speedup over FP64")
    hdr = (f"  {'preset':<28}  {'n_elem':>8}  {'FP32':>6}  {'FP16':>6}  {'BF16':>6}  "
           f"{'full FP32':>10}  {'fused FP32':>11}  {'fused BF16':>11}")
    print(hdr)
    print(f"  {'-'*28}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*11}  {'-'*11}")
    for r in results:
        sp32    = f"{r['gemm32_speedup']:.2f}x"
        sp16    = f"{r['t_gemm64_us'] / r['t_gemm16_us']:.2f}x" if r["t_gemm16_us"] else "  --  "
        sp_bf16 = f"{r['t_gemm64_us'] / r['t_gemm_bf16_us']:.2f}x" if r["t_gemm_bf16_us"] else "  --  "
        sp_full    = f"{r['full32_speedup']:.2f}x"
        sp_fused32 = f"{r['fused_fp32_speedup']:.2f}x" if r.get("fused_fp32_speedup") else "   --   "
        sp_fused16 = f"{r['fused_bf16_speedup']:.2f}x" if r.get("fused_bf16_speedup") else "   --   "
        print(f"  {r['label']:<28}  {r['n_elem']:>8,}  {sp32:>6}  {sp16:>6}  {sp_bf16:>6}  "
              f"{sp_full:>10}  {sp_fused32:>11}  {sp_fused16:>11}")
    print(f"{'-'*72}")
    print("  Scatter int32->int64 cast overhead (pre-cast once in constructor saves):")
    for r in results:
        print(f"    {r['label']}: {r['scatter_cast_overhead_pct']:.1f}% of scatter time "
              f"({r['t_scatter_cur_us']:.1f} us -> {r['t_scatter_opt_us']:.1f} us)")
    print(f"{'='*72}\n")

    print(f"{'='*72}")
    print("  Scatter int32->int64 cast overhead (pre-cast once in constructor saves):")
    for r in results:
        print(f"    {r['label']}: {r['scatter_cast_overhead_pct']:.1f}% of scatter time "
              f"({r['t_scatter_cur_us']:.1f} us -> {r['t_scatter_opt_us']:.1f} us)")
    print(f"{'='*72}\n")

    # 閳光偓閳光偓 Save CSV/JSON for the F9 profiler_bars figure 閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓閳光偓
    import csv, json
    root     = Path(__file__).resolve().parents[2]
    out_csv  = root / "experiments" / "phase3" / "profile_hotpath.csv"
    out_json = root / "experiments" / "phase3" / "profile_hotpath.json"
    if results:
        keys = list(results[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        out_json.write_text(json.dumps(results, indent=2, default=str))
        print(f"  Saved: {out_csv.relative_to(root)}")
        print(f"  Saved: {out_json.relative_to(root)}")


if __name__ == "__main__":
    main()
