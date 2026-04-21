"""
cuda_fused_matvec.py
--------------------
Phase 3 — Fused gather + GEMV + scatter-add kernel for matrix-free FEM.

The current MatrixFreeKff.matvec() has three sub-operations:
    1. gather:   u_elem[e, j] = u_full[edof[e, j]]
    2. GEMV:     f_elem[e, i] = E[e] * Σ_j KE[i, j] * u_elem[e, j]
    3. scatter:  y_full[k]   += Σ_{(e,j): edof[e,j]=k} f_elem[e, j]

At 216k (profiler): 78 + 61 (FP32) + 238 = 377 μs — scatter dominates.
At 1M (extrapolated): scatter ≈ 63% of matvec time.

This module fuses all three in a single kernel, eliminating:
  - 2 round-trips through HBM for intermediate u_elem, f_elem (saves ~400 MB/s at 1M)
  - CuPy bincount's sort+segment-reduce overhead (replaced by atomic scatter)

Two kernels:
  fused_matvec_fp32  : one thread per element, FP32 throughout
  fused_matvec_bf16  : BF16 multiplies, FP32 accumulator (tensor-core path, Phase 2)

Both assume:
  * ke_size = 24 (3D hex elements, 8 nodes × 3 DOF)
  * edof int32 (fits 2^31 DOFs = 2 billion — far beyond any realistic mesh)
  * u_full, y_full FP32
  * KE  FP32 (cast from FP64 once per solver init)

Usage:
    from gpu_fem.cuda_fused_matvec import FusedMatvec
    fm = FusedMatvec(edof_gpu, KE_unit_gpu, n_dof)   # prepares u_full/y_full buffers
    y_free = fm.matvec(u_free, E_e, free_gpu)         # returns FP32 result on free DOFs
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA source — FP32 fused gather/GEMV/scatter with shared-mem KE and atomic add
# ═══════════════════════════════════════════════════════════════════════════════

_KERNEL_SRC_FP32 = r"""
extern "C" __global__ void fused_matvec_fp32(
    const int*   __restrict__ edof,      // (n_elem, 24) int32, row-major
    const float* __restrict__ KE_global, // (24, 24)     float32, row-major
    const float* __restrict__ E,         // (n_elem,)    float32
    const float* __restrict__ u_full,    // (ndof,)      float32
    float*       __restrict__ y_full,    // (ndof,)      float32 — must be zeroed before call
    const int n_elem
) {
    // Shared: full KE (24×24 = 2304 B) loaded once per block
    __shared__ float KE_s[24*24];

    // Cooperative load of KE into shared memory
    #pragma unroll
    for (int idx = threadIdx.x; idx < 24*24; idx += blockDim.x) {
        KE_s[idx] = KE_global[idx];
    }
    __syncthreads();

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;

    // Per-thread element data in registers
    int   dofs[24];
    float u_e[24];

    const int edof_base = e * 24;
    #pragma unroll
    for (int j = 0; j < 24; ++j) {
        int d   = edof[edof_base + j];
        dofs[j] = d;
        u_e[j]  = u_full[d];
    }

    const float Ee = E[e];

    // Compute f_e = KE · u_e  (24×24 · 24) and scatter-add
    #pragma unroll 4
    for (int i = 0; i < 24; ++i) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 24; ++j) {
            acc += KE_s[i*24 + j] * u_e[j];
        }
        atomicAdd(&y_full[dofs[i]], Ee * acc);
    }
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CUDA source — BF16 tensor-core path (one warp per 16-element tile)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Design notes (Phase 2):
#   We batch 16 elements per warp: U = (16, 24) stack of u_e's, cast to BF16.
#   KE^T = (24, 24) is padded/cast to BF16 and stored in shared memory.
#   The per-warp GEMM is  F = U @ KE^T   →  (16, 24)  using WMMA 16×16×16 BF16
#   tiles.  Dimension 24 is padded to 32 (2 tiles of K=16) and N=24 padded
#   to 32 (2 tiles of N=16); the masked result extracts the 24-wide stripe.
#
#   Accumulator stays FP32 (WMMA BF16 → FP32), so the only precision loss
#   is the BF16 cast of U and KE.  BF16 has FP32 dynamic range (8 exp bits),
#   so no under/overflow for well-conditioned linear elasticity.
#
# This kernel is written separately (below) and currently left as a stub that
# the Python wrapper will finalize after the FP32 path is validated.
#
# Reference: Nvidia WMMA API, Ada/Ampere BF16 tensor cores.
# ═══════════════════════════════════════════════════════════════════════════════

_KERNEL_SRC_BF16 = r"""
// ─────────────────────────────────────────────────────────────────────────────
// Fused matvec — BF16 WMMA tensor-core path.
//
// Layout (per warp, 16 elements batched):
//     U    : (16, 32)  BF16, row-major  — cols 0..23 = u_e, cols 24..31 = pad
//     KE   : (32, 32)  BF16, row-major  — rows/cols 24..31 pad with zeros
//     F    : (16, 32)  FP32              — accumulator result, only 0..23 used
//
// WMMA tile size is (M=16, N=16, K=16).  Therefore one 16-element batch per
// warp uses exactly 1 M-tile × 2 N-tiles × 2 K-tiles = 4 mma_sync ops to
// produce the full F(16, 32) result.  Accumulator is FP32 so the only loss
// vs. FP32 fused is the BF16 cast of U and KE (BF16 has full FP32 dynamic
// range; 7-bit mantissa ≈ 3e-3 rounding noise).
//
// Shared memory per block (8 warps × 16 elems = 128 elems/block):
//     KE_s    :  32 × 32 × 2 B =  2 KB  (single copy, shared by all warps)
//     U_s     :   8 × 16 × 32 × 2 B =  8 KB
//     F_s     :   8 × 16 × 32 × 4 B = 16 KB
//     dofs_s  :   8 × 16 × 24 × 4 B = 12 KB
//     total   ≈ 38 KB   (well under 100 KB/SM on Ada SM 8.9)
// ─────────────────────────────────────────────────────────────────────────────
#include <mma.h>
#include <cuda_bf16.h>
using namespace nvcuda;

extern "C" __global__ void fused_matvec_bf16(
    const int*            __restrict__ edof,       // (n_elem, 24) int32 row-major
    const __nv_bfloat16*  __restrict__ KE_pad,     // (32, 32)     bf16 row-major
    const float*          __restrict__ E,          // (n_elem,)    float32
    const float*          __restrict__ u_full,     // (ndof,)      float32
    float*                __restrict__ y_full,     // (ndof,)      float32 — zero'd before call
    const int n_elem
) {
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int ELEMS_PER_WARP  = 16;

    __shared__ __nv_bfloat16 KE_s  [32 * 32];                               //  2 KB
    __shared__ __nv_bfloat16 U_s   [WARPS_PER_BLOCK * 16 * 32];             //  8 KB
    __shared__ float         F_s   [WARPS_PER_BLOCK * 16 * 32];             // 16 KB
    __shared__ int           dofs_s[WARPS_PER_BLOCK * 16 * 24];             // 12 KB

    const int warp_id = threadIdx.x >> 5;   // / 32
    const int lane_id = threadIdx.x & 31;   // % 32

    // Cooperative load of KE into shared (32×32 = 1024 bf16 elements)
    #pragma unroll
    for (int idx = threadIdx.x; idx < 32*32; idx += blockDim.x) {
        KE_s[idx] = KE_pad[idx];
    }
    __syncthreads();

    const int elem0 = blockIdx.x * (WARPS_PER_BLOCK * ELEMS_PER_WARP)
                    + warp_id * ELEMS_PER_WARP;

    // Gather + BF16 cast.  Lane 0..15 each handle one element row of U.
    __nv_bfloat16* U_warp = &U_s   [warp_id * 16 * 32];
    int*           D_warp = &dofs_s[warp_id * 16 * 24];
    float*         F_warp = &F_s   [warp_id * 16 * 32];

    if (lane_id < 16) {
        const int e = elem0 + lane_id;
        if (e < n_elem) {
            const int base = e * 24;
            #pragma unroll
            for (int j = 0; j < 24; ++j) {
                const int d = edof[base + j];
                D_warp[lane_id * 24 + j] = d;
                U_warp[lane_id * 32 + j] = __float2bfloat16(u_full[d]);
            }
            #pragma unroll
            for (int j = 24; j < 32; ++j) {
                U_warp[lane_id * 32 + j] = __float2bfloat16(0.0f);
            }
        } else {
            // OOB row: zero U so this warp's WMMA still runs safely
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                U_warp[lane_id * 32 + j] = __float2bfloat16(0.0f);
            }
        }
    }
    __syncwarp();

    // ── WMMA: F(16,32) = U(16,32) @ KE(32,32) ────────────────────────────
    //   1 M-tile × 2 N-tiles × 2 K-tiles
    using FragA = wmma::fragment<wmma::matrix_a,    16, 16, 16, __nv_bfloat16, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b,    16, 16, 16, __nv_bfloat16, wmma::row_major>;
    using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    FragA a_k0, a_k1;
    FragB b_k0n0, b_k0n1, b_k1n0, b_k1n1;
    FragC c_n0, c_n1;

    // A (U) loads: stride 32
    wmma::load_matrix_sync(a_k0, U_warp + 0,  32);      // cols 0..15
    wmma::load_matrix_sync(a_k1, U_warp + 16, 32);      // cols 16..31

    // B (KE) loads: stride 32
    wmma::load_matrix_sync(b_k0n0, KE_s +  0 * 32 +  0, 32);
    wmma::load_matrix_sync(b_k0n1, KE_s +  0 * 32 + 16, 32);
    wmma::load_matrix_sync(b_k1n0, KE_s + 16 * 32 +  0, 32);
    wmma::load_matrix_sync(b_k1n1, KE_s + 16 * 32 + 16, 32);

    wmma::fill_fragment(c_n0, 0.0f);
    wmma::fill_fragment(c_n1, 0.0f);

    // c_n0 += a_k0 @ b_k0n0 + a_k1 @ b_k1n0
    wmma::mma_sync(c_n0, a_k0, b_k0n0, c_n0);
    wmma::mma_sync(c_n0, a_k1, b_k1n0, c_n0);
    // c_n1 += a_k0 @ b_k0n1 + a_k1 @ b_k1n1
    wmma::mma_sync(c_n1, a_k0, b_k0n1, c_n1);
    wmma::mma_sync(c_n1, a_k1, b_k1n1, c_n1);

    // Store F into shared (row-major, stride 32)
    wmma::store_matrix_sync(F_warp + 0,  c_n0, 32, wmma::mem_row_major);
    wmma::store_matrix_sync(F_warp + 16, c_n1, 32, wmma::mem_row_major);
    __syncwarp();

    // ── Scatter-add ───────────────────────────────────────────────────────
    // Lane 0..15 scatter one row each (24 atomicAdds/lane = 384/warp)
    if (lane_id < 16) {
        const int e = elem0 + lane_id;
        if (e < n_elem) {
            const float Ee = E[e];
            #pragma unroll 4
            for (int i = 0; i < 24; ++i) {
                const int   d = D_warp[lane_id * 24 + i];
                const float f = F_warp[lane_id * 32 + i];
                atomicAdd(&y_full[d], Ee * f);
            }
        }
    }
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Python wrapper — compile kernels once, expose a matvec() method
# ═══════════════════════════════════════════════════════════════════════════════

class FusedMatvec:
    """
    Fused gather / KE·u / scatter-add kernel wrapper.

    Drop-in replacement for `MatrixFreeKff.matvec(u_free, E_e)` when
    both are FP32 and the mesh uses 24-DOF hexahedral elements.
    """

    KE_SIZE       = 24
    BLOCK_X_FP32  = 128   # FP32: one thread per element, 128 elems/block
    # BF16 path: 8 warps × 16 elems = 128 elems/block (same total throughput)
    WARPS_PER_BLOCK_BF16 = 8
    ELEMS_PER_WARP_BF16  = 16
    BLOCK_X_BF16  = 32 * WARPS_PER_BLOCK_BF16            # = 256 threads/block
    ELEMS_PER_BLOCK_BF16 = WARPS_PER_BLOCK_BF16 * ELEMS_PER_WARP_BF16   # = 128

    # Back-compat alias (used by SolverV2's init-print)
    BLOCK_X = BLOCK_X_FP32

    def __init__(self, edof_gpu, KE_unit_gpu, ndof: int):
        import cupy as cp

        n_elem, ke_size = edof_gpu.shape
        if ke_size != self.KE_SIZE:
            raise ValueError(f"FusedMatvec currently supports ke_size=24 only, got {ke_size}")

        self._edof_i32 = edof_gpu.astype(cp.int32, copy=False).ravel()   # (n_elem*24,)
        self._KE_f32   = KE_unit_gpu.astype(cp.float32, copy=False)       # (24, 24)
        self._ndof     = int(ndof)
        self._n_elem   = int(n_elem)

        # KE padded to (32, 32) BF16 for the WMMA kernel.  KE is symmetric
        # so no transpose is needed (KE == KE^T).  Stored as uint16 since
        # CuPy 12.x lacks a native BF16 dtype; kernel receives the buffer
        # as `const __nv_bfloat16*` and the bit layout matches.
        self._KE_bf16_pad = self._pack_KE_bf16_pad(self._KE_f32)

        # Pre-allocated scratch buffers (reused across matvec calls)
        self._u_full = cp.zeros(self._ndof, dtype=cp.float32)
        self._y_full = cp.zeros(self._ndof, dtype=cp.float32)

        # Compile kernels — BF16 requires SM 8.0+ (Ampere/Ada; WMMA bf16).
        # RTX 4090 is SM 8.9, so compile for sm_80 as the minimum compatible.
        self._k_fp32 = cp.RawKernel(_KERNEL_SRC_FP32, "fused_matvec_fp32",
                                    options=("-std=c++14",))
        # CuPy auto-injects -arch=compute_XX for the current device; WMMA BF16
        # requires SM 8.0+ (Ampere/Ada/Hopper), satisfied on any modern GPU
        # we target here.  No additional nvcc flags needed.
        try:
            self._k_bf16 = cp.RawKernel(
                _KERNEL_SRC_BF16, "fused_matvec_bf16",
                options=("-std=c++14",),
            )
            self._bf16_available = True
        except Exception as ex:
            # WMMA compile fail (e.g. older GPU): FP32 path still works
            self._k_bf16 = None
            self._bf16_available = False
            self._bf16_compile_err = repr(ex)

    # ------------------------------------------------------------------
    # BF16 KE padding — pure CuPy, bit-manipulation RNE-correct truncation
    # ------------------------------------------------------------------
    @staticmethod
    def _pack_KE_bf16_pad(KE_unit_f32):
        """Pad KE (24,24) fp32 to (32,32), then pack as bf16 uint16 bit-pattern.

        BF16 = FP32 with lower 16 mantissa bits dropped.  Round-to-nearest-
        even is implemented via `+ 0x7FFF + ((bits >> 16) & 1)` then `>> 16`.
        """
        import cupy as cp

        pad = cp.zeros((32, 32), dtype=cp.float32)
        pad[:24, :24] = KE_unit_f32
        bits  = pad.view(cp.uint32)
        rnd   = 0x7FFF + ((bits >> 16) & 1)
        bf16  = ((bits + rnd) >> 16).astype(cp.uint16)
        return bf16          # (32,32) uint16 — BF16 bit-pattern

    # ------------------------------------------------------------------
    # Matvec (FP32 path)
    # ------------------------------------------------------------------

    def matvec(self, u_free, E_e, free_gpu, *, dtype: str = "fp32"):
        """
        Compute y_free = K_ff · u_free (all FP32) using fused CUDA kernel.

        Parameters
        ----------
        u_free   : (n_free,) cp.ndarray float32
        E_e      : (n_elem,) cp.ndarray float32
        free_gpu : (n_free,) cp.ndarray int32
        dtype    : "fp32" (default) or "bf16" (WMMA TC path — Phase 2)

        Returns
        -------
        y_free : (n_free,) cp.ndarray float32
        """
        import cupy as cp

        u_full = self._u_full
        y_full = self._y_full

        # 1. Zero y_full; populate u_full[free] = u_free, fixed DOFs stay 0
        y_full.fill(0.0)
        u_full.fill(0.0)
        u_full[free_gpu] = u_free.astype(cp.float32, copy=False)

        # 2. Launch fused kernel
        E_f32 = E_e.astype(cp.float32, copy=False)
        self._launch(dtype, u_full, y_full, E_f32)

        # 3. Restrict to free DOFs
        return y_full[free_gpu]

    # ------------------------------------------------------------------
    # For parity testing: matvec producing a full (ndof,) output
    # ------------------------------------------------------------------

    def matvec_full(self, u_full_in, E_e, *, dtype: str = "fp32"):
        """
        Compute y_full = K · u_full (all FP32) — used for parity tests.
        Caller is responsible for u_full having zero at fixed DOFs.
        """
        import cupy as cp
        y_full = self._y_full
        y_full.fill(0.0)
        cp.copyto(self._u_full, u_full_in.astype(cp.float32, copy=False))

        E_f32 = E_e.astype(cp.float32, copy=False)
        self._launch(dtype, self._u_full, y_full, E_f32)
        return y_full.copy()

    # ------------------------------------------------------------------
    # Internal launcher — dispatches to FP32 or BF16 kernel
    # ------------------------------------------------------------------
    def _launch(self, dtype: str, u_full, y_full, E_f32):
        if dtype == "fp32":
            grid  = ((self._n_elem + self.BLOCK_X_FP32 - 1) // self.BLOCK_X_FP32,)
            block = (self.BLOCK_X_FP32,)
            self._k_fp32(grid, block,
                         (self._edof_i32, self._KE_f32, E_f32,
                          u_full, y_full, self._n_elem))
        elif dtype == "bf16":
            if not self._bf16_available:
                raise RuntimeError(
                    f"BF16 WMMA kernel failed to compile; "
                    f"use dtype='fp32'.  Reason: {getattr(self, '_bf16_compile_err', 'unknown')}"
                )
            epb = self.ELEMS_PER_BLOCK_BF16
            grid  = ((self._n_elem + epb - 1) // epb,)
            block = (self.BLOCK_X_BF16,)
            self._k_bf16(grid, block,
                         (self._edof_i32, self._KE_bf16_pad, E_f32,
                          u_full, y_full, self._n_elem))
        else:
            raise ValueError(f"Unknown dtype '{dtype}' — use 'fp32' or 'bf16'")
