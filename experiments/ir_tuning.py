"""
ir_tuning.py
------------
Find the configuration in which single precision actually pays.

The pilot established two facts: single precision cannot reach the 1e-5
equilibrium tolerance on these systems at all, and iterative refinement with a
single-precision inner solve reaches it comfortably in four outer steps. What
it did not establish is whether that costs less than simply solving in double
precision, because the pilot used the fused single-precision kernel for the
inner solve and the fused kernel is not the fastest single-precision mapping
this paper measures -- the node-owned one is.

This sweeps the inner mapping and the inner tolerance against a
double-precision control, all on the true residual.

Usage:
    python experiments/phase3/ir_tuning.py --sizes 1M,2M
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

SIZES = {"216k": (120, 60, 30), "512k": (160, 80, 40),
         "1M": (200, 100, 50), "2M": (252, 126, 63)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1M")
    ap.add_argument("--inner-paths", default="fused_fp32,fused_ai_fp32,node_fp32")
    ap.add_argument("--inner-tols", default="1e-1,1e-2,1e-3")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.pub_simp_solver import KE_UNIT_3D
    from gpu_fem.cuda_operators import OperatorSuite, NEEDS_EDOF
    from gpu_fem.simp_r2 import build_cantilever

    out_path = args.out or os.path.join(_ROOT, "results", "G6",
                                        "ir_tuning.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []

    for tag in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        nelx, nely, nelz = SIZES[tag]
        prob = build_cantilever(nelx, nely, nelz, load="patch")
        ndof, n_elem = prob["ndof"], prob["n_elem"]
        Ev = 1e-9 + (1.0 - 1e-9) * np.full(n_elem, 0.5) ** 3.0
        print(f"\n### {tag}: {n_elem:,} elements")

        def build(path, dtype):
            s = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof,
                              build_edof=(path in NEEDS_EDOF))
            m = cp.zeros(ndof, dtype=dtype)
            m[cp.asarray(prob["free"])] = 1
            bb = cp.asarray(prob["F"], dtype=dtype) * m
            E = cp.asarray(Ev, dtype=dtype)
            d = s.diagonal(E, path=path)
            Mi = cp.where(d > 0, 1.0 / cp.maximum(d, 1e-300),
                          cp.zeros_like(d)) * m

            def A(v):
                return s.matvec_full(v * m, E, path=path) * m
            return s, m, bb, A, Mi

        def cg_true(A, b, Mi, dtype, tol, maxiter, repl=200, check=50):
            nb = float(cp.linalg.norm(b))
            x = cp.zeros_like(b)
            r = b - A(x)
            z = Mi * r
            p = z.copy()
            rz = cp.dot(r, z)
            for it in range(1, maxiter + 1):
                Ap = A(p)
                pAp = cp.dot(p, Ap)
                if float(pAp) == 0.0:
                    break
                al = rz / pAp
                x = x + al * p
                r = (b - A(x)) if (repl and it % repl == 0) else (r - al * Ap)
                if it % check == 0:
                    if float(cp.linalg.norm(b - A(x)) / nb) <= tol:
                        return x, it
                z = Mi * r
                rzn = cp.dot(r, z)
                p = z + (rzn / rz) * p
                rz = rzn
            return x, it

        # double-precision control
        s64, m64, b64, A64, Mi64 = build("fused_fp64", cp.float64)
        nb = float(cp.linalg.norm(b64))
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        x, it64 = cg_true(A64, b64, Mi64, cp.float64, args.tol, 40000)
        cp.cuda.Stream.null.synchronize()
        w64 = time.perf_counter() - t0
        r64 = float(cp.linalg.norm(b64 - A64(x)) / nb)
        rows.append(dict(size=tag, scheme="fp64", inner_path=None,
                         inner_tol=None, iters=it64, true_resid=r64,
                         wall_s=w64, speedup=1.0))
        print(f"  fp64 control                       {it64:6d} it  "
              f"{r64:.2e}  {w64:7.2f}s")

        for ipath in [p.strip() for p in args.inner_paths.split(",")]:
            s32, m32, b32, A32, Mi32 = build(ipath, cp.float32)
            for itol in [float(v) for v in args.inner_tols.split(",")]:
                xx = cp.zeros_like(b64)
                cp.cuda.Stream.null.synchronize()
                t0 = time.perf_counter()
                inner, outer = 0, 0
                for outer in range(1, 41):
                    rr = b64 - A64(xx)
                    nrm = float(cp.linalg.norm(rr))
                    if nrm / nb <= args.tol:
                        break
                    d32, iti = cg_true((lambda v: A32(v)),
                                       (rr / nrm).astype(cp.float32),
                                       Mi32, cp.float32, itol, 4000)
                    inner += iti
                    xx = xx + d32.astype(cp.float64) * nrm
                cp.cuda.Stream.null.synchronize()
                w = time.perf_counter() - t0
                tr = float(cp.linalg.norm(b64 - A64(xx)) / nb)
                ok = tr <= args.tol
                rows.append(dict(size=tag, scheme="fp32_ir", inner_path=ipath,
                                 inner_tol=itol, outer=outer,
                                 inner_iters=inner, true_resid=tr,
                                 converged=ok, wall_s=w,
                                 speedup=w64 / w if w > 0 else None))
                print(f"  ir {ipath:<16s} tol {itol:<6.0e} "
                      f"{outer:3d}o/{inner:6d}i  {tr:.2e}  {w:7.2f}s  "
                      f"{'x%.2f' % (w64 / w):>7s} {'' if ok else ' NOT CONVERGED'}")
            del s32, m32, b32, Mi32
            cp.get_default_memory_pool().free_all_blocks()
        del s64, m64, b64, Mi64
        cp.get_default_memory_pool().free_all_blocks()

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwritten: {os.path.relpath(out_path, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
