"""
precision_pilot.py
------------------
Decide what single precision can and cannot do on these systems, before any
result is measured directly.

A solver that tests the recursively updated CG residual reports it as
the achieved residual. In double precision the two agree. In single precision
they do not: the recursion drifts by two to three orders over thousands of
iterations, so solves that logged 1e-5 had true residuals near 1e-3. Every
single-precision convergence statement so obtained rests on that quantity.

This measures three things per (mesh, operator), all against the true residual
||b - Ax|| / ||b|| recomputed from scratch:

  fp64      double-precision CG, as a control.
  fp32      single-precision CG with residual replacement, i.e. the recursion
            restarted from a recomputed residual periodically. This is the
            standard remedy for drift, and it answers the question the drift
            was hiding: what accuracy can single precision actually attain?
  fp32+ir   iterative refinement -- residual formed in double precision, the
            correction solved in single precision by the fast operator, the
            update applied in double. If this reaches the tolerance then the
            single-precision operator is still usable and its speed is real;
            the error is in the stopping test, not in the kernel.

Usage:
    python experiments/phase3/precision_pilot.py --sizes 216k,1M
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

SIZES = {"64k": (80, 40, 20), "216k": (120, 60, 30), "512k": (160, 80, 40),
         "1M": (200, 100, 50), "2M": (252, 126, 63)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="216k,1M")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--maxiter", type=int, default=40000)
    ap.add_argument("--replace-every", type=int, default=200)
    ap.add_argument("--ir-inner-tol", type=float, default=1e-2)
    ap.add_argument("--ir-max-outer", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.pub_simp_solver import KE_UNIT_3D
    from gpu_fem.cuda_operators import OperatorSuite, NEEDS_EDOF
    from gpu_fem.simp_r2 import build_cantilever

    out_path = args.out or os.path.join(_ROOT, "results", "G6",
                                        "precision_pilot.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []

    for tag in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        nelx, nely, nelz = SIZES[tag]
        prob = build_cantilever(nelx, nely, nelz, load="patch")
        ndof, n_elem = prob["ndof"], prob["n_elem"]
        print(f"\n### {tag}: {n_elem:,} elements, {ndof:,} DOF")

        # A representative design state: uniform density at the SIMP exponent
        # used in the production runs. Using a saved optimum would tie the
        # pilot to one run; uniform is reproducible and is the state the
        # cold-start ladder already reports.
        rho = np.full(n_elem, 0.5)
        Ev = 1e-9 + (1.0 - 1e-9) * rho ** 3.0

        def build(path, dtype):
            suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof,
                                  build_edof=(path in NEEDS_EDOF))
            mask = cp.zeros(ndof, dtype=dtype)
            mask[cp.asarray(prob["free"])] = 1
            b = cp.asarray(prob["F"], dtype=dtype) * mask
            E = cp.asarray(Ev, dtype=dtype)
            diag = suite.diagonal(E, path=path)
            Minv = cp.where(diag > 0, 1.0 / cp.maximum(diag, 1e-300),
                            cp.zeros_like(diag)) * mask

            def A(v):
                return suite.matvec_full(v * mask, E, path=path) * mask
            return suite, mask, b, A, Minv

        def cg(A, b, Minv, mask, dtype, tol, maxiter, replace_every=0,
               x0=None):
            """CG whose stopping test is the TRUE residual.

            `replace_every` restarts the recursion from a recomputed
            b - Ax, which is what stops the drift rather than merely
            reporting it.
            """
            nb = float(cp.linalg.norm(b))
            x = cp.zeros_like(b) if x0 is None else x0.astype(dtype, copy=True)
            r = b - A(x)
            z = Minv * r
            p = z.copy()
            rz = cp.dot(r, z)
            best = float(cp.linalg.norm(r) / nb)
            it = 0
            for it in range(1, maxiter + 1):
                Ap = A(p)
                pAp = cp.dot(p, Ap)
                if float(pAp) == 0.0:
                    break
                alpha = rz / pAp
                x = x + alpha * p
                if replace_every and it % replace_every == 0:
                    r = b - A(x)
                else:
                    r = r - alpha * Ap
                if it % 50 == 0 or it == maxiter:
                    true = float(cp.linalg.norm(b - A(x)) / nb)
                    best = min(best, true)
                    if true <= tol:
                        return x, it, True, true, best
                z = Minv * r
                rz_new = cp.dot(r, z)
                p = z + (rz_new / rz) * p
                rz = rz_new
            true = float(cp.linalg.norm(b - A(x)) / nb)
            return x, it, bool(true <= tol), true, min(best, true)

        # ---- control: double precision --------------------------------
        suite, mask, b, A, Minv = build("fused_fp64", cp.float64)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        x, it, ok, true, best = cg(A, b, Minv, mask, cp.float64, args.tol,
                                   args.maxiter)
        cp.cuda.Stream.null.synchronize()
        w = time.perf_counter() - t0
        rows.append(dict(size=tag, n_elem=n_elem, scheme="fp64", iters=it,
                         true_resid=true, best=best, converged=ok, wall_s=w))
        print(f"  fp64     {it:6d} it  true {true:.3e}  {'OK ' if ok else 'NO '}"
              f" {w:7.2f}s")
        del suite, mask, b, Minv
        cp.get_default_memory_pool().free_all_blocks()

        # ---- single precision with residual replacement ----------------
        suite, mask, b32, A32, Minv32 = build("fused_fp32", cp.float32)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        x32, it32, ok32, true32, best32 = cg(
            A32, b32, Minv32, mask, cp.float32, args.tol, args.maxiter,
            replace_every=args.replace_every)
        cp.cuda.Stream.null.synchronize()
        w32 = time.perf_counter() - t0
        rows.append(dict(size=tag, n_elem=n_elem, scheme="fp32_replaced",
                         iters=it32, true_resid=true32, best=best32,
                         converged=ok32, wall_s=w32))
        print(f"  fp32     {it32:6d} it  true {true32:.3e} (best {best32:.3e}) "
              f"{'OK ' if ok32 else 'FLOOR'} {w32:7.2f}s")

        # ---- iterative refinement: fp64 residual, fp32 correction ------
        suite64, mask64, b64, A64, Minv64 = build("fused_fp64", cp.float64)
        nb64 = float(cp.linalg.norm(b64))
        x64 = cp.zeros_like(b64)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        inner_total, outer, ok_ir = 0, 0, False
        for outer in range(1, args.ir_max_outer + 1):
            r64 = b64 - A64(x64)
            rel = float(cp.linalg.norm(r64) / nb64)
            if rel <= args.tol:
                ok_ir = True
                break
            r32 = (r64 / cp.linalg.norm(r64)).astype(cp.float32)
            d32, iti, _o, _t, _b = cg(A32, r32, Minv32, mask, cp.float32,
                                      args.ir_inner_tol, 3000,
                                      replace_every=args.replace_every)
            inner_total += iti
            x64 = x64 + d32.astype(cp.float64) * float(cp.linalg.norm(r64))
        cp.cuda.Stream.null.synchronize()
        wir = time.perf_counter() - t0
        true_ir = float(cp.linalg.norm(b64 - A64(x64)) / nb64)
        ok_ir = true_ir <= args.tol
        rows.append(dict(size=tag, n_elem=n_elem, scheme="fp32_ir",
                         outer=outer, inner_iters=inner_total,
                         true_resid=true_ir, converged=ok_ir, wall_s=wir))
        print(f"  fp32+ir  {outer:3d} outer / {inner_total:6d} inner  "
              f"true {true_ir:.3e}  {'OK ' if ok_ir else 'NO '} {wir:7.2f}s")

        del suite, suite64, mask, mask64, b32, b64, Minv32, Minv64
        cp.get_default_memory_pool().free_all_blocks()

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"\nwritten: {os.path.relpath(out_path, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
