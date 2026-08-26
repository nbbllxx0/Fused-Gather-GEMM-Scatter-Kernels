"""
kappa_bf16_study.py
-------------------
Condition-number estimates and the BF16 precision-boundary study, measured
on the same platform as every other number the paper reports.

Condition number: SUPERSEDED here, and off unless --with-inverse-iteration-kappa
is passed. The inverse-iteration estimator this file carries converged at 64k
and returned its 60,000-iteration inner cap at every larger mesh. Use
kappa_lanczos.py, which produces a Lanczos-Ritz lower bound in seconds. Both
write kappa_estimation.json, so running this one would overwrite a converged
bound with an unconverged estimate.

BF16: the WMMA kernel is timed and solved with directly. There is no separate
GEMM proxy: a proxy measures the GEMM, not the solver, and the question here
is whether BF16 can carry a solve at all.

Usage:
    python experiments/phase3/kappa_bf16_study.py --sizes 64k,216k,512k
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

SIZE_LADDER = {
    "64k":  (80, 40, 20),
    "216k": (120, 60, 30),
    "512k": (160, 80, 40),
    "1M":   (200, 100, 50),
}


def power_iteration(matvec, n, mask, iters, seed, dtype):
    import cupy as cp
    rng = cp.random.RandomState(seed)
    v = rng.rand(n).astype(dtype) * mask
    v /= cp.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = matvec(v)
        nw = float(cp.linalg.norm(w))
        if nw == 0.0:
            break
        v = w / nw
        lam = nw
    return lam


def smallest_eigenvalue(matvec, n, mask, M_inv, outer, tol, maxiter, seed,
                        dtype):
    """Inverse iteration; each inner solve uses the fail-closed CG."""
    import cupy as cp
    from gpu_fem.simp_r2 import pcg
    rng = cp.random.RandomState(seed)
    v = rng.rand(n).astype(dtype) * mask
    v /= cp.linalg.norm(v)
    lam = float("nan")
    inner_iters, inner_ok = [], True
    for _ in range(outer):
        w, it, ok, res = pcg(matvec, v, M_inv, x0=None, tol=tol,
                             maxiter=maxiter, mask=mask)
        inner_iters.append(it)
        inner_ok &= ok
        nw = float(cp.linalg.norm(w))
        if nw == 0.0:
            break
        v = w / nw
        lam = 1.0 / nw
    return lam, inner_iters, inner_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="64k,216k,512k")
    ap.add_argument("--penals", default="3.0,5.0")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--emin", type=float, default=1e-9)
    ap.add_argument("--pi-iters", type=int, default=200)
    ap.add_argument("--ii-outer", type=int, default=8)
    ap.add_argument("--gate", default="G6", help="name of the results/ subdirectory these runs are written "
                         "to; G6 is the one the paper reports from")
    ap.add_argument("--skip-bf16", action="store_true")
    ap.add_argument("--with-inverse-iteration-kappa", action="store_true",
                    help="re-run the superseded inverse-iteration condition "
                         "number estimator and overwrite "
                         "kappa_estimation.json with it; use kappa_lanczos.py "
                         "instead, whose bound converges")
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.pub_simp_solver import KE_UNIT_3D
    from gpu_fem.cuda_operators import OperatorSuite
    from gpu_fem.cuda_fused_matvec import FusedMatvec
    from gpu_fem.simp_r2 import build_cantilever, pcg

    outdir = os.path.join(_ROOT, "results", args.gate)
    os.makedirs(outdir, exist_ok=True)
    kappa_rows, bf16_rows = [], []

    print("=" * 78)
    print("Condition-number and BF16 precision-boundary study")
    print("=" * 78)

    for tag in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        nelx, nely, nelz = SIZE_LADDER[tag]
        prob = build_cantilever(nelx, nely, nelz, load="patch")
        n_elem, ndof = prob["n_elem"], prob["ndof"]
        print(f"\n### {tag}: {nelx}x{nely}x{nelz} = {n_elem:,} elem")

        cp.get_default_memory_pool().free_all_blocks()
        suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof)
        mask64 = cp.zeros(ndof, dtype=cp.float64)
        mask64[cp.asarray(prob["free"])] = 1

        # ---- condition number, per penalisation ------------------------
        # Superseded and off by default. This inverse-iteration estimator
        # converged at 64k and nowhere else: at 216k and 512k every inner
        # solve after the first exhausted its 60,000-iteration cap, so the
        # lambda_min it returned -- and the condition number computed from it
        # -- were the cap rather than a measurement. kappa_lanczos.py replaces
        # it and writes the same file, which is why this branch must stay off
        # unless explicitly asked for: running it would overwrite a converged
        # bound with an unconverged estimate.
        for penal in ([float(p) for p in args.penals.split(",")]
                      if args.with_inverse_iteration_kappa else []):
            E = cp.full(n_elem,
                        args.emin + (1.0 - args.emin) * args.rho ** penal,
                        dtype=cp.float64)

            def mv(v):
                return suite.matvec_full(v * mask64, E,
                                         path="fused_ai_fp64") * mask64

            diag = suite.diagonal(E, path="fused_ai_fp64")
            M_inv = cp.where(diag > 0, 1.0 / cp.maximum(diag, 1e-300),
                             cp.zeros_like(diag)) * mask64

            t0 = time.perf_counter()
            lmax = power_iteration(mv, ndof, mask64, args.pi_iters, 42,
                                   cp.float64)
            lmin, inner, ok = smallest_eigenvalue(
                mv, ndof, mask64, M_inv, args.ii_outer, 1e-10, 60000, 123,
                cp.float64)
            dt = time.perf_counter() - t0
            kappa = lmax / lmin if lmin > 0 else float("nan")
            eps_bf16 = 2.0 ** -8
            row = {
                "size": tag, "n_elem": n_elem, "ndof": ndof,
                "state": f"uniform_p{penal:g}", "rho_mean": args.rho,
                "penal": penal, "Emin": args.emin,
                "lam_max": lmax, "lam_min": lmin, "kappa": kappa,
                "eps_bf16_kappa": eps_bf16 * kappa,
                "bf16_reference_bound": 1.0 / eps_bf16,
                "exceeds_bf16_bound_by": eps_bf16 * kappa,
                "pi_iters": args.pi_iters, "ii_outer": args.ii_outer,
                "ii_inner_iters": inner, "ii_inner_converged": bool(ok),
                "elapsed_s": dt,
            }
            kappa_rows.append(row)
            print(f"  p={penal:g}: lam_max={lmax:.6g} lam_min={lmin:.6g} "
                  f"kappa={kappa:.3e}  eps*kappa={eps_bf16*kappa:.3e}  "
                  f"({dt:.1f} s)")

        # ---- BF16 precision boundary -----------------------------------
        if not args.skip_bf16:
            try:
                penal = 3.0
                E32 = cp.full(n_elem,
                              args.emin + (1.0 - args.emin) * args.rho ** penal,
                              dtype=cp.float32)
                mask32 = mask64.astype(cp.float32)
                F32 = cp.asarray(prob["F"], dtype=cp.float32) * mask32
                edof = suite._edof32.reshape(n_elem, 24)
                fm = FusedMatvec(edof, cp.asarray(KE_UNIT_3D), ndof)
                free32 = cp.asarray(prob["free"]).astype(cp.int32)

                def mv32(v):
                    return fm.matvec_full(v * mask32, E32,
                                          dtype="fp32") * mask32

                def mvbf(v):
                    return fm.matvec_full(v * mask32, E32,
                                          dtype="bf16") * mask32

                diag32 = suite.diagonal(E32, path="fused_fp32")
                Mi32 = cp.where(diag32 > 0, 1.0 / cp.maximum(diag32, 1e-30),
                                cp.zeros_like(diag32)) * mask32

                # operator-level agreement of the BF16 kernel
                rng = np.random.default_rng(7)
                vp = cp.asarray(rng.standard_normal(ndof),
                                dtype=cp.float32) * mask32
                y32, ybf = mv32(vp), mvbf(vp)
                op_err = float(cp.linalg.norm(ybf - y32)
                               / cp.linalg.norm(y32))

                # The reference has to be a converged solve, and single
                # precision cannot produce one: its attainable residual on
                # these systems floors near 1e-3, so a "reference" solved in
                # fp32 to 1e-6 is simply the iteration cap wearing a
                # tolerance's name, and every error measured against it is
                # meaningless. Solve the reference in double precision.
                E64 = cp.asarray(E32, dtype=cp.float64)
                F64r = cp.asarray(prob["F"], dtype=cp.float64) * mask64

                def mv64(v):
                    return suite.matvec_full(v * mask64, E64,
                                             path="fused_fp64") * mask64

                diag64 = suite.diagonal(E64, path="fused_fp64")
                Mi64 = cp.where(diag64 > 0, 1.0 / cp.maximum(diag64, 1e-300),
                                cp.zeros_like(diag64)) * mask64
                ref64, it_ref, ok_ref, res_ref = pcg(
                    mv64, F64r, Mi64, tol=1e-6, maxiter=40000, mask=mask64)
                if not ok_ref:
                    raise RuntimeError(
                        f"reference solve did not converge: {res_ref:.3e}")
                c_ref = float(cp.dot(F64r, ref64))
                ref = ref64.astype(cp.float32)

                out = {"size": tag, "n_elem": n_elem,
                       "operator_rel_l2_bf16_vs_fp32": op_err,
                       "reference": {"solver": "fp64", "cg_iters": it_ref,
                                     "converged": bool(ok_ref),
                                     "rel_resid": res_ref,
                                     "compliance": c_ref}}

                # Distance from the reference solution itself, alongside the
                # residual and the objective. Three numbers that disagree by
                # orders of magnitude on the same solve is the finding, and
                # only reporting all three makes the disagreement legible:
                # the residual is the error seen through K, so it is weighted
                # towards the top of the spectrum, while the objective is a
                # smooth functional weighted towards the bottom. A solve can
                # therefore look excellent in the objective, ordinary in the
                # solution, and catastrophic in the residual -- which is
                # precisely why the acceptance rule tests the residual.
                nref = float(cp.linalg.norm(ref))

                def sol_err(x):
                    return float(cp.linalg.norm(x - ref)) / nref

                # plain BF16 CG
                t0 = time.perf_counter()
                xb, itb, okb, resb = pcg(mvbf, F32, Mi32, tol=1e-6,
                                         maxiter=4000, mask=mask32)
                cb = float(cp.dot(F32, xb))
                out["plain_bf16"] = {
                    "cg_iters": itb, "converged": bool(okb),
                    "rel_resid": resb, "compliance": cb,
                    "compliance_rel_err": abs(c_ref - cb) / abs(c_ref),
                    "solution_rel_l2": sol_err(xb),
                    "wall_s": time.perf_counter() - t0}

                # BF16 inner solve inside an FP32 residual-correction loop
                for inner_tol in (1e-3, 1e-5):
                    t0 = time.perf_counter()
                    x = cp.zeros(ndof, dtype=cp.float32)
                    tot = 0
                    for _ in range(8):
                        r = (F32 - mv32(x)) * mask32
                        d, itk, _, _ = pcg(mvbf, r, Mi32, tol=inner_tol,
                                           maxiter=2000, mask=mask32)
                        tot += itk
                        x = x + d
                    cx = float(cp.dot(F32, x))
                    rr = float(cp.linalg.norm((F32 - mv32(x)) * mask32)
                               / cp.linalg.norm(F32))
                    out[f"bf16_ir_{inner_tol:g}"] = {
                        "cg_iters": tot, "outer": 8, "rel_resid": rr,
                        "compliance": cx,
                        "compliance_rel_err": abs(c_ref - cx) / abs(c_ref),
                        "solution_rel_l2": sol_err(x),
                        "wall_s": time.perf_counter() - t0}

                bf16_rows.append(out)
                print(f"  BF16 operator rel L2 vs FP32: {op_err:.3e}")
                print(f"  FP64 reference c={c_ref:.6f} ({it_ref} iters, "
                      f"resid {res_ref:.2e})")
                print(f"  plain BF16 CG   c={cb:.6f}  "
                      f"rel err {out['plain_bf16']['compliance_rel_err']:.3f}")
                for k in (1e-3, 1e-5):
                    e = out[f"bf16_ir_{k:g}"]["compliance_rel_err"]
                    print(f"  BF16-IR tol {k:g}  rel err {e:.3f}")
                del fm
            except Exception as ex:                             # noqa: BLE001
                print(f"  BF16 study failed: {ex!r}")
                bf16_rows.append({"size": tag, "error": repr(ex)})

        del suite
        cp.get_default_memory_pool().free_all_blocks()

    # Only write the condition-number file if this run actually produced one.
    # An empty list written here would erase the Lanczos bound that
    # kappa_lanczos.py puts in the same place.
    if kappa_rows:
        with open(os.path.join(outdir, "kappa_estimation.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(kappa_rows, fh, indent=2)
    with open(os.path.join(outdir, "bf16_study.json"), "w",
              encoding="utf-8") as fh:
        json.dump(bf16_rows, fh, indent=2)
    print(f"\nwritten: results/{args.gate}/kappa_estimation.json")
    print(f"         results/{args.gate}/bf16_study.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
