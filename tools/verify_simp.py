"""
verify_simp.py
--------------
Correctness harness for the SIMP driver.

Four checks, in increasing order of what they would invalidate:

  1. Filter adjoint.   <H a, b> == <a, H^T b>.  If this fails the chain rule
     is wrong and every sensitivity in the paper is wrong with it.
  2. Sensitivity by central finite differences on a sample of elements,
     against the analytic dc/drho that the OC update consumes.  This is the
     check the floor rule requires before any stiffness-floor decision, and
     it is the only direct evidence that the three-field chain
     (filter -> projection -> SIMP -> compliance) is differentiated
     correctly.
  3. Volume constraint.  The achieved *physical* volume fraction must equal
     the prescribed one.  The constraint is written on the projected field,
     so this checks that it is also enforced there, rather than assuming it.
  4. Cross-path agreement.  The same optimization run through different
     operator mappings must produce the same compliance history to solver
     tolerance, otherwise the ablation is comparing different problems.

Usage:
    python tools/verify_simp.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))


def solve_compliance(problem, rho, penal, beta, eta, Emin, E0, filt, suite,
                     path, cg_tol, cg_maxiter, mask, F):
    """Compliance at a given raw density -- converged, or it raises."""
    import cupy as cp
    from gpu_fem.simp_r2 import project, pcg, LinearSolveNotConverged

    rho_bar = filt.forward(rho)
    rho_phys = project(rho_bar, beta, eta)
    E_e = Emin + (E0 - Emin) * rho_phys ** penal
    import cupy as cp
    dt = F.dtype
    E_dev = E_e.astype(dt)

    def matvec(v):
        return suite.matvec_full(v * mask, E_dev, path=path) * mask

    diag = suite.diagonal(E_dev, path=path)
    M_inv = cp.where(diag > 0, 1.0 / cp.maximum(diag, 1e-300),
                     cp.zeros_like(diag)) * mask
    u, iters, ok, resid = pcg(matvec, F, M_inv, x0=None, tol=cg_tol,
                              maxiter=cg_maxiter, mask=mask)
    if not ok:
        raise LinearSolveNotConverged(f"FD probe solve missed tol: {resid:.2e}")
    return float(cp.dot(F, u)), iters, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="24,12,6")
    ap.add_argument("--n-probe", type=int, default=12)
    ap.add_argument("--h", type=float, default=1e-4)
    ap.add_argument("--penal", type=float, default=3.0)
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--out", default="results/G1/simp_verification.json")
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.pub_simp_solver import KE_UNIT_3D
    from gpu_fem.cuda_operators import OperatorSuite
    from gpu_fem.filter_r2 import ConeFilter
    from gpu_fem.simp_r2 import (build_cantilever, project, dproject,
                                 run_simp, LinearSolveNotConverged)

    nelx, nely, nelz = [int(v) for v in args.grid.split(",")]
    report = {"grid": [nelx, nely, nelz], "pass": True}

    print("=" * 78)
    print("SIMP driver verification")
    print("=" * 78)

    prob = build_cantilever(nelx, nely, nelz, load="patch")
    n_elem, ndof = prob["n_elem"], prob["ndof"]
    print(f"grid {nelx}x{nely}x{nelz} = {n_elem:,} elements, {ndof:,} DOF")
    print(f"load model: {prob['load_model']}, {prob['patch_nodes']} loaded "
          f"nodes, patch area {prob['patch_area']:.4f}")

    # -- 1. filter adjoint -------------------------------------------------
    filt = ConeFilter(nelx, nely, nelz, 1.5, dtype=cp.float64)
    adj = filt.check_adjoint()
    ok_adj = adj < 1e-12
    report["filter_adjoint_rel_err"] = adj
    report["filter_neighbours"] = filt.neighbours_per_element()
    report["pass"] &= ok_adj
    print(f"\n[1] filter adjoint       rel err = {adj:.3e}   "
          f"{'PASS' if ok_adj else 'FAIL'}")
    print(f"    cone support: {filt.neighbours_per_element()} neighbours/elem "
          f"at rmin={filt.rmin}")

    # -- 2. finite-difference sensitivity ---------------------------------
    path = "fused_ai_fp64"
    suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof)
    dt = cp.float64
    mask = cp.zeros(ndof, dtype=dt)
    mask[cp.asarray(prob["free"])] = 1
    F = cp.asarray(prob["F"], dtype=dt) * mask

    rng = np.random.default_rng(7)
    rho_np = 0.3 + 0.4 * rng.random(n_elem)
    rho = cp.asarray(rho_np)
    penal, beta, eta = args.penal, args.beta, 0.5
    Emin, E0 = 1e-9, 1.0
    cg_tol, cg_maxiter = 1e-11, 60000

    rho_bar = filt.forward(rho)
    rho_phys = project(rho_bar, beta, eta)
    E_e = Emin + (E0 - Emin) * rho_phys ** penal
    E_dev = E_e.astype(dt)

    def matvec(v):
        return suite.matvec_full(v * mask, E_dev, path=path) * mask

    from gpu_fem.simp_r2 import pcg
    diag = suite.diagonal(E_dev, path=path)
    M_inv = cp.where(diag > 0, 1.0 / cp.maximum(diag, 1e-300),
                     cp.zeros_like(diag)) * mask
    u, iters0, ok0, res0 = pcg(matvec, F, M_inv, x0=None, tol=cg_tol,
                               maxiter=cg_maxiter, mask=mask)
    c0 = float(cp.dot(F, u))
    print(f"\n[2] baseline solve: {iters0} CG iters, rel resid {res0:.2e}, "
          f"c = {c0:.10f}")

    ce = suite.elem_compliance(u, fused=True).astype(cp.float64)
    ce_ref = suite.elem_compliance(u, fused=False).astype(cp.float64)
    ce_err = float(cp.linalg.norm(ce - ce_ref) / cp.linalg.norm(ce_ref))
    report["elem_compliance_fused_vs_three_stage_rel_l2"] = ce_err
    print(f"    fused adjoint kernel vs materialised three-stage: "
          f"rel L2 = {ce_err:.3e}")

    dEdrp = penal * (E0 - Emin) * rho_phys ** (penal - 1.0)
    dprj = dproject(rho_bar, beta, eta)
    dc_drho = filt.transpose(-dEdrp * ce * dprj)
    dc_np = dc_drho.get()

    probe = rng.choice(n_elem, size=args.n_probe, replace=False)
    rows = []
    worst = 0.0
    for j in probe:
        j = int(j)
        rp = rho.copy(); rp[j] += args.h
        rm = rho.copy(); rm[j] -= args.h
        cp_, _, _ = solve_compliance(prob, rp, penal, beta, eta, Emin, E0,
                                     filt, suite, path, cg_tol, cg_maxiter,
                                     mask, F)
        cm_, _, _ = solve_compliance(prob, rm, penal, beta, eta, Emin, E0,
                                     filt, suite, path, cg_tol, cg_maxiter,
                                     mask, F)
        fd = (cp_ - cm_) / (2 * args.h)
        an = float(dc_np[j])
        rel = abs(fd - an) / max(abs(fd), 1e-30)
        worst = max(worst, rel)
        rows.append({"elem": j, "fd": fd, "analytic": an, "rel_err": rel})
        print(f"    e={j:6d}  FD={fd:+.6e}  analytic={an:+.6e}  "
              f"rel={rel:.2e}")

    ok_fd = worst < 1e-4
    report["fd_check"] = {"h": args.h, "worst_rel_err": worst, "rows": rows}
    report["pass"] &= ok_fd
    print(f"    worst relative error {worst:.3e}   "
          f"{'PASS' if ok_fd else 'FAIL'} (bar 1e-4)")

    # -- 3 & 4. short runs, volume constraint, cross-path agreement -------
    print("\n[3] short runs: volume constraint and cross-path agreement")
    # Only paths that can reach the tolerance belong in an agreement check.
    # The direct single-precision mappings cannot -- their attainable residual
    # floors two to three orders above it -- so they are exercised here in the
    # form the paper actually uses them: as the inner correction solve of a
    # double-precision refinement. Running them directly, as this harness did
    # against the recursive residual, makes it fail closed on runs that
    # are now *expected* to fail.
    SPECS = (("fused_ai_fp64", dict(path="fused_ai_fp64")),
             ("three_stage_fp64", dict(path="three_stage_fp64")),
             ("ir_fused_fp32", dict(ir_inner="fused_fp32")),
             ("ir_node_fp32", dict(ir_inner="node_fp32")))
    runs = {}
    for name, kw in SPECS:
        r = run_simp(prob, volfrac=0.30, rmin=1.5,
                     fixed_budget=12, cg_tol=1e-6, cg_maxiter=20000,
                     verbose=False, record_history=True, **kw)
        runs[name] = r
        print(f"    {name:<20s} c={r['final_compliance']:.8f}  "
              f"V_phys={r['final_vol_phys']:.6f}  "
              f"cg_total={r['total_cg_iters']}  "
              f"maxresid={r['max_rel_resid']:.2e}")

    # The complement, and the more important half: direct single precision must
    # still be rejected. If this ever stops raising, the stopping test has
    # regressed to the recursive residual, and no single-precision claim
    # would be supportable -- so it is verified here rather than assumed.
    print("\n[3b] direct single precision must fail closed")
    try:
        run_simp(prob, path="node_fp32", volfrac=0.30, rmin=1.5,
                 fixed_budget=1, cg_tol=1e-6, cg_maxiter=20000,
                 verbose=False, record_history=False)
    except LinearSolveNotConverged as ex:
        ok_fail = True
        # Record the residual it actually reached, not just that it failed.
        # The paper quotes this number; a stored boolean cannot support it.
        m = re.search(r"achieved relative residual ([0-9.eE+-]+)", str(ex))
        report["direct_fp32_rejected_at_resid"] = float(m.group(1)) if m \
            else None
        print(f"    rejected as expected: {str(ex)[:86]}")
    else:
        ok_fail = False
        print("    FAIL: a direct fp32 run reported convergence at 1e-6")
    report["direct_fp32_fails_closed"] = ok_fail
    report["pass"] &= ok_fail

    vol_err = max(abs(r["final_vol_phys"] - 0.30) for r in runs.values())
    ok_vol = vol_err < 5e-3
    report["volume_max_abs_err"] = vol_err
    report["pass"] &= ok_vol
    print(f"    achieved physical volume fraction within {vol_err:.2e} of "
          f"0.30   {'PASS' if ok_vol else 'FAIL'}")

    ref = runs["fused_ai_fp64"]["final_compliance"]
    spread = max(abs(r["final_compliance"] - ref) / abs(ref)
                 for r in runs.values())
    ok_cross = spread < 5e-3
    report["cross_path_compliance_spread"] = spread
    report["pass"] &= ok_cross
    print(f"    cross-path compliance spread {spread:.3e}   "
          f"{'PASS' if ok_cross else 'FAIL'}")
    report["short_runs"] = {
        p: {"final_compliance": r["final_compliance"],
            "final_vol_phys": r["final_vol_phys"],
            "total_cg_iters": r["total_cg_iters"],
            "max_rel_resid": r["max_rel_resid"],
            "solves_at_cap": r["solves_at_cap"]}
        for p, r in runs.items()}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\n" + "=" * 78)
    print("SIMP VERIFICATION: " + ("PASS" if report["pass"] else "FAIL"))
    print(f"written to {args.out}")
    print("=" * 78)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
