"""
floor_sweep.py
--------------
Stiffness-floor sweep: does the value of the floor drive the CG iteration
count, and does raising it change the design?

The question is often posed about "rho_min", but that name covers two
different quantities and they take different values here:

    rho_lb     lower bound on the raw design variable       code: 1e-3
    Emin/E0    stiffness floor in the SIMP interpolation    code: 1e-9

1e-3 is the usual choice for the design bound and 1e-9 is top88's
additive-floor convention for the interpolation.  Conflating them turns a
question about conditioning into a question about the design space.

The floor is *numerical regularization of the interpolation*, not a material
property, so it is not defended as a physics choice.  It is decided on
evidence, and the decision rule is multi-criteria and fixed in advance: a
floor is only raised if the tighter value is untenable on solver evidence
AND the looser value leaves compliance, achieved volume, topology and
sensitivities materially unchanged.  So every floor reports:

    per-solve CG iterations and achieved residual, time to tolerance,
    final compliance, achieved physical volume fraction, grayness,
    the design field (binary-mismatch fraction vs the reference floor),
    and a finite-difference sensitivity spot check.

Usage:
    python experiments/phase3/floor_sweep.py --size 512k
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

SIZE_LADDER = {
    "64k":  (80, 40, 20),
    "216k": (120, 60, 30),
    "512k": (160, 80, 40),
    "1M":   (200, 100, 50),
}


def fd_spot_check(prob, rho, penal, beta, eta, Emin, E0, filt, suite, path,
                  mask, F, n_probe=6, h=1e-4, seed=11):
    """Central-difference check of dc/drho at the given design state."""
    import cupy as cp
    import numpy as np
    from gpu_fem.simp_r2 import project, dproject, pcg

    def compliance(r):
        rb = filt.forward(r)
        rp = project(rb, beta, eta)
        Ee = (Emin + (E0 - Emin) * rp ** penal).astype(F.dtype)

        def mv(v):
            return suite.matvec_full(v * mask, Ee, path=path) * mask
        dg = suite.diagonal(Ee, path=path)
        Mi = cp.where(dg > 0, 1.0 / cp.maximum(dg, 1e-300),
                      cp.zeros_like(dg)) * mask
        u, it, ok, res = pcg(mv, F, Mi, x0=None, tol=1e-10, maxiter=60000,
                             mask=mask)
        return float(cp.dot(F, u)), u, ok

    c0, u0, ok0 = compliance(rho)
    if not ok0:
        return {"ok": False, "reason": "baseline FD probe did not converge"}
    rb = filt.forward(rho)
    rp = project(rb, beta, eta)
    ce = suite.elem_compliance(u0, fused=True).astype(cp.float64)
    dEdrp = penal * (E0 - Emin) * rp ** (penal - 1.0)
    dc = filt.transpose(-dEdrp * ce * dproject(rb, beta, eta)).get()

    rng = np.random.default_rng(seed)
    probe = rng.choice(prob["n_elem"], size=n_probe, replace=False)
    worst, rows = 0.0, []
    for j in probe:
        j = int(j)
        rp_ = rho.copy(); rp_[j] += h
        rm_ = rho.copy(); rm_[j] -= h
        cpv, _, okp = compliance(rp_)
        cmv, _, okm = compliance(rm_)
        if not (okp and okm):
            continue
        fd = (cpv - cmv) / (2 * h)
        an = float(dc[j])
        rel = abs(fd - an) / max(abs(fd), 1e-30)
        worst = max(worst, rel)
        rows.append({"elem": j, "fd": fd, "analytic": an, "rel_err": rel})
    return {"ok": True, "worst_rel_err": worst, "rows": rows, "c0": c0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="512k")
    ap.add_argument("--floors", default="1e-9,1e-6,1e-3")
    # Double precision, and not negotiable by default: a direct
    # single-precision path cannot reach cg_tol on these systems, so with the
    # a single-precision path every floor in the sweep comes back invalid and the
    # study produces nothing. The sweep is about the stiffness floor, not about
    # arithmetic, so it is run on the fastest mapping that can actually solve.
    ap.add_argument("--path", default="fused_fp64")
    ap.add_argument("--fd-path", default="fused_ai_fp64")
    ap.add_argument("--problem", default="cantilever")
    ap.add_argument("--volfrac", type=float, default=0.30)
    ap.add_argument("--cg-tol", type=float, default=1e-5)
    ap.add_argument("--cg-maxiter", type=int, default=20000)
    ap.add_argument("--gate", default="G6", help="name of the results/ subdirectory these runs are written "
                         "to; G6 is the one the paper reports from")
    args = ap.parse_args()

    import cupy as cp
    from gpu_fem.pub_simp_solver import KE_UNIT_3D
    from gpu_fem.cuda_operators import OperatorSuite, NEEDS_EDOF
    from gpu_fem.filter_r2 import ConeFilter
    from gpu_fem.simp_r2 import (build_cantilever, build_torsion, run_simp,
                                 save_result, LinearSolveNotConverged)

    nelx, nely, nelz = SIZE_LADDER[args.size]
    prob = (build_cantilever(nelx, nely, nelz, load="patch")
            if args.problem == "cantilever"
            else build_torsion(nelx, nely, nelz, load="patch"))
    outdir = os.path.join(_ROOT, "results", args.gate)
    os.makedirs(outdir, exist_ok=True)

    print("=" * 78)
    print(f"stiffness-floor sweep -- {args.problem} {args.size}, "
          f"path {args.path}")
    print("Emin/E0 is numerical regularization of the SIMP interpolation, "
          "not a material property.")
    print("=" * 78)

    floors = [float(f) for f in args.floors.split(",")]
    results = {}
    for floor in floors:
        print(f"\n### Emin/E0 = {floor:g}")
        cp.get_default_memory_pool().free_all_blocks()
        try:
            r = run_simp(prob, path=args.path, volfrac=args.volfrac,
                         Emin=floor, rho_lb=1e-3,
                         cg_tol=args.cg_tol, cg_maxiter=args.cg_maxiter,
                         verbose=False)
        except LinearSolveNotConverged as ex:
            print(f"  INVALID: {ex}")
            results[f"{floor:g}"] = {"invalid": True, "reason": str(ex)}
            continue
        save_result(r, os.path.join(
            outdir, f"floor_{args.size}_{floor:g}.json"), include_field=True)
        cg = [s["cg_iters"] for s in r["solve_log"]]
        results[f"{floor:g}"] = {
            "Emin": floor,
            "outer_iters": r["outer_iters"],
            "outer_converged": r["outer_converged"],
            "final_compliance": r["final_compliance"],
            "final_vol_phys": r["final_vol_phys"],
            "final_grayness": r["final_grayness"],
            "total_cg_iters": r["total_cg_iters"],
            "mean_cg_per_solve": sum(cg) / max(len(cg), 1),
            "max_cg_single_solve": max(cg) if cg else 0,
            "max_rel_resid": r["max_rel_resid"],
            "solves_at_cap": r["solves_at_cap"],
            "warm_start_retries": r["warm_start_retries"],
            "wall_s": r["wall_s"],
            "rho_phys_min": r["rho_phys_min"],
            "E_min_achieved": r["E_min_achieved"],
            "rho_field": r["rho_final_device"],
        }
        print(f"  {r['outer_iters']} outer iters "
              f"({'converged' if r['outer_converged'] else r['stop_reason']}), "
              f"c={r['final_compliance']:.6f}, V={r['final_vol_phys']:.4f}, "
              f"g={r['final_grayness']:.5f}, "
              f"CG mean={sum(cg)/max(len(cg),1):.1f} max={max(cg) if cg else 0}, "
              f"maxresid={r['max_rel_resid']:.2e}, {r['wall_s']:.1f} s")
        print(f"  achieved min projected density {r['rho_phys_min']:.3e}, "
              f"achieved min element stiffness {r['E_min_achieved']:.3e}")

    # -- topology comparison vs the reference floor -----------------------
    ref_key = f"{floors[0]:g}"
    ref = results.get(ref_key, {}).get("rho_field")
    for k, v in results.items():
        fld = v.pop("rho_field", None)
        if ref is not None and fld is not None:
            mism = float(((fld > 0.5) != (ref > 0.5)).mean())
            v["binary_mismatch_vs_ref"] = mism
            v["reference_floor"] = ref_key
            print(f"  binary mismatch vs Emin={ref_key}: {k} -> {mism:.5f}")

    # -- finite-difference sensitivity spot check --------------------------
    print("\n### finite-difference sensitivity spot check "
          f"({args.fd_path}, FP64)")
    fd_out = {}
    suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=prob["ndof"],
                          build_edof=(args.fd_path in NEEDS_EDOF))
    filt = ConeFilter(nelx, nely, nelz, 1.5, dtype=cp.float64)
    mask = cp.zeros(prob["ndof"], dtype=cp.float64)
    mask[cp.asarray(prob["free"])] = 1
    F = cp.asarray(prob["F"], dtype=cp.float64) * mask
    rho_probe = cp.full(prob["n_elem"], args.volfrac, dtype=cp.float64)
    for floor in floors:
        res = fd_spot_check(prob, rho_probe, 3.0, 4.0, 0.5, floor, 1.0,
                            filt, suite, args.fd_path, mask, F)
        fd_out[f"{floor:g}"] = res
        if res.get("ok"):
            print(f"  Emin={floor:g}: worst FD relative error "
                  f"{res['worst_rel_err']:.3e}")
        else:
            print(f"  Emin={floor:g}: {res.get('reason')}")

    out = {"size": args.size, "problem": args.problem, "path": args.path,
           "floors": floors, "runs": results, "fd_check": fd_out}
    out_path = os.path.join(outdir, f"floor_sweep_{args.size}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
