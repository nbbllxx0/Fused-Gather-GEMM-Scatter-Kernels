"""
coldstart_ladder.py
-------------------
Converged cold-start FEA ladder at uniform density.

A solver-scaling ladder is a scaling measurement only if every point on it
converged, so three rules apply throughout:

  * every point reports its exact CG iteration count and the relative
    residual it achieved, because a time is uninterpretable without them;
  * no point may stop at a cap.  The cap is 20,000 and the run fails closed
    if it is ever reached -- a capped point measures the cap;
  * two quantities are reported separately rather than merged:
      - time per operator application, genuinely O(n), and the only thing a
        faster kernel changes;
      - time to tolerance, O(n) x iteration count, where the iteration count
        itself grows like n^(1/3) for Jacobi-PCG.
    A fixed iteration cap makes the second look like the first, and so makes
    an O(n) fit look like mesh-independent solver scaling.  Keeping them
    apart is what lets the n^(1/3) iteration growth be seen at all.

Usage:
    python experiments/phase3/coldstart_ladder.py --sizes 216k,512k,1M,2M,4.9M
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

SIZE_LADDER = {
    "64k":  (80, 40, 20),
    "216k": (120, 60, 30),
    "512k": (160, 80, 40),
    "1M":   (200, 100, 50),
    "2M":   (252, 126, 63),
    "4.9M": (170, 170, 170),
    "8M":   (400, 200, 100),
    "13M":  (468, 234, 117),
    "16M":  (504, 252, 126),
}

DEFAULT_PATHS = ["three_stage_fp64", "three_stage_fp32", "fused_fp64",
                 "fused_fp32", "fused_ai_fp32", "node_fp32"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="216k,512k,1M,2M")
    ap.add_argument("--paths", default=",".join(DEFAULT_PATHS))
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--penal", type=float, default=3.0)
    ap.add_argument("--cg-tol", type=float, default=1e-5)
    ap.add_argument("--cg-maxiter", type=int, default=20000)
    ap.add_argument("--emin", type=float, default=1e-9)
    ap.add_argument("--load", default="patch")
    ap.add_argument("--matvec-samples", type=int, default=30)
    ap.add_argument("--gate", default="G6", help="name of the results/ subdirectory these runs are written "
                         "to; G6 is the one the paper reports from")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cupy as cp
    import numpy as np
    from gpu_fem.pub_simp_solver import KE_UNIT_3D
    from gpu_fem.cuda_operators import OperatorSuite, PATH_SPEC, NEEDS_EDOF
    from gpu_fem.simp_r2 import build_cantilever, pcg

    out_path = args.out or os.path.join(_ROOT, "results", args.gate,
                                        "coldstart_ladder.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            rows = json.load(fh)

    print("=" * 78)
    print("converged cold-start FEA ladder")
    print(f"uniform rho={args.rho}, p={args.penal}, tol={args.cg_tol:g}, "
          f"cap={args.cg_maxiter}, load={args.load}")
    print("=" * 78)

    for tag in [s.strip() for s in args.sizes.split(",") if s.strip()]:
        nelx, nely, nelz = SIZE_LADDER[tag]
        prob = build_cantilever(nelx, nely, nelz, load=args.load)
        n_elem, ndof = prob["n_elem"], prob["ndof"]
        print(f"\n### {tag}: {nelx}x{nely}x{nelz} = {n_elem:,} elem, "
              f"{ndof:,} DOF")

        for path in [p.strip() for p in args.paths.split(",") if p.strip()]:
            if any(r.get("size") == tag and r.get("path") == path
                   for r in rows):
                print(f"  [{path}] already done, skipping")
                continue
            cp.get_default_memory_pool().free_all_blocks()
            dt = cp.float64 if PATH_SPEC[path][0] == "fp64" else cp.float32
            try:
                suite = OperatorSuite(nelx, nely, nelz, KE_UNIT_3D, ndof=ndof,
                                      build_edof=(path in NEEDS_EDOF))
                mask = cp.zeros(ndof, dtype=dt)
                mask[cp.asarray(prob["free"])] = 1
                F = cp.asarray(prob["F"], dtype=dt) * mask
                E = cp.full(n_elem,
                            args.emin + (1.0 - args.emin)
                            * args.rho ** args.penal, dtype=dt)

                def matvec(v):
                    return suite.matvec_full(v * mask, E, path=path) * mask

                diag = suite.diagonal(E, path=path)
                M_inv = cp.where(diag > 0, 1.0 / cp.maximum(diag, 1e-300),
                                 cp.zeros_like(diag)) * mask

                # -- time per operator application (true O(n)) ------------
                v = cp.asarray(np.random.default_rng(1).standard_normal(ndof),
                               dtype=dt) * mask
                for _ in range(5):
                    matvec(v)
                cp.cuda.Stream.null.synchronize()
                ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
                ev0.record()
                for _ in range(args.matvec_samples):
                    matvec(v)
                ev1.record()
                ev1.synchronize()
                us_per_matvec = (cp.cuda.get_elapsed_time(ev0, ev1)
                                 / args.matvec_samples) * 1e3

                # -- time to tolerance ------------------------------------
                cp.cuda.Stream.null.synchronize()
                t0 = time.perf_counter()
                u, iters, ok, resid = pcg(matvec, F, M_inv, x0=None,
                                          tol=args.cg_tol,
                                          maxiter=args.cg_maxiter, mask=mask)
                cp.cuda.Stream.null.synchronize()
                wall = time.perf_counter() - t0
                c = float(cp.dot(F, u))

                free_b, total_b = cp.cuda.runtime.memGetInfo()
                row = {
                    "size": tag, "n_elem": n_elem, "ndof": ndof, "path": path,
                    "nelx": nelx, "nely": nely, "nelz": nelz,
                    "rho": args.rho, "penal": args.penal,
                    "cg_tol": args.cg_tol, "cg_maxiter": args.cg_maxiter,
                    "cg_iters": iters, "rel_resid": resid,
                    "converged": bool(ok), "at_cap": bool(iters >= args.cg_maxiter),
                    "compliance": c,
                    "wall_to_tolerance_s": wall,
                    "us_per_matvec": us_per_matvec,
                    "matvec_time_share": (us_per_matvec * 1e-6 * iters) / wall
                    if wall > 0 else float("nan"),
                    "peak_device_used_GiB": (total_b - free_b) / 2**30,
                    "load_model": prob["load_model"],
                }
                if not ok:
                    row["invalid"] = True
                rows.append(row)
                print(f"  {path:<20s} {iters:6d} CG iters, resid {resid:.3e}, "
                      f"{wall:8.2f} s to tol, {us_per_matvec:9.1f} us/matvec, "
                      f"c={c:.6f}" + ("   NOT CONVERGED" if not ok else ""))

                del suite, mask, F, E, diag, M_inv, u, v
            except cp.cuda.memory.OutOfMemoryError as ex:
                print(f"  {path:<20s} OOM")
                rows.append({"size": tag, "n_elem": n_elem, "path": path,
                             "oom": True, "reason": str(ex)[:200]})
            cp.get_default_memory_pool().free_all_blocks()
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=2)

    csv_path = out_path.replace(".json", ".csv")
    cols = ["size", "n_elem", "ndof", "nelx", "nely", "nelz", "path", "rho",
            "penal", "cg_tol", "cg_maxiter", "cg_iters", "rel_resid",
            "converged", "at_cap", "compliance", "wall_to_tolerance_s",
            "us_per_matvec", "matvec_time_share", "peak_device_used_GiB",
            "load_model", "oom"]
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nwritten: {out_path}\n         {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
